// doorbell_bench.cu - SigmaTrace doorbell overhead per NVSHMEM operation type
//
// Tests: put, get, atomic, pingpong, put_bulk (varying sizes)
// Each operation is timed individually inside a persistent single-thread kernel.
//
// Output: CSV per operation type for direct plotting.
//
// Run: nvshmrun -np 4 ./doorbell_bench [all|put|get|atomic|pingpong|put_bulk]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// globaltimer gives us a fixed-frequency nanosecond counter
__device__ __forceinline__ long long globaltimer_ns() {
    long long ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    return ns;
}

// ================================================================
// Stats
struct Stats {
    double mean, median, p5, p95, p99, stddev, min, max;
    int n;
};

static Stats compute_stats(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    int n = (int) v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / n;
    double sq = 0;
    for (auto x: v)
        sq += (x - mean) * (x - mean);
    return {mean,
            v[n / 2],
            v[(int) (n * 0.05)],
            v[std::min(n - 1, (int) (n * 0.95))],
            v[std::min(n - 1, (int) (n * 0.99))],
            std::sqrt(sq / n),
            v[0],
            v[n - 1],
            n};
}

// ================================================================
// Common params

constexpr int WARMUP = 1000;
constexpr int ITERS = 10000;

// ================================================================
// Per-operation kernels

// Put scalar + quiet
__global__ void kernel_put_quiet(int* remote_buf, int pe, int warmup, int iters, long long* out_ns) {
    for (int i = 0; i < warmup; ++i) {
        nvshmem_int_p(remote_buf, i, pe);
    }
    nvshmem_quiet();
    for (int i = 0; i < iters; ++i) {
        long long t0 = globaltimer_ns();
        nvshmem_int_p(remote_buf, i, pe);
        nvshmem_quiet();
        long long t1 = globaltimer_ns();
        out_ns[i] = t1 - t0;
    }
    nvshmem_quiet();
}

// Get scalar
__global__ void kernel_get(const int* remote_buf, int pe, int warmup, int iters, long long* out_ns) {
    volatile int sink;
    for (int i = 0; i < warmup; ++i) {
        sink = nvshmem_int_g(remote_buf, pe);
    }
    for (int i = 0; i < iters; ++i) {
        long long t0 = globaltimer_ns();
        sink = nvshmem_int_g(remote_buf, pe);
        long long t1 = globaltimer_ns();
        out_ns[i] = t1 - t0;
    }
    (void) sink;
}

// Atomic fetch-add
__global__ void kernel_atomic_fadd(int* remote_buf, int pe, int warmup, int iters, long long* out_ns) {
    volatile int sink;
    for (int i = 0; i < warmup; ++i) {
        sink = nvshmem_int_atomic_fetch_add(remote_buf, 1, pe);
    }
    for (int i = 0; i < iters; ++i) {
        long long t0 = globaltimer_ns();
        sink = nvshmem_int_atomic_fetch_add(remote_buf, 1, pe);
        long long t1 = globaltimer_ns();
        out_ns[i] = t1 - t0;
    }
    (void) sink;
}

// Ping-pong: put + spin-wait for remote response
__global__ void kernel_pingpong_init(volatile int* local_buf, int* remote_buf, int pe, int warmup, int iters,
                                     long long* out_ns) {
    for (int i = 1; i <= warmup; ++i) {
        nvshmem_int_p(remote_buf, i, pe);
        while (*local_buf < i) {
        }
    }
    // Sync point: both sides use values > warmup for measured phase
    for (int i = 1; i <= iters; ++i) {
        int tag = warmup + i;
        long long t0 = globaltimer_ns();
        nvshmem_int_p(remote_buf, tag, pe);
        while (*local_buf < tag) {
        }
        long long t1 = globaltimer_ns();
        out_ns[i - 1] = t1 - t0;
    }
}

__global__ void kernel_pingpong_resp(volatile int* local_buf, int* remote_buf, int pe, int warmup, int iters) {
    for (int i = 1; i <= warmup; ++i) {
        while (*local_buf < i) {
        }
        nvshmem_int_p(remote_buf, i, pe);
    }
    for (int i = 1; i <= iters; ++i) {
        int tag = warmup + i;
        while (*local_buf < tag) {
        }
        nvshmem_int_p(remote_buf, tag, pe);
    }
}

// Bulk put + quiet at varying sizes
__global__ void kernel_put_bulk_quiet(float* dest, const float* src, size_t n_elems, int pe, int warmup, int iters,
                                      long long* out_ns) {
    for (int i = 0; i < warmup; ++i) {
        nvshmem_float_put_nbi(dest, src, n_elems, pe);
        nvshmem_quiet();
    }
    for (int i = 0; i < iters; ++i) {
        long long t0 = globaltimer_ns();
        nvshmem_float_put_nbi(dest, src, n_elems, pe);
        nvshmem_quiet();
        long long t1 = globaltimer_ns();
        out_ns[i] = t1 - t0;
    }
}

// ================================================================
// Host-side runner
static void dump_csv(const char* filename, const char* op_name, const char* bytes_str,
                     const std::vector<double>& ns_vec) {
    FILE* f = fopen(filename, "a");
    if (!f) {
        return;
    }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        fprintf(f, "op,bytes,iter,ns,us\n");
    }
    const char* bstr = bytes_str ? bytes_str : "0";
    for (int i = 0; i < (int) ns_vec.size(); ++i) {
        fprintf(f, "%s,%s,%d,%.0f,%.3f\n", op_name, bstr, i, ns_vec[i], ns_vec[i] / 1000.0);
    }
    fclose(f);
}

static std::vector<double> collect_timestamps(long long* d_buf, int n) {
    std::vector<long long> h_buf(n);
    CUDA_CHECK(cudaMemcpy(h_buf.data(), d_buf, n * sizeof(long long), cudaMemcpyDeviceToHost));
    std::vector<double> ns(n);
    for (int i = 0; i < n; ++i) {
        ns[i] = (double) h_buf[i];
    }
    return ns;
}

static void run_single_op(const char* name, int mype, int npes, cudaStream_t stream, long long* d_ts,
                          const char* csv_file) {
    int target = (mype + 1) % npes;
    int* sym_buf = (int*) nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemset(sym_buf, 0, sizeof(int)));

    if (strcmp(name, "put") == 0) {
        kernel_put_quiet<<<1, 1, 0, stream>>>(sym_buf, target, WARMUP, ITERS, d_ts);
    } else if (strcmp(name, "get") == 0) {
        kernel_get<<<1, 1, 0, stream>>>(sym_buf, target, WARMUP, ITERS, d_ts);
    } else if (strcmp(name, "atomic") == 0) {
        kernel_atomic_fadd<<<1, 1, 0, stream>>>(sym_buf, target, WARMUP, ITERS, d_ts);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    nvshmem_barrier_all();

    if (mype == 0) {
        auto ns = collect_timestamps(d_ts, ITERS);
        auto s = compute_stats(ns);
        printf("  %-12s  median=%8.0f ns  p5=%8.0f  p95=%8.0f  "
               "min=%8.0f  max=%8.0f\n",
               name, s.median, s.p5, s.p95, s.min, s.max);
        dump_csv(csv_file, name, nullptr, ns);
    }

    nvshmem_free(sym_buf);
}

static void run_pingpong(int mype, int npes, cudaStream_t stream, long long* d_ts, const char* csv_file) {
    if (npes < 2) {
        if (mype == 0) {
            printf("  pingpong skipped: requires >= 2 PEs\n");
        }
        return;
    }

    // Pair PEs by XOR. With an odd npes the highest PE has no partner, but it must
    // still participate in the collective nvshmem_malloc/barrier/free calls below or
    // the active pairs will hang.
    int partner = mype ^ 1;
    bool has_partner = partner < npes;

    int* sym_buf = (int*) nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemset(sym_buf, 0, sizeof(int)));
    nvshmem_barrier_all();

    if (has_partner) {
        if (mype < partner) {
            kernel_pingpong_init<<<1, 1, 0, stream>>>((volatile int*) sym_buf, sym_buf, partner, WARMUP, ITERS, d_ts);
        } else {
            kernel_pingpong_resp<<<1, 1, 0, stream>>>((volatile int*) sym_buf, sym_buf, partner, WARMUP, ITERS);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvshmem_barrier_all();

    if (mype == 0 && has_partner) {
        auto ns = collect_timestamps(d_ts, ITERS);
        auto s = compute_stats(ns);
        printf("  %-12s  median=%8.0f ns  p5=%8.0f  p95=%8.0f  "
               "min=%8.0f  max=%8.0f\n",
               "pingpong", s.median, s.p5, s.p95, s.min, s.max);
        dump_csv(csv_file, "pingpong", nullptr, ns);
    }

    nvshmem_free(sym_buf);
}

static void run_put_bulk(int mype, int npes, cudaStream_t stream, long long* d_ts, const char* csv_file) {
    int target = (mype + 1) % npes;
    const std::vector<size_t> sizes = {16, 64, 256, 1024, 4096, 16384, 65536, 262144};

    size_t max_bytes = sizes.back();
    float* dest = (float*) nvshmem_malloc(max_bytes);
    float* src = (float*) nvshmem_malloc(max_bytes);
    CUDA_CHECK(cudaMemset(src, 0x42, max_bytes));

    if (mype == 0) {
        printf("  %-12s  %8s  %10s  %10s  %10s  %10s\n", "put_bulk", "bytes", "median_ns", "p5_ns", "p95_ns",
               "med_GB/s");
    }

    for (size_t bytes: sizes) {
        size_t n_elems = bytes / sizeof(float);
        char sz_str[32];
        snprintf(sz_str, sizeof(sz_str), "%zu", bytes);

        kernel_put_bulk_quiet<<<1, 1, 0, stream>>>(dest, src, n_elems, target, WARMUP, ITERS, d_ts);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        nvshmem_barrier_all();

        if (mype == 0) {
            auto ns = collect_timestamps(d_ts, ITERS);
            auto s = compute_stats(ns);
            double med_gbps = (s.median > 0) ? ((double) bytes / (s.median * 1e-9)) / 1e9 : 0;
            printf("  %-12s  %8zu  %10.0f  %10.0f  %10.0f  %10.3f\n", "", bytes, s.median, s.p5, s.p95, med_gbps);
            dump_csv(csv_file, "put_bulk", sz_str, ns);
        }
    }

    nvshmem_free(src);
    nvshmem_free(dest);
}

int main(int argc, char* argv[]) {
    const char* mode = (argc >= 2) ? argv[1] : "all";

    nvshmem_init();
    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    CUDA_CHECK(cudaSetDevice(mype));

    if (mype == 0) {
        printf("============================================\n");
        printf("SigmaTrace Doorbell Benchmark\n");
        printf("PEs=%d  warmup=%d  iters=%d  mode=%s\n", npes, WARMUP, ITERS, mode);
        printf("Timing: globaltimer (fixed-freq ns, DVFS-immune)\n");
        printf("============================================\n");
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    long long* d_ts;
    CUDA_CHECK(cudaMalloc(&d_ts, ITERS * sizeof(long long)));

    // CSV output file - append mode, one file per run
    const char* csv_file = "doorbell_ops.csv";
    if (mype == 0) {
        // Truncate at start of run
        FILE* f = fopen(csv_file, "w");
        if (f) {
            fclose(f);
        }
    }

    bool all = (strcmp(mode, "all") == 0);

    if (all || strcmp(mode, "put") == 0) {
        if (mype == 0) {
            printf("\n=== SCALAR OPS (per-op completion latency, ns) ===\n");
        }
        run_single_op("put", mype, npes, stream, d_ts, csv_file);
    }
    if (all || strcmp(mode, "get") == 0) {
        run_single_op("get", mype, npes, stream, d_ts, csv_file);
    }
    if (all || strcmp(mode, "atomic") == 0) {
        run_single_op("atomic", mype, npes, stream, d_ts, csv_file);
    }
    if (all || strcmp(mode, "pingpong") == 0) {
        if (mype == 0 && !all) {
            printf("\n=== PING-PONG RTT ===\n");
        }
        run_pingpong(mype, npes, stream, d_ts, csv_file);
    }
    if (all || strcmp(mode, "put_bulk") == 0) {
        if (mype == 0) {
            printf("\n=== BULK PUT (per-op completion latency, ns) ===\n");
        }
        run_put_bulk(mype, npes, stream, d_ts, csv_file);
    }

    CUDA_CHECK(cudaFree(d_ts));
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (mype == 0) {
        printf("\nResults written to %s\n", csv_file);
    }

    nvshmem_finalize();
    return 0;
}
