/*
 * Persistent Multi-GPU BFS with NVSHMEM
 *
 * One persistent kernel executes the full level-synchronous BFS.
 * Inter-level synchronization uses nvshmemx_barrier_all_block().
 * Kernel launch uses nvshmemx_collective_launch (cooperative launch).
 */

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <nvshmem.h>
#include <nvshmemx.h>

#ifdef USE_METIS
#include <metis.h>
#endif

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err__ = (call);                                                                                    \
        if (err__ != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__));               \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define NVSHMEM_CHECK(call)                                                                                            \
    do {                                                                                                               \
        int status__ = (call);                                                                                         \
        if (status__ != 0) {                                                                                           \
            fprintf(stderr, "NVSHMEM error at %s:%d: status=%d\n", __FILE__, __LINE__, status__);                      \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

struct Edge {
    int src;
    int dst;
};

struct HostCSR {
    int num_verts = 0;
    int num_edges = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
};

struct PartitionData {
    int num_global_verts = 0;
    int num_local_verts = 0;
    int max_local_verts = 0;
    std::vector<int> owner_of_vertex;
    std::vector<int> local_index_of_vertex;
    std::vector<int> base_part_of_vertex;
};

struct DeviceGraph {
    int num_local_verts = 0;
    int num_global_verts = 0;
    int max_local_verts = 0;

    int* row_offsets = nullptr;
    int* col_indices = nullptr;

    // Symmetric (nvshmem_malloc), allocated at max_local_verts on every PE.
    int* visited = nullptr;
    int* frontier_a = nullptr;
    int* frontier_b = nullptr;
    int* count_a = nullptr;
    int* count_b = nullptr;
    int* level = nullptr;

    // Device-resident global mapping arrays
    int* d_owner_of_vertex = nullptr;
    int* d_local_index_of_vertex = nullptr;
};

enum class GraphMode {
    RMAT,
    FILE_EDGELIST,
    CSR_BINARY
};

enum class PartitionMode {
    PART_1D,
    PART_METIS
};

static void usage() {
    printf("Usage: nvshmrun -np <N> ./bfs_nvshmem_persist [options]\n"
           "\n"
           "Graph options:\n"
           "  --graph rmat|file|csr   Graph source (default: rmat)\n"
           "  --graph-file <path>     Directed edge list file (src dst per line)\n"
           "  --graph-csr <path>      Directed CSR binary file\n"
           "  -s <scale>              R-MAT scale (vertices=2^scale, default 18)\n"
           "  -e <efactor>            R-MAT edge factor (default 16)\n"
           "\n"
           "Partition options:\n"
           "  --partition 1d|metis    Partition strategy (default: 1d)\n"
           "  --gpu-order a,b,c,...   Part->GPU permutation (default identity)\n"
           "  --gpu-order-file <f>    Read permutation CSV from file\n"
           "\n"
           "Run options:\n"
           "  -r <nroots>             Number of BFS roots (default 16)\n"
           "  -v                      Validate first root against CPU BFS\n"
           "  -h                      Show this help\n");
}

static bool parse_gpu_order_csv(const std::string& csv, int npes, std::vector<int>* out_order) {
    std::vector<int> vals;
    std::string token;
    std::stringstream ss(csv);
    while (std::getline(ss, token, ',')) {
        std::stringstream ts(token);
        int x;
        if (!(ts >> x)) {
            return false;
        }
        vals.push_back(x);
    }
    if ((int) vals.size() != npes)
        return false;
    std::vector<int> seen(npes, 0);
    for (int x: vals) {
        if (x < 0 || x >= npes || seen[x])
            return false;
        seen[x] = 1;
    }
    *out_order = vals;
    return true;
}

static bool read_gpu_order_file(const std::string& path, int npes, std::vector<int>* out_order) {
    std::ifstream in(path);
    if (!in)
        return false;
    std::string line;
    while (std::getline(in, line)) {
        bool all_space = true;
        for (char c: line) {
            if (!(c == ' ' || c == '\t' || c == '\r' || c == '\n')) {
                all_space = false;
                break;
            }
        }
        if (all_space)
            continue;
        if (!line.empty() && line[0] == '#')
            continue;
        return parse_gpu_order_csv(line, npes, out_order);
    }
    return false;
}

static std::vector<Edge> generate_rmat_edges(int scale, int edge_factor, uint64_t seed) {
    int nverts = 1 << scale;
    int64_t nedges = (int64_t) edge_factor * nverts;
    double a = 0.57, b = 0.19, c = 0.19;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(nedges));

    for (int64_t e = 0; e < nedges; ++e) {
        int u = 0;
        int v = 0;
        for (int depth = scale - 1; depth >= 0; --depth) {
            double r = dist(rng);
            int u_bit;
            int v_bit;
            if (r < a) {
                u_bit = 0;
                v_bit = 0;
            } else if (r < a + b) {
                u_bit = 0;
                v_bit = 1;
            } else if (r < a + b + c) {
                u_bit = 1;
                v_bit = 0;
            } else {
                u_bit = 1;
                v_bit = 1;
            }
            u |= (u_bit << depth);
            v |= (v_bit << depth);
        }
        if (u != v) {
            edges.push_back({u, v});
        }
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.src < rhs.src || (lhs.src == rhs.src && lhs.dst < rhs.dst);
    });
    edges.erase(std::unique(edges.begin(), edges.end(),
                            [](const Edge& lhs, const Edge& rhs) {
        return lhs.src == rhs.src && lhs.dst == rhs.dst;
    }),
                edges.end());

    return edges;
}

static std::vector<Edge> load_directed_edge_list(const std::string& path, int* nverts_out) {
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "Failed to open graph file: %s\n", path.c_str());
        exit(1);
    }

    std::vector<Edge> edges;
    std::string line;
    int max_v = -1;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        if (line[0] == '#')
            continue;
        std::stringstream ss(line);
        int u, v;
        if (!(ss >> u >> v))
            continue;
        if (u < 0 || v < 0 || u == v)
            continue;
        edges.push_back({u, v});
        if (u > max_v)
            max_v = u;
        if (v > max_v)
            max_v = v;
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.src < rhs.src || (lhs.src == rhs.src && lhs.dst < rhs.dst);
    });
    edges.erase(std::unique(edges.begin(), edges.end(),
                            [](const Edge& lhs, const Edge& rhs) {
        return lhs.src == rhs.src && lhs.dst == rhs.dst;
    }),
                edges.end());

    *nverts_out = max_v + 1;
    if (*nverts_out <= 0) {
        fprintf(stderr, "Graph appears empty after filtering: %s\n", path.c_str());
        exit(1);
    }
    return edges;
}

static std::vector<Edge> load_directed_csr_binary(const std::string& path, int* nverts_out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        fprintf(stderr, "Failed to open CSR file: %s\n", path.c_str());
        exit(1);
    }

    int32_t nverts32 = 0;
    int32_t nedges32 = 0;
    in.read(reinterpret_cast<char*>(&nverts32), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&nedges32), sizeof(int32_t));
    if (!in || nverts32 <= 0 || nedges32 < 0) {
        fprintf(stderr, "Invalid CSR header in file: %s\n", path.c_str());
        exit(1);
    }

    int nverts = static_cast<int>(nverts32);
    int nedges = static_cast<int>(nedges32);

    std::vector<int32_t> row_offsets(static_cast<size_t>(nverts) + 1);
    std::vector<int32_t> col_indices(static_cast<size_t>(nedges));
    in.read(reinterpret_cast<char*>(row_offsets.data()),
            static_cast<std::streamsize>((static_cast<size_t>(nverts) + 1) * sizeof(int32_t)));
    in.read(reinterpret_cast<char*>(col_indices.data()),
            static_cast<std::streamsize>(static_cast<size_t>(nedges) * sizeof(int32_t)));
    if (!in) {
        fprintf(stderr, "Failed to read CSR payload from file: %s\n", path.c_str());
        exit(1);
    }

    if (row_offsets[0] != 0 || row_offsets[nverts] != nedges) {
        fprintf(stderr, "CSR offsets are inconsistent in file: %s\n", path.c_str());
        exit(1);
    }

    std::vector<Edge> edges;
    edges.reserve(static_cast<size_t>(nedges));
    for (int u = 0; u < nverts; ++u) {
        int start = static_cast<int>(row_offsets[u]);
        int end = static_cast<int>(row_offsets[u + 1]);
        if (start > end || start < 0 || end > nedges) {
            fprintf(stderr, "CSR row bounds are invalid in file: %s\n", path.c_str());
            exit(1);
        }
        for (int e = start; e < end; ++e) {
            int v = static_cast<int>(col_indices[e]);
            if (v < 0 || v >= nverts || v == u)
                continue;
            edges.push_back({u, v});
        }
    }

    std::sort(edges.begin(), edges.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.src < rhs.src || (lhs.src == rhs.src && lhs.dst < rhs.dst);
    });
    edges.erase(std::unique(edges.begin(), edges.end(),
                            [](const Edge& lhs, const Edge& rhs) {
        return lhs.src == rhs.src && lhs.dst == rhs.dst;
    }),
                edges.end());

    *nverts_out = nverts;
    return edges;
}

static HostCSR build_local_csr_from_edges(const std::vector<Edge>& edges, const PartitionData& P, int mype) {
    HostCSR csr;
    csr.num_verts = P.num_local_verts;
    csr.row_offsets.assign(csr.num_verts + 1, 0);

    for (const auto& e: edges) {
        int owner = P.owner_of_vertex[e.src];
        if (owner == mype) {
            int lsrc = P.local_index_of_vertex[e.src];
            csr.row_offsets[lsrc + 1]++;
        }
    }
    for (int i = 1; i <= csr.num_verts; ++i)
        csr.row_offsets[i] += csr.row_offsets[i - 1];

    csr.num_edges = csr.row_offsets[csr.num_verts];
    csr.col_indices.resize(csr.num_edges);
    std::vector<int> write_pos = csr.row_offsets;
    for (const auto& e: edges) {
        int owner = P.owner_of_vertex[e.src];
        if (owner == mype) {
            int lsrc = P.local_index_of_vertex[e.src];
            int pos = write_pos[lsrc]++;
            csr.col_indices[pos] = e.dst;
        }
    }
    return csr;
}

static void compute_partition_1d(int nverts, int npes, const std::vector<int>& part_to_pe, PartitionData* P) {
    int verts_per_part = (nverts + npes - 1) / npes;

    P->num_global_verts = nverts;
    P->owner_of_vertex.resize(nverts);
    P->local_index_of_vertex.resize(nverts);
    P->base_part_of_vertex.resize(nverts);

    std::vector<int> count_per_pe(npes, 0);
    for (int v = 0; v < nverts; ++v) {
        int base_part = v / verts_per_part;
        if (base_part >= npes)
            base_part = npes - 1;
        int owner = part_to_pe[base_part];
        int local = v - base_part * verts_per_part;

        P->base_part_of_vertex[v] = base_part;
        P->owner_of_vertex[v] = owner;
        P->local_index_of_vertex[v] = local;
        count_per_pe[owner]++;
    }

    P->max_local_verts = *std::max_element(count_per_pe.begin(), count_per_pe.end());
}

#ifdef USE_METIS
static std::vector<Edge> make_undirected_edges(const std::vector<Edge>& directed) {
    std::vector<Edge> und;
    und.reserve(directed.size() * 2);
    for (const auto& e: directed) {
        if (e.src == e.dst)
            continue;
        und.push_back({e.src, e.dst});
        und.push_back({e.dst, e.src});
    }
    std::sort(und.begin(), und.end(), [](const Edge& lhs, const Edge& rhs) {
        return lhs.src < rhs.src || (lhs.src == rhs.src && lhs.dst < rhs.dst);
    });
    und.erase(std::unique(und.begin(), und.end(),
                          [](const Edge& lhs, const Edge& rhs) {
        return lhs.src == rhs.src && lhs.dst == rhs.dst;
    }),
              und.end());
    return und;
}

static std::vector<int> compute_base_parts_metis(const std::vector<Edge>& directed, int nverts, int npes) {
    if (npes <= 1) {
        return std::vector<int>(nverts, 0);
    }

    std::vector<Edge> und = make_undirected_edges(directed);
    if (und.empty()) {
        return std::vector<int>(nverts, 0);
    }

    std::vector<idx_t> xadj(static_cast<size_t>(nverts) + 1, 0);
    for (const auto& e: und)
        xadj[e.src + 1]++;
    for (int i = 1; i <= nverts; ++i)
        xadj[i] += xadj[i - 1];
    std::vector<idx_t> adjncy(xadj[nverts]);
    std::vector<idx_t> cur = xadj;
    for (const auto& e: und)
        adjncy[cur[e.src]++] = static_cast<idx_t>(e.dst);

    idx_t nvtxs = static_cast<idx_t>(nverts);
    idx_t ncon = 1;
    idx_t nparts = static_cast<idx_t>(npes);
    idx_t edgecut = 0;
    std::vector<idx_t> options(METIS_NOPTIONS);
    METIS_SetDefaultOptions(options.data());
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_CONTIG] = 0;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_UFACTOR] = 30;
    options[METIS_OPTION_SEED] = 42;

    std::vector<idx_t> part(nverts, 0);
    int status;
    if (nparts <= 8) {
        status = METIS_PartGraphRecursive(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr, &nparts,
                                          nullptr, nullptr, options.data(), &edgecut, part.data());
    } else {
        status = METIS_PartGraphKway(&nvtxs, &ncon, xadj.data(), adjncy.data(), nullptr, nullptr, nullptr, &nparts,
                                     nullptr, nullptr, options.data(), &edgecut, part.data());
    }

    if (status != METIS_OK) {
        fprintf(stderr, "METIS partition failed: status=%d\n", status);
        exit(1);
    }

    std::vector<int> out(nverts);
    for (int i = 0; i < nverts; ++i)
        out[i] = static_cast<int>(part[i]);
    return out;
}
#endif

#ifdef USE_METIS
static void compute_partition_from_base_parts(const std::vector<int>& base_part, int npes,
                                              const std::vector<int>& part_to_pe, PartitionData* P) {
    int nverts = static_cast<int>(base_part.size());
    P->num_global_verts = nverts;
    P->base_part_of_vertex = base_part;
    P->owner_of_vertex.resize(nverts);
    P->local_index_of_vertex.resize(nverts);

    std::vector<int> count_per_pe(npes, 0);
    for (int v = 0; v < nverts; ++v) {
        int owner = part_to_pe[base_part[v]];
        P->owner_of_vertex[v] = owner;
        count_per_pe[owner]++;
    }

    std::vector<int> next(npes, 0);
    for (int v = 0; v < nverts; ++v) {
        int owner = P->owner_of_vertex[v];
        P->local_index_of_vertex[v] = next[owner]++;
    }

    P->max_local_verts = *std::max_element(count_per_pe.begin(), count_per_pe.end());
}
#endif

__global__ void persistent_bfs_kernel(const int* __restrict__ row_offsets, const int* __restrict__ col_indices,
                                      const int* __restrict__ owner_of_vertex,
                                      const int* __restrict__ local_index_of_vertex, int* visited, int* frontier_a,
                                      int* frontier_b, int* count_a, int* count_b, int* level_arr, int source_idx,
                                      int source_pe, int* d_num_levels) {

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // for (int i = 0; i < 100000; ++i) {
    //     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    //         nvshmemx_barrier_all_block();
    //     // }
    // }

    // return;

    __shared__ int s_frontier_size;
    __shared__ int s_global_frontier;
    __shared__ int s_local_new_count;
    __shared__ int s_global_new_count;

    if (tid == 0) {
        count_a[0] = 0;
        count_b[0] = 0;
        if (mype == source_pe) {
            visited[source_idx] = 1;
            level_arr[source_idx] = 0;
            frontier_a[0] = source_idx;
            count_a[0] = 1;
        }
    }
    __syncthreads();

    nvshmemx_barrier_all_block();
    // if (tid == 0) {
    //     printf("%d === ok ===\n", mype);
    // }

    int current_level = 0;
    int phase = 0;
    while (true) {
        int* f_in = (phase == 0) ? frontier_a : frontier_b;
        int* f_out = (phase == 0) ? frontier_b : frontier_a;
        int* cnt_in = (phase == 0) ? count_a : count_b;
        int* cnt_out = (phase == 0) ? count_b : count_a;

        if (tid == 0)
            s_frontier_size = cnt_in[0];
        __syncthreads();
        int frontier_size = s_frontier_size;

        for (int i = tid; i < frontier_size; i += nthreads) {
            int lsrc = f_in[i];
            int row_start = row_offsets[lsrc];
            int row_end = row_offsets[lsrc + 1];

            for (int e = row_start; e < row_end; ++e) {
                int nbr = col_indices[e];
                int dest_pe = owner_of_vertex[nbr];
                int dest_idx = local_index_of_vertex[nbr];

                if (dest_pe == mype) {
                    int old = atomicCAS(&visited[dest_idx], 0, 1);
                    if (old == 0) {
                        level_arr[dest_idx] = current_level + 1;
                        int pos = atomicAdd(cnt_out, 1);
                        f_out[pos] = dest_idx;
                    }
                } else {
                    int old = nvshmem_int_atomic_compare_swap(&visited[dest_idx], 0, 1, dest_pe);
                    if (old == 0) {
                        nvshmem_int_p(&level_arr[dest_idx], current_level + 1, dest_pe);
                        int pos = nvshmem_int_atomic_fetch_add(cnt_out, 1, dest_pe);
                        nvshmem_int_p(&f_out[pos], dest_idx, dest_pe);
                    }
                }
            }
        }

        // nvshmem_quiet();
        __syncthreads();

        nvshmemx_barrier_all_block();
        // if (tid == 0) {
        //     printf("=== ok ===\n");
        // }

        if (tid == 0) {
            s_local_new_count = cnt_out[0];
        }
        __syncthreads();

        // nvshmemx_int_sum_reduce_block(NVSHMEM_TEAM_WORLD,
        //                               &s_global_new_count,
        //                               &s_local_new_count,
        //                               1);

        // Replace the nvshmemx_int_sum_reduce_block block with:
        if (tid == 0) {
            int total = 0;
            for (int p = 0; p < npes; p++) {
                if (p == mype)
                    total += cnt_out[0];
                else {
                    total += nvshmem_int_g(cnt_out, p); // RDMA get
                }
            }
            s_global_frontier = total;
        }
        __syncthreads();

        // if (tid == 0) {
        //     s_global_frontier = s_global_new_count;
        // }
        // __syncthreads();

        if (s_global_frontier == 0) {
            if (tid == 0)
                *d_num_levels = current_level + 1;
            return;
        }

        if (tid == 0)
            cnt_in[0] = 0;
        __syncthreads();
        nvshmemx_barrier_all_block();
        // if (tid == 0) {
        //     printf("=== ok ===\n");
        // }

        phase = 1 - phase;
        current_level++;
    }
}

static int pick_collective_block_size(int mype) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (!prop.cooperativeLaunch) {
        if (mype == 0) {
            fprintf(stderr, "Device does not support cooperative launch required by persistent kernel.\n");
        }
        exit(1);
    }

    const int candidates[] = {1024, 768, 512, 384, 256, 128};
    for (int block_size: candidates) {
        void* query_args[] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        int max_grid = 0;
        NVSHMEM_CHECK(nvshmemx_collective_launch_query_gridsize((const void*) persistent_bfs_kernel, dim3(block_size),
                                                                query_args, 0, &max_grid));
        if (max_grid >= 1) {
            if (mype == 0) {
                printf("  Selected block size: %d (collective max grid=%d)\n", block_size, max_grid);
                fflush(stdout);
            }
            return block_size;
        }
    }

    if (mype == 0) {
        fprintf(stderr, "Collective launch query reports no viable grid for tested block sizes.\n");
    }
    exit(1);
}

static float run_bfs(DeviceGraph& G, const PartitionData& P, int source_global, int* d_num_levels, int mype,
                     int block_size) {
    int source_pe = P.owner_of_vertex[source_global];
    int source_idx = P.local_index_of_vertex[source_global];

    CUDA_CHECK(cudaMemset(G.visited, 0, static_cast<size_t>(G.max_local_verts) * sizeof(int)));
    CUDA_CHECK(cudaMemset(G.count_a, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(G.count_b, 0, sizeof(int)));
    {
        std::vector<int> neg(static_cast<size_t>(G.max_local_verts), -1);
        CUDA_CHECK(cudaMemcpy(G.level, neg.data(), static_cast<size_t>(G.max_local_verts) * sizeof(int),
                              cudaMemcpyHostToDevice));
    }
    nvshmem_barrier_all();

    void* args[] = {&G.row_offsets, &G.col_indices, &G.d_owner_of_vertex, &G.d_local_index_of_vertex,
                    &G.visited,     &G.frontier_a,  &G.frontier_b,        &G.count_a,
                    &G.count_b,     &G.level,       &source_idx,          &source_pe,
                    &d_num_levels};

    auto t0 = std::chrono::high_resolution_clock::now();
    NVSHMEM_CHECK(
            nvshmemx_collective_launch((const void*) persistent_bfs_kernel, dim3(1), dim3(block_size), args, 0, 0));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    // if (source_pe == mype) {
    //     int seed_level = -1;
    //     CUDA_CHECK(cudaMemcpy(&seed_level, G.level + source_idx, sizeof(int), cudaMemcpyDeviceToHost));
    //     if (seed_level != 0) {
    //         fprintf(stderr, "Source level mismatch on PE %d for root %d (got %d).\n",
    //                 mype, source_global, seed_level);
    //         exit(1);
    //     }
    // }

    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

static std::vector<int> cpu_bfs(const std::vector<Edge>& edges, int nverts, int source) {
    std::vector<int> offsets(static_cast<size_t>(nverts) + 1, 0);
    for (const auto& e: edges)
        offsets[e.src + 1]++;
    for (int i = 1; i <= nverts; ++i)
        offsets[i] += offsets[i - 1];
    std::vector<int> cols(offsets[nverts]);
    std::vector<int> tmp = offsets;
    for (const auto& e: edges)
        cols[tmp[e.src]++] = e.dst;

    std::vector<int> dist(nverts, -1);
    dist[source] = 0;
    std::vector<int> q;
    q.push_back(source);
    int head = 0;
    while (head < static_cast<int>(q.size())) {
        int v = q[head++];
        for (int ei = offsets[v]; ei < offsets[v + 1]; ++ei) {
            int u = cols[ei];
            if (dist[u] == -1) {
                dist[u] = dist[v] + 1;
                q.push_back(u);
            }
        }
    }
    return dist;
}

int main(int argc, char** argv) {
    int scale = 18;
    int edge_factor = 16;
    int nroots = 16;
    bool validate = false;

    GraphMode graph_mode = GraphMode::RMAT;
    std::string graph_file;
    std::string graph_csr_file;
    PartitionMode part_mode = PartitionMode::PART_1D;

    std::string gpu_order_csv;
    std::string gpu_order_file;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            scale = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-e") && i + 1 < argc) {
            edge_factor = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-r") && i + 1 < argc) {
            nroots = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "-v")) {
            validate = true;
        } else if (!strcmp(argv[i], "--graph") && i + 1 < argc) {
            const char* mode = argv[++i];
            if (!strcmp(mode, "rmat"))
                graph_mode = GraphMode::RMAT;
            else if (!strcmp(mode, "file"))
                graph_mode = GraphMode::FILE_EDGELIST;
            else if (!strcmp(mode, "csr"))
                graph_mode = GraphMode::CSR_BINARY;
            else {
                fprintf(stderr, "Unknown --graph mode: %s\n", mode);
                return 1;
            }
        } else if (!strcmp(argv[i], "--graph-file") && i + 1 < argc) {
            graph_file = argv[++i];
        } else if (!strcmp(argv[i], "--graph-csr") && i + 1 < argc) {
            graph_csr_file = argv[++i];
        } else if (!strcmp(argv[i], "--partition") && i + 1 < argc) {
            const char* mode = argv[++i];
            if (!strcmp(mode, "1d"))
                part_mode = PartitionMode::PART_1D;
            else if (!strcmp(mode, "metis"))
                part_mode = PartitionMode::PART_METIS;
            else {
                fprintf(stderr, "Unknown --partition mode: %s\n", mode);
                return 1;
            }
        } else if (!strcmp(argv[i], "--gpu-order") && i + 1 < argc) {
            gpu_order_csv = argv[++i];
        } else if (!strcmp(argv[i], "--gpu-order-file") && i + 1 < argc) {
            gpu_order_file = argv[++i];
        } else if (!strcmp(argv[i], "-h")) {
            usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage();
            return 1;
        }
    }

    if (graph_mode == GraphMode::FILE_EDGELIST && graph_file.empty()) {
        fprintf(stderr, "--graph file requires --graph-file <path>\n");
        return 1;
    }
    if (graph_mode == GraphMode::CSR_BINARY && graph_csr_file.empty()) {
        fprintf(stderr, "--graph csr requires --graph-csr <path>\n");
        return 1;
    }

    nvshmem_init();

    int mype = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    CUDA_CHECK(cudaSetDevice(mype_node));

    std::vector<int> part_to_pe(npes);
    std::iota(part_to_pe.begin(), part_to_pe.end(), 0);
    if (!gpu_order_file.empty()) {
        if (!read_gpu_order_file(gpu_order_file, npes, &part_to_pe)) {
            if (mype == 0) {
                fprintf(stderr, "Failed to parse --gpu-order-file %s\n", gpu_order_file.c_str());
            }
            nvshmem_finalize();
            return 1;
        }
    }
    if (!gpu_order_csv.empty()) {
        if (!parse_gpu_order_csv(gpu_order_csv, npes, &part_to_pe)) {
            if (mype == 0) {
                fprintf(stderr, "Invalid --gpu-order value '%s' (must be a permutation of 0..%d)\n",
                        gpu_order_csv.c_str(), npes - 1);
            }
            nvshmem_finalize();
            return 1;
        }
    }

    int block_size = pick_collective_block_size(mype);

    if (part_mode == PartitionMode::PART_METIS) {
#ifndef USE_METIS
        if (mype == 0) {
            fprintf(stderr, "Partition mode 'metis' requested, but binary was built without USE_METIS=1.\n");
        }
        nvshmem_finalize();
        return 1;
#endif
    }

    std::vector<Edge> edges;
    int nverts = 0;
    if (graph_mode == GraphMode::RMAT) {
        nverts = 1 << scale;
        if (mype == 0) {
            printf("Generating directed R-MAT graph...\n");
            fflush(stdout);
        }
        edges = generate_rmat_edges(scale, edge_factor, 42);
    } else if (graph_mode == GraphMode::FILE_EDGELIST) {
        if (mype == 0) {
            printf("Loading directed edge-list graph from %s ...\n", graph_file.c_str());
            fflush(stdout);
        }
        edges = load_directed_edge_list(graph_file, &nverts);
    } else {
        if (mype == 0) {
            printf("Loading directed CSR graph from %s ...\n", graph_csr_file.c_str());
            fflush(stdout);
        }
        edges = load_directed_csr_binary(graph_csr_file, &nverts);
    }

    if (part_mode == PartitionMode::PART_METIS && nverts < npes) {
        if (mype == 0) {
            fprintf(stderr, "METIS partitioning requires vertices >= PEs (%d < %d).\n", nverts, npes);
        }
        nvshmem_finalize();
        return 1;
    }

    PartitionData P;
    if (part_mode == PartitionMode::PART_1D) {
        compute_partition_1d(nverts, npes, part_to_pe, &P);
    } else {
#ifdef USE_METIS
        std::vector<int> base_parts;
        size_t part_bytes = static_cast<size_t>(nverts) * sizeof(int);
        int* sym_base_parts = (int*) nvshmem_malloc(part_bytes);
        if (!sym_base_parts) {
            if (mype == 0) {
                fprintf(stderr, "nvshmem_malloc failed for METIS partition vector\n");
            }
            nvshmem_finalize();
            return 1;
        }

        if (mype == 0) {
            base_parts = compute_base_parts_metis(edges, nverts, npes);
            CUDA_CHECK(cudaMemcpy(sym_base_parts, base_parts.data(), part_bytes, cudaMemcpyHostToDevice));
        }
        nvshmem_barrier_all();

        if (mype == 0) {
            for (int p = 1; p < npes; ++p) {
                nvshmem_putmem(sym_base_parts, sym_base_parts, part_bytes, p);
            }
        }
        nvshmem_barrier_all();

        base_parts.resize(nverts);
        CUDA_CHECK(cudaMemcpy(base_parts.data(), sym_base_parts, part_bytes, cudaMemcpyDeviceToHost));

        compute_partition_from_base_parts(base_parts, npes, part_to_pe, &P);
        nvshmem_barrier_all();
        nvshmem_free(sym_base_parts);
#endif
    }

    int local_count = 0;
    for (int v = 0; v < nverts; ++v) {
        if (P.owner_of_vertex[v] == mype)
            local_count++;
    }
    P.num_local_verts = local_count;

    if (mype == 0) {
        printf("SigmaTrace Persistent BFS Benchmark\n");
        if (graph_mode == GraphMode::RMAT) {
            printf("  Graph: R-MAT directed (scale=%d, efactor=%d)\n", scale, edge_factor);
        } else if (graph_mode == GraphMode::FILE_EDGELIST) {
            printf("  Graph: directed edge list (%s)\n", graph_file.c_str());
        } else {
            printf("  Graph: directed CSR binary (%s)\n", graph_csr_file.c_str());
        }
        printf("  Vertices: %d  Directed edges: %zu\n", nverts, edges.size());
        printf("  PEs: %d  Partition: %s\n", npes, (part_mode == PartitionMode::PART_1D) ? "1d" : "metis");
        printf("  Local max verts/PE: %d  Roots: %d  Validate: %s\n", P.max_local_verts, nroots,
               validate ? "yes" : "no");
        printf("  Part->GPU order:");
        for (int i = 0; i < npes; ++i)
            printf(" %d", part_to_pe[i]);
        printf("\n");
        fflush(stdout);
    }
    nvshmem_barrier_all();

    HostCSR local_csr = build_local_csr_from_edges(edges, P, mype);

    DeviceGraph G;
    G.num_local_verts = local_csr.num_verts;
    G.num_global_verts = nverts;
    G.max_local_verts = P.max_local_verts;

    CUDA_CHECK(cudaMalloc(&G.row_offsets, static_cast<size_t>(local_csr.num_verts + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.col_indices, static_cast<size_t>(local_csr.num_edges) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(G.row_offsets, local_csr.row_offsets.data(),
                          static_cast<size_t>(local_csr.num_verts + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.col_indices, local_csr.col_indices.data(),
                          static_cast<size_t>(local_csr.num_edges) * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&G.d_owner_of_vertex, static_cast<size_t>(nverts) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.d_local_index_of_vertex, static_cast<size_t>(nverts) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(G.d_owner_of_vertex, P.owner_of_vertex.data(), static_cast<size_t>(nverts) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_local_index_of_vertex, P.local_index_of_vertex.data(),
                          static_cast<size_t>(nverts) * sizeof(int), cudaMemcpyHostToDevice));

    G.visited = (int*) nvshmem_malloc(static_cast<size_t>(G.max_local_verts) * sizeof(int));
    G.frontier_a = (int*) nvshmem_malloc(static_cast<size_t>(G.max_local_verts) * sizeof(int));
    G.frontier_b = (int*) nvshmem_malloc(static_cast<size_t>(G.max_local_verts) * sizeof(int));
    G.count_a = (int*) nvshmem_malloc(sizeof(int));
    G.count_b = (int*) nvshmem_malloc(sizeof(int));
    G.level = (int*) nvshmem_malloc(static_cast<size_t>(G.max_local_verts) * sizeof(int));
    assert(G.visited && G.frontier_a && G.frontier_b && G.count_a && G.count_b && G.level);

    int* d_num_levels = nullptr;
    CUDA_CHECK(cudaMalloc(&d_num_levels, sizeof(int)));

    nvshmem_barrier_all();

    std::vector<int> roots;
    roots.reserve(nroots);
    {
        std::mt19937 rng(123);
        if (graph_mode == GraphMode::RMAT) {
            int range = std::max(1, nverts / 100);
            for (int i = 0; i < nroots; ++i)
                roots.push_back(rng() % range);
        } else {
            for (int i = 0; i < nroots; ++i)
                roots.push_back(rng() % nverts);
        }
    }

    if (mype == 0) {
        printf("\nRunning persistent BFS...\n");
        fflush(stdout);
    }

    std::vector<float> times;
    times.reserve(nroots);

    for (int r = 0; r < nroots; ++r) {
        int source = roots[r];
        float ms = run_bfs(G, P, source, d_num_levels, mype, block_size);
        times.push_back(ms);

        int h_levels = 0;
        CUDA_CHECK(cudaMemcpy(&h_levels, d_num_levels, sizeof(int), cudaMemcpyDeviceToHost));

        int visited_count = 0;
        {
            std::vector<int> h_vis(static_cast<size_t>(G.max_local_verts));
            CUDA_CHECK(cudaMemcpy(h_vis.data(), G.visited, static_cast<size_t>(G.max_local_verts) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < G.num_local_verts; ++i) {
                if (h_vis[i])
                    visited_count++;
            }
        }

        if (mype == 0) {
            printf("  Root %8d: %8.2f ms, %d levels, PE0 visited %d\n", source, ms, h_levels, visited_count);
            fflush(stdout);
        }

        if (validate && r == 0) {
            auto ref = cpu_bfs(edges, nverts, source);
            std::vector<int> gpu_lev(static_cast<size_t>(G.max_local_verts));
            CUDA_CHECK(cudaMemcpy(gpu_lev.data(), G.level, static_cast<size_t>(G.max_local_verts) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            int bad = 0;
            for (int v = 0; v < nverts; ++v) {
                if (P.owner_of_vertex[v] != mype)
                    continue;
                int lv = P.local_index_of_vertex[v];
                if (gpu_lev[lv] != ref[v]) {
                    if (bad < 5) {
                        fprintf(stderr, "  PE %d: v=%d gpu=%d cpu=%d\n", mype, v, gpu_lev[lv], ref[v]);
                    }
                    bad++;
                }
            }
            printf("  PE %d: Validation %s (%d mismatches)\n", mype, (bad == 0) ? "PASSED" : "FAILED", bad);
            fflush(stdout);
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        std::sort(times.begin(), times.end());
        float median = times[times.size() / 2];
        float mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
        printf("\nSummary (%d roots):\n", nroots);
        printf("  Median: %.2f ms\n", median);
        printf("  Mean:   %.2f ms\n", mean);
        fflush(stdout);
    }

    nvshmem_free(G.visited);
    nvshmem_free(G.frontier_a);
    nvshmem_free(G.frontier_b);
    nvshmem_free(G.count_a);
    nvshmem_free(G.count_b);
    nvshmem_free(G.level);
    CUDA_CHECK(cudaFree(G.row_offsets));
    CUDA_CHECK(cudaFree(G.col_indices));
    CUDA_CHECK(cudaFree(G.d_owner_of_vertex));
    CUDA_CHECK(cudaFree(G.d_local_index_of_vertex));
    CUDA_CHECK(cudaFree(d_num_levels));

    nvshmem_finalize();
    return 0;
}
