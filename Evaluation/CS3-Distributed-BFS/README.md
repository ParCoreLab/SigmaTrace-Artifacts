# Description

Distributed level-synchronous BFS as a single persistent NVSHMEM kernel
across 4 GPUs (paper §VI-D, Figs. 7/8). `bfs_nvshmem_persist.cu`
traverses SNAP's `soc-LiveJournal1` graph (4.8M vertices, 68M directed
edges) under 1D vertex partitioning, using `nvshmem_int_atomic_compare_swap`
for cross-partition visit checks and `nvshmemx_barrier_all_block` between
BFS levels. Because partition 0 owns the highest-degree hub vertices,
early levels exhibit a strong RDMA traffic skew toward GPU 0 followed by
straggler barriers that are visible in SigmaTrace's per-doorbell
timeline and SQL barrier analysis.

# Result files

SigmaTrace filtered Perfetto trace in [Results/bfs-persist-livejournal-barrieronly.pftrace](Results/bfs-persist-livejournal-barrieronly.pftrace)
(Figures 7b, 8)

Nsight Systems trace (Figure 7a) is not included in this repository. The `.nsys-rep`
format embeds the profiling environment (command line, working directory) verbatim,
which prevents anonymous distribution. The trace was collected with the command shown
in the Running section below under **With Nsight Systems**.

# Experiment

### Building

```bash
# Optional
export CUDA_HOME=...
export NVSHMEM_HOME=...

mkdir build
cd build
cmake -S ..
make -j
```

### Running

Set the following environment variables to force NIC access as described in the paper
```bash
export NVSHMEM_DISABLE_P2P=1
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IB_DISABLE_DMABUF=1
export NVSHMEM_REMOTE_TRANSPORT=ibdevx
```

**Baseline:**
```bash
nvshmrun -np 4 \
  ./build/bfs_nvshmem_persist --graph csr \
  --graph-csr /path/to/soc-LiveJournal1_di.csr \
  --partition 1d -r 1
```

**With SigmaTrace:**
```bash
export SIGMA_EXE="/path/to/SigmaTrace/build/target/Profiler"

nvshmrun -np 4 \
  "$SIGMA_EXE" -f --enable-doorbell-hijack \
  ./build/bfs_nvshmem_persist --graph csr \
  --graph-csr /path/to/soc-LiveJournal1_di.csr \
  --partition 1d -r 1
```

**With Nsight Systems:**
```bash
nsys profile nvshmrun -np 4 \
  "$SIGMA_EXE" -f --enable-doorbell-hijack \
  ./build/bfs_nvshmem_persist --graph csr \
  --graph-csr /path/to/soc-LiveJournal1_di.csr \
  --partition 1d -r 1
```

### Expected output

```
[Launcher] Launching ./bfs_nvshmem_persist
[Profiler] loaded Ibverbs Trace
[Profiler] Loaded UCX Trace
[Profiler] loaded mlx5dv Trace
[Profiler] Loaded Doorbell Hijack Trace

  Selected block size: 1024 (collective max grid=108)
Loading directed CSR graph from /path/to/soc-LiveJournal1_di.csr ...
SigmaTrace Persistent BFS Benchmark
  Graph: directed CSR binary (/path/to/soc-LiveJournal1_di.csr)
  Vertices: 4847571  Directed edges: 68475391
  PEs: 4  Partition: 1d
  Local max verts/PE: 1211893  Roots: 1  Validate: no
  Part->GPU order: 0 1 2 3

Running persistent BFS...
  Root   361075: 40409.32 ms, 15 levels, PE0 visited 1179305

Summary (1 roots):
  Median: 40409.32 ms
  Mean:   40409.32 ms
[Profiler] num_nic_doorbell_accesses: 0
[Profiler] num_fake_nic_accesses: 383975
[Profiler] doorbell_submitted: 84541
[Profiler] doorbell_skipped: 299434
[Profiler] Exited Normally!
```

### Trace Visualization

Run the correlator to get perfetto output
```bash
/path/to/SigmaTrace/target/ProfilerEventReader ./EventRecorder/ bfs-nvshmem
```

This will produce `bfs-nvshmem.pftrace` (perfetto traces) and
`bfs-nvshmem-livejournal.commgraph.json` (aggregate communication in JSON).

Open the perfetto trace in https://ui.perfetto.dev/

### SQL Analysis:
See [SQL.md](SQL.md)

# Environment

### Hardware

```bash
$ nvidia-smi

Fri Apr 24 16:30:48 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-40GB          Off |   00000000:01:00.0 Off |                    0 |
| N/A   30C    P0             60W /  400W |      20MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA A100-SXM4-40GB          Off |   00000000:41:00.0 Off |                    0 |
| N/A   31C    P0             61W /  400W |     104MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA A100-SXM4-40GB          Off |   00000000:81:00.0 Off |                    0 |
| N/A   28C    P0             58W /  400W |     122MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA A100-SXM4-40GB          Off |   00000000:C1:00.0 Off |                    0 |
| N/A   28C    P0             60W /  400W |      74MiB /  40960MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

```bash
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV4     NV4     NV4     NODE    0-63,128-191    0               N/A
GPU1    NV4      X      NV4     NV4     NODE    0-63,128-191    0               N/A
GPU2    NV4     NV4      X      NV4     SYS     64-127,192-255  1               N/A
GPU3    NV4     NV4     NV4      X      SYS     64-127,192-255  1               N/A
NIC0    NODE    NODE    SYS     SYS      X
```

```bash
$ ibv_devinfo --list
1 HCA found:
        mlx5_0

$ ibv_devinfo
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         16.35.8002
        node_guid:                      xxxx:xxxx:xxxx:xxxx
        sys_image_guid:                 xxxx:xxxx:xxxx:xxxx
        vendor_id:                      0x02c9
        vendor_part_id:                 4119
        hw_ver:                         0x0
        board_id:                       MT_xxxxxxxxxx
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 14
                        port_lid:               17
                        port_lmc:               0x00
                        link_layer:             InfiniBand
```

### Software

**SigmaTrace build:**
- clang 20.1.8
- cmake 4.2.2
- abseil 20260107.0
- boost 1.89.0

**Runtime:**
- CUDA Toolkit 13.0.48 (NVHPC 25.9)
- rdma-core 1.14.58.0
- UCX 1.20.0

```bash
$ nvshmem-info -a
NVSHMEM v3.3.24

Build Information:
  CUDA API                     13000
  CUDA Driver                  13000
  Build Timestamp              Aug 19 2025 15:40:32
  Build Variables
        NVSHMEM_DEBUG=OFF NVSHMEM_DEVEL=OFF NVSHMEM_DEFAULT_PMI2=OFF
        NVSHMEM_DEFAULT_PMIX=OFF NVSHMEM_DEFAULT_UCX=OFF
        NVSHMEM_ENABLE_ALL_DEVICE_INLINING=OFF NVSHMEM_GPU_COLL_USE_LDST=OFF
        NVSHMEM_IBGDA_SUPPORT=ON NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=OFF
        NVSHMEM_IBDEVX_SUPPORT=ON NVSHMEM_IBRC_SUPPORT=ON NVSHMEM_LIBFABRIC_SUPPORT=ON
        NVSHMEM_MPI_SUPPORT=ON NVSHMEM_NVTX=ON NVSHMEM_PMIX_SUPPORT=ON
        NVSHMEM_SHMEM_SUPPORT=ON NVSHMEM_TEST_STATIC_LIB=OFF
        NVSHMEM_TIMEOUT_DEVICE_POLLING=OFF NVSHMEM_TRACE=OFF NVSHMEM_UCX_SUPPORT=ON
        NVSHMEM_USE_DLMALLOC=OFF NVSHMEM_USE_NCCL=ON NVSHMEM_USE_GDRCOPY=ON
        NVSHMEM_VERBOSE=OFF
        CUDA_HOME=/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/build/cuda
        GDRCOPY_HOME=/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/gdrcopy_build
        LIBFABRIC_HOME=/data/tmp/libfabric_install MPI_HOME=/data/tmp/openmpi_install
        NCCL_HOME=/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/nccl/build
        NVSHMEM_PREFIX=/dvs/p4/build/sw/rel/gpgpu/toolkit/r13.0/main_nvshmem/install
        PMIX_HOME=/data/tmp/pmix_install SHMEM_HOME=/data/tmp/openmpi_install
        UCX_HOME=/dvs/p4/build/sw/gpgpu/openucx/build
```