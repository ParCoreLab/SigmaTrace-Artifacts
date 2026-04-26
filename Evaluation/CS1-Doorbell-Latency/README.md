# Description

Per-operation overhead microbenchmark for SigmaTrace's UAR doorbell hijack
mechanism (paper §VI-B, Figs. 4a/4b). `doorbell_bench.cu` measures four
representative GPU-initiated NVSHMEM primitives - Put+Quiet, Get, Atomic
FetchAdd, and Ping-Pong RTT - inside a persistent single-thread CUDA kernel
using the device `globaltimer` register. `run.sh` sweeps three configurations
(uninstrumented baseline, UAR forwarding only, full WQE parse + record)
across 5 repetitions on a single node with 4xA100-40GB and one
ConnectX-5 HCA.

This experiment requires a version of SigmaTrace with only UAR forwarding
enabled to measure its overhead in isolation.

# Result files

[Results/](Results/) contains the collected timing results used in the paper

[Figures/](Figures/) contains the plotted figures (Figs. 4a/4b).

# Experiment

### Building

```bash
mkdir build
cd build
cmake -S ..
make -j
```

### Running

_in `build/`_

```bash
# If needed
export CUDA_HOME=...
export NVSHMEM_HOME=...
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/:$NVSHMEM_HOME/lib/

# Required by run.sh for the uar_only and full_trace configurations.
# Point these at the SigmaTrace Profiler binary built with the matching trace mode.
export SIGMA_EXE_UAR_ONLY=/path/to/SigmaTrace/build/target/Profiler
export SIGMA_EXE_FULL=/path/to/SigmaTrace/build/target/Profiler

../run.sh
```

After benchmark is finished:
```bash
python ../plot_doorbell.py --datadir ./results --outdir ./figures --format png
```

Generated figures will be in `build/figures/`

### Expected output
```bash
> user:CS1-Doorbell-latency/build$ ../run.sh

============================================
=  Doorbell Benchmark (globaltimer)
=  PEs: 4  Mode: all
=  Runs: 5
============================================
[2026-04-24 14:57:57] config=baseline bench=all npes=4 runs=5
  run 1/5 ...
  run 2/5 ...
  run 3/5 ...
  run 4/5 ...
  run 5/5 ...
[2026-04-24 14:58:41] Done: results/baseline.log
[2026-04-24 14:58:41] config=uar_only bench=all npes=4 runs=5
  run 1/5 ...
  run 2/5 ...
  run 3/5 ...
  run 4/5 ...
  run 5/5 ...
[2026-04-24 14:59:18] Done: results/uar_only.log
[2026-04-24 14:59:18] config=full_trace bench=all npes=4 runs=5
  run 1/5 ...
  run 2/5 ...
  run 3/5 ...
  run 4/5 ...
  run 5/5 ...
[2026-04-24 14:59:59] Done: results/full_trace.log

Results in results/
total 52708
drwx------ 2 user user    4096 Apr 24 14:59 .
drwx------ 5 user user    4096 Apr 24 14:59 ..
-rw------- 1 user user    7258 Apr 24 14:58 baseline.log
-rw------- 1 user user 3598636 Apr 24 14:58 baseline_r1.csv
-rw------- 1 user user 3585564 Apr 24 14:58 baseline_r2.csv
-rw------- 1 user user 3586444 Apr 24 14:58 baseline_r3.csv
-rw------- 1 user user 3587200 Apr 24 14:58 baseline_r4.csv
-rw------- 1 user user 3596102 Apr 24 14:58 baseline_r5.csv
-rw------- 1 user user   13652 Apr 24 14:59 full_trace.log
-rw------- 1 user user 3595618 Apr 24 14:59 full_trace_r1.csv
-rw------- 1 user user 3595556 Apr 24 14:59 full_trace_r2.csv
-rw------- 1 user user 3594914 Apr 24 14:59 full_trace_r3.csv
-rw------- 1 user user 3595732 Apr 24 14:59 full_trace_r4.csv
-rw------- 1 user user 3595124 Apr 24 14:59 full_trace_r5.csv
-rw------- 1 user user   13390 Apr 24 14:59 uar_only.log
-rw------- 1 user user 3595860 Apr 24 14:58 uar_only_r1.csv
-rw------- 1 user user 3595854 Apr 24 14:58 uar_only_r2.csv
-rw------- 1 user user 3595714 Apr 24 14:59 uar_only_r3.csv
-rw------- 1 user user 3595812 Apr 24 14:59 uar_only_r4.csv
-rw------- 1 user user 3595910 Apr 24 14:59 uar_only_r5.csv

> user:CS1-Doorbell-latency/build$ python ../plot_doorbell.py

========================================================================
DOORBELL BENCHMARK SUMMARY (median ns across runs)
========================================================================

  op                  Baseline     UAR forward      Full trace
  ────────────  ──────────────  ──────────────  ──────────────
  Put+Quiet            18208ns      18816ns(+3%)     22304ns(+22%)
  Get                  15552ns      16192ns(+4%)     19616ns(+26%)
  Atomic FetchAdd      15776ns      16128ns(+2%)     20160ns(+28%)
  Ping-Pong (RTT)      18784ns      19616ns(+4%)     27296ns(+45%)

Wrote figures/doorbell_latency.png
Wrote figures/doorbell_overhead_relative.png
Wrote figures/doorbell_overhead_absolute.png
Wrote figures/doorbell_bulk_latency.png
Wrote figures/doorbell_bulk_throughput.png
```

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

For plotting script:
- Python 3.10.12
- numpy==2.2.6
- matplotlib==3.10.9
- pandas==2.3.3
- seaborn==0.13.2

NVSHMEM build info:
```
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

