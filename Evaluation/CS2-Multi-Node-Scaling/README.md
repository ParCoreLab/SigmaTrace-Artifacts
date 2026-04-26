# Description

Multi-node tracing overhead benchmark (paper §VI-C, Figs. 5/6 and
Table I). This study runs the standard NCCL `all_reduce_perf` benchmark
from 2 to 32 nodes (8 -> 128 H100 GPUs over NDR400 InfiniBand) at 1 MB and
128 MB message sizes, comparing the uninstrumented baseline against
SigmaTrace's three trace modes (`aggregate`, `simple`, `full`). The SLURM
driver `sigma-nccl.sh` submits one job per <nodes, message size, mode,
repetition> tuple. The same workflow with `libfabric/rccl-tests` reproduces
the Slingshot/RCCL numbers quoted inline in §VI-C.

# Result files

[Results/](Results/) contains the parsed timing results for NCCL 1M, 128M, and RCCL 128M runs

[Figures/](Figures/) contains the plotted figures.

Full traces are not includes due to size limitations.

# Experiment

### Building

Ensure [NCCL tests](https://github.com/NVIDIA/nccl-tests/tree/v2.18.2) are available and built with MPI support:

Bundled repo with tag `v2.18.2`:
```bash
git submodule update --init --recursive ./nccl-tests
cd nccl-tests
make MPI=1 NAME_SUFFIX=_mpi -j
ls all_reduce_perf_mpi # Ensure the benchmark binary is available
```

### Running

Sample sbatch script is provided in `./sigma-nccl.sh`

Example submission:
```bash
export SCRATCH_DIR="/path/to/storage"

export SIGMA_EXE="/path/to/SigmaTrace/build/target/Profiler"
export NCCL_EXE="/path/to/nccl-tests/build/all_reduce_perf_mpi"

sbatch ./sigma-nccl.sh
```

### RCCL variant:

Update `./sigma-nccl.sh` with the RCCL/rccl-tests paths and ensure SigmaTrace is built
with `libfabric` support

### Parsing and plotting

The generated results can be parsed into csv files as follows:

```bash
python ./parse_nccl_results.py /path/to/output
```

The generated csv files can then be plotted as follows:

```bash
python ./generate_figures.py --datadir ./results --outdir ./figures --format png
```

# Environment

### Hardware

```bash
$ nvidia-smi

Fri Apr 24 15:04:58 2026
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10              Driver Version: 535.86.10    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA H100                    On  | 00000000:1B:00.0 Off |                    0 |
| N/A   40C    P0              68W / 700W |      2MiB / 65247MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA H100                    On  | 00000000:2C:00.0 Off |                    0 |
| N/A   40C    P0              64W / 700W |      2MiB / 65247MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   2  NVIDIA H100                    On  | 00000000:9D:00.0 Off |                    0 |
| N/A   40C    P0              65W / 700W |      2MiB / 65247MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
|   3  NVIDIA H100                    On  | 00000000:AD:00.0 Off |                    0 |
| N/A   40C    P0              71W / 700W |      2MiB / 65247MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

```bash
$ nvidia-smi topo -m
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV6     NV6     NV6     PXB     SYS     SYS     SYS     SYS     SYS     0-19,80-99      0               N/A
GPU1    NV6      X      NV6     NV6     SYS     PXB     NODE    NODE    SYS     SYS     20-39,100-119   1               N/A
GPU2    NV6     NV6      X      NV6     SYS     SYS     SYS     SYS     PXB     SYS     40-59,120-139   2               N/A
GPU3    NV6     NV6     NV6      X      SYS     SYS     SYS     SYS     SYS     PXB     60-79,140-159   3               N/A
NIC0    PXB     SYS     SYS     SYS      X      SYS     SYS     SYS     SYS     SYS
NIC1    SYS     PXB     SYS     SYS     SYS      X      NODE    NODE    SYS     SYS
NIC2    SYS     NODE    SYS     SYS     SYS     NODE     X      PIX     SYS     SYS
NIC3    SYS     NODE    SYS     SYS     SYS     NODE    PIX      X      SYS     SYS
NIC4    SYS     SYS     PXB     SYS     SYS     SYS     SYS     SYS      X      SYS
NIC5    SYS     SYS     SYS     PXB     SYS     SYS     SYS     SYS     SYS      X
```

```bash
$ ibv_devinfo --list
6 HCAs found:
        mlx5_0
        mlx5_1
        mlx5_2
        mlx5_3
        mlx5_4
        mlx5_5

$ ibv_devinfo
hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         28.36.1010
        node_guid:                      xxxx:xxxx:xxxx:xxxx
        sys_image_guid:                 xxxx:xxxx:xxxx:xxxx
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       MT_xxxxxxxxxx
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 909
                        port_lid:               1422
                        port_lmc:               0x00
                        link_layer:             InfiniBand
```

### Software

**SigmaTrace build:**
- gcc 13.2.0
- cmake 4.1.2
- boost 1.84.0

**Runtime:**
- CUDA Toolkit 12.9 (NVHPC 25.7)
- NCCL 2.24.3-1
- rdma-core 1.14.47.0
- UCX 1.16.0

For doorbell benchmark code:
- CUDA toolkit 13.0 (NVHPC v25.9)
- NVSHMEM version v3.3.24