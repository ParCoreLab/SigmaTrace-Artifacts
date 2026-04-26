# Description

Mixture-of-Experts expert-parallelism case study (paper §VI-E, Fig. 9).
Serves DeepSeek-V3-0324 (671B parameters, FP8) under SGLang inside an
Apptainer image across 4 nodes with 4xH200 GPUs (16 GPUs total). Each MoE
layer has 256 routed experts plus one shared expert with top-8 routing
per token; under expert parallelism this places 16 routed experts on each
GPU. Attention runs in data-parallel mode (`--enable-dp-attention`).
`launch_deepseek_v3_baseline.sh` and `launch_deepseek_v3_profiler.sh`
start the server (the latter under the SigmaTrace `Profiler`);
`launch_sharegpt_bench.sh` replays 200 ShareGPT prompts through
`sglang.bench_serving`.

# Result files

[Results/sglang-deepseek.commgraph.json](Results/sglang-deepseek.commgraph.json) is the aggregate communication graph.

[Results/res.matrix.png](Results/res.matrix.png) is the communication matrix heatmap shown in the paper (Fig. 9).

[Results/res.png](Results/res.png) is the full node-level communication graph (same data, edge-annotated layout).

# Experiment

### Building

Get the DeepSeek V3 model with 671B parameters and FP8
```bash
hf download deepseek-ai/DeepSeek-V3-0324 --local-dir /path/to/deepseek-ai/DeepSeek-V3-0324
```

Get the sglang apptainer image
```bash
apptainer build sglang.sif docker://lmsysorg/sglang:dev-cu13
```

Get the ShareGPT V3 benchmark: https://www.kaggle.com/datasets/roschildrui/sharegpt-v3-unfiltered-cleaned-split

### Running

Set cluster-specific env vars before launching (both are optional with sensible defaults):
```bash
# Writable path for SGLang/Triton JIT caches (default: $HOME/.cache/sglang_<GPU>)
export SGLANG_CACHE_DIR="/path/to/writable/cache"

# Apptainer bind mount for your cluster's shared filesystem (leave unset if not needed)
export CLUSTER_BIND="/scratch:/scratch"
```

**Baseline:**
Launch the sglang server in all nodes to ensure it works
```bash
export MODEL_PATH="/path/to/DeepSeek-V3-0324"
srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 \
  bash ./launch_deepseek_v3_baseline.sh
```

Test the server once up:
```bash
curl head_node:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "messages": [{"role": "user", "content": "hello"}]
  }' | jq
```

**SigmaTrace**:
Launch the SigmaTrace profiler script in all nodes
```bash
export MODEL_PATH="/path/to/DeepSeek-V3-0324"
srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=4 \
  bash ./launch_deepseek_v3_profiler.sh
```
The script above will launch the sglang server under the SigmaTrace profiler
```bash
# Example
...
apptainer exec ...
./sglang.sif \
  ${SIGMA_EXE} python3 -m sglang.launch_server 
...
```

Launch the benchmark
```bash
export HEAD_NODE=127.0.0.1
export MODEL_PATH="/path/to/DeepSeek-V3-0324"
export DATASET_PATH="/path/to/ShareGPT.json"
./launch_sharegpt_bench.sh sigma_sharegpt.json
```

### Visualization

Run the correlator to get aggregate output
```bash
/path/to/SigmaTrace/target/ProfilerEventReader "$PROFILER_OUTPUT_DIR" sglang-deepseek
```

This will produce `sglang-deepseek.commgraph.json`

You can then use the bundled visualizer in SigmaTrace to parse the output

```bash
python /path/to/SigmaTrace/Profiler/Tools/visualize_commgraph.py -o res --level node --min-edge=10000000 --max-edge=30000000000 --show-nic sglang-deepseek.commgraph.json
```

# Environment

### Hardware

```bash
$ nvidia-smi

Fri Apr 24 16:47:25 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.05              Driver Version: 580.95.05      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H200                    On  |   00000000:42:00.0 Off |                    0 |
| N/A   39C    P0             82W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H200                    On  |   00000000:55:00.0 Off |                    0 |
| N/A   38C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H200                    On  |   00000000:D4:00.0 Off |                    0 |
| N/A   38C    P0             76W /  700W |       0MiB / 143771MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H200                    On  |   00000000:E6:00.0 Off |                    0 |
| N/A   38C    P0             79W /  700W |       0MiB / 143771MiB |      0%      Default |
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
        GPU0    GPU1    GPU2    GPU3    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    CPU Affinity    NUMA Affinity     GPU NUMA ID
GPU0     X      NV6     NV6     NV6     PXB     NODE    SYS     SYS     SYS     SYS     0-31,64-95      0        N/A
GPU1    NV6      X      NV6     NV6     NODE    PXB     SYS     SYS     SYS     SYS     0-31,64-95      0        N/A
GPU2    NV6     NV6      X      NV6     SYS     SYS     NODE    NODE    PXB     NODE    32-63,96-127    1        N/A
GPU3    NV6     NV6     NV6      X      SYS     SYS     NODE    NODE    NODE    PXB     32-63,96-127    1        N/A
NIC0    PXB     NODE    SYS     SYS      X      NODE    SYS     SYS     SYS     SYS
NIC1    NODE    PXB     SYS     SYS     NODE     X      SYS     SYS     SYS     SYS
NIC2    SYS     SYS     NODE    NODE    SYS     SYS      X      PIX     NODE    NODE
NIC3    SYS     SYS     NODE    NODE    SYS     SYS     PIX      X      NODE    NODE
NIC4    SYS     SYS     PXB     NODE    SYS     SYS     NODE    NODE     X      NODE
NIC5    SYS     SYS     NODE    PXB     SYS     SYS     NODE    NODE    NODE     X
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
        fw_ver:                         28.39.2048
        node_guid:                      xxxx:xxxx:xxxx:xxxx
        sys_image_guid:                 xxxx:xxxx:xxxx:xxxx
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       LNVxxxxxxxxxx
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 54
                        port_lid:               2092
                        port_lmc:               0x00
                        link_layer:             InfiniBand
```

### Software

**SigmaTrace build:**
- gcc 12.3.0
- cmake 3.31.0
- abseil 20260107.0
- boost 1.89.0

**Runtime:**
- CUDA Toolkit 13.0.88
- rdma-core 1.14.52.0
- UCX 1.17.0
- Python 3.12.3
- sglang 0.0.0.dev1+gafb32d762
- torch 2.9.1+cu130
- triton 3.5.1
