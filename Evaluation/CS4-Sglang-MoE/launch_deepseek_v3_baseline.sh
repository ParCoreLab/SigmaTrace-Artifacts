#!/bin/bash
NNODES=${SLURM_NNODES}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NODE_RANK=$SLURM_NODEID
MODEL_PATH="${MODEL_PATH:-/path/to/DeepSeek-V3-0324}"

GPU_ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
# Override SGLANG_CACHE_DIR to a writable path on your cluster
CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang_${GPU_ARCH}}"
mkdir -p "${CACHE_DIR}/deep_gemm" "${CACHE_DIR}/triton" "${CACHE_DIR}/inductor"

# Override CLUSTER_BIND to a cluster-specific bind mount (e.g. /scratch:/scratch)
CLUSTER_BIND="${CLUSTER_BIND:-}"

SOCK_IFNAME=$(apptainer exec --nv ${CLUSTER_BIND:+--bind ${CLUSTER_BIND}} ./sglang.sif \
  awk -F: 'NR>2 && $1 !~ /lo/{gsub(/ /,"",$1); print $1; exit}' /proc/net/dev)

echo "Starting DeepSeek-V3 on node ${NODE_RANK}/${NNODES}, HEAD=${HEAD_NODE}, IFNAME=${SOCK_IFNAME}, TOTAL_GPUS=${TOTAL_GPUS}, GPU=${GPU_ARCH}"

apptainer exec --nv ${CLUSTER_BIND:+--bind ${CLUSTER_BIND}} \
  --env GLOO_SOCKET_IFNAME=${SOCK_IFNAME} \
  --env NCCL_SOCKET_IFNAME=${SOCK_IFNAME} \
  --env NCCL_IB_GID_INDEX=3 \
  --env SGLANG_ENABLE_JIT_DEEPGEMM=0 \
  --env SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  --env TRITON_CACHE_DIR=${CACHE_DIR}/triton \
  --env TRITON_HOME=${CACHE_DIR}/triton \
  --env TORCHINDUCTOR_CACHE_DIR=${CACHE_DIR}/inductor \
  --env TORCH_EXTENSIONS_DIR=${CACHE_DIR}/torch_extensions \
  --env XDG_CACHE_HOME=${CACHE_DIR} \
  ./sglang.sif python3 -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --trust-remote-code \
  --tp-size ${TOTAL_GPUS} \
  --dp-size ${TOTAL_GPUS} \
  --dist-init-addr ${HEAD_NODE}:5000 \
  --nnodes ${NNODES} \
  --node-rank ${NODE_RANK} \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --mem-fraction-static 0.75 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-flashinfer-autotune \
  --skip-server-warmup \
  --watchdog-timeout 7200 --dist-timeout 7200 \
  --host 0.0.0.0 --port 30000
