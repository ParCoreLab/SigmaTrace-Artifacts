#!/bin/bash
# Usage: bash launch_sharegpt.sh [output_file.json]

set -e

HEAD_NODE="${HEAD_NODE:-localhost}"
MODEL_PATH="${MODEL_PATH:-/path/to/DeepSeek-V3-0324}"
DATASET_PATH="${DATASET_PATH:-/path/to/ShareGPT_V3_unfiltered_cleaned_split.json}"
OUTPUT_FILE="${1:-./bench_result.json}"
# Override CLUSTER_BIND to a cluster-specific bind mount (e.g. /scratch:/scratch)
CLUSTER_BIND="${CLUSTER_BIND:-}"

echo "=== ShareGPT benchmark ==="
echo "Server: http://${HEAD_NODE}:30000"
echo "Model:  ${MODEL_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo "Start:  $(date)"
BENCH_START=$(date +%s)

apptainer exec --nv ${CLUSTER_BIND:+--bind ${CLUSTER_BIND}} ./sglang.sif python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url "http://${HEAD_NODE}:30000" \
  --model "${MODEL_PATH}" \
  --dataset-name sharegpt \
  --dataset-path "${DATASET_PATH}" \
  --num-prompts 200 \
  --request-rate 4 \
  --output-file "${OUTPUT_FILE}"

BENCH_END=$(date +%s)
echo ""
echo "End:       $(date)"
echo "Wall time: $((BENCH_END - BENCH_START))s"