#!/usr/bin/env bash
set -euo pipefail

BENCH_MODE="${1:-all}"
NPES="${NPES:-4}"
RUNS="${RUNS:-5}"
OUTDIR="results"
mkdir -p "$OUTDIR"

BENCH_EXE="$PWD/doorbell_bench"

export NVSHMEM_DISABLE_P2P=1
export NVSHMEM_DISABLE_NCCL=1
export NVSHMEM_REMOTE_TRANSPORT=ibdevx
export NVSHMEM_IB_ENABLE_IBGDA=1
export NVSHMEM_IBGDA_NIC_HANDLER=gpu
export NVSHMEM_IB_DISABLE_DMABUF=1
export NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE=gpumem
export NVSHMEM_DISABLE_GDRCOPY=1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_config() {
    local config="$1"
    echo "[$(timestamp)] config=$config bench=$BENCH_MODE npes=$NPES runs=$RUNS"
    rm -rf EventRecorder

    local logfile="$OUTDIR/${config}.log"
    echo "# config=$config date=$(timestamp)" > "$logfile"

    for run in $(seq 1 "$RUNS"); do
        echo "  run $run/$RUNS ..."
        echo "--- run $run ---" >> "$logfile"

        # Remove previous CSV so each run gets a fresh one
        rm -f doorbell_ops.csv

        case "$config" in
            baseline)
                "$NVSHMEM_HOME/bin/nvshmrun" -np "$NPES" "$BENCH_EXE" "$BENCH_MODE" >> "$logfile" 2>&1
                ;;
            uar_only)
                "$NVSHMEM_HOME/bin/nvshmrun" -np "$NPES" "$SIGMA_EXE_UAR_ONLY" -f --load-none --load-mlx5dv --enable-doorbell-hijack \
                    "$BENCH_EXE" "$BENCH_MODE" >> "$logfile" 2>&1
                ;;
            full_trace)
                "$NVSHMEM_HOME/bin/nvshmrun" -np "$NPES" "$SIGMA_EXE_FULL" -f --load-none --load-mlx5dv --enable-doorbell-hijack \
                    "$BENCH_EXE" "$BENCH_MODE" >> "$logfile" 2>&1
                ;;
        esac

        # Save per-run CSV
        [ -f doorbell_ops.csv ] && \
            mv doorbell_ops.csv "$OUTDIR/${config}_r${run}.csv" || true
        echo "" >> "$logfile"
    done
    echo "[$(timestamp)] Done: $logfile"
}

echo "============================================"
echo "=  Doorbell Benchmark (globaltimer)"
echo "=  PEs: $NPES  Mode: $BENCH_MODE"
echo "=  Runs: $RUNS"
echo "============================================"

for config in baseline uar_only full_trace; do
    run_config "$config"
done

echo ""
echo "Results in $OUTDIR/"
ls -la "$OUTDIR/"
