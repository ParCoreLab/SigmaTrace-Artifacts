#!/bin/bash
#SBATCH --account=...
#SBATCH --partition=...
#SBATCH --qos=...
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --job-name=sigma-scale
#SBATCH --output=sigma-scale_%j.out
#SBATCH --error=sigma-scale_%j.err
#SBATCH --time=02:00:00

# ================================================================
# Load CUDA, NCCL, and other dependencies
#ml load nvidia-hpc-sdk
#ml load ucx
#ml load openmpi
#ml load nccl
# ================================================================

# ================================================================
# Executable paths
# Set SIGMA_EXE to the built SigmaTrace Profiler binary before sbatch.
#   export SIGMA_EXE=/path/to/SigmaTrace/build/target/Profiler
: "${SIGMA_EXE:?Set SIGMA_EXE to the SigmaTrace Profiler binary}"
NCCL_EXE="${NCCL_EXE:-$PWD/nccl-tests/build/all_reduce_perf_mpi}"
# ================================================================

# ================================================================
# Output paths
SCRATCH_DIR="/path/to/scratch/NCCL_EXPERIMENTS/"
OUTDIR="$SCRATCH_DIR/scaling_results_${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"
# ================================================================

# ================================================================
# Parameters
NCCL_ITERS=2000
NCCL_WARMUP=500
MSG_SIZE=${MSG_SIZE:-1M}
N_REPS=5

# Optional NCCL environment for consistent behavior
# export NCCL_ALGO=Ring
# export NCCL_PROTO=Simple

# Adjust according to system
SRUN_COMMON="--exact --gpus-per-node=4"

# Only collect ibtrace
SIGMA_COMMON="-f --load-none --load-ibtrace=true"
# ================================================================

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

#
nvidia-smi

# ================================================================
# Experiment A: overhead vs. node count with fixed size
echo "================================================="
echo "=  Experiment A: Scaling at fixed message size  ="
echo "================================================="
echo "=  Nodes: 1 → $SLURM_NNODES (doubling)          ="
echo "=  Message: $MSG_SIZE, $NCCL_ITERS iters        ="
echo "================================================="

# Get sorted node list from allocation for consistent sub-selections
IFS=',' read -ra ALL_NODES <<< "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | paste -sd,)"

nodes=2
while [ $nodes -le $SLURM_NNODES ]; do
    np=$((nodes * 4))

    # Pin to first N nodes so all configs at this node count hit the same hardware
    NODELIST=$(echo "${ALL_NODES[@]:0:$nodes}" | tr ' ' ',')
    echo ""
    echo "[$(timestamp)] --- $np GPUs ($nodes nodes: $NODELIST) ---"

    for rep in $(seq 1 $N_REPS); do
        echo "[$(timestamp)]   --- rep $rep/$N_REPS --- "

        # Baseline (no SigmaTrace)
        outfile="$OUTDIR/A_baseline_${nodes}n_${np}g_r${rep}.txt"
        echo "[$(timestamp)]   baseline → $outfile"
        srun --nodes=$nodes --ntasks=$np --ntasks-per-node=4 --nodelist="$NODELIST" $SRUN_COMMON \
            "$NCCL_EXE" -b "$MSG_SIZE" -e "$MSG_SIZE" \
            -n $NCCL_ITERS -w $NCCL_WARMUP -z 0 -c 0 \
            > "$outfile" 2>&1

        # SigmaTrace modes
        for mode in aggregate simple full; do
            outfile="$OUTDIR/A_${mode}_${nodes}n_${np}g_r${rep}.txt"
            tracedir="$OUTDIR/traces/A_${mode}_${nodes}n_r${rep}"
            mkdir -p "$tracedir"
            echo "[$(timestamp)]   $mode → $outfile"
            srun --nodes=$nodes --ntasks=$np --ntasks-per-node=4 --nodelist="$NODELIST" $SRUN_COMMON \
                "$SIGMA_EXE" --trace-mode=$mode $SIGMA_COMMON \
                --output-dir="$tracedir" \
                "$NCCL_EXE" -b "$MSG_SIZE" -e "$MSG_SIZE" \
                -n $NCCL_ITERS -w $NCCL_WARMUP -z 0 -c 0 \
                > "$outfile" 2>&1
        done
    done

    nodes=$((nodes * 2))
done


## ================================================================
## Experiment B: overhead vs. message size with fixed node count
#MAX_NODES=$SLURM_NNODES
#MAX_NP=$((MAX_NODES * 4))
#
#echo ""
#echo "================================================="
#echo "=  Experiment B: Message size sweep             ="
#echo "================================================="
#echo "=  Nodes: $MAX_NODES ($MAX_NP GPUs)             ="
#echo "=  Sizes: 1K → 128M                             ="
#echo "================================================="
#
#for rep in $(seq 1 $N_REPS); do
#    echo "[$(timestamp)]   --- rep $rep/$N_REPS ---"
#
#    # Baseline
#    outfile="$OUTDIR/B_baseline_${MAX_NODES}n_sweep_r${rep}.txt"
#    echo "[$(timestamp)] baseline sweep -> $outfile"
#    srun --nodes=$MAX_NODES --ntasks=$MAX_NP --ntasks-per-node=4 $SRUN_COMMON \
#        "$NCCL_EXE" -b 1K -e 128M -f 2 -z 0 -c 0 \
#        -n 200 -w 50 \
#        > "$outfile" 2>&1
#
#    # SigmaTrace modes
#    for mode in aggregate simple full; do
#        outfile="$OUTDIR/B_${mode}_${MAX_NODES}n_sweep_r${rep}.txt"
#        tracedir="$OUTDIR/traces/B_${mode}_${MAX_NODES}n_r${rep}"
#        mkdir -p "$tracedir"
#        echo "[$(timestamp)] $mode sweep -> $outfile"
#        srun --nodes=$MAX_NODES --ntasks=$MAX_NP --ntasks-per-node=4 $SRUN_COMMON \
#            "$SIGMA_EXE" --trace-mode=$mode $SIGMA_COMMON \
#            --output-dir="$tracedir" \
#            "$NCCL_EXE" -b 1K -e 128M -f 2 -z 0 -c 0 \
#            -n 200 -w 50 \
#            > "$outfile" 2>&1
#    done
#done

# ================================================================
# Summary
echo ""
echo "[$(timestamp)] All done. Results in $OUTDIR/"
ls -lh "$OUTDIR"/A_*.txt "$OUTDIR"/B_*.txt 2>/dev/null | head -30
echo ""
echo "Trace data sizes:"
du -sh "$OUTDIR/traces/"* 2>/dev/null