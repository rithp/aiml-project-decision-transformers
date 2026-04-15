#!/bin/bash
# =============================================================================
# Cross-Task Generalization Experiment for Decision Transformer
# =============================================================================
#
# This script runs the full experiment pipeline:
#   Step 0: Setup environment
#   Step 1: Download D4RL datasets
#   Step 2: Collect target domain data
#   Step 3: Pre-train DT on source datasets
#   Step 4: Fine-tune vs from-scratch comparison
#   Step 5: Plot results
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --step 3     # Run from step 3 onwards
#   ./run_all.sh --dry-run    # Print commands without executing
#
# Estimated time on a single GPU:
#   A100:     ~1-1.5 days
#   3090/V100: ~2-3 days
#   You can reduce by running fewer source datasets (see below)
# =============================================================================

set -e  # Exit on error

# Parse arguments
START_STEP=0
DRY_RUN=false
DEVICE="cpu"
SOURCE_DATASETS="medium,medium-replay,expert"
TARGETS="all"
NUM_TARGET_TRAJECTORIES=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --step) START_STEP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        --source) SOURCE_DATASETS="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --num-target-traj) NUM_TARGET_TRAJECTORIES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

run_cmd() {
    echo ""
    echo ">>> $1"
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute: $2"
    else
        eval "$2"
    fi
}

cd "$(dirname "$0")"
echo "Working directory: $(pwd)"
echo "Device: $DEVICE"
echo "Source datasets: $SOURCE_DATASETS"
echo "Target domains: $TARGETS"
echo "Starting from step: $START_STEP"
echo ""

# ---- Step 0: Setup ----
if [ "$START_STEP" -le 0 ]; then
    echo "============================================================"
    echo "STEP 0: Environment Setup"
    echo "============================================================"
    echo "Skipping dependency installation (dt environment is already set up properly)"
    echo ""
    echo "NOTE: MuJoCo setup can be tricky. If mujoco-py fails, try:"
    echo "  pip install mujoco  # newer MuJoCo bindings"
    echo "  pip install gymnasium  # and adapt imports if needed"
    echo ""
fi

# ---- Step 1: Download D4RL datasets ----
if [ "$START_STEP" -le 1 ]; then
    echo "============================================================"
    echo "STEP 1: Download D4RL Datasets"
    echo "============================================================"
    
    # Check if datasets already exist
    NEED_DOWNLOAD=false
    for ds in medium medium-replay medium-expert; do
        if [ ! -f "data/walker2d-${ds}-v2.pkl" ]; then
            NEED_DOWNLOAD=true
            break
        fi
    done
    
    if [ "$NEED_DOWNLOAD" = true ]; then
        run_cmd "Download D4RL datasets" \
            "cd data && python download_d4rl_datasets.py --env walker2d && cd .."
    else
        echo "Datasets already downloaded, skipping."
    fi
fi

# ---- Step 2: Collect target domain data ----
if [ "$START_STEP" -le 2 ]; then
    echo "============================================================"
    echo "STEP 2: Collect Target Domain Data"
    echo "============================================================"
    echo "Collecting ${NUM_TARGET_TRAJECTORIES} trajectories per target domain"
    echo "Using source policy replay from walker2d-medium-v2 dataset"
    
    run_cmd "Collect target domain data" \
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=. python cross_task/collect_target_data.py \
            --target all \
            --num_trajectories ${NUM_TARGET_TRAJECTORIES} \
            --source_dataset data/walker2d-medium-v2.pkl \
            --output_dir data/target"
fi

# ---- Step 3: Pre-train on source datasets ----
if [ "$START_STEP" -le 3 ]; then
    echo "============================================================"
    echo "STEP 3: Pre-train DT on Source Datasets"
    echo "============================================================"
    
    run_cmd "Pre-train Decision Transformers" \
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=. python cross_task/run_cross_task.py \
            --phase pretrain \
            --source_datasets ${SOURCE_DATASETS} \
            --device ${DEVICE}"
fi

# ---- Step 4: Transfer experiments ----
if [ "$START_STEP" -le 4 ]; then
    echo "============================================================"
    echo "STEP 4: Fine-tune vs From-Scratch Transfer Experiments"
    echo "============================================================"
    
    run_cmd "Run transfer experiments" \
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=. python cross_task/run_cross_task.py \
            --phase transfer \
            --source_datasets ${SOURCE_DATASETS} \
            --target ${TARGETS} \
            --device ${DEVICE}"
fi

# ---- Step 5: Plot results ----
if [ "$START_STEP" -le 5 ]; then
    echo "============================================================"
    echo "STEP 5: Plot Results"
    echo "============================================================"
    
    run_cmd "Generate plots and summary" \
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTHONPATH=. python cross_task/plot_results.py \
            --results_file results/transfer_results.json \
            --output_dir results/plots"
fi

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results:     results/transfer_results.json"
echo "Plots:       results/plots/"
echo "Checkpoints: checkpoints/"
echo "============================================================"
