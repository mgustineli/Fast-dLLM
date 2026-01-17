#!/bin/bash
#SBATCH --job-name=reuse-layers
#SBATCH --account=pacepsdsgt-clef2026
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:1
#SBATCH --time=04:00:00
#SBATCH -qinferno
#SBATCH --output=logs/reuse_layers_%A_%a.out
#SBATCH --error=logs/reuse_layers_%A_%a.err
#SBATCH --array=0-8

# =============================================================================
# Layer Reuse Experiments - SLURM Array Job
# =============================================================================
# Runs 9 experiments in parallel (3 subsets x 3 k values):
#   Task 0: first,  k=1     Task 3: middle, k=1     Task 6: last, k=1
#   Task 1: first,  k=2     Task 4: middle, k=2     Task 7: last, k=2
#   Task 2: first,  k=3     Task 5: middle, k=3     Task 8: last, k=3
#
# Usage:
#   sbatch sbatch/eval_reuse_layers_array.sbatch              # Full evaluation
#   sbatch sbatch/eval_reuse_layers_array.sbatch --limit 10   # Test mode
#
# To run specific tasks only:
#   sbatch --array=0,3,6 sbatch/eval_reuse_layers_array.sbatch  # Only k=1
#
# Results: results/reuse_layers/<task>/<timestamp>_<config>/
# =============================================================================

set -e

# Parse optional --limit argument
LIMIT_ARG=""
NUM_SAMPLES="all"
if [ "$1" == "--limit" ] && [ -n "$2" ]; then
    LIMIT_ARG="--limit $2"
    NUM_SAMPLES=$2
    echo "[INFO] Limiting evaluation to $NUM_SAMPLES samples"
else
    echo "[INFO] Running full evaluation (no limit specified)"
fi

# Change to project directory
cd /storage/scratch1/9/mgustineli3/Fast-dLLM/v2

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME=/storage/scratch1/9/mgustineli3/.cache/huggingface
export TORCH_HOME=/storage/scratch1/9/mgustineli3/.cache/torch

# Deterministic results
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=42

# Model configuration
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
TASK="gsm8k"
EXPERIMENT="reuse_layers"

# Define experiment configurations (array index -> subset, k)
SUBSETS=("first" "first" "first" "middle" "middle" "middle" "last" "last" "last")
K_VALUES=(1 2 3 1 2 3 1 2 3)

# Get configuration for this array task
SUBSET=${SUBSETS[$SLURM_ARRAY_TASK_ID]}
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

# Build config string and output path
CONFIG="k${K}_${SUBSET}_n${NUM_SAMPLES}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/${EXPERIMENT}/${TASK}/${TIMESTAMP}_${CONFIG}"

# Export for eval.py logging
export TASK_NAME=$TASK
export RUN_TAG=$CONFIG
export RUN_TIMESTAMP=$TIMESTAMP

# Create results directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================================="
echo "SLURM Array Job: Task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Experiment: ${EXPERIMENT}"
echo "Task: ${TASK}"
echo "Config: subset=${SUBSET}, k=${K}, samples=${NUM_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "Job ID: $SLURM_JOB_ID, Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================================="

# Run evaluation
accelerate launch eval.py \
    --tasks ${TASK} \
    --model fast_dllm_v2 \
    --batch_size 1 \
    --num_fewshot 0 \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --model_args "model_path=${MODEL_PATH},reuse_k=${K},layer_subset=${SUBSET},use_block_cache=True,show_speed=True" \
    ${LIMIT_ARG} \
    --output_path ${OUTPUT_DIR}/

echo ""
echo "[INFO] Task $SLURM_ARRAY_TASK_ID complete: ${CONFIG}"
echo "[INFO] Results: ${OUTPUT_DIR}"
