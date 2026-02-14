#!/bin/bash
#SBATCH --account=gts-ylin715-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=12:00:00
#SBATCH -qinferno

# =============================================================================
# [EXPERIMENT_NAME] - Single SLURM Job
# =============================================================================
# Called by run.sh with environment variables:
#   CONFIG_NAME:     e.g., "k2_middle"
#   REUSE_K:         e.g., "2"
#   LAYER_SUBSET:    e.g., "middle"
#   LIMIT_ARG:       e.g., "--limit 10" (optional)
#   TASK_ARG:        e.g., "--task mmlu" (optional, default: gsm8k)
#   EXPERIMENT_NAME: e.g., "00_baseline"
#   EXPERIMENT_DIR:  absolute path to experiment directory
#   TASK_PATH:       e.g., "gsm8k" or "gsm8k_limit_10"
# =============================================================================

set -e

# Determine project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
    PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
fi

cd "$PROJECT_ROOT"

# Source shared utilities
source "$PROJECT_ROOT/sbatch/common.sh"

# Defaults
TASK="${TASK:-gsm8k}"
CONFIG_NAME="${CONFIG_NAME:-unknown}"
REUSE_K="${REUSE_K:-1}"
LAYER_SUBSET="${LAYER_SUBSET:-first}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(basename "$EXPERIMENT_DIR")}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-$PROJECT_ROOT/experiments/$EXPERIMENT_NAME}"

# Parse TASK_ARG if provided
if [[ "$TASK_ARG" == "--task "* ]]; then
    TASK="${TASK_ARG#--task }"
fi

TASK_PATH="${TASK_PATH:-$TASK}"

# Experiment-local output paths
OUTPUT_DIR="${EXPERIMENT_DIR}/results/${TASK_PATH}/${CONFIG_NAME}"
ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts/${TASK_PATH}/${CONFIG_NAME}"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "${EXPERIMENT_DIR}/logs/${TASK_PATH}/${CONFIG_NAME}"

# Print header
print_job_info "Layer Reuse Experiment" "$CONFIG_NAME (k=$REUSE_K, subset=$LAYER_SUBSET)" "$TASK" "$OUTPUT_DIR"

# Setup venv and environment
setup_venv
setup_environment

# Export for eval.py
export TASK_NAME=$TASK
export RUN_TAG=$CONFIG_NAME
export OUTPUT_DIR=$OUTPUT_DIR

# Model configuration
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"

# Run evaluation
accelerate launch eval.py \
    --tasks ${TASK} \
    --model fast_dllm_v2 \
    --batch_size 1 \
    --num_fewshot 0 \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --confirm_run_unsafe_code \
    --model_args "model_path=${MODEL_PATH},reuse_k=${REUSE_K},layer_subset=${LAYER_SUBSET},use_block_cache=True,show_speed=True,experiment_name=${EXPERIMENT_NAME}" \
    ${LIMIT_ARG} \
    --output_path ${OUTPUT_DIR}/

# Copy summary.json to artifacts directory for git tracking
mkdir -p "$ARTIFACTS_DIR"
if [ -f "${OUTPUT_DIR}/summary.json" ]; then
    cp "${OUTPUT_DIR}/summary.json" "$ARTIFACTS_DIR/summary.json"
    echo "[INFO] Copied summary.json to artifacts."
else
    echo "[WARN] summary.json not found in ${OUTPUT_DIR}, skipping artifact copy."
fi

echo ""
echo "[INFO] Experiment complete: $CONFIG_NAME"
echo "[INFO] Results: $OUTPUT_DIR"
