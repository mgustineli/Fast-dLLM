#!/bin/bash
#SBATCH --account=gts-ylin715-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=12:00:00
#SBATCH -qinferno

# =============================================================================
# Single Reuse Layer Experiment
# =============================================================================
# Called by run.sh with environment variables:
#   CONFIG_NAME: e.g., "k2_middle"
#   REUSE_K: e.g., "2"
#   LAYER_SUBSET: e.g., "middle"
#   LIMIT_ARG: e.g., "--limit 10" (optional)
#   TASK_ARG: e.g., "--task mmlu" (optional, default: gsm8k)
# =============================================================================

set -e

# =============================================================================
# Determine project root dynamically
# =============================================================================
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

cd "$PROJECT_ROOT"

# Defaults
TASK="${TASK:-gsm8k}"
CONFIG_NAME="${CONFIG_NAME:-unknown}"
REUSE_K="${REUSE_K:-1}"
LAYER_SUBSET="${LAYER_SUBSET:-first}"

# Parse TASK_ARG if provided
if [[ "$TASK_ARG" == "--task "* ]]; then
    TASK="${TASK_ARG#--task }"
fi

EXPERIMENT="reuse_layers"
OUTPUT_DIR="results/${EXPERIMENT}/${TASK}/${CONFIG_NAME}"

# Create output and log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "logs/${EXPERIMENT}/${TASK}/${CONFIG_NAME}"

echo "============================================================================="
echo "Layer Reuse Experiment"
echo "Config: $CONFIG_NAME (k=$REUSE_K, subset=$LAYER_SUBSET)"
echo "Task: $TASK"
echo "Output: $OUTPUT_DIR"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Node: $SLURMD_NODENAME"
fi
echo "============================================================================="

# =============================================================================
# Virtual Environment Setup (persistent venv in scratch)
# =============================================================================
SCRATCH_PROJECT="$HOME/scratch/Fast-dLLM/v2"
SCRATCH_VENV="$SCRATCH_PROJECT/.venv"
LOCK_FILE="$SCRATCH_PROJECT/.venv_setup.lock"

setup_scratch_venv() {
    echo "[INFO] Setting up venv at $SCRATCH_VENV (this may take a few minutes)..."
    mkdir -p "$SCRATCH_PROJECT"
    cd "$SCRATCH_PROJECT"
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e .
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    cd "$PROJECT_ROOT"
    echo "[INFO] Venv setup complete"
}

if [ -d "$SCRATCH_VENV" ] && [ -f "$SCRATCH_VENV/bin/activate" ]; then
    echo "[INFO] Activating venv from scratch: $SCRATCH_VENV"
    source "$SCRATCH_VENV/bin/activate"
else
    exec 200>"$LOCK_FILE"
    if flock -n 200; then
        if [ ! -d "$SCRATCH_VENV" ]; then
            setup_scratch_venv
        fi
        source "$SCRATCH_VENV/bin/activate"
        flock -u 200
    else
        echo "[INFO] Another job is setting up venv, waiting..."
        flock 200
        source "$SCRATCH_VENV/bin/activate"
        flock -u 200
    fi
fi

echo "[INFO] Python: $(which python)"

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=42

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
    --model_args "model_path=${MODEL_PATH},reuse_k=${REUSE_K},layer_subset=${LAYER_SUBSET},use_block_cache=True,show_speed=True" \
    ${LIMIT_ARG} \
    --output_path ${OUTPUT_DIR}/

echo ""
echo "[INFO] Experiment complete: $CONFIG_NAME"
echo "[INFO] Results: $OUTPUT_DIR"
