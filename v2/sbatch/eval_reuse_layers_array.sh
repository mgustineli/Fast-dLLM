#!/bin/bash
#SBATCH --job-name=reuse-layers
#SBATCH --account=gts-ylin715-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=12:00:00
#SBATCH -qinferno
#SBATCH --output=logs/slurm_%A_%a.log
#SBATCH --error=logs/slurm_%A_%a.log
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
#   sbatch sbatch/eval_reuse_layers_array.sh                        # GSM8K full
#   sbatch sbatch/eval_reuse_layers_array.sh --limit 10             # Test mode
#   sbatch sbatch/eval_reuse_layers_array.sh --task mmlu            # Different task
#   sbatch sbatch/eval_reuse_layers_array.sh --task mmlu --limit 10
#
# To run specific configurations only:
#   sbatch --array=0,3,6 sbatch/eval_reuse_layers_array.sh          # Only k=1
#
# Results: results/reuse_layers/<task>/<timestamp>_<config>/
# Logs:    logs/reuse_layers/<task>/<timestamp>_<config>/slurm.log
# =============================================================================

set -e

# =============================================================================
# Determine project root dynamically
# =============================================================================
# Use SLURM_SUBMIT_DIR if available (directory where sbatch was run),
# otherwise use the script's directory to find project root
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"  # Go up from sbatch/ to v2/
fi

cd "$PROJECT_ROOT"
echo "[INFO] Project root: $PROJECT_ROOT"

# Parse arguments
LIMIT_ARG=""
NUM_SAMPLES="all"
TASK="gsm8k"

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT_ARG="--limit $2"
            NUM_SAMPLES=$2
            shift 2
            ;;
        --task)
            TASK=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch eval_reuse_layers_array.sbatch [--task <task>] [--limit <n>]"
            exit 1
            ;;
    esac
done

echo "[INFO] Task: ${TASK}"
if [ "$NUM_SAMPLES" == "all" ]; then
    echo "[INFO] Running full evaluation"
else
    echo "[INFO] Limiting evaluation to $NUM_SAMPLES samples"
fi

# Ensure base logs directory exists
mkdir -p "logs"

# Activate existing virtual environment from scratch (persistent, large storage)
# Setup once with: cd ~/scratch/Fast-dLLM/v2 && uv venv .venv && uv pip install -e . && uv pip install torch
SCRATCH_VENV="$HOME/scratch/Fast-dLLM/v2/.venv"
if [ -d "$SCRATCH_VENV" ]; then
    echo "[INFO] Activating venv from scratch: $SCRATCH_VENV"
    source "$SCRATCH_VENV/bin/activate"
elif [ -d "$PROJECT_ROOT/.venv" ]; then
    echo "[INFO] Activating venv at $PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "[ERROR] No virtual environment found. Create one with:"
    echo "  cd ~/scratch/Fast-dLLM/v2 && uv venv .venv && uv pip install -e . && uv pip install torch"
    exit 1
fi

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
# Note: Cache directories (HF_HOME, TORCH_HOME) inherit from user environment.
# Set XDG_CACHE_HOME in ~/.bashrc to redirect caches (e.g., to ~/scratch/.cache)

# Deterministic results
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=42

# Model configuration
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
EXPERIMENT="reuse_layers"

# Define experiment configurations (array index -> subset, k)
SUBSETS=("first" "first" "first" "middle" "middle" "middle" "last" "last" "last")
K_VALUES=(1 2 3 1 2 3 1 2 3)

# Get configuration for this array task
SUBSET=${SUBSETS[$SLURM_ARRAY_TASK_ID]}
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

# Build config string and output paths (same structure for results and logs)
CONFIG="k${K}_${SUBSET}_n${NUM_SAMPLES}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${TIMESTAMP}_${CONFIG}"
OUTPUT_DIR="results/${EXPERIMENT}/${TASK}/${RUN_DIR}"
LOG_DIR="logs/${EXPERIMENT}/${TASK}/${RUN_DIR}"

# Export for eval.py
export TASK_NAME=$TASK
export RUN_TAG=$CONFIG
export OUTPUT_DIR=$OUTPUT_DIR

# Create results and logs directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

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
    --fewshot_as_multiturn \
    --apply_chat_template \
    --model_args "model_path=${MODEL_PATH},reuse_k=${K},layer_subset=${SUBSET},use_block_cache=True,show_speed=True" \
    ${LIMIT_ARG} \
    --output_path ${OUTPUT_DIR}/

# Move SLURM temp log to organized location and copy to results
SLURM_LOG="logs/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
mv "${SLURM_LOG}" "${LOG_DIR}/slurm.log" 2>/dev/null || true
cp "${LOG_DIR}/slurm.log" "${OUTPUT_DIR}/slurm.log" 2>/dev/null || true

echo ""
echo "[INFO] Task $SLURM_ARRAY_TASK_ID complete: ${CONFIG}"
echo "[INFO] Results: ${OUTPUT_DIR}"
echo "[INFO] Log: ${LOG_DIR}/slurm.log"
