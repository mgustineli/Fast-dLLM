#!/bin/bash
#SBATCH --job-name=fast-dllm-eval
#SBATCH --account=gts-ylin715-paid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:V100:1
#SBATCH --time=08:00:00
#SBATCH -qinferno
#SBATCH --output=logs/slurm_%j.log
#SBATCH --error=logs/slurm_%j.log

# =============================================================================
# Fast-dLLM Baseline Evaluation - PACE Cluster
# =============================================================================
# Runs baseline evaluation on benchmarks: MMLU, GPQA, GSM8K, Minerva Math, IFEval
#
# Usage:
#   sbatch sbatch/eval_pace_script.sbatch                           # All tasks, full eval
#   sbatch sbatch/eval_pace_script.sbatch --limit 10                # All tasks, 10 samples
#   sbatch sbatch/eval_pace_script.sbatch --task gsm8k              # Single task, full eval
#   sbatch sbatch/eval_pace_script.sbatch --task gsm8k --limit 10   # Single task, 10 samples
#
# Available tasks: mmlu, gpqa_main_n_shot, gsm8k, minerva_math, ifeval
#
# Results: results/baseline/<task>/<timestamp>_<config>/
# Logs:    logs/baseline/<task>/<timestamp>_<config>/slurm.log
# =============================================================================

set -e

# Parse arguments
LIMIT_ARG=""
NUM_SAMPLES="all"
SELECTED_TASK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT_ARG="--limit $2"
            NUM_SAMPLES=$2
            shift 2
            ;;
        --task)
            SELECTED_TASK=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: sbatch eval_pace_script.sbatch [--task <task>] [--limit <n>]"
            echo "Available tasks: mmlu, gpqa_main_n_shot, gsm8k, minerva_math, ifeval"
            exit 1
            ;;
    esac
done

# Change to project directory
cd /storage/scratch1/9/mgustineli3/Fast-dLLM/v2

# Ensure base logs directory exists
mkdir -p "logs"

# Activate virtual environment
source ../.venv/bin/activate

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
EXPERIMENT="baseline"

# Timestamp (shared across all tasks)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================================================="
echo "Fast-dLLM Baseline Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"
if [ -n "$SELECTED_TASK" ]; then
    echo "Task: ${SELECTED_TASK}"
else
    echo "Tasks: mmlu, gpqa_main_n_shot, gsm8k, minerva_math, ifeval"
fi
if [ "$NUM_SAMPLES" == "all" ]; then
    echo "Samples: full evaluation"
else
    echo "Samples: limited to $NUM_SAMPLES per task"
fi
echo "============================================================================="

# Task configurations: task_name|batch_size|num_fewshot|extra_args
# Note: batch_size=1 for generation tasks to avoid KV cache dimension issues
declare -A TASK_CONFIGS=(
    ["mmlu"]="1|5|--fewshot_as_multiturn"
    ["gpqa_main_n_shot"]="1|0|--fewshot_as_multiturn"
    ["gsm8k"]="1|0|"
    ["minerva_math"]="1|0|"
    ["ifeval"]="1|0|"
)

# Build list of tasks to run
if [ -n "$SELECTED_TASK" ]; then
    if [ -z "${TASK_CONFIGS[$SELECTED_TASK]}" ]; then
        echo "[ERROR] Unknown task: $SELECTED_TASK"
        echo "Available tasks: ${!TASK_CONFIGS[@]}"
        exit 1
    fi
    TASKS_TO_RUN=("$SELECTED_TASK")
else
    TASKS_TO_RUN=("mmlu" "gpqa_main_n_shot" "gsm8k" "minerva_math" "ifeval")
fi

# Run evaluations
for TASK in "${TASKS_TO_RUN[@]}"; do
    # Parse task configuration
    IFS='|' read -r BATCH_SIZE NUM_FEWSHOT EXTRA_ARGS <<< "${TASK_CONFIGS[$TASK]}"

    # Build config string and output paths (same structure for results and logs)
    CONFIG="threshold1_n${NUM_SAMPLES}"
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

    echo ""
    echo "============================================================================="
    echo "Experiment: ${EXPERIMENT}"
    echo "Task: ${TASK}"
    echo "Config: batch_size=${BATCH_SIZE}, fewshot=${NUM_FEWSHOT}, samples=${NUM_SAMPLES}"
    echo "Output: ${OUTPUT_DIR}"
    echo "============================================================================="

    accelerate launch eval.py \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --model fast_dllm_v2 \
        ${EXTRA_ARGS} \
        --model_args "model_path=${MODEL_PATH},threshold=1,show_speed=True" \
        ${LIMIT_ARG} \
        --output_path ${OUTPUT_DIR}/

    # Copy current SLURM log to task-specific log and results directories
    SLURM_LOG="logs/slurm_${SLURM_JOB_ID}.log"
    cp "${SLURM_LOG}" "${LOG_DIR}/slurm.log" 2>/dev/null || true
    cp "${SLURM_LOG}" "${OUTPUT_DIR}/slurm.log" 2>/dev/null || true

    echo "[INFO] Complete: ${TASK}"
    echo "[INFO] Results: ${OUTPUT_DIR}"
    echo "[INFO] Log: ${LOG_DIR}/slurm.log"
done

# Clean up temp SLURM log after all tasks complete
rm -f "logs/slurm_${SLURM_JOB_ID}.log" 2>/dev/null || true

echo ""
echo "============================================================================="
echo "[INFO] All tasks completed"
echo "[INFO] Results: results/${EXPERIMENT}/"
echo "[INFO] Logs: logs/${EXPERIMENT}/"
echo "============================================================================="
