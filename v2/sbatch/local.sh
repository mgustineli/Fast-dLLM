#!/bin/bash
set -e

# =============================================================================
# Multi-Task Baseline Evaluation (Local)
# =============================================================================
# Runs baseline evaluation on multiple benchmarks: MMLU, GPQA, GSM8K,
# Minerva Math, and IFEval.
#
# Usage:
#   bash sbatch/eval_script.sh                          # Run all tasks
#   bash sbatch/eval_script.sh --limit 10               # Test with 10 samples per task
#   bash sbatch/eval_script.sh --task gsm8k             # Single task
#   bash sbatch/eval_script.sh --task gsm8k --limit 10  # Single task, 10 samples
#
# Results: results/baseline/<task>/<timestamp>_<config>/
# =============================================================================

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
            echo "Usage: bash sbatch/eval_script.sh [--task <task>] [--limit <n>]"
            echo "Available tasks: mmlu, gpqa_main_n_shot, gsm8k, minerva_math, ifeval"
            exit 1
            ;;
    esac
done

if [ -n "$SELECTED_TASK" ]; then
    echo "[INFO] Running single task: $SELECTED_TASK"
else
    echo "[INFO] Running all tasks"
fi

if [ "$NUM_SAMPLES" == "all" ]; then
    echo "[INFO] Full evaluation"
else
    echo "[INFO] Limiting to $NUM_SAMPLES samples per task"
fi

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Model configuration
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
EXPERIMENT="baseline"

# Timestamp (shared across all tasks)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export RUN_TIMESTAMP=$TIMESTAMP

# Task configurations: task_name|batch_size|num_fewshot|extra_args
# All tasks use --fewshot_as_multiturn to match original repo
# Note: batch_size=1 for generation tasks to avoid KV cache dimension issues
TASKS=(
    "mmlu|1|5|--fewshot_as_multiturn"
    "gpqa_main_n_shot|1|0|--fewshot_as_multiturn"
    "gsm8k|1|0|--fewshot_as_multiturn"
    "minerva_math|1|0|--fewshot_as_multiturn"
    "ifeval|1|0|--fewshot_as_multiturn"
)

# Validate selected task if provided
if [ -n "$SELECTED_TASK" ]; then
    VALID_TASK=false
    for TASK_CONFIG in "${TASKS[@]}"; do
        IFS='|' read -r TASK _ _ _ <<< "$TASK_CONFIG"
        if [ "$TASK" == "$SELECTED_TASK" ]; then
            VALID_TASK=true
            break
        fi
    done
    if [ "$VALID_TASK" == "false" ]; then
        echo "[ERROR] Unknown task: $SELECTED_TASK"
        echo "Available tasks: mmlu, gpqa_main_n_shot, gsm8k, minerva_math, ifeval"
        exit 1
    fi
fi

for TASK_CONFIG in "${TASKS[@]}"; do
    # Parse task configuration
    IFS='|' read -r TASK BATCH_SIZE NUM_FEWSHOT EXTRA_ARGS <<< "$TASK_CONFIG"

    # Skip if a specific task was selected and this isn't it
    if [ -n "$SELECTED_TASK" ] && [ "$TASK" != "$SELECTED_TASK" ]; then
        continue
    fi

    # Build config string
    CONFIG="threshold1_n${NUM_SAMPLES}"
    OUTPUT_DIR="results/${EXPERIMENT}/${TASK}/${TIMESTAMP}_${CONFIG}"

    # Export for eval.py logging and summary.json
    export TASK_NAME=$TASK
    export RUN_TAG=$CONFIG
    export OUTPUT_DIR=$OUTPUT_DIR

    # Create results directory
    mkdir -p "$OUTPUT_DIR"

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
        --confirm_run_unsafe_code \
        --model fast_dllm_v2 \
        --apply_chat_template \
        ${EXTRA_ARGS} \
        --model_args model_path=${MODEL_PATH},threshold=1,show_speed=True \
        ${LIMIT_ARG} \
        --output_path ${OUTPUT_DIR}/

    echo "[INFO] Complete: ${TASK}"
done

echo ""
echo "[INFO] All tasks completed. Results: results/${EXPERIMENT}/"
