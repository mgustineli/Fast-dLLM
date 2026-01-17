#!/bin/bash
set -e

# =============================================================================
# Baseline GSM8K Evaluation
# =============================================================================
# Usage:
#   bash sbatch/eval_gsm8k.sh                        # Full evaluation
#   bash sbatch/eval_gsm8k.sh --limit 10             # Test with 10 samples
#   bash sbatch/eval_gsm8k.sh --limit 10 --batch_size 4
#
# Results: results/baseline/<task>/<timestamp>_<config>/
# =============================================================================

# Parse arguments
LIMIT_ARG=""
NUM_SAMPLES="all"
BATCH_SIZE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT_ARG="--limit $2"
            NUM_SAMPLES=$2
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE=$2
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ "$NUM_SAMPLES" == "all" ]; then
    echo "[INFO] Running full evaluation"
else
    echo "[INFO] Limiting evaluation to $NUM_SAMPLES samples"
fi

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Model configuration
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
TASK="gsm8k"
EXPERIMENT="baseline"

# Build config string
CONFIG="threshold1_n${NUM_SAMPLES}"

# Timestamp and output path
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/${EXPERIMENT}/${TASK}/${TIMESTAMP}_${CONFIG}"

# Export for eval.py logging
export TASK_NAME=$TASK
export RUN_TAG=$CONFIG
export RUN_TIMESTAMP=$TIMESTAMP

# Create results directory
mkdir -p "$OUTPUT_DIR"

echo "============================================================================="
echo "Experiment: ${EXPERIMENT}"
echo "Task: ${TASK}"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================================="

# Run evaluation
accelerate launch eval.py \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot 0 \
    ${LIMIT_ARG} \
    --model fast_dllm_v2 \
    --model_args model_path=${MODEL_PATH},threshold=1,show_speed=True \
    --output_path ${OUTPUT_DIR}/

echo "[INFO] Evaluation complete. Results: ${OUTPUT_DIR}"
