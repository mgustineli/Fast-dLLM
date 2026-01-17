#!/bin/bash
set -e

# =============================================================================
# Activation Reuse Experiments
# =============================================================================
# Reuses activations from previous k steps (without layer subsetting).
#
# Usage:
#   bash sbatch/eval_gsm8k_reuse_activations.sh              # Full evaluation
#   bash sbatch/eval_gsm8k_reuse_activations.sh --limit 10   # Test with 10 samples
#
# Results: results/reuse_activations/<task>/<timestamp>_<config>/
# =============================================================================

# Parse arguments
LIMIT_ARG=""
NUM_SAMPLES="all"

while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT_ARG="--limit $2"
            NUM_SAMPLES=$2
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

# Deterministic results
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=42

# Model configuration
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"
TASK="gsm8k"
EXPERIMENT="reuse_activations"

# Timestamp (shared across all runs in this sweep)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export RUN_TIMESTAMP=$TIMESTAMP

# Sweep over k values
for K in 1 2 3; do
    # Build config string
    CONFIG="k${K}_n${NUM_SAMPLES}"
    OUTPUT_DIR="results/${EXPERIMENT}/${TASK}/${TIMESTAMP}_${CONFIG}"

    # Export for eval.py logging
    export TASK_NAME=$TASK
    export RUN_TAG=$CONFIG

    # Create results directory
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "============================================================================="
    echo "Experiment: ${EXPERIMENT}"
    echo "Task: ${TASK}"
    echo "Config: k=${K}, samples=${NUM_SAMPLES}"
    echo "Output: ${OUTPUT_DIR}"
    echo "============================================================================="

    accelerate launch eval.py \
        --tasks ${TASK} \
        --model fast_dllm_v2 \
        --batch_size 1 \
        --num_fewshot 0 \
        --model_args "model_path=${MODEL_PATH},reuse_k=${K},use_block_cache=True,show_speed=True" \
        ${LIMIT_ARG} \
        --output_path ${OUTPUT_DIR}/

    echo "[INFO] Complete: ${CONFIG}"
done

echo ""
echo "[INFO] All experiments completed. Results: results/${EXPERIMENT}/${TASK}/"
