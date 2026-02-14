#!/bin/bash
# =============================================================================
# 00_baseline - Local Run (Interactive GPU Session)
# =============================================================================
# Runs experiment configs locally using the project venv. Supports running a
# single config or all 9 configs sequentially. Skips already-completed configs
# unless --force is used.
#
# Usage:
#   bash experiments/00_baseline/sbatch/run_locally.sh --limit 10              # all 9, skip completed
#   bash experiments/00_baseline/sbatch/run_locally.sh --config k1_first       # single config
#   bash experiments/00_baseline/sbatch/run_locally.sh --force --limit 10      # re-run all
#   bash experiments/00_baseline/sbatch/run_locally.sh --status                # check completion
#
# Requires: active interactive GPU session (salloc/pace-interact)
# =============================================================================

set -e

# Derive paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"

cd "$PROJECT_ROOT"

# Venv location
VENV_PATH="${REPO_ROOT}/.venv"
if [ ! -f "$VENV_PATH/bin/python" ]; then
    echo "[ERROR] Venv not found at $VENV_PATH"
    echo "Expected: $VENV_PATH/bin/python"
    exit 1
fi

# All experiment configs: k_value and layer_subset
declare -A CONFIGS=(
    ["k1_first"]="1|first"
    ["k1_middle"]="1|middle"
    ["k1_last"]="1|last"
    ["k2_first"]="2|first"
    ["k2_middle"]="2|middle"
    ["k2_last"]="2|last"
    ["k3_first"]="3|first"
    ["k3_middle"]="3|middle"
    ["k3_last"]="3|last"
)

# Parse arguments
CONFIG_FILTER=""
TASK="gsm8k"
LIMIT_VALUE=""
LIMIT_ARG=""
FORCE=false
SHOW_STATUS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILTER=$2
            if [[ -z "${CONFIGS[$CONFIG_FILTER]}" ]]; then
                echo "[ERROR] Unknown config: $CONFIG_FILTER"
                echo "Valid configs: ${!CONFIGS[*]}"
                exit 1
            fi
            shift 2
            ;;
        --task)
            TASK=$2
            shift 2
            ;;
        --limit)
            LIMIT_VALUE=$2
            LIMIT_ARG="--limit $2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        -h|--help)
            cat << EOF
Usage: bash experiments/${EXPERIMENT_NAME}/sbatch/run_locally.sh [OPTIONS]

Run experiment configs locally (interactive GPU session).
Without --config, runs all 9 configs sequentially, skipping completed ones.

OPTIONS:
    --config <CONFIG>   Run only this config (e.g., k1_first)
    --task <TASK>       Benchmark task (default: gsm8k)
    --limit <N>         Limit to N samples (for testing)
    --force             Re-run even if already completed
    --status            Show completion status and exit
    -h, --help          Show this help message

CONFIGS:
    k1_first, k1_middle, k1_last, k2_first, k2_middle, k2_last,
    k3_first, k3_middle, k3_last

TASKS:
    gsm8k, mmlu, minerva_math, ifeval, gpqa_main_n_shot

EXAMPLES:
    # Run all 9 configs with limit 10 (skip completed)
    bash experiments/${EXPERIMENT_NAME}/sbatch/run_locally.sh --limit 10

    # Run all 9 configs (full, no limit)
    bash experiments/${EXPERIMENT_NAME}/sbatch/run_locally.sh

    # Run single config
    bash experiments/${EXPERIMENT_NAME}/sbatch/run_locally.sh --config k1_first --limit 10

    # Force re-run all
    bash experiments/${EXPERIMENT_NAME}/sbatch/run_locally.sh --force --limit 10

    # Check status
    bash experiments/${EXPERIMENT_NAME}/sbatch/run_locally.sh --status --task gsm8k --limit 10
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

# Build task path (includes limit suffix if applicable)
TASK_PATH="$TASK"
if [[ -n "$LIMIT_VALUE" ]]; then
    TASK_PATH="${TASK}_limit_${LIMIT_VALUE}"
fi

# Check if a config has completed (summary.json exists in artifacts)
is_completed() {
    local config=$1
    local summary_file="${EXPERIMENT_DIR}/artifacts/${TASK_PATH}/${config}/summary.json"
    [[ -f "$summary_file" ]]
}

# --status: show completion and exit
if [ "$SHOW_STATUS" = true ]; then
    echo "============================================================================="
    echo "Experiment Status: $EXPERIMENT_NAME / $TASK_PATH"
    echo "============================================================================="
    completed=0
    pending=0
    lines=()
    for config in "${!CONFIGS[@]}"; do
        if is_completed "$config"; then
            lines+=("  ✓ $config")
            ((completed++)) || true
        else
            lines+=("  ✗ $config (pending)")
            ((pending++)) || true
        fi
    done
    printf '%s\n' "${lines[@]}" | sort
    echo "============================================================================="
    echo "Completed: $completed / ${#CONFIGS[@]}"
    echo "============================================================================="
    exit 0
fi

# Determine which configs to run
declare -a TO_RUN=()

for config in "${!CONFIGS[@]}"; do
    if [ -n "$CONFIG_FILTER" ]; then
        # --config specified: only that one
        if [ "$config" = "$CONFIG_FILTER" ]; then
            if [ "$FORCE" = true ] || ! is_completed "$config"; then
                TO_RUN+=("$config")
            else
                echo "[INFO] $config already completed. Use --force to re-run."
            fi
        fi
    elif [ "$FORCE" = true ]; then
        TO_RUN+=("$config")
    elif ! is_completed "$config"; then
        TO_RUN+=("$config")
    fi
done

# Sort for consistent ordering
IFS=$'\n' TO_RUN_SORTED=($(sort <<<"${TO_RUN[*]}")); unset IFS

if [ ${#TO_RUN_SORTED[@]} -eq 0 ]; then
    echo "[INFO] All configs for $TASK_PATH are already completed. Use --force to re-run."
    exit 0
fi

echo "============================================================================="
echo "Local Run: $EXPERIMENT_NAME / $TASK_PATH"
echo "Configs to run: ${#TO_RUN_SORTED[@]} / ${#CONFIGS[@]}"
for config in "${TO_RUN_SORTED[@]}"; do
    echo "  - $config"
done
echo "Venv: $VENV_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================================================="

# Environment
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=42

# Model
MODEL_PATH="Efficient-Large-Model/Fast_dLLM_v2_7B"

# Run each config sequentially
completed_count=0
failed_count=0

for config in "${TO_RUN_SORTED[@]}"; do
    IFS='|' read -r REUSE_K LAYER_SUBSET <<< "${CONFIGS[$config]}"

    OUTPUT_DIR="${EXPERIMENT_DIR}/results/${TASK_PATH}/${config}"
    ARTIFACTS_DIR="${EXPERIMENT_DIR}/artifacts/${TASK_PATH}/${config}"
    mkdir -p "$OUTPUT_DIR"

    export TASK_NAME=$TASK
    export RUN_TAG=$config
    export OUTPUT_DIR=$OUTPUT_DIR

    echo ""
    echo "============================================================================="
    echo "[$((completed_count + failed_count + 1))/${#TO_RUN_SORTED[@]}] Running: $config (k=$REUSE_K, subset=$LAYER_SUBSET)"
    echo "Task: $TASK_PATH"
    echo "Output: $OUTPUT_DIR"
    echo "============================================================================="

    if "$VENV_PATH/bin/accelerate" launch eval.py \
        --tasks ${TASK} \
        --model fast_dllm_v2 \
        --batch_size 1 \
        --num_fewshot 0 \
        --fewshot_as_multiturn \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --model_args "model_path=${MODEL_PATH},reuse_k=${REUSE_K},layer_subset=${LAYER_SUBSET},use_block_cache=True,show_speed=True,experiment_name=${EXPERIMENT_NAME}" \
        ${LIMIT_ARG} \
        --output_path ${OUTPUT_DIR}/; then

        # Copy summary to artifacts
        mkdir -p "$ARTIFACTS_DIR"
        if [ -f "${OUTPUT_DIR}/summary.json" ]; then
            cp "${OUTPUT_DIR}/summary.json" "$ARTIFACTS_DIR/summary.json"
            echo "[INFO] ✓ $config complete - summary.json saved to artifacts"
        else
            echo "[WARN] $config finished but no summary.json found"
        fi
        ((completed_count++)) || true
    else
        echo "[ERROR] ✗ $config failed - continuing with next config"
        ((failed_count++)) || true
    fi
done

echo ""
echo "============================================================================="
echo "Done: $completed_count completed, $failed_count failed (out of ${#TO_RUN_SORTED[@]})"
echo "============================================================================="
