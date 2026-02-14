#!/bin/bash
# =============================================================================
# [EXPERIMENT_NAME] - Smart Runner
# =============================================================================
# Runs only experiments that haven't completed yet.
#
# Usage:
#   bash experiments/[NAME]/sbatch/run.sh --status
#   bash experiments/[NAME]/sbatch/run.sh --dry-run
#   bash experiments/[NAME]/sbatch/run.sh
#   bash experiments/[NAME]/sbatch/run.sh --task mmlu --limit 10
#   bash experiments/[NAME]/sbatch/run.sh --config k1_first
#   bash experiments/[NAME]/sbatch/run.sh --force
# =============================================================================

set -e

# Derive paths from script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
EXPERIMENT_NAME="$(basename "$EXPERIMENT_DIR")"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Configuration
TASK="gsm8k"
LIMIT_VALUE=""

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

# Help function
show_help() {
    cat << EOF
Usage: bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh [OPTIONS]

Smart runner that only submits experiments that haven't completed yet.

OPTIONS:
    -h, --help              Show this help message
    --status                Show completion status
    --dry-run               Preview jobs without submitting
    --limit <N>             Test mode: run only N samples
    --task <TASK>           Set task (default: gsm8k)
    --config <CONFIG>       Run only specific config (e.g., k1_first)
    --force [CONFIG]        Re-run all or specific config

TASKS:
    gsm8k, mmlu, gpqa_main_n_shot, minerva_math, ifeval

CONFIGS:
    k1_first, k1_middle, k1_last, k2_first, k2_middle, k2_last,
    k3_first, k3_middle, k3_last

EXAMPLES:
    bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --status
    bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --limit 10 --dry-run
    bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --config k1_first
    bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --task mmlu --limit 10
    bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --force k2_middle
EOF
}

# Parse arguments
DRY_RUN=false
FORCE=false
FORCE_CONFIG=""
CONFIG_FILTER=""
SHOW_STATUS=false
LIMIT_ARG=""
TASK_ARG=""
STATUS_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            if [[ $2 && ! $2 =~ ^-- ]]; then
                FORCE_CONFIG=$2
                shift
            fi
            shift
            ;;
        --status)
            SHOW_STATUS=true
            shift
            ;;
        --limit)
            LIMIT_VALUE=$2
            LIMIT_ARG="--limit $2"
            STATUS_ARGS="$STATUS_ARGS --limit $2"
            shift 2
            ;;
        --task)
            TASK=$2
            TASK_ARG="--task $2"
            STATUS_ARGS="$STATUS_ARGS --task $2"
            shift 2
            ;;
        --config)
            CONFIG_FILTER=$2
            if [[ -z "${CONFIGS[$CONFIG_FILTER]}" ]]; then
                echo "Error: Unknown config '$CONFIG_FILTER'"
                echo "Valid configs: ${!CONFIGS[@]}"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run 'bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --help' for usage"
            exit 1
            ;;
    esac
done

# Define TASK_PATH for directory structures
TASK_PATH="$TASK"
if [[ -n "$LIMIT_VALUE" ]]; then
    TASK_PATH="${TASK}_limit_${LIMIT_VALUE}"
fi

# Check completion via experiment-local artifacts
is_completed() {
    local config=$1
    local summary_file="${EXPERIMENT_DIR}/artifacts/${TASK_PATH}/${config}/summary.json"
    [[ -f "$summary_file" ]]
}

# Show status and exit
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
    should_run=false
    if [ -n "$CONFIG_FILTER" ]; then
        if [ "$config" = "$CONFIG_FILTER" ]; then
            should_run=true
        fi
    elif [ "$FORCE" = true ]; then
        if [ -z "$FORCE_CONFIG" ] || [ "$config" = "$FORCE_CONFIG" ]; then
            should_run=true
        fi
    elif ! is_completed "$config"; then
        should_run=true
    fi

    if [ "$should_run" = true ]; then
        TO_RUN+=("$config")
    fi
done

IFS=$'\n' TO_RUN_SORTED=($(sort <<<"${TO_RUN[*]}")); unset IFS

if [ ${#TO_RUN_SORTED[@]} -eq 0 ]; then
    if [ -n "$CONFIG_FILTER" ]; then
        echo "[INFO] Config '$CONFIG_FILTER' for task '$TASK_PATH' is already completed. Use --force to re-run."
    else
        echo "[INFO] All experiments for task '$TASK_PATH' are already completed. Use --force to re-run."
    fi
    exit 0
fi

echo "============================================================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Task:       $TASK_PATH"
echo "-----------------------------------------------------------------------------"
echo "Configs to run: ${#TO_RUN_SORTED[@]}"
for config in "${TO_RUN_SORTED[@]}"; do
    echo "  - $config"
done
echo "============================================================================="

if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] Would submit ${#TO_RUN_SORTED[@]} jobs. Use without --dry-run to execute."
    exit 0
fi

# Submit jobs
for config in "${TO_RUN_SORTED[@]}"; do
    IFS='|' read -r K SUBSET <<< "${CONFIGS[$config]}"

    echo "[INFO] Submitting: $config (k=$K, subset=$SUBSET)"

    sbatch \
        --job-name="${EXPERIMENT_NAME}-$config" \
        --output="${EXPERIMENT_DIR}/logs/${TASK_PATH}/${config}/slurm_%j.log" \
        --export=ALL,CONFIG_NAME="$config",REUSE_K="$K",LAYER_SUBSET="$SUBSET",LIMIT_ARG="$LIMIT_ARG",TASK_ARG="$TASK_ARG",EXPERIMENT_NAME="$EXPERIMENT_NAME",EXPERIMENT_DIR="$EXPERIMENT_DIR",TASK_PATH="$TASK_PATH" \
        "$SCRIPT_DIR/_job.sh"
done

echo ""
echo "[INFO] Submitted ${#TO_RUN_SORTED[@]} jobs. Check status with: squeue -u $USER"
echo "[INFO] View progress with: bash experiments/${EXPERIMENT_NAME}/sbatch/run.sh --status$STATUS_ARGS"
