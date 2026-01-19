#!/bin/bash
# =============================================================================
# Layer Reuse Experiments - Smart Runner
# =============================================================================
# Runs only experiments that haven't completed yet.
# Results are stored by config name (not timestamp) for easy detection.
#
# Usage:
#   # Run missing experiments for the default task (gsm8k)
#   bash sbatch/run.sh
#
#   # Run for a different task or with a sample limit
#   bash sbatch/run.sh --task mmlu
#   bash sbatch/run.sh --task mmlu --limit 10
#
#   # Check status (must pass the same flags as the run you want to check)
#   bash sbatch/run.sh --status
#   bash sbatch/run.sh --status --task mmlu --limit 10
#
#   # Other options
#   bash sbatch/run.sh --experiment 01_new_exp   # Use different experiment name
#   bash sbatch/run.sh --dry-run                 # Preview without submitting
#   bash sbatch/run.sh --force                   # Re-run all configs
#   bash sbatch/run.sh --force k2_middle         # Re-run specific config
#
# Tasks: gsm8k (default), mmlu, gpqa_main_n_shot, minerva_math, ifeval
# Configs: k1_first, k1_middle, k1_last, k2_first, k2_middle, k2_last, k3_first, k3_middle, k3_last
# =============================================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Configuration
TASK="gsm8k"
# Experiment name - use --experiment flag to override (e.g., 01_adaptive_skip)
EXPERIMENT="00_baseline"
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

# Help function (for --help flag)
show_help() {
    cat << 'EOF'
Usage: bash sbatch/run.sh [OPTIONS]

Smart runner that only submits experiments that haven't completed yet.

OPTIONS:
    -h, --help              Show this help message
    --status                Show completion status for a run. Pass the same flags
                            (--task, --limit, etc.) as the original run.
    --dry-run               Preview jobs without submitting
    --limit <N>             Test mode: run only N samples
    --task <TASK>           Set task (default: gsm8k)
    --experiment <NAME>     Set experiment (default: 00_baseline)
    --force [CONFIG]        Re-run all or specific config

TASKS:
    gsm8k, mmlu, gpqa_main_n_shot, minerva_math, ifeval

CONFIGS:
    k1_first, k1_middle, k1_last, k2_first, k2_middle, k2_last,
    k3_first, k3_middle, k3_last

EXAMPLES:
    bash sbatch/run.sh                           # Run missing (gsm8k)
    bash sbatch/run.sh --status                  # Show status
    bash sbatch/run.sh --experiment 01_new       # Different experiment
    bash sbatch/run.sh --limit 10 --dry-run      # Preview test run
    bash sbatch/run.sh --force k2_middle         # Re-run one config
EOF
}

# Parse arguments
DRY_RUN=false
FORCE=false
FORCE_CONFIG=""
SHOW_STATUS=false
LIMIT_ARG=""
TASK_ARG=""
EXPERIMENT_ARG=""
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
        --experiment)
            EXPERIMENT=$2
            EXPERIMENT_ARG="$2"
            STATUS_ARGS="$STATUS_ARGS --experiment $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Run 'bash sbatch/run.sh --help' for usage"
            exit 1
            ;;
    esac
done

# Define TASK_PATH for directory structures. It includes the limit if present.
TASK_PATH="$TASK"
if [[ -n "$LIMIT_VALUE" ]]; then
    TASK_PATH="${TASK}_limit_${LIMIT_VALUE}"
fi

# Function to check if experiment completed successfully
is_completed() {
    local config=$1
    # The summary file is stored in the artifacts directory
    local summary_file="artifacts/${EXPERIMENT}/${TASK_PATH}/${config}/summary.json"

    # Check if summary.json exists - this indicates successful completion
    if [[ -f "$summary_file" ]]; then
        return 0  # Completed
    fi

    return 1  # Not completed
}

# Show status and exit
if [ "$SHOW_STATUS" = true ]; then
    echo "============================================================================="
    echo "Experiment Status: $EXPERIMENT / $TASK"
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

    # Sort and print the collected lines
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

    if [ "$FORCE" = true ]; then
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

# Sort configs for consistent ordering
IFS=$'\n' TO_RUN_SORTED=($(sort <<<"${TO_RUN[*]}")); unset IFS

if [ ${#TO_RUN_SORTED[@]} -eq 0 ]; then
    echo "[INFO] All experiments for task '$TASK_PATH' in experiment '$EXPERIMENT' are already completed. Use --force to re-run."
    exit 0
fi

echo "============================================================================="
echo "Experiment: $EXPERIMENT"
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
        --job-name="baseline-$config" \
        --output="logs/${EXPERIMENT}/${TASK_PATH}/${config}/slurm_%j.log" \
        --export=ALL,CONFIG_NAME="$config",REUSE_K="$K",LAYER_SUBSET="$SUBSET",LIMIT_ARG="$LIMIT_ARG",TASK_ARG="$TASK_ARG",EXPERIMENT_NAME="$EXPERIMENT",TASK_PATH="$TASK_PATH" \
        "$SCRIPT_DIR/_job.sh"
done

echo ""
echo "[INFO] Submitted ${#TO_RUN_SORTED[@]} jobs. Check status with: squeue -u $USER"
echo "[INFO] View progress with: bash sbatch/run.sh --status$STATUS_ARGS"
