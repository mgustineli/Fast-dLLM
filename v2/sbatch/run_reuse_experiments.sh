#!/bin/bash
# =============================================================================
# Layer Reuse Experiments - Smart Runner
# =============================================================================
# Runs only experiments that haven't completed yet.
# Results are stored by config name (not timestamp) for easy detection.
#
# Usage:
#   bash sbatch/run_reuse_experiments.sh                    # Run missing only
#   bash sbatch/run_reuse_experiments.sh --dry-run          # Show what would run
#   bash sbatch/run_reuse_experiments.sh --force            # Re-run all
#   bash sbatch/run_reuse_experiments.sh --force k2_middle  # Re-run specific config
#   bash sbatch/run_reuse_experiments.sh --status           # Show completion status
#   bash sbatch/run_reuse_experiments.sh --limit 10         # Test mode (10 samples)
#
# Configs: k1_first, k1_middle, k1_last, k2_first, k2_middle, k2_last, k3_first, k3_middle, k3_last
# =============================================================================

set -e

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Configuration
TASK="gsm8k"
EXPERIMENT="reuse_layers"
RESULTS_BASE="results/${EXPERIMENT}/${TASK}"

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
DRY_RUN=false
FORCE=false
FORCE_CONFIG=""
SHOW_STATUS=false
LIMIT_ARG=""
TASK_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
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
            LIMIT_ARG="--limit $2"
            shift 2
            ;;
        --task)
            TASK=$2
            TASK_ARG="--task $2"
            RESULTS_BASE="results/${EXPERIMENT}/${TASK}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Function to check if experiment completed successfully
is_completed() {
    local config=$1
    local results_dir="$RESULTS_BASE/$config"

    # Check if results.json exists and contains valid results
    if [ -f "$results_dir/results.json" ]; then
        # Check if the file has actual results (not empty/error)
        if grep -q "\"acc\"" "$results_dir/results.json" 2>/dev/null; then
            return 0  # Completed
        fi
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
    for config in "${!CONFIGS[@]}"; do
        if is_completed "$config"; then
            echo "  ✓ $config"
            ((completed++))
        else
            echo "  ✗ $config (pending)"
            ((pending++))
        fi
    done | sort
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
    echo "[INFO] All experiments already completed. Use --force to re-run."
    exit 0
fi

echo "============================================================================="
echo "Experiments to run: ${#TO_RUN_SORTED[@]}"
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
        --job-name="reuse-$config" \
        --output="logs/${EXPERIMENT}/${TASK}/${config}/slurm_%j.log" \
        --export=ALL,CONFIG_NAME="$config",REUSE_K="$K",LAYER_SUBSET="$SUBSET",LIMIT_ARG="$LIMIT_ARG",TASK_ARG="$TASK_ARG" \
        "$SCRIPT_DIR/eval_reuse_single.sh"
done

echo ""
echo "[INFO] Submitted ${#TO_RUN_SORTED[@]} jobs. Check status with: squeue -u $USER"
echo "[INFO] View progress with: bash sbatch/run_reuse_experiments.sh --status"
