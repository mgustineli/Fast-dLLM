#!/bin/bash
# =============================================================================
# Common SLURM Job Utilities
# =============================================================================
# Shared functions for venv setup, environment configuration, and job logging.
# Source this file from experiment _job.sh scripts:
#
#   source "$PROJECT_ROOT/sbatch/common.sh"
#   setup_venv
#   setup_environment
#   print_job_info "Experiment Name" "$CONFIG_NAME" "$TASK" "$OUTPUT_DIR"
# =============================================================================

# Setup per-job venv in TMPDIR for fast local I/O
setup_venv() {
    local job_id="${SLURM_JOB_ID:-local}"
    local venv_dir="$TMPDIR/fast-dllm-$job_id"
    local venv_path="$venv_dir/.venv"

    echo "[INFO] Setting up venv at $venv_path..."
    mkdir -p "$venv_dir"
    uv venv "$venv_path"
    source "$venv_path/bin/activate"
    uv pip install -e "$PROJECT_ROOT"
    uv pip install torch --index-url https://download.pytorch.org/whl/cu121
    echo "[INFO] Venv setup complete"
    echo "[INFO] Python: $(which python)"
}

# Setup environment variables for evaluation
setup_environment() {
    export HF_ALLOW_CODE_EVAL=1
    export HF_DATASETS_TRUST_REMOTE_CODE=true
    export CUBLAS_WORKSPACE_CONFIG=":16:8"
    export PYTHONHASHSEED=42
}

# Print job info header
# Usage: print_job_info "Title" "$CONFIG_NAME" "$TASK" "$OUTPUT_DIR"
print_job_info() {
    local title="${1:-Experiment}"
    local config="${2:-unknown}"
    local task="${3:-unknown}"
    local output_dir="${4:-}"

    echo "============================================================================="
    echo "$title"
    echo "Config: $config"
    echo "Task: $task"
    if [ -n "$output_dir" ]; then
        echo "Output: $output_dir"
    fi
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "Job ID: $SLURM_JOB_ID"
        echo "Node: $SLURMD_NODENAME"
    fi
    echo "============================================================================="
}
