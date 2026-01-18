#!/bin/bash
# =============================================================================
# Setup Virtual Environment in $TMPDIR
# =============================================================================
# Creates a fast, temporary venv in $TMPDIR for GPU jobs.
# This avoids network I/O bottlenecks from ~/scratch.
#
# Usage:
#   source v2/setup_tmpdir_venv.sh
#
# Note: $TMPDIR is wiped when the job ends. Re-run this script for each session.
# =============================================================================

set -e

# Ensure TMPDIR is set
export TMPDIR=${TMPDIR:-/tmp}

# Get the directory where this script is located (v2/) and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYPROJECT_PATH="${SCRIPT_DIR}/pyproject.toml"

echo "[INFO] TMPDIR: $TMPDIR"
echo "[INFO] Script dir: $SCRIPT_DIR"
echo "[INFO] pyproject.toml: $PYPROJECT_PATH"

# Check if pyproject.toml exists
if [ ! -f "$PYPROJECT_PATH" ]; then
    echo "[ERROR] pyproject.toml not found at $PYPROJECT_PATH"
    return 1 2>/dev/null || exit 1
fi

# Create venv in TMPDIR if it doesn't exist
VENV_PATH="$TMPDIR/.venv"
if [ ! -d "$VENV_PATH" ]; then
    echo "[INFO] Creating virtual environment at $VENV_PATH"
    uv venv "$VENV_PATH"
else
    echo "[INFO] Virtual environment already exists at $VENV_PATH"
fi

# Create symlink from repo root to the venv in TMPDIR
SYMLINK_PATH="$REPO_ROOT/.venv"
if [ -L "$SYMLINK_PATH" ]; then
    # Remove existing symlink
    rm "$SYMLINK_PATH"
fi
if [ -d "$SYMLINK_PATH" ]; then
    echo "[WARN] $SYMLINK_PATH is a real directory, skipping symlink"
else
    echo "[INFO] Creating symlink: $SYMLINK_PATH -> $VENV_PATH"
    ln -s "$VENV_PATH" "$SYMLINK_PATH"
fi

# Activate the venv
echo "[INFO] Activating virtual environment"
source "$VENV_PATH/bin/activate"

# Install dependencies from pyproject.toml
echo "[INFO] Installing dependencies from $PYPROJECT_PATH"
uv pip install -e "$SCRIPT_DIR"

# Also install torch (not in pyproject.toml but required)
echo "[INFO] Installing PyTorch"
uv pip install torch --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "============================================================================="
echo "Virtual environment ready!"
echo "Location: $VENV_PATH"
echo "Symlink: $SYMLINK_PATH -> $VENV_PATH"
echo "Python: $(which python)"
echo "============================================================================="
echo ""
echo "To activate in future shells:"
echo "  source $REPO_ROOT/.venv/bin/activate"
echo "  # or"
echo "  source $VENV_PATH/bin/activate"
