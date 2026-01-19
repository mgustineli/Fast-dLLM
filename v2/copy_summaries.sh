#!/bin/bash
# =============================================================================
# Copy summary.json files from results/ to artifacts/
# Only copies files that don't already exist in artifacts/
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

copied=0
skipped=0

for src in $(find results -name "summary.json" 2>/dev/null); do
    dest="artifacts/${src#results/}"

    if [ -f "$dest" ]; then
        echo "[SKIP] $dest already exists"
        ((skipped++)) || true
    else
        mkdir -p "$(dirname "$dest")"
        cp "$src" "$dest"
        echo "[COPY] $src -> $dest"
        ((copied++)) || true
    fi
done

echo ""
echo "Done: $copied copied, $skipped skipped"
