# GPQA Dataset Requires HuggingFace Authentication

**Status:** TODO
**Priority:** High
**Affects:** All GPQA experiments

## Issue

GPQA experiments fail with `DatasetNotFoundError` because the dataset is gated on HuggingFace Hub.

## Error Message

```
datasets.exceptions.DatasetNotFoundError: Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub.
You must be authenticated to access it.
```

## Root Cause

- The GPQA dataset (`Idavidrein/gpqa`) requires users to accept terms of use
- Job scripts don't currently pass HuggingFace authentication tokens
- Environment variable `HF_TOKEN` is not set in sbatch jobs

## Solution

### Step 1: Request Dataset Access

1. Visit https://huggingface.co/datasets/Idavidrein/gpqa
2. Click "Access repository" and accept terms of use
3. Wait for approval (usually instant)

### Step 2: Add Authentication to Job Scripts

Modify `sbatch/_job.sh` to include HuggingFace token in environment setup.

**Location:** `sbatch/_job.sh:86-90`

**Add this line:**
```bash
# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.huggingface/token 2>/dev/null)}"  # Add this
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export PYTHONHASHSEED=42
```

### Step 3: Store Your Token

Create token file in your home directory:

```bash
# Create token file
mkdir -p ~/.huggingface
echo "your_hf_token_here" > ~/.huggingface/token
chmod 600 ~/.huggingface/token
```

Get your token from: https://huggingface.co/settings/tokens

## Files to Modify

- `sbatch/_job.sh:86-90` - Add `HF_TOKEN` export to environment setup

## Verification

After implementation:
1. Run GPQA experiment with `--limit 10`
2. Check logs for successful dataset loading (no authentication errors)
3. Verify experiments complete successfully

## Related Logs

Failed experiment logs showing this issue:
- `logs/00_baseline/gpqa_main_n_shot_limit_10/k1_first/slurm_3352792.log:217`
