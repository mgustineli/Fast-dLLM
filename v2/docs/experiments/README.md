# Experiments

Documentation for Fast-dLLM v2 experiments and evaluation scripts.

## Quick Start

```bash
cd ~/eic-lab/Fast-dLLM/v2

# Check experiment status
bash sbatch/run.sh --status

# Run missing experiments (test mode)
bash sbatch/run.sh --limit 10

# Run missing experiments (full)
bash sbatch/run.sh
```

---

## Results Directory Structure

Results are stored by **config name** (not timestamp) for easy completion detection:

```
results/
├── reuse_layers/                    # Experiment type
│   └── gsm8k/                       # Task
│       ├── k1_first/                # Config name
│       │   ├── results.json         # Metrics and scores
│       │   ├── summary.json         # Throughput, config info
│       │   └── slurm.log            # Job log
│       ├── k1_middle/
│       ├── k2_first/
│       └── ...
└── baseline/
    └── gsm8k/
        └── threshold1/
```

---

## Layer Reuse Experiments (Primary)

Skip transformer layers during diffusion steps by reusing outputs from previous steps.

### Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `reuse_k` | 1, 2, 3 | Reuse every k-th step (k=1 means no reuse) |
| `layer_subset` | first, middle, last | Which 12 layers to apply reuse |

### Experiment Matrix (9 configurations)

| Subset | k=1 | k=2 | k=3 |
|--------|-----|-----|-----|
| first (layers 1-11) | k1_first | k2_first | k3_first |
| middle (layers 12-23) | k1_middle | k2_middle | k3_middle |
| last (layers 24-35) | k1_last | k2_last | k3_last |

### Running Experiments

**Primary workflow - Smart Runner:**

```bash
# Check what's completed vs pending
bash sbatch/run.sh --status

# Run only missing experiments
bash sbatch/run.sh

# Test mode (10 samples per experiment)
bash sbatch/run.sh --limit 10

# Different task
bash sbatch/run.sh --task mmlu

# Combine options
bash sbatch/run.sh --task mmlu --limit 10

# Preview without submitting
bash sbatch/run.sh --dry-run

# Force re-run all
bash sbatch/run.sh --force

# Force re-run specific config
bash sbatch/run.sh --force k2_middle
```

**Status output example:**

```
=============================================================================
Experiment Status: reuse_layers / gsm8k
=============================================================================
  ✓ k1_first
  ✓ k1_middle
  ✗ k2_first (pending)
  ✗ k2_middle (pending)
  ...
=============================================================================
Completed: 2 / 9
=============================================================================
```

---

## Baseline Experiments

Standard Fast-dLLM v2 inference with threshold-based parallel decoding.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold` | 1.0 | Confidence threshold for token unmasking |
| `use_block_cache` | True | Enable block-level KV caching |

**Script:** `sbatch/baseline.sh`

```bash
# Single task
sbatch sbatch/baseline.sh --task gsm8k

# Test mode
sbatch sbatch/baseline.sh --task gsm8k --limit 10

# All tasks (mmlu, gpqa, gsm8k, minerva_math, ifeval)
sbatch sbatch/baseline.sh
```

---

## Local Testing (No SLURM)

For interactive testing without submitting to the cluster:

**Script:** `sbatch/local.sh`

```bash
# Run locally
bash sbatch/local.sh --task gsm8k --limit 10
```

---

## Benchmark Tasks

| Task | Description | Metric |
|------|-------------|--------|
| `gsm8k` | Grade School Math 8K | Accuracy |
| `mmlu` | Massive Multitask Language Understanding | Accuracy |
| `gpqa_main_n_shot` | Graduate-level Q&A | Accuracy |
| `minerva_math` | Mathematical reasoning | Accuracy |
| `ifeval` | Instruction following | Pass rate |

---

## Monitoring Jobs

```bash
# Check job queue
squeue -u $USER

# Check experiment completion
bash sbatch/run.sh --status

# View live logs
tail -f logs/reuse_layers/gsm8k/k2_middle/slurm_*.log

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `run.sh` | Smart runner (primary) | `bash sbatch/run.sh --status` |
| `baseline.sh` | Baseline evaluation | `sbatch sbatch/baseline.sh --task gsm8k` |
| `local.sh` | Local testing | `bash sbatch/local.sh --task gsm8k` |
| `_job.sh` | Single experiment job | Internal (called by `run.sh`) |
| `_array.sh` | Array job | **Deprecated** |

---

## Expected Results

### Baseline Performance (Fast-dLLM v2 7B)

| Task | Score |
|------|-------|
| GSM8K | 83.7% |
| HumanEval | 63.4% |
| MMLU | 66.6% |
| GPQA | 31.9% |

### Layer Reuse Tradeoffs (Expected)

| Config | Expected Speedup | Expected Accuracy Impact |
|--------|------------------|-------------------------|
| k=1 | Baseline | None |
| k=2 | ~1.2-1.4x | Small drop |
| k=3 | ~1.3-1.5x | Moderate drop |

*Note: Middle layers typically show best speedup/accuracy tradeoff.*

---

## How Completion Detection Works

The smart runner (`run.sh`) checks for completion by:

1. Looking for `results/<experiment>/<task>/<config>/results.json`
2. Verifying the file contains `"acc"` (valid accuracy result)

This allows:
- Re-running only failed/incomplete experiments
- Easy tracking of progress with `--status`
- Forcing re-runs with `--force`
