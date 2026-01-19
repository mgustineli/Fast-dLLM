# Experiments

Documentation for Fast-dLLM v2 experiments and evaluation scripts.

## Quick Start

```bash
cd ~/eic-lab/Fast-dLLM/v2

# Show all options, tasks, configs, and examples
bash sbatch/run.sh --help

# Check experiment status (default: 00_baseline)
bash sbatch/run.sh --status

# Run missing experiments (test mode)
bash sbatch/run.sh --limit 10

# Run missing experiments (full)
bash sbatch/run.sh

# Run a different experiment version
bash sbatch/run.sh --experiment 01_adaptive_skip
```

---

## Results Directory Structure

Results are stored by **config name** (not timestamp) for easy completion detection:

```
results/                              # Full results (not git-tracked)
├── 00_baseline/                      # Experiment name
│   └── gsm8k/                        # Task
│       ├── k1_first/                 # Config name
│       │   ├── results.json          # Metrics and scores
│       │   └── summary.json          # Throughput, config info
│       ├── k1_middle/
│       └── ...
└── 01_adaptive_skip/
    └── gsm8k/
        └── ...

artifacts/                            # Git-tracked summaries
├── 00_baseline/
│   └── gsm8k/
│       ├── k1_first/
│       │   └── summary.json
│       └── ...
└── 01_adaptive_skip/
    └── ...

experiments/                          # Versioned generation code
├── 00_baseline/
│   └── generation_functions.py       # Frozen baseline code
└── 01_adaptive_skip/
    └── generation_functions.py       # Modified experiment code
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
Experiment Status: 00_baseline / gsm8k
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
bash sbatch/run.sh --status --experiment 01_adaptive_skip

# View live logs
tail -f logs/00_baseline/gsm8k/k2_middle/slurm_*.log

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

1. Looking for `results/<experiment>/<task>/<config>/summary.json`
2. If the file exists, the experiment is considered complete

This allows:
- Re-running only failed/incomplete experiments
- Easy tracking of progress with `--status`
- Forcing re-runs with `--force`

## Creating New Experiments

To create a new experiment with modified generation code:

```bash
# 1. Create experiment directory
mkdir -p experiments/01_adaptive_skip

# 2. Copy and modify generation_functions.py
cp generation_functions.py experiments/01_adaptive_skip/

# 3. Edit the experiment code
# vim experiments/01_adaptive_skip/generation_functions.py

# 4. Run the experiment
bash sbatch/run.sh --experiment 01_adaptive_skip --limit 10  # Test first
bash sbatch/run.sh --experiment 01_adaptive_skip             # Full run
```

The `--experiment` flag:
- Loads `generation_functions.py` from `experiments/{name}/`
- Stores results in `results/{name}/{task}/{config}/`
- Copies `summary.json` to `artifacts/{name}/{task}/{config}/`
