# Operations Guide

How to run Fast-dLLM v2 experiments on the Georgia Tech PACE cluster.

> **Prerequisites**: See [PACE Cluster Reference](reference/pace-cluster.md) for disk quotas, SLURM queues, and troubleshooting.

## Directory Structure

```
~/eic-lab/Fast-dLLM/v2/          # Code repository (can be anywhere)
├── eval.py                       # Evaluation harness (lm_eval integration)
├── generation_functions.py       # Core generation algorithms (active/dev)
├── experiments/                  # Versioned generation code per experiment
│   ├── 00_baseline/
│   │   └── generation_functions.py  # Frozen baseline code
│   └── 01_adaptive_skip/
│       └── generation_functions.py  # Modified experiment code
├── app.py                        # Gradio web interface
├── run_chatbot.py               # CLI chatbot
├── sbatch/                       # SLURM job scripts
│   ├── run.sh                    # Smart experiment runner (primary)
│   ├── baseline.sh               # Baseline evaluation
│   ├── local.sh                  # Local evaluation (no SLURM)
│   ├── _job.sh                   # Single experiment job (internal)
│   └── _array.sh                 # Array job (deprecated)
├── setup_tmpdir_venv.sh          # Interactive venv setup
├── results/                      # Experiment outputs (not tracked)
│   └── {experiment}/{task}/{config}/
├── artifacts/                    # Git-tracked results summaries
│   └── {experiment}/{task}/{config}/summary.json
└── logs/                         # SLURM logs

~/scratch/Fast-dLLM/v2/           # Persistent storage (auto-created)
└── .venv/                        # Shared venv for SLURM jobs
```

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run.sh` | Smart runner: only runs missing experiments | **Primary workflow** |
| `baseline.sh` | Baseline evaluation (multi-task) | Baseline benchmarks |
| `local.sh` | Local evaluation (no SLURM) | Interactive testing |
| `_job.sh` | Single SLURM job for one config | Internal (called by `run.sh`) |
| `_array.sh` | Array job for layer reuse | **Deprecated** |
| `setup_tmpdir_venv.sh` | Create fast TMPDIR venv | Interactive sessions only |

## Quick Start

### 1. Check Experiment Status

```bash
cd ~/eic-lab/Fast-dLLM/v2
bash sbatch/run.sh --status
```

Output:
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

### 2. Run Missing Experiments

```bash
# Preview what would run (no submission)
bash sbatch/run.sh --dry-run

# Submit only missing experiments
bash sbatch/run.sh

# Test mode (10 samples per experiment)
bash sbatch/run.sh --limit 10
```

### 3. Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# Check experiment status
bash sbatch/run.sh --status

# View logs
tail -f logs/00_baseline/gsm8k/k2_middle/slurm_*.log
```

## Environment Setup

### Automatic (SLURM Jobs)

SLURM scripts automatically:
1. Create `~/scratch/Fast-dLLM/v2/.venv` if missing (first run only)
2. Activate the venv
3. Use file locking to prevent race conditions

No manual setup required.

### Manual (Interactive Sessions)

For Jupyter/interactive work:

```bash
# Option 1: Use scratch venv (persistent, slower I/O)
source ~/scratch/Fast-dLLM/v2/.venv/bin/activate

# Option 2: Create fast TMPDIR venv (wiped after session)
cd ~/eic-lab/Fast-dLLM/v2
source setup_tmpdir_venv.sh
```

### Environment Variables

Set in `~/.bashrc`:

```bash
# Redirect caches to scratch (large storage)
export XDG_CACHE_HOME="$HOME/scratch/.cache"

# Required for lm_eval
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

## Running Experiments

### Layer Reuse Experiments (Primary)

```bash
# Check status (default: 00_baseline experiment)
bash sbatch/run.sh --status

# Run missing experiments
bash sbatch/run.sh

# Run with test mode (10 samples)
bash sbatch/run.sh --limit 10

# Force re-run specific config
bash sbatch/run.sh --force k2_middle

# Force re-run all
bash sbatch/run.sh --force

# Different task (default: gsm8k)
bash sbatch/run.sh --task mmlu

# Different experiment (loads code from experiments/{name}/)
bash sbatch/run.sh --experiment 01_adaptive_skip
bash sbatch/run.sh --experiment 01_adaptive_skip --limit 10
```

### Baseline Evaluation

```bash
# Single task
sbatch sbatch/baseline.sh --task gsm8k

# All tasks (mmlu, gpqa, gsm8k, minerva_math, ifeval)
sbatch sbatch/baseline.sh

# Test mode
sbatch sbatch/baseline.sh --task gsm8k --limit 10
```

### Local Evaluation (No SLURM)

```bash
# From v2 directory
bash sbatch/local.sh --task gsm8k --limit 10
```

### Custom Evaluation

```bash
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 1 \
    --num_fewshot 0 \
    --model fast_dllm_v2 \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --model_args "model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=1,show_speed=True" \
    --output_path results/my_experiment/
```

## Model Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `model_path` | HuggingFace path | Model to load |
| `threshold` | 0.0-1.0 | Confidence threshold for parallel decoding |
| `use_block_cache` | True/False | Enable block-level KV caching |
| `show_speed` | True/False | Display throughput metrics |
| `reuse_k` | 1,2,3 | Layer reuse parameter (skip every k-th step) |
| `layer_subset` | first/middle/last | Which 12 layers to apply reuse |
| `experiment_name` | string | Load generation code from `experiments/{name}/` |

## Results Structure

```
results/{experiment}/{task}/          # Full results (not git-tracked)
├── k1_first/
│   ├── results.json                  # Metrics (accuracy, etc.)
│   └── summary.json                  # Throughput, config
├── k2_middle/
│   └── ...
└── k3_last/
    └── ...

artifacts/{experiment}/{task}/        # Git-tracked summaries
├── k1_first/
│   └── summary.json
└── ...

experiments/{experiment}/             # Versioned generation code
└── generation_functions.py           # Frozen code for reproducibility
```

## Interactive Usage

### Gradio Web UI

```bash
# Activate venv first
source ~/scratch/Fast-dLLM/v2/.venv/bin/activate
python app.py
# Opens at http://localhost:10086
```

### CLI Chatbot

```bash
source ~/scratch/Fast-dLLM/v2/.venv/bin/activate
python run_chatbot.py
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Jobs stuck at venv setup | Check `~/scratch/Fast-dLLM/v2/.venv` exists; delete and retry |
| CUDA OOM | Use `--batch_size 1` (default) |
| I/O errors on scratch | Known PACE issue; retry or use TMPDIR for interactive |
| Missing experiments | Run `bash sbatch/run.sh` to submit missing |
| Re-run specific config | Use `--force k2_middle` flag |

See [PACE Cluster Reference](reference/pace-cluster.md) for more troubleshooting.
