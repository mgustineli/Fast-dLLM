# Operations Guide

How to run Fast-dLLM v2 experiments on the Georgia Tech PACE cluster.

> **Prerequisites**: See [PACE Cluster Reference](reference/pace-cluster.md) for disk quotas, SLURM queues, and troubleshooting.

## Directory Structure

```
~/eic-lab/Fast-dLLM/v2/          # Code repository (can be anywhere)
├── eval.py                       # Evaluation harness (lm_eval integration)
├── generation_functions.py       # Core generation algorithms
├── app.py                        # Gradio web interface
├── run_chatbot.py               # CLI chatbot
├── sbatch/                       # SLURM job scripts
│   ├── run_reuse_experiments.sh  # Smart experiment runner
│   ├── eval_reuse_single.sh      # Single experiment job
│   ├── eval_reuse_layers_array.sh # Array job (deprecated)
│   ├── eval_pace_script.sh       # Baseline evaluation
│   └── eval_script.sh            # Local evaluation
├── setup_tmpdir_venv.sh          # Interactive venv setup
├── results/                      # Experiment outputs
│   └── reuse_layers/gsm8k/
│       ├── k1_first/             # Config-named directories
│       ├── k2_middle/
│       └── ...
└── logs/                         # SLURM logs

~/scratch/Fast-dLLM/v2/           # Persistent storage (auto-created)
└── .venv/                        # Shared venv for SLURM jobs
```

## Scripts Overview

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `run_reuse_experiments.sh` | Smart runner: only runs missing experiments | **Primary workflow** |
| `eval_reuse_single.sh` | Single SLURM job for one config | Called by runner |
| `eval_pace_script.sh` | Baseline evaluation (multi-task) | Baseline benchmarks |
| `eval_script.sh` | Local evaluation (no SLURM) | Interactive testing |
| `setup_tmpdir_venv.sh` | Create fast TMPDIR venv | Interactive sessions only |

## Quick Start

### 1. Check Experiment Status

```bash
cd ~/eic-lab/Fast-dLLM/v2
bash sbatch/run_reuse_experiments.sh --status
```

Output:
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

### 2. Run Missing Experiments

```bash
# Preview what would run (no submission)
bash sbatch/run_reuse_experiments.sh --dry-run

# Submit only missing experiments
bash sbatch/run_reuse_experiments.sh

# Test mode (10 samples per experiment)
bash sbatch/run_reuse_experiments.sh --limit 10
```

### 3. Monitor Progress

```bash
# Check SLURM queue
squeue -u $USER

# Check experiment status
bash sbatch/run_reuse_experiments.sh --status

# View logs
tail -f logs/reuse_layers/gsm8k/k2_middle/slurm_*.log
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
# Check status
bash sbatch/run_reuse_experiments.sh --status

# Run missing experiments
bash sbatch/run_reuse_experiments.sh

# Run with test mode (10 samples)
bash sbatch/run_reuse_experiments.sh --limit 10

# Force re-run specific config
bash sbatch/run_reuse_experiments.sh --force k2_middle

# Force re-run all
bash sbatch/run_reuse_experiments.sh --force

# Different task (default: gsm8k)
bash sbatch/run_reuse_experiments.sh --task mmlu
```

### Baseline Evaluation

```bash
# Single task
sbatch sbatch/eval_pace_script.sh --task gsm8k

# All tasks (mmlu, gpqa, gsm8k, minerva_math, ifeval)
sbatch sbatch/eval_pace_script.sh

# Test mode
sbatch sbatch/eval_pace_script.sh --task gsm8k --limit 10
```

### Local Evaluation (No SLURM)

```bash
# From v2 directory
bash sbatch/eval_script.sh --task gsm8k --limit 10
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

## Results Structure

```
results/reuse_layers/gsm8k/
├── k1_first/
│   ├── results.json         # Metrics (accuracy, etc.)
│   ├── summary.json         # Throughput, config
│   └── slurm.log           # Job log
├── k2_middle/
│   └── ...
└── k3_last/
    └── ...
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
| Missing experiments | Run `bash sbatch/run_reuse_experiments.sh` to submit missing |
| Re-run specific config | Use `--force k2_middle` flag |

See [PACE Cluster Reference](reference/pace-cluster.md) for more troubleshooting.
