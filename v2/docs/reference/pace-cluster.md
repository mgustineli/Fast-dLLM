# PACE Cluster Reference

Quick reference for Georgia Tech PACE cluster resources.

## Storage

| Location | Quota | Speed | Persistent | Use For |
|----------|-------|-------|------------|---------|
| `~/` | 20 GB | Medium | Yes | Code, configs |
| `~/scratch/` | 15 TB | Slow I/O | Yes* | Venvs, models, outputs |
| `${TMPDIR}` | ~1 TB | Fast | No | Job-local temp |

*60-day purge policy on scratch

### Recommended Setup

```bash
# In ~/.bashrc - redirect caches to scratch
export XDG_CACHE_HOME="$HOME/scratch/.cache"
```

This automatically redirects HuggingFace, pip, and other caches to scratch.

## GPU Partitions

| Partition | GPU | VRAM | Tensor Cores | Best Dtype |
|-----------|-----|------|--------------|------------|
| `gpu-rtx6000` | RTX 6000 | 24GB | Turing | **float16** |
| `gpu-v100` | V100 | 16/32GB | Volta | float16 |
| `gpu-a100` | A100 | 40/80GB | Ampere | bfloat16 |
| `gpu-h100` | H100 | 80GB | Hopper | bfloat16 |

**Note**: RTX 6000 (Turing) doesn't support bfloat16. Our `eval.py` auto-detects GPU and uses float16 for Turing GPUs.

## Quick Commands

```bash
# Check quotas
pace-quota

# Check queue status
pace-check-queue

# Check account balance
pace-mybalance

# View your jobs
squeue -u $USER

# Cancel job
scancel <job_id>
```

## Virtual Environment Strategy

### For SLURM Jobs (Automatic)

SLURM scripts automatically create/use `~/scratch/Fast-dLLM/v2/.venv`:
- Created on first run with file locking (no race conditions)
- Shared across all jobs (no redundant installs)
- Persistent across sessions

### For Interactive Sessions

```bash
# Option 1: Use scratch venv (persistent)
source ~/scratch/Fast-dLLM/v2/.venv/bin/activate

# Option 2: Fast TMPDIR venv (wiped after session)
cd ~/eic-lab/Fast-dLLM/v2
source setup_tmpdir_venv.sh
```

## Running Experiments

### Smart Experiment Runner (Recommended)

```bash
cd ~/eic-lab/Fast-dLLM/v2

# Check what's completed vs pending
bash sbatch/run.sh --status

# Run only missing experiments
bash sbatch/run.sh

# Test mode (10 samples)
bash sbatch/run.sh --limit 10

# Force re-run specific config
bash sbatch/run.sh --force k2_middle
```

### Direct SLURM Submission

```bash
# Single baseline evaluation
sbatch sbatch/baseline.sh --task gsm8k --limit 10

# Array job (deprecated - use run.sh instead)
sbatch sbatch/_array.sh --limit 10
```

## Array Jobs (Reference)

Array jobs run multiple similar tasks as a single submission.

### Basic Syntax

```bash
#SBATCH --array=0-8           # Run 9 tasks (indices 0-8)
#SBATCH --array=1,3,5,7       # Run specific indices only
#SBATCH --array=0-8%3         # Max 3 concurrent tasks
```

### Output Naming

```bash
#SBATCH --output=logs/job_%A_%a.out   # %A=master ID, %a=task ID
```

### Key Variables

| Variable | Description |
|----------|-------------|
| `$SLURM_ARRAY_TASK_ID` | Current task index |
| `$SLURM_ARRAY_TASK_COUNT` | Total number of tasks |
| `$SLURM_ARRAY_JOB_ID` | Master job ID |

### Why We Use Individual Jobs Instead

Array jobs have issues on PACE:
1. Multiple tasks can land on same node, sharing `$TMPDIR`
2. Race conditions with shared resources
3. Hard to re-run only failed tasks

Our `run.sh` submits individual jobs:
- Checks completion before submitting
- Only runs missing experiments
- Easier to manage and re-run

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Disk quota exceeded | Move to scratch, clear `~/.cache` |
| Jobs stuck at venv setup | Delete `~/scratch/Fast-dLLM/v2/.venv`, retry |
| I/O errors on scratch | Known PACE issue; retry or reduce concurrent jobs |
| Wrong dtype (slow) | Check GPU with `nvidia-smi`; eval.py auto-detects |
| Array tasks conflict | Use `run.sh` instead |

## Useful SLURM Commands

```bash
# Detailed job info
scontrol show job <job_id>

# Job history
sacct -u $USER --starttime=today

# Cancel all your jobs
scancel -u $USER

# Check node availability
sinfo -p gpu-rtx6000
```
