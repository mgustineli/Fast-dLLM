# PACE Cluster Reference

Quick reference for Georgia Tech PACE cluster resources.

> **For writing/reviewing sbatch scripts**: See `.claude/commands/SKILLS.md` for comprehensive patterns and checklists.

## Storage

| Location | Quota | Use For |
|----------|-------|---------|
| `~/` | 20 GB | Code, configs |
| `~/scratch/` | 15 TB | Datasets, venvs, outputs (60-day purge policy) |
| `/storage/coda1/p-<alloc>/0/shared/` | Varies | Team-shared data |
| `${TMPDIR}` | ~1 TB | Job-local temp (auto-cleaned) |

## GPU Partitions

| Partition | GPU | VRAM | Availability |
|-----------|-----|------|--------------|
| `gpu-rtx6000` | RTX 6000 | 24GB | **Best** (default) |
| `gpu-v100` | V100 | 16/32GB | Good |
| `gpu-l40s` | L40S | 48GB | Moderate |
| `gpu-a100` | A100 | 40/80GB | Limited |
| `gpu-h100` | H100 | 80GB | Very limited |

## Quick Commands

```bash
# Check quotas
pace-quota

# Check queue status
pace-check-queue

# Check account balance
pace-mybalance
```

## Array Jobs

Array jobs allow running multiple similar tasks as a single job submission. Each task gets a unique `$SLURM_ARRAY_TASK_ID`.

### Basic Usage

```bash
#SBATCH --array=0-8           # Run 9 tasks (indices 0-8)
#SBATCH --array=1-5           # Run 5 tasks (indices 1-5)
#SBATCH --array=1,3,5,7       # Run specific indices only
#SBATCH --array=0-8%3         # Run max 3 tasks at once (throttling)
```

### Output File Naming

Use `%A` (master job ID) and `%a` (task ID) in output file names:

```bash
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
```

### Using Array Task ID

```bash
# Map task ID to configuration
CONFIGS=("config1" "config2" "config3")
MY_CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

# Or use task ID directly
echo "Processing task $SLURM_ARRAY_TASK_ID"
```

### Example: Layer Reuse Experiments

```bash
# Submit all 9 experiments (3 subsets x 3 k values)
sbatch sbatch/eval_reuse_layers_array.sbatch

# Submit with sample limit for testing
sbatch sbatch/eval_reuse_layers_array.sbatch --limit 10

# Run only specific experiments (e.g., k=1 for all subsets)
sbatch --array=0,3,6 sbatch/eval_reuse_layers_array.sbatch
```

### Managing Array Jobs

```bash
# Check status of array job
squeue -u $USER

# Cancel entire array job
scancel <job_id>

# Cancel specific task
scancel <job_id>_<task_id>

# Modify throttling on running job
scontrol update ArrayTaskThrottle=5 JobId=<job_id>
```

### Key Variables

| Variable | Description |
|----------|-------------|
| `$SLURM_ARRAY_TASK_ID` | Current task index |
| `$SLURM_ARRAY_TASK_COUNT` | Total number of tasks |
| `$SLURM_ARRAY_JOB_ID` | Master job ID |
| `$SLURM_JOB_ID` | Unique ID for this task |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Disk quota exceeded | Move files to scratch, clear `~/.cache/torch` |
| No module found | Use `uv run`, check `UV_PROJECT_ENVIRONMENT` |
| Jobs stuck | Try different partition, reduce resources |
| Array job logs overwrite | Ensure `%A_%a` in output filename, not just `%A` |
