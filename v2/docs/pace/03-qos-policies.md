# PACE Phoenix QOS Policies

Two QOS (Quality of Service) levels on Phoenix-Slurm: **inferno** and **embers**

Both provide access to the same resource pool, but with different job policies. Jobs are automatically assigned to appropriate nodes based on requested resources.

---

## Inferno: The Primary QOS

**Main production QOS** - Jobs consume account credits but get priority treatment.

### Policies

- **Base priority:** 250,000
- **Max jobs per user:** 500
- **Max eligible jobs per user:** 500
- **Wallclock limit:** Minimum of:
  - 21 days for CPU resources (CPU-192GB or CPU-768GB-SAS)
  - 3 days for GPU resources (GPU-V100, GPU-RTX6000, GPU-A100, GPU-H100, GPU-H200)
  - 264,960 CPU-hours ÷ Number of Requested Processors

### Key Benefits

- ✅ Higher priority - jobs start faster
- ✅ No preemption - your job won't be killed
- ✅ Longer walltime limits
- ✅ Best for: Production workflows, critical work

### How to Use

```bash
# Option 1: Explicitly specify
salloc -q inferno ...
#SBATCH -q inferno

# Option 2: Omit flag (inferno is default)
salloc ...
```

### Important Note

The scheduler will reject a job if: `nodes × cores per node × walltime > 264,960 processor-hours`

If your job completes with no output, check this calculation.

---

## Embers: The Backfill QOS

**Free backfill QOS** - Opportunistic scheduling, no credits consumed.

### Policies

- **Base priority:** 0 (lowest)
- **Max jobs per user:** 50
- **Max eligible jobs per user:** 1
- **Wallclock limit:** 8 hours
- **Preemption:** Eligible after 1 hour
  - If an inferno job needs resources, running embers jobs may be killed
  - You can resubmit if needed

### Key Characteristics

- ✅ **FREE** - No account credits consumed
- ⚠️ Lower priority
- ⚠️ May be preempted after 1 hour
- ⚠️ Shorter walltime limits
- ✅ Best for: Development, debugging, exploratory work

### How to Use

```bash
salloc -q embers ...
#SBATCH -q embers
```

### Tip

Tip: The embers QOS is ideal for exploratory work as you develop, compile, and debug your applications.

---

## Additional Running Job Constraints

Beyond per-job limits, these limits apply to concurrently running jobs:

| Constraint | Limit | Applies To |
|------------|-------|------------|
| Per-charge-account processors | 6000 | inferno only |
| Per-user GPUs | 32 | inferno only |
| Per-charge-account CPU-time | 300,000 CPU-hours | Both inferno and embers |

Jobs violating these limits will be held in the queue until currently running jobs complete.

---

## Quick Comparison

| Feature | Inferno | Embers |
|---------|---------|--------|
| **Cost** | Consumes credits | FREE |
| **Priority** | 250,000 (high) | 0 (low) |
| **Max Walltime** | 3-21 days | 8 hours |
| **Preemption** | No | Yes (after 1 hour) |
| **Max Jobs** | 500 | 50 |
| **Best For** | Production | Development/Testing |
