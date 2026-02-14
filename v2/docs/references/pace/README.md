# PACE Phoenix Slurm Documentation

Quick reference guide for using Slurm on the PACE Phoenix cluster.

---

## Table of Contents

### Getting Started

1. **[Informational Commands](01-informational-commands.md)**
   - `squeue`, `sacct`, `scancel`
   - `pace-check-queue`, `pace-job-summary`, `pace-quota`
   - Monitoring and managing jobs

2. **[Job Accounting](02-job-accounting.md)**
   - Understanding the accounting system
   - Account types and balances
   - Charge rates and billing

3. **[QOS Policies](03-qos-policies.md)**
   - **Inferno** (paid, high priority)
   - **Embers** (free, backfill)
   - Limits and best practices

### Job Types

4. **[Interactive Jobs](04-interactive-jobs.md)**
   - Using `salloc`
   - CPU and GPU interactive sessions
   - X11 forwarding

5. **[Batch Jobs](05-batch-jobs.md)**
   - Writing batch scripts
   - Using `sbatch`
   - Resource requests
   - Python example

6. **[MPI Jobs](06-mpi-jobs.md)**
   - Interactive and batch MPI
   - Using `srun` (not mpirun!)
   - Multi-node parallel computing

7. **[Array Jobs](07-array-jobs.md)**
   - Running multiple similar jobs
   - Throttling and task management
   - Handling many short tasks

### Specialized Resources

8. **[GPU Jobs](08-gpu-jobs.md)**
   - Available GPU types (V100, RTX6000, A100, H100, H200, L40S)
   - GPU request methods
   - CUDA examples
   - Multi-GPU options

9. **[Local Disk Jobs](09-local-disk-jobs.md)**
   - Using `$TMPDIR`
   - NVMe vs SAS storage
   - High I/O workloads

10. **[AMD CPU Jobs](10-amd-cpu-jobs.md)**
    - 128-core AMD Epyc nodes
    - Optimization tips
    - NUMA awareness

---

## Quick Start Examples

### Simple CPU Job (Free)

```bash
salloc -A gts-gburdell3 -q embers -N1 --ntasks-per-node=4 -t 1:00:00
```

### GPU Job (Paid)

```bash
salloc -A gts-gburdell3 -q inferno -N1 --gres=gpu:V100:1 --mem-per-gpu=12G -t 2:00:00
```

### Batch Script Template

```bash
#!/bin/bash
#SBATCH -J my_job
#SBATCH -A gts-gburdell3
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2G
#SBATCH -t 1:00:00
#SBATCH -q embers
#SBATCH -o output-%j.log

module load anaconda3
srun python my_script.py
```

---

## Common Commands Cheat Sheet

| Task | Command |
|------|---------|
| **Check your jobs** | `squeue -u $USER` |
| **Check account balance** | `pace-quota` |
| **Check queue status** | `pace-check-queue inferno` |
| **Submit batch job** | `sbatch script.sbatch` |
| **Cancel job** | `scancel <jobid>` |
| **Job details** | `scontrol show job <jobid>` |
| **Past job info** | `sacct -j <jobid>` |
| **Job summary** | `pace-job-summary <jobid>` |

---

## Quick Decision Guide

### Which QOS Should I Use?

| Scenario | QOS | Why |
|----------|-----|-----|
| Development/testing | **embers** | Free, 8hr limit |
| Production work | **inferno** | No preemption, 3-21 days |
| Quick debugging | **embers** | Free tier |
| Critical deadline | **inferno** | Higher priority |

### Which Partition Should I Use?

| Workload | Partition | Constraint |
|----------|-----------|------------|
| General CPU | `cpu-small` or `cpu-gnr` | None |
| High-memory CPU | `cpu-small` | None, request more mem |
| AMD CPU | `cpu-small` | `-C amd` |
| V100 GPU | `gpu-v100` | `--gres=gpu:V100:1` |
| RTX 6000 GPU | `gpu-rtx6000` | `--gres=gpu:RTX_6000:1` |
| A100 GPU | `gpu-a100` | `--gres=gpu:A100:1` |
| Large local disk | Any | `-C localSAS` |

---

## Best Practices

### Resource Requests

✅ **Request only what you need** - Don't request entire nodes if you only need a few cores

✅ **Estimate walltime accurately** - Jobs finish early release resources; too short kills job

✅ **Use appropriate memory** - Monitor actual usage with `seff <jobid>` after completion

✅ **Choose right QOS** - Use embers for testing, inferno for production

### Job Submission

✅ **Test with small jobs first** - Validate your script works before large-scale runs

✅ **Use array jobs wisely** - Batch short tasks together, throttle large arrays

✅ **Monitor your jobs** - Check status with `squeue`, review output files

✅ **Clean up after yourself** - Cancel unnecessary jobs, check for zombie processes

### Performance

✅ **Use `srun`** - Always prefix compute commands with `srun` in batch scripts

✅ **Use local disk** - For I/O intensive work, use `$TMPDIR`

✅ **Optimize placement** - Use CPU binding for multi-core/GPU jobs

✅ **Load modules** - Ensure all dependencies are loaded before execution

---

## Getting Help

### Check Job Status

```bash
# Why is my job pending?
scontrol show job <jobid>

# What resources did I use?
pace-job-summary <jobid>
seff <jobid>

# Check queue availability
pace-check-queue inferno
pace-check-queue embers
```

### Common Issues

| Problem | Check | Solution |
|---------|-------|----------|
| Job pending forever | Queue status | Try different partition/QOS |
| Job fails immediately | Output file | Check error messages |
| Out of memory | Resource usage | Increase `--mem-per-cpu` |
| Out of time | Walltime | Increase `-t` value |
| No credits | Account balance | `pace-quota`, contact PI |

### Contact Support

- **PACE Documentation:** https://docs.pace.gatech.edu
- **Support Email:** pace-support@oit.gatech.edu
- **Office Hours:** Check PACE website

---

## Notes

- All examples use `gts-gburdell3` as the account - **replace with your actual account**
- Use `pace-quota` to find your account names
- Job scripts should be plain text files (not Word documents!)
- Always test new scripts with short walltimes first
- Monitor your account balance to avoid job rejections

---

## Last Updated

This documentation is based on PACE Phoenix Slurm configuration as of January 2026.

For the most current information, always consult the official PACE documentation at https://docs.pace.gatech.edu

---

## Document Structure

```
docs/
├── README.md (this file)
├── 01-informational-commands.md
├── 02-job-accounting.md
├── 03-qos-policies.md
├── 04-interactive-jobs.md
├── 05-batch-jobs.md
├── 06-mpi-jobs.md
├── 07-array-jobs.md
├── 08-gpu-jobs.md
├── 09-local-disk-jobs.md
└── 10-amd-cpu-jobs.md
```

Each file is self-contained and can be referenced independently.
