# PACE Phoenix Local Disk Jobs

Every Phoenix node has local disk storage for temporary use, automatically cleared upon job completion.

---

## Why Use Local Disk?

Local disk provides **faster I/O** than network storage (home, project, scratch) for:
- Temporary working files
- Intermediate computation results
- High-frequency read/write operations

---

## Storage Types

| Node Type | Storage Type | Capacity |
|-----------|--------------|----------|
| **Standard nodes** | NVMe | 1 TB |
| **localSAS nodes** | SAS | 8 TB |

---

## Accessing Local Disk

### Using $TMPDIR Variable

Slurm automatically creates a temporary directory for each job.

**In batch script:**
```bash
#!/bin/bash
#SBATCH -J local_disk_example
#SBATCH -A gts-gburdell3
#SBATCH -N1
#SBATCH -t 1:00:00

# Copy data to local disk
cp /home/gburdell3/input_data.txt ${TMPDIR}/
cd ${TMPDIR}

# Process data on local disk
srun ./process_data input_data.txt > results.txt

# Copy results back to network storage
cp results.txt /home/gburdell3/output/
```

**In interactive session:**
```bash
cd ${TMPDIR}
ls -lh
```

The directory is **automatically deleted** when the job ends.

---

## Requesting Local Disk Space

### For Partial Node Requests

Guarantee availability with `--tmp`:

```bash
#SBATCH --tmp=100G          # Request 100GB local disk
```

**Units:** Default is MB if not specified
- `--tmp=50000` = 50,000 MB
- `--tmp=50G` = 50 GB

### Example Batch Script

```bash
#!/bin/bash
#SBATCH -J disk_intensive_job
#SBATCH -A gts-gburdell3
#SBATCH -n 4
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=200G                  # Guarantee 200GB local disk
#SBATCH -t 6:00:00
#SBATCH -q inferno

# Use local disk for temporary files
export TEMP_DIR=${TMPDIR}
srun ./my_analysis --temp-dir=$TEMP_DIR
```

---

## Requesting SAS Storage Nodes

For jobs needing **8 TB local disk**, request localSAS nodes:

```bash
#SBATCH -C localSAS         # Request SAS storage node
```

### Full Example

```bash
#!/bin/bash
#SBATCH -J large_local_storage
#SBATCH -A gts-gburdell3
#SBATCH -N1
#SBATCH -C localSAS                 # 8TB SAS storage
#SBATCH --tmp=5T                    # Request 5TB
#SBATCH -t 12:00:00
#SBATCH -q inferno

# Process large dataset on local disk
cd ${TMPDIR}
cp /scratch/gburdell3/big_dataset.tar.gz .
tar -xzf big_dataset.tar.gz

srun ./process_large_data

# Copy results back
cp -r results/ /scratch/gburdell3/output/
```

---

## Interactive Example

```bash
salloc -A gts-gburdell3 -N1 -C localSAS --tmp=1T -t 4:00:00 -q inferno
```

Once allocated:
```bash
cd ${TMPDIR}
df -h .                    # Check available space
# Do work here
```

---

## Best Practices

### 1. Copy Data In

```bash
# At start of job
cp /home/$USER/input/* ${TMPDIR}/
cd ${TMPDIR}
```

### 2. Process on Local Disk

```bash
# All I/O happens on local disk (fast!)
srun ./heavy_io_program
```

### 3. Copy Results Out

```bash
# Before job ends
cp results/* /home/$USER/output/
```

### 4. Cleanup (Optional)

```bash
# Usually not needed - automatic cleanup on job end
# But can do manual cleanup:
rm -rf ${TMPDIR}/*
```

---

## Common Use Cases

### High-Frequency I/O

```bash
#!/bin/bash
#SBATCH -A gts-gburdell3
#SBATCH --tmp=500G

cd ${TMPDIR}
# Database operations, log files, etc.
srun ./database_operations
```

### Large Temporary Files

```bash
#!/bin/bash
#SBATCH -A gts-gburdell3
#SBATCH -C localSAS
#SBATCH --tmp=3T

cd ${TMPDIR}
# Unpack large archives, intermediate results
tar -xf /scratch/$USER/huge_archive.tar
srun ./process
```

### Checkpointing

```bash
#!/bin/bash
#SBATCH -A gts-gburdell3
#SBATCH --tmp=100G

cd ${TMPDIR}
# Periodic checkpoints to local disk (fast)
# Final checkpoint to network storage (permanent)
srun ./simulation --checkpoint-dir=${TMPDIR}/checkpoints

# Save final state
cp -r ${TMPDIR}/checkpoints /home/$USER/final_checkpoint/
```

---

## Checking Disk Usage

### In Running Job

```bash
df -h ${TMPDIR}            # Check space available
du -sh ${TMPDIR}/*         # Check space used
```

### Before Submitting

```bash
pace-check-queue cpu-small -s     # See node features including disk
```

---

## Performance Comparison

| Storage Type | Typical Speed | Use Case |
|--------------|---------------|----------|
| **Local NVMe** | Very fast | Temporary, high I/O |
| **Local SAS** | Fast | Large temporary files |
| **Home** | Moderate | Permanent, small files |
| **Scratch** | Moderate-Fast | Large permanent files |
| **Project** | Moderate | Shared data |

---

## Important Notes

⚠️ **Data is deleted** when job ends - always copy important results out!

⚠️ **Not for long-term storage** - use /home, /scratch, or /project for permanent data

✅ **Automatic cleanup** - no need to manually delete files

✅ **Private to your job** - other users can't access your $TMPDIR

✅ **Best for I/O intensive jobs** - significant performance improvement

---

## Tips

Tip: Use local disk for:
- Temporary scratch space
- Unpacking large archives
- High-frequency read/write operations
- Database files during computation
- Checkpoint files (with final copy to network storage)

Tip: Don't use local disk for:
- Final results (copy them out first!)
- Input data that doesn't need modification
- Small files with infrequent access

Tip: Plan your disk needs:
- Estimate total space needed
- Add buffer (20-30% extra)
- Request with `--tmp=<size>`

Tip: Monitor usage:
- Check periodically with `df -h ${TMPDIR}`
- Ensure you don't exceed requested space
