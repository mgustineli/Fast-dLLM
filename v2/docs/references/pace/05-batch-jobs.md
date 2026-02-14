# PACE Phoenix Batch Jobs

Batch jobs are scripts submitted to Slurm for automated execution.

---

## Creating a Slurm Batch Script

Write a plain text file with Slurm directives and commands, then submit with `sbatch`.

### Required Elements

1. **Shebang:** Start with `#!/bin/bash`
2. **Charge account:** `#SBATCH -A <account>`

### Common Directives

```bash
#!/bin/bash
#SBATCH -J <job name>                           # Job name
#SBATCH --account=<account>                     # Charge account
#SBATCH -N <nodes>                              # Number of nodes
#SBATCH --ntasks-per-node=<cores>               # Cores per node
#SBATCH -n <total cores>                        # Total cores (alternative to -N)
#SBATCH --mem-per-cpu=<size>                    # Memory per core (e.g., 1G)
#SBATCH --mem-per-gpu=<size>                    # Memory per GPU (for GPU jobs)
#SBATCH -t <time>                               # Walltime (D-HH:MM:SS or minutes)
#SBATCH -q <qos>                                # QOS (inferno or embers)
#SBATCH -p <partition>                          # Partition (e.g., cpu-gnr)
#SBATCH -o <filename>                           # Output file
#SBATCH --mail-type=BEGIN,END,FAIL              # Email notifications
#SBATCH --mail-user=<email>                     # Email address
```

---

## Resource Requests

### CPU Cores

**Option 1:** Specify total cores
```bash
#SBATCH -n 4              # 4 total cores
```

**Option 2:** Specify nodes and cores per node
```bash
#SBATCH -N 2              # 2 nodes
#SBATCH --ntasks-per-node=4   # 4 cores per node (8 total)
```

### Memory

**For CPU jobs:**
```bash
#SBATCH --mem-per-cpu=1G  # 1GB per core
#SBATCH --mem=0           # All memory on node
```

**For GPU jobs:**
```bash
#SBATCH --mem-per-gpu=12G # 12GB per GPU
```

### Walltime

```bash
#SBATCH -t 1:30:00        # 1 hour 30 minutes
#SBATCH -t 2-00:00:00     # 2 days
#SBATCH -t 15             # 15 minutes (integer = minutes)
```

---

## Basic Python Example

### Python Script (test.py)

```python
#simple test script
result = 2 ** 2
print("Result of 2 ^ 2: {}".format(result))
```

### Batch Script (SlurmPythonExample.sbatch)

```bash
#!/bin/bash
#SBATCH -J SlurmPythonExample                   # Job name
#SBATCH --account=gts-gburdell3                 # Charge account
#SBATCH -N1 --ntasks-per-node=4                 # 1 node, 4 cores
#SBATCH --mem-per-cpu=1G                        # 1GB per core
#SBATCH -t 15                                   # 15 minutes
#SBATCH -q inferno                              # QOS
#SBATCH -o Report-%j.out                        # Output file (%j = job ID)
#SBATCH --mail-type=BEGIN,END,FAIL              # Email preferences
#SBATCH --mail-user=gburdell3@gatech.edu        # Email address

module load anaconda3/2022.05                   # Load modules
srun python test.py                             # Run command
```

### Submitting the Job

```bash
sbatch SlurmPythonExample.sbatch
```

### Checking Status

```bash
squeue -u gburdell3      # Check your jobs
scancel <jobid>          # Cancel a job
```

---

## Example Output File

After completion, `Report-<jobid>.out` contains:

```
---------------------------------------
Begin Slurm Prolog: Oct-07-2022 16:10:04
Job ID:    1470
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  SlurmPythonExample
Partition: cpu-small
QOS:       inferno
---------------------------------------
Result of 2 ^ 2: 4
Result of 2 ^ 2: 4
Result of 2 ^ 2: 4
Result of 2 ^ 2: 4
---------------------------------------
Begin Slurm Epilog: Oct-07-2022 16:10:06
Job ID:        1470
Array Job ID:  _4294967294
User ID:       gburdell3
Account:       gts-gburdell3
Job name:      SlurmPythonExample
Resources:     cpu=4,mem=4G,node=1
Rsrc Used:     cput=00:00:12,vmem=8K,walltime=00:00:03,mem=0,energy_used=0
Partition:     cpu-small
QOS:           inferno
Nodes:         atl1-1-02-007-30-2
---------------------------------------
```

**Note:** The command runs 4 times (once per core) because we used `srun`.

---

## CPU Partition Example

```bash
#!/bin/bash
#SBATCH -J CPUExample
#SBATCH --account=gts-gburdell3
#SBATCH -p cpu-gnr                              # CPU partition
#SBATCH -N1 --ntasks-per-node=8
#SBATCH --mem-per-cpu=2G
#SBATCH -t 2:00:00
#SBATCH -q embers                               # Free tier
#SBATCH -o output-%j.log

srun ./my_program
```

---

## Important Notes

### Using srun

⚠️ **Prefix computationally-intensive commands with `srun`** for best performance:

```bash
srun python script.py     # Good
python script.py          # Runs once only
```

### File Locations

- Submit jobs from the directory containing your scripts
- `$SLURM_SUBMIT_DIR` contains the submission directory path

### Default Values

- **Cores:** 1 core (if not specified)
- **Memory:** 1GB per core (if not specified)
- **Walltime:** 1 hour (if not specified)
- **QOS:** inferno (if not specified)

---

## Tips

Tip: Use `pace-quota` to find your charge account names

Tip: Keep `test.py` and `SlurmPythonExample.sbatch` in the same directory

Tip: Check job status with: `squeue -u $USER`

Tip: View completed job info with: `sacct -j <jobid>`

Tip: Use `pace-job-summary <jobid>` for detailed job reports

---

## Additional Resources

- **Man pages:** `man sbatch`
- **Online docs:** [sbatch documentation](https://slurm.schedmd.com/sbatch.html)
- **Help format:** `squeue --helpformat` (see available output fields)
