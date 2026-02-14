# PACE Phoenix Array Jobs

Submit multiple identical jobs without external scripting using Slurm's array job feature.

---

## Job Limit

**Maximum: 500 jobs (queued + running) per user on Phoenix**

---

## Basic Array Job Syntax

### In Batch Script

```bash
#SBATCH --array=x-y         # Range from x to y
```

### On Command Line

```bash
sbatch --array=x-y job_script.sbatch
```

Command line options override script options.

---

## Array Specifications

### Range

```bash
#SBATCH --array=1-100       # Tasks 1 through 100
```

### Comma-Separated List

```bash
#SBATCH --array=4,8,15,16,23,42    # Specific tasks only
```

Useful for rerunning failed tasks from a previous array.

### Throttling (Limiting Concurrent Tasks)

```bash
#SBATCH --array=1-200%5     # 200 tasks, max 5 running at once
```

The `%N` suffix limits active tasks to N at any time.

**Note:** The `%` specifies the actual number of concurrent tasks, not a percentage.

---

## Modifying Running Arrays

Use `scontrol` to change throttling on active jobs:

```bash
scontrol update ArrayTaskThrottle=50 JobId=123456
scontrol update ArrayTaskThrottle=0 JobId=123456    # Remove limit
```

**Important:** Reducing throttle won't affect already-running tasks, only prevents new ones from starting.

---

## Output and Error Files

Use `%A` (master job ID) and `%a` (task ID) in filenames:

```bash
#SBATCH --output=Array_test.%A_%a.out
#SBATCH --error=Array_test.%A_%a.error
```

Or combine into one file:

```bash
#SBATCH --output=Array_test.%A_%a.log
```

**⚠️ Critical:** Always use both `%A` and `%a` in log file names. Using only `%A` causes all tasks to write to a single file, drastically reducing performance.

---

## Using the Array Task ID

Slurm provides `$SLURM_ARRAY_TASK_ID` to each task.

### Example 1: Numbered Files

If files are numbered (file1.txt, file2.txt, etc.):

```bash
srun process_file.py file${SLURM_ARRAY_TASK_ID}.txt
```

### Example 2: List of Files

Process all `.txt` files in a directory:

```bash
file=$(ls *.txt | sed -n ${SLURM_ARRAY_TASK_ID}p)
srun myscript -in $file
```

### Example 3: Input List File

Use a file (`input.list`) with one sample per line:

```bash
SAMPLE_LIST=($(<input.list))
SAMPLE=${SAMPLE_LIST[${SLURM_ARRAY_TASK_ID}]}
srun process.py $SAMPLE
```

---

## Handling Many Short Tasks

**Problem:** If each task is very short (seconds/minutes), array jobs bog down the scheduler.

**Solution:** Combine array jobs with bash loops to batch short tasks.

### Example: 500 Tasks, 30 Seconds Each

Instead of 500 array tasks, use 5 array tasks that each run 100 iterations.

### Batch Script (SlurmArrayExample.sbatch)

```bash
#!/bin/bash
#SBATCH --job-name=SlurmArrayExample
#SBATCH --account=gts-gburdell3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gburdell3@gatech.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1gb
#SBATCH --time=00:10:00
#SBATCH --output=Report_%A-%a.out
#SBATCH --array=1-5                             # Only 5 tasks

# Number of runs per task
PER_TASK=100

# Calculate range for this task
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

echo "This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM"

# Loop through runs for this task
for (( run=$START_NUM; run<=END_NUM; run++ )); do
    echo "This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run"
    srun python test.py
done

date
```

### Submit and Check

```bash
cd slurm_array_example
sbatch SlurmArrayExample.sbatch
```

**Output:** `Submitted batch job 1479`

### Example Output (Report_1479-1.out)

```
---------------------------------------
Begin Slurm Prolog: Oct-07-2022 16:10:12
Job ID:    1480
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  SlurmArrayExample
Partition: cpu-small
QOS:       inferno
---------------------------------------
This is task 1, which will do runs 1 to 100
Result of 2 ^ 2: 4
This is SLURM task 1, run number 2
Result of 2 ^ 2: 4
This is SLURM task 1, run number 3
Result of 2 ^ 2: 4
...
---------------------------------------
Begin Slurm Epilog: Oct-07-2022 16:10:46
Job ID:        1480
Array Job ID:  1479_1
User ID:       gburdell3
Account:       gts-gburdell3
Job name:      SlurmArrayExample
Resources:     cpu=1,mem=1G,node=1
Rsrc Used:     cput=00:00:33,vmem=4512K,walltime=00:00:33,mem=4500K,energy_used=0
Partition:     cpu-small
QOS:           inferno
Nodes:         atl1-1-02-007-30-2
---------------------------------------
```

---

## Deleting Array Jobs

### Delete All Tasks

```bash
scancel 123456
```

### Delete Single Task

```bash
scancel 123456_1    # Cancel task 1 of job 123456
```

---

## Job ID Formats

⚠️ **Important:** Each array task has a unique job ID.

Both formats are valid when querying:
- **Job ID:** `1480` (shown in epilog)
- **Array Job ID:** `1479_1` (master job ID + task ID)

Use either format with `sacct` or `pace-job-summary`.

---

## Tips

**Efficiency:** Batch short tasks into loops rather than creating one array task per run

**Testing:** Start with a small array (`--array=1-3`) to test your script

**Throttling:** Always use `%N` for large arrays to avoid overwhelming the scheduler

**Logging:** Use `%A_%a` in output filenames to separate task outputs

**Rerunning:** Use comma-separated lists to rerun specific failed tasks

**Monitoring:** Check all tasks with `squeue -u $USER`

---

## Common Patterns

### Process All Files in Directory

```bash
#SBATCH --array=1-100

FILES=($(ls data/*.txt))
FILE=${FILES[$SLURM_ARRAY_TASK_ID-1]}  # Arrays are 0-indexed
srun process.sh $FILE
```

### Parameter Sweep

```bash
#SBATCH --array=1-10

PARAM=$(echo "scale=2; $SLURM_ARRAY_TASK_ID * 0.1" | bc)
srun simulation --parameter=$PARAM
```

### Multiple Input Files

Create `input_files.txt` with one filename per line:

```bash
#SBATCH --array=1-50

FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" input_files.txt)
srun analyze.py $FILE
```
