# PACE Phoenix Informational Commands

## squeue

Use `squeue` to check job status for pending (PD) and running (R) jobs.

**Common options:**
- `-j <job number>` - Show information about specific jobs (comma-separated for multiple)
- `-u <username>` - Show jobs belonging to a specific user (e.g., `-u gburdell3`)
- `-A <charge account>` - See jobs belonging to a specific charge account (e.g., `-A gts-gburdell3`)
- `-p <partition>` - See jobs submitted to a specific partition (e.g., `-p cpu-small`)
- `-q <QOS>` - See jobs submitted to a specific QOS (e.g., `-q inferno`)

**Documentation:** Run `man squeue` or visit the [squeue documentation page](https://slurm.schedmd.com/squeue.html)

---

## sacct

After a job has completed, use `sacct` to find information about it.

**Common options:**
- `-j <job number>` - Find information about specific jobs
- `-u <username>` - See all jobs belonging to a specific user
- `-A <charge account>` - See jobs belonging to a specific charge account
- `-X` - Show information only about the allocation, not steps inside it
- `-S <time>` - List jobs only after a specified time (formats: `YYYY-MM-DD[HH:MM[:SS]]`)
- `-o <fields>` - Specify which columns of data should appear in output
  - Run `squeue --helpformat` to see available fields

**Documentation:** Run `man sacct` or visit the [sacct documentation page](https://slurm.schedmd.com/sacct.html)

---

## scancel

To cancel a job: `scancel <job number>`

**Example:**
```bash
scancel 1440  # Cancel job 1440
```

Use `squeue` to find the job number first.

---

## pace-check-queue

Provides an overview of current utilization of each partition's nodes.

**Usage:**
```bash
pace-check-queue inferno      # For QOS
pace-check-queue cpu-small    # For partition
```

**Options:**
- `-s` - See all features of each node in the partition
- `-c` - Color-code the "Accepting Jobs?" column

**Note:** On Slurm clusters, utilized and allocated local disk (including percent utilization) are not available.

---

## pace-job-summary

Provides high-level overview about jobs processed on the cluster.

**Usage:**
```bash
pace-job-summary <JobID>
```

**Example output:**
```
---------------------------------------
Begin Slurm Job Summary for 2836
Query Executed on 2022-08-17 at 18:21:33
---------------------------------------
Job ID:     2836
User ID:    gburdell3
Account:    gts-gburdell3
Job name:   SlurmPythonExample
Resources:  cpu=4,mem=4G,node=1
Rsrc Used:  cput=00:00:08,vmem=0.8M,walltime=00:00:02,mem=0.0M,energy_used=0
Exit Code:  0:0
Partition:  cpu-small
QOS:        inferno
Nodes:      atl1-1-01-011-4-2
---------------------------------------
```

---

## pace-quota

Find your Phoenix-Slurm charge accounts and storage allocations.

**Usage:**
```bash
pace-quota
```

**What it shows:**
- Job Charge Account Balances
  - **Balance:** Current total based on completed transactions
  - **Reserved:** Sum of liens based on running jobs
  - **Available:** Total funds available for new job submissions
- Storage utilization

**Account naming convention:** Most accounts are of the form `gts-<PI username>[-<descriptor>]`

**Example:**
```
====================================================================================================
                        Job Charge Account Balances
====================================================================================================
Name                           Balance    Reserved   Available
gts-gburdell3-CODA20           291798.90  3329.35    288469.67
gts-gburdell3-phys             241264.01  69.44      241194.66
gts-gburdell3                  41.72      0.00       41.72
```
