# PACE Phoenix MPI Jobs

Message Passing Interface (MPI) jobs utilize parallel computing across multiple processors/nodes.

---

## Important Warning

**DO NOT use `mpirun` or `mpiexec` with Slurm. Use `srun` instead.**

---

## Setup Example

Download MPI "hello world" code from [MPI Tutorial](https://mpitutorial.com):

```bash
mkdir slurm_mpi_example
cd slurm_mpi_example
wget https://raw.githubusercontent.com/mpitutorial/mpitutorial/gh-pages/tutorials/mpi-hello-world/code/mpi_hello_world.c
```

---

## Interactive MPI Example

### Step 1: Allocate Resources

Request 2 nodes with 4 cores each:

```bash
salloc -A gts-gburdell3 -q inferno -N2 --ntasks-per-node=4 -t1:00:00
```

### Step 2: Wait for Allocation

```
salloc: Pending job allocation 1471
salloc: job 1471 queued and waiting for resources
salloc: Granted job allocation 1471
salloc: Waiting for resource configuration
salloc: Nodes atl1-1-02-007-30-2,atl1-1-02-018-24-2 are ready for job
---------------------------------------
Begin Slurm Prolog: Oct-07-2022 16:10:09
Job ID:    1471
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  interactive
Partition: cpu-small
QOS:       inferno
---------------------------------------
[gburdell3@atl1-1-02-007-30-2 ~]$
```

### Step 3: Load Modules and Compile

```bash
cd slurm_mpi_example
module load gcc/10.3.0 mvapich2/2.3.6
mpicc mpi_hello_world.c -o mpi_hello_world
```

### Step 4: Run MPI Job

```bash
srun mpi_hello_world
```

### Expected Output

```
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 0 out of 8 processors
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 2 out of 8 processors
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 3 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 4 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 7 out of 8 processors
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 1 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 5 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 6 out of 8 processors
```

8 total processors (2 nodes × 4 cores each)

---

## Batch MPI Example

### Batch Script (SlurmBatchMPIExample.sbatch)

```bash
#!/bin/bash
#SBATCH -J SlurmBatchMPIExample                 # Job name
#SBATCH --account=gts-gburdell3                 # Charge account
#SBATCH -N2 --ntasks-per-node=4                 # 2 nodes, 4 cores each
#SBATCH --mem-per-cpu=1G                        # 1GB per core
#SBATCH -t 1:00:00                              # 1 hour
#SBATCH -q inferno                              # QOS
#SBATCH -o Report-%j.out                        # Output file
#SBATCH --mail-type=BEGIN,END,FAIL              # Email notifications
#SBATCH --mail-user=gburdell3@gatech.edu        # Email address

cd $HOME/slurm_mpi_example                      # Change to working directory

# Compile MPI Code
module load gcc/10.3.0 mvapich2/2.3.6
mpicc mpi_hello_world.c -o mpi_hello_world

# Run MPI Code
srun mpi_hello_world
```

### Submit the Job

```bash
cd slurm_mpi_example
sbatch SlurmBatchMPIExample.sbatch
```

**Output:**
```
Submitted batch job 1473
```

### Check Results

After completion, view `Report-<job id>.out`:

```
---------------------------------------
Begin Slurm Prolog: Oct-07-2022 16:10:09
Job ID:    1473
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  SlurmBatchMPIExample
Partition: cpu-small
QOS:       inferno
---------------------------------------
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 1 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 4 out of 8 processors
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 2 out of 8 processors
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 3 out of 8 processors
Hello world from processor atl1-1-02-007-30-2.pace.gatech.edu, rank 0 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 5 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 7 out of 8 processors
Hello world from processor atl1-1-02-018-24-2.pace.gatech.edu, rank 6 out of 8 processors
---------------------------------------
Begin Slurm Epilog: Oct-07-2022 16:10:11
Job ID:        1473
Array Job ID:  _4294967294
User ID:       gburdell3
Account:       gts-gburdell3
Job name:      SlurmBatchMPIExample
Resources:     cpu=8,mem=8G,node=2
Rsrc Used:     cput=00:00:16,vmem=1104K,walltime=00:00:02,mem=0,energy_used=0
Partition:     cpu-small
QOS:           inferno
Nodes:         atl1-1-02-007-30-2,atl1-1-02-018-24-2
---------------------------------------
```

---

## Common MPI Modules

Different MPI implementations are available. Load the appropriate one:

```bash
module load gcc/10.3.0 mvapich2/2.3.6    # MVAPICH2
module load gcc/10.3.0 openmpi/4.1.1     # OpenMPI (example)
```

Use `module avail` to see available MPI modules.

---

## Key Points

✅ **Always use `srun`** to run MPI programs (not mpirun/mpiexec)

✅ **Specify nodes and cores** appropriately:
- `-N` = number of nodes
- `--ntasks-per-node` = cores per node
- Total MPI ranks = nodes × cores per node

✅ **Load required modules** before compilation and execution

✅ **Compile on compute nodes** or in the batch script itself

---

## Resource Calculation

For MPI jobs:
- **Total MPI ranks** = Number of nodes × Tasks per node
- **Example:** `-N2 --ntasks-per-node=4` = 8 MPI ranks

---

## Tips

Tip: Test with small node counts first using `embers` QOS

Tip: Use `--ntasks-per-node` to control MPI ranks per node

Tip: For CPU-bound MPI jobs, request enough memory: `--mem-per-cpu=2G`

Tip: Check job output for correct number of MPI ranks

Tip: Use `scontrol show job $SLURM_JOB_ID` to verify allocated resources
