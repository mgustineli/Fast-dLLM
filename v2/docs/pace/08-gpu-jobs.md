# PACE Phoenix GPU Jobs

Guide for requesting and using GPU resources on Phoenix.

---

## Default GPU Type

📌 **Default:** Nvidia Tesla V100 GPU (if no type specified)

To use other GPUs, you must explicitly specify the type.

---

## Available GPU Types

| GPU Model | Constraint Flag | GRES Flag | Memory | Cores/GPU |
|-----------|----------------|-----------|---------|-----------|
| **V100** (16GB) | `-C V100-16GB` | `--gres=gpu:V100:N` | 16GB | 12 |
| **V100** (32GB) | `-C V100-32GB` | `--gres=gpu:V100:N` | 32GB | 12 |
| **RTX 6000** | `-C RTX6000` or `-C gpu-rtx6000` | `--gres=gpu:RTX_6000:N` | 24GB | 6 |
| **A100** (40GB) | `-C A100-40GB` | `--gres=gpu:A100:N` | 40GB | 8 (up to 32) |
| **A100** (80GB) | `-C A100-80GB` | `--gres=gpu:A100:N` | 80GB | 8 (up to 32) |
| **H100** | `-C H100` or `-C gpu-h100` | `--gres=gpu:H100:N` | 80GB | 8 |
| **H200** | `-C H200` or `-C gpu-h200` | `--gres=gpu:H200:N` | 142GB | 8 |
| **L40S** | `-C L40S` or `-C gpu-l40s` | `--gres=gpu:L40S:N` | 48GB | 4 |
| **RTX Pro Blackwell** | `-C RTX-Pro-Blackwell` | `--gres=gpu:rtx_pro_6000_blackwell:N` | - | - |

**Note:** Some GPU nodes (8 of 10 L40S, 1 A100, 1 H100) are only available on `embers` QOS.

---

## Requesting GPUs: Two Methods

### Method 1: GRES (GPUs per Node)

```bash
--gres=gpu:<type>:<number per node>
```

**Examples:**
```bash
--gres=gpu:V100:2          # 2 V100 GPUs per node
--gres=gpu:RTX_6000:1      # 1 RTX 6000 per node
--gres=gpu:A100:4          # 4 A100 GPUs per node
--gres=gpu:H100:1          # 1 H100 per node
--gres=gpu:L40S:2          # 2 L40S per node
```

### Method 2: Total GPUs with Constraint

```bash
-G <total GPUs> -C <constraint>
```

**Examples:**
```bash
-G 2 -C V100-32GB          # 2 V100 32GB GPUs total
-G 1 -C RTX6000            # 1 RTX 6000
-G 4 -C gpu-a100           # 4 A100 GPUs (any memory)
-G 1 -C HX00               # First available H100 or H200
```

---

## Advanced GPU Allocation

### Per-Node

```bash
--gpus-per-node=<type>:<number>
```

### Per-Socket

```bash
--gpus-per-socket=<type>:<number>
```

### Per-Task

```bash
--gpus-per-task=<type>:<number>
```

See [salloc](https://slurm.schedmd.com/salloc.html) or [sbatch](https://slurm.schedmd.com/sbatch.html) documentation for details.

---

## CPU Cores per GPU

**Automatic assignment** - no need to specify `--ntasks-per-node`:

| GPU Type | Default Cores | Max Cores |
|----------|---------------|-----------|
| RTX 6000 | 6 | 6 (fixed) |
| V100 | 12 | 12 (fixed) |
| A100 | 8 | 32 (configurable) |
| H100 | 8 | 8 (fixed) |
| H200 | 8 | 8 (fixed) |
| L40S | 4 | 4 (fixed) |

**For A100 only:** You can request up to 32 cores per GPU:
```bash
--gres=gpu:A100:1 --ntasks-per-node=32
```

---

## Memory Allocation

**Recommended:** Use `--mem-per-gpu` for GPU jobs:

```bash
--mem-per-gpu=12G          # 12GB per GPU
--mem-per-gpu=48G          # 48GB per GPU
```

---

## Interactive GPU Example

### Request V100 GPU

```bash
salloc -A gts-gburdell3 -N1 --mem-per-gpu=12G -q inferno -t 0:15:00 --gres=gpu:V100:1
```

**Output:**
```
salloc: Pending job allocation 1484
salloc: job 1484 queued and waiting for resources
salloc: Granted job allocation 1484
salloc: Waiting for resource configuration
salloc: Nodes compute-node-1 are ready for job
---------------------------------------
Begin Slurm Prolog: Oct-07-2022 16:10:57
Job ID:    1484
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  interactive
Partition: gpu-v100
QOS:       inferno
---------------------------------------
[gburdell3@compute-node-1 ~]$
```

### Load CUDA and Compile

```bash
module load cuda
mkdir nvcc_example
cd nvcc_example
```

Create `hello_cuda.cu`:

```c
#include <stdio.h>

__global__ void helloCUDA()
{
    printf("Hello, world!\n");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

### Compile and Run

```bash
nvcc hello_cuda.cu -o hello_cuda
srun ./hello_cuda
```

**Output:**
```
Hello, world!
```

---

## Batch GPU Example

### Batch Script (GPU_example.sbatch)

```bash
#!/bin/bash
#SBATCH -J GPU_example                          # Job name
#SBATCH -A gts-gburdell3                        # Charge account
#SBATCH -N1                                     # 1 node
#SBATCH --gres=gpu:V100:1                       # 1 V100 GPU
#SBATCH --mem-per-gpu=12GB                      # 12GB per GPU
#SBATCH -t 1:00:00                              # 1 hour
#SBATCH -o Report-%j.out                        # Output file
#SBATCH --mail-type=BEGIN,END,FAIL              # Email notifications
#SBATCH --mail-user=gburdell3@gatech.edu        # Email address

cd ~/nvcc_example
module load gcc
module load cuda
nvcc hello_cuda.cu -o hello_cuda
srun ./hello_cuda
```

### Submit

```bash
sbatch GPU_example.sbatch
```

### Output File (Report-3210841.out)

```
---------------------------------------
Begin Slurm Prolog: Mar-13-2025 16:29:35
Job ID:    3210841
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  GPU_example
Partition: gpu-v100
QOS:       inferno
---------------------------------------
Hello, world!
---------------------------------------
Begin Slurm Epilog: Mar-13-2025 16:29:39
Job ID:        3210841
User ID:       gburdell3
Account:       gts-gburdell3
Job name:      GPU_example
Resources:     cpu=1,gres/gpu:v100=1,mem=12G,node=1
Rsrc Used:     cput=00:00:04,vmem=0,walltime=00:00:04,mem=184144K,energy_used=0
Partition:     gpu-v100
QOS:           inferno
Nodes:         compute-node-1
---------------------------------------
```

---

## Multiple GPU Options (Flexibility)

When GPU resources are in high demand, use constraint lists to accept first available:

```bash
salloc -A gts-gburdell3 -G 1 -C 'A100|V100|RTX6000'
```

**Output:**
```
salloc: Pending job allocation 1500
salloc: job 1500 queued and waiting for resources
salloc: Granted job allocation 1500
---------------------------------------
Begin Slurm Prolog: Feb-13-2024 12:00:04
Job ID:    1500
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  interactive
Partition: gpu-rtx6000,gpu-a100,gpu-v100
QOS:       inferno
---------------------------------------
[gburdell3@compute-node-1 ~]$
```

**Note:** The partition shown in the Prologue lists all options provided, not the actual assigned GPU.

### Check Actual GPU Assigned

```bash
scontrol show job $SLURM_JOB_ID | grep Partition
```

**Output:**
```
Partition=gpu-rtx6000 AllocNode:Sid=login-phoenix-slurm-1:163236
```

You're charged at the rate for the actual partition allocated (gpu-rtx6000 in this case).

---

## Common GPU Request Examples

### RTX 6000 for 2 Hours

```bash
salloc -A gts-gburdell3 -N1 --gres=gpu:RTX_6000:1 --mem-per-gpu=20G -q inferno -t 2:00:00
```

### A100 with 32 Cores

```bash
salloc -A gts-gburdell3 -N1 --gres=gpu:A100:1 --ntasks-per-node=32 --mem-per-gpu=48G -q inferno -t 4:00:00
```

### Multiple H100 GPUs

```bash
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-gpu=40G
```

### Free L40S on Embers

```bash
salloc -A gts-gburdell3 -N1 --gres=gpu:L40S:1 --mem-per-gpu=24G -q embers -t 4:00:00
```

---

## Constraint Options Summary

**Generic (any memory):**
- `-C gpu-v100` - V100 (16GB or 32GB)
- `-C gpu-rtx6000` - RTX 6000
- `-C gpu-a100` - A100 (40GB or 80GB)
- `-C gpu-h100` - H100
- `-C gpu-h200` - H200
- `-C HX00` - First available H100 or H200
- `-C gpu-l40s` - L40S

**Specific memory:**
- `-C V100-16GB` - V100 with 16GB
- `-C V100-32GB` - V100 with 32GB
- `-C A100-40GB` - A100 with 40GB
- `-C A100-80GB` - A100 with 80GB

---

## Tips

**Memory:** Always specify `--mem-per-gpu` for GPU jobs

**Cores:** No need to specify `--ntasks-per-node` (except A100 if >8 cores needed)

**Testing:** Use `embers` QOS for development/testing

**Modules:** Load required CUDA modules: `module load cuda`

**Verification:** Check assigned GPU with `nvidia-smi` on the compute node

**Flexibility:** Use constraint lists (`-C 'A100|V100|RTX6000'`) when flexible on GPU type

**Accounting:** GPU jobs are charged by GPU-hours, not CPU-hours

---

## Checking GPU Availability

```bash
pace-check-queue gpu-v100
pace-check-queue gpu-a100
pace-check-queue gpu-h100
```

Add `-s` to see all node features.

---

## Common Issues

**Problem:** Job pending forever
**Solution:** Check if requested GPU type is available, or use flexible constraints

**Problem:** Out of memory errors
**Solution:** Increase `--mem-per-gpu` value

**Problem:** Wrong GPU assigned
**Solution:** Be specific with constraint flags (e.g., `-C A100-80GB` vs `-C gpu-a100`)
