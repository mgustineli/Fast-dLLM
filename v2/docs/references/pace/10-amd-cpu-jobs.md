# PACE Phoenix AMD CPU Jobs

Phoenix provides heterogeneous compute resources including AMD CPU nodes.

---

## AMD CPU Specifications

**Processor:** Dual AMD Epyc 7713 CPUs @ 2.0 GHz

**Total Cores:** 128 cores per node (64 cores per CPU × 2 CPUs)

---

## Requesting AMD CPU Nodes

Add the AMD constraint to your job request:

```bash
#SBATCH -C amd
```

Or on command line:

```bash
salloc -C amd ...
```

---

## Interactive AMD Example

```bash
salloc -A gts-gburdell3 -N1 -C amd --ntasks-per-node=64 -t 2:00:00 -q inferno
```

**What this requests:**
- 1 AMD CPU node
- 64 cores (half the node)
- 2 hours
- Inferno QOS (paid)

---

## Batch AMD Example

### Basic CPU-Intensive Job

```bash
#!/bin/bash
#SBATCH -J amd_cpu_job
#SBATCH -A gts-gburdell3
#SBATCH -N 1                        # 1 node
#SBATCH -C amd                      # AMD CPU constraint
#SBATCH --ntasks-per-node=128       # All 128 cores
#SBATCH --mem-per-cpu=2G            # 2GB per core
#SBATCH -t 8:00:00                  # 8 hours
#SBATCH -q inferno
#SBATCH -o amd_output-%j.out

# Load required modules
module load gcc/11.2.0

# Run parallel computation
srun ./my_parallel_program
```

### Multi-Node AMD Job

```bash
#!/bin/bash
#SBATCH -J multi_amd_job
#SBATCH -A gts-gburdell3
#SBATCH -N 4                        # 4 AMD nodes
#SBATCH -C amd
#SBATCH --ntasks-per-node=128       # 128 cores per node (512 total)
#SBATCH --mem-per-cpu=1G
#SBATCH -t 12:00:00
#SBATCH -q inferno
#SBATCH -o output-%j.out

module load gcc mvapich2

# MPI job across 512 cores
srun ./mpi_application
```

---

## AMD with High Memory

Request more memory per core for memory-intensive workloads:

```bash
#!/bin/bash
#SBATCH -J amd_high_mem
#SBATCH -A gts-gburdell3
#SBATCH -N 1
#SBATCH -C amd
#SBATCH --ntasks-per-node=64        # Use half the cores
#SBATCH --mem-per-cpu=8G            # 8GB per core (512GB total)
#SBATCH -t 24:00:00
#SBATCH -q inferno

srun ./memory_intensive_app
```

---

## AMD vs Intel Considerations

### When to Use AMD

✅ **Highly parallel workloads** - 128 cores per node
✅ **Memory bandwidth intensive** - Excellent memory bandwidth
✅ **Multi-threaded applications** - Many threads can run concurrently
✅ **Cost-effective** - Good performance per credit

### Potential Differences

⚠️ **Software compatibility** - Some software optimized for Intel
⚠️ **Performance characteristics** - Different architecture may affect specific workloads
⚠️ **Module availability** - Ensure required modules support AMD

---

## Checking AMD Node Availability

```bash
pace-check-queue cpu-small -s | grep -i amd
```

Or check partition info:

```bash
sinfo -p cpu-small -o "%N %f %c %m"
```

Look for "amd" in features column.

---

## Combining AMD with Other Constraints

### AMD with Local SAS Storage

```bash
#SBATCH -C amd,localSAS
#SBATCH --tmp=2T
```

Not all AMD nodes may have SAS storage - check availability.

---

## Optimization Tips

### 1. Use AMD-Optimized Compilers

```bash
module load aocc/3.2.0          # AMD Optimizing C/C++ Compiler
```

Or use GCC with AMD-specific flags:

```bash
gcc -march=znver3 -mtune=znver3 ...
```

### 2. Thread Pinning

For optimal performance on many-core systems:

```bash
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
srun ./threaded_app
```

### 3. NUMA Awareness

AMD nodes have NUMA (Non-Uniform Memory Access):

```bash
numactl --hardware              # Check NUMA topology
srun --cpu-bind=verbose,cores ./app
```

---

## Example: Parallel Scientific Computing

```bash
#!/bin/bash
#SBATCH -J scientific_amd
#SBATCH -A gts-gburdell3
#SBATCH -N 2                        # 2 AMD nodes
#SBATCH -C amd
#SBATCH --ntasks-per-node=128       # 256 total MPI ranks
#SBATCH --mem-per-cpu=3G
#SBATCH -t 16:00:00
#SBATCH -q inferno
#SBATCH -o science_%j.log

module load gcc/11.2.0 openmpi/4.1.1

# Set thread affinity
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close

# Run with optimal binding
srun --cpu-bind=cores ./simulation input.dat
```

---

## Troubleshooting

### Issue: Job Pending

**Check:** AMD nodes availability
```bash
squeue -p cpu-small | grep amd
pace-check-queue cpu-small
```

**Solution:** AMD nodes may be in high demand, try:
- Reduce node count
- Try `embers` QOS
- Remove AMD constraint if not required

### Issue: Performance Lower Than Expected

**Check:** CPU binding and NUMA settings
```bash
srun --cpu-bind=verbose,cores ./app
```

**Solution:** Optimize thread/process placement

### Issue: Module Not Available

**Check:** Module compatibility
```bash
module spider <module_name>
```

**Solution:** Load AMD-compatible version or compile manually

---

## Resource Calculation Example

**Request:** 2 AMD nodes, 100 cores per node, 10 hours

**CPU-hours consumed:**
```
2 nodes × 100 cores × 10 hours = 2000 CPU-hours
```

Check your account has sufficient balance:
```bash
pace-quota
```

---

## Quick Reference

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Constraint** | `-C amd` | Request AMD CPU node |
| **Cores per node** | Up to 128 | Total available cores |
| **Processors** | 2 × AMD Epyc 7713 | Dual socket |
| **Base frequency** | 2.0 GHz | Clock speed |
| **Architecture** | Zen 3 | Microarchitecture |

---

## Tips

**Use all cores:** Request `--ntasks-per-node=128` to maximize utilization

**Memory planning:** AMD nodes typically have good memory capacity

**Compiler choice:** Consider AMD-optimized compilers (AOCC) for best performance

**Test first:** Run small test jobs to validate performance on AMD vs Intel

**Parallel scaling:** AMD nodes excel at highly parallel workloads

**Check documentation:** Some applications have AMD-specific tuning guides

---

## Best Practices

1. **Benchmark your application** on AMD nodes to understand performance
2. **Use appropriate compiler flags** for AMD architecture
3. **Configure thread/process binding** for NUMA-aware execution
4. **Request appropriate resources** - don't request all 128 cores if you can't use them
5. **Monitor resource usage** with `seff <jobid>` after completion

---

## Additional Resources

- AMD Epyc 7713 specifications
- AMD optimization guides
- Slurm CPU binding documentation
- PACE support for AMD-specific questions
