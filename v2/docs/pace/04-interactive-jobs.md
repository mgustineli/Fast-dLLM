# PACE Phoenix Interactive Jobs

Interactive jobs reserve resources on compute nodes for interactive use.

---

## Using salloc

**Recommended method** for allocating interactive resources.

### Required Parameters

- `--account` or `-A` - Charge account
- `--qos` or `-q` - Quality of Service (inferno or embers)

### Common Optional Parameters

- `--nodes` or `-N` - Number of nodes
- `--ntasks-per-node` - Cores per node
- `-n` - Total cores
- `--time` or `-t` - Wall time (format: `D-HH:MM:SS`)
- `--mem-per-cpu` - Memory per core
- `--partition` or `-p` - Partition (e.g., `cpu-gnr`, `cpu-small`)
- `--x11` - Enable X forwarding for GUI applications

**Documentation:** Run `man salloc` or visit [salloc documentation](https://slurm.schedmd.com/salloc.html)

---

## Basic Example

Request 1 node with 4 cores for 1 hour:

```bash
salloc -A gts-gburdell3 -q inferno -N1 --ntasks-per-node=4 -t1:00:00
```

### Expected Output

```
salloc: Pending job allocation 1464
salloc: job 1464 queued and waiting for resources
```

Once resources are available:

```
salloc: Granted job allocation 1464
salloc: Waiting for resource configuration
salloc: Nodes atl1-1-02-007-30-2 are ready for job
---------------------------------------
Begin Slurm Prolog: Oct-07-2022 16:10:49
Job ID:    1464
User ID:   gburdell3
Account:   gts-gburdell3
Job name:  interactive
Partition: cpu-small
QOS:       inferno
---------------------------------------
[gburdell3@atl1-1-02-007-30-2 ~]$
```

You're now logged into the compute node!

---

## Using Your Allocation

### Run Commands Interactively

Use `srun` to run commands on your allocated resources:

```bash
srun hostname
```

**Output (for 4 cores):**
```
atl1-1-02-007-30-2.pace.gatech.edu
atl1-1-02-007-30-2.pace.gatech.edu
atl1-1-02-007-30-2.pace.gatech.edu
atl1-1-02-007-30-2.pace.gatech.edu
```

The command runs once per core requested.

---

## Exiting an Interactive Job

Two ways to exit:

1. **Wait for time to expire** - The session ends automatically after the walltime
2. **Exit manually** - Type `exit` or press `Ctrl-D`

```bash
exit
```

**Output:**
```
exit
salloc: Relinquishing job allocation 1464
salloc: Job allocation 1464 has been revoked.
```

---

## CPU Partition Example

Request a CPU node from the `cpu-gnr` partition:

```bash
salloc -A gts-gburdell3 -q embers -p cpu-gnr -N1 --ntasks-per-node=4 -t1:00:00
```

---

## GPU Interactive Example

Request 1 node with an Nvidia Tesla V100 GPU:

```bash
salloc -A gts-gburdell3 -N1 --mem-per-gpu=12G -q inferno -t0:15:00 --gres=gpu:V100:1
```

**Note:** You don't need to specify `--ntasks-per-node` for GPUs because cores are assigned automatically:
- 6 cores per RTX6000
- 12 cores per V100
- 8 cores per A100, H100, or H200
- 4 cores per L40S

See [GPU Jobs documentation](06-gpu-jobs.md) for more details.

---

## X11 Forwarding

For GUI applications, add the `--x11` flag:

```bash
salloc -A gts-gburdell3 -q inferno -N1 --ntasks-per-node=4 -t1:00:00 --x11
```

---

## Tips

**Best Practices:**
- Use `embers` QOS for development and testing (it's free!)
- Use `inferno` QOS for production work
- Always specify an appropriate walltime to avoid wasting resources
- Remember to `exit` when done to free up resources for others

**Common Partitions:**
- `cpu-gnr` - CPU nodes with Granite Rapids processors
- `cpu-small` - Standard CPU nodes
- `gpu-v100` - V100 GPU nodes
- `gpu-rtx6000` - RTX 6000 GPU nodes
- `gpu-a100` - A100 GPU nodes
