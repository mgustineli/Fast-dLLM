# Documentation

Fast-dLLM v2 research project documentation.

> **Research Goal**: Fast-dLLM v2 adapts pretrained autoregressive models (Qwen2.5) into parallel text generation via block diffusion, targeting 2.5x+ speedup with ~1B tokens of fine-tuning.

---

## Quick Start

**New here?** Start with:
1. [STATUS.md](STATUS.md) - Current state and active experiments
2. [operations.md](operations.md) - How to run experiments
3. [concepts/](concepts/) - Research ideas we're exploring

---

## Running Experiments

### Get Help
```bash
cd ~/eic-lab/Fast-dLLM/v2
bash sbatch/run.sh --help    # Show all options, tasks, configs, examples
```

### Check Status
```bash
bash sbatch/run.sh --status
```

### Run Missing Experiments
```bash
bash sbatch/run.sh                        # Run missing only (00_baseline)
bash sbatch/run.sh --limit 10             # Test mode
bash sbatch/run.sh --experiment 01_new    # Different experiment
```

### Force Re-run
```bash
bash sbatch/run.sh --force k2_middle  # Specific config
bash sbatch/run.sh --force            # All configs
```

---

## Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `sbatch/run.sh` | Smart runner: only runs missing experiments | **Primary workflow** |
| `sbatch/baseline.sh` | Baseline evaluation (multi-task) | Baseline benchmarks |
| `sbatch/local.sh` | Local evaluation (no SLURM) | Interactive testing |
| `sbatch/_job.sh` | Single SLURM job for one config | Internal (called by `run.sh`) |
| `sbatch/_array.sh` | Array job for layer reuse | **Deprecated** |
| `setup_tmpdir_venv.sh` | Create fast TMPDIR venv | Interactive sessions only |

---

## Directory Structure

```
v2/
├── generation_functions.py    # Core generation algorithms (active/dev)
├── experiments/               # Versioned generation code per experiment
│   ├── 00_baseline/
│   │   └── generation_functions.py
│   └── 01_adaptive_skip/
│       └── generation_functions.py
├── results/                   # Experiment outputs
│   └── {experiment}/{task}/{config}/
├── artifacts/                 # Git-tracked summaries
│   └── {experiment}/{task}/{config}/summary.json
└── docs/
    ├── STATUS.md              # Current state and experiments
    ├── operations.md          # How to run experiments
    ├── concepts/              # Research ideas
    ├── experiments/           # Experiment documentation
    ├── reference/             # Cluster reference
    └── _templates/            # Document templates
```

---

## Key Optimization Strategies

| Strategy | Model Arg | Description |
|----------|-----------|-------------|
| KV Cache | `use_block_cache=True` | Block-level attention caching |
| Confidence Decoding | `threshold=0.9` | Unmask high-confidence tokens together |
| Layer Reuse | `reuse_k=2` | Skip layers every k-th diffusion step |
| Layer Subset | `layer_subset=middle` | Which 12 layers to apply reuse |
| Experiment Version | `experiment_name=01_new` | Load generation code from `experiments/{name}/` |

---

## Quick Links

| Resource | Link |
|----------|------|
| Project Status | [STATUS.md](STATUS.md) |
| Operations Guide | [operations.md](operations.md) |
| PACE Reference | [reference/pace-cluster.md](reference/pace-cluster.md) |
| Concepts | [concepts/](concepts/) |
