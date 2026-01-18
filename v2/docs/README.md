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

### Check Status
```bash
cd ~/eic-lab/Fast-dLLM/v2
bash sbatch/run_reuse_experiments.sh --status
```

### Run Missing Experiments
```bash
bash sbatch/run_reuse_experiments.sh           # Run missing only
bash sbatch/run_reuse_experiments.sh --limit 10  # Test mode
```

### Force Re-run
```bash
bash sbatch/run_reuse_experiments.sh --force k2_middle  # Specific config
bash sbatch/run_reuse_experiments.sh --force            # All configs
```

---

## Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `sbatch/run_reuse_experiments.sh` | Smart runner: only runs missing experiments | **Primary workflow** |
| `sbatch/eval_reuse_single.sh` | Single SLURM job for one config | Called by runner |
| `sbatch/eval_pace_script.sh` | Baseline evaluation (multi-task) | Baseline benchmarks |
| `sbatch/eval_script.sh` | Local evaluation (no SLURM) | Interactive testing |
| `setup_tmpdir_venv.sh` | Create fast TMPDIR venv | Interactive sessions only |
| `sbatch/eval_reuse_layers_array.sh` | Array job for layer reuse | **Deprecated** |

---

## Directory Structure

```
docs/
├── STATUS.md           # Current state and experiments
├── operations.md       # How to run experiments
├── concepts/           # Research ideas
├── experiments/        # Experiment results
├── reference/
│   └── pace-cluster.md # PACE cluster reference
└── _templates/         # Document templates
```

---

## Key Optimization Strategies

| Strategy | Model Arg | Description |
|----------|-----------|-------------|
| KV Cache | `use_block_cache=True` | Block-level attention caching |
| Confidence Decoding | `threshold=0.9` | Unmask high-confidence tokens together |
| Layer Reuse | `reuse_k=2` | Skip layers every k-th diffusion step |
| Layer Subset | `layer_subset=middle` | Which 12 layers to apply reuse |

---

## Quick Links

| Resource | Link |
|----------|------|
| Project Status | [STATUS.md](STATUS.md) |
| Operations Guide | [operations.md](operations.md) |
| PACE Reference | [reference/pace-cluster.md](reference/pace-cluster.md) |
| Concepts | [concepts/](concepts/) |
