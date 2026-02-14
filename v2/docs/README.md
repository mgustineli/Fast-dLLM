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

Each experiment is self-contained under `experiments/`:

### Check Status
```bash
cd ~/eic-lab/Fast-dLLM/v2
bash experiments/00_baseline/sbatch/run.sh --status
```

### Run Missing Experiments
```bash
bash experiments/00_baseline/sbatch/run.sh                        # Run missing only
bash experiments/00_baseline/sbatch/run.sh --limit 10             # Test mode
bash experiments/00_baseline/sbatch/run.sh --config k1_first      # Single config only
```

### Create a New Experiment
```bash
cp -r experiments/_template experiments/01_new_experiment
# Edit files in the new directory, then:
bash experiments/01_new_experiment/sbatch/run.sh --dry-run
```

---

## Directory Structure

```
v2/
├── eval.py                       # Shared evaluation harness
├── generation_functions.py       # Default generation code (chatbot/app)
├── experiments/                  # Self-contained experiments
│   ├── _template/                # Scaffold for new experiments
│   │   ├── README.md, proposal.md, results.md, tasks.md
│   │   ├── config.yaml
│   │   ├── generation_functions.py
│   │   └── sbatch/              # run.sh + _job.sh
│   └── 00_baseline/             # Layer reuse baseline
│       ├── README.md, proposal.md, results.md, tasks.md
│       ├── config.yaml
│       ├── generation_functions.py
│       ├── sbatch/              # run.sh + _job.sh
│       ├── artifacts/           # Git-tracked summary.json files
│       ├── results/             # Full results (gitignored)
│       └── logs/                # SLURM logs (gitignored)
├── sbatch/                      # Shared utilities
│   ├── common.sh                # Venv/env setup functions
│   ├── local.sh                 # Local evaluation (no SLURM)
│   └── *.sh                     # Deprecated (reference only)
└── docs/
    ├── STATUS.md                # Project status + priority stack
    ├── README.md                # This file
    ├── EXPERIMENTS_STATUS.md    # Experiment tracking
    ├── operations.md            # How to run experiments
    ├── concepts/                # Research ideas
    ├── experiments/             # Cross-cutting experiment docs
    ├── references/              # Literature + PACE cluster docs
    └── _templates/              # Document templates
```

---

## Key Optimization Strategies

| Strategy | Model Arg | Description |
|----------|-----------|-------------|
| KV Cache | `use_block_cache=True` | Block-level attention caching |
| Confidence Decoding | `threshold=0.9` | Unmask high-confidence tokens together |
| Layer Reuse | `reuse_k=2` | Skip layers every k-th diffusion step |
| Layer Subset | `layer_subset=middle` | Which 12 layers to apply reuse |
| Experiment Version | `experiment_name=00_baseline` | Load generation code from `experiments/{name}/` |

---

## Quick Links

| Resource | Link |
|----------|------|
| Project Status | [STATUS.md](STATUS.md) |
| Experiment Tracking | [EXPERIMENTS_STATUS.md](EXPERIMENTS_STATUS.md) |
| Operations Guide | [operations.md](operations.md) |
| PACE Reference | [references/pace-cluster.md](references/pace-cluster.md) |
| Concepts | [concepts/](concepts/) |
| 00_baseline | [experiments/00_baseline/](../experiments/00_baseline/) |
