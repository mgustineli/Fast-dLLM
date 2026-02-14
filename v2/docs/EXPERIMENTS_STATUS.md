# Experiments Status

**Last Updated:** 2026-02-14

Track ongoing and planned experiments to maintain context across sessions.

---

## Experiment Structure

Each experiment is a self-contained directory under `experiments/`:

```
experiments/
├── _template/              # Scaffold for new experiments
└── 00_baseline/            # Layer reuse baseline
    ├── README.md           # Quick start + status
    ├── proposal.md         # Hypothesis + method
    ├── results.md          # Findings + tables
    ├── tasks.md            # Implementation checklist
    ├── config.yaml         # Parameters
    ├── generation_functions.py
    ├── sbatch/             # run.sh + _job.sh
    ├── artifacts/          # Git-tracked summary.json files
    ├── results/            # Full results (gitignored)
    └── logs/               # SLURM logs (gitignored)
```

### Creating a New Experiment

```bash
cp -r experiments/_template experiments/01_new_experiment
# Edit the files in experiments/01_new_experiment/
bash experiments/01_new_experiment/sbatch/run.sh --dry-run
```

---

## Currently Running

None

---

## Completed Experiments

### 00_baseline - Layer Reuse Baseline

**Status**: GSM8K and MMLU complete, Minerva Math and IFEval partial (limit runs only)
**Details**: [experiments/00_baseline/](../experiments/00_baseline/)

| Task | Configs | Status |
|------|---------|--------|
| GSM8K (full) | 9/9 | Done |
| MMLU (full) | 9/9 | Done |
| MMLU (limit_10) | 9/9 | Done |
| Minerva Math (limit_10) | 9/9 | Done |
| IFEval (limit_5) | 1/9 | Partial |

**Quick check:**
```bash
bash experiments/00_baseline/sbatch/run.sh --status --task gsm8k
```

---

## Planned Next Steps

1. **Run full Minerva Math** (all 9 configs, no limit)
2. **Run full IFEval** (all 9 configs, no limit)
3. **Investigate k1_first throughput anomaly** (22.75 vs ~50 tok/s)
4. **Code generation tasks** (HumanEval, MBPP) - initial limit_10 tests showed 0% pass@1

---

## Known Issues

See [experiments/00_baseline/tasks.md](../experiments/00_baseline/tasks.md) and [experiments/tasks.md](experiments/tasks.md) for details.

- Layer reuse does NOT apply to loglikelihood tasks (MMLU, GPQA)
- GPQA dataset requires HuggingFace authentication
- Throughput metrics missing for loglikelihood tasks

---

## Quick Commands

```bash
# Check experiment status
bash experiments/00_baseline/sbatch/run.sh --status
bash experiments/00_baseline/sbatch/run.sh --status --task mmlu

# Submit jobs
bash experiments/00_baseline/sbatch/run.sh --task minerva_math

# View logs
tail -f experiments/00_baseline/logs/gsm8k/k2_middle/slurm_*.log
```
