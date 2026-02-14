# Experiments

Cross-cutting documentation for Fast-dLLM v2 experiments.

> **Note**: Experiment-specific documentation (proposal, results, tasks) now lives in each experiment's directory under `experiments/`. This directory contains cross-cutting docs that apply to multiple experiments.

## Experiment Directories

| Experiment | Status | Location |
|-----------|--------|----------|
| 00_baseline | GSM8K/MMLU done | [experiments/00_baseline/](../../experiments/00_baseline/) |

## Creating a New Experiment

```bash
cd ~/eic-lab/Fast-dLLM/v2
cp -r experiments/_template experiments/01_new_experiment
# Edit the files, then:
bash experiments/01_new_experiment/sbatch/run.sh --dry-run
```

See [experiments/_template/](../../experiments/_template/) for the scaffold.

---

## Cross-Cutting Issues

These issues affect multiple experiments and are tracked here:

| Issue | Priority | Details |
|-------|----------|---------|
| Layer reuse not applied to loglikelihood | CRITICAL | [00-layer-reuse-loglikelihood.md](00-layer-reuse-loglikelihood.md) |
| GPQA authentication required | High | [01-gpqa-authentication.md](01-gpqa-authentication.md) |
| Missing throughput for loglikelihood | High | [02-missing-throughput-tracking.md](02-missing-throughput-tracking.md) |

See [tasks.md](tasks.md) for the full task list.

---

## Benchmark Tasks

| Task | Description | Metric | Type |
|------|-------------|--------|------|
| `gsm8k` | Grade School Math 8K | Accuracy | Generative |
| `mmlu` | Massive Multitask Language Understanding | Accuracy | Loglikelihood |
| `gpqa_main_n_shot` | Graduate-level Q&A | Accuracy | Loglikelihood |
| `minerva_math` | Mathematical reasoning | Accuracy | Generative |
| `ifeval` | Instruction following | Pass rate | Generative |

---

## Experiment Matrix (00_baseline)

| Subset | k=1 (baseline) | k=2 | k=3 |
|--------|----------------|-----|-----|
| first (layers 1-11) | k1_first | k2_first | k3_first |
| middle (layers 12-23) | k1_middle | k2_middle | k3_middle |
| last (layers 24-35) | k1_last | k2_last | k3_last |
