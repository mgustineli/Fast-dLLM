# Experiment: 00_baseline

> **Status**: completed (GSM8K, MMLU) | partial (Minerva Math, IFEval - limit runs only)
> **Created**: 2026-01-18

## Quick Start

```bash
cd ~/eic-lab/Fast-dLLM/v2

# Check status
bash experiments/00_baseline/sbatch/run.sh --status
bash experiments/00_baseline/sbatch/run.sh --status --task mmlu

# Test run (10 samples)
bash experiments/00_baseline/sbatch/run.sh --limit 10 --dry-run

# Full run
bash experiments/00_baseline/sbatch/run.sh
```

## Overview

Baseline layer reuse experiment measuring the accuracy/throughput tradeoff across a 3x3 grid of configurations: reuse frequency k={1,2,3} x layer subset {first, middle, last}. k=1 is the true baseline (no reuse).

## Documents

| Document | Description |
|----------|-------------|
| [proposal.md](proposal.md) | Hypothesis, method, and evaluation plan |
| [results.md](results.md) | Findings and analysis |
| [tasks.md](tasks.md) | Implementation checklist |
| [config.yaml](config.yaml) | Experiment parameters |

## Key Results (GSM8K)

| Config | Accuracy | Tokens/s |
|--------|----------|----------|
| k1_first | 80.4% | 22.75 |
| k1_middle | 80.4% | 49.54 |
| k1_last | 80.4% | 49.63 |
| k2_first | 73.7% | 39.85 |
| k2_middle | 73.6% | 33.48 |
| k2_last | 73.8% | 40.51 |
| k3_first | 67.2% | 34.29 |
| k3_middle | 67.6% | 33.86 |
| k3_last | 67.7% | 34.05 |

k1_first throughput anomaly (22.75 vs ~50 tok/s) is under investigation.
