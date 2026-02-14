# Project Status

> **Last Updated**: February 14, 2026

## Research Goal

Fast-dLLM v2 adapts pretrained autoregressive models (Qwen2.5) into parallel text generation via block diffusion, targeting 2.5x+ speedup with ~1B tokens of fine-tuning.

## Current Phase: Layer Reuse Experiments

Running experiments to measure speedup vs accuracy tradeoff for layer reuse (k=1,2,3 x first/middle/last subsets).

## Recent Changes

### Experiment Infrastructure Refactoring (Feb 2026)
Experiments are now self-contained directories under `experiments/`:
```bash
bash experiments/00_baseline/sbatch/run.sh --status   # Check completion
bash experiments/00_baseline/sbatch/run.sh             # Run only missing
bash experiments/00_baseline/sbatch/run.sh --force     # Re-run all
```

### GPU Precision Auto-Detection
Fixed 6x performance gap between Ampere and Turing GPUs:

| GPU | Architecture | Dtype | Status |
|-----|--------------|-------|--------|
| A40 | Ampere (SM 8.0) | bfloat16 | Working |
| RTX 6000 | Turing (SM 7.5) | float16 | Fixed |

**Fix**: `get_optimal_dtype()` in `eval.py` auto-detects GPU compute capability.

## Key Results (00_baseline, GSM8K)

| Config | Accuracy | Tokens/s |
|--------|----------|----------|
| k1_first (baseline) | 80.4% | 22.75 |
| k1_middle | 80.4% | 49.54 |
| k1_last | 80.4% | 49.63 |
| k2_first | 73.7% | 39.85 |
| k2_middle | 73.6% | 33.48 |
| k2_last | 73.8% | 40.51 |
| k3_first | 67.2% | 34.29 |
| k3_middle | 67.6% | 33.86 |
| k3_last | 67.7% | 34.05 |

Full results: [experiments/00_baseline/results.md](../experiments/00_baseline/results.md)

## Next Steps

1. **[Planned]** Investigate k1_first throughput anomaly
2. **[Planned]** Run full Minerva Math and IFEval evaluations
3. **[Planned]** TRIM logic implementation

## Quick Links

- [Operations Guide](operations.md) - How to run experiments
- [PACE Reference](references/pace-cluster.md) - Cluster resources
- [Concepts](concepts/) - Research ideas
- [00_baseline Experiment](../experiments/00_baseline/) - Layer reuse baseline
