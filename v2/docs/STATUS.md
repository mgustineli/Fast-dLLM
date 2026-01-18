# Project Status

> **Last Updated**: January 18, 2026

## Research Goal

Fast-dLLM v2 adapts pretrained autoregressive models (Qwen2.5) into parallel text generation via block diffusion, targeting 2.5x+ speedup with ~1B tokens of fine-tuning.

## Current Phase: Layer Reuse Experiments

Running experiments to measure speedup vs accuracy tradeoff for layer reuse (k=1,2,3 × first/middle/last subsets).

## Recent Changes

### GPU Precision Auto-Detection
Fixed 6x performance gap between Ampere and Turing GPUs:

| GPU | Architecture | Dtype | Status |
|-----|--------------|-------|--------|
| A40 | Ampere (SM 8.0) | bfloat16 | ✓ Working |
| RTX 6000 | Turing (SM 7.5) | float16 | ✓ Fixed |

**Fix**: `get_optimal_dtype()` in `eval.py` auto-detects GPU compute capability.

### Smart Experiment Runner
New workflow avoids re-running completed experiments:

```bash
bash sbatch/run_reuse_experiments.sh --status   # Check completion
bash sbatch/run_reuse_experiments.sh            # Run only missing
bash sbatch/run_reuse_experiments.sh --force    # Re-run all
```

## Key Results

| Metric | Value |
|--------|-------|
| Throughput vs Qwen2.5-7B | 2.54x |
| GSM8K Accuracy | 83.7% |
| HumanEval | 63.4% |

## Active Experiments

| Config | Status | Accuracy | Throughput |
|--------|--------|----------|------------|
| k1_first | Pending | - | - |
| k1_middle | Pending | - | - |
| k1_last | Pending | - | - |
| k2_first | Pending | - | - |
| k2_middle | Pending | - | - |
| k2_last | Pending | - | - |
| k3_first | Pending | - | - |
| k3_middle | Pending | - | - |
| k3_last | Pending | - | - |

## Next Steps

1. **[Active]** Complete layer reuse experiments (9 configs)
2. **[Planned]** Analyze speedup vs accuracy tradeoff
3. **[Planned]** TRIM logic implementation

## Quick Links

- [Operations Guide](operations.md) - How to run experiments
- [PACE Reference](reference/pace-cluster.md) - Cluster resources
- [Concepts](concepts/) - Research ideas
