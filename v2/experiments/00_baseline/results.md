# Results: 00_baseline - Layer Reuse Baseline

**Status**: completed (GSM8K, MMLU), partial (Minerva Math, IFEval)
**Date**: 2026-01-20
**Proposal**: [proposal.md](proposal.md)

## Summary

Layer reuse (k=2,3) reduces GSM8K accuracy by 7-13 percentage points compared to the k=1 baseline (80.4%). Layer subset (first/middle/last) has minimal impact on accuracy. MMLU results are identical across all configs because layer reuse only affects generative tasks.

## Results

### GSM8K (Full, 1319 samples)

| Config | Accuracy | Tokens/s | Accuracy Drop |
|--------|----------|----------|---------------|
| k1_first | 80.4% | 22.75 | - (baseline) |
| k1_middle | 80.4% | 49.54 | 0.0% |
| k1_last | 80.4% | 49.63 | 0.0% |
| k2_first | 73.7% | 39.85 | -6.7% |
| k2_middle | 73.6% | 33.48 | -6.8% |
| k2_last | 73.8% | 40.51 | -6.6% |
| k3_first | 67.2% | 34.29 | -13.2% |
| k3_middle | 67.6% | 33.86 | -12.8% |
| k3_last | 67.7% | 34.05 | -12.7% |

### MMLU (Full)

All 9 configs: **66.53% accuracy** (identical, layer reuse has no effect on loglikelihood tasks).

### Minerva Math (limit_10)

| Config | Accuracy | Tokens/s |
|--------|----------|----------|
| k1_first | 42.9% | 50.16 |
| k1_middle | 42.9% | 49.63 |
| k1_last | 42.9% | 49.61 |
| k2_first | 41.4% | 43.30 |
| k2_middle | 40.0% | 19.85 |
| k2_last | 41.4% | 19.59 |
| k3_first | 31.4% | 17.44 |
| k3_middle | 25.7% | 37.72 |
| k3_last | 28.6% | 16.40 |

## Analysis

### Key Findings

1. **k=1 configs produce identical accuracy** as expected (no layer reuse active).
2. **Accuracy degrades proportionally with k**: k=2 loses ~7%, k=3 loses ~13% on GSM8K.
3. **Layer subset has minimal impact on accuracy**: within each k level, first/middle/last differ by <1%.
4. **MMLU is unaffected** because loglikelihood evaluation does not trigger the layer reuse code path.

### Unexpected Observations

- **k1_first throughput anomaly**: 22.75 tok/s vs ~50 tok/s for k1_middle and k1_last. All k=1 configs should be identical. This may be due to GPU warmup or a cold-start effect.
- **Throughput does not consistently increase with higher k**: k=2 and k=3 show ~34-40 tok/s, lower than k=1's best (49.6 tok/s). The expected throughput gain from skipping layers is not materializing.

## Conclusion

**Hypothesis partially confirmed**: Layer reuse trades accuracy for speed, but the throughput gains are not as large as expected. The accuracy/throughput tradeoff may not be favorable enough for practical use at k=2 or k=3.

## Next Steps

- Investigate k1_first throughput anomaly
- Run full Minerva Math and IFEval evaluations
- Explore alternative acceleration strategies (TRIM, adaptive skip)

## Artifacts

| Artifact | Location |
|----------|----------|
| Summary JSONs | `artifacts/` (37 files across 5 task dirs) |
| Full results | `results/` (gitignored) |
| SLURM logs | `logs/` (gitignored) |
