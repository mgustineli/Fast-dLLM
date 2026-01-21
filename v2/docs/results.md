# Experiment Results

This document tracks experimental results for Fast-dLLM v2 across different benchmarks, configurations, and optimization strategies.

## Overview

Fast-dLLM v2 implements several acceleration techniques:
- **Layer Reuse (k=1,2,3)**: Reuse intermediate layer activations across diffusion steps
- **Layer Subset**: Which layers to reuse (first, middle, last)
- **KV Caching**: Prefix and dual cache for attention key-value pairs
- **Threshold-based Parallel Decoding**: Unmask high-confidence tokens together
- **TRIM**: Token-level reuse minimization

## Experiment Log

### Template Entry

```markdown
#### [Date] - [Experiment Name]
- **Task**: [GSM8K/MMLU/HumanEval/GPQA/etc.]
- **Configuration**:
  - Layer Reuse: k=[1/2/3]
  - Layer Subset: [first/middle/last]
  - Other params: [threshold, cache settings, etc.]
- **Results**:
  - Accuracy: [X.XX%]
  - Throughput: [tokens/s]
  - Speedup: [Xx]
- **Notes**: [observations, issues, comparisons]
- **Logs**: [path to logs or results directory]
```

---

## GSM8K Results

### Baseline Experiments (2026-01-19)

Full 9-configuration sweep completed on GSM8K (0-shot with chain-of-thought prompting).

#### Complete Results Table

| Config | Reuse K | Layer Subset | Accuracy | Throughput (tokens/s) | Total Time (h) | Notes |
|--------|---------|--------------|----------|----------------------|----------------|-------|
| k1_first | 1 | first | 80.44% | 22.75 | 5.33 | Baseline (layers 1-11) - **Anomalously slow** ⚠️ |
| k1_middle | 1 | middle | 80.44% | 49.54 | 2.45 | Baseline (layers 12-23) |
| k1_last | 1 | last | 80.44% | 49.63 | 2.44 | Baseline (layers 24-35) |
| k2_first | 2 | first | 73.69% | 39.85 | 2.79 | 2x layer reuse |
| k2_middle | 2 | middle | 73.62% | 33.48 | 3.25 | 2x layer reuse |
| k2_last | 2 | last | 73.77% | 40.51 | 2.70 | 2x layer reuse |
| k3_first | 3 | first | 67.17% | 34.29 | 3.19 | 3x layer reuse |
| k3_middle | 3 | middle | 67.55% | 33.86 | 3.23 | 3x layer reuse |
| k3_last | 3 | last | 67.70% | 34.05 | 3.22 | 3x layer reuse |

**Key Observations:**
- **Accuracy degrades with increased layer reuse** (k=1: 80.44% → k=2: ~73.7% → k=3: ~67.5%)
- k=1 configurations all have identical accuracy (80.44%), as expected for baseline
- k1_first shows **anomalously low throughput** (22.75 tokens/s vs ~49.6 tokens/s for k1_middle/k1_last) - requires investigation
- k2 and k3 show more consistent throughput (33-40 tokens/s range)
- Layer subset choice (first/middle/last) has minimal impact on accuracy within same k value
- Total tokens generated: 436,484 across full dataset

**Performance Summary:**
- Best accuracy: k1 configs at 80.44%
- Best throughput: k1_last at 49.63 tokens/s (excluding anomalous k1_first)
- Best accuracy/speed tradeoff: k2_last at 73.77% accuracy, 40.51 tokens/s

---

## MMLU Results

### Known Issues

#### 2026-01-20 - Layer Reuse Not Applied (CRITICAL BUG)
- **Task**: MMLU (5-shot)
- **Configuration**:
  - Layer Reuse: k=1,2,3 (NOT APPLIED)
  - Layer Subset: first, middle, last (NOT APPLIED)
- **Results**:
  - **All configurations**: 66.53% accuracy (identical)
- **Notes**:
  - Layer reuse optimization not applied to loglikelihood tasks
  - All MMLU results are invalid for measuring layer reuse impact
  - See experiment doc: `experiments/00-layer-reuse-loglikelihood.md`
  - **Action Required**: Fix implementation before re-running experiments
- **Logs**: `results/00_baseline/mmlu_*/`

---

## HumanEval Results

_No experiments recorded yet._

---

## GPQA Results

### Known Issues

#### 2026-01-20 - Dataset Authentication Required
- **Task**: GPQA
- **Status**: Cannot run - requires HuggingFace authentication
- **Notes**:
  - Dataset is gated and requires accepting terms of use
  - Need to add HF_TOKEN to job scripts
  - See experiment doc: `experiments/01-gpqa-authentication.md`
- **Action Required**: Set up authentication before running experiments

---

## Performance Metrics Summary

| Benchmark | Best Accuracy | Best Throughput (tokens/s) | Configuration | Notes |
|-----------|---------------|---------------------------|---------------|-------|
| GSM8K     | 80.44%        | 49.63 (k1_last)          | k1 (all)      | Baseline with block cache |
| GSM8K (k2)| 73.77%        | 40.51 (k2_last)          | k2_last       | 2x layer reuse, -6.67pp accuracy |
| GSM8K (k3)| 67.70%        | 34.29 (k3_first)         | k3_last       | 3x layer reuse, -12.74pp accuracy |
| MMLU      | 66.53%*       | N/A                       | baseline*     | *Invalid - layer reuse bug |
| HumanEval | TBD           | TBD                       | -             | Not yet tested |
| GPQA      | N/A           | N/A                       | -             | Requires HF auth |

---

## Accuracy vs Speed Tradeoffs

### GSM8K Layer Reuse Analysis

Comparing best configurations per k-value (using k1_middle as baseline since k1_first has anomalous throughput):

| Configuration | Accuracy | Degradation | Throughput (tokens/s) | Speedup vs k1_middle |
|---------------|----------|-------------|----------------------|---------------------|
| k1_middle (baseline) | 80.44% | - | 49.54 | 1.0x |
| k2_last (best k2) | 73.77% | -6.67pp | 40.51 | 0.82x |
| k3_first (best k3) | 67.17% | -13.27pp | 34.29 | 0.69x |

**Analysis:**
- Layer reuse (k>1) **reduces throughput** instead of improving it in current implementation
- Expected: Higher k → faster inference (skip more layers)
- Observed: Higher k → slower inference
- **Conclusion**: Layer reuse implementation may have overhead issues or is not optimized correctly

**Note:** k1_first excluded from baseline comparison due to anomalously low throughput (22.75 tokens/s vs ~49.6 for k1_middle/k1_last)

---

## Pending Experiments

1. **Fix MMLU Layer Reuse Bug**
   - Priority: CRITICAL
   - Blocking: All loglikelihood task experiments
   - Issue: `experiments/00-layer-reuse-loglikelihood.md`

2. **Add Throughput Tracking for MMLU**
   - Priority: High
   - Issue: `experiments/02-missing-throughput-tracking.md`

3. **Set Up GPQA Authentication**
   - Priority: High
   - Issue: `experiments/01-gpqa-authentication.md`

4. **Complete Baseline Sweep**
   - Run HumanEval across all layer reuse configurations
   - Verify MMLU after fixing layer reuse bug
   - Run GPQA after authentication setup

---

## Notes

- All experiments use `Fast_dLLM_v2_7B` model
- Results stored in `results/` directory with structure: `results/[experiment_name]/[task]_[config]/[layer_config]/`
- Accuracy metrics extracted from `summary.json` files
- Throughput metrics (when available) also in `summary.json`

---

## References

- Main README: `../README.md`
- Experiment Documentation: `experiments/`
- Operations Guide: `operations.md`
- Status Updates: `STATUS.md`
