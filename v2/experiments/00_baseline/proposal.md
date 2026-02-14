# Proposal: 00_baseline - Layer Reuse Baseline

**Status**: completed
**Created**: 2026-01-18
**Author**: Murilo

## Hypothesis

Layer reuse (skipping transformer layer computation every k-th diffusion step by reusing cached outputs) trades accuracy for throughput in block-diffusion LLM generation. The tradeoff depends on which layers are reused (first/middle/last) and the reuse frequency (k=1,2,3).

## Background

Fast-dLLM v2 uses block diffusion to generate text in parallel. Each diffusion step runs the full transformer stack. Layer reuse skips a subset of 12 layers on non-recompute steps, using cached hidden states instead. This reduces FLOPs per step but may degrade generation quality.

## Method

### Approach

Run all 9 configurations (3x3 grid) on generative and loglikelihood benchmarks, measuring accuracy and throughput.

### Variables

- **Independent**: reuse_k (1, 2, 3) x layer_subset (first, middle, last)
- **Dependent**: accuracy (task-specific), throughput (tokens/s)
- **Controlled**: model (Fast_dLLM_v2_7B), batch_size=1, use_block_cache=True, threshold=0.95

### Setup

| Component | Details |
|-----------|---------|
| Model | Efficient-Large-Model/Fast_dLLM_v2_7B |
| Benchmarks | GSM8K, MMLU, Minerva Math, IFEval |
| Compute | RTX 6000 (Turing, SM 7.5), 64GB RAM |
| GPU Dtype | float16 (auto-detected) |

### Experiment Matrix (9 configurations)

| Subset | k=1 (baseline) | k=2 | k=3 |
|--------|----------------|-----|-----|
| first (layers 1-11) | k1_first | k2_first | k3_first |
| middle (layers 12-23) | k1_middle | k2_middle | k3_middle |
| last (layers 24-35) | k1_last | k2_last | k3_last |

Note: k=1 means every step is a full computation (no reuse), so all k=1 configs should produce identical accuracy.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | exact_match,strict-match for GSM8K; acc for MMLU |
| Throughput | Tokens per second (generation tasks only) |

### Baseline

k1_first is the true baseline (k=1 = no layer reuse).

### Success Criteria

- **Confirm if**: k=2 or k=3 shows >1.2x throughput improvement with <5% accuracy drop
- **Reject if**: all reuse configs show negligible speedup or catastrophic accuracy loss

## Limitations

- Layer reuse only applies to generative tasks (GSM8K, Minerva Math), not loglikelihood tasks (MMLU)
- MMLU results are identical across all 9 configs (layer reuse has no effect on loglikelihood evaluation)
- Throughput measurements include model loading overhead
