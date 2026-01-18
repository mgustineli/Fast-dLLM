# Project Status

> **Last Updated**: January 18, 2026

## Research Goal

Fast-dLLM v2 adapts pretrained autoregressive models (Qwen2.5) into parallel text generation via block diffusion, targeting 2.5x+ speedup with ~1B tokens of fine-tuning.

## Current Phase: GPU Precision Optimization

### Recent Fix: Auto-Dtype Detection

Resolved 6x performance gap between Ampere (A40) and Turing (RTX 6000) GPUs:

| GPU | Architecture | Old Dtype | New Dtype | Tokens/s |
|-----|--------------|-----------|-----------|----------|
| A40 | Ampere (SM 8.0) | bfloat16 | bfloat16 | ~29 |
| RTX 6000 | Turing (SM 7.5) | bfloat16 (broken) | float16 | TBD |

**Root Cause**: bfloat16 requires Ampere+. Turing GPUs fell back to FP32, bypassing Tensor Cores.

**Fix**: `get_optimal_dtype()` in `eval.py` auto-detects GPU compute capability and selects:
- SM 8.0+: bfloat16 (native support)
- SM 7.0-7.5: float16 (Tensor Core support)
- Older: float32

## Key Results

| Metric | Value |
|--------|-------|
| Throughput vs Qwen2.5-7B | 2.54x |
| GSM8K Accuracy | 83.7% |
| HumanEval | 63.4% |

## Next Steps

1. **[Pending]** Validate RTX 6000 throughput with float16 fix
2. **[Pending]** Layer reuse experiments (k=1,2,3 × first/middle/last)
3. **[Planned]** TRIM logic implementation
4. **[Planned]** Activation reuse profiling

## Active Research

| Feature | Status |
|---------|--------|
| Block Diffusion | Working |
| KV Cache (block + sub-block) | Working |
| Confidence Decoding | Working |
| Layer Reuse (k param) | Testing |
| TRIM Logic | Planned |

## Quick Links

- [Concepts](concepts/)
- [Operations](operations.md)
- [PACE Reference](reference/pace-cluster.md)
