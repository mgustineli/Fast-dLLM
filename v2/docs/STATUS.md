# Project Status

> **Last Updated**: January 2026

## Research Thesis

Fast-dLLM v2 is a block diffusion language model that efficiently adapts pretrained autoregressive models (Qwen2.5) into parallel text generation systems, achieving 2.5x speedup while requiring only ~1B tokens of fine-tuning (500x reduction vs full-attention diffusion LLMs).

## Current State

**Phase**: Acceleration Research - Layer Reuse & TRIM Logic

The base Fast-dLLM v2 model is functional with block diffusion and hierarchical caching. Current focus is on additional acceleration techniques (layer reuse, TRIM) to push speedups further.

## Key Results

| Finding | Source | Implication |
|---------|--------|-------------|
| 2.54x throughput vs Qwen2.5-7B-Instruct | Benchmark | Block diffusion + caching works |
| 5.2% accuracy improvement over LLaDA | Benchmark | Better quality than prior dLLMs |
| 83.7% on GSM8K | Eval | Strong math reasoning |
| 63.4% on HumanEval | Eval | Good code generation |

## Model Variants

| Model | Parameters | HuggingFace |
|-------|------------|-------------|
| Fast-dLLM v2 (7B) | 7B | `Efficient-Large-Model/Fast_dLLM_v2_7B` |
| Fast-dLLM v2 (1.5B) | 1.5B | - |

## Active Research Directions

1. **Layer Reuse** (k=1,2,3) - Skip transformer layers during diffusion steps
2. **TRIM Logic** - Token-level Reuse Minimization for efficient generation
3. **Activation Reuse** - Reuse intermediate activations across steps
4. **Threshold Tuning** - Optimal confidence threshold for parallel decoding

## What Works (Current Implementation)

| Feature | Status | Notes |
|---------|--------|-------|
| Block Diffusion | Working | Core architecture |
| Block-level KV Cache | Working | Historical context caching |
| Sub-block Cache | Working | Parallel generation within blocks |
| Confidence Decoding | Working | `threshold` parameter |
| Gradio Web UI | Working | `app.py` |
| CLI Chatbot | Working | `run_chatbot.py` |
| GSM8K Evaluation | Working | `eval_gsm8k.sh` |

## Experiments

| ID | Name | Status | Result |
|----|------|--------|--------|
| - | Baseline GSM8K | Complete | 83.7% accuracy |
| - | Layer Reuse (k=1,2,3) | In Progress | Testing speedup vs accuracy tradeoff |
| - | TRIM Logic | In Progress | Token-level optimization |

## Next Steps

1. **[Active]** Benchmark layer reuse with different k values
2. **[Active]** Implement and test TRIM logic
3. Profile activation reuse potential
4. Compare speedups across different batch sizes

## Benchmark Coverage

| Benchmark | Status | Score |
|-----------|--------|-------|
| GSM8K | Evaluated | 83.7% |
| HumanEval | Evaluated | 63.4% |
| MBPP | Evaluated | 63.0% |
| MMLU | Evaluated | 66.6% |
| GPQA | Evaluated | 31.9% |
| IFEval | Evaluated | 61.4% |
| Math | Evaluated | 61.6% |

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Operations | [operations.md](operations.md) |
| PACE Reference | [reference/pace-cluster.md](reference/pace-cluster.md) |
| Templates | [_templates/](_templates/) |
