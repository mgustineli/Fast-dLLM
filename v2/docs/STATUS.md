# Project Status

> **Last Updated**: January 17, 2026

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

## Known Issues

### GSM8K Accuracy = 0 with `eval_pace_script.sbatch`

**Date**: January 17, 2026

**Symptom**: Running `sbatch sbatch/eval_pace_script.sbatch --task gsm8k --limit 10` produces `accuracy=0.0` despite the model generating correct answers.

**Root Cause Analysis**:

1. **Answer Format Mismatch**
   - `eval.py:300-304` replaces the prompt to ask for `\boxed{}` format:
     ```python
     question = question.replace(
         "Answer:",
         "Please reason step by step, and put your final answer within \\boxed{{}}.",
     )
     ```
   - But lm_eval's GSM8K filter expects `#### N` format (regex: `#### (\-?[0-9\.\,]+)`)
   - Model correctly outputs `\boxed{3}`, `\boxed{540}`, etc. but the filter can't extract them

2. **Empty Answers (6 out of 10 samples)**
   - GSM8K uses `until: ["\n\n", "Question:"]` as stop sequences
   - Some generations stop immediately at `\n\n` before producing any answer
   - Working answers have long reasoning; empty ones hit `\n\n` early in generation

**Evidence** (from `logs/baseline/gsm8k/.../slurm.log`):
```
answer:   ...the total number of bolts of fiber required to make the robe is \(\boxed{3}\).  # Correct but not #### format
answer:                                                                                      # Empty - early stop
```

**Solutions**:

| Option | Description | Complexity |
|--------|-------------|------------|
| A | Change prompt to ask for `#### N` format instead of `\boxed{}` | Simple |
| B | Create custom GSM8K task YAML with `\boxed{}` filter (like minerva_math) | More robust |

**Resolution** (January 17, 2026):

1. **Missing lm_eval flags** - Found that original repo (`v2/eval_script.sh`) uses flags missing from local scripts:
   - `--apply_chat_template` - Formats prompts using model's chat template (Qwen format)
   - `--fewshot_as_multiturn` - Formats few-shot examples as conversation turns
   - Updated `sbatch/eval_pace_script.sbatch`, `sbatch/eval_script.sh`, and `sbatch/eval_reuse_layers_array.sbatch`

2. **Outdated lm_eval version** - Installed lm_eval was older than 0.4.8 and didn't support the above flags:
   - Error: `eval.py: error: unrecognized arguments: --fewshot_as_multiturn --apply_chat_template`
   - Updated `v2/pyproject.toml` to require `lm_eval>=0.4.8`
   - Run `uv pip install --upgrade "lm_eval>=0.4.8"` to fix

3. **Fixed `log_utils.py` accuracy extraction** - `summary.json` showed `accuracy: null`:
   - lm_eval writes results to `<model_name>/results_<timestamp>.json` subdirectory
   - `log_utils.py` was looking for `results.json` directly in output_dir
   - Added `find_results_file()` function to search subdirectories for timestamped results files

4. **Made sbatch scripts user-agnostic** - Replaced hardcoded paths with `~/scratch/...`

**Status**: ✅ RESOLVED - Re-test achieved **60% accuracy** on GSM8K (10 samples) using `exact_match,flexible-extract` metric.

---

## Quick Links

| Resource | Link |
|----------|------|
| Concepts | [concepts/](concepts/) |
| Operations | [operations.md](operations.md) |
| PACE Reference | [reference/pace-cluster.md](reference/pace-cluster.md) |
| Templates | [_templates/](_templates/) |
