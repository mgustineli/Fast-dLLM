# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fast-dLLM is an NVIDIA Labs diffusion-based Large Language Model (LLM) inference acceleration framework. It provides training-free acceleration techniques (KV caching, parallel decoding) for diffusion LLMs like Dream and LLaDA, achieving 2-11x speedups.

**Three versions:**
- `llada/` - LLaDA model implementation (8B Instruct)
- `dream/` - Dream model implementation (7B Base)
- `v2/` - Fast-dLLM v2 with block-diffusion, layer reuse, and TRIM logic

## Common Commands

### Environment Setup
```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
pip install -r requirements.txt
```

### Running Evaluations

**LLaDA (from llada/ directory):**
```bash
accelerate launch eval_llada.py --tasks gsm8k --num_fewshot 5 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,use_cache=True,threshold=0.9,show_speed=True
```

**Dream (from dream/ directory):**
```bash
accelerate launch eval.py --model dream \
--model_args pretrained='Dream-org/Dream-v0-Base-7B',max_new_tokens=256,diffusion_steps=256,alg=entropy,use_cache=true,show_speed=True \
--tasks gsm8k --num_fewshot 5 --batch_size 1
```

**V2 (from v2/ directory):**
```bash
bash eval_gsm8k.sh                    # Full evaluation
bash eval_gsm8k.sh --limit 10         # Test with 10 samples
```

### Interactive Chat
```bash
python llada/chat.py --gen_length 128 --steps 128 --block_size 32
python v2/run_chatbot.py              # CLI chatbot
python v2/app.py                      # Gradio web UI
```

### Training (v2)
```bash
cd v2
bash train_scripts/finetune_alpaca.sh
```

## Architecture

### Key Optimization Strategies
Model args are passed as comma-separated key=value strings:
- `use_cache=True` - Prefix KV cache (reuse attention K/V across diffusion steps)
- `dual_cache=True` - Also cache masked suffix tokens
- `threshold=0.9` - Confidence-aware parallel decoding (unmask high-confidence tokens together)
- `factor=1.0` - Factor-based parallel strategy
- `show_speed=True` - Display throughput metrics

### Core Generation Functions
Located in `llada/generate.py` and `v2/generation_functions.py`:
- `generate()` - Baseline diffusion generation
- `generate_with_prefix_cache()` - KV cache optimization
- `generate_with_dual_cache()` - Dual cache extension

### V2 Layer Reuse (Recent Development)
The `v2/` directory contains experimental layer reuse logic (k=1,2,3 parameter) and TRIM (token-level reuse minimization) for further acceleration.

### Evaluation Framework
Uses `lm_eval` with custom model registrations (`llada_dist`, `dream`, `fast_dllm_v2`). Benchmarks: GSM8K, HumanEval, MMLU, GPQA.

### Model Loading Pattern
All models use HuggingFace with `trust_remote_code=True`:
- LLaDA: `GSAI-ML/LLaDA-8B-Instruct` or `GSAI-ML/LLaDA-1.5`
- Dream: `Dream-org/Dream-v0-Base-7B`
- V2: `Efficient-Large-Model/Fast_dLLM_v2_7B`

## Code Style

- Apache 2.0 license headers on files
- Ruff for linting (v2): double quotes, space indentation
- Heavy use of argparse for CLI configuration
