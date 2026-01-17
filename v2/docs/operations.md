# Operations Guide

Operational patterns for running Fast-dLLM v2 experiments on the Georgia Tech PACE cluster.

> **Prerequisites**: See [PACE Cluster Reference](reference/pace-cluster.md) for disk quotas, SLURM queues, and troubleshooting.

## Directory Structure

```
~/scratch/Fast-dLLM/v2/
├── app.py                    # Gradio web interface
├── run_chatbot.py           # CLI chatbot
├── eval.py                  # Evaluation harness
├── eval_gsm8k.sh            # GSM8K evaluation script
├── eval_gsm8k_reuse_layers.sh   # Layer reuse experiments
├── eval_gsm8k_reuse_activations.sh  # Activation reuse experiments
├── generation_functions.py  # Core generation algorithms
├── configs/                 # Model configurations
├── data/                    # Training datasets
├── results/                 # Experiment outputs
│   └── YYYYMMDD_HHMMSS/    # Timestamped results
├── src/
│   └── lmflow/             # LMFlow training framework
└── train_scripts/          # Fine-tuning scripts
    └── finetune_alpaca.sh
```

## Quick Start

```bash
# One-time setup
cd ~/scratch/Fast-dLLM/v2
pip install -e .

# Set environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Run evaluation
bash eval_gsm8k.sh --limit 10  # Test mode
bash eval_gsm8k.sh             # Full evaluation
```

## Environment Setup

### Required Environment Variables

```bash
# Add to ~/.bashrc or set before running
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Optional: Set cache directories to scratch
export HF_HOME=~/scratch/.cache/huggingface
export TORCH_HOME=~/scratch/.cache/torch
```

### Installing Dependencies

```bash
# From the v2 directory
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Running Evaluations

### GSM8K Evaluation

```bash
# Full evaluation (all samples)
bash eval_gsm8k.sh

# Test mode (limited samples)
bash eval_gsm8k.sh --limit 10

# With custom batch size
bash eval_gsm8k.sh --limit 100 --batch_size 4
```

### Custom Evaluation

```bash
accelerate launch eval.py \
    --tasks gsm8k \
    --batch_size 1 \
    --num_fewshot 0 \
    --model fast_dllm_v2 \
    --model_args model_path=Efficient-Large-Model/Fast_dLLM_v2_7B,threshold=0.9,show_speed=True \
    --output_path results/my_experiment/
```

### Model Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `model_path` | HuggingFace path | Model to load |
| `threshold` | 0.0-1.0 | Confidence threshold for parallel decoding |
| `use_cache` | True/False | Enable KV caching |
| `dual_cache` | True/False | Enable dual caching |
| `show_speed` | True/False | Display throughput metrics |
| `k` | 1,2,3 | Layer reuse parameter |

## Interactive Usage

### Gradio Web UI

```bash
python app.py
# Opens at http://localhost:10086
```

Features:
- Real-time denoising visualization
- Adjustable parameters (block size, temperature, threshold)
- Performance metrics display

### CLI Chatbot

```bash
python run_chatbot.py
```

Commands:
- Type message + Enter to chat
- `clear` - Clear conversation history
- `exit` - Quit

## Training

### Fine-tuning on Alpaca

```bash
# Download training data
cd data && bash download.sh alpaca && cd ..

# Run fine-tuning
bash train_scripts/finetune_alpaca.sh
```

## Experiment Workflow

1. **Setup**: Ensure environment variables are set
2. **Baseline**: Run `bash eval_gsm8k.sh --limit 10` to verify setup
3. **Experiment**: Modify model args for your experiment
4. **Run**: Execute evaluation script
5. **Results**: Check `results/TIMESTAMP/` for outputs

## Results Directory Structure

```
results/
└── 20260117_120000/
    └── gsm8k_threshold1_run_raw/
        ├── results.json          # Metrics and scores
        └── samples_*.jsonl       # Individual predictions
```

## Layer Reuse Experiments

```bash
# Run layer reuse evaluation
bash eval_gsm8k_reuse_layers.sh

# Test specific k values
accelerate launch eval.py \
    --model fast_dllm_v2 \
    --model_args model_path=...,k=2
```

## SBATCH Integration

For running on PACE cluster with SLURM:

```bash
#!/bin/bash
#SBATCH --job-name=fast-dllm-eval
#SBATCH -N1 -n1 --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 120
#SBATCH -q embers
#SBATCH --output=logs/%j.out

cd ~/scratch/Fast-dLLM/v2
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

bash eval_gsm8k.sh
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce batch size, use `--batch_size 1` |
| Model not found | Check `model_path` in model_args |
| Slow download | Set `HF_HOME` to scratch directory |
| Permission denied | Check file permissions in scratch |

See [PACE Cluster Reference](reference/pace-cluster.md) for more troubleshooting.
