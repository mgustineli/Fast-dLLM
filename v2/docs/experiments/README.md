# Experiments

Documentation for Fast-dLLM v2 experiments and evaluation scripts.

## Results Directory Structure

All experiments follow a consistent output path convention:

```
results/
├── <experiment>/                # Experiment type
│   └── <task>/                  # Benchmark task
│       └── <timestamp>_<config>/    # Run with configuration
│           ├── results.json         # Metrics and scores
│           └── samples_*.jsonl      # Individual predictions
```

**Example paths:**
- `results/baseline/gsm8k/20260117_120000_threshold1_nall/`
- `results/reuse_layers/gsm8k/20260117_120000_k2_first_n100/`
- `results/reuse_activations/mmlu/20260117_120000_k1_nall/`

---

## Experiment Types

### 1. Baseline (`baseline/`)

Standard Fast-dLLM v2 inference with threshold-based parallel decoding.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `threshold` | 1.0 | Confidence threshold for token unmasking |
| `use_block_cache` | True | Enable block-level KV caching |

**Script:** `sbatch/eval_gsm8k.sh`

```bash
# Full evaluation
bash sbatch/eval_gsm8k.sh

# Test with 10 samples
bash sbatch/eval_gsm8k.sh --limit 10
```

---

### 2. Layer Reuse (`reuse_layers/`)

Skip transformer layers during diffusion steps by reusing outputs from previous k steps.

| Parameter | Values | Description |
|-----------|--------|-------------|
| `reuse_k` | 1, 2, 3 | Number of previous steps to reuse from |
| `layer_subset` | first, middle, last | Which 12 layers to reuse |

**Hypothesis:** Later diffusion steps refine but don't fundamentally change representations, so we can reuse layer outputs from previous steps.

**Experiment Matrix (9 configurations):**

| Subset | k=1 | k=2 | k=3 |
|--------|-----|-----|-----|
| first (layers 0-11) | Task 0 | Task 1 | Task 2 |
| middle (layers 12-23) | Task 3 | Task 4 | Task 5 |
| last (layers 24-35) | Task 6 | Task 7 | Task 8 |

**Scripts:**

```bash
# Sequential (single GPU, ~30 hours total)
bash sbatch/eval_gsm8k_reuse_layers.sh --limit 10

# Parallel via SLURM array job (9 GPUs, ~4 hours)
sbatch sbatch/eval_reuse_layers_array.sbatch --limit 10

# Run specific configurations only
sbatch --array=0,3,6 sbatch/eval_reuse_layers_array.sbatch  # Only k=1
sbatch --array=0-2 sbatch/eval_reuse_layers_array.sbatch    # Only first layers
```

---

### 3. Activation Reuse (`reuse_activations/`)

Reuse activations from previous k steps without layer subsetting (all layers).

| Parameter | Values | Description |
|-----------|--------|-------------|
| `reuse_k` | 1, 2, 3 | Number of previous steps to reuse from |

**Script:** `sbatch/eval_gsm8k_reuse_activations.sh`

```bash
# Full evaluation
bash sbatch/eval_gsm8k_reuse_activations.sh

# Test with 10 samples
bash sbatch/eval_gsm8k_reuse_activations.sh --limit 10
```

---

## Benchmark Tasks

| Task | Description | Metric |
|------|-------------|--------|
| `gsm8k` | Grade School Math 8K | Accuracy |
| `mmlu` | Massive Multitask Language Understanding | Accuracy |
| `gpqa_main_n_shot` | Graduate-level Q&A | Accuracy |
| `minerva_math` | Mathematical reasoning | Accuracy |
| `ifeval` | Instruction following | Pass rate |
| `humaneval` | Code generation | Pass@1 |

**Multi-task evaluation:**

```bash
# Run all benchmarks
bash sbatch/eval_script.sh

# Test mode
bash sbatch/eval_script.sh --limit 10
```

---

## Running Experiments

### Local Execution

```bash
cd ~/scratch/Fast-dLLM/v2

# Single experiment
bash sbatch/eval_gsm8k.sh --limit 10

# Layer reuse sweep (sequential)
bash sbatch/eval_gsm8k_reuse_layers.sh --limit 10
```

### SLURM Array Jobs (Parallel)

```bash
cd ~/scratch/Fast-dLLM/v2

# Submit all 9 layer reuse experiments
sbatch sbatch/eval_reuse_layers_array.sbatch

# Test mode with limited samples
sbatch sbatch/eval_reuse_layers_array.sbatch --limit 10

# Throttle to 3 concurrent jobs
sbatch --array=0-8%3 sbatch/eval_reuse_layers_array.sbatch
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/reuse_layers_<job_id>_<task_id>.out

# Cancel array job
scancel <job_id>
```

---

## Config Naming Convention

Config strings follow the pattern: `<params>_n<samples>`

| Experiment | Config Example | Meaning |
|------------|----------------|---------|
| Baseline | `threshold1_nall` | threshold=1, full dataset |
| Layer Reuse | `k2_first_n100` | k=2, first 12 layers, 100 samples |
| Activation Reuse | `k1_nall` | k=1, full dataset |

---

## Expected Results

### Baseline Performance (Fast-dLLM v2 7B)

| Task | Score |
|------|-------|
| GSM8K | 83.7% |
| HumanEval | 63.4% |
| MMLU | 66.6% |
| GPQA | 31.9% |

### Layer Reuse Tradeoffs

| Config | Expected Speedup | Expected Accuracy Drop |
|--------|-----------------|----------------------|
| k=1 | ~1.1-1.2x | Minimal |
| k=2 | ~1.2-1.4x | Small |
| k=3 | ~1.3-1.5x | Moderate |

*Note: Results vary by layer subset. Middle layers typically show best speedup/accuracy tradeoff.*

---

## Adding New Experiments

1. Create a new script in `sbatch/` following the existing pattern
2. Use the output path convention: `results/<experiment>/<task>/<timestamp>_<config>/`
3. Document the experiment in this README
4. Consider creating an array job version for parallel execution
