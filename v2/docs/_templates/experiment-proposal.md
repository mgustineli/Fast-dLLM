# Proposal: [Experiment Name]

**Status**: [proposed | approved | in-progress | completed | abandoned]
**Created**: YYYY-MM-DD
**Author**: [name]

## Hypothesis

[What we believe to be true and want to test. Be specific and falsifiable.]

## Background

[Why this matters. What concept or observation led to this experiment?]

## Method

### Approach

[High-level description of what we're going to do.]

### Setup

| Component | Details |
|-----------|---------|
| Model | [model path, parameters] |
| Data | [benchmark, splits] |
| Compute | [GPU, memory requirements] |
| Dependencies | [libraries, versions] |

### Procedure

1. [Step 1]
2. [Step 2]
3. [Step 3]

### Variables

- **Independent**: [what we're changing, e.g., k=1,2,3 for layer reuse]
- **Dependent**: [what we're measuring, e.g., accuracy, throughput]
- **Controlled**: [what we're holding constant, e.g., batch_size=1]

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Task-specific accuracy (GSM8K, HumanEval, etc.) |
| Throughput | Tokens per second |
| Latency | Time to first token, total generation time |

### Baseline

[What we're comparing against. Include specific numbers if known.]

| Baseline | Accuracy | Throughput |
|----------|----------|------------|
| threshold=1.0 | X% | Y tok/s |

### Success Criteria

- **Confirm if**: [specific threshold or observation]
- **Reject if**: [specific threshold or observation]

## Commands

```bash
# Baseline run
bash eval_gsm8k.sh --limit 100

# Experiment run
accelerate launch eval.py \
    --tasks gsm8k \
    --model fast_dllm_v2 \
    --model_args model_path=...,<experiment_param>=<value>
```

## Limitations

- [Assumption or constraint 1]
- [Assumption or constraint 2]
