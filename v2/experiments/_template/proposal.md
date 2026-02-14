# Proposal: [Experiment Name]

**Status**: proposed
**Created**: YYYY-MM-DD
**Author**: [name]

## Hypothesis

[What we believe to be true and want to test. Be specific and falsifiable.]

## Background

[Why this matters. What concept or observation led to this experiment?]

## Method

### Approach

[High-level description of what we're going to do.]

### Variables

- **Independent**: [what we're changing, e.g., k=1,2,3 for layer reuse]
- **Dependent**: [what we're measuring, e.g., accuracy, throughput]
- **Controlled**: [what we're holding constant, e.g., batch_size=1]

### Setup

| Component | Details |
|-----------|---------|
| Model | Efficient-Large-Model/Fast_dLLM_v2_7B |
| Benchmarks | GSM8K, MMLU, ... |
| Compute | RTX 6000, 64GB RAM |

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Task-specific accuracy |
| Throughput | Tokens per second |

### Baseline

| Baseline | Accuracy | Throughput |
|----------|----------|------------|
| k1_first | X% | Y tok/s |

### Success Criteria

- **Confirm if**: [specific threshold or observation]
- **Reject if**: [specific threshold or observation]

## Limitations

- [Assumption or constraint 1]
- [Assumption or constraint 2]
