# Results: [Experiment Name]

**Status**: completed
**Date**: YYYY-MM-DD
**Proposal**: [link to proposal]

## Summary

[1-2 sentence summary of what was tested and the key finding.]

## Results

### Main Results

| Configuration | Accuracy | Throughput | Speedup |
|--------------|----------|------------|---------|
| Baseline | X% | Y tok/s | 1.0x |
| Variant A | X% | Y tok/s | Z.Zx |
| Variant B | X% | Y tok/s | Z.Zx |

### Detailed Metrics

[Additional tables or figures as needed.]

## Analysis

### Key Findings

1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Unexpected Observations

- [Observation 1]
- [Observation 2]

## Conclusion

**Hypothesis [confirmed/rejected/partially confirmed]**

[Explanation of the conclusion and what it means for the project.]

## Next Steps

- [Follow-up action 1]
- [Follow-up action 2]

## Artifacts

| Artifact | Location |
|----------|----------|
| Raw results | `results/TIMESTAMP/` |
| Logs | `logs/` |
| Scripts | [link to scripts used] |

## Commands Used

```bash
# Document the exact commands used for reproducibility
bash eval_gsm8k.sh --limit 100

accelerate launch eval.py \
    --tasks gsm8k \
    --model fast_dllm_v2 \
    --model_args model_path=...
```
