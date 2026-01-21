# Missing Throughput Tracking for Loglikelihood Tasks

**Date:** 2026-01-20

**Status:** TODO
**Priority:** High
**Affects:** MMLU, GPQA (all loglikelihood-based benchmarks)

## Issue

Throughput metrics (tokens/second, total time, tokens generated) are only logged for generative tasks (GSM8K, HumanEval, etc.) but **not for loglikelihood-based tasks** (MMLU, GPQA, etc.).

## Root Cause

- `generate_until()` method (eval.py:335-470) has full throughput tracking
- `loglikelihood()` method (eval.py:305-333) lacks timing and token counting
- This results in empty `"throughput": {}` fields in summary.json for MMLU experiments

## Impact

- Cannot measure inference speed improvements for multiple-choice benchmarks
- Incomplete performance comparison across different task types
- Missing critical data for speedup analysis on MMLU, which is a key benchmark

## Solution

Add throughput tracking to `loglikelihood()` method similar to `generate_until()`:

1. Add start/end time tracking around the loglikelihood computation loop
2. Count tokens processed (prefix + target lengths)
3. Store metrics in class variable for retrieval by main script
4. Ensure `show_speed` parameter is respected

### Implementation Outline

```python
def loglikelihood(self, requests):
    # Initialize tracking
    start_time = time.time()
    num_tokens = 0

    # ... existing tokenization code ...

    out = []
    with torch.no_grad():
        for elem in tqdm(ds, desc="Computing likelihood..."):
            prefix = elem["prefix"]
            target = elem["target"]

            # Track tokens processed
            if self.show_speed:
                num_tokens += len(prefix) + len(target)

            ll = self.get_loglikelihood(prefix, target)
            out.append((ll, 0.0))

    # Store throughput metrics
    if self.show_speed:
        end_time = time.time()
        total_time = end_time - start_time
        tokens_per_s = float(num_tokens) / total_time
        Fast_dLLM_v2EvalHarness.throughput_metrics = {
            "tokens_processed": int(num_tokens),
            "total_time_s": round(total_time, 2),
            "tokens_per_s": round(tokens_per_s, 2),
        }

    torch.cuda.empty_cache()
    return out
```

## Files to Modify

- `eval.py:305-333` - `loglikelihood()` method

## Verification

After implementation:
1. Run MMLU with `--limit 10`
2. Check `summary.json` contains non-empty `throughput` field
3. Verify metrics are reasonable (tokens processed, time, tokens/s)
