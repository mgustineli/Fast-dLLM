# Layer Reuse NOT Applied to MMLU (Loglikelihood Tasks)

**Date:** 2026-01-20

**Status:** TODO
**Priority:** CRITICAL
**Affects:** MMLU, GPQA (all loglikelihood-based benchmarks)

## Issue

All MMLU experiments produce **identical accuracy** across different layer reuse configurations (k=1,2,3 and layer_subset=first,middle,last), despite GSM8K experiments showing expected variation.

This means **all current MMLU experimental results are invalid** - they only measure baseline performance without any layer reuse optimization applied.

## Evidence from Results

### MMLU (all identical - BUG!)
- k1_first: **66.53%**
- k2_middle: **66.53%** (SAME!)
- k3_last: **66.53%** (SAME!)

All 9 configurations show exactly the same accuracy to 10 decimal places.

### GSM8K (varying as expected)
- k1_first: **80.44%**
- k2_middle: **73.62%**
- k3_last: **67.70%**

Shows expected degradation as layer reuse increases.

## Root Cause

The layer reuse optimization is only applied to **generative tasks** (GSM8K, HumanEval), not **loglikelihood tasks** (MMLU, GPQA).

### Generative Path (Working Correctly)

**Location:** `eval.py:418-431`

```python
generated_ids = self.model.mdm_sample(
    batched_input_ids,
    tokenizer=self.tokenizer,
    # ... other params ...
    reuse_k=self.reuse_k,           # ✓ Passes layer reuse params
    layer_subset=self.layer_subset,  # ✓ Enables optimization
)
```

- Calls `mdm_sample()` which is aliased to `batch_sample()` (generation_functions.py:135)
- Sets up layer reuse via `_patch_layers_helper()` (generation_functions.py:161-163)
- Patches transformer layers to cache/reuse activations
- **Layer reuse works correctly**

### Loglikelihood Path (NOT Working)

**Location:** `eval.py:265-284`

```python
def get_logits(self, batch):
    with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
        logits = self.model(batch).logits  # ✗ Simple forward pass
        logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)
        return logits[:, : batch.shape[1]]

def get_loglikelihood(self, prefix, target):
    # ...
    perturbed_seq = self._forward_process(seq.clone(), prompt_index)
    mask_indices = perturbed_seq == self.mask_id

    logits = self.get_logits(perturbed_seq)  # ✗ No layer patching!
    # ...
```

- Calls standard `model.forward()` directly
- **Does NOT apply layer reuse patches**
- Bypasses all optimization logic entirely
- Always runs at baseline performance

## Impact

- **All 9 MMLU experiment runs produced duplicate results** (wasted compute time)
- Cannot measure accuracy/speed tradeoffs for multiple-choice benchmarks
- GPQA experiments will have the same issue
- **Current MMLU results are scientifically invalid** for comparing layer reuse strategies

## Solution

Modify `get_loglikelihood()` to apply layer reuse patches before inference, similar to how `mdm_sample()` does it.

### Option A: Wrap Loglikelihood with Layer Patching (Recommended)

```python
def get_loglikelihood(self, prefix, target):
    # Load generation module to access patching helpers
    gen_module = load_generation_module(self.experiment_name if hasattr(self, 'experiment_name') else None)

    # Setup layer reuse state
    reuse_state = {"count": 0, "enabled": True}

    # Apply layer patches
    original_forwards = gen_module._patch_layers_helper(
        self.model,
        self.reuse_k,
        self.layer_subset,
        reuse_state
    )

    try:
        with torch.amp.autocast(device_type="cuda", dtype=self.dtype):
            seq = torch.concatenate([prefix, target])[None, :]
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
            loss_acc = []

            perturbed_seq = self._forward_process(seq.clone(), prompt_index)
            mask_indices = perturbed_seq == self.mask_id

            # This now uses patched layers with reuse
            logits = self.get_logits(perturbed_seq)

            seq = torch.cat([
                seq.to(self.device),
                torch.full(
                    (seq.shape[0], self.bd_size - seq.shape[1] % self.bd_size),
                    -100,
                    dtype=torch.long,
                    device=self.device,
                ),
            ], dim=1)

            loss = F.cross_entropy(
                logits[mask_indices], seq[mask_indices], reduction="none"
            )
            loss = loss.sum()
            loss_acc.append(loss.item())

            return -sum(loss_acc) / len(loss_acc)
    finally:
        # Restore original forwards
        gen_module._unpatch_layers_helper(self.model, original_forwards)
```

### Option B: Create Unified Inference Method

Create a shared method that both `generate_until()` and `loglikelihood()` use for model inference with layer reuse support.

## Files to Modify

- `eval.py:272-303` - `get_loglikelihood()` method
- `eval.py:265-270` - `get_logits()` method (may need to pass reuse_state)
- Potentially `eval.py:136-212` - Constructor to store experiment_name if not already available

## Verification

After implementation:

1. **Test with MMLU limit_10:**
   ```bash
   bash sbatch/run.sh --task mmlu --limit 10 --force k1_first k2_middle k3_last
   ```

2. **Verify different configs produce different results:**
   - k1 should show highest accuracy (baseline)
   - k2 should show moderate degradation
   - k3 should show more degradation
   - **Results should NOT be identical**

3. **Check summary.json files:**
   ```bash
   for config in k1_first k2_middle k3_last; do
     acc=$(python3 -c "import json; print(json.load(open('results/00_baseline/mmlu_limit_10/${config}/summary.json'))['accuracy'])")
     echo "${config}: ${acc}"
   done
   ```

4. **Re-run full MMLU experiments** once verified working

## Post-Implementation

Once fixed, **all previous MMLU experiment results must be discarded** and re-run to get valid data for analysis.

## Related Files

- `generation_functions.py:25-122` - Layer patching implementation (`_patch_layers_helper`, `_unpatch_layers_helper`)
- `generation_functions.py:135-464` - `batch_sample()` showing correct layer reuse usage
- `eval.py:335-470` - `generate_until()` showing working generative path
