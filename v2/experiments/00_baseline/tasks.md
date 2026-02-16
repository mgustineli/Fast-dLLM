# Tasks: 00_baseline

## Phase 0: Planning [DONE]
- [x] Define 3x3 experiment grid (k x subset)
- [x] Create generation_functions.py with layer reuse logic
- [x] Set up smart runner (run.sh) and job template (_job.sh)

## Phase 1: GSM8K Full Runs [DONE]
- [x] Run all 9 configs on GSM8K (full, 1319 samples)
- [x] Verify all 9 summary.json files in artifacts/gsm8k/
- [x] Record accuracy and throughput in results.md

## Phase 2: MMLU Full Runs [DONE]
- [x] Run all 9 configs on MMLU (full)
- [x] Verify all 9 summary.json files in artifacts/mmlu/
- [x] Discover: layer reuse has no effect on loglikelihood tasks
- [x] Document finding in results.md

## Phase 3: Remaining Tasks [PARTIAL]
- [x] Run MMLU limit_10 (9 configs) - completed
- [x] Run Minerva Math limit_10 (9 configs) - completed
- [x] Run IFEval limit_5 (1 config) - completed
- [ ] Run full Minerva Math (all 9 configs)
- [ ] Run full IFEval (all 9 configs)
- [ ] Investigate k1_first throughput anomaly (22.75 vs ~50 tok/s)

---

## Phase 4: Bug Fixes [IN PROGRESS]

Code review (Feb 2026) found 4 bugs in `generation_functions.py` that invalidate
throughput measurements. Accuracy measurements are still valid. Bugs are listed
in priority order.

### Bug 1 (CRITICAL) [FIXED - Feb 15]: Reuse path calls `original_forward()` to check return type

**File**: `generation_functions.py:104-108`
**Impact**: Throughput measurements are INVALID - no computation is actually saved

On the reuse path (where we skip computation and return cached values), the code
calls `original_forward(*args, **kwargs)` just to check `isinstance(..., tuple)`.
This means every "reuse" step runs the full layer computation anyway, then throws
the result away. There is zero actual speedup from layer reuse.

```python
# BUGGY (line 104-108):
return (
    (output_tensor,)
    if isinstance(original_forward(*args, **kwargs), tuple)  # runs full forward!
    else output_tensor
)
```

**Fix**: Cache the return type alongside the output on recompute steps:
```python
# On recompute:
layer_cache["is_tuple"] = isinstance(output, tuple)

# On reuse:
if layer_cache.get("is_tuple", False):
    return (output_tensor,)
else:
    return output_tensor
```

**Verification**: k=2 and k=3 should show measurably higher tokens/s than k=1
after the fix.

### Bug 2 (MODERATE) [FIXED - Feb 15]: k=1 never uses block cache fast path

**File**: `generation_functions.py:280-288`
**Impact**: k=1 runs are slower than the original code, making it a bad baseline

The `should_recompute` condition includes `reuse_k <= 1`, which is always True
for k=1. This forces full-block forwards on every step, bypassing the original's
small-block optimization (block cache). k=1 was supposed to be the "no reuse"
control, equivalent to the original, but it's actually slower.

```python
# BUGGY:
should_recompute = (
    block_past_key_values is None
    or reuse_k <= 1                    # always True for k=1!
    or (reuse_step % reuse_k == 0)
    or (x_t[:, -block_size + small_block_start_idx] == mask_id).any()
)
```

The original code's condition is just:
```python
if block_past_key_values is None or (x_t[:, ...] == mask_id).any():
```

**Fix**: When k=1, fall through to the original logic (no extra conditions).

**Design question**: The block-level reuse (original's small-block optimization)
and layer-level reuse (our monkey-patching) are two separate mechanisms that are
currently conflated in this single `should_recompute` check. They should probably
be separated. See "Design: Separating Reuse Mechanisms" below.

### Bug 3 (MODERATE) [FIXED - Feb 15]: "first" subset patches 11 layers, not 12

**File**: `generation_functions.py:33`
**Impact**: Uneven comparison - "first" applies reuse to 11 layers while
"middle" and "last" apply to 12

```python
# BUGGY:
target_indices = list(range(1, min(n, subset_size)))  # range(1,12) = 11 layers

# FIX:
target_indices = list(range(1, subset_size + 1))  # range(1,13) = 12 layers
```

Layer 0 is skipped (it allocates cache memory), so we should extend to layer 12
to still get 12 target layers.

### Bug 4 (MINOR) [FIXED - Feb 15]: Dead code - layer caches never trimmed

**File**: `generation_functions.py:400-408`
**Impact**: After batch trimming (finished samples removed), layer caches have
stale batch entries. Currently masked by Bug 1 (caches are always overwritten).

```python
# DEAD CODE - "caches" key is never added to reuse_state:
if "caches" in reuse_state:
    for layer_id, layer_cache in reuse_state["caches"].items():
        ...
```

Layer caches live in closure-local `layer_cache` dicts inside `create_wrapper()`,
not in `reuse_state["caches"]`. Once Bug 1 is fixed, this could cause incorrect
reuse for remaining batch samples after some finish early.

**Fix**: Either store layer caches in `reuse_state["caches"]` so trim code works,
or restructure to avoid the issue.

---

## Design: Separating Reuse Mechanisms

There are two independent reuse mechanisms currently conflated:

1. **Block-level reuse** (original Fast-dLLM feature):
   When a small-block position has been unmasked, forward only the small slice
   using `block_past_key_values` instead of the full block. This is the original
   code's optimization, controlled by `use_block_cache=True`.

2. **Layer-level reuse** (our addition for the experiment):
   Within a forward pass, skip computation for specific layers and return
   cached hidden states from a previous step. Controlled by `reuse_k` and
   `layer_subset`.

Currently both decisions are merged into one `should_recompute` flag. They should
be separated so that:
- Block-level reuse works exactly as the original (always, for all k values)
- Layer-level reuse is an independent toggle on top

This would make k=1 truly equivalent to the original, and make the experiment
cleaner: we're measuring only the effect of layer reuse, not also disabling
the block cache optimization.

---

## Pre-fix Baseline (Feb 14, 2026)

Ran k1_first locally with `--limit 10` on GSM8K before fixing any bugs.
This establishes a "before" baseline for throughput comparisons.

| Config | Accuracy | Tokens/s | GPU | Notes |
|--------|----------|----------|-----|-------|
| k1_first (limit 10) | 60.0% | 47.63 | Quadro RTX 6000 | Pre-fix, buggy code |

## Post-fix Results (Feb 15, 2026)

Fixed Bug 1 (cache `is_tuple` flag instead of calling `original_forward()` on reuse
path) and Bug 3 (`range(1, subset_size + 1)` so "first" patches 12 layers).

| Config | Accuracy | Tokens/s | GPU | Notes |
|--------|----------|----------|-----|-------|
| k1_first (limit 10) | 60.0% | 47.52 | Quadro RTX 6000 | Control - no reuse (expected same) |
| k2_first (limit 10) | 70.0% | 36.74 | Quadro RTX 6000 | Slower than k1 |
| k3_first (limit 10) | 60.0% | 37.87 | Quadro RTX 6000 | Slower than k1 |

**Observation: Layer reuse does not improve throughput (and may hurt it).**

k2 and k3 are ~20-23% *slower* than k1, despite skipping layer computation on
reuse steps. Likely causes:

1. **Python monkey-patching overhead**: The wrapper closure runs on every forward
   call for 12 layers. At batch_size=1, the per-call CPU overhead (dict lookups,
   condition checks, tensor slicing) outweighs the GPU compute saved by skipping
   a single layer forward.
2. **Bug 2 still present**: The `should_recompute` flag in `batch_sample()` forces
   full-block forwards on recompute steps (every k-th step), disrupting the block
   cache optimization that the original code relies on for speed.
3. **Batch size 1**: GPU utilization is already low; skipping layers saves minimal
   wall-clock time since the GPU is not saturated.

**Decision**: Focus on accuracy for now. Throughput optimization is a separate
concern that likely requires a different approach (e.g., native CUDA-level layer
skipping, or larger batch sizes to amortize overhead). Bug 2 fix may also help
by restoring the block cache fast path.

---

## Next Steps

1. ~~**Fix Bug 1** (CRITICAL)~~ - DONE (Feb 15)
2. ~~**Fix Bug 3** (quick)~~ - DONE (Feb 15)
3. ~~**Validate fixes** with k1/k2/k3 limit-10 runs~~ - DONE (Feb 15, see above)
4. ~~**Fix Bug 2** (separate block cache from layer reuse)~~ - DONE (Feb 15).
   Simpler than expected: just removed the two extra conditions from
   `should_recompute`, restoring the original block cache logic. No refactor needed.
5. ~~**Fix Bug 4** (wire layer caches into `reuse_state["caches"]`)~~ - DONE (Feb 15).
   Registered each `layer_cache` dict into `reuse_state["caches"]` during
   `create_wrapper()`, so the existing trim code at line 395 now works.
6. **Re-run all 9 configs** on GSM8K with `--limit 10` to validate, then
   submit full SLURM runs.

---

## Known Issues
- Layer reuse does NOT apply to loglikelihood tasks (MMLU, GPQA) - see docs/experiments/00-layer-reuse-loglikelihood.md
- GPQA dataset requires HuggingFace authentication - see docs/experiments/01-gpqa-authentication.md
- Throughput metrics missing for loglikelihood tasks - see docs/experiments/02-missing-throughput-tracking.md
- `transformers>=5.0.0` breaks model loading (KeyError: 'default' in ROPE_INIT_FUNCTIONS) - pinned to `<5.0.0` in pyproject.toml
