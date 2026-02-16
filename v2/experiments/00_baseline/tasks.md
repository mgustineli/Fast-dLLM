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

## Phase 4: Bug Fixes [DONE]

Code review (Feb 2026) found 4 bugs in `generation_functions.py` that invalidate
throughput measurements. Accuracy measurements are still valid.

- [x] Bug 1 (CRITICAL): Reuse path calls `original_forward()` to check return type — zero actual speedup
- [x] Bug 2 (MODERATE): `should_recompute` conflates block cache + layer reuse — k=1 never uses small-block fast path
- [x] Bug 3 (MODERATE): "first" subset patches 11 layers instead of 12
- [x] Bug 4 (MINOR): Layer caches never trimmed — `reuse_state["caches"]` never populated
- [x] Validate fixes with k1/k2/k3 limit-10 runs (see Post-fix Results below)

See [Bug Details](#bug-details) below for full descriptions and fixes.

## Phase 5: Re-validation [PARTIAL]
- [x] Run all 9 configs on GSM8K with `--limit 10` (see results below)
- [ ] Re-run `first` subset configs (k1/k2/k3) on latest code (`6c8822f`) — currently stale (ran on `6f96d95`)
- [ ] Investigate Bug 5 (cross-small-block stale cache) — see below
- [ ] Submit full SLURM runs once validated

---

## Results: GSM8K limit_10 (Feb 15, 2026)

All 9 configs ran via `sbatch/run_locally.sh --limit 10`. Results in `artifacts/gsm8k_limit_10/`.

**WARNING: Code version discrepancy.** Configs ran across 3 different commits:

| Group | Commit | Code State | Configs |
|-------|--------|-----------|---------|
| A | `6f96d95` (dirty) | Bug 1+3 fixed, Bug 2+4 **unfixed** | k1_first, k2_first, k3_first |
| B | `6d2a9a8` | Bugs 1-4 fixed, no reuse-disable-during-full-block | k1_middle, k1_last |
| C | `6c8822f` | All fixes (latest) | k2/k3 middle, k2/k3 last |

| Config | Accuracy | Tokens/s | Tokens | Time (s) | Commit | Notes |
|--------|----------|----------|--------|----------|--------|-------|
| k1_first | 60% | 47.52 | 3,540 | 74.5 | `6f96d95` | No reuse (k=1) |
| k1_middle | 70% | 43.37 | 3,540 | 81.6 | `6d2a9a8` | No reuse (k=1) |
| k1_last | 70% | 42.94 | 3,540 | 82.4 | `6d2a9a8` | No reuse (k=1) |
| k2_first | 70% | 36.74 | 3,092 | 84.2 | `6f96d95` | OK accuracy, slow |
| k3_first | 60% | 37.87 | 3,508 | 92.6 | `6f96d95` | OK accuracy, slow |
| k2_middle | **10%** | 82.81 | 18,493 | 223.3 | `6c8822f` | **COLLAPSED** |
| k2_last | **10%** | 83.73 | 18,717 | 223.6 | `6c8822f` | **COLLAPSED** |
| k3_middle | **0%** | 96.57 | 20,350 | 210.7 | `6c8822f` | **COLLAPSED** |
| k3_last | **0%** | 88.05 | 20,350 | 231.1 | `6c8822f` | **COLLAPSED** |

### Observations

1. **k1 configs (all subsets): 60-70% accuracy.** Expected — k=1 means `_patch_layers_helper`
   returns early without patching, so subset is irrelevant. Throughput variance (~43-48 tok/s)
   is run-to-run noise.

2. **k2/k3 with `first` subset: 60-70% accuracy, normal token counts (~3K).**
   Layer reuse works without destroying quality. But these ran on **old code** (Group A)
   where Bug 2 was unfixed — the old `should_recompute` forced full-block forwards every
   k-th step, so layer wrappers cached 32-token outputs and reused sliced versions.
   **Must re-run on latest code to confirm.**

3. **k2/k3 with `middle`/`last` subset: 0-10% accuracy, ~18-20K tokens.**
   Model fails to converge — diffusion iterations don't unmask tokens properly, so
   generation runs to max_new_tokens producing gibberish. These ran on the **latest code**
   (Group C) with all bug fixes.

4. **Throughput is misleading for collapsed configs.** The high tok/s (83-97) reflects
   many wasted iterations, not useful generation speed.

### Investigation: Middle/Last Layer Collapse

Two factors contribute to the collapse:

**Factor 1: Cross-small-block stale cache (Bug 5 — code issue)**

When a full-block forward runs (`should_recompute=True`), `reuse_state["enabled"]=False`
prevents layer wrappers from updating their cache. When the next small block starts and
a layer wrapper hits a reuse step (`step % k != 0`), it returns the **stale cache** from
the previous small block's positions (e.g., returning hidden states from positions 0-8
when processing positions 8-16). The shapes match (`[B, 8, D]` == `[B, 8, D]`), so no
fallback is triggered — the wrapper silently returns wrong-position hidden states.

This affects ALL subsets, but the impact differs by layer location.

**Factor 2: Layer sensitivity (architectural)**

- **First (layers 1-12):** Stale hidden states are corrected by layers 13-27 via
  self-attention. Early layers capture lower-level features that are more position-tolerant.
- **Middle (layers 8-19) / Last (layers 16-27):** Fewer (or no) downstream layers to
  correct stale representations. These layers are output-sensitive — stale hidden states
  produce corrupted logits that compound across diffusion iterations.

**Note on Bug 2 interaction:** In Group A (Bug 2 unfixed), the old `should_recompute`
forced full-block forwards every k-th step WITH layer wrappers enabled. This meant
wrappers cached 32-token outputs, and subsequent small-block reuses sliced them correctly
via `replace_position`. In Group C (Bug 2 fixed), full-block forwards disable wrappers
entirely, so caching only happens during 8-token small-block forwards — making the
cross-small-block stale cache issue (Bug 5) more impactful.

### Next Steps

1. **Fix Bug 5**: Clear layer caches after full-block forwards to prevent stale reuse:
   ```python
   if should_recompute:
       reuse_state["enabled"] = False
       for lc in reuse_state.get("caches", {}).values():
           lc.pop("last_output", None)
       output = self.forward(...)
       reuse_state["enabled"] = True
   ```
2. **Re-run all 9 configs on the same commit** with Bug 5 fixed
3. **If middle/last still collapse after Bug 5 fix**, the conclusion is that layer reuse
   is only viable for early layers (an inherent architectural limitation)

---

## Bug Details

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

### Bug 2 (MODERATE) [FIXED - Feb 15]: k=1 never uses block cache fast path

**File**: `generation_functions.py:280-288`
**Impact**: k=1 runs are slower than the original code, making it a bad baseline

The `should_recompute` condition conflated block cache and layer reuse into one
flag. Conditions `reuse_k <= 1` and `reuse_step % reuse_k == 0` forced full-block
forwards even when the block cache could be used.

```python
# BUGGY:
should_recompute = (
    block_past_key_values is None
    or reuse_k <= 1                    # always True for k=1!
    or (reuse_step % reuse_k == 0)
    or (x_t[:, -block_size + small_block_start_idx] == mask_id).any()
)
```

**Fix**: Restore the original block cache condition (remove conditions 2 and 3).
Layer reuse is handled by the monkey-patched wrappers via `reuse_state["count"]`.
Note: block cache and layer reuse are NOT fully independent — see
[Design Constraint](#design-constraint-block-cache-vs-layer-reuse) below.

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

### Bug 4 (MINOR) [FIXED - Feb 15]: Dead code - layer caches never trimmed

**File**: `generation_functions.py:400-408`
**Impact**: After batch trimming (finished samples removed), layer caches have
stale batch entries.

The trim code checks `reuse_state["caches"]` but layer caches lived in
closure-local dicts inside `create_wrapper()`, never registered in `reuse_state`.

**Fix**: Register each `layer_cache` dict into `reuse_state["caches"]` during
`create_wrapper()`, so the existing trim code works.

---

## Design Constraint: Block Cache vs Layer Reuse

Layer reuse and block cache are **not fully independent**. Full-block forwards
(32 tokens) BUILD `block_past_key_values` — each layer's attention must run to
populate its block cache entry via `block_past_key_values.update()`. If a patched
layer is skipped (returns cached hidden states), its block cache entry is never
created, causing a `NoneType` crash when the subsequent small-block forward
tries to read it.

**Constraint**: Layer reuse must be disabled during full-block forwards.

This means layer reuse only activates on the **small-block fast path** (8 tokens),
which reads from an already-built block cache. The expensive full-block forwards
always run all layers at full cost.

```
should_recompute (original block cache logic)
  |
  +-- YES: full block (32 tokens) — BUILDS block_past_key_values
  |   Layer reuse DISABLED (all layers must run to populate block cache)
  |
  +-- NO: small block (8 tokens) — READS block_past_key_values
      Layer reuse ENABLED (skipped layers return cached hidden states)
```

**Implication for throughput**: The speedup opportunity from layer reuse is
limited to small-block iterations. The full-block forward (the most expensive
step in each denoising cycle) cannot benefit from layer skipping. This partly
explains why k=2 and k=3 show minimal throughput improvement over k=1.

---

## Known Issues
- Layer reuse does NOT apply to loglikelihood tasks (MMLU, GPQA) - see docs/experiments/00-layer-reuse-loglikelihood.md
- GPQA dataset requires HuggingFace authentication - see docs/experiments/01-gpqa-authentication.md
- Throughput metrics missing for loglikelihood tasks - see docs/experiments/02-missing-throughput-tracking.md
- `transformers>=5.0.0` breaks model loading (KeyError: 'default' in ROPE_INIT_FUNCTIONS) - pinned to `<5.0.0` in pyproject.toml
