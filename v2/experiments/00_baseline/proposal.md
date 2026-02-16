# Proposal: 00_baseline - Layer Reuse Baseline

**Status**: completed
**Created**: 2026-01-18
**Author**: Murilo

## Hypothesis

Layer reuse (skipping transformer layer computation every k-th diffusion step by reusing cached outputs) trades accuracy for throughput in block-diffusion LLM generation. The tradeoff depends on which layers are reused (first/middle/last) and the reuse frequency (k=1,2,3).

## Background

Fast-dLLM v2 uses block diffusion to generate text in parallel. Each diffusion step runs the full transformer stack. Layer reuse skips a subset of 12 layers on non-recompute steps, using cached hidden states instead. This reduces FLOPs per step but may degrade generation quality.

## Method

### Approach

Run all 9 configurations (3x3 grid) on generative and loglikelihood benchmarks, measuring accuracy and throughput.

### Variables

- **Independent**: reuse_k (1, 2, 3) x layer_subset (first, middle, last)
- **Dependent**: accuracy (task-specific), throughput (tokens/s)
- **Controlled**: model (Fast_dLLM_v2_7B), batch_size=1, use_block_cache=True, threshold=0.95

### Setup

| Component | Details |
|-----------|---------|
| Model | Efficient-Large-Model/Fast_dLLM_v2_7B |
| Benchmarks | GSM8K, MMLU, Minerva Math, IFEval |
| Compute | RTX 6000 (Turing, SM 7.5), 64GB RAM |
| GPU Dtype | float16 (auto-detected) |

### Experiment Matrix (9 configurations)

| Subset | k=1 (baseline) | k=2 | k=3 |
|--------|----------------|-----|-----|
| first (layers 1-12) | k1_first | k2_first | k3_first |
| middle (layers 12-23) | k1_middle | k2_middle | k3_middle |
| last (layers 24-35) | k1_last | k2_last | k3_last |

Note: k=1 means every step is a full computation (no reuse), so all k=1 configs should produce identical accuracy.

## Implementation Details

The experiment code lives in `generation_functions.py`, forked from the
[upstream Fast-dLLM](https://github.com/NVlabs/Fast-dLLM/blob/main/v2/generation_functions.py).
The original file contains the block diffusion generation loop (`batch_sample`)
and a Gradio visualization function. Our experiment adds the layer reuse system
on top and hardens the batch trim logic.

### What the Original Code Does

The original `batch_sample()` implements **block diffusion** — the core Fast-dLLM
algorithm. Instead of generating one token at a time (autoregressive), it generates
a **block of 32 tokens at once** by iterative denoising (mask -> unmask).

**Key parameters (original):**

| Param | Typical value | Meaning |
|-------|--------------|---------|
| `block_size` | 32 | Tokens generated per block |
| `small_block_size` | 8 | Sub-block for parallel unmasking |
| `threshold` | 0.95 | Confidence threshold for unmasking multiple tokens |
| `use_block_cache` | True | Enable the block-level KV cache optimization |

**High-level structure (original):**

```
PREFILL: process prompt, build KV cache

FOR each block:                         <- generates 32 tokens per iteration
  |
  +-- INIT: append 32 [MASK] tokens to the sequence
  |
  +-- WHILE masks remain:               <- diffusion denoising loop
  |   |
  |   +-- If ALL masks cleared -> FINALIZE block
  |   |   Run forward with update_past_key_values=True to commit
  |   |   this block's KV to the prefix cache, then predict the
  |   |   first token of the NEXT block.
  |   |
  |   +-- SMALL BLOCK LOOP:             <- the inner engine
  |       FOR each 8-token sub-block:
  |         WHILE sub-block has masks:
  |           1. Forward pass -> logits
  |           2. Sample tokens from logits
  |           3. Unmask positions above confidence threshold
  |           4. Always unmask at least the highest-confidence position
  |
  +-- COPY results back to input_ids
  |
  +-- TRIM finished samples from batch
```

**The two original optimization levels:**

*Prefix KV cache* (`past_key_values`): Standard transformer KV cache for the
prompt + all previously finalized blocks. Grows as blocks are finalized via the
`update_past_key_values=True` call in the finalize-block step.

*Block cache* (`block_past_key_values`): Within a single block's denoising loop,
caches the KV from the full-block forward so subsequent small-block forwards
(8 tokens) can reuse it instead of recomputing all 32 tokens:

```
block_past_key_values is None or position still has masks?
  +-- YES (full block, 32 tokens):
  |   -> self.forward(x_t[:, -block_size:], ...)     <- all 32 tokens
  |   -> saves block_past_key_values
  |
  +-- NO (small block, 8 tokens):
      -> self.forward(x_t[:, start:end], ..., block_past_key_values=...)
      -> much cheaper
```

**Unmasking logic** — the diffusion "denoising" step:

1. Get logits for masked positions in the current small block
2. Sample token predictions and their probabilities
3. If probability > `threshold` (0.95), unmask that position
4. **Always** unmask the single highest-probability position (guaranteed progress)
5. If any unmasked token is `stop_token`, mark that sample as finished

### What We Changed for the Experiment

Changes fall into three categories: new layer reuse infrastructure, modifications
to the generation loop, and hardened batch trim logic.

#### 1. New: Layer Reuse Patching System (lines 17-131)

Entirely new code. Two global helper functions that monkey-patch transformer
layer `.forward()` methods to cache and reuse hidden-state outputs.

**`_patch_layers_helper()` (line 17):**

1. **Pick 12 layers** based on `subset_name` (lines 31-38):
   - `"first"` — layers 1-12 (skip layer 0 — it allocates KV cache memory)
   - `"middle"` — layers centered around n/2
   - `"last"` — the final 12 layers

2. **Create a wrapper closure** for each layer (line 46). Each wrapper owns a
   private `layer_cache = {}` dict (also registered in `reuse_state["caches"]`
   so the trim logic can access it).

3. **Wrapper decision** (lines 51-113) on every forward call:
   - If `reuse_state["enabled"]` is False — run original forward, no caching
   - If `step % reuse_k == 0` — **recompute**: run original forward, cache the
     output tensor + whether it returns a tuple
   - Otherwise — **reuse**: return the cached tensor from the last recompute step,
     sliced to match the current input shape

   The slicing logic (lines 86-105) handles the shape mismatch between full-block
   forwards (`[B, 32, D]`) and small-block forwards (`[B, 8, D]`), using
   `replace_position` to extract the correct slice from the cached output.

**`_unpatch_layers_helper()` (line 127):** Restores original `.forward()` methods
after generation is done.

#### 2. Modified: `batch_sample()` Generation Loop

Added `reuse_k` and `layer_subset` parameters to the method signature (lines
146-147). Within the loop, the following changes coordinate the layer reuse
wrappers with the generation process:

- **Setup** (lines 158-165): Initialize `reuse_state` dict and apply layer
  patches before generation begins.
- **Disable reuse during finalize-block** (line 234): Set
  `reuse_state["enabled"] = False` before the correction forward pass that
  commits the block to the prefix KV cache. This ensures correct computation.
- **Enable reuse during small-block loop** (line 254): Set
  `reuse_state["enabled"] = True` for the speculative denoising iterations.
- **Step counter sync** (line 274): Set `reuse_state["count"] = reuse_step`
  before each forward call so the patched layers know which diffusion step
  they're in.
- **Step counter increment** (line 323): `reuse_step += 1` after each
  small-block iteration.
- **Teardown** (line 460): Unpatch layers after generation completes.

The block cache decision (`should_recompute`, line 280) preserves the original
logic exactly. However, block cache and layer reuse are **not fully independent**:
full-block forwards that BUILD `block_past_key_values` require every layer's
attention to run (to populate its block cache entry). Layer reuse is therefore
disabled during full-block forwards and only active during small-block forwards:

```
should_recompute (line 280) — original block cache logic
  |
  +-- YES: full block (32 tokens) — BUILDS block_past_key_values
  |   Layer reuse DISABLED (all layers must compute to populate block cache)
  |
  +-- NO: small block (8 tokens) — READS block_past_key_values
      Layer reuse ENABLED:
        For 12 PATCHED layers:
          reuse_state["count"] % k == 0?
          +-- YES: compute + cache output
          +-- NO:  return cached output
        For 16 UNPATCHED layers:
          always compute normally
```

#### 3. Hardened: Batch Trim Logic (lines 380-450)

The original trim code assumed a single KV cache format (`DynamicCache` with
`key_cache`/`value_cache` lists). Our version handles three formats to prevent
crashes across different HuggingFace / transformers versions:

- **Case A** (lines 412-419): Standard tuple/list of tuples (legacy HF)
- **Case B** (lines 421-431): `DynamicCache` with `key_cache`/`value_cache`
  (original format, preserved)
- **Case C** (lines 433-447): Modern HF `caches` list format

Additionally, we added **layer cache trimming** (lines 398-406): when finished
samples are removed from the batch, the cached hidden states in each patched
layer must also be sliced along the batch dimension to stay consistent.

#### 4. Minor: Import and Decorator Changes

- Removed unused imports (`typing`, `transformers.utils.logging`)
- Replaced `from transformers.utils import auto_docstring` with a local no-op
  identity function (line 12) to make the file self-contained

### `mdm_sample_with_visualization()` (lines 466-667)

Unchanged from upstream. Simpler variant of `batch_sample` for the Gradio demo:
no batch support, no block cache, no layer reuse. Not relevant to the experiment.

## Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | exact_match,strict-match for GSM8K; acc for MMLU |
| Throughput | Tokens per second (generation tasks only) |

### Baseline

k1_first is the true baseline (k=1 = no layer reuse).

### Success Criteria

- **Confirm if**: k=2 or k=3 shows >1.2x throughput improvement with <5% accuracy drop
- **Reject if**: all reuse configs show negligible speedup or catastrophic accuracy loss

## Limitations

- Layer reuse only applies to generative tasks (GSM8K, Minerva Math), not loglikelihood tasks (MMLU)
- MMLU results are identical across all 9 configs (layer reuse has no effect on loglikelihood evaluation)
- Throughput measurements include model loading overhead
