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
| first (layers 1-11) | k1_first | k2_first | k3_first |
| middle (layers 12-23) | k1_middle | k2_middle | k3_middle |
| last (layers 24-35) | k1_last | k2_last | k3_last |

Note: k=1 means every step is a full computation (no reuse), so all k=1 configs should produce identical accuracy.

## Implementation Details

The experiment code lives in `generation_functions.py`. It has three major parts:

1. **Layer reuse patching system** (lines 17-131) — our experiment's addition
2. **`batch_sample()` generation loop** (lines 136-464) — the main engine
3. **`mdm_sample_with_visualization()`** (lines 466-667) — Gradio demo variant (not used in experiments)

### Part 1: Layer Reuse Patching (lines 17-131)

This is the experiment's addition on top of original Fast-dLLM. The idea: skip
computation in certain transformer layers on some diffusion steps, returning
cached hidden states instead.

**`_patch_layers_helper()` (line 17)** monkey-patches `.forward()` on 12 layers:

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

   The slicing logic (lines 86-105) uses `replace_position` to know which 8-token
   slice of the cached 32-token output to return. Full-block forwards produce
   `[B, 32, D]` but small-block forwards expect `[B, 8, D]`.

**`_unpatch_layers_helper()` (line 127)** restores original `.forward()` methods
after generation is done. Called at line 460.

### Part 2: `batch_sample()` (lines 136-464)

The main generation function. Implements **block diffusion** — the core Fast-dLLM
algorithm. Instead of generating one token at a time (autoregressive), it generates
a **block of 32 tokens at once** by iterative denoising (mask -> unmask).

**Key parameters:**

| Param | Typical value | Meaning |
|-------|--------------|---------|
| `block_size` | 32 | Tokens generated per block |
| `small_block_size` | 8 | Sub-block for parallel unmasking |
| `threshold` | 0.95 | Confidence threshold for unmasking multiple tokens |
| `reuse_k` | 1, 2, or 3 | Layer reuse frequency (our experiment) |
| `use_block_cache` | True | Enable the block-level KV cache optimization |

**High-level structure:**

```
SETUP (lines 155-194)
  - Patch layers for reuse
  - Prefill: process prompt through model, build KV cache

FOR each block (line 196):              <- generates 32 tokens per iteration
  |
  +-- INIT: append 32 [MASK] tokens to the sequence
  |
  +-- WHILE masks remain (line 216):    <- diffusion denoising loop
  |   |
  |   +-- If ALL masks cleared -> FINALIZE block (lines 218-248)
  |   |   Run one more forward with update_past_key_values=True
  |   |   to commit this block's KV to the prefix cache, then
  |   |   predict the first token of the NEXT block.
  |   |
  |   +-- SMALL BLOCK LOOP (lines 256-360):    <- the inner engine
  |       FOR each 8-token sub-block:
  |         WHILE sub-block has masks:
  |           1. Forward pass -> logits
  |           2. Sample tokens from logits
  |           3. Unmask positions above confidence threshold
  |           4. Always unmask at least the highest-confidence position
  |
  +-- COPY results back to input_ids (lines 363-378)
  |
  +-- TRIM finished samples from batch (lines 380-450)

TEARDOWN: unpatch layers (line 460)
```

**The two optimization levels:**

*Prefix KV cache* (`past_key_values`, line 185): Standard transformer KV cache
for the prompt + all previously finalized blocks. Grows as blocks are finalized.
Built by the `update_past_key_values=True` call at lines 236-242.

*Block cache* (`block_past_key_values`, line 213): Within a single block's
denoising loop, caches the KV from the full-block forward so subsequent small-block
forwards (8 tokens) can reuse it instead of recomputing all 32 tokens. This is the
`should_recompute` decision at line 280:

```
should_recompute?
  +-- YES (full block, 32 tokens): first iteration, or position still has masks
  |   -> self.forward(x_t[:, -block_size:], ...)     <- all 32 tokens
  |   -> saves block_past_key_values
  |
  +-- NO (small block, 8 tokens): position unmasked, cache available
      -> self.forward(x_t[:, start:end], ..., block_past_key_values=...)
      -> much cheaper
```

**Unmasking logic (lines 342-360)** — the diffusion "denoising" step:

1. Get logits for masked positions in the current small block
2. Sample token predictions and their probabilities
3. If probability > `threshold` (0.95), unmask that position
4. **Always** unmask the single highest-probability position (guaranteed progress)
5. If any unmasked token is `stop_token`, mark that sample as finished

**Trim logic (lines 380-450)** — when a batch sample finishes (hits stop token),
remove it from all tensors to avoid wasting compute:

1. `input_ids`, `x_t`, `seq_len`, etc. — standard tensors
2. `reuse_state["caches"]` — our layer reuse caches
3. `past_key_values` — the prefix KV cache (handles multiple HF cache formats)

### How the Two Reuse Mechanisms Interact

After the Bug 2 fix, block cache and layer reuse are fully independent:

```
                    +----------------------------------+
                    |     should_recompute (line 280)   |
                    |  Block cache: what INPUT shape?   |
                    +-----------------+----------------+
                    | YES: 32 tokens  | NO: 8 tokens   |
                    +--------+--------+-------+--------+
                             |                |
                    +--------v----------------v--------+
                    |        self.forward(...)          |
                    |  Runs through all 28 layers       |
                    |                                   |
                    |  For 12 PATCHED layers:           |
                    |    reuse_state["count"] % k == 0? |
                    |    +-- YES: compute + cache output|
                    |    +-- NO:  return cached output  |
                    |                                   |
                    |  For 16 UNPATCHED layers:         |
                    |    always compute normally         |
                    +-----------------------------------+
```

- **Block cache** controls the input size (32 vs 8 tokens)
- **Layer reuse** controls which layers actually compute inside the forward pass
- The two decisions are **orthogonal** — every combination is valid

### Part 3: `mdm_sample_with_visualization()` (lines 466-667)

Simpler variant of `batch_sample` for the Gradio demo. Same block diffusion
algorithm but: no batch support, no block cache, no layer reuse, `yield`s
intermediate states for live visualization. Not relevant to the experiment.

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
