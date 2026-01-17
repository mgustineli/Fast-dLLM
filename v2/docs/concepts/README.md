# Concepts

Research ideas, hypotheses, and directions being explored for Fast-dLLM v2 acceleration.

## Concept Map

```
                    ┌─────────────────────────┐
                    │    Research Thesis      │
                    │  Accelerate diffusion   │
                    │  LLM inference          │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
 ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
 │   Caching   │        │   Layer     │        │   Token     │
 │ Strategies  │        │ Efficiency  │        │ Strategies  │
 └─────────────┘        └─────────────┘        └─────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   kv-caching             layer-reuse              trim-logic
   block-diffusion        activation-reuse    confidence-decoding
```

## Index

| Concept | Status | Description |
|---------|--------|-------------|
| [block-diffusion](block-diffusion.md) | mature | Block-wise parallel decoding with hierarchical caching |
| [kv-caching](kv-caching.md) | mature | Prefix KV cache and dual cache optimization |
| [layer-reuse](layer-reuse.md) | developing | Skip transformer layers during diffusion steps |
| [trim-logic](trim-logic.md) | developing | Token-level Reuse Minimization |
| [activation-reuse](activation-reuse.md) | nascent | Reuse intermediate activations across steps |
| [confidence-decoding](confidence-decoding.md) | mature | Threshold-based parallel token unmasking |

## Status Definitions

- **nascent**: Initial idea, minimal exploration
- **developing**: Active exploration, hypotheses forming
- **mature**: Ready for experiments, clear hypotheses
- **archived**: Explored and concluded

## Core Architecture Concepts

### Block Diffusion
The foundational mechanism: autoregressive at block level, parallel within blocks.
- Block-level KV cache stores historical context
- Sub-block cache enables parallel generation within partially decoded blocks
- Token shift mechanism retains autoregressive characteristics

### KV Caching
Two-level caching strategy:
- **Prefix cache** (`use_cache=True`): Reuse K/V for clean prefix tokens
- **Dual cache** (`dual_cache=True`): Also cache masked suffix tokens

## Active Research Concepts

### Layer Reuse (k parameter)
Skip k transformer layers during diffusion refinement steps.
- Hypothesis: Later layers refine but don't fundamentally change representations
- Trade-off: Speed vs accuracy
- Parameter: `k=1,2,3` in model_args

### TRIM Logic
Token-level Reuse Minimization:
- Identify which tokens need full computation vs reuse
- Reduce redundant computation for "stable" tokens

## Adding a New Concept

Use the template at [`../_templates/concept.md`](../_templates/concept.md).

```bash
cp docs/_templates/concept.md docs/concepts/my-new-concept.md
```

Then update this README to include the new concept in the index.
