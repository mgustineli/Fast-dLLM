# Documentation

Shared context and documentation for the Fast-dLLM v2 research project.

> **Research Thesis**: Fast-dLLM v2 is a block diffusion language model (dLLM) that efficiently adapts pretrained autoregressive models into parallel text generation systems, requiring only ~1B tokens of fine-tuning (500x reduction vs full-attention diffusion LLMs).

---

## Quick Start

**New here?** Start with:
1. [STATUS.md](STATUS.md) - What works now, what we're trying
2. [concepts/](concepts/) - Research ideas we're exploring
3. [operations.md](operations.md) - How to run experiments on PACE

---

## Directory Structure

```
docs/
├── STATUS.md           # Current state summary (start here)
├── operations.md       # PACE cluster operational patterns
├── concepts/           # Research ideas and hypotheses
│   ├── README.md       # Concept map and index
│   └── *.md            # Individual concepts
├── experiments/        # Experiment proposals and results
│   └── NNN-name/       # Each experiment gets a directory
├── reference/
│   └── pace-cluster.md # PACE cluster quick reference
└── _templates/         # Document templates
    ├── concept.md
    ├── experiment-proposal.md
    ├── experiment-result.md
    └── reference.md
```

---

## Concepts (Research Ideas)

| Concept | Description |
|---------|-------------|
| [layer-reuse](concepts/layer-reuse.md) | Skip transformer layers during diffusion steps for speedup |
| [trim-logic](concepts/trim-logic.md) | Token-level Reuse Minimization for efficient generation |
| [block-diffusion](concepts/block-diffusion.md) | Block-wise parallel decoding with hierarchical caching |
| [kv-caching](concepts/kv-caching.md) | Prefix KV cache and dual cache optimization strategies |

---

## Key Optimization Strategies

| Strategy | Model Arg | Description |
|----------|-----------|-------------|
| KV Cache | `use_cache=True` | Reuse attention K/V across diffusion steps |
| Dual Cache | `dual_cache=True` | Also cache masked suffix tokens |
| Confidence Decoding | `threshold=0.9` | Unmask high-confidence tokens together |
| Layer Reuse | `k=1,2,3` | Skip layers during diffusion steps |
| TRIM | - | Token-level reuse minimization |

---

## Common Commands

```bash
# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Full evaluation
bash eval_gsm8k.sh

# Test with limited samples
bash eval_gsm8k.sh --limit 10

# Interactive chatbot
python run_chatbot.py     # CLI
python app.py             # Gradio web UI
```

---

## Quick Links

| Resource | Link |
|----------|------|
| Project Status | [STATUS.md](STATUS.md) |
| Operations Guide | [operations.md](operations.md) |
| Concept Index | [concepts/README.md](concepts/README.md) |
| PACE Reference | [reference/pace-cluster.md](reference/pace-cluster.md) |
| Templates | [_templates/](_templates/) |
