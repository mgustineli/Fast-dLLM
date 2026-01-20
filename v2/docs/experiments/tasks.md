# Experiment Tasks & TODOs

This document tracks known issues, bugs, and improvement tasks for the Fast-dLLM v2 experiment framework.

## Critical Priority

### 1. Layer Reuse NOT Applied to Loglikelihood Tasks

**Impact:** All MMLU experimental results are invalid - layer reuse optimization not working
**Status:** TODO
**Details:** [layer-reuse-loglikelihood.md](layer-reuse-loglikelihood.md)

**Summary:** The layer reuse optimization only works for generative tasks (GSM8K) but not loglikelihood tasks (MMLU, GPQA). All 9 MMLU experiments produced identical accuracy (66.53%) regardless of configuration, when they should vary. Requires modifying `get_loglikelihood()` to apply layer patches.

---

## High Priority

### 2. GPQA Dataset Authentication Required

**Impact:** All GPQA experiments fail to load dataset
**Status:** TODO
**Details:** [gpqa-authentication.md](gpqa-authentication.md)

**Summary:** GPQA dataset is gated on HuggingFace Hub and requires authentication. Need to add `HF_TOKEN` environment variable to `sbatch/_job.sh` and request dataset access.

### 3. Missing Throughput Tracking for Loglikelihood Tasks

**Impact:** Cannot measure inference speed for MMLU/GPQA
**Status:** TODO
**Details:** [missing-throughput-tracking.md](missing-throughput-tracking.md)

**Summary:** Throughput metrics (tokens/s, total time) are only tracked for generative tasks. MMLU experiments have empty `"throughput": {}` in summary.json. Need to add timing/token counting to `loglikelihood()` method.

---

## Quick Reference

| Task | Priority | Files to Modify | Verification |
|------|----------|----------------|--------------|
| Layer reuse for loglikelihood | CRITICAL | eval.py:272-303 | MMLU configs show different accuracy |
| GPQA authentication | High | sbatch/_job.sh:86-90 | GPQA experiments complete successfully |
| Throughput tracking | High | eval.py:305-333 | summary.json has throughput metrics |

## Adding New Tasks

When documenting a new issue or TODO:

1. Create a dedicated markdown file: `issue-name.md`
2. Add entry to this tasks.md with:
   - Priority level (Critical/High/Medium/Low)
   - Brief summary
   - Link to detailed markdown file
3. Include in the dedicated file:
   - **Status:** TODO/In Progress/Completed
   - **Priority:** Critical/High/Medium/Low
   - **Affects:** What components/experiments
   - **Issue:** Clear description
   - **Root Cause:** Technical explanation
   - **Impact:** What's broken/missing
   - **Solution:** Implementation approach
   - **Files to Modify:** Specific file paths and line numbers
   - **Verification:** How to test the fix
