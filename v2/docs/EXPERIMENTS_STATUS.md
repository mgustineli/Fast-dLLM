# Experiments Status

**Last Updated:** 2026-01-21

Track ongoing and planned experiments to maintain context across sessions.

---

## Currently Running

None

---

## Planned Next Steps

1. **Code Generation Tasks - Larger Sample Size:**
   - Run HumanEval with `--limit 50` or `--limit 100` to get better accuracy estimates
   - Run MBPP with `--limit 50` or `--limit 100` to get better accuracy estimates
   - Initial tests (limit 10) showed 0% pass@1, but sample size too small to draw conclusions

2. **If larger samples confirm functionality:**
   - Run full HumanEval (all configs, no limit)
   - Run full MBPP (all configs, no limit)

3. **Other tasks to evaluate:**
   - GSM8K (check completion status)
   - MMLU (check completion status)
   - GPQA (check completion status)
   - IFEval (check completion status)

---

## Completed

### Test Runs (Limit 10) - Jan 21, 2026
Successfully verified code execution infrastructure works.

- **HumanEval** - `--limit 10`, `k1_first` config
  - Status: ✅ Completed
  - Results: 0% pass@1 (0/10), 31.64 tokens/s
  - Outcome: Infrastructure works, but sample too small for conclusions

- **MBPP** - `--limit 10`, `k1_first` config
  - Status: ✅ Completed
  - Results: 0% pass@1 (0/10), 33.55 tokens/s
  - Outcome: Infrastructure works, but sample too small for conclusions

---

## Notes

- Added `--confirm_run_unsafe_code` flag to `sbatch/_job.sh` for code execution tasks
- HumanEval and MBPP require code execution (unsafe), other tasks are text-only (safe)
- All experiments use `00_baseline` experiment name
- Results stored in: `artifacts/00_baseline/<task>_limit_<N>/<config>/`

---

## Debugging Notes (2026-01-22)

- **Issue:** HumanEval and MBPP show 0% accuracy despite `unsafe_code: true` in task config.
- **Initial Hypothesis:** Incorrect flag for `lm-evaluation-harness` to enable code execution.
  - Attempted to replace `--confirm_run_unsafe_code` with `--allow_code_execution` in `sbatch/_job.sh`.
- **Result:** Failed with `eval.py: error: unrecognized arguments: --allow_code_execution`.
- **Current Understanding:** The argument parsing in `eval.py` (specifically `extract_cli_args()`) is interfering with `lm-evaluation-harness`'s ability to process command-line arguments, including the `--allow_code_execution` flag.
- **Next Steps:** Investigate how to correctly pass `lm-evaluation-harness` arguments through `eval.py` without conflicts, or how to enable code execution directly within `eval.py`'s `Fast_dLLM_v2EvalHarness` model definition. Consider temporary modification of `eval.py` to bypass its argument parsing for debugging.

---

## Quick Commands

```bash
# Check status
bash sbatch/run.sh --status --task humaneval --limit 10

# Submit test run
bash sbatch/run.sh --task humaneval --limit 10 --config k1_first

# View logs
tail -f logs/00_baseline/humaneval_limit_10/k1_first/slurm_*.log
```
