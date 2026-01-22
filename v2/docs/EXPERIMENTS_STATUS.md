# Experiments Status

**Last Updated:** 2026-01-21

Track ongoing and planned experiments to maintain context across sessions.

---

## Currently Running

### Test Runs (Limit 10)
Testing that code generation tasks work after adding `--confirm_run_unsafe_code` flag.

- **HumanEval** - `--limit 10`, `k1_first` config
  - Status: Running
  - Purpose: Verify code execution works before full run

- **MBPP** - `--limit 10`, `k1_first` config
  - Status: Running
  - Purpose: Verify code execution works before full run

---

## Planned Next Steps

1. **If test runs succeed:**
   - Run full HumanEval (all configs)
   - Run full MBPP (all configs)

2. **Other tasks to evaluate:**
   - GSM8K (if not complete)
   - MMLU (if not complete)
   - GPQA (if not complete)
   - IFEval (if not complete)

---

## Completed

<!-- Track completed experiments here -->

---

## Notes

- Added `--confirm_run_unsafe_code` flag to `sbatch/_job.sh` for code execution tasks
- HumanEval and MBPP require code execution (unsafe), other tasks are text-only (safe)
- All experiments use `00_baseline` experiment name
- Results stored in: `artifacts/00_baseline/<task>_limit_<N>/<config>/`

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
