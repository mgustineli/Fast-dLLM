# PACE Phoenix Job Accounting

## Accounting System Overview

Phoenix's accounting system is based on the most significant processing unit on the compute node:

- **CPU and CPU-SAS nodes:** Charge rates based on CPU-hours (total number of procs × walltime)
- **GPU nodes (V100, RTX6000, A100, H100, L40S):** Charge rates based on GPU-hours (total number of GPUs × walltime)

---

## Specifying Charge Account

When submitting a job, the account to which the job should be charged **must be specified** using the `-A` flag:
- On command line: `salloc -A gts-gburdell3 ...`
- In Slurm batch file: `#SBATCH -A gts-gburdell3`

The scheduler will:
1. Verify the account has sufficient funds to run the full job length
2. Place a lien on the account when the job starts
3. Release excess funds if the job finishes early

---

## Account Types

| Account Syntax | Description | Example |
|----------------|-------------|---------|
| `gts-<PI UID>` | Institute-sponsored account with 10k CPU hours on base CPU-192GB node (credits reset monthly) | `gts-gburdell3` |
| `gts-<PI UID>-CODA20` | 2020 hardware refresh account; credits based on 5 years of cycles on refreshed hardware | `gts-gburdell3-CODA20` |
| `gts-<PI UID>-FY20PhaseN` | FY20 purchase account; credited with max of FY20 expenditures or 5-year cycle equivalent | `gts-gburdell3-FY20Phase2` |
| `gts-<PI UID>-<group>` | PI-specific child account for shared (multi-PI or school-owned) account | `gts-gburdell3-phys` |
| `gts-<group>-CODA20` | Parent account for multi-PI shared account (cannot be used for job submissions) | `gts-phys-CODA20` |
| `gts-<PI UID>-<custom>` | Postpaid billing model account (billed monthly based on actual usage) | `gts-gburdell3-paid` |
| `gts-<PI UID>-<custom>` | Prepaid billing model for state funds (deposit funds in advance) | `gts-gburdell3-startup` |

---

## Checking Account Balances

Use `pace-quota` to see available accounts and balances:

```bash
pace-quota
```

Look for the "Job Charge Account Balances" section showing:
- **Balance:** Current total
- **Reserved:** Funds tied to running jobs
- **Available:** Funds available for new submissions

---

## Resource Rates

Rates for each node class can be found in the [PACE resources page](https://docs.pace.gatech.edu/phoenix_cluster/gettingstarted_phnx/#compute-resources).

Node classes include:
- CPU-192GB
- CPU-768GB-SAS
- GPU-192GB-V100
- GPU-384GB-RTX6000
- GPU-A100
- GPU-H100
- GPU-H200
- GPU-L40S
