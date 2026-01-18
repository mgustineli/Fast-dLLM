import os
import json
import torch
import subprocess


def get_gpu_info():
    """Return a dict with GPU specs (name, memory, driver, CUDA)."""
    info = {"gpu_available": torch.cuda.is_available()}
    if not info["gpu_available"]:
        return info

    try:
        device_idx = 0
        props = torch.cuda.get_device_properties(device_idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_total_vram_gb": round(props.total_memory / (1024**3), 2),
                "cuda_device": device_idx,
                "cuda_version": torch.version.cuda,
                "driver_version": None,
            }
        )

        try:
            out = (
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
                .split("\n")[device_idx]
            )
            info["driver_version"] = out
        except Exception:
            pass

        return info
    except Exception as e:
        info["error"] = str(e)
        return info


def find_results_file(results_dir):
    """Find lm_eval results file (handles subdirectory and timestamped names)."""
    # First, check for results.json directly
    direct_path = os.path.join(results_dir, "results.json")
    if os.path.exists(direct_path):
        return direct_path

    # lm_eval writes to <model_name_sanitized>/results_<timestamp>.json
    # Look for any subdirectory containing results_*.json
    for item in os.listdir(results_dir):
        subdir = os.path.join(results_dir, item)
        if os.path.isdir(subdir):
            for filename in os.listdir(subdir):
                if filename.startswith("results_") and filename.endswith(".json"):
                    return os.path.join(subdir, filename)

    return None


def extract_accuracy(results_dir, task="gsm8k"):
    """Extract accuracy from lm_eval results.json file."""
    results_path = find_results_file(results_dir)
    if results_path is None:
        return None

    try:
        with open(results_path) as f:
            data = json.load(f)

        task_results = data.get("results", {}).get(task, {})

        # Try different accuracy keys (task-dependent)
        accuracy_keys = [
            "exact_match,flexible-extract",
            "exact_match,get-answer",
            "exact_match,strict-match",
            "exact_match",
            "acc,none",
            "acc_norm,none",
        ]

        for key in accuracy_keys:
            if key in task_results:
                return task_results[key]

        # Fallback: return first numeric metric found
        for key, value in task_results.items():
            if isinstance(value, (int, float)) and not key.endswith("_stderr"):
                return value

        return None
    except Exception as e:
        print(f"[WARN] Could not extract accuracy: {e}")
        return None


def write_summary(output_dir, task, config, throughput_metrics, timestamp):
    """Write a single summary.json combining accuracy, throughput, and config."""
    summary = {
        "task": task,
        "config": config,
        "timestamp": timestamp,
        "accuracy": extract_accuracy(output_dir, task),
        "throughput": throughput_metrics,
        "gpu_info": get_gpu_info(),
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[LOG] Summary saved to {summary_path}")
    return summary
