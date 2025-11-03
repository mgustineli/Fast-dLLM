import json
import os
import time
from glob import glob


def start_run_log(task="gsm8k", tag=None, log_dir="results"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{task}_{tag or 'run'}_{timestamp}.json"
    path = os.path.join(log_dir, filename)
    print(f"[LOG] Starting run: {filename}")
    return {
        "t0": time.time(),
        "path": path,
        "task": task,
        "tag": tag,
        "timestamp": timestamp,
    }


def extract_accuracy_from_results(results_dir):
    """Find the newest results_*.json file under the model directory and return accuracy."""
    try:
        pattern = os.path.join(results_dir, "*/results_*.json")
        result_files = sorted(glob(pattern), key=os.path.getmtime, reverse=True)
        if not result_files:
            print(f"[WARN] No result files found under {results_dir}")
            return None
        latest_file = result_files[0]
        with open(latest_file, "r") as f:
            data = json.load(f)

        gsm8k_metrics = data.get("results", {}).get("gsm8k", {})
        for key, value in gsm8k_metrics.items():
            if "flexible-extract" in key and key.startswith("exact_match"):
                print(f"[LOG] Parsed accuracy from {os.path.basename(latest_file)}: {value}")
                return value

        print(f"[WARN] flexible-extract accuracy not found in {latest_file}")
        return None
    except Exception as e:
        print(f"[WARN] Could not extract accuracy: {e}")
        return None



def end_run_log(run_info, tokens_generated=None, results_dir=None, **extra):
    """End timer, compute throughput, and optionally merge accuracy from results_dir."""
    t1 = time.time()
    duration = t1 - run_info["t0"]
    throughput = (
        (tokens_generated / duration) if (tokens_generated and duration > 0) else None
    )

    accuracy = None
    if results_dir and os.path.exists(results_dir):
        accuracy = extract_accuracy_from_results(results_dir)

    data = {
        "task": run_info["task"],
        "tag": run_info["tag"],
        "timestamp": run_info["timestamp"],
        "total_time_s": duration,
        "tokens_generated": tokens_generated,
        "tokens_per_s": throughput,
        "accuracy_flexible": accuracy,
        **extra,  # merge any new fields here
    }

    with open(run_info["path"], "w") as f:
        json.dump(data, f, indent=2)

    print(f"[LOG] Saved run summary to {run_info['path']}")
    return data
