"""Run the preregistered two-case native EE lift-tracking diagnostic."""

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

from examples.baselines.constrain.plug_orientation_ee_expert import run_case


SEEDS = (2026113003, 2026113001)
MAX_CASE_CONTROLS = 1200
MAX_GLOBAL_CONTROLS = 2400


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--demo-dir", required=True)
    args = parser.parse_args()
    experiment_dir = Path(args.experiment_dir)
    demo_dir = Path(args.demo_dir)
    started = time.monotonic()
    summary = {"seeds": list(SEEDS), "cases": [], "controls": 0,
               "started_monotonic": started}
    summary_path = experiment_dir / "native_attempt_001.json"
    try:
        for seed in SEEDS:
            case_started = time.monotonic()
            case_dir = demo_dir / f"case_seed_{seed}"
            row = {"seed": seed, "started_monotonic": case_started}
            summary["cases"].append(row)
            try:
                result = run_case(seed, case_dir)
                controls = int(result["expert_controls"] + result["replay_controls"])
                if controls > MAX_CASE_CONTROLS:
                    raise RuntimeError(f"case control cap exceeded: {controls}")
                row.update({"status": "completed", "controls": controls,
                            "expert_success": bool(result["expert"]["success"]),
                            "expert_reason": result["expert"]["reason"],
                            "expert_official_success": bool(result["expert"]["official_success"]),
                            "replay_success": bool(result["replay"]["success"]),
                            "replay_controls": int(result["replay_controls"]),
                            "finished_monotonic": time.monotonic()})
                summary["controls"] += controls
                if summary["controls"] > MAX_GLOBAL_CONTROLS:
                    raise RuntimeError("global control cap exceeded")
            except BaseException as exc:
                partial_path = case_dir / "partial_failure.json"
                partial = json.loads(partial_path.read_text()) if partial_path.exists() else None
                attempted = int(partial.get("expert_controls", 0)) if partial else 0
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "partial_expert_controls": attempted,
                            "partial_failure_path": str(partial_path) if partial else None,
                            "finished_monotonic": time.monotonic()})
                summary["controls"] += attempted
                summary["exit_code"] = 1
                summary["finished_monotonic"] = time.monotonic()
                summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
                raise
            summary["finished_monotonic"] = time.monotonic()
            summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        summary["exit_code"] = 0
    except BaseException:
        if "exit_code" not in summary:
            summary["exit_code"] = 1
        raise
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
