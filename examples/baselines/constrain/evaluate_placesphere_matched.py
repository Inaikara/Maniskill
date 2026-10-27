"""Evaluate matched-r21b PlaceSphere policies on the frozen 30-scene dev set."""

import argparse
import json
import signal
import time
from pathlib import Path

import torch

from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter, sha256_file
from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
from examples.baselines.constrain.placesphere_learning_pilot import (
    EpisodeDeadline, _episode_deadline, run_pilot_episode, summarize_pilot,
)
from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet


def _read_rows(directory, count=30):
    directory = Path(directory)
    return [json.loads((directory / f"episode_{i:03d}.json").read_text()) for i in range(count)]


def _transitions(old, new, metric):
    left = [bool(row.get("result", {}).get(metric, False)) for row in old]
    right = [bool(row.get("result", {}).get(metric, False)) for row in new]
    if metric in ("obstacle_contact", "xy_forbidden_crossing"):
        return {"cleared": sum(a and not b for a, b in zip(left, right)),
                "introduced": sum(not a and b for a, b in zip(left, right)),
                "both": sum(a and b for a, b in zip(left, right)),
                "neither": sum(not a and not b for a, b in zip(left, right))}
    return {"improved": sum(not a and b for a, b in zip(left, right)),
            "regressed": sum(a and not b for a, b in zip(left, right)),
            "both": sum(a and b for a, b in zip(left, right)),
            "neither": sum(not a and not b for a, b in zip(left, right))}


class _BasePolicy:
    def __init__(self, adapter):
        self.adapter = adapter
        self.metadata = adapter.metadata

    def get_action(self, history):
        queued = torch.as_tensor(history, dtype=torch.float32, device=self.adapter.device)[None]
        return self.adapter.get_action(queued[..., :39])


class _ConstrainPolicy:
    def __init__(self, adapter, checkpoint_path):
        self.adapter = adapter
        self.metadata = {"checkpoint_sha256": sha256_file(checkpoint_path)}

    def get_action(self, history):
        queued = torch.as_tensor(history, dtype=torch.float32, device=self.adapter.device)[None]
        return self.adapter.get_action(self.adapter.base_cond(queued))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--constrain-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-baseline", type=Path, required=True)
    parser.add_argument("--secondary-baseline", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replan-stride", type=int, choices=(2, 8), default=8)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    catalog = TransportDetourSceneSet(args.scene_dir)
    if len(catalog) != 30:
        raise ValueError(f"expected frozen 30-scene development set, found {len(catalog)}")
    base_adapter = PlaceSphereBaseDPAdapter(args.base_checkpoint, args.device, weights="ema")
    constrain_adapter = PlaceSphereConstrainAdapter(
        args.constrain_checkpoint, args.device, base_checkpoint=str(args.base_checkpoint))
    policies = {"base_dp": _BasePolicy(base_adapter),
                "constrain_dp": _ConstrainPolicy(constrain_adapter, args.constrain_checkpoint)}
    args.output_dir.mkdir(parents=True)
    frozen = {"scene_dir": str(args.scene_dir),
              "scene_manifest_sha256": sha256_file(args.scene_dir / "manifest.json"),
              "scene_count": 30,
              "checkpoints": {"base_dp": {"path": str(args.base_checkpoint), "sha256": sha256_file(args.base_checkpoint)},
                              "constrain_dp": {"path": str(args.constrain_checkpoint), "sha256": sha256_file(args.constrain_checkpoint)}},
              "primary_baseline": str(args.primary_baseline),
              "secondary_baseline": str(args.secondary_baseline),
               "device": args.device, "max_steps": 200, "episode_wall_seconds": 300,
               "execution_stride": args.replan_stride,
               "prediction_horizon": 16, "executed_action_slice": [1, 9],
               "noise_sampling": "sequential diffusion RNG per scene; more replans draw more noise",
              "selected_weights": "final EMA fixed before results", "runtime_errors_remain_in_denominator": True}
    (args.output_dir / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    rows_by_policy = {}
    for name, adapter in policies.items():
        out = args.output_dir / name
        out.mkdir()
        rows = []
        for index in range(30):
            previous = signal.signal(signal.SIGALRM, _episode_deadline)
            signal.setitimer(signal.ITIMER_REAL, 300)
            started = time.monotonic()
            trace = []
            try:
                row = run_pilot_episode(adapter, catalog, index, args.device, max_steps=200,
                                        replan_stride=args.replan_stride,
                                        trace_callback=trace.append)
            except Exception as exc:
                row = {"scene_index": index, "status": "runtime_error",
                       "error_kind": "wall_timeout" if isinstance(exc, EpisodeDeadline) else "unexpected",
                        "error": f"{type(exc).__name__}: {exc}",
                        "closed_loop_trace": trace,
                        "checkpoint_sha256": adapter.metadata["checkpoint_sha256"]}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            row["wall_seconds"] = time.monotonic() - started
            row["closed_loop_trace"] = trace
            (out / f"episode_{index:03d}.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
        rows_by_policy[name] = rows
        (out / "summary.json").write_text(json.dumps(summarize_pilot(rows, 30), indent=2) + "\n")

    metric_map = {"task": "native_success", "safe": "safe_success",
                  "contact": "obstacle_contact", "xy": "xy_forbidden_crossing"}
    comparisons = [("primary_constrain_vs_stride2_r24", args.primary_baseline / "constrain_dp", "constrain_dp"),
                   ("base_vs_stride2_r24_base", args.primary_baseline / "base_dp", "base_dp")]
    if args.secondary_baseline is not None:
        comparisons.append(("secondary_constrain_vs_100_obstacle", args.secondary_baseline / "constrain_dp", "constrain_dp"))
    comparison = {}
    for name, baseline, new_name in comparisons:
        old = _read_rows(baseline)
        if any(a.get("seed") != b.get("seed") or a.get("scene_id") != b.get("scene_id")
               for a, b in zip(old, rows_by_policy[new_name])):
            raise ValueError(f"scene pairing mismatch for {name}")
        comparison[name] = {metric: _transitions(old, rows_by_policy[new_name], field)
                            for metric, field in metric_map.items()}
    report = {"complete": all(summarize_pilot(rows, 30)["complete"] for rows in rows_by_policy.values()),
              "denominator": 30,
              "summaries": {name: summarize_pilot(rows, 30) for name, rows in rows_by_policy.items()},
              "comparisons": comparison,
              "gate": {"task_success_min": .90, "safe_success_min": .85,
                       "contact_max": .10, "xy_crossing_max": .10,
                       "passed": False}}
    # The result gate is computed from the new ConstrainNet, without checkpoint selection.
    s = report["summaries"]["constrain_dp"]
    report["gate"]["passed"] = bool(report["complete"] and
        s["task_success_count"] >= 27 and s["safe_success_count"] >= 26 and
        s["obstacle_contact_count"] <= 3 and s["xy_crossing_count"] <= 3)
    (args.output_dir / "paired_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise RuntimeError("paired evaluation has runtime errors; rows retained in denominator")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
