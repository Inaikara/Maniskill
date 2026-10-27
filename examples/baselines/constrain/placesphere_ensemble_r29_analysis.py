"""Read-only paired analysis of the frozen Round-29 ensemble trial."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(directory):
    return [json.loads((Path(directory) / f"episode_{i:03d}.json").read_text())
            for i in range(30)]


def tree_sha256(directory):
    directory = Path(directory)
    entries = sorted((f"{directory.name}/{path.name}", sha256(path))
                     for path in directory.glob("episode_*.json"))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def transition(before, after, metric, adverse=False):
    left = [bool(row.get("result", {}).get(metric, False)) for row in before]
    right = [bool(row.get("result", {}).get(metric, False)) for row in after]
    if adverse:
        keys = ("cleared", "introduced", "both", "neither")
        values = (sum(a and not b for a, b in zip(left, right)),
                  sum(not a and b for a, b in zip(left, right)),
                  sum(a and b for a, b in zip(left, right)),
                  sum(not a and not b for a, b in zip(left, right)))
    else:
        keys = ("improved", "regressed", "both", "neither")
        values = (sum(not a and b for a, b in zip(left, right)),
                  sum(a and not b for a, b in zip(left, right)),
                  sum(a and b for a, b in zip(left, right)),
                  sum(not a and not b for a, b in zip(left, right)))
    return dict(zip(keys, values))


def analyze(registration):
    reg = json.loads(Path(registration).read_text())
    for label, path in reg["inputs"].items():
        if sha256(path) != reg["input_sha256"][label]:
            raise ValueError(f"registered input changed: {label}")
    source = Path(__file__)
    if sha256(source) != reg["analysis_source_sha256"]:
        raise ValueError("analysis source identity mismatch")
    root = Path("examples/baselines/constrain")
    for name, expected in reg["helper_source_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"helper source identity mismatch: {name}")
    for policy, paths in reg["episode_trees"].items():
        for path, expected in paths.items():
            if tree_sha256(path) != expected:
                raise ValueError(f"episode tree identity mismatch: {policy} {path}")

    trial, primary = Path(reg["trial_dir"]), Path(reg["primary_dir"])
    result = {"format": "placesphere_r29_ensemble_primary_analysis_v1",
              "trial_dir": str(trial), "primary": str(primary),
              "denominator_per_policy": 30, "runtime_errors": {}, "metrics": {},
              "transitions_vs_r27": {}, "failure_details": {},
              "input_sha256": reg["input_sha256"]}
    for policy, checkpoint in reg["checkpoint_sha256"].items():
        new, old = rows(trial / policy), rows(primary / policy)
        if len(new) != 30 or len(old) != 30:
            raise ValueError(f"{policy}: fixed 30-row denominator mismatch")
        for index, (a, b) in enumerate(zip(new, old)):
            if (a.get("scene_index") != index or b.get("scene_index") != index
                    or a.get("scene_id") != b.get("scene_id") or a.get("seed") != b.get("seed")
                    or not a.get("restored_bit_exact") or not b.get("restored_bit_exact")
                    or a.get("checkpoint_sha256") != checkpoint
                    or b.get("checkpoint_sha256") != checkpoint
                    or not a.get("closed_loop_trace") or not b.get("closed_loop_trace")
                    or a["closed_loop_trace"][0].get("policy_observation")
                    != b["closed_loop_trace"][0].get("policy_observation")):
                raise ValueError(f"{policy}: paired frozen identity mismatch at scene {index}")
        result["runtime_errors"][policy] = sum(row.get("status") != "evaluated" for row in new)
        result["metrics"][policy] = {
            "task": sum(bool(row.get("result", {}).get("native_success")) for row in new),
            "safe": sum(bool(row.get("result", {}).get("safe_success")) for row in new),
            "contact": sum(bool(row.get("result", {}).get("obstacle_contact")) for row in new),
            "xy": sum(bool(row.get("result", {}).get("xy_forbidden_crossing")) for row in new),
            "all_30_paired_initial_observations_match": True,
            "all_30_restores_bit_exact": True,
            "max_single_inference_wall_seconds": max(
                row.get("max_single_inference_wall_seconds", 0.0) for row in new),
        }
        result["transitions_vs_r27"][policy] = {
            metric: transition(old, new, field, adverse=metric in ("contact", "xy"))
            for metric, field in (("task", "native_success"), ("safe", "safe_success"),
                                  ("contact", "obstacle_contact"),
                                  ("xy", "xy_forbidden_crossing"))}
        if policy == "constrain_dp":
            result["failure_details"][policy] = [
                {"scene_index": row["scene_index"], "scene_id": row["scene_id"],
                 "mechanism": row.get("terminal_mechanism", {}).get("mechanism"),
                 "release_linear_speed_m_s": row.get("release", {}).get("linear_speed_m_s"),
                 "release_relative_xyz": row.get("release", {}).get("relative_xyz"),
                 "terminal_on_bin": row.get("terminal", {}).get("on_bin"),
                 "terminal_grasped": row.get("terminal", {}).get("grasped")}
                for row in new if not row.get("result", {}).get("native_success")]

    m = result["metrics"]["constrain_dp"]
    result["gate"] = {"task_min": 27, "safe_min": 26, "contact_max": 3, "xy_max": 3,
                      "passed": m["task"] >= 27 and m["safe"] >= 26
                      and m["contact"] <= 3 and m["xy"] <= 3}
    result["interpretation"] = (
        "One matched frozen-development-set observation; the trial changes initial-plan count, "
        "random draw consumption, and inference compute together. No causal claim isolates "
        "stochastic dispersion. Arithmetic trajectory averaging can blend multimodal actions; "
        "topology preservation is not asserted without policy-trace safety evidence.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.registration)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
