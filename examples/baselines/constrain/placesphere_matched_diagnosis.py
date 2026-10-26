"""Current-checkpoint, indexed PlaceSphere matched-policy diagnosis.

This runner deliberately consumes the paired-evaluation scene set and matched
EMA checkpoints. The legacy release diagnosis remains pinned to its historical
checkpoint and is not used here.
"""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = Path("outputs/ckpt/20261026_153000_placesphere_matched_base_dp/final.pt")
CONSTRAIN = Path("outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k/final.pt")
EVAL = Path("outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval")
SCENES = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
BASE_SHA = "4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942"
CONSTRAIN_SHA = "4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88"
FAILURE_INDICES = (12, 13, 21, 22, 25, 28)
SAFE_CONTROL_INDICES = (0, 2, 4, 6)
ANGULAR_THRESHOLD_RAD_S = 0.5
LINEAR_THRESHOLD_M_S = 0.01


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_motion(linear_speed_m_s, angular_speed_rad_s):
    linear = float(linear_speed_m_s) > LINEAR_THRESHOLD_M_S
    angular = float(angular_speed_rad_s) > ANGULAR_THRESHOLD_RAD_S
    if linear and angular:
        category = "linear_and_angular"
    elif linear:
        category = "linear"
    elif angular:
        category = "angular"
    else:
        category = "within_native_thresholds"
    return {"linear_over_threshold": linear, "angular_over_threshold": angular,
            "category": category}


def classify_queue_causality(trace):
    """Count observed queued actions at failure transitions, without a TCP proxy."""
    for row in trace:
        if "queue_index" in row and not 0 <= int(row["queue_index"]) < 8:
            raise ValueError("queue index must be in the executed [0, 8) slice")
    failures = [row for row in trace if row.get("obstacle_contact") or row.get("failure_transition")]
    positive = [row for row in failures if int(row.get("queue_index", 0)) > 0]
    return {"transition_count": len(failures), "queued_transition_count": len(positive),
            "queue_indices": [int(row["queue_index"]) for row in failures],
            "staleness_isolated": bool(failures and len(positive) == len(failures))}


def _load_expert_release_windows(h5_path, manifest_path):
    """Summarize saved T+1 native states and per-action expert phase/speed metadata."""
    import h5py

    manifest = json.loads(Path(manifest_path).read_text())
    rows = []
    phase_counts = {}
    for episode in manifest["episodes"]:
        if episode.get("kind") != "obstacle":
            continue
        episode_id = int(episode["episode_id"])
        with h5py.File(h5_path, "r") as data:
            group = data[f"traj_{episode_id}"]
            metadata = json.loads(group.attrs["metadata_json"])
            actions = np.asarray(group["actions"])
            raw = np.asarray(group["raw_obs"])
            sphere_state = np.asarray(group["env_states/actors/sphere"])
            phases = metadata.get("step_phase", [])
            saved_linear = metadata.get("sphere_linear_speed_m_s", [])
            saved_angular = metadata.get("sphere_angular_speed_rad_s", [])
            if (raw.shape != (len(actions) + 1, 42) or sphere_state.shape != (len(actions) + 1, 13)
                    or not (len(phases) == len(saved_linear) == len(saved_angular) == len(actions))):
                raise ValueError(f"traj_{episode_id}: expert native-state/action trace is not aligned")
            for step in range(len(actions)):
                phase = phases[step]
                # HDF5 stores state at action boundaries. Per-action velocity is
                # the T+1 state after executing this action, matching metadata traces.
                linear_vector = sphere_state[step + 1, 7:10]
                angular_vector = sphere_state[step + 1, 10:13]
                lin, ang = np.linalg.norm(linear_vector), np.linalg.norm(angular_vector)
                if not (np.isclose(lin, float(saved_linear[step]), atol=1e-5, rtol=1e-4)
                        and np.isclose(ang, float(saved_angular[step]), atol=1e-5, rtol=1e-4)):
                    raise ValueError(f"traj_{episode_id} action {step}: saved velocity trace disagrees with T+1 native state")
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if phase in ("release", "release_hold", "settle"):
                    rows.append({"episode_id": episode_id, "step": step,
                                 "phase": phase, "linear_speed_m_s": float(lin),
                                 "angular_speed_rad_s": float(ang),
                                 "motion": classify_motion(lin, ang),
                                 "sphere_linear_velocity_m_s": linear_vector.astype(float).tolist(),
                                 "sphere_angular_velocity_rad_s": angular_vector.astype(float).tolist(),
                                 "action": actions[step].astype(float).tolist(),
                                 "raw_state_t": raw[step].astype(float).tolist(),
                                 "raw_state_t1": raw[step + 1].astype(float).tolist()})
    return {"phase_action_counts": phase_counts, "release_settle_action_count": len(rows),
            "release_settle_native_trace": rows,
            "source_h5_sha256": sha256(h5_path), "source_manifest_sha256": sha256(manifest_path),
            "thresholds": {"linear_m_s": LINEAR_THRESHOLD_M_S,
                           "angular_rad_s": ANGULAR_THRESHOLD_RAD_S}}


def run(output_dir, *, device="cuda", episode_wall_s=120, phase_wall_s=7200):
    from examples.baselines.constrain.diagnose_placesphere_failure import _child_pids, _terminate_new_children
    from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter
    from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
    from examples.baselines.constrain.placesphere_learning_pilot import (
        _bool_info, _episode_deadline, run_pilot_episode,
    )
    from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
    from examples.baselines.constrain.evaluate_placesphere_matched import _BasePolicy, _ConstrainPolicy

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if sha256(BASE) != BASE_SHA or sha256(CONSTRAIN) != CONSTRAIN_SHA:
        raise ValueError("current matched checkpoint identity mismatch")
    catalog = TransportDetourSceneSet(SCENES)
    if len(catalog) != 30:
        raise ValueError(f"expected exact frozen 30-scene set, got {len(catalog)}")
    paired_config = json.loads((EVAL / "frozen_config.json").read_text())
    if paired_config["scene_manifest_sha256"] != sha256(SCENES / "manifest.json"):
        raise ValueError("scene manifest differs from matched paired evaluation")
    prereg = {
        "kind": "current_matched_placesphere_failure_diagnosis",
        "base_checkpoint": str(BASE), "base_checkpoint_sha256": BASE_SHA,
        "constrain_checkpoint": str(CONSTRAIN), "constrain_checkpoint_sha256": CONSTRAIN_SHA,
        "scene_dir": str(SCENES), "scene_manifest_sha256": sha256(SCENES / "manifest.json"),
        "paired_evaluation_dir": str(EVAL), "paired_frozen_config_sha256": sha256(EVAL / "frozen_config.json"),
        "failure_indices": list(FAILURE_INDICES), "safe_control_indices": list(SAFE_CONTROL_INDICES),
        "per_policy_step_budget": 200, "episode_wall_seconds": episode_wall_s,
        "phase_wall_seconds": phase_wall_s, "controller_executed_slice": [1, 9],
        "queue_semantics": "8 actions, prediction starts at obs_horizon-1; queue index recorded per actual step",
        "native_velocity_thresholds": {"linear_m_s": LINEAR_THRESHOLD_M_S,
                                        "angular_rad_s": ANGULAR_THRESHOLD_RAD_S},
        "sampling": "scene seed reset independently before each policy; original evaluator adapter and 100-step diffusion sampler",
        "source_files": {}, "argv": __import__("sys").argv,
    }
    source_paths = [Path(__file__), Path(__file__).with_name("placesphere_learning_pilot.py"),
                    Path(__file__).with_name("evaluate_placesphere_matched.py"),
                    Path(__file__).with_name("placesphere_base_dp.py"),
                    Path(__file__).with_name("placesphere_constrain.py"),
                    Path(__file__).with_name("placesphere_transport_detour.py")]
    prereg["source_files"] = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    base_adapter = PlaceSphereBaseDPAdapter(BASE, device, weights="ema")
    constrain_adapter = PlaceSphereConstrainAdapter(CONSTRAIN, device, base_checkpoint=str(BASE))
    policies = {"base_dp": _BasePolicy(base_adapter),
                "constrain_dp": _ConstrainPolicy(constrain_adapter, CONSTRAIN)}
    selected = list(dict.fromkeys((*FAILURE_INDICES, *SAFE_CONTROL_INDICES)))
    deadline = time.monotonic() + phase_wall_s
    rows = []
    for policy_name, adapter in policies.items():
        for index in selected:
            trace = []
            previous = signal.signal(signal.SIGALRM, _episode_deadline)
            before = set(_child_pids())
            start = time.monotonic()
            signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError("phase wall budget exceeded")
                episode = run_pilot_episode(adapter, catalog, index, device,
                                            max_steps=200, trace_callback=trace.append)
                published = json.loads((EVAL / policy_name / f"episode_{index:03d}.json").read_text())
                a, b = episode.get("result", {}), published.get("result", {})
                reproduction = {key: a.get(key) == b.get(key) for key in (
                    "native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing",
                    "elapsed_steps", "termination_reason")}
                episode["reproduction"] = {"checks": reproduction,
                                             "mismatch": not all(reproduction.values())}
                for item in trace:
                    item["motion_class"] = classify_motion(item["sphere_linear_speed_m_s"],
                                                           item["sphere_angular_speed_rad_s"])
                episode["closed_loop_trace"] = trace
                episode["trace_summary"] = {
                    "steps": len(trace),
                    "queue_indices": [row["queue_index"] for row in trace],
                    "replan_steps": sorted(set(row["replan_step"] for row in trace)),
                    "first_contact_step": next((row["step"] for row in trace if row["obstacle_contact"]), None),
                    "max_sphere_linear_speed_m_s": max((row["sphere_linear_speed_m_s"] for row in trace), default=0),
                    "max_sphere_angular_speed_rad_s": max((row["sphere_angular_speed_rad_s"] for row in trace), default=0),
                    "release_steps": [row["step"] for row in trace if row["released_this_step"]],
                    "queue_causality": classify_queue_causality(trace),
                }
                episode["elapsed_wall_seconds"] = time.monotonic() - start
                episode["status"] = "evaluated"
            except Exception as exc:
                episode = {"scene_index": index, "policy": policy_name, "status": "runtime_error",
                           "error": f"{type(exc).__name__}: {exc}", "closed_loop_trace": trace,
                           "elapsed_wall_seconds": time.monotonic() - start,
                           "reproduction": {"mismatch": True, "checks": {}}}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
                _terminate_new_children(before)
            episode["policy"] = policy_name
            path = output / f"{policy_name}_episode_{index:03d}.json"
            path.write_text(json.dumps(episode, indent=2, sort_keys=True) + "\n")
            rows.append({"policy": policy_name, "scene_index": index, "status": episode["status"],
                         "result": episode.get("result"), "reproduction": episode["reproduction"],
                         "trace_summary": episode.get("trace_summary"), "error": episode.get("error")})
            (output / "partial_index.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    expert_path = Path("demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5")
    expert_manifest = expert_path.with_name("trajectory.json")
    expert = _load_expert_release_windows(expert_path, expert_manifest)
    (output / "expert_release_settle_native_trace.json").write_text(json.dumps(expert, indent=2) + "\n")
    complete = len(rows) == 2 * len(selected) and all(row["status"] == "evaluated" for row in rows)
    report = {"complete": complete, "denominator_per_policy": len(selected),
              "indices": selected, "episodes": rows,
            "expert_release_settle": {"action_count": expert["release_settle_action_count"],
                                        "phase_action_counts": expert["phase_action_counts"],
                                        "native_trace_path": "expert_release_settle_native_trace.json"},
              "checkpoint_sha256": {"base_dp": BASE_SHA, "constrain_dp": CONSTRAIN_SHA},
              "elapsed_seconds": phase_wall_s - max(0, deadline - time.monotonic()),
              "budgets": {"phase_wall_seconds": phase_wall_s, "episode_wall_seconds": episode_wall_s}}
    (output / "diagnosis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not complete:
        raise RuntimeError("diagnostic incomplete; runtime errors retained in denominator")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-wall-s", type=int, default=120)
    parser.add_argument("--phase-wall-s", type=int, default=7200)
    args = parser.parse_args(argv)
    report = run(args.output_dir, device=args.device,
                 episode_wall_s=args.episode_wall_s, phase_wall_s=args.phase_wall_s)
    print(json.dumps({"complete": report["complete"], "episodes": len(report["episodes"]),
                      "output": args.output_dir}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
