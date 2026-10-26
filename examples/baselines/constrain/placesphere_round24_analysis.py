"""Post-trial, read-only release joins for the R24 fixed-checkpoint trial."""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.placesphere_release_causal import release_events


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expert_release_joins(h5_path, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    events, verified = [], 0
    with h5py.File(h5_path, "r") as data:
        for item in manifest["episodes"]:
            if item.get("kind") != "obstacle":
                continue
            episode_id = int(item["episode_id"])
            group = data[f"traj_{episode_id}"]
            metadata = json.loads(group.attrs["metadata_json"])
            actions = np.asarray(group["actions"])
            raw = np.asarray(group["raw_obs"])
            sphere = np.asarray(group["env_states/actors/sphere"])
            linear = metadata["sphere_linear_speed_m_s"]
            angular = metadata["sphere_angular_speed_rad_s"]
            if raw.shape != (len(actions) + 1, 42) or sphere.shape != (len(actions) + 1, 13):
                raise ValueError(f"traj_{episode_id}: invalid T+1 records")
            for step in range(len(actions)):
                post_lin = float(np.linalg.norm(sphere[step + 1, 7:10]))
                post_ang = float(np.linalg.norm(sphere[step + 1, 10:13]))
                if not (np.isclose(post_lin, linear[step], atol=1e-5, rtol=1e-4)
                        and np.isclose(post_ang, angular[step], atol=1e-5, rtol=1e-4)):
                    raise ValueError(f"traj_{episode_id} action {step}: post-action speed mismatch")
                verified += 1
                opened = actions[step, 6] > 0
                was_grasped = raw[step, 18] > 0.5
                ungrasped = raw[step + 1, 18] <= 0.5
                if opened and was_grasped and ungrasped:
                    dt = float(item.get("control_timestep_s", 0.0) or 0.0)
                    next_dz = None if step + 2 >= len(sphere) else float(sphere[step + 2, 2] - sphere[step + 1, 2])
                    events.append({
                        "episode_id": episode_id, "action_index": step,
                        "post_action_state_index": step + 1,
                        "open_action_7d": actions[step].astype(float).tolist(),
                        "pre_open_linear_velocity": sphere[step, 7:10].astype(float).tolist(),
                        "pre_open_angular_velocity": sphere[step, 10:13].astype(float).tolist(),
                        "post_action_linear_velocity": sphere[step + 1, 7:10].astype(float).tolist(),
                        "post_action_angular_velocity": sphere[step + 1, 10:13].astype(float).tolist(),
                        "post_action_linear_speed_m_s": post_lin,
                        "post_action_angular_speed_rad_s": post_ang,
                        "next_interval_dz_m": next_dz,
                        "next_interval_control_timestep_s": dt or None,
                        "gravity_expected_dz_m": None if not dt else -0.5 * 9.81 * dt * dt,
                        "post_action_state_is_same_index_as_saved_speed": True,
                        "impulse_measured": False,
                    })
    return {"events": events, "verified_action_state_pairs": verified,
            "source_h5_sha256": sha256(h5_path), "source_manifest_sha256": sha256(manifest_path),
            "grasp_raw_observation_index": 18}


def trial_release_report(trial_dir):
    root = Path(trial_dir)
    selected = [5, 10, 24, 11, 17, 0, 1, 2, 3, 4]
    policies = {}
    for name in ("base_dp", "constrain_dp"):
        episodes = []
        for index in selected:
            row = json.loads((root / "paired_eval" / name / f"episode_{index:03d}.json").read_text())
            trace = row.get("closed_loop_trace", [])
            opens = release_events(trace)
            episodes.append({"scene_index": index, "status": row.get("status"),
                             "result": row.get("result"), "release_events": opens,
                             "trace": trace})
        policies[name] = episodes
    return {"kind": "posthoc_indexed_release_mechanics",
            "causal_claim": "not isolated; descriptive timing and kinematics only",
            "selected_indices": selected, "policies": policies}


def transitions(old, new, key):
    left = [bool(row.get("result", {}).get(key, False)) for row in old]
    right = [bool(row.get("result", {}).get(key, False)) for row in new]
    if key in ("obstacle_contact", "xy_forbidden_crossing"):
        return {"cleared": sum(a and not b for a, b in zip(left, right)),
                "introduced": sum(not a and b for a, b in zip(left, right)),
                "both": sum(a and b for a, b in zip(left, right)),
                "neither": sum(not a and not b for a, b in zip(left, right))}
    return {"improved": sum(not a and b for a, b in zip(left, right)),
            "regressed": sum(a and not b for a, b in zip(left, right)),
            "both": sum(a and b for a, b in zip(left, right)),
            "neither": sum(not a and not b for a, b in zip(left, right))}


def _load_policy_rows(root, policy):
    folder = Path(root) / policy
    return [json.loads((folder / f"episode_{i:03d}.json").read_text()) for i in range(30)]


def write_report(trial_dir):
    root = Path(trial_dir)
    result = trial_release_report(root)
    expert = expert_release_joins(
        "demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23/trajectory.h5",
        "demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23/trajectory.json")
    for name, episodes in result["policies"].items():
        for episode in episodes:
            # Keep the first open command that precedes the physical release transition.
            first_by_transition = {}
            for event in episode["release_events"]:
                trace = episode["trace"]
                at = next(i for i, row in enumerate(trace) if row["step"] == event["open_command_step"])
                grasped_before_open = at > 0 and bool(trace[at - 1].get("grasped"))
                if grasped_before_open or event["open_command_step"] == event["ungrasped_transition_step"]:
                    first_by_transition.setdefault(event["ungrasped_transition_step"], event)
            episode["release_events"] = list(first_by_transition.values())
            by_step = {int(row["step"]): row for row in episode["trace"]}
            for event in episode["release_events"]:
                dt = event["next_interval_control_timestep_s"]
                event["observed_dz_dt_m_s"] = None if not dt else event["next_interval_dz_m"] / dt
                event["gravity_expected_delta_v_m_s"] = None if not dt else -9.81 * dt
                next_row = by_step.get(int(event["ungrasped_transition_step"]) + 1)
                event["next_interval_obstacle_contact"] = None if next_row is None else bool(next_row["obstacle_contact"])
                event["next_interval_obstacle_crossing"] = None if next_row is None else bool(next_row["obstacle_crossing"])
            episode.pop("trace")
    result["expert_release_events"] = expert["events"]
    result["expert_verified_action_state_pairs"] = expert["verified_action_state_pairs"]
    result["expert_source_identity"] = {
        "h5_sha256": expert["source_h5_sha256"],
        "manifest_sha256": expert["source_manifest_sha256"],
        "join": "action index t uses native state t+1; saved per-action speed checked at exactly t",
        "control_timestep_available": False,
        "gravity_displacement_comparison": "not computed for HDF5 expert: timestep metadata is absent",
    }
    expert_pre = [float(np.linalg.norm(row["pre_open_linear_velocity"])) for row in expert["events"]]
    expert_post = [row["post_action_linear_speed_m_s"] for row in expert["events"]]
    expert_pre_ang = [float(np.linalg.norm(row["pre_open_angular_velocity"])) for row in expert["events"]]
    expert_post_ang = [row["post_action_angular_speed_rad_s"] for row in expert["events"]]
    def quantiles(values):
        return {"count": len(values), "median": float(np.median(values)),
                "p90": float(np.quantile(values, .9)), "max": float(np.max(values))}
    result["expert_release_speed_summary"] = {
        "pre_open_linear_m_s": quantiles(expert_pre), "post_action_linear_m_s": quantiles(expert_post),
        "pre_open_angular_rad_s": quantiles(expert_pre_ang), "post_action_angular_rad_s": quantiles(expert_post_ang),
        "post_action_linear_over_0_01_count": sum(x > .01 for x in expert_post),
        "post_action_angular_over_0_5_count": sum(x > .5 for x in expert_post_ang),
        "interpretation": "same-index T+1 action-aligned kinematics; speed jump temporal association, not impulse or causal attribution",
    }
    primary_dir = "outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23/paired_eval"
    secondary_dir = "outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval"
    comparisons = {}
    for label, baseline in (("primary_hold8", primary_dir), ("secondary_hold0_not_isolated", secondary_dir)):
        for policy in ("base_dp", "constrain_dp"):
            old, new = _load_policy_rows(baseline, policy), _load_policy_rows(root / "paired_eval", policy)
            comparisons[f"{label}_{policy}"] = {
                metric: transitions(old, new, field) for metric, field in (
                    ("task", "native_success"), ("safe", "safe_success"),
                    ("contact", "obstacle_contact"), ("xy", "xy_forbidden_crossing"))}
    result["comparisons"] = comparisons
    result["all_episode_metrics"] = {}
    for policy in ("base_dp", "constrain_dp"):
        rows = _load_policy_rows(root / "paired_eval", policy)
        successful_steps = [int(row["result"]["elapsed_steps"]) for row in rows
                            if row.get("status") == "evaluated" and row["result"]["native_success"]]
        result["all_episode_metrics"][policy] = {
            "denominator": 30,
            "runtime_errors": sum(row.get("status") != "evaluated" for row in rows),
            "task_success": sum(bool(row.get("result", {}).get("native_success")) for row in rows),
            "safe_success": sum(bool(row.get("result", {}).get("safe_success")) for row in rows),
            "contact": sum(bool(row.get("result", {}).get("obstacle_contact")) for row in rows),
            "xy_crossing": sum(bool(row.get("result", {}).get("xy_forbidden_crossing")) for row in rows),
            "failure_scene_indices": [row["scene_index"] for row in rows
                                      if row.get("status") == "evaluated" and not row["result"]["native_success"]],
            "successful_elapsed_steps_median_p90_max": None if not successful_steps else {
                "median": float(np.median(successful_steps)), "p90": float(np.quantile(successful_steps, .9)),
                "max": max(successful_steps)},
            "wall_seconds_sum": float(sum(row.get("wall_seconds", 0) for row in rows)),
            "first_contact_scene_phase": [{"scene_index": row["scene_index"],
                                            "phase": (row.get("first_contact") or {}).get("phase")}
                                           for row in rows if (row.get("first_contact") or {}).get("step") is not None],
        }
    result["diagnostic_limits"] = {
        "sphere_robot_contact": "not exposed in saved per-step info; obstacle contact only is recorded",
        "impulse_or_torque": "not measured; no impulse claim",
        "causal_identification": "descriptive synchronized kinematics cannot separate release actuation, contact, gravity, and pre-existing motion",
        "pre_open_vs_open": "motion measured immediately before gripper-open action and post-action; open command is not assumed to equal ungrasped transition",
    }
    result["analysis_source_sha256"] = sha256(Path(__file__))
    result["native_trial_command"] = (
        "source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill && "
        "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM --kill-after=60s 18060s "
        "python -u -m examples.baselines.constrain.evaluate_placesphere_matched "
        "--scene-dir outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes "
        "--base-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt "
        "--constrain-checkpoint outputs/ckpt/20261026_195000_placesphere_hold8_constrain_90k_r23/final.pt "
        "--output-dir outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval "
        "--primary-baseline outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23/paired_eval "
        "--secondary-baseline outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval "
        "--device cuda --replan-stride 2"
    )
    output = root / "release_causal_measurements.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    path = write_report(sys.argv[1])
    print(path)
