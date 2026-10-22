"""Frozen PickCube collection contract and retrospective base-side audit."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


def canonical_json_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def base_line_distance(start, goal, base):
    direction = np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)
    distance = np.linalg.norm(direction)
    if distance == 0:
        return float(np.linalg.norm(np.asarray(base, dtype=float) - start))
    return float(abs(np.cross(direction, np.asarray(base, dtype=float) - start)) / distance)


def audit_pilot(dataset_dir):
    """Audit retained HDF5 root poses and layouts; never modifies source artifacts."""
    root = Path(dataset_dir)
    episodes = json.loads((root / "trajectory.json").read_text())["episodes"]
    rows = []
    with h5py.File(root / "trajectory.h5", "r") as data:
        for episode in episodes:
            index = episode["episode_id"]
            state = data[f"traj_{index}/env_states/articulations/panda"][0]
            # Native articulation state starts with root position XYZ and quaternion WXYZ.
            base_xy = np.asarray(state[:2], dtype=float).tolist()
            layout = episode["layout"]
            start, goal = layout["start_xy"], layout["goal_xy"]
            signed = float(np.cross(np.asarray(goal) - start, np.asarray(base_xy) - start))
            waypoint = episode["trace"].get("waypoint_xy")
            rows.append({"episode_id": index, "kind": episode["kind"],
                         "base_xy_m": base_xy, "start_xy_m": start, "goal_xy_m": goal,
                         "base_side_signed_cross_m2": signed,
                         "base_to_segment_distance_m": base_line_distance(start, goal, base_xy),
                         "waypoint_xy_m": waypoint,
                         "waypoint_base_side_signed_cross_m2": None if waypoint is None else float(
                             np.cross(np.asarray(goal) - start, np.asarray(waypoint) - start)),
                         "minimum_base_line_distance_m": 0.06})
    classes = {}
    for kind in ("direct", "obstacle"):
        selected = [row for row in rows if row["kind"] == kind]
        classes[kind] = {
            "count": len(selected),
            "base_line_distance_min_m": min(x["base_to_segment_distance_m"] for x in selected),
            "base_line_distance_max_m": max(x["base_to_segment_distance_m"] for x in selected),
            "base_line_filter_pass_count": sum(x["base_to_segment_distance_m"] >= 0.06 for x in selected),
            "base_side_sign_counts": {str(sign): int(sum(np.sign(x["base_side_signed_cross_m2"]) == sign for x in selected))
                                      for sign in (-1.0, 1.0)},
            "waypoint_count": sum(x["waypoint_xy_m"] is not None for x in selected),
        }
    return {"audit_type": "retrospective_from_saved_native_initial_state_and_trace",
            "does_not_claim_original_metadata": True, "episodes": rows, "classes": classes,
            "common_filter_parameters": {"table_bounds_xy_m": [-0.25, -0.25, 0.25, 0.35],
                "minimum_transport_distance_m": 0.14, "minimum_base_line_distance_m": 0.06,
                "waypoint_offset_range_m": [0.20, 0.25], "waypoint_lead_fraction": 0.25,
                "waypoint_same_base_side_only": True}}


def write_audit(dataset_dir, destination):
    audit = audit_pilot(dataset_dir)
    Path(destination).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def source_fingerprint(repo_root):
    repo = Path(repo_root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    modules = ("pickcube_pilot.py", "pickcube_expert.py", "pickcube_obstacle.py", "layout_geometry.py")
    pins = {}
    for name in modules:
        path = f"examples/baselines/constrain/{name}"
        content = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=repo)
        pins[path] = hashlib.sha256(content).hexdigest()
    return {"git_commit": revision, "source_hash_basis": "git-show committed source bytes",
            "files": pins}


def current_source_fingerprint(repo_root):
    repo = Path(repo_root)
    modules = ("pickcube_pilot.py", "pickcube_expert.py", "pickcube_obstacle.py",
               "layout_geometry.py", "pickcube_provenance.py")
    return {"git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "source_hash_basis": "actual current working-tree source bytes before collection",
            "files": {f"examples/baselines/constrain/{name}": hashlib.sha256(
                (repo / "examples/baselines/constrain" / name).read_bytes()).hexdigest()
                for name in modules}}


def frozen_recipe(dataset_dir):
    root = Path(dataset_dir)
    config = json.loads((root / "collection_config.json").read_text())
    return {
        "seed": config["seed"], "target_per_class": config["target_per_class"],
        "max_attempts_per_class": config["max_attempts_per_class"],
        "max_control_steps_per_attempt": config["max_steps"],
        "max_wall_seconds": config["max_wall_seconds"],
        "rng": "numpy default_rng(seed + class_index), class order direct then obstacle",
        "start_sampling_xy_m": {"x_uniform": [-0.18, -0.02], "y_uniform": [-0.06, 0.12]},
        "goal_sampling_xy_m": {"x_uniform": [0.02, 0.18], "y_uniform": [-0.06, 0.12]},
        "geometry_candidate_attempt_limit": 32,
        "layout": {"table_bounds_xy_m": [-0.25, -0.25, 0.25, 0.35],
                   "minimum_transport_distance_m": 0.14, "minimum_base_line_distance_m": 0.06,
                   "waypoint_offset_uniform_m": config["waypoint_offset_range"],
                   "waypoint_lead_fraction": 0.25, "waypoint_candidates_per_call": 32,
                   "waypoint_side": "same signed cross-product side as Panda base; midpoint + 0.25*SG along + offset perpendicular"},
        "obstacle": {"shape": "vertical cylinder", "radius_m": config["obstacle_radius"],
                     "height_m": config["obstacle_height"], "candidate_t_fraction": [0.40, 0.60],
                     "candidate_t_count": 21, "opposite_side_offset_uniform_m": [0.0, 0.06],
                     "candidate_offset_count": 25, "blocking_segment_distance_max_m": 0.063,
                     "endpoint_exclusion_distance_m": 0.068,
                     "table_bounds_xy_m": [-0.25, -0.25, 0.25, 0.35],
                     "cube_sweep_clearance_min_m": 0.063,
                     "robot_collision_vertex_sweep_clearance_min_m": config["obstacle_radius"],
                     "ranking": "maximum measured full robot-sweep clearance; rerun decides"},
        "execution": {"robot": "Panda", "robot_init_qpos_noise": 0.02,
                      "sim_backend": "physx_cpu", "obs_mode": "state",
                      "control_mode": "pd_ee_delta_pose", "action_dim": 7,
                      "translation": "delta position / 0.1m, clamped +/-0.35",
                      "rotation": "XYZ Euler delta divided by signed controller rot_lower then clamped +/-0.35",
                      "gripper_index": 6, "max_steps_per_waypoint": 80,
                      "waypoint_tolerance_m": 0.006, "action_cap": 0.35,
                      "lift_height_m": 0.12, "grasp_steps": 10, "settle_steps": 10,
                      "grasp_geometry": "official OBB grasp, approach [0,0,-1], depth 0.025m",
                      "phases": ["grasp_approach", "grasp_pose", "gripper_close", "lift", "waypoint_if_obstacle", "goal", "settle"],
                      "obstacle_phases": "direct successful template then restored obstacle execution within shared max_steps"},
        "safety": {"contact_impulse_threshold": 1e-6, "safety_margin_m": 0.005,
                    "robot_links_and_cube_contact_checked": True,
                    "cube_forbidden_radius": "cylinder radius + cube XY collision envelope + configured safety margin; source-pinned implementation",
                    "continuous_cube_segment_check_each_simulation_step": True,
                    "native_success_required": True, "timeout_is_success": False,
                    "first_success_control_step_safety_included": True},
        "data": {"state": "native state dictionary plus pickcube_obstacle and pickcube_goal; not flattened-index assumptions",
                 "obstacle_observation": "world XY center and radius in meters, raw/un-normalized",
                 "obs_horizon": config["obs_horizon"], "prediction_horizon": config["pred_horizon"],
                 "execution_horizon": 8, "actions_T_observations_T_plus_1": True},
    }


if __name__ == "__main__":
    dataset = Path(sys.argv[1])
    repository = Path.cwd()
    recipe = frozen_recipe(dataset)
    source = source_fingerprint(repository)
    provenance = {"classification": "retrospective audit supplement; not original collection-time metadata",
                  "recipe": recipe, "recipe_sha256": canonical_json_hash(recipe),
                  "committed_generator_source": source,
                  "current_source_sha256": {name: hashlib.sha256((repository / name).read_bytes()).hexdigest()
                                             for name in source["files"]},
        "existing_artifacts_sha256": {
                      str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in [dataset / name for name in ("trajectory.h5", "trajectory.json", "collection_config.json", "contract.json", "report.json", "artifact_hashes.json")]
                      + [repository / "outputs/exp/20261022_130421_pilot" / "validation/minibatch_validation.json",
                         repository / "outputs/exp/20261022_130421_pilot" / "diagnostics/pilot_evidence.json",
                         repository / "outputs/exp/20261022_130421_pilot" / "videos/direct.mp4",
                         repository / "outputs/exp/20261022_130421_pilot" / "videos/obstacle.mp4"]
                  }}
    (dataset / "provenance_supplement.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    write_audit(dataset, dataset / "base_xy_side_audit.json")
