"""Exploratory offline diagnosis of saved fixed-validation Plug event traces.

All thresholds here are diagnostic landmarks, not task-success gates. The native
5 mm / 0.2 rad criterion remains the only success criterion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

CHECKPOINTS = (20_000, 40_000, 60_000, 80_000, 100_000)
EVENT_FIELDS = {"control", "policy_input_46d", "action_7d", "observation_46d", "native_state", "native_info"}


def _pose(state, actor):
    value = np.asarray(state["actors"][actor][0], dtype=float)
    if value.shape != (13,) or not np.isfinite(value).all():
        raise ValueError(f"invalid native actor state: {actor}")
    return value[:3], value[3:7]


def _angle(q, r):
    q, r = np.asarray(q, float), np.asarray(r, float)
    qn, rn = np.linalg.norm(q), np.linalg.norm(r)
    if qn == 0 or rn == 0: raise ValueError("zero quaternion in native state")
    return float(2 * math.acos(float(np.clip(abs(np.dot(q / qn, r / rn)), 0, 1))))


def _quat_multiply(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.asarray([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                       aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw], dtype=float)


def _relative_pose(obs):
    tcp_p, tcp_q = obs[18:21], obs[21:25]
    obj_p, obj_q = obs[25:28], obs[28:32]
    tcp_norm, obj_norm = np.linalg.norm(tcp_q), np.linalg.norm(obj_q)
    if tcp_norm < 1e-8 or obj_norm < 1e-8:
        raise ValueError("zero quaternion in policy observation")
    tcp_q = tcp_q / tcp_norm; obj_q = obj_q / obj_norm
    inv_tcp_q = tcp_q * np.asarray([1, -1, -1, -1])
    relative_p = _quat_multiply(_quat_multiply(inv_tcp_q, np.r_[0.0, obj_p-tcp_p]), tcp_q)[1:]
    relative_q = _quat_multiply(inv_tcp_q, obj_q)
    return obj_p - tcp_p, relative_p, relative_q


def _info_scalar(info, key):
    value = np.asarray(info[key], dtype=float).reshape(-1)
    if len(value) != 1 or not np.isfinite(value[0]): raise ValueError(f"invalid native_info.{key}")
    return float(value[0])


def analyze_episode(path: Path, *, claimed_success: bool):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows: raise ValueError(f"empty trace: {path}")
    if any(set(row) != EVENT_FIELDS for row in rows): raise ValueError(f"unexpected event schema: {path}")
    if [r["control"] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"non-contiguous controls: {path}")
    metrics = []
    initial_obs = np.asarray(rows[0]["policy_input_46d"], dtype=float)
    if initial_obs.shape != (46,) or not np.isfinite(initial_obs).all():
        raise ValueError(f"invalid initial policy input: {path}")
    # The first event is post-action. The paired pre-action observation retains
    # the true restored initial charger pose, avoiding a one-control lift offset.
    init_z = float(initial_obs[27])
    for row in rows:
        obs = np.asarray(row["policy_input_46d"], float)
        action = np.asarray(row["action_7d"], float)
        nxt = np.asarray(row["observation_46d"], float)
        if obs.shape != (46,) or nxt.shape != (46,) or action.shape != (7,) or not all(np.isfinite(x).all() for x in (obs, action, nxt)):
            raise ValueError(f"invalid control vectors: {path}")
        cp, cq = _pose(row["native_state"], "charger")
        goal = np.asarray(row["native_state"]["plug_orientation_target"][0], float)
        if goal.shape != (4,) or not np.isfinite(goal).all(): raise ValueError("invalid target state")
        yaw, gp = float(goal[0]), goal[1:]
        goalq = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        tcp = obs[18:21]
        observed_charger = obs[25:28]
        world_offset, tcp_offset, relative_q = _relative_pose(obs)
        info = row["native_info"]
        reported_position = _info_scalar(info, "obj_to_goal_dist")
        reported_rotation = _info_scalar(info, "obj_to_goal_angle")
        computed_position = float(np.linalg.norm(cp - gp))
        computed_rotation = _angle(cq, goalq)
        if abs(reported_position - computed_position) > 1e-5 or abs(reported_rotation - computed_rotation) > 1e-5:
            raise ValueError(f"native state/info pose disagreement at control {row['control']}: {path}")
        metrics.append({
            "control": row["control"], "position_error_m": reported_position,
            "rotation_error_rad": reported_rotation,
            "computed_position_error_m": computed_position,
            "computed_rotation_error_rad": computed_rotation,
            "charger_minus_goal_position_world_xyz_m": (cp - gp).tolist(),
            "tcp_to_charger_center_m": float(np.linalg.norm(tcp - observed_charger)),
            "object_minus_tcp_world_xyz_m": world_offset.tolist(),
            "object_minus_tcp_tcp_frame_xyz_m": tcp_offset.tolist(),
            "object_relative_quaternion_tcp_frame_wxyz": relative_q.tolist(),
            "charger_lift_m": float(cp[2] - init_z), "gripper_action": float(action[6]),
            "normalized_translation_action_xyz": action[:3].tolist(),
            "normalized_rotation_action_xyz": action[3:6].tolist(),
            "tcp_displacement_m": float(np.linalg.norm(nxt[18:21] - obs[18:21])),
            "native_success": bool(np.asarray(info.get("success", [False])).reshape(-1)[0]),
        })
    native_success = metrics[-1]["native_success"]
    if native_success != claimed_success: raise ValueError(f"trace/episode outcome mismatch: {path}")
    lifted = [m["control"] for m in metrics if m["charger_lift_m"] >= .02]
    lifted_persistently = any(all(m["charger_lift_m"] >= .02 for m in metrics[i:i+5]) for i in range(max(0, len(metrics)-4)))
    near = [m for m in metrics if m["position_error_m"] <= .05]
    aligned = [m for m in near if m["rotation_error_rad"] <= .2]
    closest = min(metrics, key=lambda m: m["position_error_m"])
    best_full = min(metrics, key=lambda m: math.hypot(m["position_error_m"] / .005, m["rotation_error_rad"] / .2))
    first_near = next((i for i, m in enumerate(metrics) if m["tcp_to_charger_center_m"] <= .03), None)
    if first_near is None:
        transform_drift = None
    else:
        ref = metrics[first_near]
        tracked = metrics[first_near:]
        transform_drift = {
            "reference_control": ref["control"],
            "max_translation_change_m": max(float(np.linalg.norm(np.asarray(m["object_minus_tcp_tcp_frame_xyz_m"]) - ref["object_minus_tcp_tcp_frame_xyz_m"])) for m in tracked),
            "max_relative_rotation_change_rad": max(_angle(m["object_relative_quaternion_tcp_frame_wxyz"], ref["object_relative_quaternion_tcp_frame_wxyz"]) for m in tracked),
            "availability": "kinematic relative-pose stability only; not a contact/grasp sensor",
        }
    if native_success: phase = "official_success"
    elif not lifted_persistently: phase = "no_sustained_20mm_lift_evidence"
    elif not near: phase = "lifted_no_50mm_goal_approach"
    elif not aligned: phase = "goal_near_rotation_over_0.2rad"
    else: phase = "aligned_approach_but_official_position_or_angle_gate"
    return {"controls": len(rows), "official_success": native_success, "failure_phase": phase,
            "sustained_20mm_lift_evidence": lifted_persistently,
            "first_tcp_within_30mm_charger_center_control": next((m["control"] for m in metrics if m["tcp_to_charger_center_m"] <= .03), None),
            "object_tcp_transform_drift_after_first_30mm_proximity": transform_drift,
            "first_50mm_goal_approach_control": near[0]["control"] if near else None,
            "first_full_alignment_within_50mm_control": aligned[0]["control"] if aligned else None,
            "closest_position_control": closest["control"], "closest_position_error_m": closest["position_error_m"],
            "closest_position_rotation_error_rad_same_control": closest["rotation_error_rad"],
            "closest_position_residual_world_xyz_m": closest["charger_minus_goal_position_world_xyz_m"],
            "best_normalized_full_pose_control": best_full["control"],
            "best_normalized_full_pose_position_error_m": best_full["position_error_m"],
            "best_normalized_full_pose_rotation_error_rad": best_full["rotation_error_rad"],
            "best_normalized_full_pose_residual_world_xyz_m": best_full["charger_minus_goal_position_world_xyz_m"],
            "last10": metrics[-10:], "last50": metrics[-50:],
            "gripper_action_quantiles": np.quantile([m["gripper_action"] for m in metrics], [0, .1, .5, .9, 1]).tolist(),
            "gripper_action_quantiles_last50": np.quantile([m["gripper_action"] for m in metrics[-50:]], [0, .1, .5, .9, 1]).tolist(),
            "tcp_control_displacement_median_last50": float(np.median([m["tcp_displacement_m"] for m in metrics[-50:]])),
            "native_grasp_status": None, "native_grasp_status_availability": "not recorded; lift is only transport evidence"}


def analyze(root: Path):
    results = {}
    for step in CHECKPOINTS:
        folder = root / f"validation_{step:06d}"
        episodes = [json.loads(line) for line in (folder / "episodes.jsonl").read_text().splitlines() if line.strip()]
        if len(episodes) != 50 or [e["index"] for e in episodes] != list(range(50)):
            raise ValueError(f"checkpoint {step}: expected 50 ordered episode records")
        rows = []
        for ep in episodes:
            event_path = folder / f"episode_{ep['index']:03d}.events.jsonl"
            episode = analyze_episode(event_path, claimed_success=bool(ep["success"]))
            if episode["controls"] != int(ep["executed_controls"]):
                raise ValueError(f"checkpoint {step} episode {ep['index']}: event/episode control count mismatch")
            rows.append({"index": ep["index"], "scene_seed": ep["seed"], "policy_seed": ep["policy_seed"],
                         **episode,
                         "event_sha256": hashlib.sha256(event_path.read_bytes()).hexdigest()})
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("episodes") != 50 or summary.get("successes") != sum(e["official_success"] for e in rows):
            raise ValueError(f"checkpoint {step}: episode traces do not match registered summary")
        results[str(step)] = {"successes": sum(e["official_success"] for e in rows),
                              "input_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                               for name in ("episodes.jsonl", "summary.json")},
                              "failure_phase_counts": {phase: sum(e["failure_phase"] == phase for e in rows)
                                                       for phase in sorted({e["failure_phase"] for e in rows})},
                              "episodes": rows}
    paired = []
    by_step = {int(k): {r["scene_seed"]: r for r in v["episodes"]} for k, v in results.items()}
    for seed in sorted(by_step[80_000]):
        a, b = by_step[80_000][seed], by_step[100_000][seed]
        paired.append({"scene_seed": seed, "80k_success": a["official_success"], "100k_success": b["official_success"],
                       "80k_phase": a["failure_phase"], "100k_phase": b["failure_phase"],
                       "80k_closest_position_error_m": a["closest_position_error_m"],
                       "100k_closest_position_error_m": b["closest_position_error_m"],
                       "80k_best_full_pose_position_error_m": a["best_normalized_full_pose_position_error_m"],
                       "100k_best_full_pose_position_error_m": b["best_normalized_full_pose_position_error_m"],
                       "80k_best_full_pose_rotation_error_rad": a["best_normalized_full_pose_rotation_error_rad"],
                       "100k_best_full_pose_rotation_error_rad": b["best_normalized_full_pose_rotation_error_rad"]})
    return {"format": "plug_base_validation_failure_analysis_v1", "exploratory_posthoc": True,
            "input_root": str(root), "native_success_gate": {"position_m_max": .005, "full_quaternion_angle_rad_max": .2},
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "diagnostic_thresholds_only": {"tcp_charger_center_m": .03, "sustained_lift_m": .02,
                                            "sustained_controls": 5, "goal_approach_m": .05,
                                            "alignment_angle_rad": .2},
            "quaternion_units": "radians; sign-invariant full-quaternion angle", "checkpoints": results,
            "paired_80k_vs_100k_same_scene_seed": paired,
            "interpretation_limit": "Phase labels are geometric landmarks, not causes; no contact/grasp force sensor was saved. Actions are native normalized 7-D values; TCP displacement is measured in meters separately."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__": main()
