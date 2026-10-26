"""Phase-A physics instrumentation for the retained Round-19 terminal traces."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

from examples.baselines.constrain.placesphere_matched_terminal import (
    NATIVE_ANG_THRESH_RAD_S,
    NATIVE_LIN_THRESH_M_S,
    _kill_new_children,
    _open_scene,
    _planned_cylinder,
    _scene_row,
    park_execution_snapshot,
)

R19 = Path("outputs/exp/20261026_100200_placesphere_matched_terminal_r19")
CASES = (
    ("confirmation", "placesphere-085", "positive_direct", "parked_full"),
    ("confirmation", "placesphere-046", "terminal_replay", "actual_suffix"),
    ("confirmation", "placesphere-123", "terminal_replay", "actual_suffix"),
    ("dev30", "placesphere-003", "positive_direct", "parked_full"),
    ("dev30", "placesphere-006", "terminal_replay", "actual_suffix"),
    ("dev30", "placesphere-019", "positive_direct", "parked_full"),
    ("dev30", "placesphere-000", "positive_direct", "parked_measurement"),
)


def classify_motion(linear_speed, angular_speed):
    linear_over = float(linear_speed) > NATIVE_LIN_THRESH_M_S
    angular_over = float(angular_speed) > NATIVE_ANG_THRESH_RAD_S
    if linear_over and angular_over:
        cause = "linear_and_angular"
    elif linear_over:
        cause = "linear"
    elif angular_over:
        cause = "angular"
    else:
        cause = "within_native_thresholds"
    return {
        "linear_over_threshold": linear_over,
        "angular_over_threshold": angular_over,
        "cause": cause,
    }


def _norm(value):
    return float(np.linalg.norm(value[0].detach().cpu().numpy()))


def _controller_identity(base):
    controllers = base.agent.controller.controllers
    result = {"control_mode": "pd_ee_delta_pose", "action_width": 7}
    for key, controller in controllers.items():
        config = controller.config
        result[key] = {
            name: getattr(config, name, None)
            for name in ("normalize_action", "use_delta", "use_target", "pos_lower", "pos_upper",
                         "rot_lower", "rot_upper", "gripper_lower", "gripper_upper", "frame")
            if hasattr(config, name)
        }
    return result


def _measure(base, scene, shared_z, step, terminated=False, truncated=False):
    info = base.evaluate()
    sphere = base.obj.pose.p[0].detach().cpu().numpy()
    bin_pose = base.bin.pose.p[0].detach().cpu().numpy()
    linear = _norm(base.obj.linear_velocity)
    angular = _norm(base.obj.angular_velocity)
    tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
    from examples.baselines.constrain.placesphere_transport_detour import _collision_bodies
    from examples.baselines.constrain.placesphere_matched_terminal import lowest_hand_vertex_z
    hand_low = lowest_hand_vertex_z(_collision_bodies(base))
    top_z = float(scene["cylinder"]["height"])
    hand_below = float(tcp[2] - hand_low)
    geometric_z = top_z + hand_below + 0.015
    return {
        "step": int(step), "elapsed_native_steps": int(step),
        "sphere_xyz": sphere.tolist(), "bin_xyz": bin_pose.tolist(),
        "bin_xy_error_m": float(np.linalg.norm(sphere[:2] - bin_pose[:2])),
        "bin_z_error_m": float(abs(sphere[2] - (bin_pose[2] + float(base.block_half_size[0]) + float(base.radius)))),
        "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
        "linear_threshold_m_s": NATIVE_LIN_THRESH_M_S,
        "angular_threshold_rad_s": NATIVE_ANG_THRESH_RAD_S,
        "motion_class": classify_motion(linear, angular),
        "on_bin": bool(info["is_obj_on_bin"][0].item()),
        "grasped": bool(info["is_obj_grasped"][0].item()),
        "native_success": bool(info["native_success"][0].item()),
        "obstacle_contact_latched": bool(base._obstacle_collision[0].item()),
        "obstacle_crossing_latched": bool(base._forbidden_crossing[0].item()),
        "terminated": bool(terminated), "truncated": bool(truncated),
        "tcp_xyz": tcp.tolist(), "hand_low_z_m": hand_low,
        "hand_below_tcp_m": hand_below, "cylinder_top_z_m": top_z,
        "measured_geometric_release_tcp_z_m": geometric_z,
        "shared_release_tcp_z_m": shared_z,
        "shared_minus_measured_geometric_z_m": None if shared_z is None else float(shared_z - geometric_z),
        "measured_hand_edge_gap_at_shared_z_m": None if shared_z is None else float(shared_z - hand_below - top_z),
        "hand_edge_gap_surface_proof": False,
    }


def _trace_record(env, base, scene, actions, *, enable_actual, shared_z, continue_steps, suffix_start=0):
    import torch
    from examples.baselines.constrain.placesphere_obstacle import PARKED_CENTER
    from examples.baselines.constrain.pickcube_expert import restore_execution_state, snapshot_execution_state

    actions = np.asarray(actions, dtype=np.float32)
    trace = []
    first_native = None
    contact_at_start = bool(base._obstacle_collision[0].item())
    for index, action in enumerate(actions):
        _, _, term, trunc, _ = env.step(torch.as_tensor(action, device=base.device))
        sample = _measure(base, scene, shared_z, index + 1, term, trunc)
        sample["action_index"] = int(index)
        sample["action"] = action.tolist()
        sample["action_phase"] = "saved_suffix" if index >= suffix_start else "saved_prefix"
        trace.append(sample)
        if sample["native_success"] and first_native is None:
            first_native = index + 1
        if term or trunc:
            break
    replay_state = snapshot_execution_state(base)
    replay_sample = trace[-1] if trace else _measure(base, scene, shared_z, 0)
    suffix_hash = hashlib.sha256(np.ascontiguousarray(actions[suffix_start:]).tobytes()).hexdigest()
    continuation = []
    for offset in range(continue_steps):
        _, _, term, trunc, _ = env.step(torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device=base.device))
        sample = _measure(base, scene, shared_z, len(actions) + offset + 1, term, trunc)
        sample["action"] = [0, 0, 0, 0, 0, 0, 1]
        sample["action_phase"] = "open_neutral_diagnostic_only"
        continuation.append(sample)
        if term or trunc or sample["native_success"]:
            break
    return {
        "controller_identity": _controller_identity(base),
        "scene_identity": {
            "scene_id": scene["scene_id"], "seed": int(scene["seed"]),
            "cylinder": scene["cylinder"], "catalog_role": "actual" if enable_actual else "parked",
            "constructed_cylinder_height_m": float(base.obstacle_collision_height),
            "obstacle_enabled": bool(base._obstacle_enabled[0].item()),
        },
        "episode_start_contact_latch": contact_at_start,
        "saved_actions_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "saved_actions_count": int(len(actions)), "suffix_start_index": int(suffix_start),
        "suffix_count": int(len(actions) - suffix_start), "suffix_actions_sha256": suffix_hash,
        "first_native_success_step": first_native,
        "saved_trace": trace, "saved_trace_terminal": replay_sample,
        "continuation_budget_steps": int(continue_steps), "open_neutral_continuation": continuation,
        "continuation_native_success_step": next((row["step"] for row in continuation if row["native_success"]), None),
        "continuation_contact": any(row["obstacle_contact_latched"] for row in continuation),
        "full_prefix_safety_claim": False,
        "probe_scope": "diagnostic replay and suffix only; no full-prefix admission",
    }


def _find_row(document, scene_id):
    return next(row for row in document["rows"] if row["layout"]["scene_id"] == scene_id)


def diagnose_case(source, scene_id, role_key, mode, *, episode_wall_s=160, settle_steps=200):
    import torch
    from examples.baselines.constrain.pickcube_expert import restore_execution_state
    from examples.baselines.constrain.placesphere_obstacle import PARKED_CENTER

    doc = json.loads((R19 / ("confirmation.json" if source == "confirmation" else "dev30_validation.json")).read_text())
    row = _find_row(doc, scene_id)
    layout = row["layout"]
    role = row[role_key]
    actions = role.get("actions")
    direct_actions = (row.get("positive_direct") or {}).get("actions")
    if actions is None:
        return {"scene_id": scene_id, "complete": False, "error": "saved_actions_missing"}
    action_array = np.asarray(actions, dtype=np.float32)
    if direct_actions is not None:
        direct_actions = np.asarray(direct_actions, dtype=np.float32)
    shared_z = role.get("release_tcp_z")
    if shared_z is None:
        shared_z = (row.get("positive_obstacle") or {}).get("release_tcp_z")
    start = time.monotonic()
    before = set(__import__("examples.baselines.constrain.placesphere_matched_terminal", fromlist=["_child_pids"])._child_pids())
    env = None
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("episode wall exceeded")))
    signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
    try:
        env, scene, initial = _open_scene(layout)
        base = env.unwrapped
        # The public r19 replay action list itself is the authoritative suffix.
        replay_actions = row.get("terminal_replay", {}).get("actions")
        if mode == "actual_suffix":
            if replay_actions is None or direct_actions is None:
                return {"scene_id": scene_id, "complete": False, "error": "suffix_actions_missing"}
            suffix = np.asarray(replay_actions, dtype=np.float32)
            if suffix.ndim != 2 or suffix.shape[1] != 7 or len(suffix) > len(direct_actions):
                return {"scene_id": scene_id, "complete": False, "error": "suffix_shape_or_length"}
            suffix_start = len(direct_actions) - len(suffix)
            if not np.allclose(direct_actions[suffix_start:], suffix, atol=0, rtol=0):
                return {"scene_id": scene_id, "complete": False, "error": "suffix_not_direct_tail"}
            parked_snapshot, _ = park_execution_snapshot(initial)
            restore_execution_state(env, parked_snapshot)
            for action in direct_actions[:suffix_start]:
                env.step(torch.as_tensor(action, device=base.device))
            snapshot = __import__("examples.baselines.constrain.pickcube_expert", fromlist=["snapshot_execution_state"]).snapshot_execution_state(env)
            restore_execution_state(env, snapshot)
            base._obstacle_enabled[:] = True
            base._set_obstacle_pose(np.asarray(scene["cylinder"]["center_xy"], dtype=np.float64))
            base._obstacle_collision[:] = False
            base._forbidden_crossing[:] = False
            base._first_obstacle_contact = None
            result = _trace_record(env, base, scene, suffix, enable_actual=True, shared_z=shared_z,
                                   continue_steps=settle_steps, suffix_start=0)
            result["prefix_replayed_parked_actions"] = int(suffix_start)
            result["prefix_replay_contact_parked"] = False
        else:
            parked_snapshot, _ = park_execution_snapshot(initial)
            restore_execution_state(env, parked_snapshot)
            result = _trace_record(env, base, scene, action_array, enable_actual=False, shared_z=shared_z,
                                   continue_steps=settle_steps if mode == "parked_full" else 0)
            if mode == "parked_measurement":
                from examples.baselines.constrain.placesphere_matched_terminal import lowest_hand_vertex_z
                from examples.baselines.constrain.placesphere_transport_detour import _collision_bodies
                tcp_z = float(base.agent.tcp.pose.p[0, 2].item())
                hand_low = lowest_hand_vertex_z(_collision_bodies(base))
                top = float(scene["cylinder"]["height"])
                result["measurement_rejection"] = {
                    "shared_release_tcp_z_m": shared_z,
                    "tcp_z_at_rejection_m": tcp_z,
                    "hand_low_z_m": hand_low,
                    "hand_below_tcp_m": tcp_z - hand_low,
                    "cylinder_top_z_m": top,
                    "measured_geometric_z_plus_prescribed_margin_m": top + (tcp_z - hand_low) + 0.015,
                    "shared_minus_required_geometric_z_m": None if shared_z is None else shared_z - (top + tcp_z - hand_low + 0.015),
                    "edge_gap_at_shared_target_m": None if shared_z is None else shared_z - (tcp_z - hand_low) - top,
                    "vertex_measurement_is_surface_proof": False,
                }
        result.update({"scene_id": scene_id, "source_split": source, "role": role_key, "mode": mode,
                       "scene_identity": {**result["scene_identity"], "catalog": layout["catalog"],
                                          "index": layout["index"]},
                       "elapsed_s": time.monotonic() - start, "episode_wall_budget_s": episode_wall_s,
                       "native_thresholds_changed": False})
        return result
    except Exception as exc:
        return {"scene_id": scene_id, "complete": False, "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": time.monotonic() - start}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if env is not None:
            env.close()
        _kill_new_children(before)


def run(output_dir, *, episode_wall_s=160, phase_wall_s=1400, settle_steps=200):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source_files = (
        "examples/baselines/constrain/placesphere_terminal_diagnosis.py",
        "examples/baselines/constrain/placesphere_matched_terminal.py",
    )
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_files}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    prereg = {
        "phase": "A_diagnosis_before_motion_fix", "identity": identity, "files": hashes,
        "cases": [{"split": a, "scene_id": b, "role": c, "mode": d,
                   "selection_use": "calibration/confirmation train evidence" if a == "confirmation" else "dev, non-selection"}
                  for a, b, c, d in CASES],
        "episode_wall_s": episode_wall_s, "phase_wall_s": phase_wall_s,
        "continuation_budget_steps": settle_steps, "native_linear_threshold_m_s": NATIVE_LIN_THRESH_M_S,
        "native_angular_threshold_rad_s": NATIVE_ANG_THRESH_RAD_S,
        "native_thresholds_changed": False, "geometry_or_motion_changed": False,
        "argv": list(__import__("sys").argv),
    }
    (output / "phase_a_preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    deadline = time.monotonic() + phase_wall_s
    results = []
    for case in CASES:
        if time.monotonic() >= deadline:
            results.append({"scene_id": case[1], "complete": False, "error": "phase_wall_exceeded"})
            continue
        result = diagnose_case(*case, episode_wall_s=episode_wall_s, settle_steps=settle_steps)
        result["complete"] = "error" not in result
        results.append(result)
        (output / "phase_a_partial.json").write_text(json.dumps({"identity": identity, "results": results}, indent=2) + "\n")
    payload = {
        "complete": len(results) == len(CASES) and all(item.get("complete") for item in results),
        "identity": identity, "phase_a_preregistration": "phase_a_preregistration.json",
        "budgets": {"episode_wall_s": episode_wall_s, "phase_wall_s": phase_wall_s,
                    "continuation_steps": settle_steps},
        "results": results,
    }
    (output / "phase_a_diagnosis.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-wall-s", type=int, default=160)
    parser.add_argument("--phase-wall-s", type=int, default=1400)
    parser.add_argument("--settle-steps", type=int, default=200)
    args = parser.parse_args()
    payload = run(args.output_dir, episode_wall_s=args.episode_wall_s,
                  phase_wall_s=args.phase_wall_s, settle_steps=args.settle_steps)
    print(json.dumps({"complete": payload["complete"], "identity": payload["identity"],
                      "cases": len(payload["results"]), "output": args.output_dir}, sort_keys=True))
    if not payload["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
