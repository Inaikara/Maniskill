"""Training-layout physics comparison for PlaceSphere post-release neutral holds."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.placesphere_matched_diagnosis import sha256

H5 = Path("demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5")
MANIFEST = H5.with_name("trajectory.json")
HOLDS = (0, 8, 16)
TRAIN_OBSTACLE_IDS = (1, 3, 5, 7, 9, 11, 13, 15)
LIN_THRESHOLD = 0.01
ANG_THRESHOLD = 0.5


def _h5_tree(node, index):
    if isinstance(node, h5py.Group):
        return {key: _h5_tree(value, index) for key, value in node.items()
                if key != "snapshot_sha256"}
    return node[index]


def _bool(info, key):
    return bool(np.asarray(info[key]).reshape(-1)[0])


def _layout(record, seed):
    meta = record
    layout = meta["layout"]
    cylinder = layout["cylinder"]
    return {
        "seed": int(seed), "start_xy": layout["start_xy"],
        "container_xy": layout["container_xy"], "center_xy": cylinder["center_xy"],
        "radius": float(cylinder["radius"]), "height": float(cylinder["height"]),
    }


def choose_physics_hold(summaries):
    """Select only from complete fixed training-layout native physics evidence."""
    if set(summaries) != {str(value) for value in HOLDS} or any(
            int(summaries[str(value)]["count"]) != len(TRAIN_OBSTACLE_IDS) for value in HOLDS):
        raise ValueError("hold selection requires complete preregistered candidates on every frozen train layout")
    return min(HOLDS, key=lambda hold: (
        -int(summaries[str(hold)]["safe_native_success_count"]),
        -int(summaries[str(hold)]["terminal_speed_pass_count"]),
        hold,
    ))


def replay_with_hold(episode_id, hold_steps, *, episode_wall_s=180):
    if hold_steps not in HOLDS:
        raise ValueError("hold must be preregistered in (0, 8, 16)")
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    started = time.monotonic()
    with h5py.File(H5, "r") as source:
        group = source[f"traj_{episode_id}"]
        metadata = json.loads(group.attrs["metadata_json"])
        seed = int(group.attrs["seed"])
        if group.attrs["kind"] != "obstacle" or metadata["kind"] != "obstacle":
            raise ValueError(f"traj_{episode_id} is not an obstacle training trajectory")
        actions = np.asarray(group["actions"], dtype=np.float32)
        raw = np.asarray(group["raw_obs"])
        state = _h5_tree(group["env_states"], 0)
        controller = _h5_tree(group["controller_states"], 0)
        controller_hash = group["controller_states"]["snapshot_sha256"][0].decode()
    layout = _layout(metadata, seed)
    env = make_env(config=PlaceSpherePilotConfig(
        seed=layout["seed"], max_steps=max(1000, len(actions) + hold_steps + 8),
        obstacle_radius=layout["radius"], obstacle_height=layout["height"]))
    try:
        env.reset(seed=layout["seed"], options={
            "sphere_xy": layout["start_xy"], "container_xy": layout["container_xy"],
            "obstacle_enabled": True, "obstacle_center": layout["center_xy"]})
        base = env.unwrapped
        base.set_state_dict(state)
        base.agent.controller.set_state(controller)
        from mani_skill.utils.common import flatten_state_dict
        restored = flatten_state_dict(base._get_obs_state_dict(base.get_info()), use_torch=True).detach().cpu().numpy()[0]
        if not np.allclose(restored, raw[0], atol=1e-6, rtol=0):
            raise ValueError(f"traj_{episode_id}: saved native initial state did not restore")
        previous_grasped = _bool(base.evaluate(), "is_obj_grasped")
        release_step = None
        release_speed = None
        hold_trace = []
        contact = crossing = native_success = False
        stop_after_hold = False
        steps = 0
        for step, action in enumerate(actions):
            _, _, terminated, truncated, info = env.step(action[None])
            steps += 1
            now_grasped = _bool(info, "is_obj_grasped")
            released = bool(previous_grasped and not now_grasped)
            contact = bool(contact or _bool(info, "obstacle_collision"))
            crossing = bool(crossing or _bool(info, "obstacle_crossing"))
            native_success = bool(native_success or _bool(info, "native_success"))
            previous_grasped = now_grasped
            if released and release_step is None:
                release_step = steps
                release_speed = {
                    "linear_m_s": float(torch.linalg.norm(base.obj.linear_velocity, dim=1)[0].item()),
                    "angular_rad_s": float(torch.linalg.norm(base.obj.angular_velocity, dim=1)[0].item()),
                }
                for offset in range(hold_steps):
                    neutral = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
                    _, _, held_term, held_trunc, held_info = env.step(neutral[None])
                    steps += 1
                    lin_vec = base.obj.linear_velocity[0].detach().cpu().numpy()
                    ang_vec = base.obj.angular_velocity[0].detach().cpu().numpy()
                    held_native = _bool(held_info, "native_success")
                    hold_trace.append({
                        "hold_step": offset + 1,
                        "sphere_xyz": base.obj.pose.p[0].detach().cpu().numpy().astype(float).tolist(),
                        "linear_velocity_m_s": lin_vec.astype(float).tolist(),
                        "angular_velocity_rad_s": ang_vec.astype(float).tolist(),
                        "linear_speed_m_s": float(np.linalg.norm(lin_vec)),
                        "angular_speed_rad_s": float(np.linalg.norm(ang_vec)),
                        "on_bin": _bool(held_info, "is_obj_on_bin"),
                        "grasped": _bool(held_info, "is_obj_grasped"),
                        "native_success": held_native,
                    })
                    contact = bool(contact or _bool(held_info, "obstacle_collision"))
                    crossing = bool(crossing or _bool(held_info, "obstacle_crossing"))
                    native_success = bool(native_success or held_native)
                    if held_term or held_trunc or held_native:
                        stop_after_hold = True
                        break
                if stop_after_hold:
                    break
            if terminated or truncated:
                break
        terminal = base.evaluate()
        linear = float(torch.linalg.norm(base.obj.linear_velocity, dim=1)[0].item())
        angular = float(torch.linalg.norm(base.obj.angular_velocity, dim=1)[0].item())
        return {
            "episode_id": episode_id, "scene_id": metadata["expert_record"]["scene_id"], "seed": layout["seed"],
            "hold_steps": hold_steps, "actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "initial_controller_sha256": controller_hash, "release_step": release_step,
            "release_speed": release_speed, "hold_trace": hold_trace,
            "executed_steps_including_hold": steps, "native_success": _bool(terminal, "native_success"),
            "on_bin": _bool(terminal, "is_obj_on_bin"), "static": _bool(terminal, "is_obj_static"),
            "grasped": _bool(terminal, "is_obj_grasped"), "contact": contact, "crossing": crossing,
            "terminal_linear_speed_m_s": linear, "terminal_angular_speed_rad_s": angular,
            "terminal_native_speed_pass": linear <= LIN_THRESHOLD and angular <= ANG_THRESHOLD,
            "safe_native_success": bool(_bool(terminal, "native_success") and not contact and not crossing),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        env.close()


def run(output_dir, *, episode_wall_s=180, phase_wall_s=2400):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if sha256(H5) != "eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0":
        raise ValueError("matched training HDF5 identity mismatch")
    manifest = json.loads(MANIFEST.read_text())
    by_id = {int(row["episode_id"]): row for row in manifest["episodes"]}
    if any(by_id[index]["kind"] != "obstacle" for index in TRAIN_OBSTACLE_IDS):
        raise ValueError("frozen training probe selection contains a non-obstacle ID")
    sources = [Path(__file__), Path(__file__).with_name("placesphere_matched_diagnosis.py"),
               Path(__file__).with_name("placesphere_pilot.py")]
    prereg = {
        "kind": "matched_training_layout_release_hold_physics_probe",
        "h5_path": str(H5), "h5_sha256": sha256(H5),
        "manifest_path": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
        "training_obstacle_ids": list(TRAIN_OBSTACLE_IDS), "hold_candidates_steps": list(HOLDS),
        "comparison": "same saved full initial native/controller state and same native expert action sequence; insert only open-neutral hold after first true grasp-to-ungrasp transition",
        "thresholds_unchanged": {"linear_m_s": LIN_THRESHOLD, "angular_rad_s": ANG_THRESHOLD},
        "selection_rule": "Among candidates with complete 8/8 runs, prefer greater safe-native-success count; ties by greater terminal native speed-threshold pass count, then smaller hold. No dev labels or results enter selection.",
        "episode_wall_seconds": episode_wall_s, "phase_wall_seconds": phase_wall_s,
        "source_hashes": {str(p.relative_to(Path.cwd())): sha256(p) for p in sources},
        "argv": __import__("sys").argv,
    }
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    deadline = time.monotonic() + phase_wall_s
    rows = []
    for episode_id in TRAIN_OBSTACLE_IDS:
        for hold in HOLDS:
            if time.monotonic() >= deadline:
                rows.append({"episode_id": episode_id, "hold_steps": hold, "status": "phase_timeout"})
                break
            previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("episode wall exceeded")))
            signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
            try:
                result = replay_with_hold(episode_id, hold, episode_wall_s=episode_wall_s)
                result["status"] = "evaluated"
            except Exception as exc:
                result = {"episode_id": episode_id, "hold_steps": hold, "status": "runtime_error",
                          "error": f"{type(exc).__name__}: {exc}"}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            rows.append(result)
            (output / "partial.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    complete = len(rows) == len(TRAIN_OBSTACLE_IDS) * len(HOLDS) and all(r["status"] == "evaluated" for r in rows)
    summaries = {}
    for hold in HOLDS:
        sample = [r for r in rows if r.get("hold_steps") == hold and r["status"] == "evaluated"]
        summaries[str(hold)] = {
            "count": len(sample), "safe_native_success_count": sum(r["safe_native_success"] for r in sample),
            "native_success_count": sum(r["native_success"] for r in sample),
            "contact_count": sum(r["contact"] for r in sample), "crossing_count": sum(r["crossing"] for r in sample),
            "terminal_speed_pass_count": sum(r["terminal_native_speed_pass"] for r in sample),
            "median_terminal_angular_speed_rad_s": None if not sample else float(np.median([r["terminal_angular_speed_rad_s"] for r in sample])),
        }
    report = {"complete": complete, "rows": rows, "summaries": summaries,
              "selection_is_train_physics_only": True,
              "selected_hold_steps": choose_physics_hold(summaries) if complete else None,
              "budgets": {"phase_wall_seconds": phase_wall_s, "episode_wall_seconds": episode_wall_s}}
    (output / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if not complete:
        raise RuntimeError("hold physics probe incomplete; failures retained in denominator")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-wall-s", type=int, default=180)
    parser.add_argument("--phase-wall-s", type=int, default=2400)
    args = parser.parse_args(argv)
    report = run(args.output_dir, episode_wall_s=args.episode_wall_s, phase_wall_s=args.phase_wall_s)
    print(json.dumps({"complete": report["complete"], "summaries": report["summaries"],
                      "output": args.output_dir}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
