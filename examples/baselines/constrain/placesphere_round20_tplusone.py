"""One matched direct/obstacle T+1 checkpoint and restore proof after the dev gate."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from examples.baselines.constrain.placesphere_matched_terminal import (
    EXPERT_MAX_STEPS,
    ROLE_NEGATIVE_DIRECT,
    ROLE_POSITIVE_DIRECT,
    ROLE_POSITIVE_OBSTACLE,
    _common_release_targets,
    _public_record,
    _write_json,
    collection_admission,
    freeze_identity,
    replay_direct_terminal,
    run_layout_expert,
)


def _run_role_pair(layout, *, episode_wall_s):
    from examples.baselines.constrain.placesphere_matched_terminal import (
        ABOVE_RELEASE_MARGINS_M,
    )

    margin = 0.015
    if margin not in ABOVE_RELEASE_MARGINS_M:
        raise RuntimeError("frozen margin is absent")
    obstacle = run_layout_expert(layout, margin_m=margin, role=ROLE_POSITIVE_OBSTACLE,
                                 episode_wall_s=episode_wall_s)
    direct = run_layout_expert(layout, margin_m=margin, role=ROLE_POSITIVE_DIRECT,
                               shared_targets=obstacle.get("shared_targets"),
                               episode_wall_s=episode_wall_s)
    targets = _common_release_targets(obstacle.get("shared_targets") or {}, direct, obstacle)
    if targets is None:
        raise RuntimeError("paired positive role clearance measurement missing")
    targets["release_hold_steps"] = 0
    old_z = obstacle.get("release_tcp_z")
    if old_z is None or abs(float(targets["release_tcp_z"]) - float(old_z)) > 1e-6:
        obstacle = run_layout_expert(layout, margin_m=margin, role=ROLE_POSITIVE_OBSTACLE,
                                     shared_targets=targets, episode_wall_s=episode_wall_s)
        direct = run_layout_expert(layout, margin_m=margin, role=ROLE_POSITIVE_DIRECT,
                                   shared_targets=targets, episode_wall_s=episode_wall_s)
    replay = replay_direct_terminal(layout, direct, episode_wall_s=episode_wall_s)
    negative = run_layout_expert(layout, margin_m=margin, role=ROLE_NEGATIVE_DIRECT,
                                 episode_wall_s=episode_wall_s)
    admission = collection_admission(direct, obstacle, replay, negative)
    if not admission["collection_ready"]:
        raise RuntimeError(f"T+1 positive pair did not pass the frozen role gate: {admission['reasons']}")
    return direct, obstacle, replay, negative, admission, targets


def _restore_one_step(layout, record):
    import torch
    from examples.baselines.constrain.pickcube_expert import restore_execution_state, snapshot_execution_state
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, _policy_observation, make_env
    from examples.baselines.constrain.placesphere_transport_detour import (
        TransportDetourSceneSet,
        _tree_hash,
    )

    raw = record["_raw"]
    env = None
    if "candidate_h5" in layout:
        from examples.baselines.constrain.placesphere_matched_terminal import _open_scene
        item = layout["candidate_scene"]
        scene = None
    else:
        scene = TransportDetourSceneSet(layout["catalog"])
        manifest = json.loads((Path(layout["catalog"]) / "manifest.json").read_text())
        item = manifest["scenes"][int(layout["index"])]
    try:
        if scene is None:
            env, _item, _ = _open_scene(layout, max_steps=EXPERT_MAX_STEPS)
        else:
            env = make_env(config=PlaceSpherePilotConfig(
                max_steps=EXPERT_MAX_STEPS, obstacle_radius=float(item["cylinder"]["radius"]),
                obstacle_height=float(item["cylinder"]["height"])))
            env, restored = scene.restore(int(layout["index"]), env)
            if not restored.get("bit_exact"):
                raise RuntimeError("catalog restore was not bit-exact")
        restore_execution_state(env, (raw["initial_state"], raw["initial_controller"]))
        actual_initial = snapshot_execution_state(env)
        initial_exact = (_tree_hash(actual_initial[0]) == _tree_hash(raw["initial_state"])
                         and _tree_hash(actual_initial[1]) == _tree_hash(raw["initial_controller"]))
        if not initial_exact:
            return {"passed": False, "initial_state_controller_exact": False}
        action = np.asarray(raw["actions"][0], dtype=np.float32)
        _obs, _reward, terminated, truncated, info = env.step(
            torch.as_tensor(action, dtype=torch.float32, device=env.unwrapped.device))
        after_state, after_controller = snapshot_execution_state(env)
        state_exact = _tree_hash(after_state) == _tree_hash(raw["states"][1])
        controller_exact = _tree_hash(after_controller) == _tree_hash(raw["controller_states"][1])
        policy = _policy_observation(env.unwrapped, info)
        observation_exact = np.array_equal(policy, np.asarray(raw["policy_observations"][1]))
        termination_exact = (bool(terminated) == bool(raw["terminated"][1])
                             and bool(truncated) == bool(raw["truncated"][1]))
        return {
            "passed": bool(state_exact and controller_exact and observation_exact and termination_exact),
            "initial_state_controller_exact": initial_exact,
            "first_action_state_exact": state_exact,
            "first_action_controller_exact": controller_exact,
            "first_action_policy_observation_exact": observation_exact,
            "first_action_terminated_exact": bool(terminated) == bool(raw["terminated"][1]),
            "first_action_truncated_exact": bool(truncated) == bool(raw["truncated"][1]),
            "first_action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
        }
    finally:
        env.close()


def _write_demo(demo_dir, records, admission, target_measurements, identity, layout, audits):
    import h5py
    from examples.baselines.constrain.placesphere_transport_detour import _stack_rows, _write_tree

    demo_dir = Path(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=False)
    trajectory = demo_dir / "matched_tplus1.h5"
    metadata = {
        "kind": "round20_matched_tplus1_checkpoint",
        "phase_b_source_identity": identity,
        "layout": layout,
        "margin_m": 0.015,
        "release_hold_steps": 0,
        "shared_release_target_measurements": target_measurements,
        "admission": admission,
        "restore_audits": audits,
        "full_prefix_safety": "each positive role is independently accepted from execution start; replay record is suffix-only",
        "native_thresholds_changed": False,
        "cylinder_geometry_changed": False,
        "gripper_semantics_changed": False,
        "training_use": False,
    }
    with h5py.File(trajectory, "w") as handle:
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        for role_name in ("positive_direct", "positive_obstacle"):
            raw = records[role_name]["_raw"]
            group = handle.create_group(role_name)
            _write_tree(group.create_group("initial_state"), raw["initial_state"])
            _write_tree(group.create_group("initial_controller"), raw["initial_controller"])
            _write_tree(group.create_group("env_states"), _stack_rows(raw["states"]))
            _write_tree(group.create_group("controller_states"), _stack_rows(raw["controller_states"]))
            group.create_dataset("actions", data=np.asarray(raw["actions"], dtype=np.float32))
            group.create_dataset("policy_observations", data=np.asarray(raw["policy_observations"], dtype=np.float32))
            group.create_dataset("raw_observations", data=np.asarray(raw["raw_observations"], dtype=np.float32))
            group.create_dataset("terminated", data=np.asarray(raw["terminated"], dtype=np.uint8))
            group.create_dataset("truncated", data=np.asarray(raw["truncated"], dtype=np.uint8))
            group.attrs["initial_state_sha256"] = __import__(
                "examples.baselines.constrain.placesphere_transport_detour",
                fromlist=["_tree_hash"])._tree_hash(raw["initial_state"])
            group.attrs["initial_controller_sha256"] = __import__(
                "examples.baselines.constrain.placesphere_transport_detour",
                fromlist=["_tree_hash"])._tree_hash(raw["initial_controller"])
            group.attrs["actions_sha256"] = hashlib.sha256(
                np.ascontiguousarray(raw["actions"], dtype=np.float32).tobytes()).hexdigest()
            group.attrs["native_success"] = int(bool(raw["terminal"]["native_success"]))
            group.attrs["safe_from_execution_start"] = int(bool(records[role_name]["from_execution_start_safe"]))
    file_sha = hashlib.sha256(trajectory.read_bytes()).hexdigest()
    _write_json(demo_dir / "manifest.json", {
        **metadata,
        "trajectory_file": trajectory.name,
        "trajectory_sha256": file_sha,
        "direct_actions": len(records["positive_direct"]["_raw"]["actions"]),
        "obstacle_actions": len(records["positive_obstacle"]["_raw"]["actions"]),
        "restore_audits_passed": all(audit["passed"] for audit in audits.values()),
    })
    return trajectory, file_sha


def run(output_dir, demo_dir, *, episode_wall_s=160):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    registration = json.loads(Path("outputs/exp/20261026_110408_placesphere_matched_terminal_r20b/preregistration.json").read_text())
    confirmed = json.loads(Path("outputs/exp/20261026_110408_placesphere_matched_terminal_r20b/confirmation.json").read_text())
    ready = json.loads(Path("outputs/exp/20261026_110408_placesphere_matched_terminal_r20b/dev30_validation.json").read_text())
    if not (confirmed.get("collection_ready") and confirmed.get("decision", {}).get("ready_count") == 16
            and ready.get("complete") and ready.get("ready_count") == 30
            and not ready.get("used_for_selection") and not ready.get("used_for_training")):
        raise RuntimeError("calibration/confirmation/dev readiness gates are not all satisfied")
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _write_json(output / "preregistration.json", {
        "kind": "matched_roles_tplus1_state_controller_restore_proof",
        "phase_b_identity": registration["identity"]["files_sha256"],
        "tplus1_script_sha256": script_sha,
        "calibration_ready": True, "confirmation_ready_16_of_16": True,
        "dev30_premises_30_of_30": True,
        "used_for_selection": False, "used_for_training": False,
        "episode_wall_s": episode_wall_s,
        "expert_action_budget_steps": EXPERT_MAX_STEPS,
        "main_policy_evaluation_budget_steps": 200,
        "pair_count": 1, "roles": [ROLE_POSITIVE_DIRECT, ROLE_POSITIVE_OBSTACLE],
        "negative_transport_block_required": True,
        "exact_restore_assertions": ["initial state", "initial controller", "T+1 state", "T+1 controller",
                                     "T+1 policy observation", "terminated", "truncated"],
        "demo_path": str(demo_dir), "argv": __import__("sys").argv,
        "no_full_collection_or_training": True,
    })
    layout = confirmed["rows"][0]["layout"]
    identity = registration["identity"]["files_sha256"]
    started = time.monotonic()
    direct, obstacle, replay, negative, admission, targets = _run_role_pair(layout, episode_wall_s=episode_wall_s)
    direct_audit = _restore_one_step(layout, direct)
    obstacle_audit = _restore_one_step(layout, obstacle)
    audits = {"positive_direct": direct_audit, "positive_obstacle": obstacle_audit}
    if not all(item["passed"] for item in audits.values()):
        _write_json(output / "restore_audit_failed.json", audits)
        raise RuntimeError("T+1 exact state/controller restoration proof failed")
    records = {"positive_direct": direct, "positive_obstacle": obstacle}
    trajectory, digest = _write_demo(
        demo_dir, records, admission, targets.get("release_target_measurements"), identity,
        layout, audits)
    result = {
        "complete": True, "phase_b_identity": identity,
        "roles": {
            "positive_direct": _public_record(direct),
            "positive_obstacle": _public_record(obstacle),
            "terminal_replay": replay,
            "negative_direct": _public_record(negative),
        },
        "admission": admission,
        "restore_audits": audits,
        "demo_path": str(trajectory), "demo_sha256": digest,
        "demo_use": "T+1 checkpoint only; no full 1000+300 collection or training",
        "elapsed_s": time.monotonic() - started,
    }
    _write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--demo-dir", required=True)
    parser.add_argument("--episode-wall-s", type=int, default=160)
    args = parser.parse_args()
    result = run(args.output_dir, args.demo_dir, episode_wall_s=args.episode_wall_s)
    print(json.dumps({"complete": result["complete"], "demo": result["demo_path"],
                      "sha256": result["demo_sha256"], "output": args.output_dir}, sort_keys=True))


if __name__ == "__main__":
    main()
