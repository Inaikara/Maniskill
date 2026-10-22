"""Fresh-load verification and bounded PlaceSphere obstacle interface rollout."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.placesphere_base_dp import (
    FORMAL_H5, FORMAL_MANIFEST, PlaceSphereBaseDPAdapter, sha256_file,
)


def _tree_hash(value):
    digest = hashlib.sha256()
    def visit(node, key=""):
        if isinstance(node, dict):
            for name in sorted(node):
                visit(node[name], f"{key}/{name}")
        else:
            digest.update(key.encode()); digest.update(np.asarray(node).tobytes())
    visit(value)
    return digest.hexdigest()


def _max_delta(left, right):
    if isinstance(left, dict):
        return max((_max_delta(left[key], right[key]) for key in left), default=0.0)
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right)), initial=0.0))


def _h5_tree(node, index):
    if isinstance(node, h5py.Group):
        return {key: _h5_tree(value, index) for key, value in node.items()
                if key != "snapshot_sha256"}
    return node[index]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist() if isinstance(value, np.ndarray) else value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    if not 1 <= args.steps <= 64:
        raise ValueError("interface rollout is bounded to 1..64 control steps")
    manifest = json.loads(FORMAL_MANIFEST.read_text())
    scene = next(ep for ep in manifest["episodes"] if ep["kind"] == "obstacle")
    scene_id, seed = int(scene["episode_id"]), int(scene["seed"])
    # Scene and episode are fixed before any policy inference.
    options = {"sphere_xy": scene["layout"]["start_xy"],
               "container_xy": scene["layout"]["container_xy"],
               "obstacle_enabled": True, "obstacle_center": scene["trace"]["obstacle_center"]}
    with h5py.File(FORMAL_H5, "r") as h5:
        group = h5[f"traj_{scene_id}"]
        saved_state = _h5_tree(group["env_states"], 0)
        saved_controller = _h5_tree(group["controller_states"], 0)
        expected_policy_obs = group["obs"][0]
        controller_hash = group["controller_states"]["snapshot_sha256"][0].decode()

    adapter = PlaceSphereBaseDPAdapter(args.checkpoint, device=args.device, weights="ema")
    torch.manual_seed(seed)
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
    cfg = PlaceSpherePilotConfig(seed=seed, max_steps=200, obstacle_radius=0.03, obstacle_height=0.08)
    env = make_env(config=cfg)
    try:
        env.reset(seed=seed, options=options)
        base = env.unwrapped
        base.set_state_dict(saved_state)
        base.agent.controller.set_state(saved_controller)
        actual_state, actual_controller = base.get_state_dict(), base.agent.controller.get_state()
        named = base._get_obs_state_dict(base.get_info())
        from mani_skill.utils.common import flatten_state_dict
        restored_raw = flatten_state_dict(named, use_torch=True).detach().cpu().numpy()[0]
        restored_policy = restored_raw[:39]
        restore_delta = {"native_state_max_abs": _max_delta(saved_state, actual_state),
                         "controller_state_max_abs": _max_delta(saved_controller, actual_controller),
                         "policy_observation_max_abs": float(np.max(np.abs(restored_policy - expected_policy_obs)))}
        if restore_delta["native_state_max_abs"] > 1e-7 or restore_delta["controller_state_max_abs"] > 1e-7:
            raise AssertionError(f"saved PlaceSphere initial state/controller failed restore tolerance: {restore_delta}")
        if restore_delta["policy_observation_max_abs"] > 1e-6:
            raise AssertionError(f"saved PlaceSphere initial observation failed restore tolerance: {restore_delta}")
        queued = torch.as_tensor(restored_raw, dtype=torch.float32, device=args.device)[None, None].repeat(1, 2, 1)
        altered = queued.clone(); altered[..., 39:] += 31
        if not torch.equal(adapter.base_cond(queued), adapter.base_cond(altered)):
            raise AssertionError("obstacle condition changed the base policy input")

        action_log, info_log = [], []
        final_info = {}
        for _ in range(args.steps):
            prediction = adapter.get_action(queued[..., :39])
            action = adapter.map_action_ee_delta_pose(prediction[:, 0])
            if action.shape != (1, 7) or not torch.isfinite(action).all() or float(action.abs().max()) > 1.000001:
                raise AssertionError("policy emitted invalid native seven-channel action")
            action_log.append(action[0].detach().cpu().tolist())
            next_obs, _, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            queued = torch.cat((queued[:, 1:], torch.as_tensor(next_obs, dtype=torch.float32,
                                  device=args.device)[:, None]), dim=1)
            final_info = info
            info_log.append({key: _jsonable(info[key]) for key in
                             ("native_success", "success", "is_obj_static", "is_obj_grasped",
                              "obstacle_collision", "obstacle_crossing") if key in info})
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break
        cpu_adapter = PlaceSphereBaseDPAdapter(args.checkpoint, device="cpu", weights="ema")
        cpu_condition = torch.as_tensor(restored_policy, dtype=torch.float32)[None, None].repeat(1, 2, 1)
        cpu_action = cpu_adapter.get_action(cpu_condition)
        if cpu_action.shape != (1, 8, 7) or not torch.isfinite(cpu_action).all():
            raise AssertionError("fresh standalone CPU load did not produce finite legal-shape (1,8,7) actions")
        cpu_native = cpu_adapter.map_action_ee_delta_pose(cpu_action)
    finally:
        env.close()

    result = {
        "purpose": "bounded PlaceSphere obstacle interface rollout; not an evaluation",
        "scene_source": str(FORMAL_MANIFEST), "scene_episode_id": scene_id, "scene_seed": seed,
        "scene_selected_before_inference": True, "scene_layout": scene["layout"],
        "obstacle_center": scene["trace"]["obstacle_center"],
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
        "weights": "final EMA", "policy_contract": adapter.metadata["contract"],
        "requested_steps": args.steps, "executed_steps": len(action_log),
        "initial_full_state_sha256": _tree_hash(saved_state),
        "initial_controller_snapshot_sha256": controller_hash,
        "restored_full_state_sha256": _tree_hash(actual_state),
        "restored_controller_state_sha256": _tree_hash(actual_controller),
        "restore_max_abs_differences": restore_delta,
        "restore_tolerance": {"native_state": 1e-7, "controller_state": 1e-7, "policy_observation": 1e-6},
        "obstacle_condition_invariance": True,
        "action_shape": [len(action_log), 7], "actions_native_pd_ee_delta_pose": action_log,
        "action_min": np.min(action_log, axis=0).tolist(), "action_max": np.max(action_log, axis=0).tolist(),
        "native_infos": info_log, "native_terminal_flags": _jsonable(final_info),
        "standalone_cpu_load": {"output_shape": list(cpu_native.shape), "finite": bool(torch.isfinite(cpu_native).all()),
                                "action_min": float(cpu_native.min()), "action_max": float(cpu_native.max())},
        "rollout_outcome": "continued" if len(action_log) == args.steps else "terminated_or_truncated",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "short_rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
