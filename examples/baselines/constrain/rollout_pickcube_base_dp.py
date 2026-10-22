"""Independent load, contract check, and 32-step real obstacle-scene rollout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import (
    FORMAL_MANIFEST, PickCubeBaseDPAdapter, sha256_file,
)


def _state_hash(value):
    h = hashlib.sha256()

    def visit(node, key=""):
        if isinstance(node, dict):
            for name in sorted(node):
                visit(node[name], f"{key}/{name}")
        elif torch.is_tensor(node):
            h.update(key.encode()); h.update(node.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(key.encode()); h.update(np.asarray(node).tobytes())
    visit(value)
    return h.hexdigest()


def _json_state(value):
    if isinstance(value, dict):
        return {str(k): _json_state(v) for k, v in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return value.item()
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    manifest = json.loads(FORMAL_MANIFEST.read_text())
    scene = next(e for e in manifest["episodes"] if e["kind"] == "obstacle")
    scene_fields = {
        "episode_id": scene["episode_id"], "seed": scene["seed"],
        "cube_xy": scene["layout"]["start_xy"], "goal_xy": scene["layout"]["goal_xy"],
        "obstacle_center": scene["trace"]["obstacle_center"],
    }
    if args.steps <= 0 or args.steps > 64:
        raise ValueError("interface rollout is bounded to 1..64 control steps")
    adapter = PickCubeBaseDPAdapter(args.checkpoint, device=args.device, weights="ema")
    torch.manual_seed(int(scene["seed"]))

    from examples.baselines.constrain.pickcube_expert import make_pickcube_env, snapshot_execution_state
    options = {"cube_xy": scene_fields["cube_xy"],
               "goal_pos": [*scene_fields["goal_xy"], 0.12],
               "obstacle_enabled": True, "obstacle_center": scene_fields["obstacle_center"]}
    env = make_pickcube_env()
    try:
        obs, _ = env.reset(seed=int(scene["seed"]), options=options)
        initial_state, initial_controller = snapshot_execution_state(env)
        initial_obs = torch.as_tensor(obs, dtype=torch.float32).clone()
        queued = initial_obs[:, None, :].repeat(1, 2, 1)
        changed = queued.clone(); changed[..., 42:] += 17
        if not torch.equal(adapter.base_cond(queued), adapter.base_cond(changed)):
            raise AssertionError("obstacle features leaked into base DP condition")
        actions_log = []
        native_safety = {"success": False, "obstacle_collision": False, "obstacle_crossing": False}
        for _ in range(args.steps):
            policy_obs = queued[..., :42].to(args.device)
            sequence = adapter.get_action(policy_obs)
            native = adapter.map_action_ee_delta_pose(sequence[:, 0])
            if native.shape != (1, 7) or not torch.isfinite(native).all():
                raise AssertionError("policy emitted invalid native action")
            if float(native.abs().max()) > 1.000001:
                raise AssertionError("policy action outside [-1,1]")
            actions_log.append(native[0].detach().cpu().tolist())
            obs, _, terminated, truncated, info = env.step(native.detach().cpu().numpy())
            queued = torch.cat((queued[:, 1:], torch.as_tensor(obs, dtype=torch.float32)[:, None]), dim=1)
            for key in native_safety:
                if key in info:
                    native_safety[key] = native_safety[key] or bool(np.asarray(info[key]).reshape(-1)[0])
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break
        controller_hash = _state_hash(initial_controller)
        state_hash = _state_hash(initial_state)
        # A fresh scene reset with identical fixed scene parameters must restore
        # the exact interface-test initial observation before any policy action.
        restore_env = make_pickcube_env()
        try:
            restored_obs, _ = restore_env.reset(seed=int(scene["seed"]), options=options)
            restored = torch.as_tensor(restored_obs, dtype=torch.float32)
            restore_match = bool(torch.equal(initial_obs, restored))
        finally:
            restore_env.close()
        result = {
            "purpose": "short obstacle-scene interface rollout; not a 100-scene evaluation",
            "scene_source": str(FORMAL_MANIFEST), "scene": scene_fields,
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
            "weights": "final EMA", "policy_contract": adapter.metadata["contract"],
            "requested_steps": args.steps, "executed_steps": len(actions_log),
            "initial_observation_sha256": hashlib.sha256(initial_obs.numpy().tobytes()).hexdigest(),
            "initial_full_state_sha256": state_hash,
            "initial_controller_state_sha256": controller_hash,
            "initial_observation": initial_obs.tolist(),
            "initial_full_state": _json_state(initial_state),
            "initial_controller_state": _json_state(initial_controller),
            "fresh_scene_restore_observation_match": restore_match,
            "obstacle_condition_invariance": True,
            "action_shape": [len(actions_log), 7],
            "action_min": np.min(actions_log, axis=0).tolist(),
            "action_max": np.max(actions_log, axis=0).tolist(),
            "actions_native_pd_ee_delta_pose": actions_log,
            "native_safety_flags": native_safety,
            "rollout_outcome": "continued" if len(actions_log) == args.steps else "terminated_or_truncated",
        }
    finally:
        env.close()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "short_rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
