"""Bounded real-scene PickCube ConstrainNet interface rollout (not evaluation)."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import FORMAL_MANIFEST, sha256_file
from examples.baselines.constrain.pickcube_constrain import PickCubeConstrainAdapter
from examples.baselines.constrain.rollout_pickcube_base_dp import _json_state, _state_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 64:
        raise ValueError("interface rollout is bounded to 1..64 control steps")
    formal = json.loads(FORMAL_MANIFEST.read_text())
    scene = next(e for e in formal["episodes"] if int(e["episode_id"]) == 1000)
    options = {"cube_xy": scene["layout"]["start_xy"],
               "goal_pos": [*scene["layout"]["goal_xy"], 0.12],
               "obstacle_enabled": True,
               "obstacle_center": scene["trace"]["obstacle_center"]}
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env, snapshot_execution_state
    adapter = PickCubeConstrainAdapter(args.checkpoint, args.device, "ema", True, args.base_checkpoint)
    torch.manual_seed(int(scene["seed"]))
    env = make_pickcube_env()
    try:
        obs, _ = env.reset(seed=int(scene["seed"]), options=options)
        initial_state, initial_controller = snapshot_execution_state(env)
        initial_obs = torch.as_tensor(obs, dtype=torch.float32).clone()
        queued = initial_obs[:, None, :].repeat(1, 2, 1)
        actions, safety = [], {"native_success": False, "obstacle_collision": False,
                               "obstacle_crossing": False}
        for _ in range(args.steps):
            condition = adapter.base_cond(queued)
            sequence = adapter.get_action(condition)
            native = adapter.map_action(sequence[:, 0])
            actions.append(native[0].detach().cpu().tolist())
            obs, _, terminated, truncated, info = env.step(native.detach().cpu().numpy())
            queued = torch.cat((queued[:, 1:], torch.as_tensor(obs, dtype=torch.float32)[:, None]), dim=1)
            for key in safety:
                if key in info:
                    safety[key] = safety[key] or bool(np.asarray(info[key]).reshape(-1)[0])
            if safety["native_success"] or safety["obstacle_collision"] or safety["obstacle_crossing"]:
                break
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break
        restore = make_pickcube_env()
        try:
            restored_obs, _ = restore.reset(seed=int(scene["seed"]), options=options)
            restore_state, restore_controller = snapshot_execution_state(restore)
            restoration = {
                "observation_exact": bool(torch.equal(initial_obs, torch.as_tensor(restored_obs, dtype=torch.float32))),
                "full_state_exact": _state_hash(initial_state) == _state_hash(restore_state),
                "controller_state_exact": _state_hash(initial_controller) == _state_hash(restore_controller),
            }
        finally:
            restore.close()
    finally:
        env.close()
    array = np.asarray(actions)
    result = {
        "purpose": "bounded real obstacle-scene interface rollout; not formal evaluation",
        "scene_episode_id": 1000, "scene_seed": int(scene["seed"]),
        "scene_manifest": str(FORMAL_MANIFEST), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint), "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256_file(args.base_checkpoint),
        "requested_steps": args.steps, "executed_steps": len(actions),
        "initial_observation_sha256": __import__("hashlib").sha256(initial_obs.numpy().tobytes()).hexdigest(),
        "initial_full_state_sha256": _state_hash(initial_state),
        "initial_controller_state_sha256": _state_hash(initial_controller),
        "fresh_scene_restoration": restoration,
        "action_shape": list(array.shape), "actions_finite": bool(np.isfinite(array).all()),
        "action_min": array.min(axis=0).tolist(), "action_max": array.max(axis=0).tolist(),
        "actions_native_pd_ee_delta_pose": actions,
        "native_success": safety["native_success"],
        "obstacle_safety": {"collision": safety["obstacle_collision"], "crossing": safety["obstacle_crossing"]},
        "full_state": _json_state(initial_state), "controller_state": _json_state(initial_controller),
        "outcome": "native_success" if safety["native_success"] else "safety_failure" if safety["obstacle_collision"] or safety["obstacle_crossing"] else "terminated_or_truncated" if len(actions) < args.steps else "step_cap",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "short_rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
