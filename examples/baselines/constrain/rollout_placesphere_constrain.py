"""Bounded restored-scene PlaceSphere ConstrainNet interface rollout (not evaluation)."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.placesphere_base_dp import FORMAL_H5, FORMAL_MANIFEST
from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
from examples.baselines.constrain.rollout_placesphere_base_dp import _h5_tree, _jsonable, _max_delta, _tree_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 32:
        raise ValueError("interface rollout is bounded to 1..32 control steps")
    manifest = json.loads(FORMAL_MANIFEST.read_text())
    scene = next(ep for ep in manifest["episodes"] if int(ep["episode_id"]) == 1000 and ep["kind"] == "obstacle")
    scene_id, seed = 1000, int(scene["seed"])
    options = {"sphere_xy": scene["layout"]["start_xy"], "container_xy": scene["layout"]["container_xy"],
               "obstacle_enabled": True, "obstacle_center": scene["trace"]["obstacle_center"]}
    with h5py.File(FORMAL_H5, "r") as h5:
        group = h5[f"traj_{scene_id}"]
        saved_state, saved_controller = _h5_tree(group["env_states"], 0), _h5_tree(group["controller_states"], 0)
        expected_obs = group["raw_obs"][0]
        controller_hash = group["controller_states"]["snapshot_sha256"][0].decode()
    # Load and choose the fixed formal obstacle before the first policy action.
    adapter = PlaceSphereConstrainAdapter(args.checkpoint, device=args.device, weights="ema", base_checkpoint=args.base_checkpoint)
    torch.manual_seed(seed)
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
    env = make_env(config=PlaceSpherePilotConfig(seed=seed, max_steps=200, obstacle_radius=.03, obstacle_height=.08))
    try:
        env.reset(seed=seed, options=options)
        base = env.unwrapped
        base.set_state_dict(saved_state); base.agent.controller.set_state(saved_controller)
        actual_state, actual_controller = base.get_state_dict(), base.agent.controller.get_state()
        from mani_skill.utils.common import flatten_state_dict
        raw = flatten_state_dict(base._get_obs_state_dict(base.get_info()), use_torch=True).detach().cpu().numpy()[0]
        restore = {"native_state_max_abs": _max_delta(saved_state, actual_state),
                   "controller_state_max_abs": _max_delta(saved_controller, actual_controller),
                   "raw_observation_max_abs": float(np.max(np.abs(raw - expected_obs)))}
        if restore["native_state_max_abs"] > 1e-7 or restore["controller_state_max_abs"] > 1e-7 or restore["raw_observation_max_abs"] > 1e-7:
            raise AssertionError(f"initial PlaceSphere state restore differs: {restore}")
        queued = torch.as_tensor(raw, dtype=torch.float32, device=args.device)[None, None].repeat(1, 2, 1)
        before = adapter.base_cond(queued); altered = queued.clone(); altered[..., 39:] += 17
        if not torch.equal(before, adapter.base_cond(altered)):
            raise AssertionError("obstacle hint leaked into base policy observation")
        # Refresh from the actual observation; restoration and one legal action are checked before proceeding.
        named = flatten_state_dict(base._get_obs_state_dict(base.get_info()), use_torch=True)
        queued = torch.as_tensor(named, dtype=torch.float32, device=args.device)[:, None].repeat(1, 2, 1)
        actions, infos, stopped = [], [], False
        for _ in range(args.steps):
            cond = adapter.base_cond(queued)
            predicted = adapter.get_action(cond)
            action = adapter.map_action(predicted[:, 0])
            if action.shape != (1, 7) or not torch.isfinite(action).all():
                raise AssertionError("invalid native PlaceSphere action")
            actions.append(action[0].detach().cpu().tolist())
            obs, _, terminated, truncated, info = env.step(action.detach().cpu().numpy())
            infos.append({key: _jsonable(info[key]) for key in ("native_success", "success", "is_obj_static", "is_obj_grasped", "obstacle_collision", "obstacle_crossing") if key in info})
            queued = torch.cat((queued[:, 1:], torch.as_tensor(obs, dtype=torch.float32, device=args.device)[:, None]), 1)
            safety = info
            if any(bool(np.asarray(info[k]).any()) for k in ("obstacle_collision", "obstacle_crossing") if k in info):
                stopped = True; break
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                stopped = True; break
        cpu = PlaceSphereConstrainAdapter(args.checkpoint, device="cpu", weights="ema")
        cpu.base_cond(torch.as_tensor(raw, dtype=torch.float32)[None, None].repeat(1, 2, 1))
        cpu_action = cpu.map_action(cpu.get_action(cpu.base_cond(torch.as_tensor(raw, dtype=torch.float32)[None, None].repeat(1, 2, 1))))
        if cpu_action.shape != (1, 8, 7) or not torch.isfinite(cpu_action).all():
            raise AssertionError("standalone CPU checkpoint load did not yield finite legal-shape actions")
    finally:
        env.close()
    result = {"purpose": "bounded interface rollout, not formal evaluation", "scene_episode_id": scene_id,
              "scene_selected_before_inference": True, "scene_seed": seed, "layout": scene["layout"],
              "obstacle_center": scene["trace"]["obstacle_center"], "requested_steps": args.steps,
              "executed_steps": len(actions), "restoration_max_abs": restore,
              "restore_tolerance": 1e-7, "initial_state_sha256": _tree_hash(saved_state),
              "controller_snapshot_sha256": controller_hash, "action_shape": [len(actions), 7],
              "actions_native_pd_ee_delta_pose": actions, "native_infos": infos,
              "safety_or_terminal_stop": stopped, "standalone_cpu_load": {"shape": list(cpu_action.shape), "finite": bool(torch.isfinite(cpu_action).all()), "native_bounds": True},
              "outcome": "stopped_on_native_safety_or_terminal" if stopped else "continued_to_step_cap"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "short_rollout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__": main()
