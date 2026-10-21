"""Restore and verify every stored scene without invoking the expert."""

import argparse
import json
from pathlib import Path

import numpy as np

import examples.baselines.constrain  # noqa: F401
from examples.baselines.constrain.scene_set import SceneSet, _state_leaves, current_state_snapshot, restore_scene
from examples.baselines.constrain.stackcube_obstacle import ENV_ID


def validate(directory):
    import gymnasium as gym

    directory = Path(directory)
    scenes = SceneSet(directory)
    if len(scenes) != 100:
        raise ValueError(f"expected 100 scenes, found {len(scenes)}")
    results = []
    for index in range(len(scenes)):
        scene = scenes.load(index)
        meta = scene.metadata
        cylinder = meta["cylinder"]
        env = gym.make(
            ENV_ID,
            num_envs=1,
            obs_mode="state",
            sim_backend="physx_cpu",
            robot_uids="panda_wristcam",
            robot_init_qpos_noise=0.02,
            control_mode="pd_ee_delta_pose",
            obstacle_radius=float(cylinder["radius"]),
            obstacle_height=float(cylinder["height"]),
            obstacle_center=cylinder["center_xy"],
            max_episode_steps=500,
        )
        try:
            restored_observation = restore_scene(env, scene)
            state = current_state_snapshot(env)
            expected_leaves = dict(_state_leaves(scene.initial_state))
            restored_leaves = dict(_state_leaves(state))
            if expected_leaves.keys() != restored_leaves.keys():
                raise ValueError(f"state leaf mismatch for {scene.scene_id}")
            errors = [
                float(np.max(np.abs(expected_leaves[key].astype(np.float64)
                                    - restored_leaves[key].astype(np.float64))))
                for key in expected_leaves
            ]
            max_state_error = max(errors, default=0.0)
            if max_state_error > 2e-7:
                raise ValueError(
                    f"state round-trip error {max_state_error} for {scene.scene_id}"
                )
            if not np.isclose(
                float(env.unwrapped.obstacle_collision_radius),
                float(cylinder["radius"]), atol=1e-6, rtol=0,
            ):
                raise ValueError(f"obstacle collision geometry mismatch for {scene.scene_id}")
            observation = np.asarray(
                restored_observation.detach().cpu() if hasattr(restored_observation, "detach")
                else restored_observation
            ).reshape(-1)
            observation_error = float(np.max(np.abs(
                observation.astype(np.float64) - scene.initial_observation.astype(np.float64)
            )))
            if observation_error > 1e-6:
                raise ValueError(f"observation round-trip mismatch for {scene.scene_id}")
            expert = meta["expert"]
            if not expert["stack_success"] or expert["obstacle_collision"] or expert["obstacle_crossing"]:
                raise ValueError(f"unsafe witness metadata for {scene.scene_id}")
            results.append({
                "scene_id": scene.scene_id,
                "status": "PASS",
                "stored_state_sha256": meta["initial_state_sha256"],
                "max_abs_state_error": max_state_error,
                "max_abs_observation_error": observation_error,
                "obstacle_collision_radius": float(env.unwrapped.obstacle_collision_radius),
            })
        finally:
            env.close()
    evidence = {
        "status": "PASS",
        "scene_count": len(results),
        "expert_execution_during_validation": False,
        "state_roundtrips_within_tolerance": len(results),
        "results": results,
    }
    (directory / "roundtrip_validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", default="outputs/wristcam_eval_scenes_20261021")
    args = parser.parse_args()
    evidence = validate(args.scene_dir)
    print(json.dumps({key: value for key, value in evidence.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
