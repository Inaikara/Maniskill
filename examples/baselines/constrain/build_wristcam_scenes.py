"""Collect and package a bounded set of reusable wristcam evaluation scenes."""

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from examples.baselines.constrain.collect_constructive import Config, collect
from examples.baselines.constrain.scene_set import export_scene_set


FROZEN_CONFIG_SHA256 = "a5390e1777aa3ea0f708bf8b414d38ac6727ed520c53e3ce53ef9c05da19d140"
FROZEN_SOURCE_SHA256 = {
    "collect_constructive.py": "b989607c6f0c434ceceba3951ec3d38cee6ec95b29473a1483ab99bc696c2142",
    "stackcube_obstacle.py": "f5c62090f968fa65fc616171dcab0109452e02d1b4d9dfd47623b3d19fde8d85",
    "obstacle_record.py": "918e2478da96bda0afdf81f2bd700a13c4e03249c79b8f20c5e626f90f86d3cb",
    "constrain_dataset.py": "4ffdbaad2d522857129d4d0d68bc2698b793937eba8353be017999a64d73a561",
}
ROOT = Path(__file__).resolve().parents[3]


def build(output_dir, seed=10000):
    output_dir = Path(output_dir)
    witness_dir = output_dir / "witness"
    config = Config(
        target_successes=100,
        reuse_count=1,
        seed=seed,
        max_templates=2000,
        max_episodes=400,
        max_cylinder_attempts=40,
        max_episode_steps=500,
    )
    if replace(config, seed=0).fingerprint() != FROZEN_CONFIG_SHA256:
        raise RuntimeError("frozen Ticket 02 configuration fingerprint changed")
    actual_sources = {}
    for filename, expected in FROZEN_SOURCE_SHA256.items():
        path = Path(__file__).with_name(filename)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen source identity mismatch: {path}")
        actual_sources[str(path.relative_to(ROOT))] = actual
    started = time.monotonic()
    report = collect(witness_dir, config)
    if not report["complete"] or report["accepted"] != 100:
        raise RuntimeError(
            "bounded collection did not produce 100 scenes; see witness/report.json"
        )
    identity = {
        "generator": "examples.baselines.constrain.collect_constructive:collect",
        "frozen_config_sha256": config.fingerprint(),
        "frozen_formal_config_sha256": FROZEN_CONFIG_SHA256,
        "frozen_source_sha256": actual_sources,
        "run_seed": seed,
        "collection_elapsed_seconds_before_scene_export": time.monotonic() - started,
        "environment_contract": {
            "env_id": "StackCubeObstacle-v1",
            "robot_uids": "panda_wristcam",
            "native_reset_noise": 0.02,
            "obs_mode": "state",
            "control_mode": "pd_ee_delta_pose",
            "sim_backend": "physx_cpu",
            "num_envs": 1,
            "max_episode_steps": 500,
            "rotation_action": [0.0, 0.0, 0.0],
        },
        "budgets": {
            "target_successes": 100, "reuse_count": 1,
            "max_templates": 2000, "max_episodes": 400,
            "max_cylinder_attempts": 40, "max_steps_per_episode": 500,
            "wall_seconds": 3600,
        },
    }
    manifest = export_scene_set(witness_dir, output_dir, identity)
    if len(manifest["scenes"]) != 100:
        raise RuntimeError("exported scene count is not 100")
    (output_dir / "build_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/wristcam_eval_scenes_20261021")
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    result = build(args.output_dir, args.seed)
    print(json.dumps({"scene_count": result["scene_count"], "output": args.output_dir}))


if __name__ == "__main__":
    main()
