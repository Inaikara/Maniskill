"""Paired policy evaluation on the immutable expert-verified scene store."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.scene_set import (
    SceneSet, _stable_hash, current_state_snapshot, restore_scene,
)

VIDEO_INDEXES = (0, 17, 42, 77)


def classify_progress(row):
    if row["stack_success"]:
        return "stacked_safe" if row["safe_success"] else "stacked_unsafe"
    if not row["ever_grasped"]:
        return "no_grasp"
    if not row["ever_lifted"]:
        return "grasp_no_lift"
    return "lift_transport_or_place_failed"


def summarize_fixed_results(rows, requested):
    evaluated = [r for r in rows if r.get("status", "evaluated") == "evaluated"]
    bins = {}
    for row in evaluated:
        name = classify_progress(row)
        bins[name] = bins.get(name, 0) + 1
    count = len(evaluated)
    return {
        "requested": requested, "denominator": requested, "evaluated": count,
        "runtime_errors": requested - count,
        "complete": count == requested and len(rows) == requested,
        "stack_success_count": sum(bool(r["stack_success"]) for r in evaluated),
        "stack_success_rate": sum(bool(r["stack_success"]) for r in evaluated) / requested,
        "safe_success_count": sum(bool(r["safe_success"]) for r in evaluated),
        "safe_success_rate": sum(bool(r["safe_success"]) for r in evaluated) / requested,
        "contact_count": sum(bool(r["obstacle_collision"]) for r in evaluated),
        "contact_rate": sum(bool(r["obstacle_collision"]) for r in evaluated) / requested,
        "crossing_count": sum(bool(r["obstacle_crossing"]) for r in evaluated),
        "crossing_rate": sum(bool(r["obstacle_crossing"]) for r in evaluated) / requested,
        "timeout_count": sum(r["termination_reason"] == "timeout" for r in evaluated),
        "progress_bins": bins,
    }


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _flag(info, name):
    return bool(np.asarray(_array(info[name])).reshape(-1)[0])


def update_step_flags(previous, info):
    """Accumulate sticky per-episode safety flags through this control step."""
    return {
        name: bool(previous[name] or _flag(info, info_name))
        for name, info_name in (("stack_success", "stack_success"),
                                ("obstacle_collision", "obstacle_collision"),
                                ("obstacle_crossing", "obstacle_crossing"))
    }


def episode_should_stop(flags, terminated, truncated, elapsed, max_steps):
    return (flags["stack_success"] or terminated or truncated or elapsed >= max_steps)


def _checkpoint_identity(adapter):
    checkpoint = adapter.metadata.get("checkpoint", adapter.metadata.get("weights_source"))
    if not checkpoint:
        return None
    path = Path(checkpoint)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class FixedSceneEvaluator:
    def __init__(self, adapter, scene_set, policy_name, output_dir, device="cpu",
                 max_steps=500, video_indexes=VIDEO_INDEXES):
        self.adapter, self.scene_set = adapter, scene_set
        self.policy_name, self.output_dir = policy_name, Path(output_dir)
        self.device, self.max_steps = device, max_steps
        self.video_indexes = set(video_indexes)
        self.env = None
        self._env_radius = None

    def _make_env(self, scene):
        import gymnasium as gym
        import examples.baselines.constrain.stackcube_obstacle  # noqa: F401
        from mani_skill.utils.wrappers import CPUGymWrapper

        cylinder = scene.metadata["cylinder"]
        env = gym.make("StackCubeObstacle-v1", reconfiguration_freq=0,
                       robot_uids="panda_wristcam", control_mode="pd_ee_delta_pose",
                       reward_mode="sparse", obs_mode="state", render_mode="rgb_array",
                       sim_backend="physx_cpu", num_envs=1,
                       robot_init_qpos_noise=0.02,
                       max_episode_steps=self.max_steps,
                       obstacle_radius=float(cylinder["radius"]),
                       obstacle_height=float(cylinder["height"]),
                       obstacle_center=cylinder["center_xy"],
                       human_render_camera_configs=dict(shader_pack="default"))
        return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)

    def run_one(self, index, scene, writer=None):
        radius = float(scene.metadata["cylinder"]["radius"])
        if self.env is not None and radius != self._env_radius:
            self.env.close()
            self.env = None
        if self.env is None:
            self.env = self._make_env(scene)
            self._env_radius = radius
        # Native reset first clears controller and sticky safety state; the saved
        # complete state is then restored without generation or expert execution.
        restore_scene(self.env, scene)
        state_before = current_state_snapshot(self.env)
        state_hash_before = _stable_hash(state_before)
        obs = np.asarray(_array(self.env.get_obs() if hasattr(self.env, "get_obs")
                                else self.env.unwrapped.get_obs()), dtype=np.float32).reshape(-1)
        expected_obs = np.asarray(scene.initial_observation).reshape(-1)
        obs_error = float(np.max(np.abs(obs - expected_obs)))
        history = np.repeat(obs[None, :], 2, axis=0)
        torch.manual_seed(int(scene.metadata["seed"]))
        if writer:
            writer.append_data(self._frame())
        ever_grasped = ever_lifted = ever_transported = False
        initial_cube_z = float(self.env.unwrapped.cubeA.pose.p[0, 2])
        flags = {"stack_success": False, "obstacle_collision": False,
                 "obstacle_crossing": False}
        elapsed = 0
        first_action = None
        termination = "timeout"
        last_obs = history
        while elapsed < self.max_steps:
            cond = self.adapter.base_cond(torch.as_tensor(last_obs[None], dtype=torch.float32, device=self.device))
            predicted = self.adapter.get_action(cond)
            actions = self.adapter.map_action_ee_delta_pose(predicted)
            for action in actions[0][:min(actions.shape[1], self.max_steps - elapsed)]:
                obs, _, terminated, truncated, info = self.env.step(_array(action)[None])
                elapsed += 1
                last_obs = np.concatenate([last_obs[1:], np.asarray(obs).reshape(1, -1)])
                flags = update_step_flags(flags, info)
                grasp = _flag(info, "is_cubeA_grasped")
                cube_pos = _array(self.env.unwrapped.cubeA.pose.p)[0]
                ever_grasped |= grasp
                ever_lifted |= float(cube_pos[2]) > initial_cube_z + 0.04
                target = _array(self.env.unwrapped.cubeB.pose.p)[0]
                ever_transported |= float(np.linalg.norm(cube_pos[:2] - target[:2])) < 0.10
                if first_action is None:
                    first_action = _array(action).astype(float).tolist()
                if writer and not (terminated or truncated):
                    writer.append_data(self._frame())
                if episode_should_stop(flags, bool(terminated), bool(truncated),
                                       elapsed, self.max_steps):
                    if flags["stack_success"]:
                        termination = "stack_success"
                    elif terminated or truncated:
                        termination = "environment_terminal"
                    break
            if episode_should_stop(flags, terminated, truncated, elapsed, self.max_steps):
                break
        stack = flags["stack_success"]
        collision = flags["obstacle_collision"]
        crossing = flags["obstacle_crossing"]
        safe = stack and not collision and not crossing
        record = {
            "scene_index": index, "scene_id": scene.scene_id, "seed": int(scene.metadata["seed"]),
            "status": "evaluated", "initial_state_sha256": scene.metadata["initial_state_sha256"],
            "restored_state_sha256": state_hash_before,
            "restored_observation_max_abs_error": obs_error,
            "result": {"elapsed_steps": elapsed, "termination_reason": termination,
                       "status": "evaluated",
                       "stack_success": stack, "safe_success": safe,
                       "obstacle_collision": collision, "obstacle_crossing": crossing,
                       "ever_grasped": ever_grasped, "ever_lifted": ever_lifted,
                       "ever_transported": ever_transported,
                       "progress_bin": classify_progress(dict(stack_success=stack,
                           safe_success=safe, ever_grasped=ever_grasped,
                           ever_lifted=ever_lifted, ever_transported=ever_transported))},
            "first_action": first_action,
        }
        return record

    def _frame(self):
        frame = _array(self.env.render())
        if frame.ndim == 4:
            frame = frame[0]
        return frame.astype(np.uint8)

    def run_all(self, count, resume_dir=None):
        import imageio.v2 as imageio
        self.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"policy": self.policy_name, "requested": count,
                    "video_scene_indexes_fixed_before_rollouts": sorted(self.video_indexes),
                    "scene_manifest_sha256": self.scene_set.manifest["manifest_sha256"],
                    "adapter_metadata": self.adapter.metadata,
                    "checkpoint_sha256": _checkpoint_identity(self.adapter),
                    "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                "device": self.device, "sim_backend": "physx_cpu",
                                "num_envs": 1, "obs_mode": "state",
                                "control_mode": "pd_ee_delta_pose", "robot_uids": "panda_wristcam",
                                "reset_noise": 0.02, "history_frames": 2,
                                "rotation_action": [0, 0, 0], "max_steps": self.max_steps,
                                "control_chunk": 8, "noise_seed_rule": "torch.manual_seed(scene.metadata.seed) per episode"},
                    "scenes": [{"scene_index": i, "scene_id": self.scene_set.load(i).scene_id}
                               for i in range(count)]}
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        video_path = self.output_dir / "representative.mp4"
        writer = imageio.get_writer(str(video_path), fps=20, codec="libx264", macro_block_size=1)
        rows = []
        try:
            for i in range(count):
                scene = self.scene_set.load(i)
                prior = (Path(resume_dir) / f"episode_{i:03d}.json") if resume_dir else None
                reused = False
                if prior is not None and prior.exists():
                    previous = json.loads(prior.read_text())
                    prior_manifest_path = Path(resume_dir) / "manifest.json"
                    prior_manifest = (json.loads(prior_manifest_path.read_text())
                                      if prior_manifest_path.is_file() else {})
                    prior_hash = prior_manifest.get("checkpoint_sha256")
                    if prior_hash is None:
                        prior_source = prior_manifest.get("adapter_metadata", {}).get(
                            "checkpoint", prior_manifest.get("adapter_metadata", {}).get("weights_source"))
                        if prior_source and Path(prior_source).is_file():
                            prior_hash = hashlib.sha256(Path(prior_source).read_bytes()).hexdigest()
                    prior_config = {
                        "policy": prior_manifest.get("policy"),
                        "checkpoint_sha256": prior_hash,
                        "scene_manifest_sha256": prior_manifest.get("scene_manifest_sha256"),
                        "runtime": prior_manifest.get("runtime"),
                        "adapter_metadata": prior_manifest.get("adapter_metadata"),
                    }
                    expected_config = {
                        "policy": self.policy_name,
                        "checkpoint_sha256": manifest["checkpoint_sha256"],
                        "scene_manifest_sha256": self.scene_set.manifest["manifest_sha256"],
                        "runtime": manifest["runtime"],
                        "adapter_metadata": self.adapter.metadata,
                    }
                    if (previous.get("status") == "evaluated"
                            and previous.get("scene_id") == scene.scene_id
                            and previous.get("seed") == int(scene.metadata["seed"])
                            and previous.get("initial_state_sha256") == scene.metadata["initial_state_sha256"]
                            and previous.get("restored_observation_max_abs_error", float("inf")) <= 1e-6
                            and prior_config == expected_config):
                        record, reused = previous, True
                        old_video = Path(resume_dir) / "representative.mp4"
                        if i in self.video_indexes and old_video.exists():
                            with imageio.get_reader(str(old_video)) as reader:
                                for frame in reader:
                                    writer.append_data(frame)
                if not reused:
                    try:
                        record = self.run_one(i, scene, writer if i in self.video_indexes else None)
                    except Exception as exc:
                        record = {"scene_index": i, "scene_id": scene.scene_id,
                                  "status": "runtime_error", "error": f"{type(exc).__name__}: {exc}"}
                (self.output_dir / f"episode_{i:03d}.json").write_text(json.dumps(record, indent=2) + "\n")
                rows.append(record.get("result", record))
        finally:
            writer.close()
            if self.env is not None:
                self.env.close()
        summary = summarize_fixed_results(rows, count)
        summary["episodes"] = rows
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--constrain-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-scenes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--resume-from", type=Path, default=None,
                   help="reuse only completed, identity-verified episodes from a prior run")
    args = p.parse_args(argv)
    from examples.baselines.constrain.base_dp import BaseDPAdapter
    from examples.baselines.constrain.constrain_adapter import ConstrainDPAdapter

    catalog = SceneSet(args.scene_dir)
    if len(catalog) < args.num_scenes:
        raise ValueError(f"need {args.num_scenes} scenes, store has {len(catalog)}")
    if args.num_scenes != 100 or args.max_steps != 500:
        raise ValueError("formal paired evaluation requires exactly 100 scenes and a 500-step cap")
    if catalog.manifest.get("manifest_sha256") != "f1aa2175bd70a86ba522b5d441b1a9897330fdbd17a5484d7a06f5eccb8335b3":
        raise ValueError("scene-store identity does not match the frozen Ticket 03 manifest")
    adapters = {
        "base_dp": BaseDPAdapter(args.base_checkpoint, device=args.device),
        "constrain_dp": ConstrainDPAdapter(args.constrain_checkpoint, device=args.device, weights="ema"),
    }
    result = {}
    checkpoint_hashes = {}
    for name, path in (("base_dp", args.base_checkpoint), ("constrain_dp", args.constrain_checkpoint)):
        checkpoint_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    started = time.time()
    for name, adapter in adapters.items():
        result[name] = FixedSceneEvaluator(adapter, catalog, name, args.output_dir / name,
                                           args.device, args.max_steps).run_all(
                                               args.num_scenes,
                                               args.resume_from / name if args.resume_from else None)
    result["paired_state_equality"] = True
    result["paired_scene_identity"] = catalog.manifest["manifest_sha256"]
    result["checkpoint_sha256"] = checkpoint_hashes
    result["video_scene_indexes_fixed_before_results"] = list(VIDEO_INDEXES)
    result["wall_seconds"] = round(time.time() - started, 3)
    for i in range(args.num_scenes):
        left = json.loads((args.output_dir / "base_dp" / f"episode_{i:03d}.json").read_text())
        right = json.loads((args.output_dir / "constrain_dp" / f"episode_{i:03d}.json").read_text())
        if (left.get("scene_id") != right.get("scene_id")
                or left.get("restored_state_sha256") != right.get("restored_state_sha256")
                or left.get("restored_observation_max_abs_error", float("inf")) > 1e-6
                or right.get("restored_observation_max_abs_error", float("inf")) > 1e-6):
            result["paired_state_equality"] = False
    result["validation"] = {"requested_scenes": args.num_scenes,
                            "paired_initial_state_fingerprints_equal": result["paired_state_equality"],
                            "all_restored_observations_within_1e-6": result["paired_state_equality"],
                            "same_step_violation_counted_non_safe": True,
                            "first_stack_success_stops_control_chunk": True,
                            "runtime_errors_retained_in_denominator": True}
    result["complete"] = all(result[k]["complete"] for k in ("base_dp", "constrain_dp"))
    (args.output_dir / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: ({m: v[m] for m in ("requested", "evaluated", "runtime_errors", "stack_success_count",
                                               "safe_success_count", "contact_count", "crossing_count", "timeout_count", "progress_bins")}
                           if isinstance(v, dict) and "requested" in v else v)
                      for k, v in result.items()}, indent=2))
    return result


if __name__ == "__main__":
    main()
