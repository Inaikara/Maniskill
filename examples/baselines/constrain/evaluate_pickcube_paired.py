"""Frozen paired Base DP / ConstrainNet evaluation on the PickCube scene set."""

import argparse
import hashlib
import json
import platform
import signal
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet, _tree_hash

VIDEO_SCENES = (0, 1, 2)
POLICIES = ("base_dp", "constrain_dp")
EPISODE_WALL_SECONDS = 120


class EpisodeDeadline(TimeoutError):
    pass


def _deadline(_signum, _frame):
    raise EpisodeDeadline(f"episode exceeded {EPISODE_WALL_SECONDS}s wall limit")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _bool(info, name):
    value = info[name]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def classify_termination(native_success, terminated, truncated, elapsed_steps, max_steps):
    """Classify completion with native success taking precedence on its step."""
    if native_success:
        return "native_success"
    if truncated:
        return "timeout"
    if terminated:
        return "environment_terminal"
    if elapsed_steps >= max_steps:
        return "control_step_limit"
    return None


def _frame(env):
    frame = env.render()
    if torch.is_tensor(frame):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame).squeeze().astype(np.uint8)


def summarize_episodes(rows, requested):
    evaluated = [row["result"] for row in rows if row.get("status") == "evaluated"]
    count = len(evaluated)
    return {
        "requested_denominator": requested, "evaluated": count,
        "runtime_errors": requested - count,
        "wall_timeout_runtime_errors": sum(
            row.get("error_kind") == "wall_timeout" for row in rows
        ),
        "task_success_count": sum(r["native_success"] for r in evaluated),
        "task_success_rate": sum(r["native_success"] for r in evaluated) / requested,
        "safe_success_count": sum(r["safe_success"] for r in evaluated),
        "safe_success_rate": sum(r["safe_success"] for r in evaluated) / requested,
        "obstacle_contact_count": sum(r["obstacle_contact"] for r in evaluated),
        "obstacle_contact_rate": sum(r["obstacle_contact"] for r in evaluated) / requested,
        "xy_crossing_count": sum(r["xy_forbidden_crossing"] for r in evaluated),
        "xy_crossing_rate": sum(r["xy_forbidden_crossing"] for r in evaluated) / requested,
        "timeout_count": sum(r["termination_reason"] == "timeout" for r in evaluated),
        "control_step_limit_count": sum(
            r["termination_reason"] == "control_step_limit" for r in evaluated
        ),
        "environment_terminal_count": sum(
            r["termination_reason"] == "environment_terminal" for r in evaluated
        ),
        "control_steps": sum(r["elapsed_steps"] for r in evaluated),
        "phase_counts": {phase: sum(r["phase"] == phase for r in evaluated)
                         for phase in ("no_grasp", "grasp_no_lift", "lift_no_transport",
                                       "transport_or_place_failed", "task_success")},
        "complete": len(rows) == requested and count == requested,
    }


def _native_env(scene, render=False):
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env

    geometry = scene["metadata"]["cylinder"]
    return make_pickcube_env(
        render_mode="rgb_array" if render else None,
        obstacle_radius=geometry["radius"], obstacle_height=geometry["height"],
    )


class PickCubePairedEvaluator:
    def __init__(self, adapter, scene_set, policy, output, device, max_steps=500):
        self.adapter, self.scene_set, self.policy = adapter, scene_set, policy
        self.output, self.device, self.max_steps = Path(output), device, max_steps

    def run_episode(self, index, scene, writer=None):
        env = _native_env(scene, render=writer is not None)
        try:
            # Catalog restore resets native/task/controller state then restores
            # the complete frozen simulation and controller snapshots.
            _, obs = self.scene_set.restore(index, env)
            obs = np.asarray(obs.detach().cpu() if torch.is_tensor(obs) else obs,
                             dtype=np.float32).reshape(-1)
            initial_obs_hash = hashlib.sha256(obs.tobytes()).hexdigest()
            restored_state = env.unwrapped.get_state_dict()
            restored_controller = env.unwrapped.agent.controller.get_state()
            initial_state_hash = _tree_hash(restored_state)
            controller_hash = _tree_hash(restored_controller)
            observation_matches = bool(np.array_equal(obs, scene["observation"].reshape(-1)))
            if initial_state_hash != scene["metadata"]["state_sha256"]:
                raise ValueError("restored native state fingerprint differs from frozen scene")
            if controller_hash != scene["metadata"]["controller_sha256"]:
                raise ValueError("restored controller fingerprint differs from frozen scene")
            if initial_obs_hash != scene["metadata"]["observation_sha256"] or not observation_matches:
                raise ValueError("restored initial observation differs from frozen scene")
            history = np.repeat(obs[None, :], 2, axis=0)
            torch.manual_seed(int(scene["metadata"]["seed"]))
            if writer is not None:
                writer.append_data(_frame(env))
            elapsed = 0
            collision = crossing = grasped = lifted = transported = False
            initial_cube_z = float(env.unwrapped.cube.pose.p[0, 2].detach().cpu())
            goal_xy = env.unwrapped.goal_site.pose.p[0, :2].detach().cpu().numpy()
            first_success_step = None
            first_action = None
            actions = []
            termination = None
            native_success = False
            terminated = truncated = False
            while elapsed < self.max_steps:
                queued = torch.as_tensor(history[None], dtype=torch.float32, device=self.device)
                condition = self.adapter.base_cond(queued)
                if hasattr(self.adapter, "agent"):
                    condition = condition.reshape(-1, 2, 42)
                prediction = self.adapter.get_action(condition)
                if hasattr(self.adapter, "map_action_ee_delta_pose"):
                    native_sequence = self.adapter.map_action_ee_delta_pose(prediction)
                else:
                    native_sequence = self.adapter.map_action(prediction)
                for action in native_sequence[0][:min(8, self.max_steps - elapsed)]:
                    action_cpu = action.detach().cpu().numpy()
                    obs, _, step_terminated, step_truncated, info = env.step(action_cpu[None])
                    elapsed += 1
                    terminated = bool(terminated or step_terminated)
                    truncated = bool(truncated or step_truncated)
                    actions.append(action_cpu.tolist())
                    if first_action is None:
                        first_action = action_cpu.tolist()
                    obs = np.asarray(obs.detach().cpu() if torch.is_tensor(obs) else obs,
                                     dtype=np.float32).reshape(-1)
                    history = np.concatenate((history[1:], obs[None]), axis=0)
                    step_success = _bool(info, "native_success")
                    native_success |= step_success
                    collision |= _bool(info, "obstacle_collision")
                    crossing |= _bool(info, "obstacle_crossing")
                    grasped |= _bool(info, "is_grasped")
                    cube_position = env.unwrapped.cube.pose.p[0].detach().cpu().numpy()
                    lifted |= bool(cube_position[2] > initial_cube_z + 0.04)
                    transported |= bool(np.linalg.norm(cube_position[:2] - goal_xy) < 0.10)
                    reason = classify_termination(
                        step_success, step_terminated, step_truncated, elapsed, self.max_steps
                    )
                    if step_success and first_success_step is None:
                        first_success_step = elapsed
                    if reason is not None:
                        termination = reason
                    if writer is not None and not (terminated or truncated):
                        writer.append_data(_frame(env))
                    if reason is not None:
                        break
                if termination is not None:
                    break
            if termination is None:
                termination = classify_termination(
                    native_success, terminated, truncated, elapsed, self.max_steps
                )
            safe_success = native_success and not collision and not crossing
            record = {
                "scene_index": index, "scene_id": scene["metadata"]["scene_id"],
                "seed": int(scene["metadata"]["seed"]), "status": "evaluated",
                "scene_state_sha256": scene["metadata"]["state_sha256"],
                "restored_state_sha256": initial_state_hash,
                "restored_controller_sha256": controller_hash,
                "restored_observation_sha256": initial_obs_hash,
                "restored_observation_matches_catalog": observation_matches,
                "native_success_step": first_success_step,
                "actions_native_pd_ee_delta_pose": actions,
                "result": {
                    "termination_reason": termination, "elapsed_steps": elapsed,
                    "terminated": terminated, "truncated": truncated,
                    "native_success": native_success, "safe_success": safe_success,
                    "obstacle_contact": collision, "xy_forbidden_crossing": crossing,
                    "ever_grasped": grasped,
                    "ever_lifted": lifted, "ever_transported": transported,
                    "phase": ("task_success" if native_success else "no_grasp" if not grasped
                              else "grasp_no_lift" if not lifted
                              else "lift_no_transport" if not transported
                              else "transport_or_place_failed"),
                },
                "first_action": first_action,
            }
            return record
        finally:
            env.close()

    def run(self, count):
        import imageio.v2 as imageio

        self.output.mkdir(parents=True, exist_ok=False)
        rows = []
        writer = imageio.get_writer(str(self.output / "representative.tmp.mp4"),
                                    fps=20, codec="libx264", macro_block_size=1)
        try:
            for index in range(count):
                scene = self.scene_set.load(index)
                path = self.output / f"episode_{index:03d}.json"
                try:
                    previous_handler = signal.signal(signal.SIGALRM, _deadline)
                    signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_SECONDS)
                    started = time.monotonic()
                    record = self.run_episode(index, scene,
                                              writer if index in VIDEO_SCENES else None)
                    record["wall_seconds"] = round(time.monotonic() - started, 3)
                except Exception as exc:
                    record = {"scene_index": index, "scene_id": scene["metadata"]["scene_id"],
                              "seed": int(scene["metadata"]["seed"]), "status": "runtime_error",
                              "error": f"{type(exc).__name__}: {exc}"}
                    if isinstance(exc, EpisodeDeadline):
                        record["error_kind"] = "wall_timeout"
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous_handler)
                _json_atomic(path, record)
                rows.append(record)
        finally:
            writer.close()
            if (self.output / "representative.tmp.mp4").exists():
                (self.output / "representative.tmp.mp4").replace(self.output / "representative.mp4")
        metrics = summarize_episodes(rows, count)
        metrics["episodes"] = rows
        _json_atomic(self.output / "summary.json", metrics)
        return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--constrain-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    catalog = PickCubeSceneSet(args.scene_dir)
    if len(catalog) != 100:
        raise ValueError(f"formal PickCube evaluation requires 100 scenes, got {len(catalog)}")
    from examples.baselines.constrain.pickcube_base_dp import PickCubeBaseDPAdapter
    from examples.baselines.constrain.pickcube_constrain import PickCubeConstrainAdapter

    adapters = {
        "base_dp": PickCubeBaseDPAdapter(args.base_checkpoint, device=args.device, weights="ema"),
        "constrain_dp": PickCubeConstrainAdapter(args.constrain_checkpoint, device=args.device),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frozen = {
        "scene_manifest_sha256": catalog.manifest["manifest_sha256"],
        "scene_file_sha256": catalog.manifest["scenes_file_sha256"],
        "checkpoint_paths": {"base_dp": str(args.base_checkpoint),
                             "constrain_dp": str(args.constrain_checkpoint)},
        "checkpoint_sha256": {"base_dp": _sha(args.base_checkpoint),
                              "constrain_dp": _sha(args.constrain_checkpoint)},
        "checkpoint_metadata": {
            "base_dp": adapters["base_dp"].metadata,
            "constrain_dp": {"checkpoint": str(args.constrain_checkpoint), "weights": "ema",
                             "format": "pickcube_constrain_native",
                             "model_config": adapters["constrain_dp"].config,
                             "control_enabled": adapters["constrain_dp"].control_enabled},
        },
        "python": platform.python_version(), "torch": str(torch.__version__),
        "device": args.device, "max_steps": 500, "history_frames": 2,
        "prediction_horizon": 16, "execution_horizon": 8,
        "episode_wall_seconds": EPISODE_WALL_SECONDS,
        "evaluator_source_sha256": _sha(__file__),
        "noise_seed_rule": "torch.manual_seed(scene seed) before inference for each scene",
        "video_scene_indexes_fixed_before_results": list(VIDEO_SCENES),
        "interpretation": "single generator and fixed base-side detour; no path-choice or isolation-generalization claim",
    }
    _json_atomic(args.output_dir / "frozen_config.json", frozen)
    _json_atomic(args.output_dir / "manifest.json", {
        "status": "running", "requested_scenes": 100,
        "scene_manifest_sha256": frozen["scene_manifest_sha256"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "video_scene_indexes_fixed_before_results": list(VIDEO_SCENES),
        "episodes_per_policy": 100,
    })
    summaries = {}
    for name in POLICIES:
        summaries[name] = PickCubePairedEvaluator(
            adapters[name], catalog, name, args.output_dir / name, args.device
        ).run(100)
    paired = {"complete": all(s["complete"] for s in summaries.values()),
              "policies": summaries,
              "paired_scene_ids_equal": True,
              "checkpoint_sha256": frozen["checkpoint_sha256"],
              "scene_manifest_sha256": frozen["scene_manifest_sha256"],
              "video_scene_indexes": list(VIDEO_SCENES)}
    fields = (("native_success", "task_success"), ("safe_success", "safe_success"))
    paired["paired_counts"] = {}
    for outcome, label in fields:
        left = [json.loads((args.output_dir / "base_dp" / f"episode_{i:03d}.json").read_text()) for i in range(100)]
        right = [json.loads((args.output_dir / "constrain_dp" / f"episode_{i:03d}.json").read_text()) for i in range(100)]
        a = [bool(x.get("result", {}).get(outcome, False)) for x in left]
        b = [bool(x.get("result", {}).get(outcome, False)) for x in right]
        paired["paired_counts"][label] = {
            "both": sum(x and y for x, y in zip(a, b)),
            "base_only": sum(x and not y for x, y in zip(a, b)),
            "constrain_only": sum(y and not x for x, y in zip(a, b)),
            "neither": sum(not x and not y for x, y in zip(a, b)),
        }
    for index in range(100):
        left = json.loads((args.output_dir / "base_dp" / f"episode_{index:03d}.json").read_text())
        right = json.loads((args.output_dir / "constrain_dp" / f"episode_{index:03d}.json").read_text())
        if (left.get("scene_id") != right.get("scene_id")
                or left.get("restored_state_sha256") != right.get("restored_state_sha256")
                or left.get("restored_controller_sha256") != right.get("restored_controller_sha256")
                or left.get("restored_observation_sha256") != right.get("restored_observation_sha256")):
            paired["paired_scene_ids_equal"] = False
    _json_atomic(args.output_dir / "paired_summary.json", paired)
    _json_atomic(args.output_dir / "manifest.json", {
        "status": "complete" if paired["complete"] else "complete_with_runtime_errors",
        "requested_scenes": 100, "scene_manifest_sha256": frozen["scene_manifest_sha256"],
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "video_scene_indexes_fixed_before_results": list(VIDEO_SCENES),
        "episodes_per_policy": 100,
    })
    print(json.dumps({name: {k: v for k, v in summary.items() if k != "episodes"}
                      for name, summary in summaries.items()}, indent=2))
    if not paired["complete"]:
        raise SystemExit("paired evaluation retained all records but reported runtime errors")
    return paired


if __name__ == "__main__":
    main()
