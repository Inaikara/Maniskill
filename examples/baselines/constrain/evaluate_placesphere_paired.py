"""Frozen paired evaluation of PlaceSphere policies on expert-verified scenes."""

import argparse
import hashlib
import json
import platform
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.placesphere_scenes import (
    PlaceSphereSceneSet, _tree_hash,
)

VIDEO_SCENES = (0, 1, 2)
POLICIES = ("base_dp", "constrain_dp")
EPISODE_WALL_SECONDS = 120
MAX_STEPS = 200


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


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=lambda item: item.item() if isinstance(item, np.generic) else str(item)) + "\n")
    temporary.replace(path)


def _bool(info, key):
    value = info[key]
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).reshape(-1)[0])


def classify_termination(native_success, terminated, truncated, elapsed, max_steps):
    if native_success:
        return "native_success"
    if truncated:
        return "timeout"
    if terminated:
        return "environment_terminal"
    if elapsed >= max_steps:
        return "control_step_limit"
    return None


def classify_phase(ever_grasped, ever_lifted, ever_transport, ever_released, native_success):
    if native_success:
        return "task_success"
    if not ever_grasped:
        return "no_grasp"
    if not ever_lifted:
        return "grasp_no_lift"
    if not ever_transport:
        return "grasped_no_transport"
    if not ever_released:
        return "transport_no_release"
    return "release_or_stability_failed"


def policy_condition(adapter, queued):
    condition = adapter.base_cond(queued)
    if hasattr(adapter, "agent"):
        return condition.reshape(-1, 2, 39)
    return condition


def summarize_episodes(rows, requested=100):
    evaluated = [r["result"] for r in rows if r.get("status") == "evaluated"]
    if requested <= 0:
        raise ValueError("requested denominator must be positive")
    count = len(evaluated)
    fields = {
        "task_success": lambda r: r["native_success"],
        "safe_success": lambda r: r["safe_success"],
        "obstacle_contact": lambda r: r["obstacle_contact"],
        "xy_forbidden_crossing": lambda r: r["xy_forbidden_crossing"],
    }
    result = {"requested_denominator": requested, "evaluated": count,
              "runtime_errors": requested-count, "complete": len(rows) == requested and count == requested,
              "control_steps": sum(r["elapsed_steps"] for r in evaluated)}
    for name, predicate in fields.items():
        n = sum(predicate(r) for r in evaluated)
        result[f"{name}_count"] = n
        result[f"{name}_rate"] = n/requested
    for reason in ("timeout", "environment_terminal", "control_step_limit"):
        result[f"{reason}_count"] = sum(r["termination_reason"] == reason for r in evaluated)
    result["wall_timeout_runtime_errors"] = sum(r.get("error_kind") == "wall_timeout" for r in rows)
    result["phase_counts"] = {phase: sum(r["phase"] == phase for r in evaluated) for phase in
        ("no_grasp", "grasp_no_lift", "grasped_no_transport", "transport_no_release",
         "release_or_stability_failed", "task_success")}
    result["mean_steps_evaluated"] = (result["control_steps"] / count) if count else None
    return result


def _frame(env):
    image = env.render()
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    return np.asarray(image).squeeze().astype(np.uint8)


def _make_env(scene, render=False):
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
    cylinder = scene["metadata"]["cylinder"]
    config = PlaceSpherePilotConfig(max_steps=MAX_STEPS,
        obstacle_radius=cylinder["radius"], obstacle_height=cylinder["height"], safety_margin=.005)
    return make_env("rgb_array" if render else None, config)


class PlaceSpherePairedEvaluator:
    def __init__(self, adapter, catalog, output, device):
        self.adapter, self.catalog = adapter, catalog
        self.output, self.device = Path(output), device
        self.xy_feedback = False

    def run_episode(self, index, scene, writer=None):
        env = _make_env(scene, render=writer is not None)
        try:
            from examples.baselines.constrain.render_only_xy_eval import (
                TaskProgressStall, consume_replan_request, should_replan_after_control,
            )
            env, restored = self.catalog.restore(index, env)
            base = env.unwrapped
            state = base.get_state_dict()
            controller = base.agent.controller.get_state()
            state_hash, controller_hash = _tree_hash(state), _tree_hash(controller)
            obs = restored["raw_observation"]
            obs_hash = hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest()
            meta = scene["metadata"]
            if (state_hash != meta["state_sha256"] or controller_hash != meta["controller_sha256"]
                    or obs_hash != meta["raw_observation_sha256"]):
                raise ValueError("restored state/controller/raw observation does not match scene catalog")
            obs = np.asarray(obs, dtype=np.float32).reshape(42)
            if not np.array_equal(obs[:39], scene["policy_observation"].reshape(39)):
                raise ValueError("restored policy observation differs from catalog")
            policy_hash = hashlib.sha256(obs[:39].tobytes()).hexdigest()
            # The two policies receive an identical, fixed two-frame history and independent
            # RNG initialization from the scene seed before any model sampling.
            history = np.repeat(obs[None, :], 2, axis=0)
            torch.manual_seed(int(meta["seed"]))
            if writer:
                writer.append_data(_frame(env))
            elapsed = 0
            collision = crossing = grasped = lifted = transported = released = False
            start_z = float(base.obj.pose.p[0, 2].detach().cpu())
            goal = base.bin.pose.p[0, :2].detach().cpu().numpy()
            first_success = None
            term = trunc = False
            termination = None
            actions = []
            task_action_count = 0
            while elapsed < MAX_STEPS:
                consume_replan_request(env, elapsed + 1)
                queued = torch.as_tensor(history[None], dtype=torch.float32, device=self.device)
                condition = policy_condition(self.adapter, queued)
                prediction = self.adapter.get_action(condition)
                mapper = getattr(self.adapter, "map_action", None)
                if mapper is None:
                    mapper = self.adapter.map_action_ee_delta_pose
                action_batch = mapper(prediction)
                chunk = action_batch[0][:min(8, MAX_STEPS-elapsed)]
                feedback = None
                if self.xy_feedback:
                    from examples.baselines.constrain.render_only_xy_eval import PlaceXYFeedback
                    feedback = PlaceXYFeedback(self.adapter, meta["seed"], self.device)
                    feedback.start_chunk(action_batch)
                auxiliary_plan = None
                auxiliary_offset = 0
                offset = 0
                while offset < len(chunk) and elapsed < MAX_STEPS:
                    action = chunk[offset]
                    a = action.detach().cpu().numpy()
                    scheduled_action = a.copy()
                    auxiliary_xy = None
                    if feedback is not None:
                        selected = feedback.action_for_control(offset, auxiliary_plan, auxiliary_offset)
                        a = selected.detach().cpu().numpy()
                        if auxiliary_plan is not None:
                            auxiliary_xy = a[:2].tolist()
                            auxiliary_offset += 1
                    try:
                        next_obs, _, terminated, truncated, info = env.step(a[None])
                    except Exception as exc:
                        if (isinstance(exc, TaskProgressStall)
                                or getattr(env, "_render_only_xy_progress_stall", None) is not None):
                            termination = "task_progress_stall"
                            break
                        raise
                    elapsed += 1
                    term, trunc = bool(terminated), bool(truncated)
                    attempt = env._render_only_xy_last_attempt
                    actions.append((attempt["filtered_action"] if getattr(
                        self, "progress_guard", False) else a.tolist()))
                    next_obs = np.asarray(next_obs.detach().cpu() if torch.is_tensor(next_obs) else next_obs,
                                          dtype=np.float32).reshape(-1)
                    history = np.concatenate((history[1:], next_obs[None]), axis=0)
                    if feedback is not None:
                        attempt["scheduled_chunk_index"] = int((elapsed - 1) // 8)
                        attempt["scheduled_chunk_offset"] = int(offset)
                        attempt["original_scheduled_action"] = scheduled_action.tolist()
                        attempt["auxiliary_xy_action"] = auxiliary_xy
                        attempt["executed_action_provenance"] = (
                            "scheduled_non_xy_plus_auxiliary_xy" if auxiliary_plan is not None
                            else "scheduled_full_action")
                    if getattr(self, "progress_guard", False):
                        attempt["task_action_index"] = int(task_action_count + 1)
                        attempt["scheduled_chunk_index"] = int(task_action_count // 8)
                        attempt["scheduled_chunk_offset"] = int(task_action_count % 8)
                        attempt["original_scheduled_action"] = scheduled_action.tolist()
                        attempt["executed_action_provenance"] = (
                            "geometric_tangent_pause" if not attempt.get("task_action_executed", True)
                            else "scheduled_full_action")
                        attempt["auxiliary_xy_control_index"] = (
                            int(elapsed) if attempt.get("xy_feedback_requested") else None)
                    collision |= _bool(info, "obstacle_collision")
                    crossing |= _bool(info, "obstacle_crossing")
                    now_grasped = _bool(info, "is_obj_grasped")
                    grasped |= now_grasped
                    position = base.obj.pose.p[0].detach().cpu().numpy()
                    lifted |= position[2] > start_z + .04
                    transported |= np.linalg.norm(position[:2]-goal) < .10
                    released |= grasped and not now_grasped
                    success = _bool(info, "native_success")
                    if success and first_success is None:
                        first_success = elapsed
                    termination = classify_termination(success, term, trunc, elapsed, MAX_STEPS)
                    if getattr(env, "_render_only_xy_progress_stall", None) is not None:
                        termination = "task_progress_stall"
                    # Include/render the whole successful control step, including safety updates.
                    if writer:
                        writer.append_data(_frame(env))
                    if termination:
                        break
                    if attempt.get("task_action_executed", True):
                        task_action_count += 1
                        offset += 1
                    if (feedback is not None and attempt.get("xy_feedback_requested")
                            and offset + 1 < len(chunk)):
                        from examples.baselines.constrain.render_only_xy_eval import auxiliary_xy_plan
                        queued = torch.as_tensor(history[None], dtype=torch.float32, device=self.device)
                        updated_condition = policy_condition(self.adapter, queued)
                        auxiliary_plan = auxiliary_xy_plan(
                            self.adapter, updated_condition, meta["seed"], elapsed, self.device)
                        auxiliary_offset = 0
                        attempt["auxiliary_query"] = {
                            "condition_history_sha256": hashlib.sha256(
                                np.ascontiguousarray(history).tobytes()).hexdigest(),
                            "seed_formula": "(scene_seed * 1000003 + control_index) % (2**63 - 1)",
                            "seed": int((int(meta["seed"]) * 1_000_003 + elapsed) % (2**63 - 1)),
                            "provided_channels": ["x", "y"],
                            "scheduled_non_xy_preserved": True,
                            "plan_sha256": hashlib.sha256(
                                np.ascontiguousarray(auxiliary_plan.detach().cpu().numpy()).tobytes()
                            ).hexdigest(),
                        }
                    if should_replan_after_control(env, termination):
                        break
                if termination:
                    break
            termination = termination or classify_termination(False, term, trunc, elapsed, MAX_STEPS)
            native_success = termination == "native_success"
            safe_success = native_success and not collision and not crossing
            is_grasped = _bool(info, "is_obj_grasped")
            on_bin, static = _bool(info, "is_obj_on_bin"), _bool(info, "is_obj_static")
            phase = classify_phase(grasped, lifted, transported, released, native_success)
            return {"scene_index": index, "scene_id": meta["scene_id"], "seed": int(meta["seed"]),
                "status": "evaluated", "scene_state_sha256": meta["state_sha256"],
                "restored_state_sha256": state_hash, "restored_controller_sha256": controller_hash,
                "restored_raw_observation_sha256": obs_hash, "restored_policy_observation_sha256": policy_hash,
                "restoration_matches_catalog": True, "native_success_step": first_success,
                "task_progress_stall": getattr(env, "_render_only_xy_progress_stall", None),
                "task_action_count": int(task_action_count),
                "progress_guard_controls": sum(
                    step.get("progress_guard_event") == "tangent_detour_pause" for step in
                    getattr(env, "_render_only_xy_trace", [])),
                "progress_guard_max_pause": max((step.get("progress_guard_pause_control", 0)
                    for step in getattr(env, "_render_only_xy_trace", [])), default=0),
                "actions_native_pd_ee_delta_pose": actions,
                "result": {"termination_reason": termination, "elapsed_steps": elapsed,
                    "terminated": term, "truncated": trunc, "native_success": native_success,
                    "safe_success": safe_success, "obstacle_contact": collision,
                    "xy_forbidden_crossing": crossing, "ever_grasped": grasped,
                    "ever_lifted": lifted, "ever_transported": transported, "ever_released": released,
                    "terminal_is_grasped": is_grasped, "terminal_on_bin": on_bin,
                    "terminal_static": static, "phase": phase}}
        finally:
            env.close()

    def run(self):
        import imageio.v2 as imageio
        out = self.output / self.policy
        out.mkdir(parents=True, exist_ok=False)
        writer = imageio.get_writer(str(out / "representative.tmp.mp4"), fps=20,
                                    codec="libx264", macro_block_size=1)
        rows = []
        try:
            for index in range(100):
                scene = self.catalog.load(index)
                previous = signal.signal(signal.SIGALRM, _deadline)
                signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_SECONDS)
                started = time.monotonic()
                try:
                    record = self.run_episode(index, scene, writer if index in VIDEO_SCENES else None)
                    record["wall_seconds"] = round(time.monotonic()-started, 3)
                except Exception as exc:
                    record = {"scene_index": index, "scene_id": scene["metadata"]["scene_id"],
                              "seed": int(scene["metadata"]["seed"]), "status": "runtime_error",
                              "error": f"{type(exc).__name__}: {exc}"}
                    if isinstance(exc, EpisodeDeadline):
                        record["error_kind"] = "wall_timeout"
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                _atomic(out / f"episode_{index:03d}.json", record)
                rows.append(record)
        finally:
            writer.close()
            tmp = out / "representative.tmp.mp4"
            if tmp.exists():
                tmp.replace(out / "representative.mp4")
        summary = summarize_episodes(rows)
        summary["episodes"] = rows
        _atomic(out / "summary.json", summary)
        return summary


def paired_audit(root):
    root = Path(root)
    loaded = {name: [json.loads((root/name/f"episode_{i:03d}.json").read_text())
                     for i in range(100)] for name in POLICIES}
    audit = {"paired_records": 100, "restore_pairs_match": True, "outcomes": {}}
    for i, (left, right) in enumerate(zip(*(loaded[name] for name in POLICIES))):
        keys = ("scene_id", "seed", "restored_state_sha256", "restored_controller_sha256",
                "restored_raw_observation_sha256", "restored_policy_observation_sha256")
        same = all(left.get(k) == right.get(k) for k in keys)
        audit["restore_pairs_match"] &= same
        if not same:
            raise ValueError(f"paired restore mismatch at scene index {i}")
    for field in ("native_success", "safe_success"):
        a = [bool(row.get("result", {}).get(field, False)) for row in loaded["base_dp"]]
        b = [bool(row.get("result", {}).get(field, False)) for row in loaded["constrain_dp"]]
        audit["outcomes"][field] = {"both": sum(x and y for x,y in zip(a,b)),
            "base_only": sum(x and not y for x,y in zip(a,b)),
            "constrain_only": sum(y and not x for x,y in zip(a,b)),
            "neither": sum(not x and not y for x,y in zip(a,b))}
    audit["videos"] = {}
    for policy in POLICIES:
        video = root/policy/"representative.mp4"
        audit["videos"][policy] = {"sha256": _sha(video), "bytes": video.stat().st_size,
                                   "scene_indices": list(VIDEO_SCENES)}
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--constrain-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    catalog = PlaceSphereSceneSet(args.scene_dir)
    if len(catalog) != 100:
        raise ValueError(f"expected 100 frozen scenes, found {len(catalog)}")
    from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter
    from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
    # Strictly load both preselected artifacts before freezing the evaluation record.
    adapters = {
        "base_dp": PlaceSphereBaseDPAdapter(args.base_checkpoint, device=args.device, weights="ema"),
        "constrain_dp": PlaceSphereConstrainAdapter(args.constrain_checkpoint, device=args.device,
            base_checkpoint="outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path(__file__).with_name("placesphere_scenes.py"),
        Path(__file__).with_name("placesphere_obstacle.py"), Path(__file__).with_name("placesphere_pilot.py"),
        Path(__file__).with_name("placesphere_base_dp.py"), Path(__file__).with_name("placesphere_constrain.py"),
        Path(__file__).with_name("constrain_denoiser.py"),
        Path("mani_skill/envs/tasks/tabletop/place_sphere.py"),
        Path("mani_skill/examples/motionplanning/panda/solutions/place_sphere.py")]
    snapshot = args.output_dir / "source_snapshot"
    for source in source_files:
        target = snapshot / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if _sha(target) != _sha(source):
            raise IOError(f"source snapshot mismatch: {source}")
    checkpoints = {"base_dp": args.base_checkpoint, "constrain_dp": args.constrain_checkpoint}
    frozen = {"status": "frozen_before_first_episode", "scene_dir": str(args.scene_dir),
        "scene_manifest_sha256": _sha(args.scene_dir/"manifest.json"),
        "scene_file_sha256": _sha(args.scene_dir/"scenes.h5"),
        "scene_count": 100, "checkpoint_paths": {k: str(v) for k,v in checkpoints.items()},
        "checkpoint_sha256": {k: _sha(v) for k,v in checkpoints.items()},
        "source_sha256": {str(p): _sha(p) for p in source_files},
        "source_snapshots": {p.name: str(snapshot/p.name) for p in source_files},
        "model_metadata": {"base_dp": adapters["base_dp"].metadata,
            "constrain_dp": {"format": "placesphere_constrain_v1", "weights": "ema",
                "base_checkpoint_sha256": _sha(checkpoints["base_dp"])}},
        "device": args.device, "python": platform.python_version(), "torch": str(torch.__version__),
        "max_steps": MAX_STEPS, "episode_wall_seconds": EPISODE_WALL_SECONDS,
        "obs_horizon": 2, "prediction_horizon": 16, "execution_horizon": 8,
        "diffusion_steps": 100, "raw_observation_dim": 42, "policy_condition_dim": 39,
        "action_dim": 7, "video_scene_indexes_fixed_before_results": list(VIDEO_SCENES),
        "rng_reset_rule": "torch.manual_seed(scene seed) independently before each policy episode",
        "scene_generation": {"seed": 40017, "expert_verified": 100, "radius_m": .03,
            "height_m": .08, "safety_margin_m": .005, "same_base_side_only": True}}
    _atomic(args.output_dir/"frozen_config.json", frozen)
    _atomic(args.output_dir/"manifest.json", {"status": "running", "requested_scenes": 100,
        "episodes_per_policy": 100, "scene_manifest_sha256": frozen["scene_manifest_sha256"],
        "checkpoint_sha256": frozen["checkpoint_sha256"]})
    summaries = {}
    started = time.monotonic()
    try:
        for name in POLICIES:
            evaluator = PlaceSpherePairedEvaluator(adapters[name], catalog, args.output_dir, args.device)
            evaluator.policy = name
            summaries[name] = evaluator.run()
        audit = paired_audit(args.output_dir)
        paired = {"complete": all(s["complete"] for s in summaries.values()),
            "policies": {k: {x:y for x,y in v.items() if x != "episodes"} for k,v in summaries.items()},
            "paired_restore_audit": audit, "checkpoint_sha256": frozen["checkpoint_sha256"],
            "scene_manifest_sha256": frozen["scene_manifest_sha256"],
            "video_scene_indexes": list(VIDEO_SCENES), "elapsed_seconds": time.monotonic()-started}
        _atomic(args.output_dir/"paired_summary.json", paired)
        # Keep the completed manifest outside the file index to avoid a
        # self-referential checksum; it pins the checksum-index digest instead.
        files = [p for p in args.output_dir.rglob("*") if p.is_file()
                 and p.name not in ("checksums.json", "manifest.json")]
        _atomic(args.output_dir/"checksums.json", {p.relative_to(args.output_dir).as_posix(): _sha(p)
            for p in sorted(files)})
        _atomic(args.output_dir/"manifest.json", {"status": "complete" if paired["complete"] else "complete_with_runtime_errors",
            "requested_scenes": 100, "episodes_per_policy": 100,
            "scene_manifest_sha256": frozen["scene_manifest_sha256"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "paired_summary_sha256": _sha(args.output_dir/"paired_summary.json"),
            "checksums_sha256": _sha(args.output_dir/"checksums.json")})
        if not paired["complete"]:
            raise RuntimeError("evaluation completed with retained runtime error rows")
        print(json.dumps(paired, indent=2), flush=True)
        return paired
    except BaseException:
        if (args.output_dir/"manifest.json").exists():
            _atomic(args.output_dir/"manifest.json", {"status": "interrupted_or_failed",
                "requested_scenes": 100, "episodes_per_policy": 100,
                "scene_manifest_sha256": frozen["scene_manifest_sha256"],
                "checkpoint_sha256": frozen["checkpoint_sha256"]})
        raise


if __name__ == "__main__":
    main()
