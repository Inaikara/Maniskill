"""Collect and replay orientation-conditioned PegInsertionSide experts."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from transforms3d.quaternions import quat2mat

from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common
from mani_skill.utils.wrappers.record import RecordEpisode

from examples.baselines.constrain import peg_orientation


ENV_IDS = {
    "fixed": peg_orientation.ENV_ID,
    "random": peg_orientation.RANDOM_ENV_ID,
}
PROVENANCE_FILES = (
    "examples/baselines/constrain/peg_orientation.py",
    "examples/baselines/constrain/peg_orientation_preflight.py",
    "mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py",
    "mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py",
    "mani_skill/trajectory/utils/actions/conversion.py",
    "mani_skill/trajectory/replay_trajectory.py",
    "mani_skill/envs/tasks/tabletop/peg_insertion_side.py",
)


def yaw_from_pose(pose):
    quat = np.asarray(pose.q[0].detach().cpu().numpy())
    axis = quat2mat(quat)[:, 0]
    return float(np.arctan2(axis[1], axis[0]))


def install_mplib_base_pose_compat(planner_class=None):
    """Adapt ManiSkill's seven-vector base pose to MPLib's Pose binding."""
    import mplib

    planner_class = mplib.Planner if planner_class is None else planner_class
    if getattr(planner_class, "_maniskill_base_pose_compat", False):
        return
    original = planner_class.set_base_pose
    original_plan_screw = planner_class.plan_screw

    def set_base_pose(planner, pose):
        if isinstance(pose, np.ndarray):
            pose = mplib.Pose(
                p=np.asarray(pose[:3], dtype=np.float64),
                q=np.asarray(pose[3:7], dtype=np.float64),
            )
        return original(planner, pose)

    planner_class.set_base_pose = set_base_pose

    def plan_screw(planner, *args, **kwargs):
        use_point_cloud = kwargs.pop("use_point_cloud", False)
        if use_point_cloud:
            raise NotImplementedError("this MPLib binding cannot plan with point clouds")
        if args and isinstance(args[0], np.ndarray) and args[0].shape == (7,):
            args = (
                mplib.Pose(
                    p=np.asarray(args[0][:3], dtype=np.float64),
                    q=np.asarray(args[0][3:], dtype=np.float64),
                ),
                *args[1:],
            )
        return original_plan_screw(planner, *args, **kwargs)

    planner_class.plan_screw = plan_screw
    planner_class._maniskill_base_pose_compat = True


def collect(args):
    install_mplib_base_pose_compat()
    from mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side import solve

    env_id = ENV_IDS[args.distribution]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = gym.make(
        env_id, obs_mode="state", control_mode="pd_joint_pos",
        sim_backend="physx_cpu", reward_mode="sparse", max_episode_steps=100,
    )
    recorder = RecordEpisode(
        env, output_dir=str(output), trajectory_name="joint_expert",
        save_trajectory=True, save_video=False, save_on_reset=False,
        record_env_state=True, source_type="motionplanning",
        source_desc="Official PegInsertionSide motion-planning expert; native pd_joint_pos",
    )
    attempts = []
    accepted = 0
    started = time.monotonic()
    try:
        for attempt in range(args.max_attempts):
            if accepted >= args.target_successes:
                break
            if time.monotonic() - started >= args.wall_seconds:
                break
            seed = args.seed + attempt
            recorder.reset(seed=seed)
            task = recorder.unwrapped
            hole_center = task.box_hole_pose.p[0].detach().cpu().numpy().tolist()
            peg_yaw = yaw_from_pose(task.peg.pose)
            target_yaw = float(task.target_yaw)
            peg_half_size = task.peg_half_sizes[0].detach().cpu().numpy().tolist()
            hole_radius = float(task.box_hole_radii[0].detach().cpu().item())
            try:
                result = solve(recorder, seed=seed, debug=False, vis=False)
                success = result != -1 and bool(np.asarray(result[-1]["success"].cpu()).item())
                reason = "expert_success" if success else "expert_failure"
            except Exception as exc:
                success = False
                reason = f"{type(exc).__name__}: {exc}"
            steps = len(recorder._trajectory_buffer.done) - recorder._trajectory_buffer.env_episode_ptr[0] - 1
            recorder.flush_trajectory(save=success)
            row = {
                "attempt": attempt, "seed": seed, "accepted_by_expert": success,
                "reason": reason, "steps": int(steps), "peg_initial_yaw": peg_yaw,
                "target_yaw": target_yaw, "hole_center": hole_center,
                "peg_half_size": peg_half_size, "hole_radius": hole_radius,
            }
            attempts.append(row)
            if success:
                accepted += 1
            (output / "attempts.jsonl").open("a").write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        recorder.close()
    report = {
        "distribution": args.distribution, "env_id": env_id,
        "target_successes": args.target_successes, "expert_accepted": accepted,
        "attempt_count": len(attempts), "attempt_limit": args.max_attempts,
        "wall_seconds_limit": args.wall_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "complete": accepted == args.target_successes, "attempts": attempts,
        "joint_h5_sha256": sha256(output / "joint_expert.h5"),
        "config": {"seed": args.seed, "sim_backend": "physx_cpu", "obs_mode": "state",
                   "expert_control_mode": "pd_joint_pos", "converted_control_mode": "pd_ee_delta_pose",
                   "robot_uids": "panda_wristcam", "max_episode_steps": 100},
        "runtime": {"python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name)
                                 for name in ("numpy", "torch", "gymnasium", "mani-skill", "mplib")}},
        "source_sha256": {name: sha256(Path(name)) for name in PROVENANCE_FILES},
        "action_contract": {"control_mode": "pd_joint_pos", "converted_to": "pd_ee_delta_pose"},
    }
    (output / "collection_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise RuntimeError(f"expert collection incomplete: {accepted}/{args.target_successes}")
    return report


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_valid_episodes(dataset_dir, count):
    dataset_dir = Path(dataset_dir)
    candidate = dataset_dir / "joint_expert.state.pd_ee_delta_pose.physx_cpu.h5"
    candidate_json = candidate.with_suffix(".json")
    final_h5 = dataset_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    final_json = final_h5.with_suffix(".json")
    candidate_meta = json.loads(candidate_json.read_text())
    selected = candidate_meta["episodes"][:count]
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} successful episodes available; need {count}")
    with h5py.File(candidate, "r") as source, h5py.File(final_h5, "w") as target:
        for episode_id, row in enumerate(selected):
            source_key = f"traj_{row['episode_id']}"
            if source_key not in source:
                raise ValueError(f"successful replay metadata points to missing {source_key}")
            source.copy(source_key, target, name=f"traj_{episode_id}")
            row["episode_id"] = episode_id
    candidate_meta["episodes"] = selected
    final_json.write_text(json.dumps(candidate_meta, indent=2, sort_keys=True) + "\n")
    collection = json.loads((dataset_dir / "collection_report.json").read_text())
    replay_result = json.loads((dataset_dir / "replay_audit.json").read_text())
    selected_rows = replay_result["episodes"][:count]
    manifest = {
        "dataset": dataset_dir.name, "episode_count": count,
        "control_mode": "pd_ee_delta_pose", "observation_mode": "state",
        "sim_backend": "physx_cpu", "robot_uids": "panda_wristcam",
        "source_joint_h5_sha256": sha256(dataset_dir / "joint_expert.h5"),
        "all_converted_candidates_sha256": sha256(candidate),
        "trajectory_h5_sha256": sha256(final_h5),
        "trajectory_json_sha256": sha256(final_json),
        "selected_seeds": [int(row["episode_seed"]) for row in selected],
        "selected_target_yaws": [float(row["target_yaw"]) for row in selected_rows],
        "selected_peg_initial_yaws": [float(row["peg_initial_yaw"]) for row in selected_rows],
        "selected_episode_lengths": [int(row["T"]) for row in selected_rows],
        "all_observation_action_hint_alignment": all(
            row["state_count"] == row["observation_count"] == row["hint_count"] == row["T"] + 1
            and row["action_shape"] == [row["T"], 7]
            for row in selected_rows
        ),
        "all_selected_episodes_have_rotation_control": all(
            row["rotation_nonzero_steps"] > 0 for row in selected_rows
        ),
        "all_selected_episodes_pass_state_restore": all(
            len(row["restore_checks"]) == 2 for row in selected_rows
        ),
        "collection_attempts": collection["attempt_count"],
        "expert_successes": collection["expert_accepted"],
        "converted_replay_attempts": len(replay_result["attempts"]),
        "converted_replay_successes": replay_result["native_successful_replays"],
        "replay_failures": [row for row in replay_result["attempts"]
                            if not row["converted_replay_success"]],
        "source_sha256": collection["source_sha256"],
    }
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def summarize_pair(fixed_dir, random_dir):
    fixed_dir, random_dir = Path(fixed_dir), Path(random_dir)
    fixed_collection = json.loads((fixed_dir / "collection_report.json").read_text())
    random_collection = json.loads((random_dir / "collection_report.json").read_text())
    fixed_manifest = json.loads((fixed_dir / "dataset_manifest.json").read_text())
    random_manifest = json.loads((random_dir / "dataset_manifest.json").read_text())
    fixed_attempts = {row["seed"]: row for row in fixed_collection["attempts"]}
    random_attempts = {row["seed"]: row for row in random_collection["attempts"]}
    if set(fixed_attempts) != set(random_attempts):
        raise ValueError("fixed and random collections do not share the same paired seeds")
    center_differences = {
        str(seed): float(np.max(np.abs(
            np.asarray(fixed_attempts[seed]["hole_center"])
            - np.asarray(random_attempts[seed]["hole_center"])
        )))
        for seed in sorted(fixed_attempts)
    }
    max_center_difference = max(center_differences.values(), default=0.0)
    if max_center_difference > 1e-6:
        raise ValueError(f"hole center changed with target yaw: {max_center_difference}")
    for manifest in (fixed_manifest, random_manifest):
        if manifest["episode_count"] != 50:
            raise ValueError("each materialized orientation dataset must contain exactly 50 episodes")
        if not (manifest["all_observation_action_hint_alignment"]
                and manifest["all_selected_episodes_have_rotation_control"]
                and manifest["all_selected_episodes_pass_state_restore"]):
            raise ValueError("dataset action/state/hint or restore audit failed")
    fixed_yaws = [row["target_yaw"] for row in fixed_collection["attempts"]]
    random_yaws = [row["target_yaw"] for row in random_collection["attempts"]]
    def geometry_coverage(collection):
        rows = collection["attempts"]
        return {
            "hole_center_axis_ranges_m": [
                [min(row["hole_center"][axis] for row in rows),
                 max(row["hole_center"][axis] for row in rows)]
                for axis in range(3)
            ],
            "peg_half_size_axis_ranges_m": [
                [min(row["peg_half_size"][axis] for row in rows),
                 max(row["peg_half_size"][axis] for row in rows)]
                for axis in range(3)
            ],
            "hole_radius_range_m": [
                min(row["hole_radius"] for row in rows),
                max(row["hole_radius"] for row in rows),
            ],
        }
    if not np.allclose(fixed_yaws, peg_orientation.TARGET_YAW, atol=1e-6):
        raise ValueError("fixed target yaw was not held at 90 degrees")
    if (min(random_yaws) < peg_orientation.TARGET_YAW - peg_orientation.YAW_HALF_RANGE
            or max(random_yaws) > peg_orientation.TARGET_YAW + peg_orientation.YAW_HALF_RANGE):
        raise ValueError("random target yaw escaped the registered range")
    result = {
        "preflight_pass": True,
        "datasets": {"fixed": fixed_manifest, "random": random_manifest},
        "paired_scene_audit": {
            "paired_seed_count": len(fixed_attempts),
            "max_hole_center_difference_m": max_center_difference,
            "all_initial_peg_yaws_match_by_seed": all(
                abs(fixed_attempts[seed]["peg_initial_yaw"]
                    - random_attempts[seed]["peg_initial_yaw"]) < 1e-7
                for seed in fixed_attempts
            ),
            "center_differences_by_seed": center_differences,
        },
        "coverage": {
            "fixed_target_yaw_rad": [min(fixed_yaws), max(fixed_yaws)],
            "random_target_yaw_rad": [min(random_yaws), max(random_yaws)],
            "fixed_peg_initial_yaw_rad": [
                min(row["peg_initial_yaw"] for row in fixed_collection["attempts"]),
                max(row["peg_initial_yaw"] for row in fixed_collection["attempts"]),
            ],
            "random_peg_initial_yaw_rad": [
                min(row["peg_initial_yaw"] for row in random_collection["attempts"]),
                max(row["peg_initial_yaw"] for row in random_collection["attempts"]),
            ],
            "random_target_yaw_histogram_5_bins": np.histogram(
                random_yaws,
                bins=np.linspace(peg_orientation.TARGET_YAW - peg_orientation.YAW_HALF_RANGE,
                                 peg_orientation.TARGET_YAW + peg_orientation.YAW_HALF_RANGE, 6),
            )[0].tolist(),
            "fixed_geometry": geometry_coverage(fixed_collection),
            "random_geometry": geometry_coverage(random_collection),
        },
        "expert_reachability": {
            "fixed_expert_successes": fixed_collection["expert_accepted"],
            "fixed_attempts": fixed_collection["attempt_count"],
            "fixed_converted_replay_successes": fixed_manifest["converted_replay_successes"],
            "fixed_converted_replay_candidates": fixed_manifest["converted_replay_attempts"],
            "random_expert_successes": random_collection["expert_accepted"],
            "random_attempts": random_collection["attempt_count"],
            "random_converted_replay_successes": random_manifest["converted_replay_successes"],
            "random_converted_replay_candidates": random_manifest["converted_replay_attempts"],
        },
    }
    return result


def audit_converted(path):
    rows = []
    env = None
    with h5py.File(path, "r") as h5:
        json_path = Path(path).with_suffix(".json")
        metadata = json.loads(json_path.read_text())
        env_kwargs = dict(metadata["env_info"]["env_kwargs"])
        env_kwargs["sim_backend"] = "physx_cpu"
        env_kwargs["control_mode"] = "pd_ee_delta_pose"
        env = gym.make(metadata["env_info"]["env_id"], **env_kwargs)
        for key in sorted(h5, key=lambda item: int(item.split("_")[1])):
            traj = h5[key]
            actions = np.asarray(traj["actions"])
            states = traj["env_states"]
            yaw_state = np.asarray(states[peg_orientation.STATE_KEY])
            peg_pose = np.asarray(states["actors"]["peg"])
            observations = traj["obs"]
            T = len(actions)
            if isinstance(observations, h5py.Group):
                obs_count = len(observations[next(iter(observations))])
            else:
                obs_count = len(observations)
            if (actions.shape != (T, 7) or len(yaw_state) != T + 1
                    or len(peg_pose) != T + 1 or obs_count != T + 1):
                raise ValueError(f"{key}: converted action/state/hint lengths are misaligned")
            peg_yaws = []
            for raw in peg_pose:
                quat = raw[3:7]
                axis = quat2mat(quat)[:, 0]
                peg_yaws.append(float(np.arctan2(axis[1], axis[0])))
            target_yaws = yaw_state[:, 0]
            hints = peg_orientation.build_orientation_hint(np.asarray(peg_yaws), target_yaws)
            if not np.isfinite(actions).all() or not np.isfinite(hints).all():
                raise ValueError(f"{key}: non-finite converted action or hint")
            if not np.any(np.abs(actions[:, 3:6]) > 1e-6):
                raise ValueError(f"{key}: converted rotation channels are all zero")
            restored = []
            for state_index in (0, T):
                state = trajectory_utils.index_dict(states, state_index)
                env.reset(seed=int(metadata["episodes"][int(key.split("_")[1])]["episode_seed"]))
                env.unwrapped.set_state_dict(common.batch(state))
                restored_hint = env.unwrapped.get_orientation_hint()
                restored_center = env.unwrapped.box_hole_pose.p[0].detach().cpu().numpy()
                if (not np.allclose(restored_hint, hints[state_index], atol=1e-5)
                        or not np.allclose(restored_center, yaw_state[state_index, 1:], atol=1e-5)):
                    raise ValueError(f"{key}: state restoration changed hole center or orientation hint")
                restored.append({"state_index": state_index, "hint": restored_hint.tolist(),
                                 "hole_center": restored_center.tolist()})
            rows.append({
                "trajectory": key, "T": T, "state_count": T + 1,
                "action_shape": list(actions.shape),
                "rotation_nonzero_steps": int(np.any(np.abs(actions[:, 3:6]) > 1e-6, axis=1).sum()),
                "hint_count": len(hints), "initial_hint": hints[0].tolist(),
                "final_hint": hints[-1].tolist(),
                "hint_sha256": hashlib.sha256(hints.astype(np.float64).tobytes()).hexdigest(),
                "observation_count": obs_count,
                "restore_checks": restored,
                "target_yaw": float(target_yaws[0]), "peg_initial_yaw": peg_yaws[0],
                "hole_center": np.asarray(states[peg_orientation.STATE_KEY])[0, 1:].tolist(),
            })
    if env is not None:
        env.close()
    return rows


def replay(args):
    # Importing the task module above registers the custom ID before official replay.
    from mani_skill.trajectory.replay_trajectory import Args, main

    source = Path(args.dataset_dir) / "joint_expert.h5"
    replay_args = Args(
        traj_path=str(source), sim_backend="physx_cpu", obs_mode="state",
        target_control_mode="pd_ee_delta_pose", save_traj=True,
        count=args.count, max_retry=0, allow_failure=False,
    )
    main(replay_args)
    converted = source.with_name("joint_expert.state.pd_ee_delta_pose.physx_cpu.h5")
    rows = audit_converted(converted) if converted.exists() else []
    source_manifest = json.loads(source.with_suffix(".json").read_text())
    converted_json = converted.with_suffix(".json")
    replayed_seeds = set()
    if converted_json.exists():
        replayed_seeds = {
            int(row["episode_seed"])
            for row in json.loads(converted_json.read_text())["episodes"]
        }
    attempt_outcomes = [
        {"seed": int(row["episode_seed"]), "expert_success": bool(row["success"]),
         "converted_replay_success": int(row["episode_seed"]) in replayed_seeds,
         "reason": "official_replay_success" if int(row["episode_seed"]) in replayed_seeds
         else "official_replay_native_success_failed"}
        for row in source_manifest["episodes"]
    ]
    result = {
        "source_h5_sha256": sha256(source), "converted_h5": str(converted),
        "converted_h5_sha256": sha256(converted) if converted.exists() else None,
        "target_control_mode": "pd_ee_delta_pose", "attempts": attempt_outcomes,
        "native_successful_replays": len(rows), "minimum_successes": args.min_successes,
        "complete": len(rows) >= args.min_successes, "episodes": rows,
    }
    (Path(args.dataset_dir) / "replay_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["complete"]:
        raise RuntimeError(f"official replay accepted {len(rows)}; requires {args.min_successes}")
    materialize_count = (result["native_successful_replays"] if args.materialize_all
                        else args.min_successes)
    if materialize_count:
        result["dataset"] = materialize_valid_episodes(args.dataset_dir, materialize_count)
        (Path(args.dataset_dir) / "replay_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main_cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--distribution", choices=ENV_IDS, required=True)
    collect_parser.add_argument("--output-dir", required=True)
    collect_parser.add_argument("--target-successes", type=int, required=True)
    collect_parser.add_argument("--max-attempts", type=int, required=True)
    collect_parser.add_argument("--wall-seconds", type=float, default=1600)
    collect_parser.add_argument("--seed", type=int, default=2026111100)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--dataset-dir", required=True)
    replay_parser.add_argument("--count", type=int, required=True)
    replay_parser.add_argument("--min-successes", type=int, default=0)
    replay_parser.add_argument("--materialize-all", action="store_true")
    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--fixed-dir", required=True)
    summary_parser.add_argument("--random-dir", required=True)
    summary_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args)
    elif args.command == "replay":
        replay(args)
    else:
        result = summarize_pair(args.fixed_dir, args.random_dir)
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main_cli()
