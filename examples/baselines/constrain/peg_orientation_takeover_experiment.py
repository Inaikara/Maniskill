"""Immutable-preflight driver for bounded Peg live-expert handoff controls."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from examples.baselines.constrain import peg_orientation
from examples.baselines.constrain.peg_orientation_takeover import (
    capture_physical_state,
    native_grasped,
    run_live_expert_continuation,
    run_official_geometry_continuation,
    build_official_planner,
)


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "demos/PegInsertionSide-orientation-random-500"
H5 = DATA / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
META = DATA / "trajectory.state.pd_ee_delta_pose.physx_cpu.json"
MAX_PREFIX = 300
MAX_SUFFIX = 200
MAX_CONTROL_TOTAL = 2000
MAX_WALL_SECONDS = 900
HEAD_X_LOW = -0.10
HEAD_X_HIGH = -0.015
STATE_TOLERANCE = 1e-5
EPISODE_IDS = (0, 1)
SOURCE_PATHS = (
    "examples/baselines/constrain/peg_orientation_takeover.py",
    "examples/baselines/constrain/peg_orientation_takeover_experiment.py",
    "examples/baselines/constrain/peg_orientation_preflight.py",
    "examples/baselines/constrain/peg_orientation.py",
    "examples/baselines/constrain/peg_orientation_data.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover.py",
    "mani_skill/agents/base_agent.py",
    "mani_skill/agents/controllers/base_controller.py",
    "mani_skill/agents/controllers/pd_ee_pose.py",
    "mani_skill/agents/controllers/pd_joint_pos.py",
    "mani_skill/agents/robots/panda/panda.py",
    "mani_skill/agents/robots/panda/panda_wristcam.py",
    "mani_skill/envs/sapien_env.py",
    "mani_skill/envs/tasks/tabletop/peg_insertion_side.py",
    "mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py",
    "mani_skill/examples/motionplanning/panda/motionplanner.py",
    "mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py",
    "mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return repr(value)


def _physical(task):
    return capture_physical_state(task)


def _head_in_hole(task):
    head_in_hole = task.box_hole_pose.inv() * task.peg_head_pose
    xyz = np.asarray(head_in_hole.p[0].detach().cpu(), dtype=np.float64)
    inserted = bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any())
    return xyz, inserted


def _is_success(info):
    success = info.get("success", False)
    if isinstance(success, torch.Tensor):
        success = success.detach().cpu().numpy()
    return bool(np.asarray(success).any())


def _new_env():
    return gym.make(
        peg_orientation.RANDOM_ENV_ID,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu",
        reconfiguration_freq=1,
        max_episode_steps=600,
    )


def _trajectory(index):
    with h5py.File(H5, "r") as handle:
        group = handle[f"traj_{index}"]
        obs = np.asarray(group["obs"])
        actions = np.asarray(group["actions"])
        successes = np.asarray(group["success"])
    metadata = json.loads(META.read_text())["episodes"][index]
    return metadata, obs, actions, successes


def _reset_and_verify(env, metadata, recorded_obs):
    obs, _ = env.reset(seed=int(metadata["episode_seed"]), options=metadata["reset_kwargs"].get("options"))
    actual = np.asarray(obs.detach().cpu() if isinstance(obs, torch.Tensor) else obs)
    expected = recorded_obs[0]
    error = float(np.max(np.abs(actual.reshape(-1) - expected.reshape(-1))))
    if error > STATE_TOLERANCE:
        raise RuntimeError(f"initial observation mismatch {error} > {STATE_TOLERANCE}")
    return error


def _branch_row(task, obs_history, actions, control_index):
    local_head, inserted = _head_in_hole(task)
    return {
        "control_index": int(control_index),
        "head_in_hole_xyz_m": local_head.tolist(),
        "native_insertion": inserted,
        "native_grasp": native_grasped(task),
        "physical_state": _jsonable(_physical(task)),
        "observation_history": [_jsonable(item) for item in obs_history[-2:]],
        "prefix_action_sha256": hashlib.sha256(np.asarray(actions[:control_index]).tobytes()).hexdigest(),
    }


def _run_prefix(env, metadata, recorded_obs, actions, episode_cap):
    task = env.unwrapped
    initial_obs_error = _reset_and_verify(env, metadata, recorded_obs)
    obs_history = [np.asarray(recorded_obs[0]).copy()]
    max_prefix = min(MAX_PREFIX, episode_cap, len(actions))
    for index, action in enumerate(actions[:max_prefix], start=1):
        obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        obs_np = np.asarray(obs.detach().cpu() if isinstance(obs, torch.Tensor) else obs).reshape(-1)
        obs_history.append(obs_np.copy())
        local_head, inserted = _head_in_hole(task)
        grasped = native_grasped(task)
        eligible = grasped and not inserted and HEAD_X_LOW <= float(local_head[0]) < HEAD_X_HIGH
        if eligible:
            return {
                "status": "eligible",
                "controls": index,
                "initial_observation_max_abs_error": initial_obs_error,
                "branch": _branch_row(task, obs_history, actions, index),
                "obs_history": obs_history,
                "next_action": index,
            }
        if _is_success(info) or bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            return {
                "status": "no_eligible_trigger_before_terminal",
                "controls": index,
                "initial_observation_max_abs_error": initial_obs_error,
                "branch": _branch_row(task, obs_history, actions, index),
                "obs_history": obs_history,
                "next_action": index,
            }
    return {
        "status": "no_eligible_trigger_within_prefix_cap",
        "controls": max_prefix,
        "initial_observation_max_abs_error": initial_obs_error,
        "branch": _branch_row(task, obs_history, actions, max_prefix),
        "obs_history": obs_history,
        "next_action": max_prefix,
    }


def _same_branch(expected, actual):
    errors = {}
    for name in ("robot_qpos", "robot_qvel", "robot_root_pose", "tcp_pose", "peg_pose", "hole_pose"):
        before = np.asarray(expected["physical_state"][name], dtype=np.float64)
        after = np.asarray(actual["physical_state"][name], dtype=np.float64)
        errors[name] = float(np.max(np.abs(before - after)))
    expected_obs = np.asarray(expected["observation_history"], dtype=np.float64)
    actual_obs = np.asarray(actual["observation_history"], dtype=np.float64)
    if expected_obs.shape != actual_obs.shape:
        raise RuntimeError(f"matched branch observation history shape differs: {expected_obs.shape} vs {actual_obs.shape}")
    errors["two_observation_history"] = float(np.max(np.abs(expected_obs - actual_obs)))
    if expected["control_index"] != actual["control_index"]:
        raise RuntimeError("matched branch prefix control index differs")
    if expected["prefix_action_sha256"] != actual["prefix_action_sha256"]:
        raise RuntimeError("matched branch prefix action hash differs")
    max_error = max(errors.values(), default=0.0)
    if max_error > STATE_TOLERANCE:
        raise RuntimeError(f"matched prefix state error {max_error} exceeds tolerance: {errors}")
    return errors


def _run_replay_arm(index, metadata, recorded_obs, actions):
    env = _new_env()
    try:
        prefix = _run_prefix(env, metadata, recorded_obs, actions, 600)
        result = {"arm": "recorded_expert_suffix", "episode_id": index, **prefix}
        if prefix["status"] != "eligible":
            return result
        task = env.unwrapped
        controls = prefix["controls"]
        native_success = False
        max_suffix = min(MAX_SUFFIX, 600 - controls)
        suffix_count = min(max_suffix, len(actions) - prefix["next_action"])
        for action in actions[prefix["next_action"]:prefix["next_action"] + suffix_count]:
            obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            controls += 1
            if _is_success(info):
                native_success = True
                break
            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break
        result.update({"controls": controls, "suffix_controls": controls - prefix["controls"], "native_success": native_success})
        result["recorded_suffix_action_sha256"] = hashlib.sha256(
            np.asarray(actions[prefix["next_action"]:prefix["next_action"] + result["suffix_controls"]]).tobytes()
        ).hexdigest()
        return result
    finally:
        env.close()


def _run_live_arm(index, metadata, recorded_obs, actions, replay_branch):
    env = _new_env()
    try:
        prefix = _run_prefix(env, metadata, recorded_obs, actions, 600)
        result = {"arm": "live_official_expert_continuation", "episode_id": index, **prefix}
        if prefix["status"] != "eligible":
            return result
        state_errors = _same_branch(replay_branch, prefix["branch"])
        task = env.unwrapped
        live = run_live_expert_continuation(
            env,
            build_official_planner,
            run_official_geometry_continuation,
            max_controls=MAX_SUFFIX,
        )
        result.update({
            "controls": prefix["controls"] + live["controls"],
            "suffix_controls": live["controls"],
            "matched_branch_state_errors": state_errors,
            "live_expert": live,
            "native_success": live["status"] == "native_success",
        })
        return result
    finally:
        env.close()


def _source_closure():
    closure = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        closure[relative] = sha256(path)
    return closure


def _episode_registration(index):
    metadata, observations, actions, successes = _trajectory(index)
    return {
        "episode_id": index,
        "episode_seed": int(metadata["episode_seed"]),
        "source_dataset": metadata["source_dataset"],
        "target_yaw_rad": None,
        "trajectory_action_count": int(len(actions)),
        "trajectory_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
        "trajectory_obs_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
        "recorded_success": bool(np.asarray(successes).any()),
    }


def prepare(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    episode_rows = [_episode_registration(index) for index in EPISODE_IDS]
    for row in episode_rows:
        metadata = json.loads(META.read_text())["episodes"][row["episode_id"]]
        with h5py.File(H5, "r") as handle:
            yaw = float(handle[f"traj_{row['episode_id']}"]["env_states"]["peg_orientation_target"][0, 0])
        row["target_yaw_rad"] = yaw
        row["source_dataset"] = metadata["source_dataset"]
    source_hashes = _source_closure()
    package_lock = subprocess.run(
        ["conda", "list", "--explicit"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    (outdir / "runtime_conda_explicit.txt").write_text(package_lock)
    snapshots = outdir / "registered_sources"
    snapshots.mkdir()
    for relative in SOURCE_PATHS:
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    registration = {
        "format": "peg-orientation-takeover-preflight-v1",
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Validate recorded expert-suffix and live-state official-expert continuation on two fixed training episodes; no policy score or recovery claim.",
        "scope": "diagnostic only; frozen 500-demo candidate/data/checkpoints unchanged; no training, data collection, validation/test/fixed-retention scene access, Plug, or ManiSkill source edits",
        "hypotheses": {
            "H1": "An ideal expert can continue from a physically matched held-peg near-hole branch and reach official native success; falsified if matched live expert continuation fails its bounded plan/control run while recorded expert suffix succeeds.",
            "H2": "The fixed Constrain candidate's residual failures may reflect lack of recovery rather than an unreachable expert state; this two-episode oracle study does not test policy intervention or generalize recoverability.",
        },
        "episodes": episode_rows,
        "source_data": {
            "hdf5": {"path": str(H5.relative_to(ROOT)), "sha256": sha256(H5)},
            "json": {"path": str(META.relative_to(ROOT)), "sha256": sha256(META)},
        },
        "candidate_checkpoint_loaded": None,
        "candidate_checkpoint_reference_only": {
            "update": 25000,
            "sha256": "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df",
            "loaded": False,
        },
        "source_closure_sha256": source_hashes,
        "runtime": {
            "python": sys.version,
            "gymnasium": importlib.metadata.version("gymnasium"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "mani_skill_git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "native_backend": "physx_cpu",
            "conda_explicit_sha256": sha256(outdir / "runtime_conda_explicit.txt"),
        },
        "handoff": {
            "trigger": "first postaction state with native is_grasping true, official peg-head-in-hole x in [-0.10,-0.015) m, and official has_peg_inserted false",
            "mode_switch": "public task.agent.set_control_mode('pd_joint_pos') then public task.agent.controller.reset(); zero physical controls; compare robot qpos/qvel/root, TCP, peg, and hole poses at max absolute error <=1e-5; retain switch as controller confound",
            "planner": "stock PandaArmMotionPlanningSolver initialized after switch; gripper_state=CLOSED without close_gripper controls; official geometry is reproduced from current tcp-to-peg transform, goal/preinsert/refine/insert waypoints",
            "action": "public env.step control-mode action dict with explicitly batched [1,8] pd_joint_pos action",
        },
        "positive_controls": {
            "A": "same initial seed and recorded native pd_ee_delta_pose prefix to trigger, then replay recorded demo suffix; official success is positive control",
            "B": "independent env, same initial seed and exact action prefix; compare two obs histories plus qpos/qvel/root/TCP/peg/hole within 1e-5; switch controller and execute live official expert continuation",
            "no_eligible_trigger": "retain selected episode ID and mark no-eligible; do not replace based on outcome",
        },
        "budgets": {
            "selected_episodes": list(EPISODE_IDS),
            "prefix_controls_per_arm_max": MAX_PREFIX,
            "suffix_controls_per_arm_max": MAX_SUFFIX,
            "controls_per_arm_max": MAX_PREFIX + MAX_SUFFIX,
            "arms_max": 2 * len(EPISODE_IDS),
            "aggregate_native_controls_max": MAX_CONTROL_TOTAL,
            "aggregate_wall_seconds_max": MAX_WALL_SECONDS,
            "environment_time_limit": 600,
            "episode_control_cap": 500,
            "planning_timeout": "outer process timeout 900 s; TERM then KILL after 20 s",
        },
        "provenance_limits": [
            "Only two preselected training episodes; not a general recoverability or policy-effect estimate.",
            "Live expert mode-switches from pd_ee_delta_pose to pd_joint_pos; mode/controller target change is an explicit confound, even when physical states match.",
            "Recorded suffix arm is an expert positive control, not a recovery oracle.",
            "No hidden reset, teleport, direct qpos/actor pose assignment, learned-policy edit, checkpoint modification, or training-data modification.",
            "All episodes are from training data and all comparisons are nonblind diagnostics.",
        ],
    }
    registration["registration_sha256"] = canonical_hash(registration)
    (outdir / "preregistration.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outdir": str(outdir), "registration_sha256": registration["registration_sha256"], "episodes": episode_rows}, indent=2))


def run(outdir, attempt_id):
    outdir = Path(outdir)
    registration = json.loads((outdir / "preregistration.json").read_text())
    recorded_hash = registration.pop("registration_sha256")
    if canonical_hash(registration) != recorded_hash:
        raise RuntimeError("preregistration checksum mismatch")
    registration["registration_sha256"] = recorded_hash
    current_sources = _source_closure()
    if current_sources != registration["source_closure_sha256"]:
        raise RuntimeError("registered source closure changed after freeze")
    attempt = outdir / f"attempt_{attempt_id}"
    attempt.mkdir(parents=False, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    rows = []
    ledger = {"registration_sha256": recorded_hash, "started_at_utc": started_utc, "arms": []}
    ledger_path = attempt / "execution_ledger.json"
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    try:
        for index in EPISODE_IDS:
            metadata, observations, actions, _ = _trajectory(index)
            arm_start = time.monotonic()
            arm_ledger = {
                "episode_id": index,
                "arm": "recorded_expert_suffix",
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "started_monotonic_seconds": arm_start - started,
                "status": "running",
            }
            ledger["arms"].append(arm_ledger)
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
            replay = _run_replay_arm(index, metadata, observations, actions)
            replay["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            arm_ledger.update({
                "completed_at_utc": replay["completed_at_utc"],
                "completed_monotonic_seconds": time.monotonic() - started,
                "status": "completed",
                "native_controls": replay.get("controls", 0),
                "outcome": replay["status"],
            })
            rows.append(replay)
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
            if replay["status"] == "eligible":
                arm_start = time.monotonic()
                arm_ledger = {
                    "episode_id": index,
                    "arm": "live_official_expert_continuation",
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "started_monotonic_seconds": arm_start - started,
                    "controller_switch_mode": "pd_ee_delta_pose -> pd_joint_pos",
                    "status": "running",
                }
                ledger["arms"].append(arm_ledger)
                ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
                live = _run_live_arm(index, metadata, observations, actions, replay["branch"])
                live["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
                arm_ledger.update({
                    "completed_at_utc": live["completed_at_utc"],
                    "completed_monotonic_seconds": time.monotonic() - started,
                    "status": "completed",
                    "native_controls": live.get("controls", 0),
                    "outcome": live["status"],
                })
                rows.append(live)
                ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
            if sum(int(row.get("controls", 0)) for row in rows) > MAX_CONTROL_TOTAL:
                raise RuntimeError("aggregate native control cap exceeded")
        elapsed = time.monotonic() - started
        if elapsed > MAX_WALL_SECONDS:
            raise RuntimeError("aggregate stage wall-time cap exceeded")
        output = {
            "status": "completed",
            "started_at_utc": started_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "registration_sha256": recorded_hash,
            "source_closure_sha256_before": current_sources,
            "source_closure_sha256_after": _source_closure(),
            "native_controls_total": sum(int(row.get("controls", 0)) for row in rows),
            "rows": rows,
        }
        output = _jsonable(output)
        output["results_sha256"] = canonical_hash(output)
        (attempt / "results.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        ledger.update({
            "completed_at_utc": output["completed_at_utc"],
            "elapsed_seconds": elapsed,
            "total_native_controls": output["native_controls_total"],
            "status": "completed",
            "source_closure_sha256_after": output["source_closure_sha256_after"],
        })
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        print(json.dumps(output, indent=2, sort_keys=True))
    except BaseException as exc:
        output = {
            "status": "failed_or_interrupted",
            "started_at_utc": started_utc,
            "failure_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "registration_sha256": recorded_hash,
            "rows_completed_or_partial": rows,
            "native_controls_total": sum(int(row.get("controls", 0)) for row in rows),
            "source_closure_sha256_after": _source_closure(),
        }
        (attempt / "attempt_failure.json").write_text(json.dumps(_jsonable(output), indent=2, sort_keys=True) + "\n")
        ledger.update({
            "failure_at_utc": output["failure_at_utc"],
            "elapsed_seconds": output["elapsed_seconds"],
            "total_native_controls": output["native_controls_total"],
            "status": "failed_or_interrupted",
            "error_type": output["error_type"],
            "error": output["error"],
            "source_closure_sha256_after": output["source_closure_sha256_after"],
        })
        ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--outdir", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--outdir", required=True)
    run_parser.add_argument("--attempt-id", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.outdir)
    else:
        run(args.outdir, args.attempt_id)


if __name__ == "__main__":
    main()
