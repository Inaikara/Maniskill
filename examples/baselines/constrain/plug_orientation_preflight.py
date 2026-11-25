"""Bounded PlugCharger expert-preflight runner; native execution is opt-in."""

import argparse
import json
import random
import shutil
import time
import traceback
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from mani_skill.utils.wrappers.record import RecordEpisode

from examples.baselines.constrain import plug_orientation

PLANNED_SCENES = 10
MAX_ATTEMPTS = 20
MAX_CONTROLS_PER_ATTEMPT = 600
MAX_STAGE_SECONDS = 3600
MAX_WHOLE_SECONDS = 9000
MAX_STAGE_CONTROLS = 12000
FIXED_SEED_START = 2026113000
RANDOM_SEED_START = FIXED_SEED_START
RANDOM_TARGET_YAWS = tuple(np.linspace(-np.pi / 8, np.pi / 8, 10).tolist())


def scene_plan(distribution):
    if distribution not in ("fixed", "random"):
        raise ValueError(distribution)
    return [
        {"seed": FIXED_SEED_START + i, "target_yaw": 0.0 if distribution == "fixed"
         else RANDOM_TARGET_YAWS[i]}
        for i in range(PLANNED_SCENES)
    ]


class _BoundedRecorder:
    """Inject preplanned yaw and observe true resets/controls around official solve."""

    def __init__(self, recorder, target_yaw, control_cap, on_reset,
                 stage_controls=None, stage_control_cap=MAX_STAGE_CONTROLS,
                 planner_seed=None, mplib_seed=None):
        self._recorder = recorder
        self._target_yaw = float(target_yaw)
        self._control_cap = int(control_cap)
        self._on_reset = on_reset
        self._stage_controls = stage_controls
        self._stage_control_cap = stage_control_cap
        self.controls = 0
        self._planner_seed, self._mplib_seed = planner_seed, mplib_seed

    def __getattr__(self, name):
        return getattr(self._recorder, name)

    def reset(self, *, seed=None, options=None, **kwargs):
        options = dict(options or {})
        options["target_yaw"] = self._target_yaw
        result = self._recorder.reset(seed=seed, options=options, **kwargs)
        if self._planner_seed is not None:
            random.seed(self._planner_seed)
            np.random.seed(self._planner_seed)
            torch.manual_seed(self._planner_seed)
            if self._mplib_seed is not None:
                self._mplib_seed(self._planner_seed)
        task = self._recorder.unwrapped
        state = task.get_state_dict()
        self._on_reset(task, state)
        return result

    def step(self, action):
        if self.controls >= self._control_cap:
            raise RuntimeError(
                f"official expert exceeded {self._control_cap}-control attempt cap"
            )
        if self._stage_controls is not None and self._stage_controls["controls"] >= self._stage_control_cap:
            raise RuntimeError(f"expert exceeded {self._stage_control_cap}-control stage cap")
        self.controls += 1
        if self._stage_controls is not None:
            self._stage_controls["controls"] += 1
        return self._recorder.step(action)


class _ControlLimitedEnv(gym.Wrapper):
    def __init__(self, env, control_cap, counter=None, stage_control_cap=MAX_STAGE_CONTROLS,
                 initial_capture=None):
        super().__init__(env)
        self.control_cap = int(control_cap)
        self.controls = 0
        self.counter = counter if counter is not None else {"controls": 0}
        self.stage_control_cap = stage_control_cap
        self.initial_capture = initial_capture

    def step(self, action):
        if self.controls >= self.control_cap:
            raise RuntimeError(f"replay exceeded {self.control_cap}-control episode cap")
        if self.counter["controls"] >= self.stage_control_cap:
            raise RuntimeError(f"replay exceeded {self.stage_control_cap}-control stage cap")
        if self.controls == 0 and self.initial_capture is not None:
            task = self.unwrapped
            control_mode = getattr(task, "control_mode", None)
            if control_mode == "pd_ee_delta_pose":
                self.initial_capture["control_mode"] = control_mode
                self.initial_capture["state"] = _copy_tree(task.get_state_dict())
                self.initial_capture["observation"] = _copy_tree(task.get_obs(unflattened=True))
        self.controls += 1
        self.counter["controls"] += 1
        return self.env.step(action)


def collect(distribution, output_dir, *, solver=None, wall_seconds=3600,
            max_attempts=MAX_ATTEMPTS, max_stage_controls=MAX_STAGE_CONTROLS,
            deadline=None, expert_grasp_tilt_deg=None, planned_scenes=None):
    """Collect attempted episodes with the stock official PlugCharger planner."""
    injected_solver = solver is not None
    if solver is None:
        from examples.baselines.constrain.peg_orientation_preflight import install_mplib_base_pose_compat
        install_mplib_base_pose_compat()
        from mani_skill.examples.motionplanning.panda.solutions.plug_charger import solve
        solver = solve
        if expert_grasp_tilt_deg is not None:
            from examples.baselines.constrain.plug_orientation_expert_candidate import solve as candidate_solve
            solver = lambda env, **kwargs: candidate_solve(
                env, grasp_tilt_deg=expert_grasp_tilt_deg, **kwargs
            )
        import mplib
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    env = gym.make(
        plug_orientation.RANDOM_ENV_ID if distribution == "random" else plug_orientation.ENV_ID,
        obs_mode="state", control_mode="pd_joint_pos", sim_backend="physx_cpu",
        reward_mode="sparse", max_episode_steps=200,
    )
    recorder = RecordEpisode(
        env, output_dir=str(output), trajectory_name="expert", save_trajectory=True,
        save_video=False, save_on_reset=False, record_env_state=True,
        source_type="motionplanning",
        source_desc="Official PlugCharger planner; native pd_joint_pos",
    )
    rows = []
    started = time.monotonic()
    stage_controls = {"controls": 0}
    plan = list(planned_scenes) if planned_scenes is not None else scene_plan(distribution)
    try:
        for i, scene in enumerate(plan[:max_attempts]):
            if (time.monotonic() - started >= min(wall_seconds, MAX_STAGE_SECONDS)
                    or (deadline is not None and time.monotonic() >= deadline)):
                break
            captures = []

            def capture(task, state):
                captures.append({
                    "target_yaw": float(task.target_yaw),
                    "initial_state": _jsonable(state),
                    "goal_position": _jsonable(task.goal_pose.p),
                    "charger_pose": _jsonable(task.charger.pose.raw_pose),
                    "receptacle_pose": _jsonable(task.receptacle.pose.raw_pose),
                })

            bounded = _BoundedRecorder(recorder, scene["target_yaw"],
                MAX_CONTROLS_PER_ATTEMPT, capture, stage_controls, max_stage_controls,
                planner_seed=int(scene["seed"]) + 1,
                mplib_seed=None if injected_solver else mplib.set_global_seed)
            planner_seed = int(scene["seed"]) + 1
            random.seed(planner_seed)
            np.random.seed(planner_seed)
            torch.manual_seed(planner_seed)
            if not injected_solver:
                mplib.set_global_seed(planner_seed)
            try:
                # The stock solve performs reset itself. Its reset is intercepted,
                # target selection injected, and exact post-reset state captured.
                result = solver(bounded, seed=scene["seed"], debug=False, vis=False)
                info = result[-1] if result != -1 else {}
                success = bool(torch.as_tensor(info.get("success", False)).item())
                reason = "expert_success" if success else "official_success_false"
                exception_traceback = None
            except Exception as exc:
                success = False
                reason = f"{type(exc).__name__}: {exc}"
                exception_traceback = traceback.format_exc()
            steps = bounded.controls
            saved_episode_id = None
            if steps:
                recorder.flush_trajectory(save=True)
                saved_episode_id = getattr(recorder, "_episode_id", None)
            rows.append({
                "planned_index": i, **scene, "expert_success": success,
                "planner_seed": planner_seed,
                "saved_episode_id": saved_episode_id,
                "reason": reason, "controls": steps,
                "exception_traceback": exception_traceback,
                "reset_capture": captures[0] if captures else None,
            })
            (output / "attempts.jsonl").open("a").write(
                json.dumps(rows[-1], sort_keys=True, allow_nan=False) + "\n"
            )
    finally:
        recorder.close()
    _annotate_source_episodes(output / "expert.json", rows)
    report = {
        "distribution": distribution, "planned_denominator": len(plan),
        "planned_scenes": plan, "attempt_limit": min(max_attempts, MAX_ATTEMPTS),
        "attempts": rows, "attempt_count": len(rows),
        "expert_successes": sum(row["expert_success"] for row in rows),
        "stage_controls": stage_controls["controls"],
        "max_stage_controls": max_stage_controls,
        "complete_plan": len(rows) == len(plan),
        "elapsed_seconds": time.monotonic() - started,
        "max_controls_per_attempt": MAX_CONTROLS_PER_ATTEMPT,
        "stage_deadline_seconds": min(wall_seconds, MAX_STAGE_SECONDS),
        "whole_deadline_seconds": MAX_WHOLE_SECONDS,
        "native_replay": "not run by collection command",
        "expert_method": "injected_test_solver" if injected_solver
        else ("official_plug_charger_copy_parameterized_grasp_tilt" if expert_grasp_tilt_deg is not None
              else "official_plug_charger_solver"),
        "expert_grasp_tilt_deg": expert_grasp_tilt_deg,
        "planner_seed_policy": "scene_seed_plus_one_before_reset_and_planner_creation",
    }
    (output / "collection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def replay_successes(source_h5, output_dir, *, wall_seconds=MAX_STAGE_SECONDS,
                     max_stage_controls=MAX_STAGE_CONTROLS, deadline=None):
    """Convert each successful joint-space episode in an isolated CPU replay."""
    from mani_skill.trajectory.replay_trajectory import Args, main as replay_main
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common

    source_h5 = Path(source_h5)
    source_meta_path = source_h5.with_suffix(".json")
    source_meta = json.loads(source_meta_path.read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    stage_controls = {"controls": 0}
    selected_rows = _select_expert_successes(
        source_meta["episodes"], _read_attempts(source_h5.parent / "attempts.jsonl"))
    for row in selected_rows:
        if (time.monotonic() - started >= min(wall_seconds, MAX_STAGE_SECONDS)
                or (deadline is not None and time.monotonic() >= deadline)):
            break
        episode_id = int(row["episode_id"])
        source_key = f"traj_{episode_id}"
        episode_dir = output_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir()
        isolated_h5 = episode_dir / "expert.h5"
        isolated_meta = json.loads(json.dumps(source_meta))
        isolated_row = dict(row, episode_id=0)
        isolated_meta["episodes"] = [isolated_row]
        with h5py.File(source_h5, "r") as source, h5py.File(isolated_h5, "w") as target:
            source.copy(source_key, target, name="traj_0")
            initial_state = trajectory_utils.dict_to_list_of_dicts(
                source[source_key]["env_states"]
            )[0]
        isolated_h5.with_suffix(".json").write_text(
            json.dumps(isolated_meta, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        orientation = np.asarray(initial_state.get(plug_orientation.STATE_KEY, []))
        if orientation.size != 4 or not np.isfinite(orientation).all():
            raise ValueError(f"source episode {episode_id} lacks finite initial orientation state")
        outcome = {"source_episode_id": episode_id,
                   "episode_seed": int(row["episode_seed"]),
                   "target_yaw": float(orientation.reshape(-1)[0])}
        try:
            original_make = gym.make

            initial_capture = {}
            def guarded_make(*args, **kwargs):
                return _ControlLimitedEnv(
                    original_make(*args, **kwargs), MAX_CONTROLS_PER_ATTEMPT,
                    stage_controls, max_stage_controls, initial_capture,
                )

            gym.make = guarded_make
            try:
                replay_main(Args(
                    traj_path=str(isolated_h5), sim_backend="physx_cpu", obs_mode="state",
                    target_control_mode="pd_ee_delta_pose", save_traj=True, count=1,
                    max_retry=0, allow_failure=True, use_first_env_state=True,
                ))
            finally:
                gym.make = original_make
            # ManiSkill's public main returns None. _main derives this exact
            # name from input stem, obs mode, control mode, and backend.
            converted_h5 = isolated_h5.with_name(
                "expert.state.pd_ee_delta_pose.physx_cpu.h5"
            )
            if not converted_h5.is_file():
                raise FileNotFoundError(f"public replay did not produce {converted_h5}")
            raw_h5 = episode_dir / "expert.raw_recording.state.pd_ee_delta_pose.physx_cpu.h5"
            shutil.copy2(converted_h5, raw_h5)
            with h5py.File(converted_h5, "r") as converted:
                actions = np.asarray(converted["traj_0"]["actions"])
                states = converted["traj_0"]["env_states"]
                state_rows = trajectory_utils.dict_to_list_of_dicts(states)
                if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                    raise ValueError(f"converted actions must be finite T×7, got {actions.shape}")
                if len(state_rows) != len(actions) + 1:
                    raise ValueError("converted state/action alignment must be T+1/T")
                if (initial_capture.get("control_mode") != "pd_ee_delta_pose"
                        or "state" not in initial_capture
                        or "observation" not in initial_capture):
                    raise ValueError("public replay did not capture actual destination initial state")
                source_initial = _normalise_batched_tree(initial_capture["state"])
                if not _trees_close(_normalise_batched_tree(initial_state), source_initial, atol=1e-5):
                    raise ValueError("actual replay initial state differs from canonical source state 0")
                flat_initial = np.asarray(common.flatten_state_dict(
                    initial_capture["observation"], use_torch=True).detach().cpu())
                if not np.isfinite(flat_initial).all():
                    raise ValueError("actual initial observation contains non-finite values")
            alignment = _align_initial_record(converted_h5, source_initial, flat_initial,
                                              raw_sha256=_sha256(raw_h5))
            alignment["corrected_sha256"] = _sha256(converted_h5)
            converted_h5.with_suffix(".alignment.json").write_text(
                json.dumps(alignment, indent=2, sort_keys=True, allow_nan=False) + "\n")
            with h5py.File(converted_h5, "r") as aligned:
                state_rows = trajectory_utils.dict_to_list_of_dicts(aligned["traj_0"]["env_states"])
                actions = np.asarray(aligned["traj_0"]["actions"])
                if "obs" not in aligned["traj_0"]:
                    raise ValueError("converted trajectory is missing recorded observations")
                obs_audit = _audit_observation_group(
                    aligned["traj_0"]["obs"], len(actions) + 1
                )
                for state_index, state in enumerate(state_rows):
                    if not _finite_tree(state):
                        raise ValueError(f"converted state {state_index} contains non-finite values")
                if not np.allclose(np.asarray(state_rows[0][plug_orientation.STATE_KEY]).reshape(-1),
                                   orientation.reshape(-1), atol=1e-6, rtol=0):
                    raise ValueError("converted initial orientation state differs from source")
                task_env = gym.make(
                    isolated_meta["env_info"]["env_id"], obs_mode="state",
                    control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                    reward_mode="sparse",
                )
                try:
                    task_env.reset(seed=int(row["episode_seed"]))
                    task_env.unwrapped.set_state_dict(common.batch(state_rows[0]))
                    # BaseEnv defaults to flattened state tensors even in state mode.
                    # Leak auditing requires the unflattened public observation tree.
                    initial_obs = task_env.unwrapped.get_obs(unflattened=True)
                    _assert_no_orientation_leaks(initial_obs)
                    initial_flat = common.flatten_state_dict(initial_obs, use_torch=True)
                    if not np.isfinite(initial_flat).all():
                        raise ValueError("initial base observation contains non-finite values")
                    outcome["base_observation_dim"] = int(initial_flat.shape[-1])
                    task_env.unwrapped.set_state_dict(common.batch(state_rows[-1]))
                    official = task_env.unwrapped.evaluate()
                    official_success = bool(torch.as_tensor(official["success"]).item())
                    outcome["official_position_error_m"] = float(
                        torch.as_tensor(official["obj_to_goal_dist"]).item()
                    )
                    outcome["official_rotation_error_rad"] = float(
                        torch.as_tensor(official["obj_to_goal_angle"]).item()
                    )
                finally:
                    task_env.close()
                outcome.update({
                    "replay_success": official_success,
                    "action_shape": list(actions.shape),
                    "action_count": int(len(actions)),
                    "finite_actions": True,
                    "state_count": len(state_rows),
                    "observation_leaf_count": obs_audit["leaf_count"],
                    "observation_flat_dimensions": obs_audit["flat_dimensions"],
                    "reason": "official_full_pose_success" if official_success
                    else "official_full_pose_success_false",
                })
        except Exception as exc:
            outcome.update({"replay_success": False,
                            "reason": f"{type(exc).__name__}: {exc}"})
        rows.append(outcome)
        (output_dir / "replay_attempts.jsonl").open("a").write(
            json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n"
        )
    report = {
        "source_h5": str(source_h5), "source_h5_sha256": _sha256(source_h5),
        "attempt_count": len(rows), "replay_successes": sum(r["replay_success"] for r in rows),
        "attempts": rows, "sim_backend": "physx_cpu", "target_control_mode": "pd_ee_delta_pose",
        "isolation": "one source episode per official replay invocation",
        "elapsed_seconds": time.monotonic() - started,
        "stage_controls": stage_controls["controls"],
        "max_stage_controls": max_stage_controls,
        "stage_deadline_seconds": min(wall_seconds, MAX_STAGE_SECONDS),
        "expected_source_success_count": len(selected_rows),
        "retained_source_failure_count": len(source_meta["episodes"]) - len(selected_rows),
        "source_episode_denominator": len(source_meta["episodes"]),
        "complete": len(rows) == len(selected_rows),
    }
    (output_dir / "replay_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _copy_tree(value):
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def _normalise_batched_tree(value):
    if isinstance(value, dict):
        return {key: _normalise_batched_tree(item) for key, item in value.items()}
    array = np.asarray(value)
    return array[0] if array.ndim and array.shape[0] == 1 else array


def _trees_close(left, right, atol):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _trees_close(left[key], right[key], atol) for key in left
        )
    try:
        return np.allclose(np.asarray(left), np.asarray(right), atol=atol, rtol=0)
    except (TypeError, ValueError):
        return left == right


def _read_attempts(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _select_expert_successes(episodes, attempts):
    by_id = {int(item["saved_episode_id"]): item for item in attempts
             if item.get("saved_episode_id") is not None}
    selected = []
    for row in episodes:
        episode_id = int(row["episode_id"])
        declarations = []
        attempt = by_id.get(episode_id)
        if attempt is not None and "expert_success" in attempt:
            declarations.append(attempt["expert_success"])
        for key in ("expert_success", "final_success", "success"):
            if key in row:
                declarations.append(row[key])
        if any(not isinstance(value, bool) for value in declarations):
            raise ValueError(f"episode {episode_id} success provenance must be an explicit bool")
        if not declarations:
            raise ValueError(f"episode {episode_id} is missing success provenance")
        if any(value != declarations[0] for value in declarations[1:]):
            raise ValueError(f"episode {episode_id} has conflicting success provenance")
        if declarations[0]:
            selected.append(row)
    return selected


def _annotate_source_episodes(path, attempts):
    if not path.is_file():
        return
    by_id = {int(row["saved_episode_id"]): row for row in attempts
             if row.get("saved_episode_id") is not None}
    metadata = json.loads(path.read_text())
    for episode in metadata.get("episodes", []):
        attempt = by_id.get(int(episode["episode_id"]))
        if attempt is not None:
            episode.update(expert_success=bool(attempt["expert_success"]),
                           planner_seed=int(attempt["planner_seed"]),
                           target_yaw=float(attempt["target_yaw"]))
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _align_initial_record(path, actual_state, actual_observation, *, raw_sha256):
    with h5py.File(path, "r+") as h5:
        trajectory = h5["traj_0"]
        action_hash = _sha256_array(np.asarray(trajectory["actions"]))
        state_group = trajectory["env_states"]
        updates = []
        def replace(node, value):
            if isinstance(node, h5py.Group):
                if set(node.keys()) != set(value):
                    raise ValueError("actual initial state structure differs from recorded state")
                for key in node:
                    replace(node[key], value[key])
                return
            expected = np.asarray(value)
            if node.shape[0] < 1 or int(np.prod(node.shape[1:])) != expected.size:
                raise ValueError(f"actual initial state shape mismatch at {node.name}")
            if not np.isfinite(expected).all():
                raise ValueError(f"actual initial state is non-finite at {node.name}")
            updates.append((node, expected.reshape(node.shape[1:])))
        replace(state_group, actual_state)
        obs = trajectory["obs"]
        if isinstance(obs, h5py.Group):
            leaves = []
            obs.visititems(lambda _, node: leaves.append(node) if isinstance(node, h5py.Dataset) else None)
            if len(leaves) != 1:
                raise ValueError("cannot align structured state observation")
            obs = leaves[0]
        if obs.ndim < 2 or obs.shape[0] < 1 or obs[0].size != actual_observation.size:
            raise ValueError("actual initial observation shape differs from recorded observation")
        if not np.isfinite(actual_observation).all():
            raise ValueError("actual initial observation is non-finite")
        # Validate every leaf before the first mutation.
        for node, value in updates:
            node[0] = value
        obs[0] = actual_observation.reshape(obs.shape[1:])
        if action_hash != _sha256_array(np.asarray(trajectory["actions"])):
            raise ValueError("alignment unexpectedly changed converted actions")
        capture_sha256 = _tree_sha256({"state": actual_state, "observation": actual_observation})
        trajectory.attrs["initial_record_alignment"] = json.dumps({
            "raw_recording_sha256": raw_sha256,
            "actions_sha256": action_hash,
            "actual_initial_capture_sha256": capture_sha256,
            "method": "captured_actual_destination_state_before_first_control",
        }, sort_keys=True)
    return {"raw_recording_sha256": raw_sha256, "actions_sha256": action_hash,
            "actual_initial_capture_sha256": capture_sha256,
            "method": "captured_actual_destination_state_before_first_control"}


def _sha256_array(value):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _tree_sha256(value):
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    import hashlib
    return hashlib.sha256(payload).hexdigest()


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (np.ndarray, np.number, float, int)):
        return bool(np.isfinite(value).all())
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return True


def _assert_no_orientation_leaks(value):
    forbidden = {"receptacle_pose", "goal_pose", "goal_quaternion", "target_yaw"}
    if isinstance(value, dict):
        leaked = forbidden.intersection(value)
        if leaked:
            raise ValueError(f"orientation leaked into base observation: {sorted(leaked)}")
        for child in value.values():
            _assert_no_orientation_leaks(child)


def _audit_observation_group(group, expected_rows):
    leaves = []
    def visit(node):
        if isinstance(node, h5py.Dataset):
            values = np.asarray(node)
            if values.ndim == 0 or values.shape[0] != expected_rows:
                raise ValueError(f"observation {node.name} must have {expected_rows} rows")
            if not np.isfinite(values).all():
                raise ValueError(f"observation {node.name} contains non-finite values")
            leaves.append(int(np.prod(values.shape[1:])))
        else:
            for key in node:
                if key in {"receptacle_pose", "goal_pose", "goal_quaternion", "target_yaw"}:
                    raise ValueError(f"orientation leaked into recorded observation: {key}")
                visit(node[key])
    visit(group)
    if not leaves:
        raise ValueError("recorded observation is empty")
    return {"leaf_count": len(leaves), "flat_dimensions": sum(leaves)}


def _sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", choices=("fixed", "random"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wall-seconds", type=int, default=MAX_STAGE_SECONDS)
    parser.add_argument("--max-stage-controls", type=int, default=MAX_STAGE_CONTROLS)
    parser.add_argument("--replay-source", help="replay an expert HDF5 instead of collecting")
    parser.add_argument("--replay-output-dir")
    parser.add_argument("--expert-grasp-tilt-deg", type=float,
                        help="explicitly select the experiment-side official-solver copy")
    args = parser.parse_args()
    if args.replay_source:
        if not args.replay_output_dir:
            parser.error("--replay-source requires --replay-output-dir")
        replay_successes(args.replay_source, args.replay_output_dir,
                         wall_seconds=args.wall_seconds,
                         max_stage_controls=args.max_stage_controls)
    else:
        collect(args.distribution, args.output_dir, wall_seconds=args.wall_seconds,
                max_stage_controls=args.max_stage_controls,
                expert_grasp_tilt_deg=args.expert_grasp_tilt_deg)


if __name__ == "__main__":
    main()
