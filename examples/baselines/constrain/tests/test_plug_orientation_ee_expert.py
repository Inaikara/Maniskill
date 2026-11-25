import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.baselines.constrain import plug_orientation_ee_expert as expert


def test_goal_tcp_target_uses_actual_charger_to_tcp_transform_and_full_goal_pose():
    charger = np.array([0.3, -0.2, 0.1, 0.9238795, 0, 0, 0.3826834])
    tcp = np.array([0.4, -0.1, 0.2, 0.8660254, 0.2886751, 0.2886751, 0.2886751])
    goal = np.array([0.1, 0.2, 0.3, 0.5, 0.5, 0.5, 0.5])
    got = expert.charger_relative_tcp_target(charger, tcp, goal)
    assert got.shape == (7,) and np.isfinite(got).all()
    relative_position = Rotation.from_quat(charger[[4, 5, 6, 3]]).inv().apply(
        tcp[:3] - charger[:3]
    )
    expected_position = goal[:3] + Rotation.from_quat(goal[[4, 5, 6, 3]]).apply(
        relative_position
    )
    assert np.allclose(got[:3], expected_position, atol=1e-6)
    expected_rotation = Rotation.from_quat(goal[[4, 5, 6, 3]]) * Rotation.from_quat(
        charger[[4, 5, 6, 3]]
    ).inv() * Rotation.from_quat(tcp[[4, 5, 6, 3]])
    assert abs(np.dot(got[[4, 5, 6, 3]], expected_rotation.as_quat())) > 1 - 1e-6


def test_native_phase_limits_and_preregistration_are_fixed():
    assert expert.MAX_CONTROLS == 600
    assert expert.MAX_ALIGN_CONTROLS == 180
    assert expert.MAX_INSERT_CONTROLS == 160
    assert expert.PHASES == ("reach", "grasp", "lift", "align", "insert", "settle")


def test_planned_path_streams_every_waypoint_then_holds_planned_endpoint(monkeypatch):
    targets = [np.array([float(i), 0, 0, 1, 0, 0, 0]) for i in range(3)]
    calls = []
    # Simulate contact drift: the measured TCP is deliberately not the final plan target.
    measured = np.array([99., 0, 0, 1, 0, 0, 0])

    def physical_step(env, target, gripper, trace, phase, **kwargs):
        calls.append((np.asarray(target).copy(), gripper, kwargs))
        trace.append({"phase": phase, "measured_tcp_after": measured.copy()})
        return None, False, {}

    monkeypatch.setattr(expert, "_step", physical_step)
    trace = []
    final = expert._follow_planned_targets(object(), targets, 1., trace, "reach")
    assert np.array_equal(final, targets[-1])
    assert len(calls) == len(targets) + 10
    assert all(np.array_equal(calls[i][0], targets[i]) for i in range(len(targets)))
    assert all(np.array_equal(row[0], targets[-1]) for row in calls[-10:])
    assert all(row[2].get("target_hold") for row in calls[-10:])

    for _ in range(expert.GRIPPER_CLOSE_STEPS):
        physical_step(None, final, -1., trace, "grasp_close",
                      waypoint_index=len(targets) - 1, planned_target=final,
                      target_hold=True)
    assert len(calls) == len(targets) + 10 + expert.GRIPPER_CLOSE_STEPS
    assert all(np.array_equal(row[0], targets[-1]) and row[1] == -1.
               for row in calls[-expert.GRIPPER_CLOSE_STEPS:])


def test_planned_path_does_not_abort_on_intermediate_tracking_error(monkeypatch):
    targets = [np.array([float(i), 0, 0, 1, 0, 0, 0]) for i in range(2)]
    calls = []

    def physical_step(env, target, gripper, trace, phase, **kwargs):
        calls.append(np.asarray(target).copy())
        # No convergence is reported; intermediate errors are diagnostic only.
        return None, False, {}

    monkeypatch.setattr(expert, "_step", physical_step)
    result = expert._follow_planned_targets(object(), targets, 1., [], "reach")
    assert result is not None
    assert len(calls) == 12


def test_seven_dimensional_command_inverts_actual_native_controller_with_full_roll_yaw():
    import gymnasium as gym
    import torch
    from scipy.spatial.transform import Rotation
    from examples.baselines.constrain import plug_orientation
    from examples.baselines.constrain.peg_orientation_near_hole_perturb import (
        _native_config, _pose_value, _rotation, native_pose_action,
    )

    env = gym.make(plug_orientation.ENV_ID, obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                   reward_mode="sparse")
    try:
        env.reset(seed=2026113001, options={"target_yaw": 0.})
        task = env.unwrapped
        controller, geometry = _native_config(env)
        current = _pose_value(task.agent.tcp.pose)
        root_rotation = _rotation(_pose_value(task.agent.robot.pose).rotation)
        curr_rot = Rotation.from_quat(np.r_[current.rotation[1:], current.rotation[0]])
        desired_rot = Rotation.from_euler("xyz", [0.021, -0.017, 0.026]) * curr_rot
        q = desired_rot.as_quat()
        desired = type(current)(current.position + [0.004, -0.003, 0.002],
                                np.r_[q[3], q[:3]])
        encoded = native_pose_action(current, desired, geometry, root_rotation)
        assert encoded["action"].shape == (7,) and np.isfinite(encoded["action"]).all()
        raw = torch.as_tensor(encoded["action"][:6][None], dtype=torch.float32,
                              device=task.device)
        scaled = controller._clip_and_scale_action(raw)
        actual = controller.compute_target_pose(task.agent.tcp.pose, scaled)
        assert np.allclose(actual.p[0].detach().cpu().numpy(), desired.position, atol=1e-6, rtol=0)
        actual_q = actual.q[0].detach().cpu().numpy()
        assert abs(np.dot(actual_q, desired.rotation)) > 1 - 1e-6
    finally:
        env.close()


def test_joint_planner_is_dry_run_only_and_installs_experiment_mplib_adapter(monkeypatch):
    from types import SimpleNamespace
    from examples.baselines.constrain import peg_orientation_preflight
    from mani_skill.examples.motionplanning.panda import motionplanner

    events = []
    class Planner:
        def __init__(self, *args, **kwargs):
            events.append(("construct", kwargs))
        def move_to_pose_with_screw(self, pose, dry_run=False):
            events.append(("plan", dry_run))
            return {"status": "Success", "position": np.zeros((2, 7))}
        def close(self):
            events.append(("close",))

    monkeypatch.setattr(peg_orientation_preflight, "install_mplib_base_pose_compat",
                        lambda: events.append(("adapter",)))
    monkeypatch.setattr(expert, "_seed_planner", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(motionplanner, "PandaArmMotionPlanningSolver", Planner)
    monkeypatch.setattr(expert, "_fk_waypoint_targets", lambda task, path: ["fk1", "fk2"])
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        agent=SimpleNamespace(robot=SimpleNamespace(pose=object()))
    ))
    assert expert._plan_arm_path(env, object(), 42) == ["fk1", "fk2"]
    assert events[0] == ("adapter",)
    assert events[1] == ("seed", 42)
    assert events[2][0] == "construct"
    assert events[3] == ("plan", True)


def test_native_target_is_safely_bounded_before_each_physical_command():
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation

    current = sapien.Pose([0.01, -0.02, 0.03], [1, 0, 0, 0])
    desired_rot = Rotation.from_euler("xyz", [0.4, -0.2, 0.5]).as_quat()
    desired = sapien.Pose([0.03, -0.04, 0.01], np.r_[desired_rot[3], desired_rot[:3]])
    bounded = expert._bounded_target(current, desired)
    assert np.all(np.abs(bounded.position - np.asarray(current.p)) <= 0.005 + 1e-9)
    rot_delta = (Rotation.from_quat(np.r_[bounded.rotation[1:], bounded.rotation[0]])
                 * Rotation.from_quat(np.r_[current.q[1:], current.q[0]]).inv())
    assert rot_delta.magnitude() <= 0.03 + 1e-9


def test_step_native_target_calibration_with_nonidentity_tcp_roll_yaw_and_gripper():
    import gymnasium as gym
    import torch
    import sapien.core as sapien
    from examples.baselines.constrain import plug_orientation
    from examples.baselines.constrain.peg_orientation_near_hole_perturb import (
        _native_config, _pose_value, _rotation, native_pose_action,
    )

    env = gym.make(plug_orientation.ENV_ID, obs_mode="state",
                   control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
                   reward_mode="sparse")
    try:
        env.reset(seed=2026113001, options={"target_yaw": 0.})
        task = env.unwrapped
        controller, geometry = _native_config(env)
        root = _pose_value(task.agent.robot.pose)
        current = _pose_value(task.agent.tcp.pose)
        current_euler = Rotation.from_quat(np.r_[current.rotation[1:], current.rotation[0]]).as_euler("xyz")
        assert abs(current_euler[0]) > 0.01 and abs(current_euler[2]) > 0.01
        desired_rotation = Rotation.from_euler("xyz", [0.02, -0.01, 0.03]) * Rotation.from_quat(
            np.r_[current.rotation[1:], current.rotation[0]])
        q = desired_rotation.as_quat()
        desired = type(current)(current.position + [0.004, -0.003, 0.002],
                                np.r_[q[3], q[:3]])
        desired_sapien = sapien.Pose(desired.position, desired.rotation)
        bounded = expert._bounded_target(current, desired_sapien)
        encoded = native_pose_action(current, bounded, geometry, _rotation(root.rotation))
        assert not encoded["action_clipped"]
        scaled = controller._clip_and_scale_action(torch.as_tensor(
            encoded["action"][:6][None], dtype=torch.float32, device=task.device))
        actual_target = controller.compute_target_pose(task.agent.tcp.pose, scaled)
        assert np.allclose(actual_target.p[0].detach().cpu().numpy(), bounded.position,
                           atol=1e-6, rtol=0)
        assert abs(np.dot(actual_target.q[0].detach().cpu().numpy(), bounded.rotation)) > 1-1e-6
        # Execute the actual step method; do not patch env.step away from this API boundary.
        trace = []
        _, done, _ = expert._step(
            env, np.r_[desired.position, desired.rotation], -1., trace, "test")
        assert not done
        assert trace[0]["action"].shape == (7,)
        assert trace[0]["action"][-1] == -1.
    finally:
        env.close()


def _install_fake_run_case(monkeypatch, success):
    import sys
    import types
    import pathlib
    import h5py
    import torch

    actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.],
                        [-0.1, 0.0, 0.1, 0.0, -0.2, 0.1, 1.]], dtype=np.float32)
    envs = []

    class FakeEnv:
        def __init__(self):
            self.unwrapped = self
            self.state = np.zeros(2, dtype=np.float32)
            self.teleports = 0
            self.close_count = 0
            envs.append(self)

        def reset(self, seed=None, options=None):
            self.state[:] = 0
            return self.get_obs()

        def get_state_dict(self):
            return self.state.copy()

        def get_obs(self, unflattened=False):
            return torch.full((1, 46), float(self.state[0]), dtype=torch.float32)

        def step(self, action):
            self.state += [1., float(np.asarray(action)[0])]
            return self.get_obs(), 0., False, False, {"success": False}

        def evaluate(self):
            return {"success": np.array([False]), "obj_to_goal_dist": np.array([0.1]),
                    "obj_to_goal_angle": np.array([0.3])}

        def close(self):
            self.close_count += 1

    class FakeRecorder:
        def __init__(self, env, output_dir, trajectory_name, **kwargs):
            self.env, self.output_dir, self.trajectory_name = env, output_dir, trajectory_name
            self.states, self.observations, self.actions = [], [], []

        @property
        def unwrapped(self):
            return self.env

        def reset(self, **kwargs):
            result = self.env.reset(**kwargs)
            self._capture()
            return result

        def _capture(self):
            self.states.append(self.env.get_state_dict())
            self.observations.append(self.env.get_obs().cpu().numpy()[0].copy())

        def step(self, action):
            self.actions.append(np.asarray(action).copy())
            result = self.env.step(action)
            self._capture()
            return result

        def flush_trajectory(self, save=True):
            path = pathlib.Path(self.output_dir) / f"{self.trajectory_name}.h5"
            with h5py.File(path, "w") as h5:
                traj = h5.create_group("traj_0")
                traj.create_dataset("actions", data=np.asarray(self.actions, dtype=np.float32))
                traj.create_dataset("env_states", data=np.asarray(self.states, dtype=np.float32))
                traj.create_dataset("obs", data=np.asarray(self.observations, dtype=np.float32))

        def close(self):
            self.env.close()

    monkeypatch.setitem(sys.modules, "gymnasium", types.SimpleNamespace(
        make=lambda *args, **kwargs: FakeEnv()))
    monkeypatch.setitem(sys.modules, "mani_skill.utils.wrappers.record", types.SimpleNamespace(
        RecordEpisode=FakeRecorder))

    def fake_solve(recorder, seed, target_yaw, *, trace):
        recorder.reset(seed=seed, options={"target_yaw": target_yaw})
        trace.extend({"action": row.copy()} for row in actions)
        for row in actions:
            recorder.step(row.copy())
        return {"success": success, "reason": "native_success" if success else "grasp_tracking_failed",
                "controls": len(actions), "official_success": success,
                "official_position_error_m": 0.001, "official_rotation_error_rad": 0.01,
                "trace": trace}

    monkeypatch.setattr(expert, "solve", fake_solve)
    return actions, envs


def test_run_case_fake_hdf5_exact_action_replay_has_no_initial_teleport(tmp_path, monkeypatch):
    import json
    import pathlib
    import h5py

    actions, envs = _install_fake_run_case(monkeypatch, success=True)
    result = expert.run_case(123, tmp_path / "success")
    assert result["expert"]["success"]
    assert result["replay"]["actions_exact_match"]
    assert result["replay"]["initial_observation_match_max_abs"] == 0
    assert result["replay_status"] == "replayed"
    assert len(envs) == 2 and all(env.teleports == 0 for env in envs)
    with h5py.File(tmp_path / "success/replay/replay.h5", "r") as h5:
        assert np.array_equal(h5["traj_0/actions"][:], actions)
        assert len(h5["traj_0/env_states"]) == len(actions) + 1


def test_run_case_failed_expert_still_validates_flat_hdf5_observations_and_replays(tmp_path, monkeypatch):
    actions, envs = _install_fake_run_case(monkeypatch, success=False)
    result = expert.run_case(124, tmp_path / "failed")
    assert not result["expert"]["success"]
    assert result["replay_status"] == "diagnostic_expert_failed"
    assert result["replay"]["actions_exact_match"]
    assert result["replay"]["controls"] == len(actions)
    assert len(envs) == 2 and all(env.teleports == 0 for env in envs)


def test_run_case_deadline_stops_before_physical_control(tmp_path, monkeypatch):
    import time
    _, envs = _install_fake_run_case(monkeypatch, success=True)
    with pytest.raises(TimeoutError, match="before control"):
        expert.run_case(125, tmp_path / "expired", deadline=time.monotonic() - 1)
    assert len(envs) == 1 and np.array_equal(envs[0].state, np.zeros(2, dtype=np.float32))


def test_run_case_records_replay_not_started_for_actual_expert_planning_exception(tmp_path, monkeypatch):
    import json
    _install_fake_run_case(monkeypatch, success=False)

    def planning_failure(recorder, seed, target_yaw, *, trace):
        recorder.reset(seed=seed, options={"target_yaw": target_yaw})
        raise RuntimeError("MPLib screw planning failed")

    monkeypatch.setattr(expert, "solve", planning_failure)
    output = tmp_path / "planning-failure"
    with pytest.raises(RuntimeError, match="MPLib screw planning failed"):
        expert.run_case(126, output)
    payload = json.loads((output / "partial_failure.json").read_text())
    assert payload["expert_controls"] == 0
    assert payload["stage"] == "expert_failed_before_replay"
    assert payload["replay_status"] == "not_started"
    assert payload["replay_controls"] == 0


def _lift_tracking_env(object_moves):
    from types import SimpleNamespace
    import torch

    class Pose:
        def __init__(self, raw):
            self.raw_pose = torch.as_tensor(raw, dtype=torch.float32)[None]

    class Actor:
        def __init__(self):
            self.pose = Pose([0., 0., 0., 1., 0., 0., 0.])

        def set_pose(self, raw):
            self.pose = Pose(raw)

    charger = Actor()
    tcp = Actor()
    task = SimpleNamespace(charger=charger, agent=SimpleNamespace(tcp=tcp,
        is_grasping=lambda _charger: False))
    env = SimpleNamespace(unwrapped=task)
    return env, charger, object_moves


def test_lift_progress_does_not_abort_when_classifier_is_false_but_charger_moves(monkeypatch):
    env, charger, _ = _lift_tracking_env(True)

    def target_from_object(_charger, _tcp, desired):
        return np.asarray(desired).copy()

    def physical_step(_env, target, _gripper, trace, phase, **kwargs):
        if env.unwrapped.charger_moves:
            charger.set_pose(np.asarray(target).copy())
        trace.append({"phase": phase, "charger_pose": expert._actor_pose(charger)})
        return None, False, {}

    env.unwrapped.charger_moves = True
    monkeypatch.setattr(expert, "charger_relative_tcp_target", target_from_object)
    monkeypatch.setattr(expert, "_step", physical_step)
    desired = np.array([0., 0., 0.03, 1., 0., 0., 0.])
    assert expert._charger_target_loop(env, desired, -1., [], "lift", 80)


def test_lift_tracking_still_fails_when_tcp_target_is_followed_but_charger_does_not_move(monkeypatch):
    env, charger, _ = _lift_tracking_env(False)

    def target_from_object(_charger, _tcp, desired):
        return np.asarray(desired).copy()

    def physical_step(_env, target, _gripper, trace, phase, **kwargs):
        # The commanded TCP follows perfectly; only the physical object is static.
        trace.append({"phase": phase, "charger_pose": expert._actor_pose(charger),
                      "tcp_pose": np.asarray(target).copy()})
        return None, False, {}

    monkeypatch.setattr(expert, "charger_relative_tcp_target", target_from_object)
    monkeypatch.setattr(expert, "_step", physical_step)
    desired = np.array([0., 0., 0.03, 1., 0., 0., 0.])
    assert not expert._charger_target_loop(env, desired, -1., [], "lift", 80)


def test_step_exception_preserves_attempted_native_control_in_trace(monkeypatch):
    from types import SimpleNamespace
    from scipy.spatial.transform import Rotation

    current = expert.PoseValue(np.zeros(3), np.array([1., 0., 0., 0.]))
    geometry = SimpleNamespace(frame="root", use_delta=True, pos_lower=np.full(3, -1.),
                               pos_upper=np.full(3, 1.), rot_lower=np.full(3, -1.),
                               rot_upper=np.full(3, 1.))
    task = SimpleNamespace(agent=SimpleNamespace(tcp=SimpleNamespace(pose=None),
                                                 robot=SimpleNamespace(pose=None)),
                          charger=object())
    env = SimpleNamespace(unwrapped=task, step=lambda _action: (_ for _ in ()).throw(
        RuntimeError("injected step failure")))
    monkeypatch.setattr(expert, "_native_config", lambda _env: (None, geometry))
    monkeypatch.setattr(expert, "_pose_value", lambda _pose: current)
    monkeypatch.setattr(expert, "_tcp", lambda _task: np.zeros(7))
    monkeypatch.setattr(expert, "_rotation", lambda _rotation: Rotation.identity())
    monkeypatch.setattr(expert, "native_pose_action", lambda *args: {
        "action_clipped": False, "action": np.zeros(7)})
    trace = []
    try:
        expert._step(env, np.array([0., 0., 0., 1., 0., 0., 0.]), -1., trace, "lift")
    except RuntimeError as exc:
        assert str(exc) == "injected step failure"
    else:
        raise AssertionError("injected native step must fail")
    assert len(trace) == 1
    assert trace[0]["control_attempted"] and not trace[0]["control_completed"]
    assert trace[0]["step_exception"] == "RuntimeError: injected step failure"
