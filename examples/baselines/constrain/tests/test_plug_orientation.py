"""Plug orientation experiment contracts and preflight orchestration."""

import numpy as np
import torch
import gymnasium as gym
import json
from pathlib import Path

from examples.baselines.constrain import plug_orientation
from examples.baselines.constrain import plug_orientation_preflight
from examples.baselines.constrain import plug_orientation_pipeline


def test_goal_yaw_preserves_native_goal_position_and_uses_full_rotation_contract():
    goal_position = np.array([0.2, -0.1, 0.3])
    goal = plug_orientation.goal_pose_components(goal_position, np.pi + 0.1)
    assert np.allclose(goal[0], goal_position)
    assert np.isclose(plug_orientation.world_x_yaw(goal[1]), -np.pi + 0.1)


def test_hint_is_current_charger_plus_x_minus_target_goal_plus_x():
    current = np.pi / 4
    target = 0.0
    hint = plug_orientation.build_orientation_hint(current, target)
    assert np.allclose(hint, [np.sqrt(0.5), np.sqrt(0.5)])
    assert np.allclose(plug_orientation.build_orientation_hint(current, target), hint)


def test_target_options_do_not_move_native_goal_position():
    assert plug_orientation.choose_target_yaw({}, random_target=False, rng=None) == 0
    assert plug_orientation.choose_target_yaw(
        {"target_yaw": 0.2}, random_target=True, rng=None
    ) == 0.2


def test_state_geometry_is_constant_and_carries_no_world_orientation():
    geometry = plug_orientation.local_geometry_observation("cpu")
    assert torch.allclose(geometry["charger_base_half_size"], torch.tensor([[0.02, 0.015, 0.012]]))
    assert torch.allclose(geometry["charger_peg_half_size"], torch.tensor([[0.008, 0.00075, 0.0032]]))
    assert torch.allclose(geometry["receptacle_half_size"], torch.tensor([[0.01, 0.05, 0.05]]))


def test_observation_contract_removes_both_world_orientation_leaks():
    obs = {
        "agent": {"qpos": np.zeros(9)},
        "tcp_pose": np.zeros(7),
        "charger_pose": np.zeros(7),
        "receptacle_pose": np.zeros(7),
        "goal_pose": np.zeros(7),
    }
    result = plug_orientation.filter_orientation_observation(obs)
    assert set(result) == {"agent", "tcp_pose", "charger_pose"}


def test_preflight_plan_pairs_scenes_and_stratifies_random_yaw_extremes():
    fixed = plug_orientation_preflight.scene_plan("fixed")
    random = plug_orientation_preflight.scene_plan("random")
    assert len(fixed) == len(random) == 10
    assert [row["seed"] for row in fixed] == [row["seed"] for row in random]
    assert all(row["target_yaw"] == 0 for row in fixed)
    assert np.isclose(random[0]["target_yaw"], -np.pi / 8)
    assert np.isclose(random[-1]["target_yaw"], np.pi / 8)
    assert len({row["target_yaw"] for row in random}) == 10


def test_bounded_recorder_injects_target_and_captures_true_reset():
    class Task:
        def get_state_dict(self):
            return {"marker": torch.tensor([[3]])}

        target_yaw = 0.4

    class Recorder:
        unwrapped = Task()

        def reset(self, **kwargs):
            self.options = kwargs["options"]
            return "reset"

        def step(self, action):
            return "step"

    captures = []
    recorder = Recorder()
    bounded = plug_orientation_preflight._BoundedRecorder(
        recorder, 0.2, 1, lambda task, state: captures.append((task, state))
    )
    assert bounded.reset(seed=13) == "reset"
    assert recorder.options == {"target_yaw": 0.2}
    assert captures[0][1]["marker"].item() == 3
    bounded.step(None)
    with np.testing.assert_raises_regex(RuntimeError, "1-control"):
        bounded.step(None)


def test_replay_environment_enforces_control_cap_before_physics_step():
    class FakeEnv(gym.Env):
        action_space = gym.spaces.Box(-1, 1, shape=(1,))
        observation_space = gym.spaces.Box(-1, 1, shape=(1,))

        def __init__(self):
            self.calls = 0

        def step(self, action):
            self.calls += 1
            return np.zeros(1), 0.0, False, False, {}

        def reset(self, **kwargs):
            return np.zeros(1), {}

    env = FakeEnv()
    bounded = plug_orientation_preflight._ControlLimitedEnv(env, 1)
    bounded.step(np.zeros(1))
    with np.testing.assert_raises_regex(RuntimeError, "1-control"):
        bounded.step(np.zeros(1))
    assert env.calls == 1


def test_task_reset_rotates_actual_receptacle_and_goal_about_same_center(monkeypatch):
    from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv

    original_init = PlugChargerEnv._initialize_episode

    def sampled_reset(self, env_idx, options):
        self.goal_pose = plug_orientation.Pose.create_from_pq(
            torch.tensor([[0.3, -0.2, 0.1]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        )
        self.receptacle.pose = plug_orientation.Pose.create_from_pq(
            torch.tensor([[0.3, -0.2, 0.1]]), torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        )

    monkeypatch.setattr(PlugChargerEnv, "_initialize_episode", sampled_reset)
    env = object.__new__(plug_orientation.PlugChargerOrientationEnv)
    env.num_envs, env.device, env._episode_rng = 1, "cpu", None
    env.receptacle = type("Receptacle", (), {
        "pose": None, "set_pose": lambda self, pose: setattr(self, "pose", pose)
    })()
    env.goal_pose = None
    env._initialize_episode(torch.tensor([0]), {"target_yaw": 0.4})
    assert np.allclose(env.receptacle.pose.p.numpy(), [[0.3, -0.2, 0.1]])
    assert np.isclose(plug_orientation.world_x_yaw(env.receptacle.pose.q[0].numpy()), 0.4 - np.pi)
    assert np.isclose(plug_orientation.world_x_yaw(env.goal_pose.q[0].numpy()), 0.4)
    assert original_init is not None


def test_custom_orientation_state_roundtrips_unbatched_and_batched(monkeypatch):
    from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv

    monkeypatch.setattr(PlugChargerEnv, "set_state_dict", lambda self, state, env_idx=None: None)
    monkeypatch.setattr(PlugChargerEnv, "get_state_dict", lambda self: {})
    env = object.__new__(plug_orientation._PlugOrientationEnv)
    env.num_envs, env.device = 1, "cpu"
    env.goal_pose = plug_orientation.Pose.create_from_pq(
        torch.tensor([[0.1, 0.2, 0.3]]), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    env.set_state_dict({plug_orientation.STATE_KEY: torch.tensor([0.25, 0.4, 0.5, 0.6])})
    assert np.isclose(env.target_yaw, 0.25)
    assert np.allclose(env.goal_pose.p.numpy(), [[0.4, 0.5, 0.6]])
    restored = env.get_state_dict()
    assert np.allclose(restored[plug_orientation.STATE_KEY].numpy(), [[0.25, 0.4, 0.5, 0.6]])
    env.set_state_dict({plug_orientation.STATE_KEY: torch.tensor([[0.75, -0.4, 0.2, 0.6]])})
    assert np.isclose(env.target_yaw, 0.75)
    assert np.allclose(env.goal_pose.p.numpy(), [[-0.4, 0.2, 0.6]])


def test_reconstructed_plan_preserves_state_zero_and_targets_each_following_row():
    from examples.baselines.constrain import plug_orientation_reconstruction as reconstruction

    actions = np.zeros((2, 8), dtype=np.float32)
    states = [
        {"articulations": {"panda_wristcam": np.arange(31, dtype=np.float32) + i * 100}}
        for i in range(3)
    ]
    initial, targets = reconstruction.split_source_trajectory(states, actions)
    assert np.array_equal(initial["articulations"]["panda_wristcam"], np.arange(31))
    assert len(targets) == len(actions) == 2
    assert [int(row["articulations"]["panda_wristcam"][0]) for row in targets] == [100, 200]


def test_reconstructed_plan_rejects_t_plus_one_alignment_loss():
    from examples.baselines.constrain import plug_orientation_reconstruction as reconstruction

    with np.testing.assert_raises_regex(ValueError, "T\\+1 states"):
        reconstruction.split_source_trajectory([{}], np.zeros((1, 8)))


def test_native_pose_encoder_uses_controller_bounds_and_negative_rotation_scale():
    from examples.baselines.constrain import plug_orientation_reconstruction as reconstruction

    class Controller:
        action_space_low = torch.tensor([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
        action_space_high = -action_space_low

    # Identity current pose to a known world translation and a -0.1 rad X delta.
    action = reconstruction.encode_native_pose_delta(
        Controller(),
        current_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        target_pose=np.array([0.05, -0.02, 0.01, np.cos(-0.05), np.sin(-0.05), 0.0, 0.0]),
    )
    assert action.shape == (6,)
    assert np.isfinite(action).all()
    assert np.allclose(action[:3], [0.5, -0.2, 0.1], atol=1e-6)
    assert action[3] > 0
    assert action[3] * -0.1 < 0  # Native controller multiplies by negative rot_lower.


def test_native_pose_encoder_matches_official_controller_conversion():
    import sapien
    from transforms3d.quaternions import qinverse, qmult
    from mani_skill.trajectory.utils.actions.conversion import delta_pose_to_pd_ee_delta
    from examples.baselines.constrain import plug_orientation_reconstruction

    env = gym.make(
        plug_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu", reward_mode="sparse",
    )
    try:
        env.reset(seed=2026113001)
        controller = env.unwrapped.agent.controller.controllers["arm"]
        current = env.unwrapped.agent.tcp.pose.raw_pose[0].detach().cpu().numpy()
        target = current.copy()
        target[:3] += [0.013, -0.007, 0.004]
        target[3:] = qmult(target[3:], np.array([np.cos(0.04), 0.0, np.sin(0.04), 0.0]))
        actual = plug_orientation_reconstruction.encode_native_pose_delta(
            controller, current, target
        )
        delta_q = qmult(current[3:], qinverse(target[3:]))
        expected = delta_pose_to_pd_ee_delta(
            controller, sapien.Pose(target[:3] - current[:3], delta_q)
        )
        assert np.allclose(actual, expected, atol=1e-6, rtol=0)
    finally:
        env.close()


def test_reconstruction_control_loop_caps_controls_and_records_initial_row():
    from examples.baselines.constrain import plug_orientation_reconstruction as reconstruction

    class FakeEnv:
        def __init__(self):
            self.calls = 0
            self.unwrapped = self

        def get_state_dict(self):
            return {"row": np.array([self.calls])}

        def step(self, action):
            self.calls += 1
            return np.array([self.calls], dtype=np.float32), 0, False, False, {}

    env = FakeEnv()
    obs, states, actions = reconstruction.run_capped_waypoint_loop(
        env,
        [1.0, 2.0, 3.0],
        initial_observation=np.array([0.0]),
        initial_state={"row": np.array([0])},
        make_action=lambda target, gripper: np.array([target[0], gripper]),
        error=lambda target: 1.0,
        tolerance=0.1,
        max_controls=2,
        gripper=1.0,
    )
    assert env.calls == 2 == len(actions)
    assert len(states) == len(obs) == len(actions) + 1
    assert states[0]["row"].tolist() == [0]
    assert obs[0].tolist() == [0.0]


def test_pipeline_runs_four_real_subprocess_stages_and_accumulates_budgets(tmp_path):
    calls = []

    class Child:
        pid = 123
        returncode = 0

        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            calls.append(self)

        def communicate(self, timeout=None):
            command = self.command
            if "--replay-source" in command:
                out = Path(command[command.index("--replay-output-dir") + 1])
                out.mkdir(parents=True)
                (out / "replay_report.json").write_text(json.dumps({"stage_controls": 6, "complete": True}))
            else:
                out = Path(command[command.index("--output-dir") + 1])
                out.mkdir(parents=True)
                (out / "collection_report.json").write_text(json.dumps({"stage_controls": 5, "complete_plan": True}))
            return "fake child ok", None

    result = plug_orientation_pipeline.run_pipeline(
        tmp_path / "exp", tmp_path / "dataset", popen_factory=Child
    )
    assert [(c.command[c.command.index("--distribution") + 1],
             "--replay-source" in c.command) for c in calls] == [
        ("fixed", False), ("random", False), ("fixed", True), ("random", True)]
    assert [c.kwargs["start_new_session"] for c in calls] == [True] * 4
    assert result["expert_controls"] == 10
    assert result["replay_controls"] == 12
    assert calls[1].command[-1] == str(plug_orientation_preflight.MAX_STAGE_CONTROLS - 5)
    assert calls[3].command[-1] == str(plug_orientation_preflight.MAX_STAGE_CONTROLS - 6)
    assert result["status"] == "complete"
    assert json.loads((tmp_path / "exp" / "preregistration.json").read_text())["planned_denominator_per_profile"] == 10
    prereg = json.loads((tmp_path / "exp" / "preregistration.json").read_text())
    assert prereg["source_count"] == len(prereg["source_closure"])
    assert "native/mani_skill/examples/motionplanning/panda/solutions/plug_charger.py" in prereg["source_closure"]
    assert "native/mani_skill/agents/robots/panda/panda_wristcam.py" in prereg["source_closure"]


def test_pipeline_forwards_explicit_official_default_tilt_to_both_collectors(tmp_path):
    calls = []
    class Child:
        pid, returncode = 123, 0
        def __init__(self, command, **kwargs): self.command = command; calls.append(command)
        def communicate(self, timeout=None):
            if "--replay-source" in self.command:
                out = Path(self.command[self.command.index("--replay-output-dir") + 1])
                out.mkdir(parents=True)
                (out / "replay_report.json").write_text(json.dumps({"stage_controls": 0, "complete": True}))
            else:
                out = Path(self.command[self.command.index("--output-dir") + 1])
                out.mkdir(parents=True)
                (out / "collection_report.json").write_text(json.dumps({"stage_controls": 0, "complete_plan": True}))
            return "", None
    plug_orientation_pipeline.run_pipeline(tmp_path / "exp", tmp_path / "data", popen_factory=Child,
                                            expert_grasp_tilt_deg=15)
    collectors = [command for command in calls if "--replay-source" not in command]
    assert len(collectors) == 2
    assert all(float(command[command.index("--expert-grasp-tilt-deg") + 1]) == 15 for command in collectors)


def test_pipeline_stops_after_nonzero_child_and_preserves_partial(tmp_path):
    calls = []

    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            self.command = command
            self.returncode = 7
            calls.append(command)
        def communicate(self, timeout=None):
            return "failed", None

    with np.testing.assert_raises(Exception):
        plug_orientation_pipeline.run_pipeline(tmp_path / "exp", tmp_path / "data", popen_factory=Child)
    report = json.loads((tmp_path / "exp" / "pipeline_report.json").read_text())
    assert len(calls) == len(report["phases"]) == 1
    assert report["status"] == "partial"
    assert (tmp_path / "exp" / "collect_fixed.log").read_text() == "failed"


def test_pipeline_timeout_terminates_then_kills_process_group(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(plug_orientation_pipeline.os, "killpg", lambda pid, sig: sent.append(sig))

    class Child:
        pid = 23
        returncode = -9
        def __init__(self, command, **kwargs): pass
        def communicate(self, timeout=None):
            if timeout == 20:
                raise subprocess.TimeoutExpired("child", timeout)
            if timeout is not None and timeout < 20:
                raise subprocess.TimeoutExpired("child", timeout)
            return "captured-before-timeout", None

    import subprocess
    with np.testing.assert_raises(TimeoutError):
        plug_orientation_pipeline.run_pipeline(tmp_path / "exp", tmp_path / "data",
                                               popen_factory=Child, whole_seconds=1)
    assert sent == [plug_orientation_pipeline.signal.SIGTERM, plug_orientation_pipeline.signal.SIGKILL]
    report = json.loads((tmp_path / "exp" / "pipeline_report.json").read_text())
    assert report["phases"][0]["launched"] is True
    assert report["phases"][0]["returncode"] == -9
    assert report["phases"][0]["error"] == "TimeoutError"
    assert report["phases"][0]["controls_status"] == "unknown_if_no_report"
    assert report["incomplete_phase_count"] == 1
    assert report["partial_total_status"].startswith("incomplete")
    assert "expert_controls" not in report and "replay_controls" not in report
    assert (tmp_path / "exp" / "collect_fixed.log").read_text() == "captured-before-timeout"
    assert report["measured_completed_expert_controls"] == 0


def test_expert_shared_control_counter_stops_before_next_action():
    class Recorder:
        def step(self, action):
            raise AssertionError("must not reach recorder")
    bounded = plug_orientation_preflight._BoundedRecorder(
        Recorder(), 0, 2, lambda *_: None, {"controls": 2}, 2
    )
    with np.testing.assert_raises_regex(RuntimeError, "stage cap"):
        bounded.step(None)


def test_replay_state_finiteness_audit_checks_nested_arrays():
    assert plug_orientation_preflight._finite_tree({"x": np.array([1.0, 2.0])})
    assert not plug_orientation_preflight._finite_tree({"x": [1.0, float("nan")]})


def test_recorded_observation_audit_rejects_world_goal_leak():
    import h5py
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5") as stream:
        with h5py.File(stream.name, "w") as h5:
            group = h5.create_group("obs")
            group.create_dataset("goal_pose", data=np.zeros((2, 7)))
            with np.testing.assert_raises_regex(ValueError, "leaked"):
                plug_orientation_preflight._audit_observation_group(group, 2)


def test_recorded_observation_audit_rejects_wrong_length_and_nonfinite():
    import h5py
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5") as stream:
        with h5py.File(stream.name, "w") as h5:
            group = h5.create_group("obs")
            group.create_dataset("short", data=np.zeros((1, 3)))
            with np.testing.assert_raises_regex(ValueError, "rows"):
                plug_orientation_preflight._audit_observation_group(group, 2)
        with h5py.File(stream.name, "w") as h5:
            group = h5.create_group("obs")
            group.create_dataset("nan", data=np.array([[np.nan], [0.]]))
            with np.testing.assert_raises_regex(ValueError, "non-finite"):
                plug_orientation_preflight._audit_observation_group(group, 2)


def test_replay_source_without_initial_orientation_fails_closed(tmp_path):
    import h5py
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as h5:
        traj = h5.create_group("traj_0")
        states = traj.create_group("env_states")
        states.create_dataset("unrelated", data=np.zeros((2, 1)))
    source.with_suffix(".json").write_text(json.dumps({"episodes": [{
        "episode_id": 0, "episode_seed": 1, "success": True,
    }]}))
    with np.testing.assert_raises_regex((KeyError, ValueError), "orientation|plug_orientation_target"):
        plug_orientation_preflight.replay_successes(source, tmp_path / "out")


def test_pipeline_rejects_existing_output_without_overwrite(tmp_path):
    output = tmp_path / "exp"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("preserve")
    with np.testing.assert_raises(FileExistsError):
        plug_orientation_pipeline.run_pipeline(output, tmp_path / "data")
    assert marker.read_text() == "preserve"


def test_source_seal_rejects_changed_archived_bytes(tmp_path):
    import hashlib
    source = tmp_path / "source.py"
    archive = tmp_path / "archive"
    archive.mkdir()
    source.write_text("original")
    archived = archive / "own" / "source.py"
    archived.parent.mkdir()
    archived.write_text("tampered")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "preregistration.json"
    manifest.write_text(json.dumps({"source_closure": {"own/source.py": digest}}))
    with np.testing.assert_raises_regex(RuntimeError, "archive seal mismatch"):
        plug_orientation_pipeline._validate_source_seal(
            manifest, archive, {"own/source.py": digest}, {"own/source.py": source})


def test_source_seal_rejects_changed_current_bytes(tmp_path):
    import hashlib
    source = tmp_path / "source.py"
    archive = tmp_path / "archive"
    archive.mkdir()
    source.write_text("changed")
    archived = archive / "own" / "source.py"
    archived.parent.mkdir()
    archived.write_text("original")
    digest = hashlib.sha256(archived.read_bytes()).hexdigest()
    manifest = tmp_path / "preregistration.json"
    manifest.write_text(json.dumps({"source_closure": {"own/source.py": digest}}))
    with np.testing.assert_raises_regex(RuntimeError, "current source seal mismatch"):
        plug_orientation_pipeline._validate_source_seal(
            manifest, archive, {"own/source.py": digest}, {"own/source.py": source})


def test_actual_collect_runs_official_solver_seam_and_journals_attempts(monkeypatch, tmp_path):
    from types import SimpleNamespace
    task = SimpleNamespace(
        target_yaw=0.0,
        get_state_dict=lambda: {plug_orientation.STATE_KEY: torch.tensor([[0., 1., 2., 3.]])},
        goal_pose=SimpleNamespace(p=torch.tensor([[1., 2., 3.]])),
        charger=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
        receptacle=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
    )

    class Env:
        unwrapped = task
        def reset(self, **kwargs): return np.zeros(1), {}
        def step(self, action): return np.zeros(1), 0., False, False, {}
        def close(self): pass

    class Recorder:
        unwrapped = task
        def __init__(self, env, **kwargs): self.env = env
        def reset(self, **kwargs): return self.env.reset(**kwargs)
        def step(self, action): return self.env.step(action)
        def flush_trajectory(self, save): self.saved = save
        def close(self): pass

    monkeypatch.setattr(plug_orientation_preflight.gym, "make", lambda *a, **k: Env())
    monkeypatch.setattr(plug_orientation_preflight, "RecordEpisode", Recorder)
    def solver(env, **kwargs):
        env.reset(seed=kwargs["seed"])
        env.step(np.zeros(1))
        return (None, None, None, None, {"success": True})

    report = plug_orientation_preflight.collect("fixed", tmp_path / "collect", solver=solver,
                                                max_attempts=2)
    assert report["attempt_count"] == report["expert_successes"] == 2
    assert report["stage_controls"] == 2
    rows = [json.loads(line) for line in (tmp_path / "collect" / "attempts.jsonl").read_text().splitlines()]
    assert [row["target_yaw"] for row in rows] == [0., 0.]
    assert all(row["reset_capture"]["initial_state"][plug_orientation.STATE_KEY] == [[0., 1., 2., 3.]] for row in rows)


def test_default_collect_adapts_installed_mplib_before_official_solver(monkeypatch, tmp_path):
    import sys
    from types import ModuleType, SimpleNamespace

    class Pose:
        def __init__(self, p, q):
            self.p, self.q = p, q

    class Planner:
        def set_base_pose(self, pose):
            assert isinstance(pose, Pose)
        def plan_screw(self, goal, **kwargs):
            assert isinstance(goal, Pose)
            assert "use_point_cloud" not in kwargs
            return None

    mplib = ModuleType("mplib")
    mplib.Pose, mplib.Planner = Pose, Planner
    seed_events = []
    mplib.set_global_seed = lambda seed: seed_events.append(("mplib_seed", seed))
    monkeypatch.setitem(sys.modules, "mplib", mplib)

    task = SimpleNamespace(
        target_yaw=0.0,
        get_state_dict=lambda: {plug_orientation.STATE_KEY: torch.tensor([[0., 1., 2., 3.]])},
        goal_pose=SimpleNamespace(p=torch.tensor([[1., 2., 3.]])),
        charger=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
        receptacle=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
    )
    class Env:
        unwrapped = task
        def reset(self, **kwargs):
            seed_events.append(("reset", kwargs["seed"]))
            return np.zeros(1), {}
        def close(self): pass
    class Recorder:
        unwrapped = task
        def __init__(self, env, **kwargs): self.env = env
        def reset(self, **kwargs): return self.env.reset(**kwargs)
        def flush_trajectory(self, save): pass
        def close(self): pass
    monkeypatch.setattr(plug_orientation_preflight.gym, "make", lambda *a, **k: Env())
    monkeypatch.setattr(plug_orientation_preflight, "RecordEpisode", Recorder)

    solution = ModuleType("mani_skill.examples.motionplanning.panda.solutions.plug_charger")
    def official_solve(env, **kwargs):
        env.reset(seed=kwargs["seed"])
        seed_events.append(("planner", kwargs["seed"]))
        planner = Planner()
        planner.set_base_pose(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32))
        planner.plan_screw(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), use_point_cloud=False)
        return (None, None, None, None, {"success": False})
    solution.solve = official_solve
    monkeypatch.setitem(sys.modules, solution.__name__, solution)

    report = plug_orientation_preflight.collect("fixed", tmp_path / "collect", max_attempts=1)
    assert report["attempt_count"] == 1
    assert seed_events.index(("mplib_seed", 2026113001)) < seed_events.index(("reset", 2026113000))
    assert seed_events.index(("reset", 2026113000)) < seed_events.index(("mplib_seed", 2026113001), 1)
    assert seed_events.index(("mplib_seed", 2026113001), 1) < seed_events.index(("planner", 2026113000))
    row = json.loads((tmp_path / "collect" / "attempts.jsonl").read_text())
    assert row["reason"] == "official_success_false", row["reason"]


def test_failed_nonempty_expert_attempt_is_saved_not_success_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    task = SimpleNamespace(
        target_yaw=0.0,
        get_state_dict=lambda: {plug_orientation.STATE_KEY: torch.tensor([[0., 1., 2., 3.]])},
        goal_pose=SimpleNamespace(p=torch.tensor([[1., 2., 3.]])),
        charger=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
        receptacle=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
    )
    class Env:
        unwrapped = task
        def reset(self, **kwargs): return np.zeros(1), {}
        def step(self, action): return np.zeros(1), 0., False, False, {}
        def close(self): pass
    class Recorder:
        unwrapped = task
        def __init__(self, env, **kwargs): self._episode_id = -1
        def reset(self, **kwargs): return Env.reset(self, **kwargs)
        def step(self, action): return Env.step(self, action)
        def flush_trajectory(self, save):
            self.saved = save
            if save: self._episode_id += 1
        def close(self): pass
    monkeypatch.setattr(plug_orientation_preflight.gym, "make", lambda *a, **k: Env())
    monkeypatch.setattr(plug_orientation_preflight, "RecordEpisode", Recorder)
    def failed_solver(env, **kwargs):
        env.reset(seed=kwargs["seed"])
        env.step(np.array([1.]))
        return (None, None, None, None, {"success": False})
    report = plug_orientation_preflight.collect("fixed", tmp_path / "failed", solver=failed_solver,
                                                max_attempts=1)
    attempt = report["attempts"][0]
    assert attempt["expert_success"] is False
    assert attempt["controls"] == 1 and attempt["saved_episode_id"] == 0
    assert attempt["planner_seed"] == 2026113001
    assert report["planner_seed_policy"] == "scene_seed_plus_one_before_reset_and_planner_creation"


def test_replay_source_selection_uses_sparse_saved_episode_ids_and_keeps_failures_in_denominator():
    episodes = [{"episode_id": 0}, {"episode_id": 2}, {"episode_id": 5}]
    attempts = [
        {"saved_episode_id": 0, "expert_success": False},
        {"saved_episode_id": 2, "expert_success": True},
        {"saved_episode_id": 5, "expert_success": False},
    ]
    selected = plug_orientation_preflight._select_expert_successes(episodes, attempts)
    assert [row["episode_id"] for row in selected] == [2]
    assert len(episodes) == 3  # planned-source denominator is not success-selected


def test_replay_source_selection_rejects_missing_success_provenance():
    with np.testing.assert_raises_regex(ValueError, "success provenance"):
        plug_orientation_preflight._select_expert_successes([{"episode_id": 7}], [])


def test_replay_source_selection_rejects_conflicting_success_provenance():
    episodes = [{"episode_id": 7, "expert_success": False}]
    attempts = [{"saved_episode_id": 7, "expert_success": True}]
    with np.testing.assert_raises_regex(ValueError, "conflicting"):
        plug_orientation_preflight._select_expert_successes(episodes, attempts)


def test_malformed_initial_capture_is_rejected_without_partial_hdf5_alignment(tmp_path):
    import h5py
    converted = tmp_path / "converted.h5"
    with h5py.File(converted, "w") as h5:
        trajectory = h5.create_group("traj_0")
        trajectory.create_dataset("actions", data=np.zeros((1, 7)))
        states = trajectory.create_group("env_states")
        states.create_dataset("first", data=np.array([[1.], [2.]]))
        states.create_dataset("wrong_shape", data=np.zeros((2, 2)))
        trajectory.create_dataset("obs", data=np.zeros((2, 3)))
    with np.testing.assert_raises_regex(ValueError, "shape mismatch"):
        plug_orientation_preflight._align_initial_record(
            converted, {"first": np.array([9.]), "wrong_shape": np.array([1., 2., 3.])},
            np.ones(3), raw_sha256="raw")
    with h5py.File(converted, "r") as h5:
        assert h5["traj_0"]["env_states"]["first"][0, 0] == 1.
        assert np.all(h5["traj_0"]["obs"][0] == 0.)


def test_collect_journals_full_unexpected_solver_traceback(monkeypatch, tmp_path):
    from types import SimpleNamespace
    task = SimpleNamespace(
        target_yaw=0.0,
        get_state_dict=lambda: {plug_orientation.STATE_KEY: torch.tensor([[0., 1., 2., 3.]])},
        goal_pose=SimpleNamespace(p=torch.tensor([[1., 2., 3.]])),
        charger=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
        receptacle=SimpleNamespace(pose=SimpleNamespace(raw_pose=torch.zeros((1, 7)))),
    )
    class Env:
        unwrapped = task
        def reset(self, **kwargs): return np.zeros(1), {}
        def close(self): pass
    class Recorder:
        unwrapped = task
        def __init__(self, env, **kwargs): self.env = env
        def reset(self, **kwargs): return self.env.reset(**kwargs)
        def flush_trajectory(self, save): pass
        def close(self): pass
    monkeypatch.setattr(plug_orientation_preflight.gym, "make", lambda *a, **k: Env())
    monkeypatch.setattr(plug_orientation_preflight, "RecordEpisode", Recorder)
    def solver(*args, **kwargs):
        raise TypeError("planner binding rejected pose")

    plug_orientation_preflight.collect("fixed", tmp_path / "trace", solver=solver,
                                        max_attempts=1)
    row = json.loads((tmp_path / "trace" / "attempts.jsonl").read_text())
    assert row["reason"] == "TypeError: planner binding rejected pose"
    assert "Traceback (most recent call last):" in row["exception_traceback"]
    assert "TypeError: planner binding rejected pose" in row["exception_traceback"]


def test_actual_replay_successes_calls_public_api_and_audits_real_conversion(monkeypatch, tmp_path):
    import h5py
    from types import SimpleNamespace
    from mani_skill.trajectory import replay_trajectory, utils as trajectory_utils

    source = tmp_path / "expert.h5"
    orientation = np.array([[0.2, 1., 2., 3.], [0.3, 1., 2., 3.]])
    markers = np.array([[7., 8.], [9., 10.]])
    with h5py.File(source, "w") as h5:
        states = h5.create_group("traj_0").create_group("env_states")
        states.create_dataset(plug_orientation.STATE_KEY, data=orientation)
        states.create_dataset("marker", data=markers)
    source_hash = plug_orientation_preflight._sha256(source)
    source.with_suffix(".json").write_text(json.dumps({
        "env_info": {"env_id": "fake"}, "episodes": [{
            "episode_id": 0, "episode_seed": 12, "success": True,
        }]
    }))

    class Env:
        def __init__(self): self.unwrapped = self; self.calls = 0
        control_mode = "pd_ee_delta_pose"
        def reset(self, **kwargs): return np.zeros(1), {}
        def step(self, action): self.calls += 1; return np.zeros(1), 0., False, False, {}
        def set_state_dict(self, state): self.state = state
        def get_state_dict(self): return self.state
        def get_obs(self, info=None, unflattened=False):
            self.get_obs_kwargs = {"info": info, "unflattened": unflattened}
            if unflattened:
                return {"agent": {"qpos": torch.zeros((1, 2))}, "tcp_pose": torch.ones((1, 7))}
            return torch.zeros((1, 9))
        def evaluate(self):
            return {"success": torch.tensor([True]), "obj_to_goal_dist": torch.tensor([0.]),
                    "obj_to_goal_angle": torch.tensor([0.])}
        def close(self): pass

    envs = []
    def make(*args, **kwargs):
        env = Env()
        env.state = {plug_orientation.STATE_KEY: orientation[0:1].copy(),
                     "marker": markers[0:1].copy()}
        envs.append(env)
        return env
    monkeypatch.setattr(plug_orientation_preflight.gym, "make", make)
    monkeypatch.setattr(trajectory_utils, "dict_to_list_of_dicts", lambda group: [
        {key: np.asarray(group[key][i]) for key in group}
        for i in range(len(group[plug_orientation.STATE_KEY]))
    ])
    class Args:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    monkeypatch.setattr(replay_trajectory, "Args", Args)
    calls = []
    def public_main(args):
        calls.append(args)
        replay_env = plug_orientation_preflight.gym.make()
        limited = plug_orientation_preflight._ControlLimitedEnv(
            replay_env, plug_orientation_preflight.MAX_CONTROLS_PER_ATTEMPT,
            public_main.counter)
        limited.step(np.zeros(1))
        out = Path(args.traj_path).with_name("expert.state.pd_ee_delta_pose.physx_cpu.h5")
        with h5py.File(out, "w") as h5:
            traj = h5.create_group("traj_0")
            traj.create_dataset("actions", data=np.zeros((1, 7)))
            states = traj.create_group("env_states")
            states.create_dataset(plug_orientation.STATE_KEY, data=orientation[::-1])
            states.create_dataset("marker", data=markers[::-1])
            obs = traj.create_group("obs")
            obs.create_dataset("flat", data=np.zeros((2, 9)))
        return None
    public_main.counter = {"controls": 0}
    monkeypatch.setattr(replay_trajectory, "main", public_main)

    report = plug_orientation_preflight.replay_successes(source, tmp_path / "replayed")
    assert len(calls) == 1 and calls[0].max_retry == 0
    assert report["replay_successes"] == report["attempt_count"] == 1, report["attempts"][0].get("reason")
    row = report["attempts"][0]
    assert row["target_yaw"] == 0.2 and row["state_count"] == 2
    assert report["stage_controls"] == 1
    assert envs[-1].get_obs_kwargs["unflattened"] is True
    converted = tmp_path / "replayed" / "episode_0000" / "expert.state.pd_ee_delta_pose.physx_cpu.h5"
    raw = tmp_path / "replayed" / "episode_0000" / "expert.raw_recording.state.pd_ee_delta_pose.physx_cpu.h5"
    alignment = json.loads(converted.with_suffix(".alignment.json").read_text())
    with h5py.File(raw, "r") as raw_h5, h5py.File(converted, "r") as fixed_h5:
        assert raw_h5["traj_0"]["env_states"][plug_orientation.STATE_KEY][0, 0] == .3
        assert fixed_h5["traj_0"]["env_states"][plug_orientation.STATE_KEY][0, 0] == .2
        assert np.array_equal(fixed_h5["traj_0"]["env_states"]["marker"][0], [7., 8.])
        assert np.array_equal(fixed_h5["traj_0"]["obs"]["flat"][0], [0, 0, 1, 1, 1, 1, 1, 1, 1])
        assert np.array_equal(raw_h5["traj_0"]["actions"], fixed_h5["traj_0"]["actions"])
        assert "initial_record_alignment" in fixed_h5["traj_0"].attrs
    assert alignment["raw_recording_sha256"] == plug_orientation_preflight._sha256(raw)
    assert alignment["corrected_sha256"] == plug_orientation_preflight._sha256(converted)
    assert len(alignment["actual_initial_capture_sha256"]) == 64
    assert plug_orientation_preflight._sha256(source) == source_hash
