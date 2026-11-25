import numpy as np
import pytest
import json

from examples.baselines.constrain.peg_orientation_takeover import (
    run_live_expert_continuation,
)
from examples.baselines.constrain import peg_orientation_takeover_experiment as experiment


def test_run_finalizes_numpy_arm_rows_and_persists_canonical_hash(tmp_path, monkeypatch):
    outdir = tmp_path / "study"
    attempt = outdir / "attempt_mock"
    outdir.mkdir()
    registration = {"source_closure_sha256": {}}
    registration["registration_sha256"] = experiment.canonical_hash(registration)
    (outdir / "preregistration.json").write_text(json.dumps(registration))

    monkeypatch.setattr(experiment, "_source_closure", lambda: {})
    monkeypatch.setattr(experiment, "EPISODE_IDS", (7,))
    monkeypatch.setattr(experiment, "_trajectory", lambda index: ({}, np.zeros((2, 3)), np.zeros((1, 2)), []))
    monkeypatch.setattr(experiment, "_run_replay_arm", lambda *args: {
        "status": "eligible", "controls": 1, "native_success": np.bool_(True),
        "obs_history": [np.array([1.0, 2.0])],
        "branch": {"planner_arrays": np.array([[3.0, 4.0]])},
    })
    monkeypatch.setattr(experiment, "_run_live_arm", lambda *args: {
        "status": "eligible", "controls": 2,
        "obs_history": [np.array([5.0])],
        "branch": {"planner_arrays": np.array([6.0])},
    })

    experiment.run(outdir, "mock")

    saved = json.loads((attempt / "results.json").read_text())
    digest = saved.pop("results_sha256")
    assert digest == experiment.canonical_hash(saved)
    assert len(saved["rows"]) == 2
    assert saved["rows"][0]["obs_history"] == [[1.0, 2.0]]
    assert saved["rows"][0]["native_success"] is True
    ledger = json.loads((attempt / "execution_ledger.json").read_text())
    assert len(ledger["arms"]) == 2
    assert all(row["status"] == "completed" for row in ledger["arms"])


class _Controller:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _Agent:
    def __init__(self):
        self.control_mode = "pd_ee_delta_pose"
        self.controller = _Controller()
        self.mode_changes = []

    def set_control_mode(self, mode):
        self.mode_changes.append(mode)
        self.control_mode = mode
        self.controller = _Controller()


class _Env:
    def __init__(self, successes=()):
        self.agent = _Agent()
        self.physical = np.array([0.1, -0.2, 0.3, 1.0, 2.0])
        self.grasped = True
        self.successes = set(successes)
        self.actions = []
        self.reset_count = 0

    @property
    def unwrapped(self):
        return self

    @property
    def control_mode(self):
        return self.agent.control_mode

    def snapshot(self):
        return self.physical.copy()

    def is_grasping(self):
        return self.grasped

    def reset(self, *args, **kwargs):
        self.reset_count += 1
        raise AssertionError("handoff must not reset the environment")

    def step(self, action):
        assert action["control_mode"] == "pd_joint_pos"
        assert action["action"].shape == (1, 8)
        self.actions.append(action["action"].copy())
        self.physical[0] += 0.001
        success = len(self.actions) in self.successes
        return None, 0.0, False, False, {"success": success}


class _Planner:
    CLOSED = -1

    def __init__(self, env, actions=1, status=0):
        self.env = env
        self.actions = actions
        self.status = status
        self.closed = False
        self.gripper_state = 1

    def run(self):
        for _ in range(self.actions):
            self.env.step(np.zeros(8))
        return self.status

    def close(self):
        self.closed = True


def test_handoff_switches_public_controller_without_physical_step_and_preserves_state():
    env = _Env()
    planners = []

    def factory(proxy):
        planner = _Planner(proxy)
        planners.append(planner)
        return planner

    result = run_live_expert_continuation(
        env, factory, lambda task, planner: planner.run(), max_controls=2,
        state_reader=lambda task: {"physical": task.snapshot()},
        grasp_check=lambda task: task.is_grasping(),
    )

    assert result["status"] == "plan_complete"
    assert result["controls"] == 1
    assert result["controller_mode_before"] == "pd_ee_delta_pose"
    assert result["controller_mode_after"] == "pd_joint_pos"
    assert result["controller_switch_physical_steps"] == 0
    assert result["switch_state_max_abs_error"] == 0
    assert env.agent.mode_changes == ["pd_joint_pos"]
    assert env.agent.controller.reset_count == 1
    assert env.reset_count == 0
    assert planners[0].gripper_state == -1
    assert planners[0].closed


def test_live_expert_stops_at_native_success_and_counts_only_executed_controls():
    env = _Env(successes={2})
    planner = _Planner
    result = run_live_expert_continuation(
        env, lambda proxy: planner(proxy, actions=5),
        lambda task, solver: solver.run(), max_controls=5,
        state_reader=lambda task: {"physical": task.snapshot()},
        grasp_check=lambda task: task.is_grasping(),
    )

    assert result["status"] == "native_success"
    assert result["controls"] == 2
    assert len(env.actions) == 2


def test_live_expert_never_exceeds_control_cap_and_closes_planner():
    env = _Env()
    planners = []

    def factory(proxy):
        planner = _Planner(proxy, actions=4)
        planners.append(planner)
        return planner

    result = run_live_expert_continuation(
        env, factory, lambda task, planner: planner.run(), max_controls=2,
        state_reader=lambda task: {"physical": task.snapshot()},
        grasp_check=lambda task: task.is_grasping(),
    )

    assert result["status"] == "control_cap"
    assert result["controls"] == 2
    assert len(env.actions) == 2
    assert planners[0].closed


def test_plan_failure_is_distinct_and_planner_always_closes():
    env = _Env()
    planners = []

    def factory(proxy):
        planner = _Planner(proxy, actions=0, status=-1)
        planners.append(planner)
        return planner

    result = run_live_expert_continuation(
        env, factory, lambda task, planner: planner.run(), max_controls=2,
        state_reader=lambda task: {"physical": task.snapshot()},
        grasp_check=lambda task: task.is_grasping(),
    )

    assert result["status"] == "plan_failure"
    assert result["controls"] == 0
    assert planners[0].closed


def test_handoff_refuses_ungrasped_state_without_reset_or_mode_change():
    env = _Env()
    env.grasped = False

    with pytest.raises(ValueError, match="already-grasped"):
        run_live_expert_continuation(
            env, lambda proxy: _Planner(proxy), lambda task, planner: planner.run(),
            max_controls=2, state_reader=lambda task: {"physical": task.snapshot()},
            grasp_check=lambda task: task.is_grasping(),
        )

    assert env.agent.mode_changes == []
    assert env.reset_count == 0


def test_handoff_refuses_controller_switch_that_moves_physical_state():
    env = _Env()
    reads = 0

    def state_reader(task):
        nonlocal reads
        reads += 1
        state = task.snapshot()
        if reads == 2:
            state[0] += 2e-5
        return {"physical": state}

    with pytest.raises(RuntimeError, match="changed physical branch state"):
        run_live_expert_continuation(
            env, lambda proxy: _Planner(proxy), lambda task, planner: planner.run(),
            max_controls=2, state_reader=state_reader,
            grasp_check=lambda task: task.is_grasping(),
        )

    assert env.actions == []
