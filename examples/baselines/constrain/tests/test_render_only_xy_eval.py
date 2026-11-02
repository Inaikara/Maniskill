import numpy as np
import pytest

from examples.baselines.constrain.render_only_xy_eval import instrument_env


class _Actor:
    def __init__(self, xy):
        self.pose = type("Pose", (), {"p": np.array([[xy[0], xy[1], 0.4]])})()


class _Task:
    def __init__(self):
        self.obstacle_mode = "render_only"
        visual = type("Visual", (), {"visibility": 1.0})()
        obj = type("Object", (), {"find_component_by_type": lambda _self, _kind: visual})()
        self.obstacle = type("Obstacle", (), {"_bodies": [], "_objs": [obj]})()
        self.agent = type("Agent", (), {"tcp": _Actor([0.08, 0.0])})()
        self._obstacle_center = np.array([[0.0, 0.0]])
        self.obstacle_collision_radius = 0.03
        self._cube_xy_radius = np.sqrt(2) * .02
        self.obstacle_safety_margin = 0.005
        self.radius = .02


class _Env:
    def __init__(self):
        self.unwrapped = _Task()
        self.calls = []

    def step(self, action):
        self.calls.append(np.asarray(action).copy())
        self.unwrapped.agent.tcp.pose.p[0, :2] += np.asarray(action)[0, :2] * 0.1
        return None, 0.0, False, False, {"native_success": False}


def test_each_native_control_step_uses_measured_tcp_and_preserves_other_channels():
    env = _Env()
    trace = instrument_env(env, "pickcube")
    requested = np.array([[-0.35, 0.0, 0.1, 0.2, 0.0, 0.0, -1.0]])
    env.step(requested)
    env.step(requested)
    assert len(trace) == len(env.calls) == 2
    assert env._render_only_xy_radius == pytest.approx(.03 + np.sqrt(2) * .02 + .005)
    assert trace[0]["tcp_xy_before"] == pytest.approx([0.08, 0.0])
    assert trace[1]["tcp_xy_before"] == pytest.approx(env.calls[0][0, :2] * .1 + [.08, 0.0])
    assert np.array_equal(env.calls[0][0, 2:], requested[0, 2:])
    assert np.array_equal(env.calls[1][0, 2:], requested[0, 2:])
    assert all(row["filtered_action"] != row["requested_action"] for row in trace)


def test_constraint_does_not_bypass_when_adapter_control_is_disabled():
    env = _Env()
    trace = instrument_env(env, "placesphere", control_enabled=False)
    action = np.array([[-0.35, 0.0, 0, 0, 0, 0, -1.0]])
    env.step(action)
    assert len(trace) == 1
    assert not np.array_equal(env.calls[0], action)


def test_source_registration_is_immutable_and_validates_sources(tmp_path):
    from examples.baselines.constrain.render_only_xy_eval import freeze_registration, validate_registration

    source = tmp_path / "source.py"
    source.write_text("frozen")
    path = tmp_path / "registration.json"
    freeze_registration(path, "pickcube", [0], {"source": source})
    with pytest.raises(FileExistsError):
        freeze_registration(path, "pickcube", [0], {"source": source})
    source.write_text("changed")
    with pytest.raises(ValueError, match="source"):
        validate_registration(path)


def test_failure_denominator_counts_missing_rows():
    from examples.baselines.constrain.render_only_xy_eval import summarize_rows

    result = summarize_rows([{"status": "evaluated", "native_success": True,
                              "xy_zero": True}, {"status": "runtime_error"}], requested=3)
    assert result["denominator"] == 3
    assert result["runtime_errors"] == 2
    assert result["native_success_count"] == 1
