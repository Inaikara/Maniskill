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
        self.agent = type("Agent", (), {"tcp": _Actor([0.055, 0.0])})()
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
    assert env._render_only_xy_radius == pytest.approx(.03 + .005)
    assert trace[0]["tcp_xy_before"] == pytest.approx([0.055, 0.0])
    assert trace[1]["tcp_xy_before"] == pytest.approx(env.calls[0][0, :2] * .1 + [.055, 0.0])
    assert np.array_equal(env.calls[0][0, 2:], requested[0, 2:])
    assert np.array_equal(env.calls[1][0, 2:], requested[0, 2:])
    assert all(row["filtered_action"] != row["requested_action"] for row in trace)


def test_geometry_is_read_live_after_scene_restore_and_tcp_only_profile():
    env = _Env()
    trace = instrument_env(env, "pickcube")
    # The environment wrapper is installed before the saved scene is restored.
    env.unwrapped._obstacle_center[:] = [[0.2, 0.0]]
    env.unwrapped.obstacle_collision_radius = 0.04
    env.step(np.array([[0.0, 0.0, 0, 0, 0, 0, -1.0]]))
    assert trace[0]["obstacle_center_xy"] == pytest.approx([0.2, 0.0])
    assert trace[0]["forbidden_radius_m"] == pytest.approx(0.045)
    assert env._render_only_xy_radius == pytest.approx(0.045)


def test_independent_trace_scorer_rejects_cached_wrong_center_and_scores_chords():
    from examples.baselines.constrain.render_only_xy_eval import score_native_xy_trace

    trace = [{"tcp_xy_before": [-0.1, 0.0], "tcp_xy_after": [0.1, 0.0],
              "native_success": False,
              "obstacle_center_xy": [0.0, 0.0], "forbidden_radius_m": 0.02}]
    result = score_native_xy_trace(trace, [-0.1, 0.0], [0.0, 0.0], 0.02)
    assert result["zero_xy_violations"] is False
    assert result["violating_segments"] == [0]
    with pytest.raises(ValueError, match="geometry"):
        score_native_xy_trace(trace, [-0.1, 0.0], [0.5, 0.0], 0.02)


def test_constraint_does_not_bypass_when_adapter_control_is_disabled():
    env = _Env()
    trace = instrument_env(env, "placesphere", control_enabled=False)
    action = np.array([[-0.9, 0.0, 0, 0, 0, 0, -1.0]])
    env.step(action)
    assert len(trace) == 1
    assert env.calls[0][0, 0] != pytest.approx(action[0, 0])
    assert abs(env.calls[0][0, 0]) <= .35


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


def test_explicit_catalog_is_pinned_bounded_and_unique(tmp_path):
    import hashlib
    import json
    from examples.baselines.constrain.render_only_xy_eval import validate_explicit_catalog

    scene_dir = tmp_path / "scenes"
    scene_dir.mkdir()
    h5 = scene_dir / "scenes.h5"
    h5.write_bytes(b"catalog")
    body = {"format": "pickcube-transport-detour-scenes-v2", "scene_count": 1,
            "scenes_file": "scenes.h5", "scenes": [{}]}
    body["manifest_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, indent=2).encode()).hexdigest()
    manifest = scene_dir / "manifest.json"
    manifest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    registration = {"inputs": {
        "scene_manifest": {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "scene_h5": {"path": str(h5), "sha256": hashlib.sha256(h5.read_bytes()).hexdigest()}}}
    validate_explicit_catalog(registration, "pickcube", [0], scene_dir)
    with pytest.raises(ValueError, match="unique"):
        validate_explicit_catalog(registration, "pickcube", [0, 0], scene_dir)
    with pytest.raises(ValueError, match="outside"):
        validate_explicit_catalog(registration, "pickcube", [1], scene_dir)
    h5.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="input mismatch"):
        validate_explicit_catalog(registration, "pickcube", [0], scene_dir)
