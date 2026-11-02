import numpy as np
import pytest

from examples.baselines.constrain.render_only_xy_eval import (
    consume_replan_request, instrument_env, should_replan_after_control,
)


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


def test_filter_intervention_requests_replan_and_next_plan_consumes_it():
    env = _Env()
    trace = instrument_env(env, "pickcube", event_replan=True)
    class FakeAdapter:
        def __init__(self):
            self.observations = []

        def get_action(self, observation):
            self.observations.append(observation)
            return list(range(8))

    adapter = FakeAdapter()
    observation = "before"
    plans = 0
    while plans < 2:
        consume_replan_request(env, len(env.calls) + 1)
        actions = adapter.get_action(observation)
        plans += 1
        for action in actions:
            if action == 0:
                env.step(np.array([[-0.35, 0.0, 0.1, 0.2, 0.0, 0.0, -1.0]]))
                observation = "after-control-1"
            else:
                break
            if should_replan_after_control(env, None):
                break
        else:
            continue
        if plans == 2:
            break
    assert adapter.observations == ["before", "after-control-1"]
    assert trace[0]["requested_replan"] is True
    assert trace[0]["replan_consumed_at_control_index"] == 2


def test_unmodified_action_does_not_request_event_replan():
    env = _Env()
    env.unwrapped.agent.tcp.pose.p[0, :2] = [0.2, 0.2]
    trace = instrument_env(env, "pickcube", event_replan=True)
    env.step(np.array([[0.0, 0.0, 0.1, 0.2, 0.0, 0.0, -1.0]]))
    assert trace[0]["requested_replan"] is False
    assert env._render_only_xy_replan_requested is False
    assert should_replan_after_control(env, None) is False


def test_without_intervention_the_original_eight_action_chunk_runs_to_completion():
    env = _Env()
    env.unwrapped.agent.tcp.pose.p[0, :2] = [0.2, 0.2]
    trace = instrument_env(env, "pickcube", event_replan=True)
    fake_plan = [np.array([[0.0, 0.0, 0, 0, 0, 0, -1.0]]) for _ in range(8)]
    plans = 1
    for action in fake_plan:
        env.step(action)
        if should_replan_after_control(env, None):
            break
    assert plans == 1
    assert len(trace) == 8
    assert all(step["replan_consumed_at_control_index"] is None for step in trace)


def test_failed_second_filter_retains_initial_point_and_first_executed_segment():
    from examples.baselines.constrain.render_only_xy_eval import score_native_xy_trace

    env = _Env()
    def lagged_step(action):
        env.calls.append(np.asarray(action).copy())
        env.unwrapped.agent.tcp.pose.p[0, :2] = [0.02, 0.0]
        return None, 0.0, False, False, {"native_success": False}
    env.step = lagged_step
    trace = instrument_env(env, "pickcube")
    requested = np.array([[0.35, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]])
    env.step(requested)
    with pytest.raises(ValueError, match="starts inside"):
        env.step(requested)

    assert len(trace) == 1
    assert trace[0]["env_step_executed"] is True
    assert trace[0]["requested_action"] == requested[0].tolist()
    assert trace[0]["filtered_action"] is not None
    assert trace[0]["tcp_xy_after"] == pytest.approx([0.02, 0.0])
    assert env._render_only_xy_initial_tcp_xy == pytest.approx([0.055, 0.0])
    score = score_native_xy_trace(trace, env._render_only_xy_initial_tcp_xy,
                                  [0.0, 0.0], 0.035)
    assert score["violating_samples"] == [1]
    assert score["violating_segments"] == [0]


def test_native_env_step_exception_retains_attempt_action_and_latest_pose():
    env = _Env()
    def fail_after_motion(action):
        env.calls.append(np.asarray(action).copy())
        env.unwrapped.agent.tcp.pose.p[0, :2] = [0.06, 0.01]
        raise RuntimeError("native-step-fixture")

    env.step = fail_after_motion
    trace = instrument_env(env, "pickcube")
    with pytest.raises(RuntimeError, match="native-step-fixture"):
        env.step(np.array([[0.0, 0.0, 0, 0, 0, 0, -1.0]]))
    assert len(trace) == 1
    attempt = trace[0]
    assert attempt["env_step_executed"] is True
    assert attempt["requested_action"] is not None
    assert attempt["filtered_action"] is not None
    assert attempt["tcp_xy_after"] == pytest.approx([0.06, 0.01])


def test_failure_row_keeps_partial_trace_without_fabricating_success():
    from examples.baselines.constrain import render_only_xy_eval as evaluator

    env = _Env()
    trace = instrument_env(env, "pickcube")
    env.step(np.array([[0.0, 0.0, 0, 0, 0, 0, -1.0]]))
    row = evaluator._runtime_error_row(7, RuntimeError("fixture"))
    assert row["status"] == "runtime_error"
    assert row["native_success"] is False
    assert row["scene_index"] == 7
    assert row["initial_tcp_xy"] == pytest.approx([0.055, 0.0])
    assert row["executed_trace"] == trace
    assert row["failed_control_index"] is None


def test_trace_validation_includes_native_first_success_endpoint_and_chord():
    from examples.baselines.constrain.render_only_xy_eval import score_native_xy_trace

    trace = [{"tcp_xy_after": [-0.03, 0.03], "native_success": True,
              "obstacle_center_xy": [0.0, 0.0], "forbidden_radius_m": 0.02}]
    result = score_native_xy_trace(trace, [-0.04, 0.0], [0.0, 0.0], 0.02)
    assert result["first_success_step"] == 1
    assert result["first_success_step_valid"] is True
    assert result["zero_xy_violations"] is True


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


def test_common_initial_noise_reuses_latent_with_current_condition_and_isolates_rng():
    import torch
    from examples.baselines.constrain.render_only_xy_eval import CommonInitialNoise

    class Adapter:
        def __init__(self):
            self.latents = []

        def get_action(self, condition):
            latent = torch.randn((1, 16, 7), device=condition.device)
            self.latents.append(latent.clone())
            return latent + condition

    adapter = Adapter()
    torch.manual_seed(7123)
    expected_first = adapter.get_action(torch.zeros(1, 1, 1))
    torch.manual_seed(7123)
    adapter = Adapter()
    policy = CommonInitialNoise(adapter, "cpu")
    first = policy.get_action(torch.zeros(1, 1, 1))
    assert torch.equal(first, expected_first)

    global_state = torch.get_rng_state().clone()
    second = policy.get_action(torch.ones(1, 1, 1))
    assert torch.equal(adapter.latents[0], adapter.latents[1])
    assert torch.allclose(second - first, torch.ones_like(first))
    assert torch.equal(torch.get_rng_state(), global_state)
    assert policy.plan_count == 2


def test_common_initial_noise_is_episode_local_and_cuda_rng_isolated_when_available():
    import torch
    from examples.baselines.constrain.render_only_xy_eval import CommonInitialNoise

    class Adapter:
        def get_action(self, condition):
            return torch.randn((1, 2), device=condition.device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(41)
    first_policy = CommonInitialNoise(Adapter(), device)
    first = first_policy.get_action(torch.zeros(1, device=device))
    before = torch.cuda.get_rng_state(torch.device(device)).clone() if device == "cuda" else torch.get_rng_state().clone()
    first_policy.get_action(torch.ones(1, device=device))
    after = torch.cuda.get_rng_state(torch.device(device)) if device == "cuda" else torch.get_rng_state()
    assert torch.equal(before, after)

    torch.manual_seed(41)
    second_policy = CommonInitialNoise(Adapter(), device)
    assert torch.equal(first, second_policy.get_action(torch.zeros(1, device=device)))


def test_pick_profile_does_not_install_common_initial_noise():
    import torch
    from examples.baselines.constrain.render_only_xy_eval import configure_place_common_noise

    class Adapter:
        def get_action(self, condition):
            return torch.randn(1)

    adapter = Adapter()
    original = adapter.get_action
    assert configure_place_common_noise(adapter, "pickcube", True, "cpu") is None
    assert adapter.get_action == original


def test_event_replan_uses_new_observation_with_the_same_initial_latent():
    import torch
    from examples.baselines.constrain.render_only_xy_eval import (
        configure_place_common_noise, consume_replan_request,
    )

    class Adapter:
        def __init__(self):
            self.conditions, self.latents = [], []

        def get_action(self, condition):
            latent = torch.randn((1, 1), device=condition.device)
            self.conditions.append(condition.clone())
            self.latents.append(latent)
            return latent + condition

    env = _Env()
    trace = instrument_env(env, "placesphere", event_replan=True)
    adapter = Adapter()
    policy = configure_place_common_noise(adapter, "placesphere", True, "cpu")
    observation = torch.tensor([[0.0]])
    first = policy.get_action(observation)
    env.step(np.array([[-0.35, 0.0, 0.1, 0.2, 0.0, 0.0, -1.0]]))
    assert should_replan_after_control(env, None)
    assert consume_replan_request(env, 2)
    observation = torch.tensor([[1.0]])
    second = policy.get_action(observation)

    assert trace[0]["replan_consumed_at_control_index"] == 2
    assert torch.equal(adapter.latents[0], adapter.latents[1])
    assert torch.equal(adapter.conditions[0], torch.tensor([[0.0]]))
    assert torch.equal(adapter.conditions[1], observation)
    assert torch.allclose(second - first, torch.ones_like(first))


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
