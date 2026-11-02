import pytest

from examples.baselines.constrain.evaluate_placesphere_paired import (
    classify_phase, classify_termination, paired_audit, policy_condition, summarize_episodes,
)


def test_native_success_wins_only_after_step_and_unsafe_success_is_not_safe():
    assert classify_termination(False, False, False, 200, 200) == "control_step_limit"
    assert classify_termination(True, False, False, 34, 200) == "native_success"
    summary = summarize_episodes([{"status": "evaluated", "result": {
        "termination_reason": "native_success", "elapsed_steps": 34,
        "native_success": True, "safe_success": False, "obstacle_contact": True,
        "xy_forbidden_crossing": False, "phase": "task_success"}}], 1)
    assert summary["task_success_count"] == 1
    assert summary["safe_success_count"] == 0
    assert summary["obstacle_contact_count"] == 1


def test_release_and_stability_phases_are_distinct_and_cumulative():
    assert classify_phase(True, True, True, False, False) == "transport_no_release"
    assert classify_phase(True, True, True, True, False) == "release_or_stability_failed"


def test_base_and_control_adapters_receive_their_frozen_condition_shape():
    class Base:
        agent = object()
        def base_cond(self, queued):
            return queued[..., :39].flatten(1)

    class Control:
        def base_cond(self, queued):
            return queued[..., :39].flatten(1)

    queued = __import__("torch").zeros(1, 2, 42)
    assert policy_condition(Base(), queued).shape == (1, 2, 39)
    assert policy_condition(Control(), queued).shape == (1, 78)


def test_runtime_errors_stay_in_fixed_denominator_and_flags_are_not_inferred():
    summary = summarize_episodes([
        {"status": "evaluated", "result": {"native_success": False, "safe_success": False,
         "obstacle_contact": False, "xy_forbidden_crossing": False,
         "termination_reason": "control_step_limit", "elapsed_steps": 200, "phase": "no_grasp"}},
        {"status": "runtime_error", "error": "step failed"}], requested=2)
    assert summary["requested_denominator"] == 2
    assert summary["runtime_errors"] == 1
    assert summary["task_success_rate"] == 0
    assert summary["control_step_limit_count"] == 1
    with pytest.raises(ValueError, match="positive"):
        summarize_episodes([], requested=0)


def test_paired_restore_audit_compares_actual_fingerprints(tmp_path):
    for policy in ("base_dp", "constrain_dp"):
        folder = tmp_path / policy
        folder.mkdir()
        for i in range(100):
            row = {"scene_id": f"s{i}", "seed": i,
                   "restored_state_sha256": "state", "restored_controller_sha256": "ctrl",
                   "restored_raw_observation_sha256": "raw",
                   "restored_policy_observation_sha256": "policy",
                   "status": "evaluated", "result": {"native_success": False, "safe_success": False}}
            (folder / f"episode_{i:03d}.json").write_text(__import__("json").dumps(row))
        (folder / "representative.mp4").write_bytes(b"video")
    assert paired_audit(tmp_path)["restore_pairs_match"] is True
    path = tmp_path / "constrain_dp/episode_017.json"
    row = __import__("json").loads(path.read_text())
    row["restored_controller_sha256"] = "different"
    path.write_text(__import__("json").dumps(row))
    with pytest.raises(ValueError, match="scene index 17"):
        paired_audit(tmp_path)


@pytest.mark.parametrize("native_success,expected_steps", [(True, 1), (False, 200)])
def test_physical_rollout_without_render_instrumentation_counts_executed_task_actions(
        monkeypatch, native_success, expected_steps):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from examples.baselines.constrain import evaluate_placesphere_paired as rollout

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.agent = SimpleNamespace(controller=SimpleNamespace(get_state=lambda: {}))
            self.obj = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., .1]])))
            self.bin = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[.1, 0., 0.]])))
            self.closed = False

        def reset(self, *args, **kwargs):
            return None

        def get_state_dict(self):
            return {}

        def step(self, action):
            self.steps += 1
            success = native_success and self.steps == 1
            info = {key: torch.tensor([value]) for key, value in {
                "obstacle_collision": False, "obstacle_crossing": False,
                "is_obj_grasped": False, "native_success": success,
                "is_obj_on_bin": success, "is_obj_static": success}.items()}
            return torch.zeros(42), 0., success, False, info

        def close(self):
            self.closed = True

    env = Env()
    env.steps = 0
    monkeypatch.setattr(rollout, "_make_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(rollout, "_tree_hash", lambda _value: "fixed")
    observation = np.zeros(42, dtype=np.float32)
    scene = {"metadata": {"scene_id": "legacy-physical", "seed": 9,
              "state_sha256": "fixed", "controller_sha256": "fixed",
              "raw_observation_sha256": __import__("hashlib").sha256(observation.tobytes()).hexdigest()},
             "policy_observation": observation[:39]}

    class Catalog:
        def restore(self, _index, _env):
            return _env, {"raw_observation": observation}

    class Adapter:
        def base_cond(self, queued): return queued
        def get_action(self, _condition): return torch.zeros((1, 16, 7))
        def map_action_ee_delta_pose(self, action): return action

    row = rollout.PlaceSpherePairedEvaluator(Adapter(), Catalog(), ".", "cpu").run_episode(0, scene)
    assert row["task_action_count"] == expected_steps
    assert row["result"]["elapsed_steps"] == expected_steps
    assert len(row["actions_native_pd_ee_delta_pose"]) == expected_steps
    assert env.closed


def test_progress_guard_annotations_follow_executed_task_actions_through_pause_and_terminal(
        monkeypatch):
    import hashlib
    import numpy as np
    import torch
    from types import SimpleNamespace
    from examples.baselines.constrain import evaluate_placesphere_paired as rollout

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.agent = SimpleNamespace(controller=SimpleNamespace(get_state=lambda: {}))
            self.obj = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0., 0., .1]])))
            self.bin = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[.3, 0., 0.]])))
            self._render_only_xy_trace = []
            self.closed = False
            self.steps = 0

        def reset(self, *args, **kwargs):
            return None

        def get_state_dict(self):
            return {}

        def step(self, _action):
            self.steps += 1
            paused = self.steps == 3
            success = self.steps == 10
            attempt = {"task_action_executed": not paused,
                       "progress_guard_event": "tangent_detour_pause" if paused else None,
                       "filtered_action": [0.] * 7}
            self._render_only_xy_last_attempt = attempt
            self._render_only_xy_trace.append(attempt)
            info = {key: torch.tensor([value]) for key, value in {
                "obstacle_collision": False, "obstacle_crossing": False,
                "is_obj_grasped": False, "native_success": success,
                "is_obj_on_bin": success, "is_obj_static": success}.items()}
            return torch.zeros(42), 0., success, False, info

        def close(self):
            self.closed = True

    env = Env()
    monkeypatch.setattr(rollout, "_make_env", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(rollout, "_tree_hash", lambda _value: "fixed")
    observation = np.zeros(42, dtype=np.float32)
    scene = {"metadata": {"scene_id": "guard-trace", "seed": 9,
              "state_sha256": "fixed", "controller_sha256": "fixed",
              "raw_observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest()},
             "policy_observation": observation[:39]}

    class Catalog:
        def restore(self, _index, _env):
            return _env, {"raw_observation": observation}

    class Adapter:
        def base_cond(self, queued): return queued
        def get_action(self, _condition): return torch.zeros((1, 16, 7))
        def map_action_ee_delta_pose(self, action): return action

    evaluator = rollout.PlaceSpherePairedEvaluator(Adapter(), Catalog(), ".", "cpu")
    evaluator.progress_guard = True
    row = evaluator.run_episode(0, scene)
    trace = env._render_only_xy_trace
    assert row["task_action_count"] == 9
    assert row["result"]["native_success"] is True
    assert [step["task_action_index"] for step in trace] == [1, 2, 3, 3, 4, 5, 6, 7, 8, 9]
    assert [step["scheduled_chunk_index"] for step in trace] == [0] * 9 + [1]
    assert [step["scheduled_chunk_offset"] for step in trace] == [0, 1, 2, 2, 3, 4, 5, 6, 7, 0]
    assert env.closed
