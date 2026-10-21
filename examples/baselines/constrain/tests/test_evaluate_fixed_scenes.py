import pytest
import numpy as np
import torch
from types import SimpleNamespace

from examples.baselines.constrain import evaluate_fixed_scenes as fixed_eval
from examples.baselines.constrain.evaluate_fixed_scenes import (
    FixedSceneEvaluator,
    _checkpoint_identity,
    classify_progress,
    episode_should_stop,
    summarize_fixed_results,
    update_step_flags,
)


def test_resume_checkpoint_identity_hashes_adapter_weight_file(tmp_path):
    import hashlib

    from types import SimpleNamespace

    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"first weights")
    adapter = SimpleNamespace(metadata={"checkpoint": str(checkpoint)})
    first = _checkpoint_identity(adapter)
    assert first == hashlib.sha256(b"first weights").hexdigest()
    checkpoint.write_bytes(b"different weights")
    assert _checkpoint_identity(adapter) != first


class _RolloutAdapter:
    def __init__(self):
        self.inferences = []
        self.histories = []

    def base_cond(self, history):
        self.histories.append(history.detach().cpu().numpy().copy())
        return history

    def get_action(self, condition):
        self.inferences.append(1)
        return torch.zeros(1, 8, 4)

    def map_action_ee_delta_pose(self, actions):
        return torch.zeros(1, 8, 7)


class _RolloutEnv:
    def __init__(self, step_infos, initial_obs):
        self.step_infos = iter(step_infos)
        self.initial_obs = np.asarray(initial_obs, dtype=np.float32)
        self.step_calls = 0
        self.cubeA = SimpleNamespace(pose=SimpleNamespace(p=np.array([[0., 0., 0.1]])))
        self.cubeB = SimpleNamespace(pose=SimpleNamespace(p=np.array([[0.2, 0., 0.1]])))
        self.unwrapped = self
        self.closed = False

    def get_obs(self):
        return self.initial_obs.copy()

    def close(self):
        self.closed = True

    def step(self, action):
        self.step_calls += 1
        info = next(self.step_infos, None)
        assert info is not None, "rollout took more actions than its finite script"
        return self.initial_obs.copy(), 0., False, False, info


def _scene(index, radius=0.04, seed=100):
    return SimpleNamespace(
        scene_id=f"scene-{index}", initial_observation=np.full(51, index, dtype=np.float32),
        metadata={"seed": seed + index, "initial_state_sha256": f"initial-{index}",
                  "cylinder": {"radius": radius}},
    )


def _step_info(*, stack=False, collision=False, crossing=False):
    return {"stack_success": stack, "obstacle_collision": collision,
            "obstacle_crossing": crossing, "is_cubeA_grasped": False}


def _run_one_fixture(monkeypatch, tmp_path, env, *, max_steps):
    adapter = _RolloutAdapter()
    evaluator = FixedSceneEvaluator(adapter, None, "stub", tmp_path, device="cpu",
                                    max_steps=max_steps)
    evaluator.env = env
    evaluator._env_radius = 0.04
    restored = []
    monkeypatch.setattr(fixed_eval, "restore_scene", lambda e, scene: (
        restored.append(scene.scene_id), setattr(e, "initial_obs", scene.initial_observation.copy())
    ))
    monkeypatch.setattr(fixed_eval, "current_state_snapshot", lambda e: {"scene": e.initial_obs[0]})
    return evaluator, adapter, restored


def test_run_one_stops_inside_action_chunk_on_same_step_stack_violation(monkeypatch, tmp_path):
    infos = [_step_info(crossing=True), _step_info(),
             _step_info(stack=True, collision=True)]
    env = _RolloutEnv(infos, np.zeros(51))
    evaluator, adapter, _ = _run_one_fixture(monkeypatch, tmp_path, env, max_steps=10)

    record = evaluator.run_one(0, _scene(0))
    result = record["result"]
    assert env.step_calls == 3
    assert len(adapter.inferences) == 1
    assert result["elapsed_steps"] == 3
    assert result["termination_reason"] == "stack_success"
    assert result["stack_success"] is True
    assert result["obstacle_collision"] is True
    assert result["obstacle_crossing"] is True
    assert result["safe_success"] is False


def test_run_one_caps_partial_final_chunk_and_resets_history_between_episodes(
    monkeypatch, tmp_path
):
    env = _RolloutEnv([_step_info() for _ in range(20)], np.zeros(51))
    evaluator, adapter, restored = _run_one_fixture(monkeypatch, tmp_path, env, max_steps=10)
    first = evaluator.run_one(0, _scene(0))
    second = evaluator.run_one(1, _scene(1))

    assert env.step_calls == 20
    assert len(adapter.inferences) == 4  # two 8-action chunks per episode
    assert [first["result"]["elapsed_steps"], second["result"]["elapsed_steps"]] == [10, 10]
    assert first["result"]["termination_reason"] == "timeout"
    assert second["result"]["termination_reason"] == "timeout"
    assert restored == ["scene-0", "scene-1"]
    assert np.all(adapter.histories[0] == 0)
    assert np.all(adapter.histories[2] == 1)


def test_run_one_rebuilds_environment_when_scene_radius_changes(monkeypatch, tmp_path):
    first_env = _RolloutEnv([_step_info()], np.zeros(51))
    evaluator, _, restored = _run_one_fixture(monkeypatch, tmp_path, first_env, max_steps=1)
    second_env = _RolloutEnv([_step_info()], np.ones(51))
    created = []

    def make_env(scene):
        created.append(scene.metadata["cylinder"]["radius"])
        return second_env

    evaluator._make_env = make_env
    evaluator.run_one(0, _scene(0, radius=0.04))
    evaluator.run_one(1, _scene(1, radius=0.06))

    assert first_env.closed is True
    assert created == [0.06]
    assert second_env.step_calls == 1
    assert restored == ["scene-0", "scene-1"]


def test_progress_failure_bins_are_disjoint_and_stack_can_be_unsafe():
    rows = [
        dict(stack_success=True, safe_success=False, ever_grasped=True,
             ever_lifted=True, ever_transported=True, obstacle_collision=True,
             obstacle_crossing=False, termination_reason="stack_success"),
        dict(stack_success=False, ever_grasped=False, ever_lifted=False,
             ever_transported=False, safe_success=False, obstacle_collision=False,
             obstacle_crossing=False, termination_reason="timeout"),
        dict(stack_success=False, ever_grasped=True, ever_lifted=False,
             ever_transported=False, safe_success=False, obstacle_collision=False,
             obstacle_crossing=False, termination_reason="timeout"),
        dict(stack_success=False, ever_grasped=True, ever_lifted=True,
             ever_transported=False, safe_success=False, obstacle_collision=False,
             obstacle_crossing=False, termination_reason="timeout"),
    ]
    assert [classify_progress(row) for row in rows] == [
        "stacked_unsafe", "no_grasp", "grasp_no_lift", "lift_transport_or_place_failed"
    ]
    summary = summarize_fixed_results(rows, requested=4)
    assert summary["denominator"] == 4
    assert summary["stack_success_count"] == 1
    assert summary["safe_success_count"] == 0
    assert summary["progress_bins"] == {
        "stacked_unsafe": 1, "no_grasp": 1, "grasp_no_lift": 1,
        "lift_transport_or_place_failed": 1,
    }


def test_incomplete_or_failed_episodes_never_shrink_requested_denominator():
    summary = summarize_fixed_results(
        [dict(status="evaluated", stack_success=False, safe_success=False,
              obstacle_collision=False, obstacle_crossing=False,
              termination_reason="max_steps", ever_grasped=False,
              ever_lifted=False, ever_transported=False),
         dict(status="runtime_error", error="boom")], requested=2
    )
    assert summary["requested"] == 2
    assert summary["evaluated"] == 1
    assert summary["runtime_errors"] == 1
    assert summary["complete"] is False


def test_progress_classification_requires_explicit_task_state():
    with pytest.raises(KeyError):
        classify_progress({"stack_success": False})


def test_same_step_stack_and_safety_violation_is_stacked_but_unsafe():
    flags = update_step_flags(
        {"stack_success": False, "obstacle_collision": False,
         "obstacle_crossing": False},
        {"stack_success": True, "obstacle_collision": True,
         "obstacle_crossing": False},
    )
    assert flags == {"stack_success": True, "obstacle_collision": True,
                     "obstacle_crossing": False}
    row = dict(**flags, safe_success=(flags["stack_success"]
                                     and not flags["obstacle_collision"]
                                     and not flags["obstacle_crossing"]),
               ever_grasped=True, ever_lifted=True, ever_transported=True)
    assert row["stack_success"] is True
    assert classify_progress(row) == "stacked_unsafe"
    assert episode_should_stop(flags, False, False, 37, 500)


def test_episode_continues_until_stack_or_500_step_cap():
    flags = {"stack_success": False, "obstacle_collision": False,
             "obstacle_crossing": False}
    assert not episode_should_stop(flags, False, False, 499, 500)
    assert episode_should_stop(flags, False, False, 500, 500)
