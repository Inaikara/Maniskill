import pytest

from examples.baselines.constrain.evaluate_pickcube_paired import (
    classify_termination, summarize_episodes,
)


@pytest.mark.parametrize("terminated,truncated,steps,expected", [
    (False, True, 500, "timeout"),
    (False, True, 17, "timeout"),
    (True, False, 17, "environment_terminal"),
    (False, False, 500, "control_step_limit"),
])
def test_termination_classification(terminated, truncated, steps, expected):
    assert classify_termination(False, terminated, truncated, steps, 500) == expected


def test_native_success_precedes_same_step_truncation_and_unsafe_result():
    assert classify_termination(True, False, True, 500, 500) == "native_success"
    summary = summarize_episodes([{"status": "evaluated", "result": {
        "termination_reason": "native_success", "elapsed_steps": 500,
        "native_success": True, "safe_success": False,
        "obstacle_contact": True, "xy_forbidden_crossing": False,
        "phase": "task_success",
    }}], requested=1)
    assert summary["task_success_rate"] == 1.0
    assert summary["safe_success_rate"] == 0.0


def test_live_time_limit_flags_follow_the_same_classifier():
    gym = pytest.importorskip("gymnasium")
    from examples.baselines.constrain.evaluate_pickcube_paired import _native_env
    from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet

    catalog = PickCubeSceneSet("outputs/exp/20261022_152157_pickcube_scenes")
    env = gym.wrappers.TimeLimit(_native_env(catalog.load(0)), max_episode_steps=1)
    try:
        env.reset(seed=0)
        obs, _ = env.reset()
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        # The wrapper's real API flags enter the exact classifier used by rollout.
        assert truncated
        assert classify_termination(False, terminated, truncated, 1, 500) == "timeout"
    finally:
        env.close()


def test_runtime_errors_remain_in_paired_policy_denominator():
    summary = summarize_episodes([
        {"status": "evaluated", "result": {
            "native_success": True, "safe_success": False,
            "obstacle_contact": True, "xy_forbidden_crossing": False,
            "termination_reason": "native_success", "elapsed_steps": 31,
            "phase": "task_success",
        }},
        {"status": "runtime_error", "error": "native step failed"},
    ], requested=2)

    assert summary["requested_denominator"] == 2
    assert summary["task_success_rate"] == 0.5
    assert summary["safe_success_rate"] == 0.0
    assert summary["obstacle_contact_rate"] == 0.5
    assert summary["runtime_errors"] == 1
    assert summary["control_steps"] == 31
    assert summary["complete"] is False


def test_invalid_empty_denominator_is_rejected():
    with pytest.raises(ZeroDivisionError):
        summarize_episodes([], requested=0)


def test_timeout_step_limit_and_wall_timeout_have_distinct_reporting():
    summary = summarize_episodes([
        {"status": "evaluated", "result": {"native_success": False,
         "safe_success": False, "obstacle_contact": False,
         "xy_forbidden_crossing": False, "termination_reason": "timeout",
         "elapsed_steps": 12, "phase": "no_grasp"}},
        {"status": "evaluated", "result": {"native_success": False,
         "safe_success": False, "obstacle_contact": False,
         "xy_forbidden_crossing": False, "termination_reason": "control_step_limit",
         "elapsed_steps": 500, "phase": "no_grasp"}},
        {"status": "runtime_error", "error_kind": "wall_timeout"},
    ], requested=3)
    assert summary["requested_denominator"] == 3
    assert summary["timeout_count"] == 1
    assert summary["control_step_limit_count"] == 1
    assert summary["wall_timeout_runtime_errors"] == 1
    assert summary["runtime_errors"] == 1
