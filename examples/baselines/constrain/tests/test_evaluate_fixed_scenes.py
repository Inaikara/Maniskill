import pytest

from examples.baselines.constrain.evaluate_fixed_scenes import (
    classify_progress,
    episode_should_stop,
    summarize_fixed_results,
    update_step_flags,
)


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
