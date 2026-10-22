import pytest

from examples.baselines.constrain.evaluate_pickcube_paired import summarize_episodes


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
