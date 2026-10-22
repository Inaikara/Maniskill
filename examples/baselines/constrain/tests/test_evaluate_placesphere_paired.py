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
