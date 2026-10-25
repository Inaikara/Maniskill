"""Seams for the frozen 90k closed-loop contact diagnosis."""

import json

import numpy as np
import pytest

from examples.baselines.constrain.pickcube_closed_loop_diagnosis import (
    CONDITION_DRIFT_M,
    CONTACT_INDICES,
    EXPOSURE_CKPT_SHA,
    ISOLATION_COUNT,
    PREDECLARED_STRIDE,
    STALE_QUEUE_INDEX_MIN,
    assert_reproduction,
    build_diagnosis_index,
    build_diagnosis_preregistration,
    classify_contact_mechanism,
    condition_drift_m,
    decide_execution_factor,
    evaluator_contract,
    kinematic_queue_endpoint,
    main,
    queue_heads_toward_obstacle,
)


def _obs(tcp_xy, obstacle_xy=(0.0, 0.1), radius=0.03):
    obs = np.zeros(45, dtype=np.float32)
    obs[19:21] = tcp_xy
    obs[42:44] = obstacle_xy
    obs[44] = radius
    return obs


def _row(**overrides):
    row = {
        "queue_index": 6,
        "condition_drift_m": 0.08,
        "clearance_drop_m": 0.04,
        "expert_tcp_distance_m": 0.05,
        "expert_physx_contact": False,
        "queue_heads_toward_obstacle": True,
        "is_surface_clearance": False,
        "evaluator_mismatch": False,
    }
    row.update(overrides)
    return row


def _episode(index=3, mechanism_row=None, matched=True):
    contact = mechanism_row or _row()
    return {
        "scene_index": index,
        "scene_id": f"pickcube-{index:03d}",
        "reproduction": {"matched": matched},
        "is_surface_clearance": False,
        "steps": [{"step": 1, "queue_index": 0}],
        "contact": {
            "step": 37,
            "queue_index": contact["queue_index"],
            "entities": ["panda_hand"],
            "tcp_xyz": [0.1, 0.2, 0.3],
            "cube_xyz": [0.1, 0.2, 0.04],
            "grasp_stage": "transport",
            "condition_drift_m": contact["condition_drift_m"],
            **contact,
        },
        "evaluator_mismatch": contact["evaluator_mismatch"],
    }


def test_condition_drift_is_unnormalized_relative_xy_meters():
    replan = _obs((0.0, 0.0))
    current = _obs((0.03, 0.04))
    assert condition_drift_m(replan, current) == pytest.approx(0.05)
    assert condition_drift_m(replan, replan) == 0.0


def test_kinematic_endpoint_sums_position_actions_and_is_not_a_pose():
    endpoint = kinematic_queue_endpoint(np.array([0.0, 0.1]), np.array([[1.0, 0.0, 0.0, 0, 0, 0, 0], [0.0, -0.5, 0.0, 0, 0, 0, 1]]))
    assert endpoint["xy"] == pytest.approx([0.1, 0.05])
    assert endpoint["is_executed_pose"] is False
    assert endpoint["is_surface_clearance"] is False
    assert endpoint["pos_scale_m"] == 0.1


def test_queue_heading_uses_geometry_clearance_not_surface():
    assert queue_heads_toward_obstacle(0.04, 0.01) is True
    assert queue_heads_toward_obstacle(0.04, 0.04) is False


def test_staleness_requires_late_queue_drift_and_a_left_expert_path():
    assert classify_contact_mechanism(_row())["mechanism"] == "queued_overshoot"
    assert classify_contact_mechanism(_row(queue_index=STALE_QUEUE_INDEX_MIN - 1))["mechanism"] != "queued_overshoot"
    assert classify_contact_mechanism(_row(condition_drift_m=CONDITION_DRIFT_M - 0.01))["mechanism"] != "queued_overshoot"
    assert classify_contact_mechanism(_row(expert_tcp_distance_m=0.01))["mechanism"] == "narrow_tracking"
    assert classify_contact_mechanism(_row(queue_index=0, condition_drift_m=0.0, expert_tcp_distance_m=0.05, queue_heads_toward_obstacle=False, clearance_drop_m=0.0))["mechanism"] == "fresh_error"
    with pytest.raises(ValueError, match="surface"):
        classify_contact_mechanism(_row(is_surface_clearance=True))


def test_execution_factor_is_stride_2_only_when_staleness_is_isolated():
    stale = [_episode(index, _row()) for index in CONTACT_INDICES]
    decision = decide_execution_factor(stale)
    assert decision["trial"] is True
    assert decision["stride"] == PREDECLARED_STRIDE == 2
    assert decision["comparator_stride"] == 8
    assert decision["swept"] is False
    assert decision["seed_rule_changes"] is True
    mixed = [_episode(index, _row(queue_index=1, condition_drift_m=0.0, clearance_drop_m=0.0, queue_heads_toward_obstacle=False)) for index in CONTACT_INDICES]
    refused = decide_execution_factor(mixed)
    assert refused["trial"] is False and refused["stride"] is None
    broken = [_episode(index, _row(evaluator_mismatch=True)) for index in CONTACT_INDICES]
    mismatch = decide_execution_factor(broken)
    assert mismatch["trial"] is False and mismatch["reason"] == "evaluator_mismatch"
    assert ISOLATION_COUNT == 8


def test_reproduction_requires_the_published_contact_identity():
    published = {
        "scene_index": 3, "scene_id": "pickcube-003", "seed": 2026107010,
        "first_contact": {"step": 37, "phase": "transport", "link": "panda_hand"},
        "result": {"native_success": True, "obstacle_contact": True, "xy_forbidden_crossing": False,
                   "safe_success": False, "elapsed_steps": 102, "termination_reason": "native_success"},
    }
    traced = dict(published)
    assert_reproduction(traced, published)
    changed = json.loads(json.dumps(traced))
    changed["first_contact"]["step"] = 38
    with pytest.raises(ValueError, match="reproduction"):
        assert_reproduction(changed, published)


def test_evaluator_contract_rejects_width_clip_and_late_success():
    ok = evaluator_contract({
        "action_shape": [8, 7], "action_abs_max": 1.0, "action_clipped": False,
        "success_step": 102, "elapsed_steps": 102, "native_success": True,
        "controller_matches": True, "obs_tcp_error_m": 0.0,
        "normalization_matches": True, "executed_slice": [1, 9],
    })
    assert ok["passed"] is True and ok["action_clipped"] is False
    bad = dict(ok)
    bad["passed"] = False
    for broken in (
        {"action_shape": [8, 4]},
        {"action_abs_max": 1.2, "action_clipped": True},
        {"success_step": 100, "elapsed_steps": 108, "native_success": True},
        {"controller_matches": False},
        {"executed_slice": [0, 8]},
    ):
        payload = {
            "action_shape": [8, 7], "action_abs_max": 1.0, "action_clipped": False,
            "success_step": 102, "elapsed_steps": 102, "native_success": True,
            "controller_matches": True, "obs_tcp_error_m": 0.0,
            "normalization_matches": True, "executed_slice": [1, 9],
        }
        payload.update(broken)
        report = evaluator_contract(payload)
        assert report["passed"] is False


def test_index_requires_fifteen_reproduced_contact_traces():
    episodes = [_episode(index) for index in CONTACT_INDICES]
    protocol = {"stale_queue_index_min": STALE_QUEUE_INDEX_MIN, "condition_drift_m": CONDITION_DRIFT_M}
    index = build_diagnosis_index(episodes, protocol, {"files_sha256": "abc"}, {"passed": True})
    assert index["prose_only"] is False
    assert index["formal_performance_claim"] is False
    assert index["checkpoint_sha256"] == EXPOSURE_CKPT_SHA
    assert index["vertex_gap_is_surface_clearance"] is False
    assert index["contact_count"] == 15
    short = episodes[:-1]
    with pytest.raises(ValueError, match="15"):
        build_diagnosis_index(short, protocol, {"files_sha256": "abc"}, {"passed": True})
    unmatched = json.loads(json.dumps(episodes))
    unmatched[0]["reproduction"]["matched"] = False
    with pytest.raises(ValueError, match="reproduction"):
        build_diagnosis_index(unmatched, protocol, {"files_sha256": "abc"}, {"passed": True})
    surface = json.loads(json.dumps(episodes))
    surface[0]["is_surface_clearance"] = True
    with pytest.raises(ValueError, match="surface"):
        build_diagnosis_index(surface, protocol, {"files_sha256": "abc"}, {"passed": True})


def test_preregistration_freezes_the_90k_checkpoint_and_forbids_training():
    payload = build_diagnosis_preregistration({"exp": "outputs/exp/x", "diagnosis": "outputs/exp/x/diagnosis"})
    assert payload["kind"] == "authorized_closed_loop_diagnosis"
    assert payload["no_collection"] is True and payload["no_training"] is True
    assert payload["formal_performance_claim"] is False
    assert payload["checkpoint_sha256"] == EXPOSURE_CKPT_SHA
    assert payload["contact_indices"] == list(CONTACT_INDICES)
    assert payload["protocol"]["predeclared_stride"] == 2
    assert payload["protocol"]["comparator_stride"] == 8
    assert payload["trial_authorized_by_this_file"] is False


def test_diagnose_refuses_identity_mismatch_before_writing(tmp_path):
    payload = build_diagnosis_preregistration({"exp": str(tmp_path), "diagnosis": str(tmp_path / "diagnosis")})
    payload["source_identity"] = {"files_sha256": "not-the-running-tree"}
    path = tmp_path / "preregistration.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="source identity"):
        main(["diagnose", "--registration", str(path), "--device", "cpu"])
    assert not (tmp_path / "diagnosis").exists()
