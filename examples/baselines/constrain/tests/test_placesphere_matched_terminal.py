"""Seam for the opt-in matched terminal. Legacy order and native thresholds stay fixed."""

import inspect

import numpy as np
import pytest

from examples.baselines.constrain.placesphere_matched_terminal import (
    ABOVE_RELEASE_MARGINS_M,
    NATIVE_XY_M,
    RADIUS_M,
    above_release_candidates,
    bounded_xy_actions,
    clearance_tcp_z,
    COLLECTION_READY_IFF,
    ROLE_NEGATIVE_DIRECT,
    ROLE_POSITIVE_DIRECT,
    ROLE_POSITIVE_OBSTACLE,
    ROLE_TERMINAL_REPLAY,
    collection_admission,
    confirmation_decision,
    direct_terminal_suffix_accepted,
    matched_terminal_accepted,
    negative_direct_admits,
    positive_direct_accepted,
    terminal_replay_accepted,
    pose_delta_action,
    retreat_after_release,
    select_terminal_candidate,
    select_representative,
    shared_matched_terminal,
    shared_release_tcp_z,
    side_offset_candidates,
    terminal_phase_plan,
)
from examples.baselines.constrain.placesphere_transport_detour import run_transport_expert


def test_legacy_terminal_order_and_default_are_unchanged():
    assert terminal_phase_plan("legacy") == (
        "container_approach", "retreat", "release", "settle")
    assert inspect.signature(run_transport_expert).parameters["terminal_mode"].default == "legacy"


def test_above_release_clears_the_measured_hand_and_does_not_widen_success():
    tcp_z = clearance_tcp_z(0.127, tcp_z=0.120, lowest_hand_vertex_z=0.095, margin_m=0.005)

    assert tcp_z == pytest.approx(0.157)
    rows = above_release_candidates(
        0.127, tcp_z=0.120, lowest_hand_vertex_z=0.095, transport_exit_z=0.120,
        margins=ABOVE_RELEASE_MARGINS_M)
    assert [row["margin_m"] for row in rows] == [0.005, 0.015, 0.030]
    assert rows[0]["release_tcp_z"] == pytest.approx(0.157)
    assert rows[0]["descends_below_transport_exit"] is False
    assert rows[0]["geometric_is_surface_clearance"] is False
    assert rows[0]["radius_m"] == RADIUS_M == 0.03
    assert rows[0]["native_xy_m"] == NATIVE_XY_M == 0.005
    assert rows[0]["native_thresholds_changed"] is False


def test_above_release_does_not_descend_when_transport_exit_already_clears():
    rows = above_release_candidates(
        0.10, tcp_z=0.16, lowest_hand_vertex_z=0.14, transport_exit_z=0.16)
    assert rows[0]["geometric_tcp_z"] == pytest.approx(0.125)
    assert rows[0]["release_tcp_z"] == pytest.approx(0.16)
    assert rows[0]["descends_below_transport_exit"] is False


def test_common_release_uses_both_measured_role_requirements_without_a_fixed_raise():
    assert shared_release_tcp_z(
        direct_required_z_m=0.1518737539972212,
        obstacle_required_z_m=0.15177056475995387,
        transport_exit_z_m=0.12,
    ) == pytest.approx(0.1518737539972212)


def test_retreat_stops_at_the_first_native_success_and_logs_the_plan():
    skipped = retreat_after_release(
        native_success_before_retreat=True, retreat_steps_executed=0, retreat_steps_planned=12)
    assert skipped["truncated_after_native_success"] is True
    assert skipped["retreat_steps_executed"] == 0
    assert skipped["continued_after_first_native_success"] is False
    assert skipped["retreat_required"] is True

    partial = retreat_after_release(
        native_success_before_retreat=False, retreat_steps_executed=3, retreat_steps_planned=12,
        native_success_during_retreat=True)
    assert partial["truncated_after_native_success"] is True
    assert partial["retreat_steps_executed"] == 3
    assert partial["continued_after_first_native_success"] is False


def test_xy_perturbation_is_a_bounded_action_not_a_state_jump():
    actions = bounded_xy_actions([0.02, -0.01])
    assert len(actions) == 1
    assert actions[0].shape == (7,)
    assert float(actions[0][0]) == pytest.approx(0.2)
    assert float(actions[0][1]) == pytest.approx(-0.1)
    assert float(actions[0][2]) == 0.0
    assert float(actions[0][6]) < 0
    integrated = np.sum([np.asarray(row[:2], dtype=np.float64) * 0.1 for row in actions], axis=0)
    assert np.allclose(integrated, [0.02, -0.01])
    assert pose_delta_action([0.01, 0.0, 0.0], -1.0)[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        bounded_xy_actions([0.03, 0.0])


def test_selection_keeps_the_lowest_margin_and_rejects_contact_or_loosened_thresholds():
    def row(margin, *, success=True, contact=False, xy=0.005):
        return {
            "margin_m": margin, "native_success": success, "on_bin": success, "static": success,
            "ungrasped": success, "obstacle_contact": contact, "obstacle_crossing": False,
            "native_xy_m": xy, "radius_m": 0.03, "native_thresholds_changed": xy != 0.005,
            "closed_descent_below_clearance": False, "from_execution_start": True,
        }

    chosen = select_terminal_candidate([
        row(0.030), row(0.005, contact=True), row(0.015),
    ])
    assert chosen["selected_margin_m"] == 0.015
    assert chosen["side_offset_authorized"] is False
    blocked = select_terminal_candidate([row(0.005, success=False), row(0.015, contact=True), row(0.030, xy=0.02)])
    assert blocked["selected_margin_m"] is None
    assert blocked["side_offset_authorized"] is True
    assert blocked["collection_ready"] is False


def test_side_offset_is_finite_and_rejected_inside_the_column():
    rows = side_offset_candidates(
        bin_xy=[0.10, 0.0], cylinder_xy=[0.0, 0.0], hand_xy_extent_m=0.04, radius_m=0.03)
    assert len(rows) == 3
    assert {row["heading_rad"] for row in rows} == {0.0, 0.6, -0.6}
    assert all(row["radius_m"] == 0.03 for row in rows)
    inside = side_offset_candidates(
        bin_xy=[0.02, 0.0], cylinder_xy=[0.0, 0.0], hand_xy_extent_m=0.04, radius_m=0.03,
        offset_m=0.01)
    assert all(row["column_clear"] is False for row in inside)
    assert all(row["executable"] is False for row in inside)


def test_direct_and_obstacle_share_the_new_terminal_and_still_require_native_release():
    targets = {
        "lift_xyz": [0.0, 0.0, 0.12], "transport_exit_xyz": [0.1, 0.0, 0.12],
        "clearance_xyz": [0.1, 0.0, 0.16], "release_tcp_z": 0.16, "retreat_xyz": [0.1, 0.0, 0.24],
    }
    direct = {"targets": dict(targets), "waypoint_xyz": None, "native_success": True,
              "on_bin": True, "static": True, "ungrasped": True, "radius_m": 0.03}
    obstacle = {"targets": dict(targets), "waypoint_xyz": [0.0, 0.2, 0.12], "native_success": True,
                "on_bin": True, "static": True, "ungrasped": True, "radius_m": 0.03}
    ok, reason = shared_matched_terminal(direct, obstacle)
    assert ok and reason is None
    obstacle["ungrasped"] = False
    ok, reason = shared_matched_terminal(direct, obstacle)
    assert not ok and reason == "native_success_not_released"


def _positive_obstacle(**overrides):
    record = {
        "role": ROLE_POSITIVE_OBSTACLE, "native_success": True, "on_bin": True, "static": True,
        "ungrasped": True, "obstacle_contact": False, "obstacle_crossing": False,
        "transport_contact": False, "terminal_contact": False, "release_reached": True,
        "retreat_required": True, "truncated_after_native_success": True, "retreat_reached": False,
        "continued_after_first_native_success": False, "radius_m": 0.03, "native_xy_m": 0.005,
        "from_execution_start": True, "from_execution_start_safe": True,
        "prefix_latch_cleared_as_proof": False, "cylinder_parked": False, "actual_cylinder": True,
        "obstacle_relocated": False, "contact_checks_ran": True, "path_uses_waypoint": True,
        "action_width": 7, "native_thresholds_changed": False,
        "measured_geometric_release_tcp_z_m": 0.152,
    }
    record.update(overrides)
    return record


def test_acceptance_allows_logged_retreat_truncation_and_rejects_a_missing_release():
    assert matched_terminal_accepted(_positive_obstacle())
    blocked = _positive_obstacle(release_reached=False)
    assert matched_terminal_accepted(blocked) is False


def test_terminal_suffix_must_complete_its_own_feedback_to_first_native_success():
    assert terminal_replay_accepted(_replay(
        fixed_suffix_complete=False, fixed_suffix_action_count=11,
        feedback_settle_steps=5, first_native_success_step=16,
        terminal_feedback_completed=True, claim_scope="suffix_only"))
    assert terminal_replay_accepted(_replay(terminal_feedback_completed=False)) is False
    assert terminal_replay_accepted(_replay(feedback_settle_steps=801)) is False


def test_transport_contact_suffix_is_not_admitted():
    """Round-18 confirmation placesphere-152: native success with transport contact."""
    record = {
        "native_success": True, "on_bin": True, "static": True, "ungrasped": True,
        "transport_contact": True, "terminal_contact": False, "terminal_crossing": False,
        "obstacle_contact": True, "obstacle_crossing": False,
        "release_reached": True, "continued_after_first_native_success": False,
        "radius_m": 0.03, "native_xy_m": 0.005, "from_execution_start_safe": False,
        "cylinder_parked": False, "actual_cylinder": True, "role": "negative_direct",
    }
    assert direct_terminal_suffix_accepted(record) is False
    assert matched_terminal_accepted(record) is False


def _positive_direct(**overrides):
    record = {
        "role": ROLE_POSITIVE_DIRECT, "native_success": True, "on_bin": True, "static": True,
        "ungrasped": True, "obstacle_contact": False, "obstacle_crossing": False,
        "transport_contact": False, "terminal_contact": False, "release_reached": True,
        "continued_after_first_native_success": False, "radius_m": 0.03, "native_xy_m": 0.005,
        "from_execution_start": True, "from_execution_start_safe": True,
        "prefix_latch_cleared_as_proof": False, "cylinder_parked": True, "actual_cylinder": False,
        "obstacle_relocated": False, "planned_cylinder_recorded": True,
        "release_height_from_planned_geometry": True, "planned_cylinder_contact_proof": False,
        "path_uses_waypoint": False, "action_width": 7, "native_thresholds_changed": False,
        "release_tcp_z": 0.152, "waypoint_xyz": None,
        "measured_geometric_release_tcp_z_m": 0.152,
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.12], "transport_exit_xyz": [0.1, 0.0, 0.12],
            "clearance_xyz": [0.1, 0.0, 0.152], "release_tcp_z": 0.152, "retreat_xyz": [0.1, 0.0, 0.232],
        },
    }
    record.update(overrides)
    return record


def _replay(**overrides):
    record = {
        "role": ROLE_TERMINAL_REPLAY, "actual_cylinder": True, "cylinder_parked": False,
        "prefix_latches_labeled": True, "prefix_latch_cleared_as_proof": False,
        "full_prefix_safe": False, "native_success": True, "on_bin": True, "static": True,
        "terminal_feedback_completed": True, "suffix_contact_free": True,
        "feedback_settle_steps": 0,
        "ungrasped": True, "terminal_contact": False, "terminal_crossing": False,
        "radius_m": 0.03, "native_xy_m": 0.005, "native_thresholds_changed": False,
        "action_width": 7, "same_planned_geometry": True, "release_tcp_z": 0.152,
    }
    record.update(overrides)
    return record


def _negative_transport_block(**overrides):
    record = {
        "role": ROLE_NEGATIVE_DIRECT, "actual_cylinder": True,
        "blocked_in_transport": True, "native_success": False, "admits_demos": False,
    }
    record.update(overrides)
    return record


def test_blocking_cylinder_direct_and_cleared_latch_do_not_admit():
    blocking = _positive_direct(
        role=ROLE_NEGATIVE_DIRECT, cylinder_parked=False, actual_cylinder=True,
        transport_contact=True, obstacle_contact=True, from_execution_start_safe=False,
        native_success=False)
    assert direct_terminal_suffix_accepted(blocking) is False
    assert negative_direct_admits(blocking) is False
    cleared = _positive_obstacle(prefix_latch_cleared_as_proof=True, from_execution_start_safe=True)
    assert matched_terminal_accepted(cleared) is False
    parked_missing = _positive_direct(cylinder_parked=False, actual_cylinder=True)
    assert positive_direct_accepted(parked_missing) is False
    claimed = _replay(full_prefix_safe=True, prefix_latch_cleared_as_proof=True)
    assert terminal_replay_accepted(claimed) is False


def test_collection_admits_only_the_paired_safe_roles():
    direct = _positive_direct()
    obstacle = _positive_obstacle(
        release_tcp_z=0.152, waypoint_xyz=[0.0, 0.2, 0.12], targets=dict(direct["targets"]))
    replay = _replay()
    decision = collection_admission(direct, obstacle, replay, _negative_transport_block())
    assert decision["collection_ready"] is True
    assert decision["negative_admits_demos"] is False
    mismatch = collection_admission(direct, obstacle, _replay(release_tcp_z=0.2), None)
    assert mismatch["collection_ready"] is False
    assert "release_tcp_z_mismatch" in mismatch["reasons"]
    no_negative = collection_admission(direct, obstacle, replay, None)
    assert "negative_transport_block_not_proven" in no_negative["reasons"]
    obstacle_only = confirmation_decision([
        {"obstacle_accepted": True, "positive_obstacle": _positive_obstacle(release_tcp_z=0.152,
         waypoint_xyz=[0.0, 0.2, 0.12], targets=dict(direct["targets"]))}
        for _ in range(16)
    ])
    assert obstacle_only["collection_ready"] is False
    assert COLLECTION_READY_IFF.startswith("every confirmation layout admits only when positive_direct")


def test_collection_rejects_a_shared_release_below_either_measured_role_clearance():
    shared_z = 0.15177056475995387
    direct = _positive_direct(
        release_tcp_z=shared_z,
        measured_geometric_release_tcp_z_m=0.1518737539972212,
        targets={
            "lift_xyz": [0.0, 0.0, 0.12], "transport_exit_xyz": [0.1, 0.0, 0.12],
            "clearance_xyz": [0.1, 0.0, shared_z], "release_tcp_z": shared_z,
            "retreat_xyz": [0.1, 0.0, shared_z + 0.08],
        })
    obstacle = _positive_obstacle(
        release_tcp_z=shared_z,
        measured_geometric_release_tcp_z_m=shared_z,
        waypoint_xyz=[0.0, 0.2, 0.12], targets=dict(direct["targets"]))
    replay = _replay(release_tcp_z=shared_z)

    decision = collection_admission(direct, obstacle, replay)

    assert decision["collection_ready"] is False
    assert "shared_release_below_measured_clearance" in decision["reasons"]


def test_representative_subset_is_deterministic_and_excludes_dev_seeds():
    rows = []
    for index in range(30):
        rows.append({
            "source": "a" if index < 10 else "b", "index": index, "seed": 1000 + index,
            "height_m": 0.127 if index % 5 else 0.20,
            "distance_m": 0.08 + 0.01 * (index % 7),
        })
    first = select_representative(rows, count=12, excluded_seeds={1001, 1002}, seed=7)
    second = select_representative(rows, count=12, excluded_seeds={1001, 1002}, seed=7)
    assert first == second
    assert len(first) == 12
    assert all(row["seed"] not in {1001, 1002} for row in first)
    assert {row["source"] for row in first} == {"a", "b"}
