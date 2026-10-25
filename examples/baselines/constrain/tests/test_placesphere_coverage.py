"""Seams for PlaceSphere contact coverage against the 300 expert windows."""

import numpy as np
import pytest

from examples.baselines.constrain.placesphere_coverage import (
    component_distances,
    contact_identity,
    coverage_decision,
    decision_window,
    expert_training_window,
    index_queries,
    leave_one_scene_out,
    nearest_coverage,
    quat_geodesic,
    traces_have_full_observations,
)


def _state(**updates):
    state = np.zeros(42, dtype=np.float64)
    state[22] = 1.0
    state[32] = 1.0
    state[19:22] = [0.1, 0.2, 0.3]
    state[26:29] = [0.4, 0.5, 0.05]
    state[29:32] = [0.4, 0.5, 0.06]
    state[39:42] = [0.0, 0.0, 0.03]
    state.update(updates) if False else None
    for key, value in updates.items():
        state[key] = value
    return state


def _window(episode_id, tcp_z=0.3, grip=1.0, delta_z=0.1, phase_shift=0.0):
    latest = _state()
    latest[18] = 1.0
    latest[19:22] = [0.1, 0.2, tcp_z]
    latest[29:32] = [0.4 + phase_shift, 0.5, 0.06]
    previous = latest.copy()
    action = np.zeros(7, dtype=np.float64)
    action[2] = delta_z
    action[6] = grip
    return {
        "episode_id": episode_id,
        "split": "train",
        "latest": latest,
        "action": action,
        "policy_phase": "release",
        "state_phase": "held_near_bin",
    }


def test_diagnosis_traces_without_a_decision_window_are_not_full_observations():
    episodes = [{"scene_index": 1, "first_contact": {"step": 90}, "result": {"obstacle_contact": True}}]
    assert traces_have_full_observations(episodes) is False
    complete = [{
        "decision_window": {
            "frames": 2,
            "policy_condition_shape": [2, 39],
            "latest_state_index": 1,
            "future_frames_used": 0,
        }
    }]
    assert traces_have_full_observations(complete) is True


def test_decision_window_uses_both_frames_and_ignores_future_rows():
    obs = np.zeros((6, 42), dtype=np.float64)
    obs[0, 19:22] = [0.0, 0.0, 0.20]
    obs[1, 19:22] = [0.0, 0.0, 0.21]
    obs[1, 18] = 1.0
    obs[2:, 19:22] = [9.0, 9.0, 9.0]
    obs[3, 18] = 0.0
    actions = np.zeros((5, 7), dtype=np.float64)
    actions[0, 2] = -0.1
    actions[0, 6] = 1.0
    actions[1:, 6] = -1.0
    record = decision_window(obs, actions, start=0)
    assert record["frames"] == 2
    assert record["raw_dim"] == 42
    assert record["policy_condition_shape"] == [2, 39]
    assert record["policy_condition_dim"] == 78
    assert record["latest_state_index"] == 1
    assert record["future_frames_used"] == 0
    assert record["latest_state"][19:22] == pytest.approx([0.0, 0.0, 0.21])
    assert record["delta_z"] == pytest.approx(-0.1)
    assert record["grip"] == pytest.approx(1.0)
    assert record["policy_phase"] == "release"
    obs[4, 0] = 7.0
    actions[3, 2] = 0.35
    again = decision_window(obs, actions, start=0)
    assert again["latest_state"][19:22] == pytest.approx([0.0, 0.0, 0.21])
    assert again["delta_z"] == pytest.approx(-0.1)


def test_xy_within_3cm_is_not_labeled_3d_and_z_gap_is_3d():
    left = _state()
    right = _state()
    right[19] = left[19] + 0.02
    flat = component_distances(left, right)
    assert flat["xy_only_m"] == pytest.approx(0.02)
    assert flat["position_3d_m"] == pytest.approx(0.02)
    assert flat["xy_within_3cm"] is True
    assert flat["xy_only_is_3d"] is False
    assert flat["xy_within_3cm_is_3d_close"] is False
    assert flat["is_3d_close"] is False
    raised = _state()
    raised[21] = left[21] + 0.1
    height = component_distances(left, raised)
    assert height["xy_only_m"] == pytest.approx(0.0)
    assert height["position_3d_m"] == pytest.approx(0.1)
    assert height["xy_only_is_3d"] is False
    assert height["is_3d_close"] is False


def test_quaternion_geodesic_is_pi_for_opposite_z_and_zero_for_double_cover():
    assert quat_geodesic([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]) == pytest.approx(np.pi)
    assert quat_geodesic([1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_leave_one_scene_out_radius_comes_from_train_windows_not_an_arbitrary_bar():
    scales = {name: 1.0 for name in (
        "position_3d", "object_position_3d", "rotation", "proprio_qpos", "proprio_qvel", "normalized_hint")}
    near = [_window(1000, tcp_z=0.30), _window(1001, tcp_z=0.30)]
    report = leave_one_scene_out(near, scales, quantile=0.95)
    assert report["radius"] == pytest.approx(0.0)
    assert report["position_3d_radius_m"] == pytest.approx(0.0)
    assert report["uses_dev_labels"] is False
    assert report["reward_claim"] is False
    assert report["quantile"] == 0.95
    assert report["source"] == "train_leave_one_scene_out_same_policy_phase"
    apart = [_window(1000, tcp_z=0.30), _window(1001, tcp_z=0.40)]
    apart_report = leave_one_scene_out(apart, scales, quantile=0.95)
    assert apart_report["radius"] == pytest.approx(0.1)
    assert apart_report["position_3d_radius_m"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="300 obstacle"):
        leave_one_scene_out([_window(5)], scales)
    dev = _window(1000)
    dev["split"] = "dev"
    with pytest.raises(ValueError, match="dev"):
        leave_one_scene_out([dev, _window(1001)], scales)


def test_nearest_neighbor_reports_action_grip_and_delta_z_without_a_reward_claim():
    scales = {name: 1.0 for name in (
        "position_3d", "object_position_3d", "rotation", "proprio_qpos", "proprio_qvel", "normalized_hint")}
    query = _window(None, tcp_z=0.30, grip=-1.0, delta_z=-0.35)
    windows = [
        _window(1000, tcp_z=0.30, grip=1.0, delta_z=0.2),
        _window(1001, tcp_z=0.30, grip=-1.0, delta_z=-0.2),
        _window(1002, tcp_z=1.30, grip=1.0, delta_z=0.2),
    ]
    hit = nearest_coverage(query, windows, scales, radius=0.05, position_bar=0.05)
    assert hit["reward_claim"] is False
    assert hit["xy_only_is_3d"] is False
    assert hit["within_radius"] is True
    assert hit["ambiguous_grip"] is True
    assert hit["neighbor_count_within_radius"] == 2
    assert hit["nearest"]["episode_id"] in (1000, 1001)
    assert hit["nearest"]["grip_sign"] in ("open", "close")
    assert "delta_z" in hit["nearest"]
    assert hit["nearest"]["policy_phase"] == "release"
    assert hit["components"]["xy_within_3cm_is_3d_close"] is False
    assert hit["is_3d_close"] is True
    far = nearest_coverage(query, [_window(1002, tcp_z=1.30, grip=1.0, delta_z=0.2)], scales, radius=0.05, position_bar=0.05)
    assert far["within_radius"] is False
    assert far["is_3d_close"] is False
    assert far["components"]["position_3d_m"] == pytest.approx(1.0)


def test_near_ambiguous_contacts_do_not_authorize_collection():
    contacts = [{"within_radius": True, "ambiguous_grip": True, "ambiguous_delta_z": False} for _ in range(6)]
    safes = [{"within_radius": True, "ambiguous_grip": False, "ambiguous_delta_z": False} for _ in range(4)]
    decision = coverage_decision(contacts, safes)
    assert decision["collect_recovery"] is False
    assert decision["action_ambiguity"] is True
    assert decision["strong_off_path"] is False
    assert decision["reward_claim"] is False
    off = [{"within_radius": False, "ambiguous_grip": False, "ambiguous_delta_z": False} for _ in range(6)]
    covered = [{"within_radius": True, "ambiguous_grip": False, "ambiguous_delta_z": False} for _ in range(4)]
    collect = coverage_decision(off, covered)
    assert collect["strong_off_path"] is True
    assert collect["collect_recovery"] is True
    assert collect["reward_claim"] is False


def test_expert_window_labels_the_executed_action_and_ignores_future_rows():
    obs = np.zeros((5, 42), dtype=np.float64)
    obs[1, 18] = 1.0
    obs[1, 19:22] = [0.0, 0.0, 0.21]
    obs[2, 19:22] = [9.0, 9.0, 9.0]
    actions = np.zeros((4, 7), dtype=np.float64)
    actions[0, 6] = -1.0
    actions[1, 6] = 1.0
    actions[1, 2] = -0.1
    actions[2, 6] = -1.0
    record = expert_training_window(obs, actions, start=0)
    assert record["latest_state"][19:22] == pytest.approx([0.0, 0.0, 0.21])
    assert record["grip"] == pytest.approx(1.0)
    assert record["delta_z"] == pytest.approx(-0.1)
    assert record["policy_phase"] == "release"
    assert record["future_frames_used"] == 0
    assert record["previous_action_used_as_label"] is False
    obs[3, 0] = 5.0
    actions[3, 2] = 0.35
    again = expert_training_window(obs, actions, start=0)
    assert again["grip"] == pytest.approx(1.0)
    assert again["latest_state"][19:22] == pytest.approx([0.0, 0.0, 0.21])


def test_published_contact_steps_are_the_identity_not_a_count_only():
    rows = [
        {"scene_index": index, "obstacle_contact": True, "contact_step": step}
        for index, step in ((1, 90), (3, 146), (5, 113), (9, 146), (19, 92), (21, 162))
    ]
    assert contact_identity(rows)["match"] is True
    assert contact_identity(rows)["count"] == 6
    rows[0]["contact_step"] = 91
    assert contact_identity(rows)["match"] is False


def test_index_uses_the_train_phase_radius_and_makes_no_reward_claim():
    scales = {name: 1.0 for name in (
        "position_3d", "object_position_3d", "rotation", "proprio_qpos", "proprio_qvel", "normalized_hint")}
    train = [_window(1000, tcp_z=0.30), _window(1001, tcp_z=0.30)]
    query = _window(None, tcp_z=0.30, grip=-1.0, delta_z=-0.35)
    query["scene_index"] = 1
    query["role"] = "contact"
    query["policy_phase"] = "release"
    reports = index_queries([query], train, scales)
    assert reports[0]["within_radius"] is True
    assert reports[0]["radius_choice"]["source"] == "train_leave_one_scene_out_same_policy_phase"
    assert reports[0]["radius_choice"]["uses_dev_labels"] is False
    assert reports[0]["reward_claim"] is False
    assert reports[0]["scene_index"] == 1
