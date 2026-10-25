"""Seams for the frozen PlaceSphere release-contact diagnosis."""

import numpy as np
import pytest

from examples.baselines.constrain.placesphere_release_diagnosis import (
    action_semantics,
    body_origin_gap,
    contact_surface_record,
    contact_vs_expert,
    controller_contract,
    decide_release_factor,
    label_training_window,
    perturbation_required,
    phase_inverse_weights,
    reconcile_native_failure,
    vertex_column_gap,
)


def test_grasped_no_transport_is_not_incorrect_release():
    row = reconcile_native_failure(
        phase="grasped_no_transport", mechanism="incorrect_release",
        ever_released=False, terminal_grasped=True, transported=False, terminal_on_bin=False)
    assert row["operative"] == "grasped_no_transport"
    assert row["mistaken_incorrect_release"] is True
    assert row["replaces_native_success"] is False


def test_off_bin_release_stays_incorrect_release():
    row = reconcile_native_failure(
        phase="release_or_stability_failed", mechanism="incorrect_release",
        ever_released=True, terminal_grasped=False, transported=True, terminal_on_bin=False)
    assert row["operative"] == "incorrect_release"
    assert row["mistaken_incorrect_release"] is False


def test_xy_within_3cm_is_not_3d_close():
    gap = body_origin_gap([0.02, 0.0, 0.20], center_xy=[0.0, 0.0], radius=0.03, top_z=0.12)
    assert gap["xy_distance_m"] == pytest.approx(0.02)
    assert gap["xy_within_3cm"] is True
    assert gap["is_3d_close"] is False
    assert gap["is_surface_clearance"] is False
    assert gap["vertical_gap_m"] == pytest.approx(0.08)
    assert gap["radial_gap_m"] == pytest.approx(-0.01)
    assert gap["method"] == "body_origin_vs_cylinder_axis"


def test_high_point_inside_the_xy_disk_is_not_a_top_cap_contact():
    record = contact_surface_record(
        {"entity": "panda_hand", "position": [0.01, 0.0, 0.20], "separation_m": -0.001,
         "impulse": 0.2, "surface": {"method": "post_step_physx_contact_with_impulse",
                                      "is_surface_clearance": True, "separation_m": -0.001}},
        center_xy=[0.0, 0.0], radius=0.03, top_z=0.12)
    assert record["location"] == "above_column"
    assert record["is_3d_close"] is False
    assert record["xy_within_3cm_is_3d_close"] is False


def test_physx_point_on_the_top_plane_is_exact_surface_not_an_xy_label():
    record = contact_surface_record(
        {"entity": "panda_hand", "position": [0.01, 0.0, 0.127], "separation_m": -0.001338,
         "impulse": 0.4, "surface": {"method": "post_step_physx_contact_with_impulse",
                                      "is_surface_clearance": True, "separation_m": -0.001338}},
        center_xy=[0.0, 0.0], radius=0.03, top_z=0.127)
    assert record["location"] == "top_cap"
    assert record["is_surface_clearance"] is True
    assert record["is_3d_close"] is False
    assert record["vertical_gap_m"] == pytest.approx(0.0, abs=1e-12)
    assert record["radial_gap_m"] == pytest.approx(0.01 - 0.03)


def test_cached_contact_is_not_called_surface_clearance():
    record = contact_surface_record(
        {"entity": "panda_hand", "position": [0.0, 0.0, 0.08], "separation_m": 0.0, "impulse": 0.1,
         "surface": {"method": "post_step_contact_cache_impulse_cleared",
                      "is_surface_clearance": False, "separation_m": 0.0}},
        center_xy=[0.0, 0.0], radius=0.03, top_z=0.08)
    assert record["is_surface_clearance"] is False
    assert record["location"] == "top_cap"


def test_action_semantics_names_grip_and_does_not_treat_rotation_as_meters():
    report = action_semantics([0.1, -0.2, 0.05, 0.01, -0.02, 0.0, 1.0])
    assert report["channels"] == [
        "delta_x", "delta_y", "delta_z", "delta_euler_x", "delta_euler_y", "delta_euler_z", "gripper"]
    assert report["grip_sign"] == "open"
    assert report["rotation_is_meters"] is False
    assert report["translation_is_already_meters"] is False
    assert report["pose_action_cap"] == 0.35
    assert report["controller_pos_scale_m"] == 0.1
    assert report["controller_rot_scale_rad"] == 0.1
    assert action_semantics([0, 0, 0, 0, 0, 0, -1.0])["grip_sign"] == "close"


def test_window_label_saves_the_entry_and_does_not_call_radial_gap_3d_close():
    obs = np.zeros((5, 42), np.float32)
    obs[:, 18] = 1
    obs[:, 19:22] = (0.0, 0.0, 0.2)
    obs[:, 26:29] = (0.2, 0.0, 0.0)
    obs[:, 29:32] = (0.0, 0.0, 0.2)
    obs[:, 39:42] = (0.0, 0.0, 0.03)
    actions = np.zeros((4, 7), np.float32)
    actions[2, 6] = 1.0
    row = label_training_window(obs, actions, start=0, episode_id=1100)
    assert row["episode_id"] == 1100
    assert row["window_start"] == 0
    assert row["phase"] == "release"
    assert row["near"] is True
    assert row["is_3d_close"] is False
    assert row["is_surface_clearance"] is False
    assert row["height_in_observation"] is False
    assert row["far_near_basis"] == "tcp_center_radial_gap"


def test_ungrasped_opening_is_not_a_release_window():
    obs = np.zeros((4, 42), np.float32)
    obs[:, 39:42] = (1.0, 0.0, 0.03)
    obs[:, 19:22] = (0.0, 0.0, 0.1)
    actions = np.zeros((3, 7), np.float32)
    actions[:, 6] = 1.0
    row = label_training_window(obs, actions, start=0, episode_id=7)
    assert row["phase"] == "approach"
    assert row["near"] is False


def test_phase_weights_are_inverse_frequency_and_mean_one():
    weights = phase_inverse_weights({"approach": 10, "transport": 30, "release": 10})
    assert weights["release"] == pytest.approx(50 / 30)
    assert weights["transport"] == pytest.approx(50 / 90)
    total = 10 * weights["approach"] + 30 * weights["transport"] + 10 * weights["release"]
    assert total / 50 == pytest.approx(1.0)


def test_vertex_gap_is_an_approximation_and_ignores_points_outside_the_column():
    vertices = np.array([[0.0, 0.0, 0.15], [0.20, 0.0, 0.01]])
    gap = vertex_column_gap(vertices, center_xy=[0.0, 0.0], radius=0.03, top_z=0.12)
    assert gap["is_surface_clearance"] is False
    assert gap["is_3d_close"] is False
    assert gap["min_vertical_gap_m"] == pytest.approx(0.03)
    assert gap["vertices_in_column"] == 1


def test_tcp_center_distance_under_3cm_is_not_3d_close():
    near = contact_vs_expert([0.0, 0.0, 0.2], [0.02, 0.0, 0.2], grip_opening=True, expert_physx=False)
    far = contact_vs_expert([0.0, 0.0, 0.2], [0.05, 0.0, 0.2], grip_opening=True, expert_physx=False)
    assert near["class"] == "execution_track"
    assert near["is_3d_close"] is False
    assert near["xy_within_3cm_is_3d_close"] is False
    assert near["basis"] == "tcp_center_distance_not_surface"
    assert far["class"] == "prediction_far"
    assert far["is_surface_clearance"] is False


def test_penetration_does_not_infer_a_margin_or_request_perturbation():
    assert perturbation_required(separation_m=-0.001, margin_inferred=False) is False
    assert perturbation_required(separation_m=None, margin_inferred=True) is True


def _evidence(*, release_fraction, ratio, prediction, execution, mismatch=None):
    weights = phase_inverse_weights({"approach": 70, "transport": 20, "held_near_bin": 5, "release": 5})
    return {
        "reproduction_mismatch": False,
        "expert_physx_contact_count": 0,
        "release_window_fraction": release_fraction,
        "release_epsilon": ratio,
        "other_epsilon": 1.0,
        "prediction_far_contacts": prediction,
        "execution_track_contacts": execution,
        "contact_count": 6,
        "failure_13_operative": "incorrect_release",
        "failure_14_operative": "incorrect_release",
        "failure_16_operative": "grasped_no_transport",
        "failure_16_mistaken_incorrect_release": True,
        "phase_weights": weights,
        "controller_mismatch": mismatch,
        "window_index_saved": True,
    }


def test_decide_trains_only_when_release_supervision_is_rare_and_worse():
    decision = decide_release_factor(_evidence(release_fraction=0.08, ratio=2.1, prediction=6, execution=0))
    assert decision["trial"] is True
    assert decision["factor"] == "phase_inverse_frequency"
    assert decision["weights"]["release"] == pytest.approx(100 / 20)
    assert decision["architecture_change"] is False
    assert decision["action_shield"] is False
    assert decision["new_data"] is False
    assert decision["updates"] == 90000
    assert decision["base_checkpoint_sha256"].startswith("2bbb7f5e")
    assert decision["stride_swept"] is False


def test_decide_refuses_execution_tracking_and_does_not_start_a_stride_sweep():
    decision = decide_release_factor(_evidence(release_fraction=0.08, ratio=2.1, prediction=1, execution=5))
    assert decision["trial"] is False
    assert decision["factor"] is None
    assert decision["stride"] is None


def test_named_controller_mismatch_is_the_only_execution_trial():
    decision = decide_release_factor(_evidence(
        release_fraction=0.08, ratio=2.1, prediction=0, execution=6, mismatch="executed_slice"))
    assert decision["trial"] is True
    assert decision["factor"] == "executed_slice"
    assert decision["same_checkpoint"] is True
    assert decision["stride_swept"] is False


def test_common_release_windows_do_not_authorize_reweighting():
    decision = decide_release_factor(_evidence(release_fraction=0.40, ratio=2.1, prediction=6, execution=0))
    assert decision["trial"] is False
    assert decision["reason"] == "release_supervision_not_rare_or_not_worse"


def test_controller_contract_matches_the_panda_delta_pose_defaults():
    report = controller_contract({
        "control_mode": "pd_ee_delta_pose", "normalize_action": True, "use_delta": True,
        "use_target": False, "pos_lower": -0.1, "pos_upper": 0.1, "rot_lower": -0.1,
        "rot_upper": 0.1, "gripper_lower": -0.01, "gripper_upper": 0.04,
        "executed_slice": [1, 9], "obs_horizon": 2, "pred_horizon": 16, "act_horizon": 8,
        "seed_rule": "scene_seed",
    })
    assert report["matches_expert_contract"] is True
    assert report["pose_0.35_is_radians"] is False
    assert report["rotation_action_is_normalized_then_scaled_by_rot_lower"] is True
    assert report["mismatch"] is None
