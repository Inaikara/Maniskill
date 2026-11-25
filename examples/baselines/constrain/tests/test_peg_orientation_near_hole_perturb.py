import numpy as np
import pytest
import torch
from types import SimpleNamespace
from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseController
from mani_skill.utils.structs import Pose

from examples.baselines.constrain import peg_orientation_near_hole_perturb as perturb


def test_native_action_builder_reaches_requested_root_delta_and_yaw_sign():
    controller = perturb.NativePoseCommandGeometry(
        pos_lower=np.full(3, -0.1), pos_upper=np.full(3, 0.1),
        rot_lower=np.full(3, -0.1), rot_upper=np.full(3, 0.1),
        frame="root_translation:root_aligned_body_rotation", use_delta=True,
    )
    current = perturb.PoseValue([0.31, -0.17, 0.42], perturb.quat_from_euler_xyz([0.2, -0.3, 0.4]))
    command = perturb.native_pose_action(
        current, perturb.PoseValue([0.32, -0.17, 0.42], perturb.quat_from_euler_xyz([0.2, -0.3, 0.4 + 0.01])), controller
    )
    target = perturb.compute_native_target(current, command["action"], controller)
    np.testing.assert_allclose(target.position, [0.32, -0.17, 0.42], atol=1e-6)
    np.testing.assert_allclose(target.rotation, perturb.quat_from_euler_xyz([0.2, -0.3, 0.41]), atol=1e-6)
    assert command["requested_rotation_euler_xyz_rad"][2] > 0
    assert command["action"][5] < 0  # native rot_lower is negative


@pytest.mark.parametrize("yaw", [np.deg2rad(5), -np.deg2rad(5)])
def test_builder_matches_installed_native_controller_target_pose(yaw):
    controller = perturb.NativePoseCommandGeometry(
        pos_lower=np.full(3, -0.1), pos_upper=np.full(3, 0.1),
        rot_lower=np.full(3, -0.1), rot_upper=np.full(3, 0.1),
        frame="root_translation:root_aligned_body_rotation", use_delta=True,
    )
    current = perturb.PoseValue(np.array([0.2, -0.3, 0.4]), perturb.quat_from_euler_xyz([0.1, -0.2, 0.3]))
    desired = perturb.PoseValue(np.array([0.205, -0.3, 0.4]), perturb.quat_from_euler_xyz([0.1, -0.2, 0.3+yaw]))
    command = perturb.native_pose_action(current, desired, controller)
    scaled = np.r_[controller.pos_lower + (command["action"][:3]+1) *
                   (controller.pos_upper-controller.pos_lower)/2,
                   np.clip(command["action"][3:6], -1, 1) * controller.rot_lower]
    actual_current = Pose.create_from_pq(torch.tensor(np.asarray([current.position]), dtype=torch.float32),
                                         torch.tensor(np.asarray([current.rotation]), dtype=torch.float32))
    actual_target = PDEEPoseController.compute_target_pose(
        SimpleNamespace(config=SimpleNamespace(use_delta=True, frame=controller.frame)),
        actual_current, torch.tensor(np.asarray([scaled]), dtype=torch.float32))
    expected = perturb.compute_native_target(current, command["action"], controller)
    np.testing.assert_allclose(actual_target.p.detach().numpy()[0], expected.position, atol=1e-6)
    actual_quat = actual_target.q.detach().numpy()[0]
    assert abs(np.dot(actual_quat, expected.rotation)) == pytest.approx(1.0, abs=1e-6)


def test_command_rotation_pivots_tcp_around_peg_head_and_reports_clipping():
    controller = perturb.NativePoseCommandGeometry(
        pos_lower=np.full(3, -0.1), pos_upper=np.full(3, 0.1),
        rot_lower=np.full(3, -0.1), rot_upper=np.full(3, 0.1),
        frame="root_translation:root_aligned_body_rotation", use_delta=True,
    )
    current = perturb.PoseValue([0.4, 0.2, 0.5], perturb.quat_from_euler_xyz([0, 0, 0]))
    head = np.array([0.45, 0.2, 0.5])
    desired = perturb.rotate_tcp_about_head(current, head, np.pi / 36)
    command = perturb.native_pose_action(current, desired, controller)
    realized = perturb.compute_native_target(current, command["action"], controller)
    np.testing.assert_allclose(realized.position, desired.position, atol=1e-6)
    assert command["realized_rotation_euler_xyz_rad"][2] > 0
    assert command["action_clipped"] is False


def test_outside_entry_predicate_uses_true_front_plane_not_box_center():
    assert perturb.entry_clearance([-0.105, 0.0, 0.0], peg_half_length=0.085) == pytest.approx(0.020)
    assert not perturb.entry_clearance([-0.045, 0.0, 0.0], peg_half_length=0.085) >= 0.019


def test_prospective_stability_uses_vector_norm_and_axis_specific_bounds():
    target = np.zeros(3)
    assert perturb.head_position_eligible([0.0, .002, 0.0], target)
    assert perturb.head_position_eligible([.001, .002, .002], target)
    assert not perturb.head_position_eligible([.0021, 0, 0], target)
    assert not perturb.head_position_eligible([0, .0021, 0], target)
    assert not perturb.head_position_eligible([0, 0, .0031], target)


def test_pose_only_qualification_does_not_relax_full_stability():
    assert perturb.pose_qualified([0, 0, 0], [0, 0, 0], 0, grasped=True, inserted=False)
    assert not perturb.pose_qualified([0, 0, 0], [0, 0, 0], 0, grasped=False, inserted=False)
    assert not perturb.pose_qualified([0, 0, 0], [0, 0, 0], np.deg2rad(1.01), True, False)


def test_velocity_measurement_retains_raw_vectors_and_norms():
    task = SimpleNamespace(
        peg=SimpleNamespace(get_linear_velocity=lambda: torch.tensor([[.003, .004, 0.]]),
                            get_angular_velocity=lambda: torch.tensor([[0., 0., .04]])),
        agent=SimpleNamespace(robot=SimpleNamespace(get_qvel=lambda: torch.tensor([[.01, -.02, 0., 0., 0., 0., .005]]))),
    )
    measurement = perturb._velocity_measurement(task)
    assert measurement["peg_linear_velocity_m_s"] == pytest.approx([.003, .004, 0])
    assert measurement["peg_linear_speed_m_s"] == pytest.approx(.005)
    assert measurement["peg_angular_velocity_rad_s"] == pytest.approx([0, 0, .04])
    assert measurement["peg_angular_speed_rad_s"] == pytest.approx(.04)
    assert measurement["arm_qvel_rad_s"] == pytest.approx([.01, -.02, 0, 0, 0, 0, .005])


def test_lateral_admission_bands_are_signed_and_disjoint_at_boundaries():
    assert perturb.lateral_admission_eligible(.003, +1)
    assert perturb.lateral_admission_eligible(.007, +1)
    assert not perturb.lateral_admission_eligible(.002, +1)
    assert not perturb.lateral_admission_eligible(.0020001, +1)
    assert not perturb.lateral_admission_eligible(.0070001, +1)
    assert perturb.lateral_admission_eligible(-.003, -1)
    assert perturb.lateral_admission_eligible(-.007, -1)
    assert not perturb.lateral_admission_eligible(-.002, -1)
    assert not perturb.lateral_admission_eligible(-.0070001, -1)
    assert not perturb.lateral_admission_eligible(.005, -1)


def test_pose_window_boundary_checks_contiguity_span_and_quaternion_sign():
    rows = []
    for i in range(11):
        rows.append({"control_index": i, "monotonic_seconds": i*.025, "simulation_elapsed_seconds": i*.025,
            "head_error_xyz_m": [0., 0., 0.], "entry_clearance_m": .020,
            "yaw_error_rad": 0., "grasped": True, "official_inserted": False,
            "head_hole_xyz_m": [0., 0., 0.], "peg_center_world_m": [1., 0., 0.],
            "peg_quaternion_wxyz": ([1., 0., 0., 0.] if i % 2 == 0 else [-1., 0., 0., 0.]),
            "arm_qpos_rad": [0.]*7})
    assert perturb.minimal_window_admission(rows, .025)
    assert not perturb.minimal_window_admission(rows[:-1], .025)
    gap = [dict(row) for row in rows]
    gap[5]["control_index"] += 1
    assert not perturb.minimal_window_admission(gap, .025)
    bad = [dict(row) for row in rows]
    bad[4]["head_hole_xyz_m"] = [np.nan, 0., 0.]
    assert not perturb.minimal_window_admission(bad, .025)


def test_pose_window_is_not_legacy_instant_velocity_gate():
    rows = [{"control_index": i, "monotonic_seconds": i*.025, "simulation_elapsed_seconds": i*.025,
        "head_error_xyz_m": [0., 0., 0.], "entry_clearance_m": .020,
        "yaw_error_rad": 0., "grasped": True, "official_inserted": False,
        "head_hole_xyz_m": [0., 0., 0.], "peg_center_world_m": [1., 0., 0.],
        "peg_quaternion_wxyz": [1., 0., 0., 0.], "arm_qpos_rad": [0.]*7,
        "peg_linear_speed_m_s": .03, "peg_angular_speed_rad_s": .5,
        "max_arm_qvel_rad_s": .03} for i in range(11)]
    assert perturb.minimal_window_admission(rows, .025)


@pytest.mark.parametrize("field,value", [
    ("yaw_error_rad", np.nan),
    ("peg_quaternion_wxyz", [0., 0., 0., 0.]),
])
def test_pose_window_rejects_nonfinite_yaw_and_zero_quaternion(field, value):
    rows = [{"control_index": i, "monotonic_seconds": i*.025, "simulation_elapsed_seconds": i*.025,
        "head_error_xyz_m": [0., 0., 0.], "entry_clearance_m": .020,
        "yaw_error_rad": 0., "grasped": True, "official_inserted": False,
        "head_hole_xyz_m": [0., 0., 0.], "peg_center_world_m": [1., 0., 0.],
        "peg_quaternion_wxyz": [1., 0., 0., 0.], "arm_qpos_rad": [0.]*7} for i in range(11)]
    rows[4][field] = value
    assert not perturb.minimal_window_admission(rows, .025)


def test_case_finalizer_preserves_partial_rows_and_original_exception(monkeypatch):
    task = SimpleNamespace(
        peg_half_sizes=torch.tensor([[.085, .01, .01]]), control_mode="pd_ee_delta_pose",
        peg=object(), agent=SimpleNamespace(controller=SimpleNamespace(get_state=lambda: np.array([], dtype=np.float32))),
        has_peg_inserted=lambda: (torch.tensor([False]),),
    )
    env = SimpleNamespace(unwrapped=task, get_obs=lambda: np.asarray([np.bool_(True)]))
    def fail_after_prefix(*args):
        telemetry = args[-1]
        telemetry["executed_prefix_actions"] = np.asarray([[np.bool_(True)]], dtype=object)
        telemetry["observation_history"] = [np.asarray([np.bool_(True)])]
        raise RuntimeError("injected after prefix")
    monkeypatch.setattr(perturb, "_run_condition_body", fail_after_prefix)
    monkeypatch.setattr(perturb, "capture_physical_state", lambda _task: {"qpos": np.asarray([1.0])})
    monkeypatch.setattr(perturb, "_task_head", lambda _task: np.asarray([-.105, 0, 0]))
    monkeypatch.setattr(perturb, "native_grasped", lambda _task: np.bool_(True))
    with pytest.raises(RuntimeError, match="injected after prefix") as caught:
        perturb._run_condition(env, None, np.zeros((2, 7)), {"prefix_controls": 2}, "aligned", [0, 1])
    evidence = caught.value.case_evidence
    assert evidence["partial_case"] is True
    assert evidence["physical_state"]["qpos"] == [1.0]
    assert evidence["native_grasped"] is True
    assert evidence["official_inserted"] is False
    assert evidence["actual_observation"] == [1.0]
    assert evidence["observation_history"] == [[1.0]]
    assert evidence["executed_prefix_actions"] == [[True]]
    assert evidence["final_entry_clearance_m"] == pytest.approx(.020)
