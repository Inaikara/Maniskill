import hashlib
import json

import numpy as np
import pytest
import torch

from examples.baselines.constrain.pickcube_clearance_proxy_r43 import (
    BODY_PREDICTION_P95_M, GEOMETRY_TOLERANCE_M, PRECHECK_SCHEMA,
    RESPONSE_IMPROVEMENT, TRAINING_SCHEMA, carried_cube_xy_gap, clearance_proxy_loss,
    swept_clearance_components,
    native_forbidden_radius, padded_cylinder_sdf, transform_link_points_to_world,
    transform_link_points_to_world_inverse_bug, verify_training_admission,
)
from examples.baselines.constrain import pickcube_clearance_train_r43 as training


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_link_local_points_use_tcp_to_link_not_inverse():
    tcp = np.eye(4)
    tcp[:3, 3] = [1.0, 2.0, 3.0]
    t_tcp_link = np.eye(4)
    t_tcp_link[:3, 3] = [0.0, 0.0, -0.2]
    points_link = np.array([[0.1, 0.0, 0.0]])
    world = transform_link_points_to_world(points_link, tcp, t_tcp_link)
    np.testing.assert_allclose(world, [[1.1, 2.0, 2.8]], atol=1e-12)
    assert np.linalg.norm(world - transform_link_points_to_world_inverse_bug(points_link, tcp, t_tcp_link)) > 0.3


def test_registered_sphere_padding_is_subtracted_from_sdf():
    point = torch.tensor([[0.11, 0.0, 0.05]])
    physical = padded_cylinder_sdf(point, [0.0, 0.0], 0.1, 0.1) + 0.005
    padded = padded_cylinder_sdf(point, [0.0, 0.0], 0.1, 0.1)
    torch.testing.assert_close(physical, torch.tensor([0.01]))
    torch.testing.assert_close(padded, torch.tensor([0.005]))


def test_clearance_kernel_has_finite_outward_gradients_and_grasp_mask():
    body = torch.tensor([[[0.11, 0.0, 0.05]]], requires_grad=True)
    raw = torch.zeros(2, 45)
    raw[0, 18] = 1
    raw[0, 29:31] = torch.tensor([0.0, 0.0])
    raw[1, 18] = 0
    body_cost, cube_cost = clearance_proxy_loss(body, [0.0, 0.0], 0.1, 0.1,
                                                raw[:1], 0.1)
    (body_cost + cube_cost).backward()
    assert torch.isfinite(body_cost) and torch.isfinite(cube_cost)
    assert body.grad is not None and body.grad[0, 0, 0] < 0
    assert cube_cost.item() > 0
    no_grasp_body = body.detach().clone().requires_grad_(True)
    _, inactive = clearance_proxy_loss(no_grasp_body, [0.0, 0.0], 0.1, 0.1,
                                       raw[1:2], 0.1)
    assert inactive.item() == 0


def test_swept_kernel_detects_midpath_body_and_cube_violation_with_finite_gradients():
    body = torch.tensor([[[[-0.2, 0.0, 0.05]]]], requires_grad=False)
    delta = torch.tensor([[0.44, 0.0, 0.0]], requires_grad=True)
    cube_xy = torch.tensor([[-0.2, 0.0]])
    body_cost, cube_cost, stats = swept_clearance_components(
        body, delta, cube_xy, torch.tensor([[0.0, 0.0]]),
        torch.tensor([0.05]), torch.tensor([0.1]), torch.tensor([True]),
        torch.tensor([0.08]))
    assert body_cost > 0 and cube_cost > 0
    assert stats["body_min_gap_m"] < 0 and stats["cube_min_gap_m"] < 0
    (body_cost + cube_cost).backward()
    assert torch.isfinite(delta.grad).all()
    assert delta.grad[0, 0] < 0


def test_swept_kernel_empty_masks_return_connected_zero():
    body = torch.tensor([[[[0.5, 0.5, 0.5]]]], requires_grad=True)
    delta = torch.zeros((1, 3), requires_grad=True)
    body_cost, cube_cost, _ = swept_clearance_components(
        body, delta, torch.tensor([[0.5, 0.5]]), torch.tensor([[0.0, 0.0]]),
        torch.tensor([0.02]), torch.tensor([0.1]), torch.tensor([False]),
        torch.tensor([0.08]))
    assert body_cost.item() == cube_cost.item() == 0
    (body_cost + cube_cost).backward()
    assert torch.isfinite(delta.grad).all()


def test_carried_cube_xy_forbidden_constraint_applies_above_cylinder_top():
    raw = torch.zeros(2, 45)
    raw[:, 18] = 1
    raw[0, 29:31] = torch.tensor([0.0, 0.0])
    raw[0, 31] = 2.0
    raw[1, 29:31] = torch.tensor([0.3, 0.0])
    gaps = carried_cube_xy_gap(raw, [0.0, 0.0], 0.1)
    torch.testing.assert_close(gaps[0], torch.tensor(-0.1))
    torch.testing.assert_close(gaps[1], torch.tensor(0.2))
    assert native_forbidden_radius(0.03, 0.02, 0.005) == pytest.approx(0.03 + np.sqrt(2)*0.02 + 0.005)


def test_admission_rejects_mutated_true_flag_inverse_error_and_missing_coverage(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("frozen source\n")
    source_files = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    data = {"h5_sha256": "h5", "manifest_sha256": "manifest", "validation_sha256": "validation"}
    assets = {"urdf": "u", "robot": "r", "meshes": "m"}
    report = {"schema": PRECHECK_SCHEMA, "status": "failed", "training_permitted": False,
              "registration_sha256": "frozen-registration",
              "source_files": source_files, "data": data, "asset_hashes": assets,
              "metrics": {"native_point_max_abs_m": 0.424431, "heldout_body_prediction_p95_m": 0.0,
                          "response_improvement_fraction": 0.96, "all_four_links_covered": False,
                          "heldout_body_mean_improvement_fraction": 0.9,
                          "manifest_radius_max_abs_error_m": 0.0,"manifest_center_max_abs_error_m":0.0,
                          "manifest_height_max_abs_error_m":0.0,
                          "cube_xy_kernel_exercised": False, "padding_applied": False},
              "native_fk_validation":{"states":36}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    registration = {"schema": TRAINING_SCHEMA, "precheck_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    "precheck_registration_sha256": "frozen-registration",
                    "source_files": source_files, "data": data, "asset_hashes": assets}
    registration["registration_sha256"] = _digest(registration)
    registration_path = tmp_path / "training_registration.json"
    registration_path.write_text(json.dumps(registration))
    report["training_permitted"] = True
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_training_admission(registration_path, report_path, source_files, data, assets)


def test_admission_accepts_only_pinned_fresh_semantically_valid_report(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("frozen source\n")
    source_files = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    data = {"h5_sha256": "h5", "manifest_sha256": "manifest", "validation_sha256": "validation"}
    assets = {"urdf": "u", "robot": "r", "meshes": "m"}
    report = {"schema": PRECHECK_SCHEMA, "status": "passed", "training_permitted": True,
              "registration_sha256": "frozen-registration",
              "source_files": source_files, "data": data, "asset_hashes": assets,
              "metrics": {"native_point_max_abs_m": GEOMETRY_TOLERANCE_M,
                          "heldout_body_prediction_p95_m": BODY_PREDICTION_P95_M,
                          "response_improvement_fraction": RESPONSE_IMPROVEMENT,
                          "heldout_body_mean_improvement_fraction": RESPONSE_IMPROVEMENT,
                          "manifest_radius_max_abs_error_m":0.0,"manifest_center_max_abs_error_m":0.0,
                          "manifest_height_max_abs_error_m":0.0,
                          "all_four_links_covered": True, "cube_xy_kernel_exercised": True,
                          "padding_applied": True}, "native_fk_validation":{"states":36}}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    registration = {"schema": TRAINING_SCHEMA, "precheck_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                    "precheck_registration_sha256": "frozen-registration",
                    "source_files": source_files, "data": data, "asset_hashes": assets}
    registration["registration_sha256"] = _digest(registration)
    registration_path = tmp_path / "training_registration.json"
    registration_path.write_text(json.dumps(registration))
    assert verify_training_admission(registration_path, report_path, source_files, data, assets)["verified"]
    source.write_text("mutated source\n")
    with pytest.raises(ValueError):
        verify_training_admission(registration_path, report_path, source_files, data, assets)


def test_production_x0_path_uses_action_index_one_and_backpropagates_proxy():
    raw=torch.zeros((1,2,45));raw[:,:,22]=1
    raw[:,:,19]=-.2;raw[:,:,21]=.05;raw[:,:,18]=1;raw[:,:,29]=-.2;raw[:,:,31]=.05
    response={"feature_mean":[0.0]*21,"target_mean":[0.0]*6,
        "coefficient":[[0.0]*6 for _ in range(21)]}
    response["coefficient"][0][0]=1.0
    clouds={name:np.array([[0.0,0.0,0.0]]) for name in
        ("panda_link7","panda_hand","panda_leftfinger","panda_rightfinger")}
    finger={name:{"qpos_index":idx,"q_reference_m":0.0,
        "T_tcp_finger_at_reference":np.eye(4).tolist(),"axis_tcp_m_per_q":[0,0,0]}
        for name,idx in (("panda_leftfinger",7),("panda_rightfinger",8))}
    x0=torch.zeros((1,16,7),requires_grad=True)
    with torch.no_grad():x0[0,1,0]=.44
    body,cube,stats=training.clearance_proxy_from_x0(x0,raw,raw,response,clouds,finger,
        {"center":torch.tensor([[0.,0.]]),"radius":torch.tensor([.05]),
         "height":torch.tensor([.1])},torch.tensor([True]))
    assert body.item()>0 and cube.item()>0 and int(stats["active_rows"])==1
    (body+cube).backward()
    assert torch.isfinite(x0.grad).all() and x0.grad[0,1,0]!=0


def test_actor_forward_never_calls_training_geometry(monkeypatch):
    actor=training.ClearanceActor.__new__(training.ClearanceActor)
    actor.device=torch.device("cpu");actor.stats={}
    class Model(torch.nn.Module):
        def forward(self,sample,timestep,*,global_cond,hint):
            assert global_cond.shape==(1,84) and hint.shape==(1,16,9)
            return torch.zeros_like(sample)
    class Scheduler:
        timesteps=[torch.tensor(0)]
        def step(self,eps,timestep,sample):
            return type("Out",(),{"prev_sample":sample})
    actor.model=Model();actor.scheduler=Scheduler()
    monkeypatch.setattr(training.engine,"normalize_observation",lambda raw,stats: torch.zeros_like(raw))
    def forbidden(*args,**kwargs):raise AssertionError("inference called training-only geometry")
    monkeypatch.setattr(training,"swept_clearance_components",forbidden)
    monkeypatch.setattr(training,"clearance_proxy_from_x0",forbidden,raising=False)
    raw=torch.zeros((1,2,45));raw[:,:,22]=1
    assert actor.get_action(raw).shape==(1,8,7)
