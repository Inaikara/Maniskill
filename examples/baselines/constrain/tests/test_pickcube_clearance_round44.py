import numpy as np

from examples.baselines.constrain import pickcube_clearance_round44 as r44


def test_quaternion_log_exp_is_sign_invariant_and_round_trips():
    q = r44.quat_exp(np.array([0.2, -0.1, 0.05]))
    np.testing.assert_allclose(r44.quat_log(q), r44.quat_log(-q), atol=1e-12)
    np.testing.assert_allclose(r44.quat_exp(r44.quat_log(q)), q, atol=1e-12)


def test_response_features_use_only_two_history_frames_and_exact_horizon_shapes():
    previous = np.zeros(45)
    current = np.zeros(45)
    previous[22] = current[22] = 1
    previous[19:22] = [0.1, 0.2, 0.3]
    current[19:22] = [0.11, 0.19, 0.32]
    current[:18] = np.arange(18)
    actions = np.zeros((8, 7))
    actions[:, 0] = 1
    features = r44.make_features(previous, current, actions[:, :6])
    assert features.shape == (72,)
    np.testing.assert_allclose(features[:48].reshape(8, 6)[:, 0], .1)
    np.testing.assert_allclose(features[48:66], current[:18])
    np.testing.assert_allclose(features[66:69], current[19:22]-previous[19:22])
    future = np.repeat(current[None], 8, axis=0)
    targets = r44.make_targets(current, future)
    assert targets.shape == (48,)
    np.testing.assert_allclose(targets, 0, atol=1e-12)


def test_controller_contract_is_root_translation_and_root_aligned_rotation():
    import inspect
    import torch
    from mani_skill.agents.robots.panda import panda

    source = inspect.getsource(panda)
    assert "rot_lower=-0.1" in source
    np.testing.assert_allclose(r44.action_to_physical(np.array([1,0,0,1,0,0])), [.1,0,0,-.1,0,0])
    from examples.baselines.constrain.pickcube_clearance_train_r44 import _native_rotation_deltas, _torch_qmatrix
    from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, matrix_to_quaternion
    native = torch.tensor([[[0.2, -0.1, 0.3]]], dtype=torch.float64)
    actual = _torch_qmatrix(_native_rotation_deltas(torch.cat((torch.zeros_like(native),native),dim=-1)))[0,0].numpy()
    effective = native.numpy()[0,0] * -.1
    expected = euler_angles_to_matrix(torch.tensor(effective,dtype=torch.float64),"XYZ").numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-10)


def test_horizon8_proxy_has_xyz_and_rotation_gradients_but_no_gripper_gradient():
    import torch

    from examples.baselines.constrain.pickcube_clearance_train_r44 import horizon8_clearance_components

    raw = torch.zeros((1, 2, 45), dtype=torch.float32)
    raw[:, :, 22] = 1
    raw[:, :, 21] = 0.12
    raw[:, 1, 18] = 1
    x0 = torch.zeros((1, 16, 7), dtype=torch.float32, requires_grad=True)
    coefficient = np.zeros((72, 48), dtype=np.float64)
    coefficient[0, 0] = 1.0   # planned x translation -> horizon-1 x displacement
    coefficient[5, 5] = 1.0   # planned z rotation -> horizon-1 world rotation log
    response = {"feature_mean": np.zeros(72).tolist(), "feature_scale": np.ones(72).tolist(),
                "target_mean": np.zeros(48).tolist(), "coefficient": coefficient.tolist()}
    clouds = {name: np.zeros((0, 3), dtype=np.float64) for name in
              ("panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger")}
    clouds["panda_hand"] = np.array([[0.0, 0.10, 0.0]])
    finger = {name: {"T_tcp_finger_at_reference": np.eye(4).tolist(),
                     "axis_tcp_m_per_q": [0, 0, 0], "qpos_index": 7 if "left" in name else 8,
                     "q_reference_m": 0.0}
              for name in ("panda_leftfinger", "panda_rightfinger")}
    cylinders = {"center": torch.tensor([[0.025, 0.10]]),
                 "radius": torch.tensor([0.04]), "height": torch.tensor([0.3])}
    body, cube, _ = horizon8_clearance_components(x0, raw, response, clouds, finger,
        {key: value.to(x0.device) for key, value in cylinders.items()}, torch.tensor([True]))
    gradient = torch.autograd.grad(body + cube, x0)[0]
    assert gradient[:, 1:9, :3].norm() > 0
    assert gradient[:, 1:9, 3:6].norm() > 0
    assert gradient[:, 1:9, 6].norm() == 0
