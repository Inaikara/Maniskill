import numpy as np

from examples.baselines.constrain.pickcube_clearance_proxy_r42 import (
    cylinder_sdf, pose_delta, quaternion_rotation_vector, ridge_fit, ridge_predict,
    response_features,
)


def test_finite_cylinder_distance_handles_cap_and_outward_gradient():
    points = np.array([[0.0, 0.0, 0.25], [0.2, 0.0, 0.05], [0.0, 0.0, 0.05]])
    distances = cylinder_sdf(points, [0.0, 0.0], 0.1, 0.1)
    np.testing.assert_allclose(distances, [0.15, 0.1, -0.05], atol=1e-12)
    point = np.array([[0.12, 0.0, 0.05]])
    eps = 1e-6
    gradient = (cylinder_sdf(point + [[eps, 0, 0]], [0, 0], .1, .1)
                - cylinder_sdf(point - [[eps, 0, 0]], [0, 0], .1, .1)) / (2*eps)
    assert gradient[0] > 0.99


def test_quaternion_log_is_sign_invariant_and_pose_delta_is_world_frame():
    q = np.array([np.cos(.2), 0, 0, np.sin(.2)])
    np.testing.assert_allclose(quaternion_rotation_vector(q), quaternion_rotation_vector(-q), atol=1e-12)
    a = np.zeros(45); b = a.copy()
    a[19:22] = [1, 2, 3]; b[19:22] = [1.1, 2, 3]
    a[22:26] = [1, 0, 0, 0]; b[22:26] = q
    np.testing.assert_allclose(pose_delta(a[None], b[None])[0], [.1, 0, 0, 0, 0, .4], atol=1e-12)


def test_response_uses_current_executable_action_index_one():
    obs = np.zeros((4, 45)); actions = np.zeros((3, 7))
    obs[:, 22] = 1
    obs[:, 19] = [0, 1, 2, 3]
    actions[:, 0] = [10, 20, 30]
    features = response_features(obs, actions)
    np.testing.assert_allclose(features[:, 0], [20, 30])
    np.testing.assert_allclose(features[:, 6], [1, 1])
    assert features.shape == (2, 21)


def test_ridge_uses_trajectory_mask_only_and_zero_baseline_is_fixed():
    x = np.arange(8*21, dtype=np.float64).reshape(8, 21) / 100
    y = x[:, :6] * 2
    mask = np.array([True]*6 + [False]*2)
    fit = ridge_fit(x, y, mask)
    assert np.max(np.abs(ridge_predict(fit, x[:6])-y[:6])) < 1e-5
    assert fit["coefficient"].shape == (21, 6)
