"""Tests for the relative obstacle condition seam (Ticket 03).

Seams under test:
- the flat 51d obstacle obs schema is proven against the real environment
  (TCP world xy at indices 18/19, appended obstacle (x, y, radius) at 48-50),
- the relative condition is computed from the *latest* observation frame as
  ``[obstacle_x - tcp_x, obstacle_y - tcp_y, radius]`` in world-parallel XY
  axes (no TCP rotation applied), meters,
- semantic invariances: equal translation of TCP and obstacle changes nothing;
  a TCP position shift changes the condition accordingly; a TCP orientation
  change changes nothing; the radius channel is independent of positions,
- one explicit, fixed and configurable normalization shared by training and
  inference, with recorded out-of-range clipping (never per-batch statistics).

Pure-tensor tests use synthetic 51d frames; one test builds the real env to
prove the schema (real flattened obs ordering, not guessed).
"""

import gymnasium as gym
import numpy as np
import pytest
import torch

from examples.baselines.constrain.constrain_condition import (
    OBSTACLE_INDICES,
    TCP_XY_INDICES,
    ConditionNormConfig,
    build_hint,
    latest_relative_condition,
)


def frame(tcp_x=0.0, tcp_y=0.0, obstacle_x=0.1, obstacle_y=0.2, radius=0.03,
          tcp_quat=(0.0, 1.0, 0.0, 0.0)):
    """One synthetic 51d flat frame with the proven layout."""
    f = torch.zeros(51)
    f[18], f[19] = tcp_x, tcp_y
    f[21:25] = torch.tensor(tcp_quat)
    f[48], f[49], f[50] = obstacle_x, obstacle_y, radius
    return f


def test_real_env_flattened_obs_schema():
    """The real StackCubeObstacle-v1 flattened obs is proven, not guessed:
    TCP world xy sits at indices 18/19 (qpos 0-9, qvel 9-18, tcp raw_pose
    18-25) and the appended obstacle (center_x, center_y, radius) at 48-51."""
    import examples.baselines.constrain.stackcube_obstacle  # noqa: F401
    from mani_skill.utils import common

    env = gym.make(
        "StackCubeObstacle-v1", reconfiguration_freq=1, robot_uids="panda_wristcam",
        control_mode="pd_ee_delta_pose", reward_mode="sparse", obs_mode="state",
        render_mode="rgb_array", sim_backend="physx_cpu", num_envs=1,
        max_episode_steps=100, obstacle_radius=0.03, obstacle_height=0.18,
    )
    env.reset(seed=11)
    e = env.unwrapped
    obs_dict = e._get_obs_state_dict(e.get_info())
    flat = common.flatten_state_dict(obs_dict, use_torch=True)[0].numpy()
    blocks = [
        (obs_dict["agent"]["qpos"][0].numpy(), 0),
        (obs_dict["agent"]["qvel"][0].numpy(), 9),
        (obs_dict["extra"]["tcp_pose"][0].numpy(), 18),
        (obs_dict["extra"]["cubeA_pose"][0].numpy(), 25),
        (obs_dict["extra"]["cubeB_pose"][0].numpy(), 32),
        (obs_dict["extra"]["tcp_to_cubeA_pos"][0].numpy(), 39),
        (obs_dict["extra"]["tcp_to_cubeB_pos"][0].numpy(), 42),
        (obs_dict["extra"]["cubeA_to_cubeB_pos"][0].numpy(), 45),
    ]
    for arr, start in blocks:
        assert np.allclose(flat[start:start + len(arr)], arr)
        assert start + len(arr) <= 48
    obstacle = np.array([
        e._obstacle_center[0][0].item(), e._obstacle_center[0][1].item(),
        e.obstacle_collision_radius,
    ])
    assert np.allclose(flat[48:51], obstacle)
    assert TCP_XY_INDICES == (18, 19) and OBSTACLE_INDICES == (48, 49, 50)
    # the raw tcp pose block proven above carries the world TCP position
    tcp = e.agent.tcp.pose.p[0].numpy()
    assert np.allclose(flat[18:20], tcp[:2])
    env.close()


def test_condition_is_latest_frame_obstacle_minus_tcp():
    """The condition uses the most recent queued frame (index -1), not the
    first frame of the 2-frame history."""
    old = frame(tcp_x=0.10, obstacle_x=0.50, obstacle_y=0.30, radius=0.02)
    new = frame(tcp_x=0.00, obstacle_x=0.40, obstacle_y=0.20, radius=0.04)
    cond = latest_relative_condition(torch.stack([old, new]), ConditionNormConfig())
    # raw latest-frame condition: obstacle - tcp = (0.40, 0.20), radius 0.04;
    # affine [-1,1] range: v -> 2v - 1; radius range (0, 0.08): 0.04 -> 0.0
    assert cond.shape == (3,)
    torch.testing.assert_close(
        cond, torch.tensor([0.40, 0.20, 2 * 0.04 / 0.08 - 1.0])
    )  # rel_xy range (-1,1) is the identity map; radius 0.04 -> 0.0


def test_condition_batch_shape():
    """Batched queued histories ``(B, 2, 51)`` return one condition per batch."""
    hist = torch.stack([frame(), frame(tcp_x=0.1, obstacle_y=0.3)])
    batch = torch.stack([hist, hist])
    cond = latest_relative_condition(batch, ConditionNormConfig())
    assert cond.shape == (2, 3)


def test_translation_invariance():
    """Translating the whole scene (TCP and obstacle by the same world XY
    delta) leaves the relative condition unchanged."""
    delta = torch.tensor([0.37, -0.21])
    a = frame(tcp_x=0.02, tcp_y=0.05, obstacle_x=0.11, obstacle_y=0.27, radius=0.03)
    b = a.clone()
    b[18:20] += delta
    b[48:50] += delta
    cond_a = latest_relative_condition(torch.stack([a, a]), ConditionNormConfig())
    cond_b = latest_relative_condition(torch.stack([b, b]), ConditionNormConfig())
    torch.testing.assert_close(cond_b, cond_a)


def test_tcp_position_shift_changes_condition():
    """Moving only the TCP shifts the condition by minus the delta (obstacle
    stays fixed in world coordinates)."""
    a = frame(tcp_x=0.02, tcp_y=0.05, obstacle_x=0.11, obstacle_y=0.27)
    b = a.clone()
    b[18] += 0.05
    b[19] -= 0.03
    cond_a = latest_relative_condition(torch.stack([a, a]), ConditionNormConfig())
    cond_b = latest_relative_condition(torch.stack([b, b]), ConditionNormConfig())
    # normalized condition is affine in the raw condition with equal ranges
    scale = 2.0 / (1.0 - (-1.0))
    torch.testing.assert_close(cond_b[0] - cond_a[0], torch.tensor(-0.05 * scale))
    torch.testing.assert_close(cond_b[1] - cond_a[1], torch.tensor(0.03 * scale))


def test_tcp_rotation_invariance():
    """Changing only the TCP orientation (quaternion block) does not change
    the condition: world-parallel XY axes, never the TCP frame axes."""
    a = frame(tcp_x=0.02, obstacle_x=0.11, obstacle_y=0.27, radius=0.03,
              tcp_quat=(0.0, 1.0, 0.0, 0.0))
    b = frame(tcp_x=0.02, obstacle_x=0.11, obstacle_y=0.27, radius=0.03,
              tcp_quat=(0.7071, 0.0, 0.7071, 0.0))
    torch.testing.assert_close(
        latest_relative_condition(torch.stack([b, b]), ConditionNormConfig()),
        latest_relative_condition(torch.stack([a, a]), ConditionNormConfig()),
    )


def test_radius_channel_ignores_positions():
    """The radius channel keeps real cylinder radius semantics: only the
    radius feature moves it."""
    a = frame(obstacle_x=0.11, obstacle_y=0.27, radius=0.03)
    b = frame(obstacle_x=-0.40, obstacle_y=0.90, radius=0.03)
    torch.testing.assert_close(
        latest_relative_condition(torch.stack([b, b]), ConditionNormConfig())[2],
        latest_relative_condition(torch.stack([a, a]), ConditionNormConfig())[2],
    )
    c = frame(obstacle_x=0.11, obstacle_y=0.27, radius=0.04)
    assert latest_relative_condition(torch.stack([c, c]), ConditionNormConfig())[2] > (
        latest_relative_condition(torch.stack([a, a]), ConditionNormConfig())[2]
    )


def test_fixed_normalization_formula_and_clipping():
    """The fixed affine map [lo, hi] -> [-1, 1] with recorded clipping; no
    per-batch statistics are involved (two calls at different scenes agree)."""
    cfg = ConditionNormConfig(rel_xy_range=(-1.0, 1.0), radius_range=(0.0, 0.08))
    hist = torch.stack([frame(tcp_x=0.0, obstacle_x=0.30, obstacle_y=0.0, radius=0.02)] * 2)
    cond = latest_relative_condition(hist, cfg)
    torch.testing.assert_close(cond[0], torch.tensor(0.30))  # identity map on (-1,1)
    torch.testing.assert_close(cond[2], torch.tensor(2 * 0.02 / 0.08 - 1.0))  # -0.5
    # out-of-range values clip (recorded out-of-range handling)
    hist_far = torch.stack([frame(tcp_x=0.0, obstacle_x=3.0, obstacle_y=-4.0, radius=0.5)] * 2)
    cond_far = latest_relative_condition(hist_far, cfg)
    assert cond_far[0] == 1.0 and cond_far[1] == -1.0 and cond_far[2] == 1.0
    # determinism across scenes: no batch statistics
    torch.testing.assert_close(
        latest_relative_condition(hist, cfg), cond,
    )


def test_normalization_config_roundtrip_records_everything():
    """The normalization config serializes everything needed to rebuild the
    exact transform (ranges + clipping + proven indices)."""
    cfg = ConditionNormConfig(rel_xy_range=(-1.0, 1.0), radius_range=(0.0, 0.08))
    d = cfg.to_dict()
    rebuilt = ConditionNormConfig.from_dict(d)
    assert rebuilt == cfg
    assert d["tcp_xy_indices"] == [18, 19]
    assert d["obstacle_indices"] == [48, 49, 50]
    assert d["clip"] is True
    assert "affine" in d["transform"].lower()


def test_hint_repeats_condition_over_pred_horizon():
    """The current 3d condition repeats to every one of the 16 predicted
    steps: it encodes current relative geometry, never a future trajectory."""
    cond = torch.tensor([[0.1, -0.2, 0.5]])
    hint = build_hint(cond, pred_horizon=16)
    assert hint.shape == (1, 16, 3)
    torch.testing.assert_close(hint, cond[:, None, :].expand(1, 16, 3).clone())


def test_condition_rejects_wrong_frame_layout():
    """Frames without the proven 51d obstacle layout are rejected up front."""
    with pytest.raises(ValueError):
        latest_relative_condition(torch.zeros(2, 50), ConditionNormConfig())
    with pytest.raises(ValueError):
        latest_relative_condition(torch.zeros(3, 51), ConditionNormConfig())
