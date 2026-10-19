"""Separate physical controller-mapping check for the base DP adapter.

The numeric 4d -> 7d dimensional checks in ``test_base_dp.py`` verify tensor
shapes only; this file verifies at the real physics seam that the mapped 7d
``pd_ee_delta_pose`` action actually makes the official controller realize the
base policy's semantics: a world/root-frame xyz end-effector delta, zero
rotation, and the gripper command.
"""

import gymnasium as gym
import numpy as np
import torch

from examples.baselines.constrain.base_dp import BaseDPAdapter


def make_env(seed):
    import examples.baselines.constrain.stackcube_obstacle  # noqa: F401 (registers env)

    env = gym.make(
        "StackCubeObstacle-v1",
        reconfiguration_freq=1,
        robot_uids="panda_wristcam",
        control_mode="pd_ee_delta_pose",
        obs_mode="state",
        sim_backend="physx_cpu",
        num_envs=1,
    )
    env.reset(seed=seed)
    env.action_space.seed(0)
    return env


def tcp_state(env):
    agent = env.unwrapped.agent
    pose = agent.tcp.pose
    fingers = agent.robot.get_qpos()[0, -2].item()
    return pose.p[0].cpu().numpy().copy(), pose.q[0].cpu().numpy().copy(), fingers


def mapped(env, base_action):
    adapter = BaseDPAdapter(device="cpu")
    action = adapter.map_action_ee_delta_pose(torch.as_tensor(base_action))
    return action.numpy().astype(np.float32)


def test_zero_rotation_maps_to_constant_tcp_orientation():
    """Zero rotation deltas keep the TCP orientation constant through real
    controller dynamics while the arm translates."""
    env = make_env(seed=23)
    _, q0, _ = tcp_state(env)
    action = mapped(env, [0.0, 0.0, 0.3, 0.0])
    q = q0
    for _ in range(10):
        env.step(action)
        _, q, _ = tcp_state(env)
    dot = float(np.abs(np.dot(q0, q)))
    assert dot >= 1.0 - 1e-4, f"TCP orientation drifted (dot={dot})"
    env.close()


def test_xyz_delta_maps_to_axis_predominant_translation_and_gripper():
    """A pure +z base delta drives the TCP predominantly up through real IK/PD
    dynamics, and the gripper component opens/closes the real fingers."""
    env = make_env(seed=23)
    p0, _, f_open0 = tcp_state(env)
    action_up = mapped(env, [0.0, 0.0, 0.3, 1.0])
    for _ in range(10):
        env.step(action_up)
    p_up, q_up, f_open1 = tcp_state(env)
    dp = p_up - p0
    assert dp[2] > 0.02, f"expected grip frame to rise along +z, got dp={dp}"
    assert abs(dp[0]) < 0.01 and abs(dp[1]) < 0.01, f"cross-axis drift {dp}"
    assert f_open1 >= 0.035, f"open command did not keep fingers open: {f_open1}"

    f_open_end = f_open1
    action_close = mapped(env, [0.0, 0.0, 0.0, -1.0])
    for _ in range(10):
        env.step(action_close)
    _, _, f_close = tcp_state(env)
    assert f_close < f_open_end - 0.01, (
        f"close command did not close fingers: open={f_open_end} closed={f_close}"
    )
    env.close()
