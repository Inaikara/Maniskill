import gymnasium as gym
import numpy as np
from transforms3d.euler import euler2quat
from mani_skill.utils.structs import Pose

from examples.baselines.constrain.peg_orientation import (
    ENV_ID,
    build_orientation_hint,
)


def test_orientation_hint_uses_peg_minus_target_yaw_with_wrapped_period():
    hint = build_orientation_hint(np.deg2rad(10), np.deg2rad(350))
    np.testing.assert_allclose(hint, [np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20))])


def test_registered_env_hides_target_orientation_and_restores_scene_state():
    env = gym.make(ENV_ID, obs_mode="state", sim_backend="physx_cpu", reconfiguration_freq=1)
    try:
        env.reset(seed=11, options={"target_yaw": np.pi / 2})
        task = env.unwrapped
        obs = task.get_obs(unflattened=True)
        state = obs["extra"]
        assert "box_hole_pose" not in state
        assert "peg_pose" in state and "hole_center" in state
        assert "peg_half_size" in state and "box_hole_radius" in state
        assert task.action_space.shape == (7,)
        assert task.agent.controller.controllers["arm"].action_space.shape == (6,)
        assert task.agent.controller.controllers["gripper"].action_space.shape == (1,)
        saved = task.get_state_dict()
        peg_pose = task.peg.pose.raw_pose.clone()
        task.reset(seed=12)
        task.set_state_dict(saved)
        np.testing.assert_allclose(task.peg.pose.raw_pose.cpu().numpy(), peg_pose.cpu().numpy())
        np.testing.assert_allclose(task.get_obs(unflattened=True)["extra"]["hole_center"].cpu().numpy(),
                                   state["hole_center"].cpu().numpy())
    finally:
        env.close()


def test_task_orientation_hint_tracks_current_world_peg_axis_and_restoration():
    env = gym.make(ENV_ID, obs_mode="state", sim_backend="physx_cpu", reconfiguration_freq=1)
    try:
        env.reset(seed=11, options={"target_yaw": np.pi / 2})
        task = env.unwrapped
        peg_position = task.peg.pose.p.clone()
        task.peg.set_pose(Pose.create_from_pq(
            peg_position, euler2quat(0, 0, np.deg2rad(80))
        ))
        before = task.get_orientation_hint()
        np.testing.assert_allclose(before, [np.sin(np.deg2rad(-10)), np.cos(np.deg2rad(-10))], atol=1e-6)
        saved = task.get_state_dict()

        task.peg.set_pose(Pose.create_from_pq(
            peg_position, euler2quat(0, 0, np.deg2rad(110))
        ))
        np.testing.assert_allclose(
            task.get_orientation_hint(),
            [np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20))], atol=1e-6,
        )
        task.set_state_dict(saved)
        np.testing.assert_allclose(task.get_orientation_hint(), before, atol=1e-6)
        obs = task.get_obs(unflattened=True)["extra"]
        assert "orientation_hint" not in obs
        assert "box_hole_pose" not in obs
    finally:
        env.close()


def test_orientation_task_rejects_batched_environments():
    try:
        gym.make(ENV_ID, num_envs=2, obs_mode="state", sim_backend="physx_cpu")
    except ValueError as exc:
        assert "single environment" in str(exc)
    else:
        raise AssertionError("orientation task must reject batched environments")


def test_hole_center_sampling_is_independent_of_requested_target_yaw():
    env = gym.make(ENV_ID, obs_mode="state", sim_backend="physx_cpu", reconfiguration_freq=1)
    try:
        centers = []
        for yaw in (0.0, np.pi / 2):
            env.reset(seed=29, options={"target_yaw": yaw})
            centers.append(env.unwrapped.get_obs(unflattened=True)["extra"]["hole_center"].cpu().numpy())
        np.testing.assert_allclose(centers[0], centers[1], atol=1e-6, rtol=0)
    finally:
        env.close()


def test_full_base_observation_does_not_encode_target_yaw():
    env = gym.make(ENV_ID, obs_mode="state", sim_backend="physx_cpu", reconfiguration_freq=1)
    try:
        observations = []
        for yaw in (np.deg2rad(75), np.deg2rad(105)):
            observation, _ = env.reset(seed=2026112199, options={"target_yaw": yaw})
            observations.append(observation.cpu().numpy())
        np.testing.assert_allclose(observations[0], observations[1], atol=1e-6, rtol=0)
    finally:
        env.close()
