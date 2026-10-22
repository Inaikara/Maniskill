import gymnasium as gym
import pytest
import torch

from examples.baselines.constrain.pickcube_obstacle import ENV_ID, OBSTACLE_STATE_KEY


def make_env(**kwargs):
    params = dict(
        num_envs=1,
        obs_mode="state_dict",
        sim_backend="physx_cpu",
        control_mode="pd_ee_delta_pose",
        max_episode_steps=500,
    )
    params.update(kwargs)
    return gym.make(ENV_ID, **params)


def clone(value):
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return value.clone()


def test_pickcube_native_task_layout_safety_and_state_restore():
    env = make_env()
    try:
        options = {
            "cube_xy": [0.12, -0.04],
            "goal_pos": [-0.12, 0.08, 0.12],
            "obstacle_center": [0.0, 0.2],
            "obstacle_enabled": True,
        }
        obs0, info0 = env.reset(seed=23, options=options)
        base = env.unwrapped
        assert base.robot_uids == "panda"
        assert base.control_mode == "pd_ee_delta_pose"
        assert base.action_space.shape == (7,)
        assert info0["success"].shape == (1,)
        assert torch.allclose(obs0["extra"][OBSTACLE_STATE_KEY], torch.tensor([[0.0, 0.2, 0.03]]))
        assert torch.allclose(base.cube.pose.p[0, :2], torch.tensor(options["cube_xy"]))
        assert torch.allclose(base.goal_site.pose.p[0], torch.tensor(options["goal_pos"]))

        saved = clone(base.get_state_dict())
        action = torch.tensor([0.1, 0, 0, 0, 0.1, 0, 0.25])
        env.step(action)
        base.set_state_dict(saved)
        restored_obs = base.get_obs()
        for section, value in obs0.items():
            for key, expected in value.items():
                assert torch.allclose(restored_obs[section][key], expected, atol=1e-6), key
        restored_state = clone(base.get_state_dict())
        assert restored_state.get("controller", {}) == saved.get("controller", {})
        _, _, _, _, info = env.step(action)
        after_first = clone(base.get_state_dict())
        base.set_state_dict(saved)
        env.step(action)
        after_restored = base.get_state_dict()
        for category in ("actors", "articulations"):
            for key, value in after_first[category].items():
                assert torch.allclose(value, after_restored[category][key], atol=1e-5)
        base.set_state_dict(saved)
        _, _, _, _, info = env.step(torch.zeros(7))
        assert not bool(info["obstacle_collision"].item())
        assert not bool(info["obstacle_crossing"].item())

        # Restore native success conditions: no obstacle/release condition can
        # replace PickCube's placement + static-robot success semantics.
        state = clone(saved)
        state["actors"]["cube"][0, :3] = state["pickcube_goal"][0]
        base.set_state_dict(state)
        base.agent.robot.set_qvel(torch.zeros_like(base.agent.robot.get_qvel()))
        native = base.evaluate()
        assert bool(native["success"].item())
        assert bool(native["is_obj_placed"].item())
        assert "is_grasped" in native
    finally:
        env.close()


def test_obstacle_contact_and_swept_forbidden_disk_are_sticky_and_disabled_is_safe():
    env = make_env()
    try:
        env.reset(seed=3, options={"obstacle_center": [0.0, 0.0], "obstacle_enabled": True})
        base = env.unwrapped
        safe_state = clone(base.get_state_dict())
        state = clone(base.get_state_dict())
        state["actors"]["cube"][0, :3] = torch.tensor([0.045, 0.0, 0.02])
        base.set_state_dict(state)
        _, _, _, _, info = env.step(torch.zeros(7))
        assert bool(info["obstacle_collision"].item())
        assert not bool(info["success"].item())

        env.reset(seed=3, options={"obstacle_center": [0.0, 0.0], "obstacle_enabled": True})
        base = env.unwrapped
        state = clone(base.get_state_dict())
        state["actors"]["cube"][0, :3] = torch.tensor([-0.15, 0.0, 0.4])
        base.set_state_dict(state)
        base._grasp_started[:] = True
        base._prev_cube_xy[:] = torch.tensor([[-0.15, 0.0]])
        state = clone(base.get_state_dict())
        state["actors"]["cube"][0, :3] = torch.tensor([0.15, 0.0, 0.4])
        base.set_state_dict(state)
        # set_state_dict rebases the path for restoration; emulate a real cube
        # displacement occurring during the following physics control step.
        base._prev_cube_xy[:] = torch.tensor([[-0.15, 0.0]])
        _, _, _, _, info = env.step(torch.zeros(7))
        assert bool(info["obstacle_crossing"].item())
        assert not bool(info["success"].item())
        base.set_state_dict(safe_state)
        restored = base.get_state_dict()
        assert torch.equal(restored[OBSTACLE_STATE_KEY], safe_state[OBSTACLE_STATE_KEY])
        assert not bool(base.evaluate()["obstacle_crossing"].item())

        _, info = env.reset(seed=3, options={"obstacle_center": [0.0, 0.0], "obstacle_enabled": False})
        assert not bool(info["obstacle_collision"].item())
        assert not bool(info["obstacle_crossing"].item())
        assert not bool(info["obstacle_enabled"].item())
    finally:
        env.close()


def test_robot_link_contact_is_collected_and_native_success_survives_safe_obstacle():
    env = make_env()
    try:
        _, info = env.reset(
            seed=9,
            options={"obstacle_center": [-0.615, 0.0], "obstacle_enabled": True},
        )
        base = env.unwrapped
        for _ in range(2):
            _, _, _, _, info = env.step(torch.zeros(7))
        assert bool(info["obstacle_collision"].item())
        assert not bool(info["success"].item())

        # Clear the episode flags by resetting to a safe obstacle, then place
        # the native cube at the native target and stop the robot.
        base.reset(seed=9, options={"obstacle_center": [0.25, 0.25], "obstacle_enabled": True})
        state = clone(base.get_state_dict())
        state["actors"]["cube"][0, :3] = state["pickcube_goal"][0]
        base.set_state_dict(state)
        base.agent.robot.set_qvel(torch.zeros_like(base.agent.robot.get_qvel()))
        native = base.evaluate()
        assert bool(native["native_success"].item())
        assert bool(native["success"].item())
        assert "is_grasped" in native
    finally:
        env.close()


def test_pickcube_obstacle_rejects_unsupported_simulation():
    with pytest.raises(NotImplementedError):
        make_env(sim_backend="physx_gpu")
    with pytest.raises(NotImplementedError):
        make_env(num_envs=2)


def test_time_limit_is_truncation_not_native_success():
    env = make_env(max_episode_steps=1)
    try:
        env.reset(seed=13, options={"obstacle_enabled": False})
        _, _, terminated, truncated, info = env.step(torch.zeros(7))
        assert bool(truncated)
        assert not bool(terminated)
        assert not bool(info["native_success"].item())
        assert not bool(info["success"].item())
    finally:
        env.close()
