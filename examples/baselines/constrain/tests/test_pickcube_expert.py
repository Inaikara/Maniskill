import torch

from examples.baselines.constrain.pickcube_expert import (
    make_pickcube_env,
    restore_execution_state,
    run_pickcube_expert,
    snapshot_execution_state,
)


def test_pickcube_expert_returns_full_rollout_and_preserves_pose_action_channels():
    env = make_pickcube_env()
    try:
        result = run_pickcube_expert(
            env,
            cube_xy=(-0.04, 0.05),
            goal_xy=(0.12, 0.20),
            obstacle_enabled=False,
        )
        assert result["actions"].ndim == 2
        assert result["actions"].shape[1] == 7
        assert len(result["states"]) == len(result["actions"]) + 1
        assert len(result["observations"]) == len(result["actions"]) + 1
        assert result["trace"]["swept_xy"]
        assert torch.isfinite(torch.as_tensor(result["actions"])).all()
        assert torch.any(torch.abs(torch.as_tensor(result["actions"][:, 3:6])) > 0)
    finally:
        env.close()


def test_native_success_and_safety_violation_on_same_control_step_is_not_safe_success():
    env = make_pickcube_env()
    try:
        env.reset(
            seed=51,
            options={
                "cube_xy": [-0.1, 0.1],
                "goal_pos": [-0.1, 0.1, 0.02],
                "obstacle_enabled": True,
                "obstacle_center": [0.0, 0.1],
            },
        )
        base = env.unwrapped
        state = base.get_state_dict()
        state["actors"]["cube"][0, :3] = state["pickcube_goal"][0]
        base.set_state_dict(state)
        base.agent.robot.set_qvel(torch.zeros_like(base.agent.robot.get_qvel()))
        base._grasp_started[:] = True
        base._prev_cube_xy[:] = torch.tensor([[0.1, 0.1]])
        _, _, terminated, truncated, info = env.step(torch.zeros(7))
        assert not bool(terminated)
        assert not bool(truncated)
        assert bool(info["obstacle_crossing"].item())
        # Establish native success in the same still-live episode after the
        # real physics step registered the forbidden swept segment.
        base.goal_site.set_pose(base.cube.pose)
        base.agent.robot.set_qvel(torch.zeros_like(base.agent.robot.get_qvel()))
        info = base.evaluate()
        assert bool(info["native_success"].item())
        assert bool(info["obstacle_crossing"].item())
        assert not bool(info["success"].item())
    finally:
        env.close()


def test_explicit_controller_snapshot_restores_same_action_continuation():
    env = make_pickcube_env()
    try:
        env.reset(seed=7, options={"obstacle_enabled": False})
        base = env.unwrapped
        snapshot = snapshot_execution_state(env)
        action = torch.tensor([0.2, -0.1, 0.1, 0.15, 0.0, -0.1, 1.0])
        env.step(action)
        first = base.get_state_dict()
        restore_execution_state(env, snapshot)
        _assert_nested_tensors_equal(snapshot[1], base.agent.controller.get_state())
        env.step(action)
        replayed = base.get_state_dict()
        for group in ("actors", "articulations"):
            for key, expected in first[group].items():
                assert torch.allclose(expected, replayed[group][key], atol=1e-5)
    finally:
        env.close()


def _assert_nested_tensors_equal(expected, actual):
    assert expected.keys() == actual.keys()
    for key in expected:
        if isinstance(expected[key], dict):
            _assert_nested_tensors_equal(expected[key], actual[key])
        else:
            assert torch.equal(expected[key], actual[key]), key
