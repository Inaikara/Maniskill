import numpy as np
import pytest
import torch

from examples.baselines.constrain.placesphere_obstacle import (
    PlaceSphereObstacleEnv,
    segment_enters_forbidden_disk,
)
from examples.baselines.constrain.placesphere_pilot import _continuous_sweep_clearance


def test_sphere_forbidden_disk_uses_native_radius_and_checks_post_release_segments():
    # Native sphere radius is 0.02 m; cylinder radius 0.03 m and safety margin
    # 0.005 m yield an exact 0.055 m forbidden center distance.
    assert segment_enters_forbidden_disk(
        [-0.2, 0.0], [0.2, 0.0], [0.0, 0.0], 0.055
    )
    assert not segment_enters_forbidden_disk(
        [-0.2, 0.06], [0.2, 0.06], [0.0, 0.0], 0.055
    )
    # Endpoint sampling would miss a sphere that passes between controls.
    assert _continuous_sweep_clearance(
        [0.0, 0.0], np.asarray([[-0.2, 0.0], [0.2, 0.0]])
    ) == pytest.approx(0.0)


def test_place_sphere_env_preserves_native_geometry_and_success_contract():
    import gymnasium as gym

    import examples.baselines.constrain.placesphere_obstacle

    env = gym.make(
        "PlaceSphereObstacle-v1", num_envs=1, obs_mode="state_dict",
        sim_backend="physx_cpu", control_mode="pd_ee_delta_pose",
    )
    try:
        env.reset(seed=4)
        task = env.unwrapped
        assert task.radius == pytest.approx(0.02)
        assert task.inner_side_half_len == pytest.approx(0.02)
        assert task.block_half_size == [0.0025, 0.025, 0.025]
        assert task.obstacle_collision_radius == pytest.approx(0.03)
        assert task.evaluate()["success"].shape == (1,)
        full_state = task.get_full_state_dict()
        state = full_state["environment"]
        assert "placesphere_obstacle" in state
        assert "placesphere_goal" in state
        env.step(torch.tensor([0.1, 0, 0, 0, 0.1, 0, 0.25]))
        task.set_full_state_dict(full_state)
        restored = task.get_state_dict()
        for key in ("placesphere_obstacle", "placesphere_goal"):
            assert torch.equal(state[key], restored[key])
        assert torch.equal(state["actors"]["sphere"], restored["actors"]["sphere"])
        assert torch.equal(state["actors"]["bin"], restored["actors"]["bin"])
        assert task.agent.controller.get_state().keys() == full_state["controller"].keys()
    finally:
        env.close()


def test_table_bounds_include_full_container_and_layout_rejects_short_transport():
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSphereLayoutConfig,
        validate_placesphere_layout,
    )
    from examples.baselines.constrain.layout_geometry import sample_same_side_waypoint

    config = PlaceSphereLayoutConfig(table_bounds=(-0.25, -0.25, 0.25, 0.25))
    assert not validate_placesphere_layout(
        [-0.1, 0.0], [-0.02, 0.0], [-0.615, 0.0], config,
        container_half_extent=0.025,
    )[0]
    assert validate_placesphere_layout(
        [-0.15, 0.05], [0.12, 0.05], [-0.615, -0.1], config,
        container_half_extent=0.025,
    )[0]
    assert not validate_placesphere_layout(
        [-0.15, 0.0], [0.23, 0.0], [-0.615, -0.1], config,
        container_half_extent=0.025,
    )[0]
    diagonal_start, diagonal_goal, base = [-0.15, -0.1], [0.1, 0.1], [-0.615, 0.0]
    valid, _, _, layout_config = validate_placesphere_layout(
        diagonal_start, diagonal_goal, base, config, container_half_extent=0.025)
    assert valid
    waypoint = sample_same_side_waypoint(
        diagonal_start, diagonal_goal, base, np.random.default_rng(11), layout_config)
    assert waypoint.waypoint_xy is not None
    assert np.cross(np.subtract(diagonal_goal, diagonal_start),
                    np.subtract(waypoint.waypoint_xy, diagonal_start)) * np.cross(
                        np.subtract(diagonal_goal, diagonal_start),
                        np.subtract(base, diagonal_start)) > 0


def test_grasped_release_then_rolling_through_disk_is_sticky_and_native_success_is_unmodified():
    import gymnasium as gym
    import examples.baselines.constrain.placesphere_obstacle

    env = gym.make("PlaceSphereObstacle-v1", num_envs=1, obs_mode="state_dict",
                   sim_backend="physx_cpu", control_mode="pd_ee_delta_pose")
    try:
        env.reset(seed=7, options={"obstacle_enabled": True, "obstacle_center": [0.0, 0.0]})
        task = env.unwrapped
        # A real post-release sphere pose displacement is swept from the previous
        # physics sample, even though the grasp flag has already become false.
        task._grasp_started[:] = True
        task._prev_sphere_xy[:] = torch.tensor([[-0.12, 0.0]], device=task.device)
        pose = task.obj.pose.raw_pose.clone()
        pose[0, :2] = torch.tensor([0.12, 0.0], device=task.device)
        task.obj.set_pose(type(task.obj.pose).create(pose))
        task._check_sphere_segment()
        assert bool(task._forbidden_crossing.item())
        task._prev_sphere_xy[:] = task.obj.pose.p[:, :2]
        pose = task.obj.pose.raw_pose.clone()
        pose[0, :2] = torch.tensor([0.12, 0.12], device=task.device)
        task.obj.set_pose(type(task.obj.pose).create(pose))
        task._check_sphere_segment()
        assert bool(task._forbidden_crossing.item())

        # Native success is only the native placement/static/ungrasped result;
        # with a sticky safety violation the wrapper rejects it, without changing
        # the native result or geometry.
        pose = task.bin.pose.p.clone()
        pose[0, 2] += task.radius + task.block_half_size[0]
        task.obj.set_pose(type(task.obj.pose).create_from_pq(pose, task.obj.pose.q))
        task.agent.robot.set_qvel(torch.zeros_like(task.agent.robot.get_qvel()))
        native = super(type(task), task).evaluate()
        wrapped = task.evaluate()
        assert bool(native["success"].item())
        assert not bool(native["is_obj_grasped"].item())
        assert bool(wrapped["native_success"].item())
        assert not bool(wrapped["success"].item())
    finally:
        env.close()


def test_robot_links_contact_static_cylinder_is_recorded():
    import gymnasium as gym
    import examples.baselines.constrain.placesphere_obstacle

    env = gym.make("PlaceSphereObstacle-v1", num_envs=1, obs_mode="state_dict",
                   sim_backend="physx_cpu", control_mode="pd_ee_delta_pose")
    try:
        env.reset(seed=9, options={"obstacle_enabled": True, "obstacle_center": [-0.615, 0.0]})
        _, _, _, _, info = env.step(torch.zeros(7))
        assert bool(info["obstacle_collision"].item())
        assert not bool(info["success"].item())
    finally:
        env.close()


def test_signed_xyz_rotation_action_uses_runtime_negative_controller_scale():
    from examples.baselines.constrain.pickcube_expert import scale_native_euler_action

    action = scale_native_euler_action(
        [0.04, -0.03, 0.02], [-0.08, -0.06, 0.04], 0.35
    )
    np.testing.assert_allclose(action, [-0.35, 0.35, 0.35])


def test_policy_state_keeps_native_sphere_and_container_fields_without_obstacle_hint():
    import gymnasium as gym
    from examples.baselines.constrain.placesphere_pilot import (
        _policy_observation, _policy_observation_schema,
    )
    import examples.baselines.constrain.placesphere_obstacle

    env = gym.make("PlaceSphereObstacle-v1", num_envs=1, obs_mode="state",
                   sim_backend="physx_cpu", control_mode="pd_ee_delta_pose")
    try:
        _, info = env.reset(seed=14, options={"obstacle_enabled": True,
                                               "obstacle_center": [0.1, 0.1]})
        task = env.unwrapped
        vector = _policy_observation(task, info)
        fields = _policy_observation_schema(task, info)
        assert vector.shape == (39,)
        paths = [field["path"] for field in fields]
        assert "/extra/obj_pose" in paths
        assert "/extra/bin_pos" in paths
        assert not any("placesphere_obstacle" in path for path in paths)
        assert sum(field["flattened_dimension"] for field in fields) == len(vector)
    finally:
        env.close()


def test_dataset_rejects_missing_controller_or_timeout_terminal_evidence(tmp_path):
    import h5py
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotDataset

    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("traj_0")
        group.attrs["kind"] = "direct"
        group.create_dataset("obs", data=np.zeros((2, 39), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
        group.create_dataset("success", data=np.array([0, 1], dtype=np.uint8))
        group.create_dataset("terminated", data=np.array([0, 1], dtype=np.uint8))
        group.create_dataset("truncated", data=np.zeros(2, dtype=np.uint8))
        group.create_group("env_states").create_dataset("native", data=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="controller"):
        PlaceSpherePilotDataset(path)


def test_collector_rejects_nonpositive_obstacle_safety_geometry_before_output(tmp_path):
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSpherePilotConfig, collect_pilot,
    )

    output = tmp_path / "invalid-safety-geometry"
    config = PlaceSpherePilotConfig(safety_margin=-0.001)
    with pytest.raises(ValueError, match="obstacle geometry"):
        collect_pilot(output, config, attempt_runner=lambda *args: (None, "unused"))
    assert not output.exists()


def test_formal_targets_are_independent_and_default_to_pilot_target():
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig

    config = PlaceSpherePilotConfig(target_per_class=20, direct_target=1000,
                                    obstacle_target=100)
    assert config.target("direct") == 1000
    assert config.target("obstacle") == 100
    assert PlaceSpherePilotConfig(target_per_class=20).target("direct") == 20


def test_collector_freezes_contract_before_attempt_and_uses_each_target(tmp_path):
    import json
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSpherePilotConfig, collect_pilot,
    )

    output = tmp_path / "formal"
    calls = []

    def accepted(kind, seed, rng, config):
        contract = json.loads((output / "frozen_contract.json").read_text())
        assert contract["limits"]["targets_per_class"] == {"direct": 2, "obstacle": 1}
        calls.append(kind)
        info = {"native_success": [True], "success": [True],
                "is_obj_on_bin": [True], "is_obj_static": [True],
                "is_obj_grasped": [False], "obstacle_collision": [False],
                "obstacle_crossing": [False]}
        return {"actions": np.zeros((1, 7), np.float32),
                "obs": np.zeros((2, 39), np.float32),
                "raw_obs": np.zeros((2, 42), np.float32),
                "states": [{"native": np.zeros((1, 3), np.float32)}] * 2,
                "controller_states": [{"controller": np.zeros((1,), np.float32)}] * 2,
                "infos": [info, info], "terminated": np.array([0, 1], np.uint8),
                "truncated": np.zeros(2, np.uint8), "native_success": True,
                "safe_success": True, "layout": {"start_xy": [0, 0],
                    "container_xy": [1, 1]}, "trace": {}}, None

    config = PlaceSpherePilotConfig(direct_target=2, obstacle_target=1,
                                    max_attempts_per_class=3)
    report = collect_pilot(output, config, accepted)
    assert report["complete"]
    assert calls == ["direct", "direct", "obstacle"]
    assert [json.loads(line)["class"] for line in
            (output / "attempts.jsonl").read_text().splitlines()] == calls
    manifest = json.loads((output / "trajectory.json").read_text())
    assert len(manifest["episodes"]) == 3


def test_native_seven_channel_action_limits_allow_full_gripper_range():
    from examples.baselines.constrain.placesphere_pilot import _actions_are_legal

    actions = np.zeros((2, 7), dtype=np.float32)
    actions[0, 6], actions[1, 6] = -1.0, 1.0
    assert _actions_are_legal(actions)
    actions[0, 0] = 0.351
    assert not _actions_are_legal(actions)


def test_serialized_native_state_restore_uses_recorded_float_tolerance():
    from examples.baselines.constrain.placesphere_formal import _tree_close

    saved = {"robot": np.array([[0.0, 1.0]], dtype=np.float32)}
    restored = {"robot": np.array([[1e-8, 1.0]], dtype=np.float64)}
    assert _tree_close(saved, restored, atol=1e-7)
    restored["robot"][0, 0] = 2e-7
    assert not _tree_close(saved, restored, atol=1e-7)


def test_training_window_loader_rejects_time_limit_instead_of_native_terminal(tmp_path):
    import h5py
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotDataset

    path = tmp_path / "time-limit.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("traj_0")
        group.attrs["kind"] = "direct"
        group.create_dataset("obs", data=np.zeros((2, 39), dtype=np.float32))
        group.create_dataset("actions", data=np.zeros((1, 7), dtype=np.float32))
        group.create_dataset("success", data=np.array([0, 1], dtype=np.uint8))
        group.create_dataset("terminated", data=np.zeros(2, dtype=np.uint8))
        group.create_dataset("truncated", data=np.array([0, 1], dtype=np.uint8))
        group.create_group("env_states").create_dataset("native", data=np.zeros((2, 3)))
        group.create_group("controller_states").create_dataset("native", data=np.zeros((2, 1)))
    with pytest.raises(ValueError, match="time limit"):
        PlaceSpherePilotDataset(path)
