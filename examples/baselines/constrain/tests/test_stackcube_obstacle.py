"""Behavioural tests for the StackCubeObstacle task.

Tests are written at the environment / recording seam:
``reset``, ``step``, ``get_obs``, ``get_state_dict``, ``set_state_dict`` and the
official ``RecordEpisode`` HDF5/JSON output. They exercise real physx_cpu
physics, a real cylindrical collider and the real Panda grasps.
"""

import json

import gymnasium as gym
import h5py
import numpy as np
import pytest
import sapien
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from examples.baselines.constrain.obstacle_record import ObstacleRecordEpisode
from examples.baselines.constrain.stackcube_obstacle import (
    ENV_ID,
    OBSTACLE_STATE_KEY,
)
from mani_skill.utils.wrappers.record import RecordEpisode

OBSTACLE_CENTER = [0.0, 0.25]
OBSTACLE_RADIUS = 0.03
SAFETY_MARGIN = 0.005
CUBE_HALF = 0.02
# Forbidden disk radius = real cylinder radius + cube XY circumscribed radius + margin
CUBE_XY_RADIUS = float(np.hypot(CUBE_HALF, CUBE_HALF))
FORBIDDEN_RADIUS = OBSTACLE_RADIUS + CUBE_XY_RADIUS + SAFETY_MARGIN
# Panda root link (panda_link0) is fixed at this world XY.
PANDA_BASE_XY = [-0.615, 0.0]


def make_env(**kwargs):
    params = dict(
        num_envs=1,
        obs_mode="state_dict",
        sim_backend="physx_cpu",
        robot_uids="panda",
        obstacle_radius=OBSTACLE_RADIUS,
        safety_margin=SAFETY_MARGIN,
    )
    params.update(kwargs)
    return gym.make(ENV_ID, **params)


def _teleport_actor(env, actor_name, xyz):
    base = env.unwrapped
    state = base.get_state_dict()
    leaf = state["actors"][actor_name]
    leaf[0, :3] = torch.as_tensor(xyz, dtype=leaf.dtype, device=leaf.device)
    leaf[0, 7:13] = 0
    base.set_state_dict(state)


def _hold_action(base, close=False):
    qpos = base.agent.robot.get_qpos().clone()
    action = torch.zeros(base.single_action_space.shape)
    action[:7] = qpos[0, :7]
    action[-1] = -1.0 if close else 1.0
    return action


def _establish_real_grasp(env):
    """Drive a real Panda grasp on cubeA through the public env interface.

    The cube is placed at the current TCP pose, the gripper command is closed and
    the environment is stepped until the agent reports a grasp via contact forces.
    """
    base = env.unwrapped
    qpos = base.agent.robot.get_qpos().clone()
    base.cubeA.pose = base.agent.tcp.pose.raw_pose
    qpos[0, -2:] = 0.01
    base.agent.robot.set_qpos(qpos)
    action = _hold_action(base, close=True)
    for _ in range(3):
        env.step(action)
    assert bool(base.agent.is_grasping(base.cubeA).item())
    return action


def _assert_state_equal(state_a, state_b):
    assert set(state_a) == set(state_b)
    for key in state_a:
        if isinstance(state_a[key], dict):
            _assert_state_equal(state_a[key], state_b[key])
        else:
            assert torch.allclose(state_a[key], state_b[key], atol=1e-6), key


def _clone_state(state):
    return {
        key: _clone_state(value) if isinstance(value, dict) else value.clone()
        for key, value in state.items()
    }


def _np_segment_distance(point, start, end):
    point, start, end = map(np.asarray, (point, start, end))
    direction = end - start
    length_sq = float(direction @ direction)
    if length_sq == 0.0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip((point - start) @ direction / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + t * direction)))


def _flatten_obs(obs):
    out = {}
    for section, value in obs.items():
        if isinstance(value, dict):
            for key, tensor in value.items():
                out[f"{section}/{key}"] = np.asarray(tensor)
        else:
            out[section] = np.asarray(value)
    return out


def test_registration_and_deterministic_layout_restore():
    env = make_env()
    obs, _ = env.reset(seed=7, options={"obstacle_center": list(OBSTACLE_CENTER)})
    base = env.unwrapped

    state = base.get_state_dict()
    assert list(state.keys()) == ["actors", "articulations", OBSTACLE_STATE_KEY]
    assert state[OBSTACLE_STATE_KEY].shape == (1, 3)
    center_x, center_y, radius = state[OBSTACLE_STATE_KEY][0].tolist()
    assert center_x == pytest.approx(OBSTACLE_CENTER[0])
    assert center_y == pytest.approx(OBSTACLE_CENTER[1])
    assert radius == pytest.approx(OBSTACLE_RADIUS)
    # State radius must match the actually built collision and visual geometry.
    collision_shape = base.obstacle._bodies[0].collision_shapes[0]
    assert collision_shape.radius == pytest.approx(radius)
    assert base.obstacle_visual_radius == pytest.approx(radius)
    assert base.obstacle_collision_half_height == pytest.approx(0.25 / 2)

    obs0 = _flatten_obs(obs)
    for _ in range(3):
        env.step(env.action_space.sample())

    base.set_state_dict(state)
    obs1 = _flatten_obs(base.get_obs())
    assert set(obs0) == set(obs1)
    for key in obs0:
        assert np.allclose(obs0[key], obs1[key], atol=1e-5), key

    # Repeated reset with the same seed/options restores the same full layout.
    env.reset(seed=7, options={"obstacle_center": list(OBSTACLE_CENTER)})
    state_repeat = base.get_state_dict()
    _assert_state_equal(state, state_repeat)
    env.close()


def test_state_dict_keeps_original_prefix_and_appends_obstacle():
    env = make_env()
    env.reset(seed=3, options={"obstacle_center": list(OBSTACLE_CENTER)})
    state = env.unwrapped.get_state_dict()

    assert list(state.keys()) == ["actors", "articulations", OBSTACLE_STATE_KEY]
    # Original StackCube actor order is untouched.
    assert list(state["actors"].keys()) == ["table-workspace", "cubeA", "cubeB"]
    assert list(state["articulations"].keys()) == ["panda"]
    assert state["actors"]["cubeA"].shape == (1, 13)
    assert state[OBSTACLE_STATE_KEY].shape == (1, 3)
    env.close()


def test_obstacle_is_a_vertical_cylinder_of_configured_height():
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": list(OBSTACLE_CENTER)})
    base = env.unwrapped

    # The render geometry is a SAPIEN X-aligned unit cylinder; the body rotation
    # must turn its flat axis into world +z, otherwise the obstacle lies flat and
    # the vertical-axis forbidden-disk model is meaningless.
    render_body = base.obstacle._objs[0].find_component_by_type(
        sapien.render.RenderBodyComponent
    )
    shape = render_body.render_shapes[0]
    vertices = np.asarray(shape.get_vertices())
    unique = [len(np.unique(np.round(vertices[:, axis], 4))) for axis in range(3)]
    flat_axis = int(np.argmin(unique))
    body_rotation = quat2mat(
        np.asarray(base.obstacle._objs[0].pose.q, dtype=np.float64)
    )
    world_axis = body_rotation @ np.eye(3)[:, flat_axis]
    assert abs(float(world_axis @ np.array([0.0, 0.0, 1.0]))) == pytest.approx(
        1.0, abs=1e-6
    )

    # Behavioural: a cube dropped on the axis rests on the 0.25 m top, not on a
    # horizontal side (which would leave it around z = 0.175).
    top = 0.25
    _teleport_actor(
        env, "cubeA", [OBSTACLE_CENTER[0], OBSTACLE_CENTER[1], top + CUBE_HALF + 0.001]
    )
    for _ in range(12):
        _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert float(base.cubeA.pose.p[0, 2]) == pytest.approx(
        top + CUBE_HALF, abs=0.01
    )
    env.close()


def test_obstacle_fields_present_in_state_obs_every_step():
    env = make_env()
    obs, info = env.reset(seed=11, options={"obstacle_center": list(OBSTACLE_CENTER)})
    base = env.unwrapped
    expected = base.get_state_dict()[OBSTACLE_STATE_KEY]
    assert "obstacle" in obs["extra"]
    assert np.allclose(obs["extra"]["obstacle"].tolist(), expected.tolist(), atol=1e-6)

    for _ in range(3):
        obs, _, _, _, info = env.step(env.action_space.sample())
        expected = base.get_state_dict()[OBSTACLE_STATE_KEY]
        assert np.allclose(
            obs["extra"]["obstacle"].tolist(), expected.tolist(), atol=1e-6
        )
        assert info["obstacle_collision"].shape == (1,)
    env.close()


def test_rejects_gpu_and_multi_env_backend():
    # GPU simulation would silently skip the per-substep checks, so it must be
    # rejected up front rather than produce apparently-valid trajectories.
    with pytest.raises(NotImplementedError):
        make_env(sim_backend="physx_gpu")
    with pytest.raises(NotImplementedError):
        make_env(sim_backend="gpu")
    with pytest.raises(NotImplementedError):
        make_env(num_envs=2)
    with pytest.raises(NotImplementedError):
        make_env(num_envs=2, sim_backend="physx_cpu")


@pytest.mark.parametrize("radius", [0.02, 0.05])
def test_state_radius_matches_built_geometry_across_radii(radius, tmp_path):
    env = ObstacleRecordEpisode(
        make_env(obstacle_radius=radius),
        output_dir=str(tmp_path),
        trajectory_name=f"traj_{radius}",
        save_video=False,
        save_on_reset=True,
    )
    env.reset(seed=7, options={"obstacle_center": list(OBSTACLE_CENTER)})
    base = env.unwrapped
    state = base.get_state_dict()

    assert state[OBSTACLE_STATE_KEY][0, 2].item() == pytest.approx(radius)
    assert base.obstacle_collision_radius == pytest.approx(radius)
    assert base.obstacle_visual_radius == pytest.approx(radius)
    assert base.obstacle_collision_half_height == pytest.approx(0.125)
    assert base._forbidden_radius == pytest.approx(
        radius + CUBE_XY_RADIUS + SAFETY_MARGIN
    )

    # Exact recovery of every state field, including the real radius.
    base.set_state_dict(_clone_state(state))
    _assert_state_equal(base.get_state_dict(), state)

    for _ in range(2):
        env.step(torch.zeros(base.single_action_space.shape))
    env.reset(seed=8, options={"obstacle_center": list(OBSTACLE_CENTER)})
    env.close()

    meta = json.loads((tmp_path / f"traj_{radius}.json").read_text())
    config = meta["env_info"]["obstacle_config"]
    assert config["radius"] == pytest.approx(radius)
    assert config["height"] == pytest.approx(0.25)
    assert config["center_z"] == pytest.approx(0.125)
    assert config["units"] == "meters"
    assert config["state_field_order"] == ["center_x", "center_y", "radius"]
    assert config["safety_margin"] == pytest.approx(SAFETY_MARGIN)
    assert config["cube_xy_radius"] == pytest.approx(CUBE_XY_RADIUS)
    assert config["forbidden_radius"] == pytest.approx(
        radius + CUBE_XY_RADIUS + SAFETY_MARGIN
    )
    assert config["geometry_restore"]["radius"] == pytest.approx(radius)
    assert config["geometry_restore"]["half_length"] == pytest.approx(0.125)


def test_set_state_dict_rejects_mismatched_radius_before_mutation():
    env = make_env(obstacle_radius=0.02)
    env.reset(seed=7, options={"obstacle_center": list(OBSTACLE_CENTER)})
    base = env.unwrapped
    before = _clone_state(base.get_state_dict())

    mismatched = _clone_state(before)
    mismatched[OBSTACLE_STATE_KEY][0, 2] = 0.05
    mismatched["actors"]["cubeA"][0, :3] = torch.tensor(
        [0.2, 0.2, 0.4], dtype=mismatched["actors"]["cubeA"].dtype
    )

    with pytest.raises(ValueError, match="radius"):
        base.set_state_dict(mismatched)

    # Nothing may be mutated when the construction geometry does not match.
    _assert_state_equal(base.get_state_dict(), before)
    env.close()


def test_real_obstacle_collision_invalid_even_after_stacking():
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    base = env.unwrapped

    # Overlap the cube with the cylinder side (radius 0.03 + half size 0.02).
    _teleport_actor(env, "cubeA", [0.048, 0.1, 0.125])
    _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert bool(info["obstacle_collision"].item())
    assert not bool(info["success"].item())

    # A later successful stack must not erase the sticky collision flag.
    state = base.get_state_dict()
    state["actors"]["cubeB"][0, :3] = torch.tensor(
        [0.0, 0.0, 0.02], dtype=state["actors"]["cubeB"].dtype
    )
    state["actors"]["cubeB"][0, 7:13] = 0
    state["actors"]["cubeA"][0, :3] = torch.tensor(
        [0.0, 0.0, 0.06], dtype=state["actors"]["cubeA"].dtype
    )
    state["actors"]["cubeA"][0, 7:13] = 0
    base.set_state_dict(state)
    for _ in range(5):
        _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert bool(info["stack_success"].item())
    assert bool(info["obstacle_collision"].item())
    assert not bool(info["success"].item())
    env.close()


def test_robot_link_contact_sets_collision():
    # The static cylinder placed over the panda base link is a real robot contact.
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": PANDA_BASE_XY})
    base = env.unwrapped
    action = _hold_action(base)
    for _ in range(3):
        _, _, _, _, info = env.step(action)
    assert bool(info["obstacle_collision"].item())
    env.close()


def test_gripper_contact_sets_collision():
    env = make_env()
    env.reset(seed=7)
    base = env.unwrapped
    # Re-place the cylinder where the gripper currently is, then close the gripper.
    # Both fingers end up contacting opposite sides of the cylinder: the net contact
    # force nearly cancels, so this only registers if individual impulses are used.
    tcp_xy = base.agent.tcp.pose.p[0, :2].tolist()
    env.reset(seed=7, options={"obstacle_center": tcp_xy})
    action = _hold_action(base, close=True)
    for _ in range(8):
        _, _, _, _, info = env.step(action)
    assert bool(info["obstacle_collision"].item())
    env.close()


def test_table_and_stack_contacts_are_not_flagged():
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": [0.25, 0.25]})
    base = env.unwrapped
    action = _hold_action(base)
    for _ in range(4):
        _, _, _, _, info = env.step(action)
    # cubeA resting on the table is a normal contact and must not be flagged.
    assert not bool(info["obstacle_collision"].item())

    state = base.get_state_dict()
    state["actors"]["cubeB"][0, :3] = torch.tensor(
        [-0.1, -0.1, 0.02], dtype=state["actors"]["cubeB"].dtype
    )
    state["actors"]["cubeB"][0, 7:13] = 0
    state["actors"]["cubeA"][0, :3] = torch.tensor(
        [-0.1, -0.1, 0.06], dtype=state["actors"]["cubeA"].dtype
    )
    state["actors"]["cubeA"][0, 7:13] = 0
    base.set_state_dict(state)
    for _ in range(8):
        _, _, _, _, info = env.step(action)
    # cubeA/cubeB and cube/table contacts are not obstacle contacts.
    assert bool(info["stack_success"].item())
    assert not bool(info["obstacle_collision"].item())
    assert not bool(info["obstacle_crossing"].item())
    env.close()


def test_contact_impulse_threshold_controls_detection():
    # A huge impulse threshold suppresses an otherwise genuine overlap, proving
    # detection is driven by contact impulses rather than mere body proximity.
    env = make_env(contact_impulse_threshold=1e9)
    env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    base = env.unwrapped
    _teleport_actor(env, "cubeA", [0.048, 0.1, 0.125])
    for _ in range(3):
        _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert not bool(info["obstacle_collision"].item())
    env.close()


def test_over_top_crossing_without_contact_is_invalid():
    env = make_env()
    env.reset(seed=5, options={"obstacle_center": list(OBSTACLE_CENTER)})
    action = _establish_real_grasp(env)

    # Both sampled XY positions lie outside the forbidden disk; the segment
    # between them passes straight over the cylinder above its 0.25 m top.
    start_xy = [0.15, OBSTACLE_CENTER[1]]
    end_xy = [-0.15, OBSTACLE_CENTER[1]]
    for xy in (start_xy, end_xy):
        assert np.linalg.norm(np.array(xy) - OBSTACLE_CENTER) > FORBIDDEN_RADIUS
    assert (
        _np_segment_distance(OBSTACLE_CENTER, start_xy, end_xy) < FORBIDDEN_RADIUS
    )

    _teleport_actor(env, "cubeA", [start_xy[0], start_xy[1], 0.4])
    _, _, _, _, info = env.step(action)
    assert not bool(info["obstacle_crossing"].item())

    _teleport_actor(env, "cubeA", [end_xy[0], end_xy[1], 0.4])
    _, _, _, _, info = env.step(action)
    assert bool(info["obstacle_crossing"].item())
    assert not bool(info["obstacle_collision"].item())
    assert not bool(info["success"].item())
    env.close()


def test_forbidden_disk_radius_boundary():
    env = make_env()
    env.reset(seed=5, options={"obstacle_center": list(OBSTACLE_CENTER)})
    action = _establish_real_grasp(env)

    outside = [OBSTACLE_CENTER[0] + FORBIDDEN_RADIUS + 0.01, OBSTACLE_CENTER[1], 0.4]
    inside = [OBSTACLE_CENTER[0] + FORBIDDEN_RADIUS - 0.01, OBSTACLE_CENTER[1], 0.4]

    _teleport_actor(env, "cubeA", outside)
    _, _, _, _, info = env.step(action)
    assert not bool(info["obstacle_crossing"].item())

    _teleport_actor(env, "cubeA", inside)
    _, _, _, _, info = env.step(action)
    assert bool(info["obstacle_crossing"].item())
    assert not bool(info["obstacle_collision"].item())
    env.close()


def test_no_crossing_before_grasp():
    env = make_env()
    env.reset(seed=5, options={"obstacle_center": list(OBSTACLE_CENTER)})
    base = env.unwrapped
    action = _hold_action(base)

    # The cube sweeps across the forbidden disk while the gripper is open. The
    # 2D constraint only applies after a real grasp, so nothing may be flagged.
    for xy in [[-0.15, OBSTACLE_CENTER[1]], [0.15, OBSTACLE_CENTER[1]]]:
        _teleport_actor(env, "cubeA", [xy[0], xy[1], 0.3])
        _, _, _, _, info = env.step(action)
        assert not bool(base.agent.is_grasping(base.cubeA).item())
        assert not bool(info["obstacle_crossing"].item())
    env.close()


def test_safe_side_path_is_not_flagged_invalid():
    env = make_env()
    env.reset(seed=5, options={"obstacle_center": list(OBSTACLE_CENTER)})
    action = _establish_real_grasp(env)

    safe_waypoints = [
        [0.2, 0.25],
        [0.2, 0.35],
        [0.0, 0.4],
        [-0.2, 0.35],
        [-0.2, 0.25],
    ]
    for first, second in zip(safe_waypoints, safe_waypoints[1:]):
        assert _np_segment_distance(OBSTACLE_CENTER, first, second) > FORBIDDEN_RADIUS

    for xy in safe_waypoints:
        _teleport_actor(env, "cubeA", [xy[0], xy[1], 0.3])
        _, _, _, _, info = env.step(action)

    assert not bool(info["obstacle_crossing"].item())
    assert not bool(info["obstacle_collision"].item())
    env.close()


def test_validity_flags_are_sticky_until_reset():
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    base = env.unwrapped

    _teleport_actor(env, "cubeA", [0.048, 0.1, 0.125])
    _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert bool(info["obstacle_collision"].item())

    # Moving the cube back to a safe, harmless place does not clear the flag.
    _teleport_actor(env, "cubeA", [0.0, -0.15, 0.02])
    for _ in range(3):
        _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert bool(info["obstacle_collision"].item())

    _, reset_info = env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    assert not bool(reset_info["obstacle_collision"].item())
    assert not bool(reset_info["obstacle_crossing"].item())
    env.close()


def test_reset_to_env_states_restores_layout_and_clears_flags():
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    base = env.unwrapped
    saved_state = _clone_state(base.get_state_dict())

    _teleport_actor(env, "cubeA", [0.048, 0.1, 0.125])
    _, _, _, _, info = env.step(torch.zeros(base.single_action_space.shape))
    assert bool(info["obstacle_collision"].item())

    _, reset_info = env.reset(
        options={"reset_to_env_states": {"env_states": saved_state}}
    )
    assert not bool(reset_info["obstacle_collision"].item())
    assert not bool(reset_info["obstacle_crossing"].item())
    restored = base.get_state_dict()
    assert torch.allclose(
        restored[OBSTACLE_STATE_KEY], saved_state[OBSTACLE_STATE_KEY], atol=1e-6
    )
    env.close()


def test_reset_to_env_states_does_not_bridge_previous_episode():
    env = make_env()
    env.reset(seed=7, options={"obstacle_center": [0.25, 0.0]})
    base = env.unwrapped

    # Save a valid grasped state near the origin.
    action = _establish_real_grasp(env)
    saved_state = _clone_state(base.get_state_dict())

    # Move the (still sticky) grasped cube far across the forbidden disk.
    _teleport_actor(env, "cubeA", [0.5, 0.0, 0.4])
    _, _, _, _, info = env.step(action)
    assert bool(info["obstacle_crossing"].item())

    # Restoring the start state must rebase the cube path, not bridge the old
    # episode's last position to the restored one.
    env.reset(options={"reset_to_env_states": {"env_states": saved_state}})
    _, _, _, _, info = env.step(action)
    assert not bool(info["obstacle_crossing"].item())
    env.close()


def test_record_episode_hdf5_json_t_plus_one_and_original_info(tmp_path):
    env = RecordEpisode(
        make_env(),
        output_dir=str(tmp_path),
        trajectory_name="traj",
        save_video=False,
        save_on_reset=True,
    )
    env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    num_steps = 5
    base = env.unwrapped
    for _ in range(num_steps):
        env.step(torch.zeros(base.single_action_space.shape))
    # Trigger a flush of the finished trajectory.
    env.reset(seed=8, options={"obstacle_center": [0.0, 0.1]})
    env.close()

    h5_path = tmp_path / "traj.h5"
    json_path = tmp_path / "traj.json"
    assert h5_path.exists() and json_path.exists()

    with h5py.File(h5_path, "r") as h5:
        assert "traj_0" in h5
        traj = h5["traj_0"]
        assert traj["actions"].shape == (num_steps, 8)
        assert traj["env_states"]["obstacle"].shape == (num_steps + 1, 3)
        assert traj["env_states"]["actors"]["cubeA"].shape == (num_steps + 1, 13)
        assert traj["env_states"]["articulations"]["panda"].shape[0] == num_steps + 1
        assert traj["obs"]["extra"]["obstacle"].shape == (num_steps + 1, 3)
        assert traj["success"].shape == (num_steps,)

    with open(json_path) as f:
        meta = json.load(f)
    assert meta["env_info"]["env_id"] == ENV_ID
    episode = meta["episodes"][0]
    assert episode["elapsed_steps"] == num_steps
    assert "success" in episode
    assert episode["reset_kwargs"]["options"]["obstacle_center"] == [0.0, 0.1]
    env.close()


def test_recorder_preserves_component_validity_and_config(tmp_path):
    env = ObstacleRecordEpisode(
        make_env(),
        output_dir=str(tmp_path),
        trajectory_name="traj",
        save_video=False,
        save_on_reset=True,
    )
    base = env.unwrapped

    # Episode 0: a real obstacle collision, which is a failed example.
    env.reset(seed=7, options={"obstacle_center": [0.0, 0.1]})
    _teleport_actor(env, "cubeA", [0.048, 0.1, 0.125])
    for _ in range(3):
        env.step(torch.zeros(base.single_action_space.shape))

    # Episode 1: an uninterrupted stack success.
    env.reset(seed=8, options={"obstacle_center": [0.25, 0.25]})
    state = base.get_state_dict()
    state["actors"]["cubeB"][0, :3] = torch.tensor(
        [-0.1, -0.1, 0.02], dtype=state["actors"]["cubeB"].dtype
    )
    state["actors"]["cubeB"][0, 7:13] = 0
    state["actors"]["cubeA"][0, :3] = torch.tensor(
        [-0.1, -0.1, 0.06], dtype=state["actors"]["cubeA"].dtype
    )
    state["actors"]["cubeA"][0, 7:13] = 0
    base.set_state_dict(state)
    for _ in range(8):
        env.step(torch.zeros(base.single_action_space.shape))

    env.reset(seed=9, options={"obstacle_center": [0.25, 0.25]})
    env.close()

    meta = json.loads((tmp_path / "traj.json").read_text())
    config = meta["env_info"]["obstacle_config"]
    assert config["radius"] == pytest.approx(OBSTACLE_RADIUS)
    assert config["height"] == pytest.approx(0.25)
    assert config["center_z"] == pytest.approx(0.125)
    assert config["safety_margin"] == pytest.approx(SAFETY_MARGIN)
    assert config["units"] == "meters"
    assert config["state_field_order"] == ["center_x", "center_y", "radius"]
    assert config["geometry_restore"]["radius"] == pytest.approx(OBSTACLE_RADIUS)

    episodes = meta["episodes"]
    assert episodes[0]["stack_success"] is False
    assert episodes[0]["obstacle_collision"] is True
    assert episodes[0]["obstacle_crossing"] is False
    assert episodes[0]["success"] is False

    assert episodes[1]["stack_success"] is True
    assert episodes[1]["obstacle_collision"] is False
    assert episodes[1]["obstacle_crossing"] is False
    assert episodes[1]["success"] is True


def test_episode_obstacle_config_matches_hdf5_and_reset_kwargs(tmp_path):
    """Per-episode config reports the actual (not construction-time) obstacle.

    Multiple episodes with different radius and centre must each record their
    own centre/world pose, and the recorded JSON must agree with both the HDF5
    ``env_states.obstacle`` rows and the reset kwargs that produced the episode.
    """
    env = ObstacleRecordEpisode(
        make_env(),
        output_dir=str(tmp_path),
        trajectory_name="varied",
        save_video=False,
        save_on_reset=True,
    )
    base = env.unwrapped
    episodes = [([0.0, 0.25], 0.03), ([0.05, 0.20], 0.04), ([-0.05, 0.30], 0.02)]
    for seed, (center, radius) in enumerate(episodes):
        env.reset(
            seed=seed,
            options={
                "obstacle_center": center,
                "obstacle_radius": radius,
                "reconfigure": True,
            },
        )
        for _ in range(2):
            env.step(torch.zeros(base.single_action_space.shape))
    env.close()

    meta = json.loads((tmp_path / "varied.json").read_text())
    assert len(meta["episodes"]) == len(episodes)
    with h5py.File(tmp_path / "varied.h5", "r") as f:
        for index, (center, radius) in enumerate(episodes):
            episode = meta["episodes"][index]
            config = episode["obstacle_config"]
            assert config["center"] == pytest.approx(center)
            assert config["radius"] == pytest.approx(radius)
            assert config["world_pose"] == pytest.approx(
                [center[0], center[1], 0.125, *euler2quat(0.0, np.pi / 2, 0.0)]
            )
            assert config["geometry_restore"]["initial_pose"] == pytest.approx(
                [center[0], center[1], 0.125]
            )
            assert config["geometry_restore"]["initial_rotation"] == pytest.approx(
                list(euler2quat(0.0, np.pi / 2, 0.0))
            )
            options = episode["reset_kwargs"]["options"]
            assert options["obstacle_center"] == pytest.approx(center)
            assert options["obstacle_radius"] == pytest.approx(radius)

            recorded = f[f"traj_{index}"]["env_states"]["obstacle"]
            assert recorded.shape[1] == 3
            assert np.allclose(recorded[:, :2], center, atol=1e-5)
            assert np.allclose(recorded[:, 2], radius, atol=1e-5)
