import numpy as np
import pytest
import gymnasium as gym
import sapien

from examples.baselines.constrain.inference_xy_constraint_r45 import (
    XYConstraint,
    validate_tcp_trace,
)


@pytest.mark.parametrize("env_id,module", [
    ("PickCubeObstacle-v1", "pickcube_obstacle"),
    ("PlaceSphereObstacle-v1", "placesphere_obstacle"),
])
def test_render_only_mode_keeps_visual_actor_and_restorable_forbidden_geometry(env_id, module):
    __import__(f"examples.baselines.constrain.{module}")
    env = gym.make(env_id, num_envs=1, obs_mode="state_dict", sim_backend="physx_cpu",
                   control_mode="pd_ee_delta_pose", obstacle_mode="render_only")
    try:
        env.reset(seed=17, options={"obstacle_enabled": True, "obstacle_center": [0.1, 0.2]})
        task = env.unwrapped
        body = task.obstacle._bodies[0]
        assert len(body.collision_shapes) == 0
        visual = task.obstacle._objs[0].find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        assert visual is not None and visual.visibility > 0
        assert task.obstacle_collision_radius == pytest.approx(0.03)
        saved = task.get_state_dict()
        task.set_state_dict(saved)
        assert task.get_state_dict()["pickcube_obstacle" if "PickCube" in env_id
                                     else "placesphere_obstacle"][0, 2] == pytest.approx(0.03)
    finally:
        env.close()


def test_filter_preserves_nonposition_channels_and_avoids_a_grazing_chord():
    constraint = XYConstraint(translation_scale_m=0.1, action_limit=1.0)
    action = np.array([0.8, 0.0, 0.25, 0.1, -0.2, 0.3, 0.75])
    result = constraint.filter_action(action, [0.06, 0.0], [0.0, 0.0], 0.05)
    assert np.array_equal(result[2:], action[2:])
    assert not constraint.segment_violates([0.06, 0.0], [0.06, 0.0] + result[:2] * 0.1,
                                           [0.0, 0.0], 0.05)


def test_lag_buffer_changes_safe_boundary_target_before_actual_crossing():
    constraint = XYConstraint(translation_scale_m=.1, action_limit=1,
                               xy_action_bounds=(-1, 1), working_clearance_m=.003)
    # The old 1 um target was nominally safe, but a 2.614 mm plant error crossed.
    result = constraint.filter_action([-.20, 0, .3, .1, -.2, .3, -1],
                                      [.055001, 0], [0, 0], .05)
    endpoint = np.array([.055001, 0]) + result[:2] * .1
    assert np.linalg.norm(endpoint) >= .053 - 1e-9
    direction = endpoint - [.055001, 0]
    fraction = np.clip(np.dot(-np.array([.055001, 0]), direction) / np.dot(direction, direction), 0, 1)
    chord_gap = np.linalg.norm(np.array([.055001, 0]) + fraction * direction) - .05
    assert chord_gap >= .003 - 1e-8
    # Under the measured worst-case 2.614 mm response error, actual clearance
    # remains positive; the old radius+1 um target would cross.
    assert np.linalg.norm(endpoint) - .002614 > .05
    assert not constraint.segment_violates([.055001, 0], endpoint, [0, 0], .05)
    assert np.array_equal(result[2:], [.3, .1, -.2, .3, -1])


def test_initial_point_inside_working_buffer_is_directed_outward_and_task_bounds_apply():
    constraint = XYConstraint(translation_scale_m=.1, action_limit=1,
                               xy_action_bounds=(-1, 1), working_clearance_m=.003)
    result = constraint.filter_action([-.8, 0, 0, 0, 0, 0, 1], [.0501, 0], [0, 0], .05)
    assert result[0] > 0
    assert abs(result[0]) <= 1
    with pytest.raises(ValueError, match="starts inside"):
        constraint.filter_action([0, 0, 0, 0, 0, 0, 1], [.049, 0], [0, 0], .05)
    place = XYConstraint(xy_action_bounds=(-.35, .35))
    far = place.filter_action([.8, -.8, .2, 0, 0, 0, -1], [.2, .2], [0, 0], .05)
    assert far[:2] == pytest.approx([.35, -.35])
    pick = XYConstraint(xy_action_bounds=(-1, 1))
    far_pick = pick.filter_action([.8, -.8, .2, 0, 0, 0, -1], [.2, .2], [0, 0], .05)
    assert far_pick[:2] == pytest.approx([.8, -.8])


def test_trace_validator_checks_every_actual_sample_and_segment_including_success():
    report = validate_tcp_trace(
        [[-0.2, 0.06], [-0.02, 0.0], [0.2, 0.06]], [0.0, 0.0], 0.05,
        first_success_step=2,
    )
    assert report["violating_samples"] == [1]
    assert report["violating_segments"] == [0, 1]
    assert report["first_success_step_valid"] is False


@pytest.mark.parametrize("action,position", [
    ([np.nan, 0, 0, 0, 0, 0, 0], [0.2, 0]),
    ([0, 0, 0, 0, 0, 0, 0], [np.inf, 0]),
])
def test_filter_rejects_nonfinite_controller_input(action, position):
    with pytest.raises(ValueError, match="finite"):
        XYConstraint().filter_action(action, position, [0, 0], 0.05)
