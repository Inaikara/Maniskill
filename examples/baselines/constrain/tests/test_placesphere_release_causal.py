import pytest

from examples.baselines.constrain.placesphere_release_causal import release_events


def _row(step, *, grip=0.0, grasped=True, released=False, z=0.2, queue=0):
    return {
        "step": step, "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, grip],
        "queue_index": queue, "released_this_step": released,
        "grasped": grasped, "sphere_xyz": [0.0, 0.0, z],
        "sphere_linear_speed_m_s": 0.2, "sphere_angular_speed_rad_s": 0.7,
        "sphere_linear_velocity_m_s": [0.0, 0.0, 0.0],
        "sphere_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "tcp_xyz": [0.1, 0.2, 0.3], "control_timestep_s": 0.02,
        "obstacle_contact": False, "obstacle_crossing": False,
    }


def test_release_measurement_keeps_open_and_physical_transition_indices_distinct():
    event = release_events([
        _row(3), _row(4, grip=0.2, queue=1),
        _row(5, grasped=False, released=True, queue=2), _row(6, grasped=False, z=0.196),
    ])[0]
    assert (event["open_command_step"], event["ungrasped_transition_step"]) == (4, 5)
    assert event["open_queue_index"] == 1
    assert event["transition_queue_index"] == 2
    assert event["next_interval_dz_m"] == pytest.approx(-0.004)
    assert event["gravity_expected_dz_m"] == pytest.approx(-0.001962)
    assert event["impulse_measured"] is False
    assert event["staleness_causal"] is False


def test_release_event_rejects_wrong_action_width():
    row = _row(0, grip=0.2)
    row["action"] = [1.0]
    with pytest.raises(ValueError, match="seven-dimensional"):
        release_events([row])
