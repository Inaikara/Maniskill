"""Indexed kinematic release measurements; these are not force/impulse estimates."""

import numpy as np


def release_events(trace, *, gravity_m_s2=9.81):
    """Join the open command, ungrasped transition and following gravity interval."""
    rows = sorted(trace, key=lambda row: int(row["step"]))
    events = []
    for index, row in enumerate(rows):
        action = np.asarray(row["action"], dtype=np.float64)
        if action.shape != (7,):
            raise ValueError("release trace actions must be seven-dimensional")
        if action[6] <= 0:
            continue
        if index == 0:
            pre = None
        else:
            pre = rows[index - 1]
        transition = next((item for item in rows[index:] if item.get("released_this_step")), None)
        if transition is None:
            continue
        after = next((item for item in rows if int(item["step"]) == int(transition["step"]) + 1), None)
        dt = float(transition.get("control_timestep_s", 0.0))
        dz = None if after is None else float(after["sphere_xyz"][2] - transition["sphere_xyz"][2])
        events.append({
            "open_command_step": int(row["step"]),
            "open_queue_index": int(row["queue_index"]),
            "open_action_7d": action.tolist(),
            "pre_open_sphere_linear_speed_m_s": None if pre is None else pre["sphere_linear_speed_m_s"],
            "pre_open_sphere_angular_speed_rad_s": None if pre is None else pre["sphere_angular_speed_rad_s"],
            "post_open_sphere_linear_speed_m_s": row["sphere_linear_speed_m_s"],
            "post_open_sphere_angular_speed_rad_s": row["sphere_angular_speed_rad_s"],
            "open_tcp_xyz": row["tcp_xyz"],
            "open_tcp_quaternion_wxyz": row.get("tcp_quaternion_wxyz"),
            "open_sphere_quaternion_wxyz": row.get("sphere_quaternion_wxyz"),
            "ungrasped_transition_step": int(transition["step"]),
            "transition_queue_index": int(transition["queue_index"]),
            "transition_sphere_linear_velocity_m_s": transition["sphere_linear_velocity_m_s"],
            "transition_sphere_angular_velocity_rad_s": transition["sphere_angular_velocity_rad_s"],
            "transition_obstacle_contact": bool(transition.get("obstacle_contact", False)),
            "transition_obstacle_crossing": bool(transition.get("obstacle_crossing", False)),
            "next_interval_dz_m": dz,
            "next_interval_control_timestep_s": dt or None,
            "gravity_expected_dz_m": None if not dt else -0.5 * gravity_m_s2 * dt * dt,
            "gravity_model_is_comparison_only": True,
            "impulse_measured": False,
            "staleness_causal": False,
        })
    return events
