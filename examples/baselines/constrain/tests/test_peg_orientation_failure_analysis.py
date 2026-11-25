from examples.baselines.constrain.peg_orientation_failure_analysis import summarize_episode


def test_summary_uses_all_pre_post_and_terminal_samples_once():
    episode = {
        "success": False,
        "phase": {"ever_grasped": True, "contact_losses": 1, "first_grasp_control": 2},
        "phase_trace": [{
            "is_grasping": True,
            "contact_lost": True,
            "stage_geometry_preaction": {
                "official_peg_head_in_hole_m": [-0.04, 0.03, 0.0],
                "official_insertion": False,
                "wrapped_peg_plus_x_minus_box_plus_x_yaw_rad": 0.2,
            },
            "stage_geometry_postaction": {
                "official_peg_head_in_hole_m": [-0.02, 0.01, 0.0],
                "official_insertion": False,
                "wrapped_peg_plus_x_minus_box_plus_x_yaw_rad": -0.1,
            },
        }],
        "stage_geometry_terminal": {
            "official_peg_head_in_hole_m": [-0.01, 0.0, 0.0],
            "official_insertion": True,
            "wrapped_peg_plus_x_minus_box_plus_x_yaw_rad": -0.05,
        },
    }
    result = summarize_episode(episode)
    assert result["snapshot_count"] == 3
    assert result["minimum_head_distance_m"] == 0.01
    assert result["minimum_head_xyz_m"] == [-0.01, 0.0, 0.0]
    assert result["terminal_yaw_rad"] == -0.05
    assert result["trace_native_insertion_seen"] is True
    assert result["contact_loss_transitions"] == 1
