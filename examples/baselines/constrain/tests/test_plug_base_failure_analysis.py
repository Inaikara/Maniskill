import json

import numpy as np
import pytest

from examples.baselines.constrain import plug_base_failure_analysis as analysis


def _event(control, *, charger_z=0.0, position=0.1, angle=0.5, success=False):
    obs = np.zeros(46, np.float32)
    obs[18:21] = [0.0, 0.0, 0.2]
    obs[21] = 1.0
    obs[25:28] = [position, 0, charger_z]
    obs[28] = np.cos(angle / 2)
    obs[31] = np.sin(angle / 2)
    charger = [position, 0.0, charger_z, np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)] + [0.0] * 6
    receptacle = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 6
    return {"control": control, "policy_input_46d": obs.tolist(), "action_7d": [0] * 6 + [-1],
            "observation_46d": obs.tolist(),
            "native_state": {"actors": {"charger": [charger], "receptacle": [receptacle]},
                             "plug_orientation_target": [[0.0, 0.0, 0.0, 0.0]]},
            "native_info": {"obj_to_goal_dist": [np.hypot(position, charger_z)], "obj_to_goal_angle": [angle], "success": [success]}}


def test_trace_labels_failed_alignment_and_keeps_same_control_pose_errors(tmp_path):
    path = tmp_path / "episode.events.jsonl"
    path.write_text("\n".join(json.dumps(_event(i, charger_z=(0 if i == 1 else .03), position=.004, angle=.3)) for i in range(1, 7)))
    report = analysis.analyze_episode(path, claimed_success=False)
    assert report["failure_phase"] == "goal_near_rotation_over_0.2rad"
    assert report["closest_position_rotation_error_rad_same_control"] == pytest.approx(.3)
    assert report["closest_position_error_m"] == pytest.approx(.004)
    assert report["closest_position_residual_world_xyz_m"] == pytest.approx([.004, 0, 0])
    assert report["sustained_20mm_lift_evidence"] is True
    assert report["native_grasp_status"] is None


def test_trace_rejects_event_count_alignment_and_success_mismatch(tmp_path):
    path = tmp_path / "episode.events.jsonl"
    path.write_text(json.dumps(_event(2)))
    with pytest.raises(ValueError, match="non-contiguous"):
        analysis.analyze_episode(path, claimed_success=False)
    path.write_text(json.dumps(_event(1, success=True)))
    with pytest.raises(ValueError, match="outcome mismatch"):
        analysis.analyze_episode(path, claimed_success=False)
