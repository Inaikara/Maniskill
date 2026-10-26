import json

import h5py
import numpy as np
import pytest

from examples.baselines.constrain.placesphere_matched_diagnosis import (
    ANGULAR_THRESHOLD_RAD_S,
    BASE_SHA,
    CONSTRAIN_SHA,
    FAILURE_INDICES,
    SAFE_CONTROL_INDICES,
    classify_motion,
    classify_queue_causality,
    _load_expert_release_windows,
)


def test_diagnostic_is_pinned_to_the_current_matched_emas_and_frozen_cases():
    assert BASE_SHA == "4484b76f1767a4723c7a60f17c1cdd2b25afc6876bc0f732f66a0997919c6942"
    assert CONSTRAIN_SHA == "4c8f18c79c00188da785740bae5750c5989b8516a2f741712f3d5c833d915b88"
    assert FAILURE_INDICES == (12, 13, 21, 22, 25, 28)
    assert SAFE_CONTROL_INDICES == (0, 2, 4, 6)


def test_native_motion_classification_keeps_linear_and_angular_units_separate():
    assert ANGULAR_THRESHOLD_RAD_S == 0.5
    assert classify_motion(0.011, 0.6) == {
        "linear_over_threshold": True,
        "angular_over_threshold": True,
        "category": "linear_and_angular",
    }
    assert classify_motion(0.0, 0.51)["category"] == "angular"
    assert classify_motion(0.01, 0.5)["category"] == "within_native_thresholds"


def test_queue_causality_requires_transition_evidence_and_reports_actual_queue_index():
    rows = [
        {"obstacle_contact": False, "failure_transition": False, "queue_index": 6},
        {"obstacle_contact": True, "failure_transition": True, "queue_index": 3},
        {"obstacle_contact": False, "failure_transition": True, "queue_index": 0},
    ]
    assert classify_queue_causality(rows) == {
        "transition_count": 2,
        "queued_transition_count": 1,
        "queue_indices": [3, 0],
        "staleness_isolated": False,
    }


def test_queue_index_out_of_range_is_not_silently_accepted():
    with pytest.raises(ValueError, match="queue index"):
        classify_queue_causality([{"obstacle_contact": True, "queue_index": 8}])


def test_expert_velocity_is_aligned_to_post_action_t_plus_one_state(tmp_path):
    h5_path, manifest_path = tmp_path / "trajectory.h5", tmp_path / "trajectory.json"
    manifest_path.write_text(json.dumps({"episodes": [{"episode_id": 0, "kind": "obstacle"}]}))
    metadata = {"expert_record": {}, "step_phase": ["release", "settle"],
                "sphere_linear_speed_m_s": [0.4, 0.1],
                "sphere_angular_speed_rad_s": [0.3, 0.2]}
    states = np.zeros((3, 13), dtype="float32")
    states[1, 7:10] = [0.4, 0, 0]
    states[1, 10:13] = [0.3, 0, 0]
    states[2, 7:10] = [0, 0.1, 0]
    states[2, 10:13] = [0, 0.2, 0]
    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("traj_0")
        group.attrs["metadata_json"] = json.dumps(metadata)
        group["actions"] = np.zeros((2, 7), dtype="float32")
        group["raw_obs"] = np.zeros((3, 42), dtype="float32")
        group.create_group("env_states").create_group("actors")["sphere"] = states

    report = _load_expert_release_windows(h5_path, manifest_path)
    rows = report["release_settle_native_trace"]
    assert [row["phase"] for row in rows] == ["release", "settle"]
    assert [row["linear_speed_m_s"] for row in rows] == pytest.approx([0.4, 0.1])
    assert [row["angular_speed_rad_s"] for row in rows] == pytest.approx([0.3, 0.2])
