import json

import numpy as np
import pytest

from examples.baselines.constrain.retain_orientation_results import (
    indices_with_outcome, recorded_actions, representative_index,
)


def test_representative_index_uses_lowest_success_without_visual_choice():
    rows = [{"index": 2, "success": False}, {"index": 0, "success": False}, {"index": 4, "success": True}]
    assert representative_index(rows) == 4
    assert representative_index([{"index": 3, "success": False}, {"index": 1, "success": False}]) == 1


def test_indices_with_outcome_keeps_lowest_matching_scenes():
    rows = [{"index": 4, "success": False}, {"index": 1, "success": True}, {"index": 0, "success": False}]
    assert indices_with_outcome(rows, False, 2) == [0, 4]
    assert indices_with_outcome([{"scene_index": 2, "success": True}], True, 1) == [2]
    assert representative_index(rows) == 1


def test_recorded_actions_reads_native_controls_only(tmp_path):
    path = tmp_path / "episode.jsonl"
    path.write_text("\n".join([
        json.dumps({"event": "episode_start"}),
        json.dumps({"event": "control", "action": [0, 0, 0, 0, 0, 0, 1]}),
    ]) + "\n")
    actions = recorded_actions(path)
    assert len(actions) == 1 and actions[0].shape == (7,) and actions[0].dtype == np.float32


def test_recorded_actions_rejects_malformed_action(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"event": "control", "action": [1, 2]}) + "\n")
    with pytest.raises(ValueError):
        recorded_actions(path)
