import numpy as np

import json

import pytest

from examples.baselines.constrain.peg_orientation_near_hole import (
    _jsonable, add_no_candidate_rows, finalize_saved_bank, near_hole_geometry_eligible,
)


def test_near_hole_requires_grasp_and_uninserted_head_in_official_hole_frame_window():
    assert near_hole_geometry_eligible(np.array([-0.03, 0.01, 0.0]), True, False)
    assert not near_hole_geometry_eligible(np.array([-0.01, 0.0, 0.0]), True, False)
    assert not near_hole_geometry_eligible(np.array([-0.03, 0.051, 0.0]), True, False)
    assert not near_hole_geometry_eligible(np.array([-0.03, 0.0, 0.0]), False, False)
    assert not near_hole_geometry_eligible(np.array([-0.03, 0.0, 0.0]), True, True)


def test_json_normalization_preserves_numpy_boolean_and_array_values():
    normalized = _jsonable({"success": np.bool_(True), "pose": np.array([1.0, 2.0])})
    assert normalized == {"success": True, "pose": [1.0, 2.0]}
    assert type(normalized["success"]) is bool


def test_bank_finalization_persists_numpy_values_from_complete_journal(tmp_path):
    journal = tmp_path / "branches.jsonl"
    journal.write_text("".join(json.dumps({"catalog_index": i, "eligible": i == 0,
                                             "prefix_controls": 2,
                                             "physical_state": {"qpos": [float(i)]}}) + "\n"
                             for i in range(20)))
    result = finalize_saved_bank(journal, tmp_path / "bank.json", split="validation_random",
                                 catalog_manifest_sha256="catalog", source_h5_sha256="source",
                                 elapsed_seconds=1.0)
    saved = json.loads((tmp_path / "bank.json").read_text())
    assert result["eligible_count"] == saved["eligible_count"] == 1
    assert saved["controls"] == 40
    assert saved["branches"][0]["physical_state"]["qpos"] == [0.0]


def test_bank_finalization_rejects_incomplete_journal(tmp_path):
    journal = tmp_path / "branches.jsonl"
    journal.write_text(json.dumps({"catalog_index": 0, "eligible": False, "prefix_controls": 0}) + "\n")
    with pytest.raises(ValueError, match="20 frozen catalog indices"):
        finalize_saved_bank(journal, tmp_path / "bank.json", split="validation_random",
                             catalog_manifest_sha256="catalog", source_h5_sha256="source",
                             elapsed_seconds=1.0)


def test_no_candidate_scenes_remain_as_explicit_ineligible_denominator_rows():
    metadata = [{"split": "validation_random", "scene_id": f"scene-{i}", "seed": 100 + i}
                for i in range(20)]
    rows = [{"catalog_index": i, "eligible": True, "prefix_controls": 3} for i in range(2, 20)]
    complete = add_no_candidate_rows(rows, metadata)
    assert [row["catalog_index"] for row in complete] == list(range(20))
    assert complete[0]["reason"] == complete[1]["reason"] == "no_expert_replay_candidate"
    assert complete[0]["seed"] == 100
