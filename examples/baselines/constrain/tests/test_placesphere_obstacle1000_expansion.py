import json

import h5py
import numpy as np
import pytest

from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_HOLD8_OBSTACLE1000_VERSION,
    MATCHED_HOLD8_OBSTACLE500_VERSION,
    _counts_for_version,
)
from examples.baselines.constrain.placesphere_obstacle1000_expansion import (
    next_unseen_seed,
    preview_candidate_layout,
    frozen_pool_selection,
    task_layout_conflict,
    unique_layout_audit,
    verify_obstacle_candidate,
)


def test_new_version_is_explicit_and_does_not_reinterpret_historical_profiles():
    assert _counts_for_version(MATCHED_HOLD8_OBSTACLE1000_VERSION) == (
        {"direct": 1000, "obstacle": 1000}, 2000)
    assert _counts_for_version("placesphere_matched_terminal_r23_hold8_v1") == (
        {"direct": 1000, "obstacle": 300}, 1300)
    with pytest.raises(ValueError, match="unsupported"):
        _counts_for_version("unregistered")


def test_frozen_candidate_selection_is_seed_ordered_and_never_refills():
    rows = [{"seed": 1000 - i, "original_id": i, "start_xy": [i, 0],
             "container_xy": [i, 1]} for i in range(239)]
    selection = frozen_pool_selection(rows)
    assert len(selection["ordered_candidates"]) == 239
    assert selection["ordered_candidates"][0]["seed"] == min(row["seed"] for row in rows)
    with pytest.raises(ValueError, match="exactly 239"):
        frozen_pool_selection(rows[:-1])


def test_500_profile_keeps_the_existing_1300_episode_profile_unchanged():
    assert _counts_for_version(MATCHED_HOLD8_OBSTACLE500_VERSION) == (
        {"direct": 1000, "obstacle": 500}, 1500)
    assert _counts_for_version(MATCHED_HOLD8_OBSTACLE1000_VERSION) == (
        {"direct": 1000, "obstacle": 1000}, 2000)
    assert _counts_for_version("placesphere_matched_terminal_r23_hold8_v1") == (
        {"direct": 1000, "obstacle": 300}, 1300)


def test_superseded_candidate_sampling_paths_fail_closed():
    from examples.baselines.constrain.placesphere_obstacle1000_expansion import (
        continue_candidate_catalog,
        generate_candidate_catalog,
        preregister,
        preregister_candidate_resume,
        preregister_full_candidate_resume,
        run_candidate_canary,
        run_registered_candidate_batch,
    )

    with pytest.raises(RuntimeError, match="prohibited"):
        generate_candidate_catalog("unused", "unused")
    with pytest.raises(RuntimeError, match="superseded"):
        run_registered_candidate_batch("unused", "unused")
    with pytest.raises(RuntimeError, match="superseded"):
        preregister("unused")
    with pytest.raises(RuntimeError, match="superseded"):
        preregister_candidate_resume("unused", first_catalog="unused", interrupted_catalog="unused",
                                     recovered_catalog="unused", first_journal="unused",
                                     interrupted_journal="unused", timestamp="unused")
    with pytest.raises(RuntimeError, match="prohibited"):
        run_candidate_canary("unused", "unused", "unused")
    with pytest.raises(RuntimeError, match="superseded"):
        preregister_full_candidate_resume("unused", recovered_catalog="unused",
                                          first_journal="unused", interrupted_journal="unused",
                                          canary_journal="unused")
    with pytest.raises(RuntimeError, match="prohibited"):
        continue_candidate_catalog("unused", "unused", "unused")


def test_task_identity_uses_seed_or_both_task_endpoints_not_obstacle_pose():
    left = {"episode_id": 0, "seed": 1, "start_xy": [0.0, 0.0],
            "container_xy": [0.2, 0.1]}
    same_seed_different_pose = {**left, "seed": 2, "start_xy": [0.1, 0.1]}
    near_both_endpoints = {**left, "seed": 3, "start_xy": [0.01, 0.0],
                           "container_xy": [0.2, 0.11]}
    assert not task_layout_conflict(left, same_seed_different_pose)
    assert task_layout_conflict(left, near_both_endpoints)
    assert unique_layout_audit([left, same_seed_different_pose, near_both_endpoints])["greedy_unique"] == 2
    nested = [{"episode_id": index, "seed": row["seed"],
               "layout": {k: row[k] for k in ("start_xy", "container_xy")}}
              for index, row in enumerate((left, same_seed_different_pose, near_both_endpoints))]
    assert unique_layout_audit(nested)["greedy_unique"] == 2


def test_resume_cursor_uses_all_attempted_seeds_and_reports_overlap(tmp_path):
    first, interrupted, canary = (tmp_path / "first.jsonl", tmp_path / "interrupted.jsonl",
                                  tmp_path / "canary.jsonl")
    first.write_text('\n'.join(json.dumps({"seed": seed}) for seed in (100, 101, 102)) + '\n')
    interrupted.write_text('\n'.join(json.dumps({"seed": seed}) for seed in (101, 102, 103, 104)) + '\n')
    canary.write_text('\n'.join(json.dumps({"seed": seed}) for seed in (104, 105)) + '\n')
    cursor = next_unseen_seed(first, interrupted, canary)
    assert cursor["next_unseen_seed"] == 106
    assert cursor["cross_journal_seed_overlap"] == 3
    assert cursor["overlapping_seeds"] == [101, 102, 104]


def test_preview_reproduces_recorded_seed_start_and_goal_without_physics():
    from pathlib import Path

    source = Path("outputs/exp/20261027_100700_placesphere_obstacle1000_r25/recovered_candidates_239/manifest.json")
    scenes = json.loads(source.read_text())["scenes"]
    assert len(scenes) == 239
    for row in scenes:
        proposal = preview_candidate_layout(row["seed"], row["base_xy"])
        assert proposal is not None
        assert np.allclose(proposal["start_xy"], row["start_xy"], atol=1e-12, rtol=0)
        assert np.allclose(proposal["container_xy"], row["container_xy"], atol=1e-12, rtol=0)


def test_candidate_audit_requires_native_positive_first_success_and_restore(tmp_path):
    h5_path, manifest_path = tmp_path / "candidate.h5", tmp_path / "candidate.json"
    row = {"episode_id": 0, "kind": "obstacle", "seed": 17,
           "layout": {"start_xy": [0.0, 0.0], "container_xy": [0.2, 0.1]}}
    expert = {"native_success": True, "terminated": True, "truncated": False,
              "from_execution_start_safe": True, "obstacle_contact": False,
              "obstacle_crossing": False, "on_bin": True, "static": True,
              "ungrasped": True}
    metadata = {"expert_record": expert,
                "t_plus_one_restore_audit": {"passed": True}, "layout": row["layout"]}
    with h5py.File(h5_path, "w") as h5:
        group = h5.create_group("traj_0")
        group.attrs["kind"] = "obstacle"
        group.attrs["metadata_json"] = json.dumps(metadata)
        group.create_dataset("actions", data=np.zeros((2, 7), np.float32))
        group.create_dataset("obs", data=np.zeros((3, 39), np.float32))
        group.create_dataset("raw_obs", data=np.zeros((3, 42), np.float32))
    manifest_path.write_text(json.dumps({"episodes": [row]}))
    audit = verify_obstacle_candidate(h5_path, manifest_path, expected_count=1)
    assert audit["task_identity"]["greedy_unique"] == 1
    assert audit["lengths"]["max"] == 2

    metadata["expert_record"]["obstacle_contact"] = True
    with h5py.File(h5_path, "a") as h5:
        h5["traj_0"].attrs["metadata_json"] = json.dumps(metadata)
    with pytest.raises(ValueError, match="full-prefix safety"):
        verify_obstacle_candidate(h5_path, manifest_path, expected_count=1)
