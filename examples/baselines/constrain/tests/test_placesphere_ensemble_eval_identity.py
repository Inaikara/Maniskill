import json

import pytest

from examples.baselines.constrain.evaluate_placesphere_matched import _validate_primary_baseline


def test_primary_baseline_requires_same_checkpoint_pair_and_initial_observation(tmp_path):
    observation = [0.0] * 42
    for policy, checkpoint in (("base_dp", "base"), ("constrain_dp", "control")):
        folder = tmp_path / policy
        folder.mkdir()
        for index in range(30):
            row = {
                "scene_index": index,
                "scene_id": f"scene-{index}",
                "seed": 2026109004 + index,
                "restored_bit_exact": True,
                "checkpoint_sha256": checkpoint,
                "closed_loop_trace": [{"policy_observation": observation}],
            }
            (folder / f"episode_{index:03d}.json").write_text(json.dumps(row))

    audit = _validate_primary_baseline(tmp_path, "base", "control")
    assert audit["matching_scene_ids_seeds_and_initial_observations"] is True
    assert audit["all_restored_bit_exact"] is True

    path = tmp_path / "constrain_dp" / "episode_003.json"
    row = json.loads(path.read_text())
    row["closed_loop_trace"][0]["policy_observation"][0] = 1.0
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="initial observation mismatch at scene 3"):
        _validate_primary_baseline(tmp_path, "base", "control")
