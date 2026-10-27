import json

import pytest

from examples.baselines.constrain.placesphere_r27_primary_analysis import (
    ROOT,
    verify_registered_inputs,
    validate_primary_configs,
)


def frozen(path, base_sha="base", constrain_sha="constrain"):
    return {
        "scene_dir": "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes",
        "scene_manifest_sha256": "scene-manifest",
        "scene_count": 30,
        "checkpoints": {
            "base_dp": {"path": "base.pt", "sha256": base_sha},
            "constrain_dp": {"path": "constrain.pt", "sha256": constrain_sha},
        },
        "max_steps": 200,
        "episode_wall_seconds": 300,
        "execution_stride": 2,
        "prediction_horizon": 16,
        "executed_action_slice": [1, 9],
        "runtime_errors_remain_in_denominator": True,
        "primary_baseline": path,
    }


def test_primary_comparison_requires_r26_same_checkpoints_and_rollout_contract():
    r26 = frozen("outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval")
    ddim = frozen("outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2")
    assert validate_primary_configs(ddim, r26) is True

    wrong_checkpoint = frozen(
        "outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2",
        constrain_sha="another-model",
    )
    with pytest.raises(ValueError, match="checkpoint"):
        validate_primary_configs(ddim, wrong_checkpoint)


def test_r24_cannot_be_used_as_the_same_checkpoint_primary():
    r24 = frozen("outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval")
    ddim = frozen("outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2")
    with pytest.raises(ValueError, match="R26 primary"):
        validate_primary_configs(
            ddim, r24,
            primary_run_path="outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval",
        )


def test_registered_input_gate_rejects_append_after_pin_and_accepts_frozen_copy(tmp_path):
    mutable_record = tmp_path / "round27_execution_record.json"
    mutable_record.write_bytes(b'{"status":"complete"}\n')
    pinned_sha = "bc3483a62d0a1496b93d003cd73bb2dcdce1081c08c2d608b6aa90ff3187fd9d"
    registration = {
        "inputs": {
            "r27_original_execution_record": {
                "path": str(mutable_record), "sha256": pinned_sha,
            }
        }
    }

    with pytest.raises(ValueError, match="r27_original_execution_record"):
        verify_registered_inputs(registration, root=tmp_path)

    frozen = tmp_path / "snapshot" / "round27_execution_record.json"
    frozen.parent.mkdir()
    frozen.write_bytes(mutable_record.read_bytes())
    registration["inputs"]["r27_original_execution_record"] = {
        "path": str(frozen), "sha256": "",
    }
    import hashlib

    registration["inputs"]["r27_original_execution_record"]["sha256"] = hashlib.sha256(
        frozen.read_bytes()
    ).hexdigest()
    assert verify_registered_inputs(registration, root=tmp_path) is True

    frozen.write_bytes(b'{"status":"changed"}\n')
    with pytest.raises(ValueError, match="r27_original_execution_record"):
        verify_registered_inputs(registration, root=tmp_path)


def test_retained_r27_registration_fails_but_superseding_snapshot_is_addressable():
    old_registration = json.loads((
        ROOT / "outputs/exp/20261027_213400_placesphere_r27_primary_correction/analysis_registration.json"
    ).read_text())
    new_registration = json.loads((
        ROOT / "outputs/exp/20261027_220000_placesphere_r27_primary_correction_rerun/analysis_registration.json"
    ).read_text())

    with pytest.raises(ValueError, match="r27_original_execution_record"):
        verify_registered_inputs(old_registration)
    assert verify_registered_inputs(new_registration) is True
