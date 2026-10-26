import hashlib
import json

import pytest

from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_HOLD8_VERSION,
    load_matched_training_identity,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows():
    return ([{"episode_id": i, "kind": "direct"} for i in range(1000)]
            + [{"episode_id": i, "kind": "obstacle"} for i in range(1000, 1300)])


def _episode_ids():
    return {"direct": list(range(1000)), "obstacle": list(range(1000, 1300))}


def test_hold8_training_identity_binds_validated_artifact_hashes_and_protocol(tmp_path):
    h5 = tmp_path / "trajectory.h5"
    manifest = tmp_path / "trajectory.json"
    identity = tmp_path / "training_identity.json"
    h5.write_bytes(b"matched native states")
    rows = _manifest_rows()
    manifest.write_text(json.dumps({"episodes": rows}))
    payload = {
        "format": "placesphere_matched_training_identity_v1",
        "dataset_version": MATCHED_HOLD8_VERSION,
        "release_hold_steps": 8,
        "h5_sha256": _sha(h5),
        "manifest_sha256": _sha(manifest),
        "class_counts": {"direct": 1000, "obstacle": 300},
        "episode_ids": _episode_ids(),
        "validation_complete": True,
        "admitted_for_training": True,
        "native_thresholds_changed": False,
        "geometry_changed": False,
        "registration_sha256": "a" * 64,
        "dataset_validation_sha256": "b" * 64,
    }
    identity.write_text(json.dumps(payload))
    assert load_matched_training_identity(
        identity, h5, manifest, expected_version=MATCHED_HOLD8_VERSION) == payload


@pytest.mark.parametrize("change", [
    {"release_hold_steps": 16},
    {"validation_complete": False},
    {"admitted_for_training": False},
    {"geometry_changed": True},
    {"h5_sha256": "0" * 64},
])
def test_hold8_identity_rejects_protocol_or_artifact_mismatch(tmp_path, change):
    h5 = tmp_path / "trajectory.h5"
    manifest = tmp_path / "trajectory.json"
    identity = tmp_path / "training_identity.json"
    h5.write_bytes(b"native")
    rows = _manifest_rows()
    manifest.write_text(json.dumps({"episodes": rows}))
    payload = {
        "format": "placesphere_matched_training_identity_v1",
        "dataset_version": MATCHED_HOLD8_VERSION,
        "release_hold_steps": 8,
        "h5_sha256": _sha(h5),
        "manifest_sha256": _sha(manifest),
        "class_counts": {"direct": 1000, "obstacle": 300},
        "episode_ids": _episode_ids(),
        "validation_complete": True,
        "admitted_for_training": True,
        "native_thresholds_changed": False,
        "geometry_changed": False,
        "registration_sha256": "a" * 64,
        "dataset_validation_sha256": "b" * 64,
    } | change
    identity.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="identity"):
        load_matched_training_identity(identity, h5, manifest,
                                       expected_version=MATCHED_HOLD8_VERSION)
