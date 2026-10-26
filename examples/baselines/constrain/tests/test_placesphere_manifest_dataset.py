import hashlib
import json

import h5py
import pytest

from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_H5_SHA256,
    MATCHED_MANIFEST_SHA256,
    PlaceSphereManifestDataset,
)


DATA = "demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b"


def test_matched_reader_selects_interleaved_direct_ids_and_returns_official_window():
    dataset = PlaceSphereManifestDataset(
        f"{DATA}/trajectory.h5", f"{DATA}/trajectory.json", "direct")

    assert len(dataset.trajectory_ids) == 1000
    assert dataset.trajectory_ids != list(range(1000))
    with h5py.File(f"{DATA}/trajectory.h5", "r") as h5:
        assert all(h5[f"traj_{episode_id}"].attrs["kind"] == "direct"
                   for episode_id in dataset.trajectory_ids)
        assert any(h5[f"traj_{episode_id}"].attrs["kind"] == "obstacle"
                   for episode_id in set(range(1300)) - set(dataset.trajectory_ids))
    sample = dataset[0]
    assert sample["observations"].shape == (2, 39)
    assert sample["raw_observations"].shape == (2, 42)
    assert sample["actions"].shape == (16, 7)


def test_matched_reader_rejects_unpinned_hdf5_identity():
    with pytest.raises(ValueError, match="HDF5 identity"):
        PlaceSphereManifestDataset(
            f"{DATA}/trajectory.h5", f"{DATA}/trajectory.json", "obstacle",
            expected_h5_sha256="0" * 64,
            expected_manifest_sha256=MATCHED_MANIFEST_SHA256)


def test_matched_reader_rejects_unpinned_manifest_identity():
    with pytest.raises(ValueError, match="manifest identity"):
        PlaceSphereManifestDataset(
            f"{DATA}/trajectory.h5", f"{DATA}/trajectory.json", "obstacle",
            expected_h5_sha256=MATCHED_H5_SHA256,
            expected_manifest_sha256="0" * 64)


def test_matched_reader_rejects_manifest_class_disagreement_with_hdf5(tmp_path):
    manifest = json.loads(open(f"{DATA}/trajectory.json").read())
    direct = next(item for item in manifest["episodes"] if item["kind"] == "direct")
    obstacle = next(item for item in manifest["episodes"] if item["kind"] == "obstacle")
    direct["kind"], obstacle["kind"] = "obstacle", "direct"
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="manifest/HDF5 class mismatch"):
        PlaceSphereManifestDataset(
            f"{DATA}/trajectory.h5", path, "direct",
            expected_h5_sha256=MATCHED_H5_SHA256,
            expected_manifest_sha256=digest)


def test_matched_reader_rejects_missing_class_labels(tmp_path):
    manifest = json.loads(open(f"{DATA}/trajectory.json").read())
    manifest["episodes"][0].pop("kind")
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="missing episode_id or kind"):
        PlaceSphereManifestDataset(
            f"{DATA}/trajectory.h5", path, "direct",
            expected_h5_sha256=MATCHED_H5_SHA256,
            expected_manifest_sha256=digest)
