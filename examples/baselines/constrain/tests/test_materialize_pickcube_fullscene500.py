import hashlib
import json

import h5py
import numpy as np

from examples.baselines.constrain.materialize_pickcube_fullscene500 import (
    PickCubeFullScene500Dataset, write_tree,
)


def test_native_state_series_writer_preserves_nested_t_plus_one_arrays(tmp_path):
    path = tmp_path / "states.h5"
    rows = [
        {"actors": {"cube": np.asarray([1, 2], dtype=np.float32)},
         "task": np.asarray([0], dtype=np.int32)},
        {"actors": {"cube": np.asarray([3, 4], dtype=np.float32)},
         "task": np.asarray([1], dtype=np.int32)},
    ]
    with h5py.File(path, "w") as handle:
        write_tree(handle, "env_states", rows)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["env_states/actors/cube"][:], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(handle["env_states/task"][:], [[0], [1]])


def test_obstacle_dataset_can_expose_raw_45d_frames_without_changing_default_base_view(tmp_path):
    h5_path, manifest_path = tmp_path / "trajectory.h5", tmp_path / "trajectory.json"
    episodes = []
    with h5py.File(h5_path, "w") as handle:
        for episode_id in range(1500):
            kind = "direct" if episode_id < 1000 else "obstacle"
            group = handle.create_group(f"traj_{episode_id}")
            group.attrs["kind"] = kind
            obs = np.zeros((17, 45), np.float32)
            obs[:, 42:] = episode_id
            group["obs"] = obs
            group["actions"] = np.zeros((16, 7), np.float32)
            group["success"] = np.asarray([0] * 15 + [1], np.uint8)
            group["truncated"] = np.zeros(16, np.uint8)
            episodes.append({"episode_id": episode_id, "kind": kind})
    h5_sha = hashlib.sha256(h5_path.read_bytes()).hexdigest()
    manifest = {"format": "pickcube-fullscene500-v1", "trajectory_sha256": h5_sha,
                "class_ids": {"direct": list(range(1000)), "obstacle": list(range(1000,1500))},
                "episodes": episodes}
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    base_view = PickCubeFullScene500Dataset(
        h5_path, manifest_path, expected_h5_sha256=h5_sha,
        expected_manifest_sha256=manifest_sha, kind="obstacle")
    raw_view = PickCubeFullScene500Dataset(
        h5_path, manifest_path, expected_h5_sha256=h5_sha,
        expected_manifest_sha256=manifest_sha, kind="obstacle", include_obstacle=True)
    assert base_view[0]["observations"].shape == (2, 42)
    raw = raw_view[0]["observations"]
    assert raw.shape == (2, 45)
    assert bool((raw[..., 42:] == 1000).all())
