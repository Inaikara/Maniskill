import json

import h5py
import numpy as np
import pytest

from examples.baselines.constrain.scene_set import SceneSet, _stable_hash


def _write_scene_set(directory):
    state = {"actors": {"cubeA": np.arange(13, dtype=np.float32)}}
    with h5py.File(directory / "scenes.h5", "w") as handle:
        scene = handle.create_group("wristcam-000")
        leaf = scene.create_group("env_states").create_group("actors")
        leaf.create_dataset("cubeA", data=state["actors"]["cubeA"][None])
        scene.create_dataset("initial_observation", data=np.array([1.0, 2.0]))
    (directory / "manifest.json").write_text(json.dumps({
        "format": "maniskill-wristcam-scenes-v1",
        "scene_count": 1,
        "scenes_file": "scenes.h5",
        "scenes": [{
            "scene_id": "wristcam-000",
            "initial_state_sha256": _stable_hash(state),
            "cylinder": {"radius": 0.03, "height": 0.18},
        }],
    }))


def test_scene_set_loads_complete_state_without_collection_assets(tmp_path):
    _write_scene_set(tmp_path)

    catalog = SceneSet(tmp_path)
    scene = catalog.load(0)

    assert len(catalog) == 1
    np.testing.assert_array_equal(scene.initial_state["actors"]["cubeA"], np.arange(13))
    np.testing.assert_array_equal(scene.initial_observation, [1.0, 2.0])
    assert scene.metadata["cylinder"]["radius"] == pytest.approx(0.03)


def test_scene_set_fails_closed_when_physical_state_changes(tmp_path):
    _write_scene_set(tmp_path)
    with h5py.File(tmp_path / "scenes.h5", "a") as handle:
        handle["wristcam-000/env_states/actors/cubeA"][0, 0] = -1

    with pytest.raises(ValueError, match="checksum"):
        SceneSet(tmp_path).load(0)
