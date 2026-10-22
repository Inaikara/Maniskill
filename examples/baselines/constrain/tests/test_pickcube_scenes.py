import hashlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain.pickcube_expert import make_pickcube_env
from examples.baselines.constrain.pickcube_scenes import (
    CONTRACT, RUNTIME_CONTRACT, PickCubeSceneSet, _sha, _tree_hash, _write_tree,
)


def _json(value):
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _fixture(directory, env, unsafe=False, unsafe_expert=False):
    base = env.unwrapped
    env.reset(seed=27, options={"obstacle_enabled": True, "obstacle_center": [0.0, 0.22]})
    state = base.get_state_dict()
    if unsafe:
        state["pickcube_obstacle"][0, 4] = 1
    controller = base.agent.controller.get_state()
    observation = base.get_obs().detach().cpu().numpy().reshape(-1)
    with h5py.File(directory / "scenes.h5", "w") as handle:
        scene = handle.create_group("pickcube-000")
        _write_tree(scene.create_group("state"), state)
        scene.create_dataset("initial_observation", data=observation)
        scene.attrs["controller_json"] = json.dumps(_json(controller))
    (directory / "config.json").write_text("{}\n")
    row = {"scene_id": "pickcube-000", "seed": 27, "start_xy": [-0.12, 0.07],
           "goal_xy": [0.12, 0.07],
           "cylinder": {"center_xy": [0.0, 0.22], "radius": 0.03, "height": 0.25},
           "state_sha256": _tree_hash(state),
           "controller_sha256": _tree_hash(controller),
           "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
           "expert": {"native_success": not unsafe_expert,
                      "obstacle_collision": unsafe_expert,
                      "obstacle_crossing": False}}
    manifest = {"format": "pickcube-obstacle-scenes-v1", "scenes_file": "scenes.h5",
                "scenes": [row], "scenes_file_sha256": _sha(directory / "scenes.h5"),
                "config_sha256": _sha(directory / "config.json"),
                "frozen_contract_sha256": _sha(CONTRACT),
                "runtime_contract_sha256": _sha(RUNTIME_CONTRACT)}
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, indent=2).encode()).hexdigest()
    (directory / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))


def test_unsafe_initial_state_is_rejected_even_when_hashes_match():
    env = make_pickcube_env()
    try:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            _fixture(path, env, unsafe=True)
            with pytest.raises(ValueError, match="unsafe"):
                PickCubeSceneSet(path).load(0)
    finally:
        env.close()


def test_scene_with_expert_safety_failure_is_rejected_even_with_valid_manifest_hash():
    env = make_pickcube_env()
    try:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            _fixture(path, env, unsafe_expert=True)
            with pytest.raises(ValueError, match="expert acceptance"):
                PickCubeSceneSet(path).load(0)
    finally:
        env.close()


def test_scene_restore_checks_real_geometry_and_restores_state_and_controller(tmp_path):
    env = make_pickcube_env()
    wrong_geometry = make_pickcube_env(obstacle_radius=0.04)
    try:
        _fixture(tmp_path, env)
        catalog = PickCubeSceneSet(tmp_path)
        with pytest.raises(ValueError, match="geometry"):
            catalog.restore(0, wrong_geometry)
        before = catalog.load(0)
        env.step(torch.zeros(7))
        restored, observation = catalog.restore(0, env)
        assert restored is env
        np.testing.assert_array_equal(observation.detach().cpu().numpy().reshape(-1),
                                      before["observation"])
        assert _tree_hash(env.unwrapped.get_state_dict()) == before["metadata"]["state_sha256"]
        assert _tree_hash(env.unwrapped.agent.controller.get_state()) == before["metadata"]["controller_sha256"]
    finally:
        env.close()
        wrong_geometry.close()
