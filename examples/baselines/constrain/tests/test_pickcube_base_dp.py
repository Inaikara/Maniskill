import json

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain.pickcube_base_dp import (
    PickCubeBaseDPAdapter,
    PickCubeBaseDataset,
    contract,
)


def test_contract_keeps_native_pickcube_dimensions_and_horizons():
    cfg = contract()
    assert cfg["policy"] == {
        "robot": "panda", "obs_dim": 42, "obstacle_obs_dim": 45,
        "act_dim": 7, "obs_horizon": 2, "pred_horizon": 16,
        "act_horizon": 8, "control_mode": "pd_ee_delta_pose",
    } | {"observation_mode": "state", "observation_fields": [
        ["agent/qpos", 9], ["agent/qvel", 9], ["extra/is_grasped", 1],
        ["extra/tcp_pose", 7], ["extra/goal_pos", 3], ["extra/obj_pose", 7],
        ["extra/tcp_to_obj_pos", 3], ["extra/obj_to_goal_pos", 3],
        ["extra/pickcube_obstacle", 3]],
        "obstacle_feature_semantics": "pickcube_obstacle world XY center and radius",
        "action_fields": ["delta_x", "delta_y", "delta_z", "delta_euler_x",
                          "delta_euler_y", "delta_euler_z", "gripper"],
        "translation_scale_m": 0.1,
        "rotation_scale": "signed controller XYZ Euler range [-0.1,0.1] rad",
        "action_bounds": [-1.0, 1.0]}


def test_direct_only_windows_use_official_padding_and_keep_all_action_channels(tmp_path):
    path = tmp_path / "trajectory.h5"
    manifest = tmp_path / "trajectory.json"
    with h5py.File(path, "w") as f:
        for i in range(1100):
            g = f.create_group(f"traj_{i}")
            g.attrs["kind"] = "direct" if i < 1000 else "obstacle"
            g["obs"] = np.full((4, 45), i, np.float32)
            g["actions"] = np.full((3, 7), 0.1, np.float32)
            g["success"] = np.array([0, 0, 1], np.uint8)
            g["terminated"] = np.array([0, 0, 1], np.uint8)
            g["truncated"] = np.zeros(3, np.uint8)
    manifest.write_text(json.dumps({"episodes": [
        {"episode_id": i, "kind": "direct" if i < 1000 else "obstacle",
         "terminated": [0, 0, 1], "truncated": [0, 0, 0]} for i in range(1100)
    ]}))
    ds = PickCubeBaseDataset(path, manifest, obs_horizon=2, pred_horizon=4)
    assert len(ds.trajectory_ids) == 1000
    assert 1000 not in ds.trajectory_ids
    assert len(ds) == 2000
    first = ds[0]
    assert first["observations"].shape == (2, 42)
    assert first["actions"].shape == (4, 7)
    assert torch.equal(first["actions"][0], first["actions"][1])
    padded = ds[1]["actions"]
    assert padded[-1, :6].eq(0).all()
    assert padded[-1, 6] == pytest.approx(0.1)


def test_adapter_discards_only_verified_named_obstacle_tail_and_rejects_bad_shapes():
    adapter = object.__new__(PickCubeBaseDPAdapter)
    adapter.obs_horizon = 2
    adapter.obs_dim = 42
    adapter.obstacle_obs_dim = 45
    data = torch.randn(2, 2, 45)
    conditioned = adapter.base_cond(data)
    changed = data.clone()
    changed[..., 42:] += 100
    assert torch.equal(conditioned, adapter.base_cond(changed))
    with pytest.raises(ValueError, match="45"):
        adapter.base_cond(torch.randn(2, 2, 48))


def test_checkpoint_metadata_mismatch_fails_strictly(tmp_path):
    ckpt = tmp_path / "bad.pt"
    torch.save({"format": "pickcube_base_dp_v1", "config": {"policy": {"act_dim": 4}}}, ckpt)
    with pytest.raises(ValueError, match="contract"):
        PickCubeBaseDPAdapter.validate_checkpoint(ckpt)


def test_native_action_adapter_rejects_illegal_values():
    adapter = object.__new__(PickCubeBaseDPAdapter)
    valid = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -1.0]])
    assert torch.equal(adapter.map_action_ee_delta_pose(valid), valid)
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        adapter.map_action_ee_delta_pose(torch.tensor([[0.0] * 6 + [1.01]]))
