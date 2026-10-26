import json

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain.placesphere_base_dp import (
    FORMAT,
    MATCHED_HOLD8_VERSION,
    NAMED_FIELDS,
    PlaceSphereBaseDPAdapter,
    PlaceSphereBaseDataset,
    TRUSTED_CONTRACT_SHA256,
    TRUSTED_H5_SHA256,
    TRUSTED_MANIFEST_SHA256,
    TRUSTED_REPORT_SHA256,
    contract,
)


def _formal_fixture(tmp_path, bad_id=None):
    path, manifest = tmp_path / "trajectory.h5", tmp_path / "trajectory.json"
    episodes = []
    with h5py.File(path, "w") as handle:
        for episode_id in range(1100):
            kind = "direct" if episode_id < 1000 else "obstacle"
            group = handle.create_group(f"traj_{episode_id}")
            group.attrs["kind"] = "obstacle" if episode_id == bad_id else kind
            obs = np.arange(4 * 42, dtype=np.float32).reshape(4, 42) + episode_id
            actions = np.zeros((3, 7), dtype=np.float32)
            actions[:, 0] = [0.1, 0.2, 0.3]  # retain post-release retreat
            actions[:, 3:6] = [0.11, -0.12, 0.13]
            actions[:, 6] = [-1, 1, 1]  # release, then native open
            group["raw_obs"] = obs
            group["obs"] = obs[:, :39]
            group["actions"] = actions
            group["success"] = np.array([0, 0, 1], np.uint8)
            group["terminated"] = np.array([0, 0, 1], np.uint8)
            group["truncated"] = np.zeros(3, np.uint8)
            episodes.append({"episode_id": episode_id, "kind": kind,
                             "terminated": [0, 0, 1], "truncated": [0, 0, 0]})
    manifest.write_text(json.dumps({"episodes": episodes}))
    return path, manifest


def test_placesphere_contract_is_native_39_of_42_and_seven_action_channels():
    cfg = contract()
    assert cfg["policy"]["obs_dim"] == 39
    assert cfg["policy"]["raw_obs_dim"] == 42
    assert cfg["policy"]["act_dim"] == 7
    assert cfg["policy"]["control_mode"] == "pd_ee_delta_pose"
    assert cfg["policy"]["action_fields"][3:6] == ["delta_euler_x", "delta_euler_y", "delta_euler_z"]


def test_dataset_uses_only_direct_ids_and_open_zero_arm_tail_padding(tmp_path):
    path, manifest = _formal_fixture(tmp_path)
    ds = PlaceSphereBaseDataset(path, manifest, obs_horizon=2, pred_horizon=4)
    assert ds.trajectory_ids == list(range(1000))
    sample = ds[0]
    assert sample["observations"].shape == (2, 39)
    assert sample["actions"].shape == (4, 7)
    assert torch.equal(sample["actions"][0], sample["actions"][1])
    # Initial window repeats first action; subsequent windows retain release/retreat.
    assert sample["actions"][0, 0] == pytest.approx(0.1)
    assert sample["actions"][1, 0] == pytest.approx(0.1)
    assert ds[1]["actions"][1, 0] == pytest.approx(0.2)
    assert ds[1]["actions"][2, 0] == pytest.approx(0.3)
    assert ds[1]["actions"][1, 6] == pytest.approx(1.0)  # native release remains present
    assert ds[1]["actions"][2, 6] == pytest.approx(1.0)  # post-release retreat remains present
    padded = ds[1]["actions"]
    assert torch.equal(padded[-1, :6], torch.zeros(6))
    assert padded[-1, 6] == pytest.approx(1.0)


def test_dataset_rejects_obstacle_episode_in_direct_manifest_selection(tmp_path):
    path, manifest = _formal_fixture(tmp_path, bad_id=17)
    with pytest.raises(ValueError, match="direct"):
        PlaceSphereBaseDataset(path, manifest, obs_horizon=2, pred_horizon=4)


def test_adapter_condition_excludes_named_obstacle_and_strictly_rejects_contract(tmp_path):
    adapter = object.__new__(PlaceSphereBaseDPAdapter)
    adapter.obs_horizon, adapter.obs_dim, adapter.raw_obs_dim = 2, 39, 42
    raw = torch.randn(3, 2, 42)
    raw[..., 39:] = 4
    cond = adapter.base_cond(raw)
    changed = raw.clone()
    changed[..., 39:] = -9
    assert torch.equal(cond, adapter.base_cond(changed))
    with pytest.raises(ValueError, match="42"):
        adapter.base_cond(torch.randn(2, 41))
    ckpt = tmp_path / "wrong.pt"
    wrong_contract = contract()
    wrong_contract["policy"]["act_dim"] = 4
    torch.save({"format": FORMAT, "contract": wrong_contract}, ckpt)
    with pytest.raises(ValueError, match="contract"):
        PlaceSphereBaseDPAdapter.validate_checkpoint(ckpt)


def test_checkpoint_loader_requires_exact_formal_direct_ids_and_trusted_data(tmp_path):
    checkpoint = tmp_path / "metadata.pt"
    data = {"selected_class": "direct", "selected_episode_ids": list(range(1000)),
            "episodes": 1000, "h5_sha256": TRUSTED_H5_SHA256,
            "manifest_sha256": TRUSTED_MANIFEST_SHA256,
            "frozen_contract_sha256": TRUSTED_CONTRACT_SHA256,
            "collection_report_sha256": TRUSTED_REPORT_SHA256}
    schema = {"base_dim": 39, "flat_dim": 42,
              "obstacle_field": "extra/placesphere_obstacle",
              "field_order": [field[0] for field in NAMED_FIELDS]}
    valid = {"format": FORMAT, "contract": contract(), "data_identity": data,
             "named_state_schema": schema}
    torch.save(valid, checkpoint)
    assert PlaceSphereBaseDPAdapter.validate_checkpoint(checkpoint)["format"] == FORMAT
    for changed_data in (
        {**data, "selected_episode_ids": list(range(1, 1001))},
        {**data, "h5_sha256": "0" * 64},
        {**data, "manifest_sha256": "0" * 64},
    ):
        torch.save({**valid, "data_identity": changed_data}, checkpoint)
        with pytest.raises(ValueError, match="contract|identity|IDs|trusted formal"):
            PlaceSphereBaseDPAdapter.validate_checkpoint(checkpoint)


def test_checkpoint_loader_accepts_only_versioned_hold8_matched_direct_identity(tmp_path):
    checkpoint = tmp_path / "matched_hold8.pt"
    direct_ids = list(range(0, 600, 2)) + list(range(600, 1300))
    data = {"dataset_version": MATCHED_HOLD8_VERSION, "release_hold_steps": 8,
            "selected_class": "direct", "selected_episode_ids": direct_ids,
            "episodes": 1000, "h5_sha256": "a" * 64, "manifest_sha256": "b" * 64}
    schema = {"base_dim": 39, "flat_dim": 42,
              "obstacle_field": "extra/placesphere_obstacle",
              "field_order": [field[0] for field in NAMED_FIELDS]}
    valid = {"format": FORMAT, "contract": contract(), "data_identity": data,
             "named_state_schema": schema}
    torch.save(valid, checkpoint)
    assert PlaceSphereBaseDPAdapter.validate_checkpoint(checkpoint)["data_identity"] == data
    for changed in ({**data, "release_hold_steps": 16},
                    {**data, "dataset_version": "unregistered"},
                    {**data, "episodes": 999},
                    {**data, "selected_episode_ids": direct_ids[:-1] + [1300]}):
        torch.save({**valid, "data_identity": changed}, checkpoint)
        with pytest.raises(ValueError, match="contract"):
            PlaceSphereBaseDPAdapter.validate_checkpoint(checkpoint)


def test_native_action_mapping_enforces_placesphere_pose_bounds_without_dropping_rotation():
    action = torch.tensor([[0.5, -0.5, 0.2, 0.4, -0.4, 0.1, -1.0]])
    native = PlaceSphereBaseDPAdapter.map_action_ee_delta_pose(action)
    assert native.shape == (1, 7)
    assert torch.equal(native[0, 3:6], torch.tensor([0.35, -0.35, 0.1]))
    assert torch.equal(native[0, :3], torch.tensor([0.35, -0.35, 0.2]))
    assert native[0, 6] == -1.0
