"""Tests for the obstacle demonstration dataset seam (Ticket 03).

Seams under test (public interface of ``ConstrainDemoDataset``):
- the raw 100-trajectory obstacle demonstration HDF5 is read directly (no
  holdout split, no re-collection, no rewritten data files),
- every trajectory's three rotation action channels are verified zero at
  load (a violated file fails the build),
- the 7d ``pd_ee_delta_pose`` actions map to the base 4d (xyz + gripper),
- the official ManiSkill DP trajectory windowing is kept exactly: the same
  ``(traj_idx, start, end)`` slice set, pad-before repeats the first
  observation/action, pad-after zeroes the xyz displacement and holds the
  last gripper value.
"""

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain.constrain_dataset import (
    DEMO_ROTATION_CHANNELS,
    ConstrainDemoDataset,
    official_window_slices,
)

REAL_DEMO = "outputs/stackcube_obstacle/trajectory.h5"


def write_fake_demo(path, lengths=(6, 9), obs_dim=51, act_dim=7):
    """A minimal raw demonstration file with the official episode layout:
    ``L`` actions and ``L + 1`` observations per trajectory."""
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for i, L in enumerate(lengths):
            ep = f.create_group(f"traj_{i}")
            ep.create_dataset("obs", data=rng.standard_normal((L + 1, obs_dim)).astype(np.float32))
            actions = rng.standard_normal((L, act_dim)).astype(np.float32)
            actions[:, 3:6] = 0.0  # constructive expert commands zero rotation
            ep.create_dataset("actions", data=actions)


def test_official_window_slices_formula():
    """The slice set matches the official DP windowing exactly: start runs
    over ``[-(obs_horizon-1), L - pred + pad_after)`` per trajectory."""
    # official formula from examples/baselines/diffusion_policy/train.py
    obs_horizon, pred_horizon = 2, 16
    pad_before = obs_horizon - 1
    pad_after = pred_horizon - obs_horizon
    lengths = [6, 9, 30]
    expected = []
    for traj_idx, L in enumerate(lengths):
        expected += [
            (traj_idx, start, start + pred_horizon)
            for start in range(-pad_before, L - pred_horizon + pad_after)
        ]
    assert official_window_slices(lengths, obs_horizon=2, pred_horizon=16) == expected
    assert len(expected) == sum(L - 1 for L in lengths)


def test_real_dataset_all_100_no_holdout():
    """The real obstacle demo file loads fully: 100 trajectories, every
    transition slice available (no holdout), 51d obs and 4d action windows."""
    ds = ConstrainDemoDataset(REAL_DEMO)
    assert len(ds.trajectory_lengths) == 100
    assert len(ds) == 10456 - 100  # sum(L) - num_traj slices
    item = ds[0]
    assert item["observations"].shape == (2, 51)
    assert item["actions"].shape == (16, 4)
    assert item["observations"].dtype == torch.float32
    assert item["actions"].dtype == torch.float32


def test_real_dataset_rotation_channels_verified_zero():
    """The real demos carry exactly zero rotation commands; the dataset
    verified this at load (see the guard test below)."""
    import h5py as _h5

    with _h5.File(REAL_DEMO, "r") as f:
        for key in f:
            actions = f[key]["actions"][:]
            assert np.abs(actions[:, 3:6]).max() == 0.0


def test_rotation_check_rejects_violating_file(tmp_path):
    """A demonstration file with nonzero rotation commands fails the build
    instead of silently training on rotation targets."""
    path = tmp_path / "bad.h5"
    write_fake_demo(path)
    with h5py.File(path, "a") as f:
        actions = f["traj_0"]["actions"]
        actions[0, 4] = 0.01
    with pytest.raises(ValueError, match="rotation"):
        ConstrainDemoDataset(path)


def test_action_mapping_7d_to_4d(tmp_path):
    """The 4d base action is ``[x, y, z, gripper]``: xyz displacement channels
    and the last gripper channel; the rotation channels map nowhere."""
    path = tmp_path / "demo.h5"
    write_fake_demo(path)
    ds = ConstrainDemoDataset(path)
    with h5py.File(path, "r") as f:
        raw = f["traj_0"]["actions"][:]
    item = ds[0]  # (traj 0, start=-1, end=15): [act0, act0, act1..act5, pad...]
    assert DEMO_ROTATION_CHANNELS == (3, 4, 5)
    # rows 2..6 replay the true 4d actions 1..5 (pad-before duplicates act0)
    torch.testing.assert_close(item["actions"][2:7], torch.tensor(raw[1:, [0, 1, 2, 6]]))
    torch.testing.assert_close(item["actions"][0], torch.tensor(raw[0, [0, 1, 2, 6]]))
    torch.testing.assert_close(item["actions"][1], torch.tensor(raw[0, [0, 1, 2, 6]]))


def test_official_padding_semantics(tmp_path):
    """Pad-before repeats the first observation/action; pad-after zeroes the
    xyz displacement and holds the last gripper value (official DP behavior
    on 4d actions)."""
    path = tmp_path / "demo.h5"
    write_fake_demo(path, lengths=(20, 20))  # long enough for unpadded + tail windows
    ds = ConstrainDemoDataset(path)
    # traj slice list for L=20: start in [-1, 18) -> first slice pads before
    first = ds[0]  # (traj 0, start=-1, end=15)
    with h5py.File(path, "r") as f:
        obs0 = f["traj_0"]["obs"][:]
        act0 = f["traj_0"]["actions"][:]
    torch.testing.assert_close(first["observations"][0], first["observations"][1])
    torch.testing.assert_close(first["observations"][1], torch.tensor(obs0[0]))
    torch.testing.assert_close(first["actions"][0], torch.tensor(act0[0, [0, 1, 2, 6]]))
    torch.testing.assert_close(first["actions"][1], torch.tensor(act0[0, [0, 1, 2, 6]]))
    torch.testing.assert_close(first["actions"][2], torch.tensor(act0[1, [0, 1, 2, 6]]))
    # a tail window pads after: xyz zeros + last gripper
    slices = ds.slices
    last_traj0 = [s for s in slices if s[0] == 0][-1]  # start=17, end=33 > 20
    tail = ds[slices.index(last_traj0)]
    torch.testing.assert_close(tail["actions"][-1, :3], torch.zeros(3))
    torch.testing.assert_close(tail["actions"][-1, 3], torch.tensor(act0[-1, 6]))
    torch.testing.assert_close(tail["actions"][-2, 3], torch.tensor(act0[-1, 6]))
    # observations never pad after (official: only actions pad after); the
    # unpadded tail window holds obs[17:19]
    torch.testing.assert_close(tail["observations"][1], torch.tensor(obs0[18]))
    assert tail["actions"].shape == (16, 4)
    assert tail["observations"].shape == (2, 51)


def test_short_trajectory_fully_padded(tmp_path):
    """A trajectory shorter than the prediction horizon produces two windows
    with the official both-sides padding."""
    path = tmp_path / "short.h5"
    write_fake_demo(path, lengths=(3, 3))
    ds = ConstrainDemoDataset(path)
    assert len(ds) == 4  # (L - 1) slices per trajectory, two trajectories
    item = ds[0]  # (traj 0, start=-1, end=15)
    assert item["actions"].shape == (16, 4)
    with h5py.File(path, "r") as f:
        act0 = f["traj_0"]["actions"][:]
    # [act0, act0, act1, act2] then pad-after rows holding act2's gripper
    torch.testing.assert_close(item["actions"][0], torch.tensor(act0[0, [0, 1, 2, 6]]))
    torch.testing.assert_close(item["actions"][1], torch.tensor(act0[0, [0, 1, 2, 6]]))
    torch.testing.assert_close(item["actions"][3], torch.tensor(act0[2, [0, 1, 2, 6]]))
    torch.testing.assert_close(item["actions"][4, :3], torch.zeros(3))
    torch.testing.assert_close(item["actions"][-1, :3], torch.zeros(3))
    torch.testing.assert_close(item["actions"][-1, 3], torch.tensor(act0[2, 6]))


def test_real_dataset_window_counts_match_official(tmp_path):
    """The real dataset's per-trajectory slice counts equal the official
    ``L - 1`` window count."""
    ds = ConstrainDemoDataset(REAL_DEMO)
    counts = {}
    for traj_idx, _, _ in ds.slices:
        counts[traj_idx] = counts.get(traj_idx, 0) + 1
    assert all(count == L - 1 for traj_idx, count in counts.items()
               for L in [ds.trajectory_lengths[traj_idx]])
    assert set(counts) == set(range(100))
