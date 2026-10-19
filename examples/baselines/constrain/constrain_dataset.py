"""Raw obstacle demonstration dataset for the constrain policy (Ticket 03).

Reads the raw 100-trajectory ``StackCubeObstacle-v1`` demonstration HDF5
produced by ``collect_constructive`` directly — no holdout split, no
re-collection, no rewritten data files — and keeps the official ManiSkill
Diffusion Policy training windowing exactly:

- all trajectories are used (``num_demos=None`` semantics),
- every trajectory's three rotation action channels (3:6) are verified zero
  at load; a violated file fails the build,
- the 7d ``pd_ee_delta_pose`` actions map to the base 4d
  ``[x, y, z, gripper]`` (channels 0, 1, 2, 6),
- the official ``(traj_idx, start, end)`` slice set from
  ``examples/baselines/diffusion_policy/train.py`` (pad-before repeats the
  first observation/action, pad-after zeroes the xyz displacement and holds
  the last gripper value), so the epsilon denoising objective and horizon
  semantics match the base run.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

OBS_DIM = 51
DEMO_ROTATION_CHANNELS = (3, 4, 5)
DEMO_GRIPPER_CHANNEL = 6
# Official 4d base action layout: xyz delta + gripper.
ACT_DIM = 4


def official_window_slices(trajectory_lengths, obs_horizon=2, pred_horizon=16):
    """The official DP ``(traj_idx, start, end)`` window slices.

    Replicates ``SmallDemoDataset_DiffusionPolicy`` from
    ``examples/baselines/diffusion_policy/train.py``: per trajectory of
    length ``L`` the start runs over ``[-(obs_horizon-1), L - pred_horizon +
    pad_after)`` with ``pad_before = obs_horizon - 1`` and
    ``pad_after = pred_horizon - obs_horizon``.
    """
    pad_before = obs_horizon - 1
    pad_after = pred_horizon - obs_horizon
    slices = []
    for traj_idx, L in enumerate(trajectory_lengths):
        slices += [
            (traj_idx, start, start + pred_horizon)
            for start in range(-pad_before, L - pred_horizon + pad_after)
        ]
    return slices


class ConstrainDemoDataset(Dataset):
    """All obstacle demonstrations as official DP training windows.

    Returns ``{'observations': (obs_horizon, 51), 'actions': (pred_horizon, 4)}``;
    the relative condition is computed from the window's latest observation
    frame by the shared condition seam at loss time, exactly as the rollout
    adapter does at inference.
    """

    def __init__(
        self,
        data_path,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        act_dim: int = ACT_DIM,
    ):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.act_dim = act_dim
        trajectories_obs = []
        trajectories_act = []
        lengths = []
        with h5py.File(data_path, "r") as f:
            keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
            for key in keys:
                ep = f[key]
                obs = ep["obs"][:]  # (L + 1, 51)
                actions = ep["actions"][:]  # (L, 7)
                if actions.shape[1] != 7:
                    raise ValueError(
                        f"{key}: expected 7d pd_ee_delta_pose actions, got shape {actions.shape}"
                    )
                if np.abs(actions[:, DEMO_ROTATION_CHANNELS]).max() != 0.0:
                    raise ValueError(
                        f"{key}: demonstration rotation channels "
                        f"{DEMO_ROTATION_CHANNELS} are not all zero; refusing to "
                        "train on rotation targets"
                    )
                if obs.shape[0] != actions.shape[0] + 1:
                    raise ValueError(
                        f"{key}: expected L + 1 observations for L actions, got "
                        f"{obs.shape[0]} obs for {actions.shape[0]} actions"
                    )
                trajectories_obs.append(torch.tensor(obs, dtype=torch.float32))
                # 7d -> 4d: [x, y, z, gripper]; rotation channels map nowhere
                trajectories_act.append(
                    torch.tensor(
                        actions[:, [0, 1, 2, DEMO_GRIPPER_CHANNEL]], dtype=torch.float32
                    )
                )
                lengths.append(int(actions.shape[0]))
        self.trajectory_lengths = lengths
        self.trajectories_obs = trajectories_obs
        self.trajectories_act = trajectories_act
        # All 100 trajectories, no holdout.
        self.slices = official_window_slices(
            lengths, obs_horizon=obs_horizon, pred_horizon=pred_horizon
        )

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L = self.trajectory_lengths[traj_idx]
        obs = self.trajectories_obs[traj_idx]
        act = self.trajectories_act[traj_idx]

        obs_seq = obs[max(0, start):start + self.obs_horizon]
        act_seq = act[max(0, start):end]
        if start < 0:  # pad before the trajectory
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:  # pad after the trajectory
            # official: arm displacement stays still (zeros), gripper holds
            pad_action = torch.cat(
                (torch.zeros(3), act_seq[-1, -1][None]), dim=0
            )
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        assert obs_seq.shape[0] == self.obs_horizon
        assert act_seq.shape[0] == self.pred_horizon
        return {"observations": obs_seq, "actions": act_seq}
