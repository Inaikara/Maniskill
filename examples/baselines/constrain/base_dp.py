"""Reusable base policy adapter for the obstacle environment.

Loads a trained obstacle-free StackCube DP (official ManiSkill diffusion_policy
baseline) into a fresh official ``Agent`` build at the exact trained channels,
records the base weight source, and adapts the world interface only:

- consume the queued stacked 51d obstacle-env flat obs, trim to the base 48d
  condition; the appended obstacle features (x, y, radius) map nowhere,
- keep the official epsilon prediction, two-frame obs history, 100-step
  squaredcos DDPM scheduler, 16-step prediction horizon and 8-step executed
  horizon,
- map the 4d base action (xyz delta + gripper) to the 7d pd_ee_delta_pose env
  action by appending three zero rotations at execution.

The base weight source is configurable and recorded in the adapter metadata.
Checkpoint loading is strict: mismatched base weight tensors reject instead of
a silent ignore.
"""

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
CONSTRAIN_DIR = _REPO_ROOT / "examples" / "baselines" / "constrain"
OBSTACLE_STATIC_FEATURES = 3

DEFAULT_CKPT = _REPO_ROOT / "outputs" / "ckpt" / "base_dp_stackcube_980" / "best_eval_success_at_end.pt"

_OFFICIAL_DP_DIR = _REPO_ROOT / "examples" / "baselines" / "diffusion_policy"


class BasePolicyArgs:
    """Reuse-only DP config stub matching the exactly trained channels."""

    def __init__(
        self,
        obs_dim=48,
        act_dim=4,
        obs_horizon=2,
        act_horizon=8,
        pred_horizon=16,
        diffusion_step_embed_dim=64,
        n_groups=8,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.pred_horizon = pred_horizon
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.n_groups = n_groups
        self.unet_dims = [64, 128, 256]


class _PolicyEnvStub:
    """Reuse-only env stub for the official Agent build (obs/action spaces)."""

    def __init__(self, obs_dim, act_dim, obs_horizon):
        import gymnasium as gym

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_horizon, obs_dim), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )


def build_official_agent(
    obs_dim=48, act_dim=4, obs_horizon=2, act_horizon=8, pred_horizon=16, device="cpu"
):
    """Build the official BC_Diffusion_state_UNet Agent at the exact channels."""
    if str(_OFFICIAL_DP_DIR) not in sys.path:
        sys.path.insert(0, str(_OFFICIAL_DP_DIR))
    from train import Agent

    agent = Agent(
        _PolicyEnvStub(obs_dim, act_dim, obs_horizon),
        BasePolicyArgs(
            obs_dim=obs_dim,
            act_dim=act_dim,
            obs_horizon=obs_horizon,
            act_horizon=act_horizon,
            pred_horizon=pred_horizon,
        ),
    )
    return agent.to(device)


def map_action_4d_to_7d(base_seq):
    """Map a 4d base action (xyz delta + gripper) to the 7d pd_ee_delta_pose
    action by appending three zero rotations on the same device."""
    base_seq = torch.as_tensor(base_seq)
    zeros = torch.zeros(
        (*base_seq.shape[:-1], 3),
        dtype=base_seq.dtype,
        device=base_seq.device,
    )
    return torch.cat([base_seq[..., :3], zeros, base_seq[..., 3:4]], dim=-1)


class BaseDPAdapter:
    """Reusable base policy adapter consumable by later tickets at the
    obstacle environment rollout seam."""

    def __init__(
        self,
        checkpoint=DEFAULT_CKPT,
        weights_subkey="ema_agent",
        device="cpu",
        obs_dim=48,
        act_dim=4,
        obs_horizon=2,
        act_horizon=8,
        pred_horizon=16,
    ):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        if weights_subkey not in ck:
            raise ValueError(f"Checkpoint {checkpoint} lacks {weights_subkey!r}")
        self.agent = build_official_agent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            obs_horizon=obs_horizon,
            act_horizon=act_horizon,
            pred_horizon=pred_horizon,
            device=device,
        )
        self.agent.load_state_dict(ck[weights_subkey])
        self.agent.eval()
        cond_input_dim = self.agent.noise_pred_net.mid_modules[0].cond_encoder[1].weight.shape[1]
        global_cond_dim = cond_input_dim - BasePolicyArgs().diffusion_step_embed_dim
        base_obstacle_flat_dim = global_cond_dim // obs_horizon
        self.metadata = {
            "weights_source": str(checkpoint),
            "weights_subkey": weights_subkey,
            "weights_tag": "best_eval_success_at_end",
            "act_dim": int(self.agent.act_dim),
            "base_obs_dim": int(base_obstacle_flat_dim),
            "obstacle_flat_dim": int(base_obstacle_flat_dim) + OBSTACLE_STATIC_FEATURES,
            "obs_horizon": obs_horizon,
            "act_horizon": act_horizon,
            "pred_horizon": pred_horizon,
            "num_diffusion_iters": 100,
        }

    @property
    def noise_pred_net(self):
        return self.agent.noise_pred_net

    def net_state_dict(self):
        return self.agent.state_dict()

    def base_cond(self, queued_stacked):
        """Trim the 51d obstacle stacked flat obs to the base policy condition.

        queued shape (obs_horizon, 51) or (batch, obs_horizon, 51). Returns the
        (batch, obs_horizon * 48) condition input; the appended obstacle
        features (x, y, radius) map nowhere (never enter the condition).
        """
        stacked = torch.as_tensor(queued_stacked)
        obs_horizon = self.agent.obs_horizon
        if stacked.shape[-2] != obs_horizon:
            raise ValueError(
                f"Expected the queued {obs_horizon}-frame history "
                f"(shape[-2] == {obs_horizon}), got {tuple(stacked.shape)}."
            )
        if stacked.shape[-1] != 48 + OBSTACLE_STATIC_FEATURES:
            raise ValueError(
                "Expected the 51d obstacle flat layout "
                f"(..., {48 + OBSTACLE_STATIC_FEATURES}), got {tuple(stacked.shape)}."
            )
        trimmed = stacked[..., :48]
        if trimmed.dim() == 2:
            trimmed = trimmed[None]
        return trimmed.flatten(start_dim=1)

    def map_action_ee_delta_pose(self, base_seq):
        """Execute the 4d base action (xyz delta + gripper) in the 7d
        pd_ee_delta_pose env: [xyz, zeros(3) rotations, gripper]."""
        return map_action_4d_to_7d(base_seq)

    def select_executed(self, pred_seq):
        """Keep the official executed horizon start=end+act_horizon slices off the
        16-step prediction."""
        pred_seq = torch.as_tensor(pred_seq)
        start = self.agent.obs_horizon - 1
        end = start + self.agent.act_horizon
        return pred_seq[:, start:end]

    def get_action(self, obs_seq):
        """The official DP rollout: a fresh torch randn noise stream then 100 DDPM
        steps at the same noise timesteps, slicing the executed horizon."""
        obs_seq = torch.as_tensor(obs_seq)
        with torch.no_grad():
            executed = self.agent.get_action(obs_seq)
        return executed
