"""Closed-loop rollout adapter for the constrain policy (Ticket 03).

``ConstrainDPAdapter`` plugs the trained ConstrainNet into the exact Ticket 01
/ Ticket 02 rollout seam — ``base_cond(stacked) -> get_action(cond) ->
map_action_ee_delta_pose(executed)`` — so the existing evaluator and rollout
CLIs run it without changes:

- ``base_cond`` performs the same unchanged 2x51 -> (B, 96) base-condition
  trim and *captures the full 2x51 stacked observation*: the relative
  condition hint is read from its most recent frame afterwards, so the
  appended obstacle features (x, y, radius) are never lost by the premature
  48d trim,
- ``get_action`` mirrors the official ``Agent.get_action`` epsilon/DDPM loop
  (100 squared-cos diffusion steps, prediction horizon 16, executed slice
  ``[obs_horizon-1, obs_horizon-1+act_horizon)``) while feeding the repeated
  3d relative condition as the control-branch hint; the same shared
  normalization config recorded in the checkpoint is used here, exactly as at
  training time,
- with ``control_enabled=False`` the control branch is bypassed entirely and
  the adapter reproduces the base DP adapter's rollout under the same noise
  stream,
- ``map_action_ee_delta_pose`` reuses the shared 4d -> 7d mapping.

The adapter loads strictly from the standalone constrain checkpoint (frozen
base weights stored inside; no dependency on the original base checkpoint
file) and records its config/metadata.
"""

from pathlib import Path

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from examples.baselines.constrain.base_dp import map_action_4d_to_7d
from examples.baselines.constrain.constrain_condition import (
    ConditionNormConfig,
    build_hint,
    latest_relative_condition,
)
from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
from examples.baselines.constrain.constrain_ckpt import CHECKPOINT_FORMAT, load_constrain_denoiser

DEFAULT_CKPT = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "ckpt"
    / "constrainnet_wristcam_seed1"
    / "final.pt"
)


class ConstrainDPAdapter:
    """Rollout adapter for the obstacle-conditioned ConstrainNet.

    The adapter is intentionally stateful only between ``base_cond`` and the
    following ``get_action`` call (the exact call pattern the Ticket 02
    evaluator uses): ``base_cond`` captures the full stacked observation and
    ``get_action`` consumes its latest frame for the hint.
    """

    def __init__(self, checkpoint, device="cpu", weights="ema", control_enabled=True):
        self.denoiser, self.checkpoint = load_constrain_denoiser(
            checkpoint, device=device, weights=weights
        )
        self.denoiser.eval()
        config = self.checkpoint["config"]
        policy = config["policy"]
        self.control_enabled = control_enabled
        self.obs_horizon = int(policy["obs_horizon"])
        self.act_horizon = int(policy["act_horizon"])
        self.pred_horizon = int(policy["pred_horizon"])
        self.base_obs_dim = int(policy["base_obs_dim"])
        self.act_dim = int(policy["act_dim"])
        self.num_diffusion_iters = int(config["model"]["num_diffusion_iters"])
        self.norm_cfg = ConditionNormConfig.from_dict(config["normalization"])
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self._latest_stacked = None
        self.metadata = dict(
            checkpoint=str(checkpoint),
            format=CHECKPOINT_FORMAT,
            weights=weights,
            step=int(self.checkpoint["step"]),
            base_weights=config["base_weights"],
            act_dim=self.act_dim,
            base_obs_dim=self.base_obs_dim,
            obs_horizon=self.obs_horizon,
            act_horizon=self.act_horizon,
            pred_horizon=self.pred_horizon,
            num_diffusion_iters=self.num_diffusion_iters,
            control_enabled=control_enabled,
            normalization=config["normalization"],
        )

    # -- Ticket 02 evaluator seam ------------------------------------------- #

    def relative_condition(self, stacked):
        """Public condition seam: normalized latest-frame relative condition
        (shared training/inference implementation)."""
        return latest_relative_condition(stacked, self.norm_cfg)

    def pending_hint(self):
        """The relative-condition hint of the most recent ``base_cond``
        capture, repeated over the prediction horizon (test seam;
        ``get_action`` consumes exactly this)."""
        if self._latest_stacked is None:
            raise RuntimeError(
                "pending_hint requires a prior base_cond call to capture the "
                "full stacked observation"
            )
        return build_hint(
            latest_relative_condition(self._latest_stacked, self.norm_cfg),
            pred_horizon=self.pred_horizon,
        ).to(next(self.denoiser.parameters()).device)

    def base_cond(self, queued_stacked):
        """Trim the 2x51 stacked obs to the base policy condition and capture
        the full stacked history for the relative hint."""
        stacked = torch.as_tensor(queued_stacked)
        if stacked.shape[-2] != self.obs_horizon or stacked.shape[-1] != self.base_obs_dim + 3:
            raise ValueError(
                f"Expected the queued ({self.obs_horizon}, "
                f"{self.base_obs_dim + 3}) stacked obs, got {tuple(stacked.shape)}"
            )
        self._latest_stacked = stacked.detach().clone().to(next(self.denoiser.parameters()).device)
        trimmed = stacked[..., : self.base_obs_dim]
        if trimmed.dim() == 2:
            trimmed = trimmed[None]
        return trimmed.flatten(start_dim=1)

    def get_action(self, obs_cond):
        """Official DP rollout with the relative-condition hint.

        Must follow a ``base_cond`` call (the hint's source). Returns the
        ``(B, act_horizon, 4)`` executed slice. The hint is derived from the
        most recent ``base_cond`` capture; the evaluator's per-chunk
        ``base_cond -> get_action`` pattern always refreshes it.
        """
        if self._latest_stacked is None:
            raise RuntimeError(
                "get_action requires a prior base_cond call to capture the "
                "full stacked observation for the relative hint"
            )
        obs_cond = torch.as_tensor(obs_cond)
        B = obs_cond.shape[0]
        device = next(self.denoiser.parameters()).device
        hint = self.pending_hint()
        conditional = obs_cond.to(device)
        sample = torch.randn((B, self.pred_horizon, self.act_dim), device=device)
        with torch.no_grad():
            for k in self.noise_scheduler.timesteps:
                sample = self.noise_scheduler.step(
                    model_output=self.denoiser(
                        sample, k, global_cond=conditional, hint=hint,
                        enabled=self.control_enabled,
                    ),
                    timestep=k,
                    sample=sample,
                ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return sample[:, start:end]

    def map_action_ee_delta_pose(self, base_seq):
        """4d base action -> 7d pd_ee_delta_pose action (three zero rotations)."""
        return map_action_4d_to_7d(base_seq)

    def select_executed(self, pred_seq):
        """Executed-horizon slicing kept for base-adapter seam parity."""
        pred_seq = torch.as_tensor(pred_seq)
        start = self.obs_horizon - 1
        return pred_seq[:, start:start + self.act_horizon]
