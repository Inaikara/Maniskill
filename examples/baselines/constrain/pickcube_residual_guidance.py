"""Exploratory epsilon-space residual guidance for the frozen PickCube model."""

import math

import torch


GUIDANCE_KIND = "learned_residual_guidance"
REGISTERED_SCALE = 1.5
MAX_SCALE = 2.0


def validate_guidance_scale(scale):
    value = float(scale)
    if not math.isfinite(value) or not 0.0 <= value <= MAX_SCALE:
        raise ValueError(f"guidance scale must be finite and within [0,{MAX_SCALE}]")
    return value


def combine_epsilon(eps_base, eps_control, scale):
    """Extrapolate the learned Control-minus-Base epsilon residual."""
    scale = validate_guidance_scale(scale)
    if scale == 0.0:
        return eps_base
    if scale == 1.0:
        return eps_control
    return eps_base + scale * (eps_control - eps_base)


class ResidualGuidanceAdapter:
    """Wrap a position-unit Control adapter without changing its checkpoint."""

    def __init__(self, adapter, scale=REGISTERED_SCALE):
        self.adapter = adapter
        self.scale = validate_guidance_scale(scale)
        self.device = adapter.device
        self.scheduler = adapter.scheduler
        self.metadata = dict(adapter.metadata)
        self.metadata.update({"guidance_kind": GUIDANCE_KIND,
                              "guidance_scale": self.scale,
                              "guidance_semantics": "epsilon_base + scale * (epsilon_control - epsilon_base)"})

    def get_action(self, raw_queued):
        raw = torch.as_tensor(raw_queued, dtype=torch.float32, device=self.device)
        if raw.ndim == 2:
            raw = raw[None]
        if raw.shape[-2:] != (2, 45):
            raise ValueError("expected raw native (batch,2,45) observations")
        from examples.baselines.constrain.pickcube_zscore_round37 import prepare_control_observation

        obs42, hint_condition = prepare_control_observation(raw, self.adapter.stats)
        condition = obs42.flatten(1)
        hint = hint_condition[:, None, :].expand(-1, 16, -1)
        sample = torch.randn((raw.shape[0], 16, 7), device=self.device)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                eps_control = self.adapter.denoiser(
                    sample, timestep, global_cond=condition, hint=hint, enabled=True
                )
                if self.scale == 1.0:
                    # Preserve the prior default trajectory bit-for-bit.
                    epsilon = eps_control
                else:
                    eps_base = self.adapter.denoiser(
                        sample, timestep, global_cond=condition, hint=hint, enabled=False
                    )
                    epsilon = combine_epsilon(eps_base, eps_control, self.scale)
                sample = self.scheduler.step(epsilon, timestep, sample).prev_sample
        return sample[:, 1:9]
