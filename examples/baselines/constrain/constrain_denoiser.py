"""ControlNet-style constrain denoiser over the official ManiSkill DP UNet.

The frozen base is the official ``ConditionalUnet1D`` trained by the ManiSkill
diffusion_policy baseline (Ticket 01 weight source). The trainable control
branch follows the conventional ControlNet layout (confirmed seam for this
ticket):

- a trainable **control copy** of the base ``down_modules`` and
  ``mid_modules``, initialized to the exact base weights,
- a small trainable **hint encoder** mapping the repeated 3d relative
  condition to the control input channels,
- **zero-initialized** 1x1 residual convolutions that add the control
  features to every frozen down-path skip connection and to the frozen mid
  output before the frozen decoder consumes them.

At initialization the residuals are zero and the denoiser output equals the
frozen base DP output exactly at the same sample, diffusion timestep and
global condition; an explicit ``enabled=False`` forward bypasses the control
branch entirely with the same guarantee. Because the residual convolutions
start at zero, the first optimizer step updates only those convolutions; from
the second update on, gradients reach the upstream control path and hint
encoder (the standard ControlNet training dynamic).

The base network is never modified or retrained: its parameters are frozen
and keep the loaded values bit-exact. ``MODEL_CONFIG`` pins the exact trained
channels of the base run and ``build_constrain_denoiser`` rebuilds the model
from a config dict, which is what the checkpoint save/load seam and the
independent-process inference reload use.
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_DP_DIR = _REPO_ROOT / "examples" / "baselines" / "diffusion_policy"

# Exact trained channels of the base run (outputs/runs/dp_StackCube-v1_980_1),
# also stored in every checkpoint config for strict reloads.
MODEL_CONFIG = dict(
    input_dim=4,
    global_cond_dim=96,
    diffusion_step_embed_dim=64,
    down_dims=[64, 128, 256],
    n_groups=8,
    cond_dim=3,
    num_diffusion_iters=100,
)


def _official_conditional_unet1d():
    if str(_OFFICIAL_DP_DIR) not in sys.path:
        sys.path.insert(0, str(_OFFICIAL_DP_DIR))
    from diffusion_policy.conditional_unet1d import ConditionalUnet1D

    return ConditionalUnet1D


def zero_conv1d(in_channels, out_channels):
    """Zero-initialized 1x1 residual convolution (weight and bias)."""
    conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv


def trainable_parameters(denoiser: "ConstrainDenoiser"):
    """Module-level seam: the denoiser's trainable parameters (never the base)."""
    return denoiser.trainable_parameters()


def build_constrain_denoiser(model_config=None, device="cpu"):
    """Build the denoiser from a config dict (checkpoint reload seam).

    The returned network holds a fresh official ``ConditionalUnet1D`` base
    built at the configured channels; loading a checkpoint state dict on top
    restores both the frozen base weights and the control branch without any
    dependency on the original base checkpoint file.
    """
    cfg = dict(MODEL_CONFIG if model_config is None else model_config)
    ConditionalUnet1D = _official_conditional_unet1d()
    base_net = ConditionalUnet1D(
        input_dim=cfg["input_dim"],
        global_cond_dim=cfg["global_cond_dim"],
        diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
        down_dims=list(cfg["down_dims"]),
        n_groups=cfg["n_groups"],
    )
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=cfg["cond_dim"])
    return denoiser.to(device)


class ConstrainDenoiser(nn.Module):
    """Frozen official DP UNet plus a trainable ControlNet control branch.

    ``forward(sample, timestep, global_cond, hint, enabled)`` mirrors the
    official ``ConditionalUnet1D.forward`` loop (same timestep handling, same
    FiLM global conditioning, same up-path skip consumption) with the control
    residuals injected; all frozen submodules are the official ones.
    """

    def __init__(self, base_net: nn.Module, cond_dim: int):
        super().__init__()
        self.base_net = base_net
        for p in self.base_net.parameters():
            p.requires_grad_(False)
        # Exact trainable copies of the frozen encoder stack. The hint
        # encoder ends at the UNet input channels so the copied down path can
        # consume it with the copied weights (no channel mismatch).
        self.control_down_modules = copy.deepcopy(base_net.down_modules)
        self.control_mid_modules = copy.deepcopy(base_net.mid_modules)
        for p in self.control_down_modules.parameters():
            p.requires_grad_(True)
        for p in self.control_mid_modules.parameters():
            p.requires_grad_(True)
        in_dim = base_net.down_modules[0][0].blocks[0].block[0].in_channels
        hidden = max(in_dim, 8)
        self.hint_encoder = nn.Sequential(
            nn.Conv1d(cond_dim, hidden, kernel_size=1),
            nn.Mish(),
            nn.Conv1d(hidden, in_dim, kernel_size=1),
            nn.Mish(),
        )
        # One zero residual per frozen-down-level skip connection that the
        # official decoder actually consumes (the official up path pops only
        # ``down_modules[1:]`` — ``in_out[1:]`` — so the first skip h[0] is
        # dead in the base UNet and must not carry a residual), plus the mid
        # level (added to the frozen mid output).
        down_dims = [base_net.down_modules[i][0].blocks[0].block[0].out_channels
                     for i in range(len(base_net.down_modules))]
        self.skip_residual_levels = list(range(1, len(down_dims)))
        self.zero_skip_convs = nn.ModuleList(
            zero_conv1d(down_dims[i], down_dims[i]) for i in self.skip_residual_levels
        )
        mid_dim = down_dims[-1]
        self.zero_mid_conv = zero_conv1d(mid_dim, mid_dim)
        self.cond_dim = cond_dim

    # -- parameters --------------------------------------------------------- #

    def trainable_parameters(self):
        """The control branch, hint encoder and zero convs (never the base)."""
        return [p for p in self.parameters() if p.requires_grad]

    # -- forward ------------------------------------------------------------ #

    def _global_feature(self, timesteps, sample, global_cond):
        base = self.base_net
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = base.diffusion_step_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)
        return global_feature

    def forward(self, sample, timestep, global_cond=None, hint=None, enabled=True):
        """Official DP forward with control residuals injected.

        sample: (B, T, act_dim); timestep: scalar or (B,); global_cond:
        (B, obs_horizon * 48) base condition; hint: (B, T, cond_dim) repeated
        relative condition (required when ``enabled``); ``enabled=False``
        bypasses the control branch entirely (pure base DP).
        """
        base = self.base_net
        sample = sample.moveaxis(-1, -2)  # (B, C, T)

        global_feature = self._global_feature(timestep, sample, global_cond)

        if enabled:
            if hint is None:
                raise ValueError("enabled control branch requires a hint tensor")
            ctrl = self.hint_encoder(hint.moveaxis(-1, -2))
        x = sample
        h = []
        residual_slot = 0
        for idx, (resnet, resnet2, downsample) in enumerate(base.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            if enabled and idx in self.skip_residual_levels:
                c = self.control_down_modules[idx][0](ctrl, global_feature)
                c = self.control_down_modules[idx][1](c, global_feature)
                h.append(x + self.zero_skip_convs[residual_slot](c))
                ctrl = self.control_down_modules[idx][2](c)
                residual_slot += 1
            else:
                h.append(x)
                if enabled:
                    c = self.control_down_modules[idx][0](ctrl, global_feature)
                    c = self.control_down_modules[idx][1](c, global_feature)
                    ctrl = self.control_down_modules[idx][2](c)
            x = downsample(x)

        for mid_module in base.mid_modules:
            x = mid_module(x, global_feature)
        if enabled:
            for control_mid in self.control_mid_modules:
                ctrl = control_mid(ctrl, global_feature)
            x = x + self.zero_mid_conv(ctrl)

        for idx, (resnet, resnet2, upsample) in enumerate(base.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = base.final_conv(x)
        return x.moveaxis(-1, -2)  # (B, T, act_dim)
