"""Tests for the ConstrainNet-style denoiser seam (Ticket 03).

Seams under test (public interface of ``ConstrainDenoiser``):
- the base official ``ConditionalUnet1D`` (loaded from the trained base
  checkpoint through ``BaseDPAdapter``-equivalent channels) is fully frozen:
  every base parameter has ``requires_grad=False`` and keeps its loaded value;
- the control path is an exact weight copy of the base ``down_modules`` and
  ``mid_modules`` and is trainable, together with the hint encoder and the
  zero-initialized multiscale residual convolutions;
- at initialization the zero residual and an explicitly disabled control
  branch reproduce the base DP output exactly at the same sample, diffusion
  timestep and global condition,
- after one optimizer step the zero convs have moved and a second backward
  pass produces nonzero gradients reaching the upstream control path and hint
  encoder (gradient path through the updated zero conv),
- the model is rebuildable from a config dict (strict reload seam used by
  checkpoint save/load and independent-process inference).

Uses the real trained base checkpoint (same weight source as Ticket 01).
"""

import sys
from pathlib import Path

import pytest
import torch

from examples.baselines.constrain.base_dp import DEFAULT_CKPT, build_official_agent
from examples.baselines.constrain.constrain_denoiser import (
    MODEL_CONFIG,
    ConstrainDenoiser,
    build_constrain_denoiser,
    trainable_parameters,
)

OFFICIAL_DP_DIR = Path(__file__).resolve().parents[3] / "examples" / "baselines" / "diffusion_policy"
if str(OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DP_DIR))


def load_base_net():
    """The official ConditionalUnet1D with the exact trained base weights."""
    agent = build_official_agent(device="cpu")
    ck = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)
    agent.load_state_dict(ck["ema_agent"])
    agent.eval()
    return agent.noise_pred_net


@pytest.fixture(scope="module")
def base_net():
    return load_base_net()


@pytest.fixture(scope="module")
def denoiser(base_net):
    d = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    d.eval()
    return d


def denoiser_inputs(seed=0):
    """Fixed (sample, timestep, global_cond, hint) batch for equivalence checks."""
    g = torch.Generator().manual_seed(seed)
    sample = torch.randn(2, 16, 4, generator=g)
    global_cond = torch.randn(2, 96, generator=g)
    hint = torch.randn(2, 16, 3, generator=g).clamp(-1, 1)
    return sample, 37, global_cond, hint


def test_model_config_matches_trained_channels():
    """The model config pins the exact trained channels of the base run."""
    assert MODEL_CONFIG == dict(
        input_dim=4,
        global_cond_dim=96,
        diffusion_step_embed_dim=64,
        down_dims=[64, 128, 256],
        n_groups=8,
        cond_dim=3,
        num_diffusion_iters=100,
    )
    rebuilt = build_constrain_denoiser(MODEL_CONFIG)
    assert isinstance(rebuilt, ConstrainDenoiser)


def test_base_frozen_and_control_trainable(denoiser):
    """Every base parameter is frozen; the control path, hint encoder and
    zero convs are the only trainable parameters."""
    for name, p in denoiser.base_net.named_parameters():
        assert not p.requires_grad, name
    trainable = {n: p for n, p in denoiser.named_parameters() if p.requires_grad}
    assert trainable, "no trainable parameters"
    assert all(p.requires_grad for p in trainable.values())
    assert all(not k.startswith("base_net.") for k in trainable)
    for section in ("control_down_modules", "control_mid_modules", "hint_encoder", "zero_skip_convs", "zero_mid_conv"):
        assert any(k.startswith(section + ".") for k in trainable), section


def test_control_path_is_exact_weight_copy(denoiser):
    """The trainable control down/mid modules hold the exact copied base
    weights at initialization."""
    for ctrl, base in zip(denoiser.control_down_modules, denoiser.base_net.down_modules):
        for c, b in zip(ctrl, base):
            for (cn, cp), (bn, bp) in zip(c.named_parameters(), b.named_parameters()):
                assert cn == bn
                torch.testing.assert_close(cp, bp)
    for ctrl, base in zip(denoiser.control_mid_modules, denoiser.base_net.mid_modules):
        for (cn, cp), (bn, bp) in zip(ctrl.named_parameters(), base.named_parameters()):
            assert cn == bn
            torch.testing.assert_close(cp, bp)


def test_zero_residual_output_equals_base(denoiser, base_net):
    """At initialization (zero convs) the denoiser output with a hint equals
    the base DP output exactly at the same sample/timestep/global condition."""
    sample, timestep, global_cond, hint = denoiser_inputs()
    with torch.no_grad():
        out = denoiser(sample, timestep, global_cond=global_cond, hint=hint)
        base_out = base_net(sample, timestep, global_cond=global_cond)
    assert out.shape == base_out.shape == (2, 16, 4)
    torch.testing.assert_close(out, base_out)


def test_disabled_control_branch_equals_base(denoiser, base_net):
    """The explicitly disabled control branch reproduces the base DP output
    exactly, and agrees with the zero-residual output."""
    sample, timestep, global_cond, _ = denoiser_inputs()
    with torch.no_grad():
        disabled = denoiser(sample, timestep, global_cond=global_cond, enabled=False)
        base_out = base_net(sample, timestep, global_cond=global_cond)
    torch.testing.assert_close(disabled, base_out)


def test_zero_convs_are_zero_initialized(denoiser):
    """All multiscale skip residuals and the mid residual start at exact zero
    weights and biases."""
    for conv in denoiser.zero_skip_convs:
        assert torch.equal(conv.weight, torch.zeros_like(conv.weight))
        assert torch.equal(conv.bias, torch.zeros_like(conv.bias))
    assert torch.equal(denoiser.zero_mid_conv.weight, torch.zeros_like(denoiser.zero_mid_conv.weight))
    assert torch.equal(denoiser.zero_mid_conv.bias, torch.zeros_like(denoiser.zero_mid_conv.bias))
    # multiscale: residuals exactly on the skip connections the official
    # decoder consumes (the official up path pops down_modules[1:] only, so
    # level 0's skip is dead and must not carry a residual) plus the mid level
    assert denoiser.skip_residual_levels == [1, 2]
    assert len(denoiser.zero_skip_convs) == len(denoiser.base_net.down_modules) - 1


def test_enabled_requires_hint(denoiser):
    """An enabled forward without a hint is an error, not silent base output."""
    sample, timestep, global_cond, _ = denoiser_inputs()
    with pytest.raises(ValueError):
        denoiser(sample, timestep, global_cond=global_cond, hint=None, enabled=True)


def test_update_flows_through_zero_conv_to_upstream(denoiser):
    """After one optimizer step the zero convs have moved and a second
    backward pass delivers nonzero gradients to the control path, the hint
    encoder and the upstream zero convs; the frozen base keeps its exact
    loaded values throughout."""
    base_state = {k: v.clone() for k, v in denoiser.base_net.state_dict().items()}
    sample, timestep, global_cond, hint = denoiser_inputs()
    params = list(trainable_parameters(denoiser))
    opt = torch.optim.AdamW(params, lr=1e-3)

    # first update: only the zero convs receive gradient (they read the
    # nonzero control features); upstream control gets zero grad at first
    denoiser.train()
    pred = denoiser(sample, timestep, global_cond=global_cond, hint=hint)
    loss = (pred - sample).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    zero_conv_grads = {
        n: p.grad for n, p in denoiser.named_parameters()
        if n.startswith("zero_") and p.grad is not None
    }
    assert zero_conv_grads and any(torch.any(g != 0) for g in zero_conv_grads.values())
    opt.step()
    zero_conv_moved = any(
        not torch.equal(p, torch.zeros_like(p))
        for n, p in denoiser.named_parameters() if n.startswith("zero_") and n.endswith("weight")
    )
    assert zero_conv_moved

    # second update: gradient now reaches the upstream control path
    pred = denoiser(sample, timestep, global_cond=global_cond, hint=hint)
    loss = (pred - sample).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    ctrl_grads = {
        n: p.grad for n, p in denoiser.named_parameters()
        if n.startswith("control_down_modules.") and p.grad is not None
    }
    assert ctrl_grads and any(torch.any(g != 0) for g in ctrl_grads.values())
    hint_grads = {
        n: p.grad for n, p in denoiser.named_parameters()
        if n.startswith("hint_encoder.") and p.grad is not None
    }
    assert any(torch.any(g != 0) for g in hint_grads.values())

    # frozen base untouched by the updates
    for k, v in denoiser.base_net.state_dict().items():
        torch.testing.assert_close(v, base_state[k])


def test_nonzero_control_changes_output(base_net):
    """Sanity: once a zero conv carries weight, a changed hint changes the
    output — the condition actually enters the frozen decoder (fresh model,
    no reliance on other tests' updates)."""
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    with torch.no_grad():
        denoiser.zero_mid_conv.weight.add_(1.0)
    sample, timestep, global_cond, hint = denoiser_inputs()
    with torch.no_grad():
        out_a = denoiser(sample, timestep, global_cond=global_cond, hint=hint)
        out_b = denoiser(sample, timestep, global_cond=global_cond, hint=hint + 0.5)
    assert not torch.allclose(out_a, out_b)
