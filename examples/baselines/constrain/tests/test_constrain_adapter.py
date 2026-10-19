"""Tests for the ConstrainDPAdapter rollout seam (Ticket 03).

Seams under test:
- the adapter plugs into the existing Ticket 02 evaluator call pattern
  ``adapter.base_cond(stacked) -> adapter.get_action(cond) ->
  adapter.map_action_ee_delta_pose(executed)`` with the unchanged base
  condition trim (2x51 -> 2x48 flattened) and the unchanged 4d->7d action
  mapping,
- the relative-condition hint is derived from the most recent frame of the
  full 2x51 stacked observation (captured at ``base_cond``, never lost by the
  premature 48d trim): with an active control branch two histories sharing
  the same trimmed base condition but differing obstacle/TCP features produce
  different actions,
- with the control branch disabled the adapter reproduces the base DP
  adapter's rollout exactly under the same torch noise stream (official
  epsilon/DDPM loop, executed-horizon slice),
- the adapter loads a standalone checkpoint strictly (no base checkpoint
  dependency), reusing the recorded normalization and policy horizons.
"""

import sys
from pathlib import Path

import pytest
import torch

import examples.baselines.constrain as _pkg  # noqa: F401

from examples.baselines.constrain.base_dp import DEFAULT_CKPT, BaseDPAdapter
from examples.baselines.constrain.constrain_adapter import ConstrainDPAdapter
from examples.baselines.constrain.constrain_condition import ConditionNormConfig
from examples.baselines.constrain.constrain_ckpt import (
    CHECKPOINT_FORMAT,
    frozen_base_sha16,
    save_constrain_checkpoint,
)
from examples.baselines.constrain.constrain_denoiser import (
    ConstrainDenoiser,
    build_constrain_denoiser,
    trainable_parameters,
)

OFFICIAL_DP_DIR = Path(__file__).resolve().parents[3] / "examples" / "baselines" / "diffusion_policy"
if str(OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DP_DIR))


def base_frame(tcp_x, tcp_y, obstacle_x, obstacle_y, radius):
    """A synthetic 51d frame using the proven index layout."""
    f = torch.zeros(51)
    f[18], f[19] = tcp_x, tcp_y
    f[48], f[49], f[50] = obstacle_x, obstacle_y, radius
    return f


@pytest.fixture(scope="module")
def base_net():
    agent = build_constrain_denoiser().base_net
    sd = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)["ema_agent"]
    agent.load_state_dict({k[len("noise_pred_net."):]: v for k, v in sd.items()})
    agent.eval()
    return agent


@pytest.fixture(scope="module")
def active_ckpt(tmp_path_factory, base_net):
    """A standalone checkpoint whose control branch is active (zero residual
    moved off zero) so the hint provably enters the frozen decoder."""
    run_dir = tmp_path_factory.mktemp("adapter_run")
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    with torch.no_grad():
        # a large "trained" control residual so the hint's effect survives
        # the 100-step clipped DDPM loop
        denoiser.zero_mid_conv.weight.add_(5.0)
        denoiser.zero_mid_conv.bias.add_(0.1)
    params = trainable_parameters(denoiser)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    from diffusers.training_utils import EMAModel

    ema = EMAModel(parameters=params, power=0.75)
    path = save_constrain_checkpoint(
        run_dir / "active.pt",
        denoiser=denoiser,
        ema=ema,
        optimizer=optimizer,
        lr_scheduler=None,
        step=12,
        rng={"torch": torch.get_rng_state()},
        config={
            "model": dict(
                input_dim=4, global_cond_dim=96, diffusion_step_embed_dim=64,
                down_dims=[64, 128, 256], n_groups=8, cond_dim=3,
                num_diffusion_iters=100,
            ),
            "policy": dict(obs_horizon=2, act_horizon=8, pred_horizon=16,
                           base_obs_dim=48, act_dim=4),
            "normalization": ConditionNormConfig().to_dict(),
            "base_weights": {
                "source": str(DEFAULT_CKPT), "subkey": "ema_agent",
                "state_sha16": frozen_base_sha16(denoiser),
            },
        },
    )
    return path


def stacked_from_frames(old, new):
    return torch.stack([old, new])[None]  # (1, 2, 51)


def test_adapter_evaluator_seam_shapes(active_ckpt):
    """The evaluator call pattern works against this adapter: same trimmed
    base condition and 7d action mapping shapes as the base adapter."""
    adapter = ConstrainDPAdapter(active_ckpt, device="cpu")
    old = base_frame(0.0, 0.0, 0.10, 0.20, 0.03)
    new = base_frame(0.01, 0.0, 0.11, 0.20, 0.03)
    stacked = stacked_from_frames(old, new)
    cond = adapter.base_cond(stacked)
    assert cond.shape == (1, 96)
    base = BaseDPAdapter(device="cpu")
    torch.testing.assert_close(cond, base.base_cond(stacked))
    torch.manual_seed(0)
    executed = adapter.get_action(cond)
    assert executed.shape == (1, 8, 4)
    env_actions = adapter.map_action_ee_delta_pose(executed)
    assert env_actions.shape == (1, 8, 7)
    torch.testing.assert_close(env_actions[:, :, 3:6], torch.zeros(1, 8, 3))
    torch.testing.assert_close(env_actions[:, :, 6], executed[:, :, 3])
    torch.testing.assert_close(env_actions[:, :, :3], executed[:, :, :3])


def test_hint_uses_latest_frame_not_lost_by_trim(active_ckpt):
    """The captured hint is read from the most recent frame of the full 2x51
    stacked observation, exactly matching the trimmed base condition for the
    same frames — the appended obstacle features are never lost by the
    premature 48d trim (content-level, at the pending_hint seam)."""
    adapter = ConstrainDPAdapter(active_ckpt, device="cpu")
    shared_first = base_frame(0.5, -0.5, 9.9, 9.9, 0.999)  # stale first frame
    hist_a = stacked_from_frames(
        shared_first,
        base_frame(0.00, 0.00, 0.10, 0.20, 0.03),  # latest frame A
    )
    other_first = shared_first.clone()
    other_first[50] = 0.004  # different stale first-frame obstacle feature
    hist_a2 = stacked_from_frames(
        other_first, base_frame(0.00, 0.00, 0.10, 0.20, 0.03),
    )
    hist_b = stacked_from_frames(
        shared_first,
        base_frame(0.00, 0.00, 0.14, 0.20, 0.03),  # latest frame B: obstacle x
    )
    cond_a = adapter.base_cond(hist_a)
    # the hint is captured BEFORE hist_b, so read it right after its capture
    hint_a = adapter.pending_hint()
    cond_b = adapter.base_cond(hist_b)
    hint_b = adapter.pending_hint()
    # both trims are identical: the base condition cannot distinguish them
    torch.testing.assert_close(cond_a, cond_b)
    assert hint_a.shape == (1, 16, 3) and hint_b.shape == (1, 16, 3)
    torch.testing.assert_close(hint_a[0, 0], torch.tensor([0.10, 0.20, -0.25]))
    torch.testing.assert_close(hint_b[0, 0], torch.tensor([0.14, 0.20, -0.25]))
    assert not torch.allclose(hint_a, hint_b)
    # the stale first frame maps nowhere into the hint (latest-frame rule)
    cond_a2 = adapter.base_cond(hist_a2)  # subsequent captures refresh the hint
    torch.testing.assert_close(cond_a2, cond_a)
    # the stale first frame maps nowhere into the hint (latest-frame rule)
    torch.testing.assert_close(adapter.pending_hint(), hint_a)


def test_enabled_control_branch_actually_drives_rollout(active_ckpt):
    """With the control branch live the adapter's rollout diverges from the
    disabled rollout under the same noise stream and capture: the captured
    hint provably flows through the DDPM loop into the executed actions."""
    enabled = ConstrainDPAdapter(active_ckpt, device="cpu", control_enabled=True)
    disabled = ConstrainDPAdapter(active_ckpt, device="cpu", control_enabled=False)
    stacked = stacked_from_frames(base_frame(0.0, 0.0, 0.10, 0.20, 0.03),
                                  base_frame(0.00, 0.00, 0.11, 0.20, 0.03))
    torch.manual_seed(3)
    out_on = enabled.get_action(enabled.base_cond(stacked))
    torch.manual_seed(3)
    out_off = disabled.get_action(disabled.base_cond(stacked))
    assert not torch.allclose(out_on, out_off)


def test_same_history_reproduces_actions(active_ckpt):
    adapter = ConstrainDPAdapter(active_ckpt, device="cpu")
    stacked = stacked_from_frames(base_frame(0.0, 0.0, 0.10, 0.20, 0.03),
                                  base_frame(0.0, 0.0, 0.11, 0.20, 0.03))
    cond = adapter.base_cond(stacked)
    torch.manual_seed(3)
    first = adapter.get_action(cond)
    torch.manual_seed(3)
    second = adapter.get_action(adapter.base_cond(stacked))
    torch.testing.assert_close(first, second)


def test_disabled_control_equals_base_adapter(active_ckpt):
    """With the control branch disabled the constrained adapter equals the
    base DP adapter under the same torch noise stream and obs (the whole
    zero-residual guarantee at the rollout seam)."""
    constrain = ConstrainDPAdapter(active_ckpt, device="cpu", control_enabled=False)
    base = BaseDPAdapter(device="cpu")
    stacked = stacked_from_frames(base_frame(0.0, 0.0, 0.10, 0.20, 0.03),
                                  base_frame(0.00, 0.00, 0.11, 0.20, 0.03))
    torch.manual_seed(3)
    via_constrain = constrain.get_action(constrain.base_cond(stacked))
    torch.manual_seed(3)
    via_base = base.get_action(base.base_cond(stacked))
    torch.testing.assert_close(via_constrain, via_base)


def test_adapter_records_config_and_metadata(active_ckpt):
    adapter = ConstrainDPAdapter(active_ckpt, device="cpu")
    meta = adapter.metadata
    assert meta["step"] == 12
    assert meta["weights"] == "ema"
    assert meta["num_diffusion_iters"] == 100
    assert meta["act_dim"] == 4 and meta["base_obs_dim"] == 48
    assert meta["obs_horizon"] == 2 and meta["act_horizon"] == 8 and meta["pred_horizon"] == 16
    assert adapter.norm_cfg == ConditionNormConfig()
    # the adapter is independent of the original base checkpoint file: the
    # reload reads only the standalone artifact (frozen base stored inside)
    ck = torch.load(active_ckpt, map_location="cpu", weights_only=False)
    assert ck["format"] == CHECKPOINT_FORMAT
    assert Path(ck["config"]["base_weights"]["source"]).exists()


def test_get_action_without_base_cond_rejected(active_ckpt):
    """get_action requires a prior base_cond (the hint source is the latest
    stacked frame; absent capture is an error, never a stale condition)."""
    adapter = ConstrainDPAdapter(active_ckpt, device="cpu")
    with pytest.raises(RuntimeError):
        adapter.get_action(torch.zeros(1, 96))


def test_relative_condition_semantics_at_adapter(active_ckpt):
    """Obstacle-position-only changes shift the hint; TCP-orientation-only
    changes leave it untouched (world-parallel axes) at the adapter's own
    capture seam."""
    adapter = ConstrainDPAdapter(active_ckpt, device="cpu")
    a = stacked_from_frames(base_frame(0.0, 0.0, 0.20, 0.10, 0.03),
                            base_frame(0.00, 0.00, 0.20, 0.10, 0.03))
    b = stacked_from_frames(base_frame(0.0, 0.0, 0.26, 0.10, 0.03),
                            base_frame(0.00, 0.00, 0.26, 0.10, 0.03))
    adapter.base_cond(a)
    hint_a = adapter.pending_hint()
    adapter.base_cond(b)
    hint_b = adapter.pending_hint()
    torch.testing.assert_close(hint_b[0, 0] - hint_a[0, 0], torch.tensor([0.06, 0.0, 0.0]))
    # rotation-only change leaves the hint unchanged (world-parallel axes)
    a_q = torch.clone(a)
    a_q[0, -1, 21:25] = torch.tensor([0.0, 0.0, 0.7071, 0.7071])
    adapter.base_cond(a_q)
    torch.testing.assert_close(adapter.pending_hint(), hint_a)
