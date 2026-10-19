"""Tests for the constrain training/checkpoint seam (Ticket 03).

Seams under test:
- ``compute_epsilon_loss`` keeps the official epsilon denoising objective: the
  loss is the MSE between the denoiser output and the exact noise sample, at
  the same relative-condition hint; at initialization (zero residual) it
  equals the frozen base DP loss on the identical batch,
- ``save_constrain_checkpoint`` stores the complete standalone state — model,
  EMA, optimizer, lr scheduler, training step, RNG streams, model config,
  normalization config and base weight source — and the reload builds the
  model strictly from the saved config and tensors (no dependency on the
  original base checkpoint file),
- strict checkpoint loading rejects mismatched tensors,
- resuming preserves the frozen base bit-exactly (verified against the
  original base checkpoint ``ema_agent`` weights) and continues the step
  count, optimizer state and EMA.

Tests run the real dataset and the real base checkpoint on CPU.
"""

import hashlib
import sys
from pathlib import Path

import pytest
import torch

from examples.baselines.constrain.base_dp import DEFAULT_CKPT
from examples.baselines.constrain.constrain_ckpt import (
    CHECKPOINT_FORMAT,
    frozen_base_sha16,
    load_constrain_denoiser,
    save_constrain_checkpoint,
)
from examples.baselines.constrain.constrain_condition import ConditionNormConfig
from examples.baselines.constrain.constrain_denoiser import (
    ConstrainDenoiser,
    build_constrain_denoiser,
    trainable_parameters,
)
from examples.baselines.constrain.constrain_dataset import ConstrainDemoDataset
from examples.baselines.constrain.train_constrain import (
    ConstrainTrainer,
    compute_epsilon_loss,
)

OFFICIAL_DP_DIR = Path(__file__).resolve().parents[3] / "examples" / "baselines" / "diffusion_policy"
if str(OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DP_DIR))

REAL_DEMO = "outputs/stackcube_obstacle/trajectory.h5"


@pytest.fixture(scope="module")
def base_net():
    agent_sd = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)["ema_agent"]
    net_sd = {k[len("noise_pred_net."):]: v for k, v in agent_sd.items()}
    agent = build_constrain_denoiser().base_net  # official net at trained channels
    agent.load_state_dict(net_sd)
    agent.eval()
    return agent


def loss_inputs(seed=5, batch=3):
    ds = ConstrainDemoDataset(REAL_DEMO)
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(batch)
    obs = torch.stack([ds[int(i)]["observations"] for i in idx])
    act = torch.stack([ds[int(i)]["actions"] for i in idx])
    noise = torch.randn(batch, 16, 4, generator=g)
    timesteps = torch.randint(0, 100, (batch,), generator=g).long()
    return obs, act, noise, timesteps


def test_epsilon_loss_matches_manual_mse(base_net):
    """The loss equals the MSE between the denoiser epsilon output and the
    exact noise passed in (independent manual computation)."""
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    denoiser.eval()
    obs, act, noise, timesteps = loss_inputs()
    cfg = ConditionNormConfig()
    scheduler = ConstrainTrainer.build_scheduler(num_diffusion_iters=100)
    loss = compute_epsilon_loss(denoiser, cfg, obs, act, noise, timesteps, scheduler)
    # manual recomputation through the same public pieces
    from examples.baselines.constrain.constrain_condition import build_hint, latest_relative_condition
    obs_cond = obs[:, :, :48].flatten(1)
    hint = build_hint(latest_relative_condition(obs, cfg), pred_horizon=16)
    noisy = scheduler.add_noise(act, noise, timesteps)
    pred = denoiser(noisy, timesteps, global_cond=obs_cond, hint=hint)
    manual = torch.nn.functional.mse_loss(pred, noise)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, manual)


def test_epsilon_loss_at_init_equals_base_loss(base_net):
    """At zero-residual initialization the constrain loss equals the frozen
    base DP loss on the identical batch (base gets no obstacle condition)."""
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    denoiser.eval()
    obs, act, noise, timesteps = loss_inputs()
    scheduler = ConstrainTrainer.build_scheduler(num_diffusion_iters=100)
    constrain_loss = compute_epsilon_loss(
        denoiser, ConditionNormConfig(), obs, act, noise, timesteps, scheduler
    )
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    noisy = scheduler.add_noise(act, noise, timesteps)
    base_pred = base_net(noisy, timesteps, global_cond=obs[:, :, :48].flatten(1))
    base_loss = torch.nn.functional.mse_loss(base_pred, noise)
    torch.testing.assert_close(constrain_loss, base_loss)


def test_checkpoint_roundtrip_stores_complete_state(tmp_path, base_net):
    """The checkpoint carries model, EMA, optimizer, lr scheduler, step, RNG,
    model config, normalization config and base weight source."""
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    params = trainable_parameters(denoiser)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    from diffusers.training_utils import EMAModel
    ema = EMAModel(parameters=params, power=0.75)
    obs, act, noise, timesteps = loss_inputs()
    scheduler = ConstrainTrainer.build_scheduler(num_diffusion_iters=100)
    loss = compute_epsilon_loss(
        denoiser, ConditionNormConfig(), obs, act, noise, timesteps, scheduler
    )
    loss.backward()
    optimizer.step()
    ema.step(params)
    step = 7
    path = tmp_path / "ck.pt"
    save_constrain_checkpoint(
        path,
        denoiser=denoiser,
        ema=ema,
        optimizer=optimizer,
        lr_scheduler=None,
        step=step,
        rng={"torch": torch.get_rng_state()},
        config={
            "model": dict(input_dim=4, global_cond_dim=96, diffusion_step_embed_dim=64,
                          down_dims=[64, 128, 256], n_groups=8, cond_dim=3,
                          num_diffusion_iters=100),
            "policy": dict(obs_horizon=2, act_horizon=8, pred_horizon=16,
                           base_obs_dim=48, act_dim=4),
            "normalization": ConditionNormConfig().to_dict(),
            "base_weights": {
                "source": str(DEFAULT_CKPT),
                "subkey": "ema_agent",
                "state_sha16": frozen_base_sha16(denoiser),
            },
        },
    )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["format"] == CHECKPOINT_FORMAT
    for key in ("model", "ema", "optimizer", "step", "rng", "config"):
        assert key in ck, key
    assert ck["step"] == 7
    assert ck["config"]["base_weights"]["subkey"] == "ema_agent"
    assert "normalization" in ck["config"] and "model" in ck["config"]
    assert "rel_xy_range" in ck["config"]["normalization"]
    assert ck["config"]["policy"]["pred_horizon"] == 16


def test_reload_strict_and_weight_fidelity(tmp_path, base_net):
    """Reload builds the model from the saved config only (no base checkpoint
    dependency), loads strictly, and restores both raw and EMA weights
    bit-exactly."""
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    params = trainable_parameters(denoiser)
    # make the raw weights distinctive
    with torch.no_grad():
        params[0].add_(0.125)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    from diffusers.training_utils import EMAModel
    ema = EMAModel(parameters=params, power=0.75)
    with torch.no_grad():
        for p in params:
            p.add_(1.0)
    # diffusers EMA decay starts at zero over the first steps, so the EMA
    # copy is made distinctive directly in its shadow state
    ema.step(params)
    with torch.no_grad():
        ema.shadow_params[0].sub_(0.5)
    path = tmp_path / "ck.pt"
    save_constrain_checkpoint(
        path, denoiser=denoiser, ema=ema, optimizer=optimizer, lr_scheduler=None,
        step=3, rng={"torch": torch.get_rng_state()},
        config={
            "model": dict(input_dim=4, global_cond_dim=96, diffusion_step_embed_dim=64,
                          down_dims=[64, 128, 256], n_groups=8, cond_dim=3,
                          num_diffusion_iters=100),
            "normalization": ConditionNormConfig().to_dict(),
            "base_weights": {"source": str(DEFAULT_CKPT), "subkey": "ema_agent",
                             "state_sha16": frozen_base_sha16(denoiser)},
        },
    )
    raw_model, _ = load_constrain_denoiser(path, device="cpu", weights="model")
    torch.testing.assert_close(
        raw_model.trainable_parameters()[0], denoiser.trainable_parameters()[0]
    )
    ema_model, loaded_ck = load_constrain_denoiser(path, device="cpu", weights="ema")
    assert loaded_ck["step"] == 3
    # EMA shadow restored into the trainable parameters (distinct from raw)
    raw0 = raw_model.trainable_parameters()[0]
    ema0 = ema_model.trainable_parameters()[0]
    assert not torch.equal(ema0, raw0)
    torch.testing.assert_close(ema0, ema.shadow_params[0])
    # the frozen base is identical to the fresh base in both reloads
    torch.testing.assert_close(ema_model.base_net.state_dict(), base_net.state_dict())
    # raw reload reproduces outputs exactly at the same inputs (same weights)
    g = torch.Generator().manual_seed(2)
    sample = torch.randn(1, 16, 4, generator=g)
    cond = torch.randn(1, 96, generator=g)
    hint = torch.randn(1, 16, 3, generator=g)
    with torch.no_grad():
        torch.testing.assert_close(
            raw_model(sample, 5, global_cond=cond, hint=hint),
            denoiser(sample, 5, global_cond=cond, hint=hint),
        )


def test_reload_rejects_corrupted_tensor(tmp_path, base_net):
    """Strict loading: a mismatched control tensor fails the reload."""
    denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3)
    path = tmp_path / "ck.pt"
    save_constrain_checkpoint(
        path, denoiser=denoiser, ema=None, optimizer=None, lr_scheduler=None,
        step=0, rng={}, config={
            "model": dict(input_dim=4, global_cond_dim=96, diffusion_step_embed_dim=64,
                          down_dims=[64, 128, 256], n_groups=8, cond_dim=3,
                          num_diffusion_iters=100),
            "normalization": ConditionNormConfig().to_dict(),
            "base_weights": {"source": str(DEFAULT_CKPT), "subkey": "ema_agent",
                             "state_sha16": frozen_base_sha16(denoiser)},
        },
    )
    ck = torch.load(path, map_location="cpu", weights_only=False)
    key = next(k for k in ck["model"] if k.startswith("control_down_modules"))
    ck["model"][key] = torch.zeros((3,) + tuple(ck["model"][key].shape[1:]))
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    with pytest.raises(RuntimeError):
        load_constrain_denoiser(bad, device="cpu", weights="model")


def test_frozen_base_sha_detects_tampering():
    """The sha is deterministic and differs when any frozen tensor moves
    (own network: the shared base fixture must stay pristine for others)."""
    denoiser = build_constrain_denoiser()
    sha = frozen_base_sha16(denoiser)
    assert sha == frozen_base_sha16(denoiser)  # deterministic
    with torch.no_grad():
        next(denoiser.base_net.parameters()).add_(0.01)
    assert sha != frozen_base_sha16(denoiser)


@pytest.fixture(scope="module")
def trained_ckpt(tmp_path_factory, base_net):
    run_dir = tmp_path_factory.mktemp("train_run")
    trainer = ConstrainTrainer(
        demo_path=REAL_DEMO,
        base_checkpoint=DEFAULT_CKPT,
        device="cpu",
        batch_size=4,
        total_iters=2,
        run_dir=run_dir,
        seed=1,
    )
    trainer.setup()
    losses = trainer.train_steps(2)
    path = trainer.save_checkpoint(run_dir / "ck.pt")
    return trainer, path, losses, run_dir, base_net


def test_resume_preserves_frozen_base_and_state(trained_ckpt):
    trainer, path, losses, run_dir, base_net = trained_ckpt
    assert len(losses) == 2 and all(torch.isfinite(l) for l in losses)
    # trainer's own frozen base matches the original base checkpoint exactly
    base_state = trainer.denoiser.base_net.state_dict()
    for k, v in base_state.items():
        torch.testing.assert_close(v, base_net.state_dict()[k])
    # resume into a fresh trainer
    resumed = ConstrainTrainer(
        demo_path=REAL_DEMO, base_checkpoint=DEFAULT_CKPT, device="cpu",
        batch_size=4, total_iters=3, run_dir=run_dir, seed=1, resume=path,
    )
    resumed.setup()
    assert resumed.step == 2
    # frozen base still bit-identical to the original base checkpoint
    for k, v in resumed.denoiser.base_net.state_dict().items():
        torch.testing.assert_close(v, base_net.state_dict()[k])
    # optimizer state carried over (nonempty momentum buffers)
    opt_state = resumed.optimizer.state_dict()["state"]
    assert opt_state and any("exp_avg" in s for s in opt_state.values())
    continued = resumed.train_steps(1)
    assert resumed.step == 3
    assert torch.isfinite(continued[0])


def test_resume_rejects_base_drift(tmp_path, base_net):
    """A checkpoint whose recorded base sha does not match the current frozen
    base fails resume instead of silently training on drifted weights."""
    trainer = ConstrainTrainer(
        demo_path=REAL_DEMO, base_checkpoint=DEFAULT_CKPT, device="cpu",
        batch_size=4, total_iters=1, run_dir=tmp_path, seed=1,
    )
    trainer.setup()
    # tamper the checkpoint's recorded sha
    path = trainer.save_checkpoint(tmp_path / "ck.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["config"]["base_weights"]["state_sha16"] = "deadbeefdeadbeef"
    torch.save(ck, path)
    with pytest.raises(ValueError, match="base"):
        ConstrainTrainer(
            demo_path=REAL_DEMO, base_checkpoint=DEFAULT_CKPT, device="cpu",
            batch_size=4, total_iters=2, run_dir=tmp_path, seed=1, resume=path,
        ).setup()


def test_resume_rejects_corrupted_model(tmp_path, base_net):
    trainer = ConstrainTrainer(
        demo_path=REAL_DEMO, base_checkpoint=DEFAULT_CKPT, device="cpu",
        batch_size=4, total_iters=1, run_dir=tmp_path, seed=1,
    )
    trainer.setup()
    path = trainer.save_checkpoint(tmp_path / "ck.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    key = next(k for k in ck["model"] if k.startswith("hint_encoder"))
    ck["model"][key] = torch.zeros((3,) + tuple(ck["model"][key].shape[1:]))
    torch.save(ck, path)
    with pytest.raises(RuntimeError):
        ConstrainTrainer(
            demo_path=REAL_DEMO, base_checkpoint=DEFAULT_CKPT, device="cpu",
            batch_size=4, total_iters=2, run_dir=tmp_path, seed=1, resume=path,
        ).setup()
