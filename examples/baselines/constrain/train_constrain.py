"""Training CLI for the obstacle-conditioned constrain policy (Ticket 03).

Trains the ControlNet control branch (control copy, hint encoder, zero
convolutions) on all 100 obstacle demonstrations while the frozen base DP is
never updated. The loss keeps the official ManiSkill DP epsilon objective:
sample per-window noise and diffusion timesteps, add noise to the 4d action
window and predict the noise, with the relative-condition hint built from the
window's latest observation frame by the shared condition seam.

Checkpoints carry the complete standalone state (model, EMA, optimizer, lr
scheduler, step, RNG, model/normalization config, base weight source) and can
be resumed with ``--resume``; resuming re-verifies the frozen-base checksum
(see ``constrain_ckpt.py``).

Defaults mirror the official training configuration (lr 1e-4, AdamW betas
(0.95, 0.999), weight decay 1e-6, cosine schedule with 500-step warmup,
batch 1024, EMA power 0.75, 30000 updates); short runs pass smaller
``--total-iters``. The full-budget run is deliberately deferred (Ticket 04).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

from examples.baselines.constrain.base_dp import (
    DEFAULT_CKPT,
    _OFFICIAL_DP_DIR,
    build_official_agent,
)
from examples.baselines.constrain.constrain_ckpt import (
    frozen_base_sha16,
    save_constrain_checkpoint,
    verify_base_weights,
)
from examples.baselines.constrain.constrain_condition import (
    ConditionNormConfig,
    build_hint,
    latest_relative_condition,
)
from examples.baselines.constrain.constrain_denoiser import (
    ConstrainDenoiser,
    build_constrain_denoiser,
)
from examples.baselines.constrain.constrain_dataset import ConstrainDemoDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_DP_DIR))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-path", type=Path,
                   default=_REPO_ROOT / "outputs" / "stackcube_obstacle" / "trajectory.h5")
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-iters", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--ema-power", type=float, default=0.75)
    p.add_argument("--save-every", type=int, default=None,
                   help="periodic checkpoint interval (a final checkpoint is always saved)")
    p.add_argument("--loss-log-size", type=int, default=200,
                   help="number of most recent losses stored in the checkpoint")
    return p.parse_args(argv)


def compute_epsilon_loss(
    denoiser,
    norm_cfg,
    obs_seq,
    act_seq,
    noise,
    timesteps,
    scheduler,
    base_obs_dim=48,
):
    """Official epsilon denoising loss with the relative-condition hint.

    obs_seq: (B, obs_horizon, 51) raw obstacle flat obs windows; the base
    condition stays the unchanged 2x48 trim while the hint comes from the
    window's latest frame via the shared condition seam (the same
    implementation the rollout adapter uses at inference). act_seq/noise:
    (B, pred_horizon, 4); timesteps: (B,).
    """
    obs_cond = obs_seq[..., :base_obs_dim].flatten(start_dim=1)
    hint = build_hint(
        latest_relative_condition(obs_seq, norm_cfg), pred_horizon=act_seq.shape[1]
    )
    noisy = scheduler.add_noise(act_seq, noise, timesteps)
    pred = denoiser(noisy, timesteps, global_cond=obs_cond, hint=hint)
    return torch.nn.functional.mse_loss(pred, noise)


def capture_rng(device):
    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        rng["torch_cuda"] = torch.cuda.get_rng_state()
    return rng


def restore_rng(rng, device):
    if "torch" in rng:
        torch.set_rng_state(rng["torch"].cpu() if torch.is_tensor(rng["torch"]) else rng["torch"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "python" in rng:
        random.setstate(rng["python"])
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state(
            rng["torch_cuda"].cpu() if torch.is_tensor(rng["torch_cuda"]) else rng["torch_cuda"]
        )


class ConstrainTrainer:
    """Training state for the constrain policy (frozen base + control branch)."""

    def __init__(
        self,
        demo_path,
        base_checkpoint,
        device="cpu",
        batch_size=1024,
        total_iters=30000,
        lr=1e-4,
        warmup=500,
        ema_power=0.75,
        seed=1,
        run_dir=None,
        norm_cfg=None,
        resume=None,
        loss_log_size=200,
    ):
        self.demo_path = Path(demo_path)
        self.base_checkpoint = Path(base_checkpoint)
        self.device = device
        self.batch_size = batch_size
        self.total_iters = total_iters
        self.lr = lr
        self.warmup = warmup
        self.ema_power = ema_power
        self.seed = seed
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.norm_cfg = norm_cfg if norm_cfg is not None else ConditionNormConfig()
        self.resume = Path(resume) if resume is not None else None
        self.loss_log_size = loss_log_size
        self.loss_history = []
        self.step = 0
        self.dataset = None
        self.dataloader = None
        self.denoiser = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ema = None

    @staticmethod
    def build_scheduler(num_diffusion_iters=100):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        return DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    @staticmethod
    def _model_config():
        return dict(
            input_dim=4,
            global_cond_dim=96,
            diffusion_step_embed_dim=64,
            down_dims=[64, 128, 256],
            n_groups=8,
            cond_dim=3,
            num_diffusion_iters=100,
        )

    def _load_base_net(self):
        agent = build_official_agent(device="cpu")
        ck = torch.load(self.base_checkpoint, map_location="cpu", weights_only=False)
        agent.load_state_dict(ck["ema_agent"])
        agent.eval()
        return agent.noise_pred_net

    def setup(self):
        """Build dataset, frozen base, control branch, optimizer/EMA and loader."""
        if self.dataloader is not None:
            return
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import EMAModel
        from examples.baselines.constrain.constrain_ckpt import load_constrain_denoiser

        self.dataset = ConstrainDemoDataset(self.demo_path)
        if self.resume is not None:
            _, ck = load_constrain_denoiser(self.resume, device=self.device, weights="model")
            self.denoiser = build_constrain_denoiser(
                ck["config"]["model"], device=self.device
            )
            self.denoiser.load_state_dict(
                {k: v.to(self.device) for k, v in ck["model"].items()}, strict=True
            )
            verify_base_weights(self.denoiser, ck["config"])
            self.norm_cfg = ConditionNormConfig.from_dict(ck["config"]["normalization"])
            self.step = int(ck["step"])
            self.loss_history = list(ck.get("config", {}).get("training", {}).get("loss_history", []))
            self.base_info = dict(ck["config"]["base_weights"])
            self._build_trainables(self.denoiser.trainable_parameters())
            if ck.get("optimizer") is not None:
                self.optimizer.load_state_dict(ck["optimizer"])
            if ck.get("lr_scheduler") is not None:
                self.lr_scheduler.load_state_dict(ck["lr_scheduler"])
            if ck.get("ema") is not None:
                self.ema.load_state_dict(ck["ema"])
            if ck.get("rng"):
                restore_rng(ck["rng"], self.device)
        else:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            base_net = self._load_base_net()
            self.denoiser = ConstrainDenoiser(base_net=base_net, cond_dim=3).to(self.device)
            self.base_info = {
                "source": str(self.base_checkpoint),
                "subkey": "ema_agent",
                "state_sha16": frozen_base_sha16(self.denoiser),
            }
            self._build_trainables(self.denoiser.trainable_parameters())
        self.dataloader = self._build_dataloader()

    def _build_trainables(self, params):
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import EMAModel

        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, betas=(0.95, 0.999), weight_decay=1e-6
        )
        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup,
            num_training_steps=self.total_iters,
        )
        self.ema = EMAModel(parameters=params, power=self.ema_power)

    def _build_dataloader(self):
        sampler = RandomSampler(self.dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=self.batch_size, drop_last=True)
        return DataLoader(self.dataset, batch_sampler=batch_sampler, num_workers=0)

    def train_steps(self, num_steps):
        """Run ``num_steps`` updates over a fresh shuffled pass; returns losses."""
        assert self.dataloader is not None, "call setup() first"
        scheduler = self.build_scheduler()
        it = iter(self.dataloader)
        losses = []
        for _ in range(num_steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.dataloader)
                batch = next(it)
            losses.append(self.train_batch(batch, scheduler))
        return losses

    def train_batch(self, batch, scheduler=None):
        """One update: epsilon loss -> backward -> optimizer/lr/EMA step."""
        scheduler = scheduler if scheduler is not None else self.build_scheduler()
        device = next(self.denoiser.parameters()).device
        obs = batch["observations"].to(device)
        act = batch["actions"].to(device)
        B = obs.shape[0]
        noise = torch.randn((B, act.shape[1], act.shape[2]), device=device)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        loss = compute_epsilon_loss(
            self.denoiser, self.norm_cfg, obs, act, noise, timesteps, scheduler
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {self.step}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.ema.step(self.denoiser.trainable_parameters())
        self.step += 1
        self.loss_history.append(float(loss.item()))
        self.loss_history = self.loss_history[-self.loss_log_size:]
        return loss

    def checkpoint_config(self):
        config = {
            "model": self._model_config(),
            "policy": dict(
                obs_horizon=2, act_horizon=8, pred_horizon=16,
                base_obs_dim=48, act_dim=4,
            ),
            "normalization": self.norm_cfg.to_dict(),
            "training": dict(
                demo_path=str(self.demo_path),
                total_iters=self.total_iters,
                batch_size=self.batch_size,
                lr=self.lr,
                warmup=self.warmup,
                ema_power=self.ema_power,
                seed=self.seed,
                loss_history=self.loss_history,
            ),
        }
        base_info = getattr(self, "base_info", None)
        if base_info is None:
            raise RuntimeError("base weight source not recorded (call setup() first)")
        config["base_weights"] = base_info
        return config

    def save_checkpoint(self, path):
        """Save the complete standalone checkpoint (verifying the frozen base)."""
        verify_base_weights(self.denoiser, self.checkpoint_config())
        return save_constrain_checkpoint(
            path,
            denoiser=self.denoiser,
            ema=self.ema,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            step=self.step,
            rng=capture_rng(self.device),
            config=self.checkpoint_config(),
        )


def training_evidence(trainer):
    """Tangible evidence for the ticket invariants after a training run:
    the frozen base is unchanged, the zero convs moved, and a backward pass
    at the trained weights reaches the upstream control branch and hint
    encoder."""
    scheduler = ConstrainTrainer.build_scheduler()
    batch = next(iter(trainer.dataloader))
    device = next(trainer.denoiser.parameters()).device
    obs = batch["observations"].to(device)
    act = batch["actions"].to(device)
    B = obs.shape[0]
    noise = torch.randn((B, act.shape[1], act.shape[2]), device=device)
    timesteps = torch.randint(0, 100, (B,), device=device).long()
    trainer.denoiser.train()
    trainer.optimizer.zero_grad()
    loss = compute_epsilon_loss(
        trainer.denoiser, trainer.norm_cfg, obs, act, noise, timesteps, scheduler
    )
    loss.backward()
    zeros_moved = [
        bool(torch.any(p != 0))
        for n, p in trainer.denoiser.named_parameters()
        if n.startswith("zero_") and n.endswith("weight")
    ]
    upstream = {}
    for prefix in ("control_down_modules.", "control_mid_modules.", "hint_encoder."):
        grads = [
            p.grad for n, p in trainer.denoiser.named_parameters()
            if n.startswith(prefix) and p.grad is not None
        ]
        upstream[prefix.rstrip(".")] = bool(grads) and any(
            torch.any(g != 0) for g in grads
        )
    return {
        "loss_finite": bool(torch.isfinite(loss)),
        "evidence_loss": float(loss.item()),
        "zero_conv_weights_moved": all(zeros_moved) and len(zeros_moved) > 0,
        "upstream_gradients_nonzero": upstream,
        "frozen_base_unchanged": frozen_base_sha16(trainer.denoiser) == trainer.base_info["state_sha16"],
    }


def main(argv=None):
    args = parse_args(argv)
    from diffusion_policy.utils import IterationBasedBatchSampler

    run_dir = args.run_dir if args.run_dir is not None else (
        _REPO_ROOT / "outputs" / "constrain_runs" / f"run_seed{args.seed}_iters{args.total_iters}"
    )
    trainer = ConstrainTrainer(
        demo_path=args.demo_path,
        base_checkpoint=args.base_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        total_iters=args.total_iters,
        lr=args.lr,
        warmup=args.warmup,
        ema_power=args.ema_power,
        seed=args.seed,
        run_dir=run_dir,
        resume=args.resume,
        loss_log_size=args.loss_log_size,
    )
    trainer.setup()

    sampler = RandomSampler(trainer.dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    dataloader = DataLoader(trainer.dataset, batch_sampler=batch_sampler, num_workers=0)
    scheduler = ConstrainTrainer.build_scheduler()

    # ``--total-iters`` is the target total update count for both fresh runs
    # and resumes (a resume fast-forwards the sampler to the saved step).
    for iteration, batch in enumerate(dataloader):
        if iteration < trainer.step:
            continue
        trainer.train_batch(batch, scheduler)
        if args.save_every is not None and (iteration + 1) % args.save_every == 0:
            path = trainer.save_checkpoint(Path(run_dir) / "checkpoints" / "periodic.pt")
            print(f"saved periodic checkpoint at step {trainer.step}: {path}", flush=True)
    final = trainer.save_checkpoint(Path(run_dir) / "checkpoints" / "final.pt")
    evidence = training_evidence(trainer)
    history = trainer.loss_history
    summary = dict(
        step=trainer.step,
        all_finite=bool(history) and all(np.isfinite(history)),
        num_recorded_losses=len(history),
        first_loss=history[0] if history else None,
        last_loss=history[-1] if history else None,
        evidence=evidence,
        checkpoint=str(final),
        config=trainer.checkpoint_config(),
    )
    if not all(evidence.values() if not isinstance(v, dict) else all(v.values()) for v in [evidence]) or not summary["all_finite"]:
        print(json.dumps(summary, indent=2))
        raise RuntimeError("training evidence failed; see the printed summary")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
    print(f"saved: {final}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
