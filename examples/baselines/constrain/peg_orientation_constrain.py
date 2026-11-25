"""Peg-specific ConstrainNet training and live-hint inference.

This keeps the generic obstacle trainer untouched: Peg uses 39d observations,
7d native pose actions and a 2d wrapped-yaw condition.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from examples.baselines.constrain.constrain_denoiser import ConstrainDenoiser
from examples.baselines.constrain.constrain_ckpt import frozen_base_sha16, verify_base_weights
from examples.baselines.constrain.peg_orientation_base_dp import (
    ACT_HORIZON, ACT_DIM, OBS_DIM, OBS_HORIZON, PRED_HORIZON,
    build_agent, sha256_file,
)

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt"
DATA = ROOT / "demos/PegInsertionSide-orientation-random-100/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
MANIFEST = ROOT / "demos/PegInsertionSide-orientation-random-100/training_coverage.json"
FORMAT = "peg_orientation_constrain_v1"
AUTHORIZED_RESUME_SHA256 = "6e26647b5ee1ad356d57f06464f77a1884fcd76b1fad52c8ed181e9e0d1de1de"
MODEL_CONFIG = dict(input_dim=7, global_cond_dim=78, diffusion_step_embed_dim=64,
                    down_dims=[64, 128, 256], n_groups=8, cond_dim=2,
                    num_diffusion_iters=100)


def _hint(obs, target_yaw):
    """Derive world +X peg yaw from the current 39d observation, then wrap error."""
    from transforms3d.quaternions import quat2mat

    quat = np.asarray(obs[28:32], dtype=np.float64)
    axis = quat2mat(quat)[:, 0]
    peg_yaw = np.arctan2(axis[1], axis[0])
    error = np.arctan2(np.sin(peg_yaw - target_yaw), np.cos(peg_yaw - target_yaw))
    return np.asarray([np.sin(error), np.cos(error)], dtype=np.float32)


class PegConstrainDataset(Dataset):
    """Official 2/16 windowing over the configured trajectories with current hints."""

    def __init__(self, data_path=DATA, manifest_path=MANIFEST, expected_episodes=100):
        metadata = json.loads(Path(manifest_path).read_text())
        rows = metadata["episodes"]
        if len(rows) != expected_episodes:
            raise ValueError(f"expected {expected_episodes} Peg training episodes")
        self.obs, self.actions, self.hints, self.slices = [], [], [], []
        with h5py.File(data_path, "r") as f:
            for episode, row in enumerate(rows):
                group = f[f"traj_{episode}"]
                obs, actions = group["obs"][:], group["actions"][:]
                if obs.shape != (len(actions) + 1, OBS_DIM) or actions.shape[1:] != (ACT_DIM,):
                    raise ValueError(f"traj_{episode}: expected T+1 39d obs and T 7d actions")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError(f"traj_{episode}: non-finite data")
                hints = np.stack([_hint(frame, float(row["target_yaw_rad"])) for frame in obs])
                self.obs.append(torch.tensor(obs, dtype=torch.float32))
                self.actions.append(torch.tensor(actions, dtype=torch.float32))
                self.hints.append(torch.tensor(hints, dtype=torch.float32))
                length = len(actions)
                self.slices.extend((episode, start) for start in range(-1, length - 2))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        episode, start = self.slices[index]
        obs, act, hint = self.obs[episode], self.actions[episode], self.hints[episode]
        obs_start = max(0, start)
        observations = obs[obs_start:start + 2]
        if start < 0:
            observations = torch.cat((obs[:1], observations))
        actions = act[max(0, start):start + 16]
        hints = hint[max(0, start):start + 2]
        if start < 0:
            actions = torch.cat((act[:1], actions))
            hints = torch.cat((hint[:1], hints))
        if len(actions) < 16:
            pad = torch.cat((torch.zeros(6), actions[-1, 6:7]))
            actions = torch.cat((actions, pad.repeat(16 - len(actions), 1)))
        return {"observations": observations, "actions": actions, "hint": hints[-1]}


def build_peg_denoiser(base_net):
    return ConstrainDenoiser(base_net=base_net, cond_dim=2)


def compute_peg_epsilon_loss(denoiser, obs, actions, hints, noise, timesteps, scheduler):
    hint = hints[:, None, :].expand(-1, actions.shape[1], -1)
    noisy = scheduler.add_noise(actions, noise, timesteps)
    pred = denoiser(noisy, timesteps, global_cond=obs.flatten(start_dim=1), hint=hint)
    return torch.nn.functional.mse_loss(pred, noise)


def balanced_replay_batch(random_batch, fixed_batch):
    """Join equally sized windows from the two replay populations."""
    return {key: torch.cat((random_batch[key], fixed_batch[key]), dim=0)
            for key in random_batch}


def restore_training_state(checkpoint, denoiser, optimizer, ema, device):
    """Restore raw trainable weights and optimizer/EMA/RNG state for a new phase."""
    denoiser.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.load_state_dict(checkpoint["ema"])
    ema.to(device)
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if "torch_cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng["torch_cuda"].cpu())


def validate_resume(checkpoint, args, base_sha256, random_sha256, random_manifest_sha256,
                    fixed_sha256, fixed_manifest_sha256, parent_sha256):
    if parent_sha256 != AUTHORIZED_RESUME_SHA256:
        raise ValueError("resume checkpoint identity mismatch")
    previous = checkpoint["config"]["training"]
    expected = {
        "random_episodes": args.random_episodes,
        "data_sha256": random_sha256,
        "manifest_sha256": random_manifest_sha256,
        "fixed_episodes": args.fixed_episodes,
        "fixed_data_sha256": fixed_sha256,
        "fixed_manifest_sha256": fixed_manifest_sha256,
    }
    for key, value in expected.items():
        if previous.get(key) != value:
            raise ValueError(f"resume checkpoint {key} mismatch")
    if checkpoint["config"]["base_weights"]["sha256"] != base_sha256:
        raise ValueError("resume checkpoint Base identity mismatch")
    if int(checkpoint["step"]) != 30000 or args.updates <= 30000:
        raise ValueError("resume requires update-30000 checkpoint and cumulative target > 30000")


def _load_base(checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    agent = build_agent("cpu")
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    return agent.noise_pred_net.to(device)


class PegConstrainAdapter:
    """Full 7d native-action sampler; hint is refreshed on every live observation."""

    def __init__(self, checkpoint, device="cpu", control_enabled=True, task=None,
                 hint_mode="correct", hint_assignment_sha256=None,
                 record_all_hints=False):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        if ck.get("format") != FORMAT:
            raise ValueError("not a Peg orientation Constrain checkpoint")
        cfg = ck["config"]["model"]
        base = build_agent("cpu").noise_pred_net
        self.denoiser = ConstrainDenoiser(base, cond_dim=2).to(device)
        self.denoiser.load_state_dict(ck["model"], strict=True)
        verify_base_weights(self.denoiser, ck["config"])
        self.denoiser.eval()
        self.device = torch.device(device)
        self.control_enabled = control_enabled
        self.task = task
        if hint_mode not in ("correct", "fixed_standard", "shuffled"):
            raise ValueError(f"unknown hint mode: {hint_mode}")
        self.hint_mode = hint_mode
        self.actual_target_yaw = None
        self.shuffled_target_yaw = None
        self.hint_samples = []
        self.sampler_calls = 0
        self.record_all_hints = bool(record_all_hints)
        self.scheduler = self._scheduler(cfg["num_diffusion_iters"])
        if ck.get("ema") is not None:
            from diffusers.training_utils import EMAModel
            ema = EMAModel(parameters=self.denoiser.trainable_parameters(), power=.75)
            ema.load_state_dict(ck["ema"])
            ema.copy_to(self.denoiser.trainable_parameters())
            self.denoiser.eval()
        self.metadata = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                         "weights": "ema control branch + frozen base", "step": int(ck["step"]),
                         "base_weights": ck["config"]["base_weights"],
                         "act_dim": 7, "base_obs_dim": 39, "obs_horizon": 2,
                          "act_horizon": 8, "pred_horizon": 16,
                          "num_diffusion_iters": int(cfg["num_diffusion_iters"]),
                          "control_enabled": bool(control_enabled), "hint_mode": hint_mode,
                          "hint_assignment_sha256": hint_assignment_sha256}

    def set_episode_context(self, scene_index, scene_metadata, shuffled_target_yaw=None):
        self.actual_target_yaw = float(scene_metadata["target_yaw_rad"])
        self.shuffled_target_yaw = (None if shuffled_target_yaw is None
                                    else float(shuffled_target_yaw))
        self.hint_samples = []
        self.sampler_calls = 0

    def _live_hint(self):
        if self.hint_mode == "correct":
            return self.task.get_orientation_hint()
        if self.actual_target_yaw is None:
            raise RuntimeError("Peg episode context must be set before intervention sampling")
        from examples.baselines.constrain.peg_orientation_data import (
            _pose_yaw, build_hint_for_mode,
        )
        peg_yaw = _pose_yaw(self.task.peg.pose)
        return build_hint_for_mode(self.hint_mode, peg_yaw, self.actual_target_yaw,
                                   self.shuffled_target_yaw)

    @staticmethod
    def _scheduler(steps):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        return DDPMScheduler(num_train_timesteps=steps, beta_schedule="squaredcos_cap_v2",
                             clip_sample=True, prediction_type="epsilon")

    def get_action(self, observations, hint=None):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if hint is None:
            if self.task is None:
                raise RuntimeError("live Peg task is required to refresh the orientation hint")
            hint = self._live_hint()
        hint = torch.as_tensor(hint, dtype=torch.float32, device=self.device)
        if obs.shape[-2:] != (2, 39) or hint.shape[-1] != 2:
            raise ValueError("expected (...,2,39) observations and (...,2) live yaw hint")
        if obs.ndim == 2:
            obs, hint = obs[None], hint[None]
        elif hint.ndim == 1:
            hint = hint[None]
        self.sampler_calls += 1
        task = self.task
        peg_yaw = None
        if task is not None and hasattr(task, "peg"):
            from examples.baselines.constrain.peg_orientation_data import _pose_yaw
            peg_yaw = _pose_yaw(task.peg.pose)
        sample = {"sampler_call": self.sampler_calls, "peg_yaw_rad": peg_yaw,
                  "actual_target_yaw_rad": self.actual_target_yaw,
                  "shuffled_target_yaw_rad": self.shuffled_target_yaw,
                  "hint": hint[0].detach().cpu().tolist()}
        if self.record_all_hints:
            self.hint_samples.append(sample)
        elif self.sampler_calls <= 2:
            self.hint_samples.append(sample)
        else:
            if len(self.hint_samples) == 2:
                self.hint_samples.append(sample)
            else:
                self.hint_samples[-1] = sample
        repeated = hint[:, None, :].expand(-1, PRED_HORIZON, -1)
        sample = torch.randn((len(obs), PRED_HORIZON, 7), device=self.device)
        with torch.no_grad():
            for step in self.scheduler.timesteps:
                noise = self.denoiser(sample, step, obs.flatten(1), hint=repeated,
                                      enabled=self.control_enabled)
                sample = self.scheduler.step(noise, step, sample).prev_sample
        return sample[:, 1:1 + ACT_HORIZON]


def train(args):
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel

    if args.resume_from is not None:
        if args.fixed_data is None or args.fixed_manifest is None:
            raise ValueError("resume requires the original balanced fixed replay")
        if args.ckpt.resolve() == args.resume_from.resolve().parent:
            raise ValueError("resume checkpoint directory cannot be used as output")
    else:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dataset = PegConstrainDataset(args.data, args.manifest,
                                  expected_episodes=args.random_episodes)
    loader = DataLoader(dataset, batch_size=args.batch_size // 2 if args.fixed_data else args.batch_size,
                        shuffle=True, drop_last=True,
                        num_workers=0)
    fixed_loader = None
    if args.fixed_data is not None:
        fixed_dataset = PegConstrainDataset(args.fixed_data, args.fixed_manifest,
                                            expected_episodes=args.fixed_episodes)
        if args.batch_size % 2:
            raise ValueError("balanced replay requires an even batch size")
        fixed_loader = DataLoader(fixed_dataset, batch_size=args.batch_size // 2,
                                  shuffle=True, drop_last=True, num_workers=0)
    base = _load_base(args.base, device)
    for p in base.parameters(): p.requires_grad_(False)
    denoiser = build_peg_denoiser(base).to(device)
    base_before = {k: v.detach().clone() for k, v in base.state_dict().items()}
    params = denoiser.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=1e-4, betas=(.95, .999), weight_decay=1e-6)
    ema = EMAModel(parameters=params, power=.75)
    resume = None
    if args.resume_from is not None:
        resume = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        validate_resume(resume, args, sha256_file(args.base), sha256_file(args.data),
                        sha256_file(args.manifest), sha256_file(args.fixed_data),
                        sha256_file(args.fixed_manifest), sha256_file(args.resume_from))
        restore_training_state(resume, denoiser, optimizer, ema, device)
        phase_updates = args.updates - int(resume["step"])
        phase_start = int(resume["step"])
    else:
        phase_updates = args.updates
        phase_start = 0
    lr_schedule = get_scheduler("cosine", optimizer, num_warmup_steps=500,
                                num_training_steps=phase_updates)
    scheduler = PegConstrainAdapter._scheduler(100)
    args.run.mkdir(parents=True, exist_ok=True); args.ckpt.mkdir(parents=True, exist_ok=True)
    if (args.run / "training.jsonl").exists() or any(args.ckpt.glob("update_*.pt")):
        raise FileExistsError("refusing to overwrite an existing Peg exploratory training run")
    if args.resume_from is not None:
        config = dict(resume["config"])
        config["training"] = dict(config["training"])
        config["training"].update({"total_iters": args.updates,
            "continuation": {"parent_checkpoint": str(args.resume_from),
                "parent_sha256": sha256_file(args.resume_from), "parent_step": phase_start,
                "remaining_updates": phase_updates, "phase_lr_schedule": "cosine",
                "phase_warmup": 500, "phase_peak_lr": 1e-4,
                "old_scheduler_state": resume["lr_scheduler"],
                "optimizer_restored": True, "ema_restored": True,
                "sampler_restart": True, "exact_trajectory_resume": False}})
    else:
        config = {"model": MODEL_CONFIG, "policy": {"obs_horizon": 2, "act_horizon": 8,
              "pred_horizon": 16, "base_obs_dim": 39, "act_dim": 7},
              "base_weights": {"source": str(args.base), "subkey": "ema_agent",
                               "sha256": sha256_file(args.base), "state_sha16": frozen_base_sha16(denoiser)},
              "training": {"seed": args.seed, "total_iters": args.updates,
                           "batch_size": args.batch_size, "lr": 1e-4,
                           "warmup": 500, "ema_power": .75,
                            "random_data_path": str(args.data),
                            "random_manifest_path": str(args.manifest),
                            "random_episodes": args.random_episodes,
                            "random_windows": len(dataset),
                            "data_sha256": sha256_file(args.data),
                           "manifest_sha256": sha256_file(args.manifest),
                            "coverage_sha256": sha256_file(args.manifest),
                            "fixed_data_path": str(args.fixed_data) if args.fixed_data else None,
                            "fixed_manifest_path": str(args.fixed_manifest) if args.fixed_manifest else None,
                            "fixed_episodes": args.fixed_episodes if args.fixed_data else None,
                            "fixed_windows": len(fixed_dataset) if fixed_loader is not None else None,
                            "fixed_data_sha256": sha256_file(args.fixed_data) if args.fixed_data else None,
                            "fixed_manifest_sha256": sha256_file(args.fixed_manifest) if args.fixed_manifest else None,
                             "replay_mix_per_batch": ([args.batch_size // 2, args.batch_size // 2]
                                                       if fixed_loader is not None else None)}}
    iterator = iter(loader)
    fixed_iterator = iter(fixed_loader) if fixed_loader is not None else None
    for phase_step in range(1, phase_updates + 1):
        step = phase_start + phase_step
        try: batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        if fixed_loader is not None:
            try: fixed_batch = next(fixed_iterator)
            except StopIteration:
                fixed_iterator = iter(fixed_loader); fixed_batch = next(fixed_iterator)
            batch = balanced_replay_batch(batch, fixed_batch)
        obs = batch["observations"].to(device); actions = batch["actions"].to(device)
        hint = batch["hint"].to(device)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(obs),), device=device)
        loss = compute_peg_epsilon_loss(denoiser, obs, actions, hint, noise, timesteps, scheduler)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at update {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        lr_schedule.step(); ema.step(params)
        if any(not torch.equal(base_before[k], v) for k, v in base.state_dict().items()):
            raise RuntimeError("frozen Base changed during optimizer update")
        if step % 100 == 0:
            with (args.run / "training.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.item())}) + "\n")
        if step % args.save_every == 0 or step == args.updates:
            payload = {"format": FORMAT, "model": denoiser.state_dict(), "ema": ema.state_dict(),
                       "optimizer": optimizer.state_dict(), "lr_scheduler": lr_schedule.state_dict(),
                        "step": step, "phase_step": phase_step, "rng": {"torch": torch.get_rng_state(),
                       "numpy": np.random.get_state(), "python": random.getstate()}, "config": config}
            if torch.cuda.is_available(): payload["rng"]["torch_cuda"] = torch.cuda.get_rng_state()
            path = args.ckpt / f"update_{step:05d}.pt"
            tmp = path.with_suffix(".tmp"); torch.save(payload, tmp); tmp.replace(path)
    return {"updates": step, "phase_updates": phase_updates, "base_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--base", type=Path, default=BASE); parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--fixed-data", type=Path)
    parser.add_argument("--fixed-manifest", type=Path)
    parser.add_argument("--fixed-episodes", type=int, default=1000)
    parser.add_argument("--random-episodes", type=int, default=100)
    parser.add_argument("--run", type=Path); parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--updates", type=int, default=30000); parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-eval", action="store_true")
    parser.add_argument("--seed-base", type=int, default=1)
    parser.add_argument("--episode-cap", type=int, default=300)
    parser.add_argument("--hint-mode", choices=("correct", "fixed_standard", "shuffled"), default="correct")
    parser.add_argument("--hint-assignment", type=Path)
    parser.add_argument("--scene-indices", type=int, nargs="+")
    parser.add_argument("--record-stage-geometry", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-scene-indices", type=int, nargs="+")
    args = parser.parse_args(argv)
    if args.mode == "evaluate":
        if args.catalog is None or args.output is None:
            parser.error("evaluate requires --catalog and --output")
        from examples.baselines.constrain.peg_orientation_base_dp import evaluate
        if args.base_eval:
            evaluate(args.base, args.catalog, args.output, device=args.device,
                     max_episode_steps=args.episode_cap, env_time_limit=args.episode_cap,
                      seed=args.seed_base, scene_indices=args.scene_indices,
                      record_phase_trace=args.record_stage_geometry,
                      record_stage_geometry=args.record_stage_geometry,
                      video=args.video, video_scene_indices=args.video_scene_indices)
        else:
            if args.checkpoint is None:
                parser.error("Constrain evaluation requires --checkpoint")
            assignment = (json.loads(args.hint_assignment.read_text())
                          if args.hint_assignment is not None else None)
            if args.hint_mode == "shuffled" and assignment is None:
                parser.error("shuffled evaluation requires --hint-assignment")
            assignment_sha = (hashlib.sha256(json.dumps(
                assignment, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest() if assignment is not None else None)
            evaluate(args.checkpoint, args.catalog, args.output, device=args.device,
                      max_episode_steps=args.episode_cap, env_time_limit=args.episode_cap,
                       seed=args.seed_base, scene_indices=args.scene_indices,
                       record_phase_trace=args.record_stage_geometry,
                       record_stage_geometry=args.record_stage_geometry,
                       video=args.video, video_scene_indices=args.video_scene_indices,
                     policy_factory=lambda task: PegConstrainAdapter(
                         args.checkpoint, device=args.device, task=task,
                         hint_mode=args.hint_mode,
                         hint_assignment_sha256=assignment_sha),
                     hint_mode=args.hint_mode, hint_assignment=assignment)
        return
    if args.run is None or args.ckpt is None:
        parser.error("training requires --run and --ckpt")
    if (args.fixed_data is None) != (args.fixed_manifest is None):
        parser.error("--fixed-data and --fixed-manifest must be supplied together")
    if args.seed != 1 or args.batch_size != 1024 or args.save_every != 5000:
        parser.error("Peg training fixes seed=1, batch=1024, save-every=5000")
    if args.resume_from is None and args.updates != 30000:
        parser.error("fresh exploratory training fixes updates=30000")
    if args.resume_from is not None and args.updates != 50000:
        parser.error("the approved update-30000 continuation target is cumulative update 50000")
    print(json.dumps(train(args), sort_keys=True))


if __name__ == "__main__":
    main()
