"""Task-specific PickCube obstacle ConstrainNet training and inference."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

from examples.baselines.constrain.constrain_ckpt import (
    frozen_base_sha16, save_constrain_checkpoint, verify_base_weights,
)
from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
from examples.baselines.constrain.pickcube_base_dp import (
    FORMAL_H5, FORMAL_MANIFEST, FORMAL_DIR, MODEL_CONFIG as BASE_MODEL_CONFIG,
    NAMED_FIELDS, PickCubeBaseDPAdapter, build_agent, contract, sha256_file,
)

BASE_CKPT = Path("outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt")
TRUSTED_H5_SHA256 = "fa846f52b5ae08de0243cfa52b40a4b0e7ce160881aa9db59ebaf4952c02904e"
TRUSTED_MANIFEST_SHA256 = "8ef0d2739d6bf5675316ab37f42f1437cc19d80739e6f9102973beeabd6d1e58"
TRUSTED_BASE_SHA256 = "b379573970712bcc17279ed237602a7be823acce9f7134ba125041ba5478389c"
TRUSTED_BASE_STATE_SHA256 = "d3184b571c83f5c53d32d80cc2bc4441c66337d5658e241cbdc12bb410860bb8"
FORMAT = "pickcube_constrain_v1"
POLICY_CONTRACT = contract()["policy"]
MODEL_CONFIG = {
    "input_dim": 7, "global_cond_dim": 84, "diffusion_step_embed_dim": 64,
    "down_dims": [64, 128, 256], "n_groups": 8, "cond_dim": 3,
    "num_diffusion_iters": 100,
}


def _field_offsets(fields):
    offsets = {}
    cursor = 0
    for name, width in fields:
        offsets[name] = (cursor, cursor + width)
        cursor += width
    return offsets


_OFFSETS = _field_offsets(NAMED_FIELDS)
TCP_XY = tuple(range(_OFFSETS["extra/tcp_pose"][0], _OFFSETS["extra/tcp_pose"][0] + 2))
OBSTACLE_XY = tuple(range(_OFFSETS["extra/pickcube_obstacle"][0], _OFFSETS["extra/pickcube_obstacle"][0] + 2))
OBSTACLE_RADIUS = _OFFSETS["extra/pickcube_obstacle"][1] - 1


@dataclass(frozen=True)
class PickCubeConditionNorm:
    relative_xy_scale_m: float = 0.5
    radius_scale_m: float = 0.08
    clip: bool = True

    def __post_init__(self):
        if self.relative_xy_scale_m <= 0 or self.radius_scale_m <= 0:
            raise ValueError("condition normalization scales must be positive")

    def to_dict(self):
        return {
            "definition": "[obstacle_x-tcp_x, obstacle_y-tcp_y, cylinder_radius]",
            "axes": "world-parallel XY; relative offset anchored at latest TCP",
            "units": "meters",
            "formula": "clip(value / [relative_xy_scale_m, relative_xy_scale_m, radius_scale_m], -1, 1)",
            "relative_xy_scale_m": self.relative_xy_scale_m,
            "radius_scale_m": self.radius_scale_m,
            "clip": self.clip,
            "tcp_field": "extra/tcp_pose[:2]",
            "obstacle_field": "extra/pickcube_obstacle[x,y,radius]",
            "named_fields": NAMED_FIELDS,
        }

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("PickCube condition normalization config must be a mapping")
        cfg = cls(value["relative_xy_scale_m"], value["radius_scale_m"], value["clip"])
        if value != cfg.to_dict():
            raise ValueError("PickCube condition normalization contract mismatch")
        return cfg


def pickcube_relative_condition(queued_observation, norm=PickCubeConditionNorm()):
    """Extract normalized latest-frame obstacle-relative-to-TCP condition."""
    obs = torch.as_tensor(queued_observation)
    if obs.shape[-1] != 45 or obs.shape[-2] != 2:
        raise ValueError(f"expected queued PickCube native state (2,45), got {tuple(obs.shape)}")
    latest = obs[..., -1, :]
    tcp = latest[..., list(TCP_XY)]
    obstacle = latest[..., list(OBSTACLE_XY)]
    radius = latest[..., OBSTACLE_RADIUS:OBSTACLE_RADIUS + 1]
    scale = obs.new_tensor([norm.relative_xy_scale_m, norm.relative_xy_scale_m, norm.radius_scale_m])
    condition = torch.cat((obstacle - tcp, radius), dim=-1) / scale
    return condition.clamp(-1, 1) if norm.clip else condition


class PickCubeObstacleDataset(Dataset):
    """Strictly load the frozen 100 safe obstacle episodes, no direct/pilot data."""

    def __init__(self, data_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST):
        self.data_path, self.manifest_path = Path(data_path), Path(manifest_path)
        self.h5_sha256, self.manifest_sha256 = sha256_file(self.data_path), sha256_file(self.manifest_path)
        if self.h5_sha256 != TRUSTED_H5_SHA256 or self.manifest_sha256 != TRUSTED_MANIFEST_SHA256:
            raise ValueError("formal PickCube obstacle dataset identity is not the trusted Ticket 02 dataset")
        manifest = json.loads(self.manifest_path.read_text())
        episodes = manifest.get("episodes", [])
        ids = [int(e["episode_id"]) for e in episodes]
        if sorted(ids) != list(range(1100)) or len(set(ids)) != 1100:
            raise ValueError("formal manifest must contain IDs 0..1099 exactly once")
        by_id = {int(e["episode_id"]): e for e in episodes}
        selected = list(range(1000, 1100))
        if any(by_id[i].get("kind") != "obstacle" for i in selected):
            raise ValueError("training selection must be the exact obstacle IDs 1000..1099")
        self.trajectory_lengths, self.slices, self.observations, self.actions = [], [], [], []
        with h5py.File(self.data_path, "r") as handle:
            if len(handle) != 1100:
                raise ValueError(f"formal HDF5 must contain exactly 1100 episodes, got {len(handle)}")
            for tid in selected:
                key, episode = f"traj_{tid}", handle[f"traj_{tid}"]
                if episode.attrs.get("kind") != "obstacle":
                    raise ValueError(f"{key}: HDF5 class metadata differs from formal manifest")
                obs, action = episode["obs"][:], episode["actions"][:]
                if obs.ndim != 2 or obs.shape[1] != 45 or action.ndim != 2 or action.shape[1] != 7:
                    raise ValueError(f"{key}: native PickCube observation/action schema mismatch")
                if len(obs) != len(action) + 1 or not np.isfinite(obs).all() or not np.isfinite(action).all():
                    raise ValueError(f"{key}: malformed or non-finite native trajectory")
                if np.max(np.abs(action)) > 1.00001:
                    raise ValueError(f"{key}: action outside native normalized bounds")
                if "success" not in episode or not bool(episode["success"][-1]) or not bool(episode["terminated"][-1]) or bool(episode["truncated"][:].any()):
                    raise ValueError(f"{key}: missing safe successful terminal evidence")
                record = by_id[tid]
                if record.get("terminated", [0])[-1] != 1 or record.get("truncated", [1])[-1] != 0:
                    raise ValueError(f"{key}: manifest terminal evidence is not successful")
                infos = record.get("infos", [])
                if any(bool(info.get(flag, [False])[-1]) for info in infos
                       for flag in ("obstacle_collision", "obstacle_crossing")):
                    raise ValueError(f"{key}: unsafe obstacle demonstration rejected")
                self.observations.append(torch.tensor(obs, dtype=torch.float32))
                self.actions.append(torch.tensor(action, dtype=torch.float32))
                length = len(action)
                self.trajectory_lengths.append(length)
                for start in range(-1, length - 16 + 14):
                    self.slices.append((len(self.actions) - 1, start, start + 16))

    @property
    def num_trajectories(self):
        return len(self.trajectory_lengths)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start, end = self.slices[index]
        obs, actions, length = self.observations[trajectory], self.actions[trajectory], self.trajectory_lengths[trajectory]
        obs_seq = obs[max(0, start):start + 2]
        action_seq = actions[max(0, start):end]
        if start < 0:
            obs_seq = torch.cat((obs_seq[:1].repeat(-start, 1), obs_seq))
            action_seq = torch.cat((action_seq[:1].repeat(-start, 1), action_seq))
        if end > length:
            padding = torch.cat((torch.zeros(6), action_seq[-1, 6:7])).repeat(end - length, 1)
            action_seq = torch.cat((action_seq, padding))
        return {"observations": obs_seq, "actions": action_seq}


def _state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _trusted_base(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file() or sha256_file(checkpoint) != TRUSTED_BASE_SHA256:
        raise ValueError("base checkpoint is not the trusted Ticket 03 final EMA")
    ck = PickCubeBaseDPAdapter.validate_checkpoint(checkpoint, expected_episode_ids=list(range(1000)), trusted_h5_sha256=TRUSTED_H5_SHA256)
    if ck.get("step") != 30000 or ck.get("selected_weights") not in (None, "ema"):
        # The checkpoint is the final EMA artifact; some version metadata lives in its embedded config.
        if ck.get("step") != 30000:
            raise ValueError("PickCube base checkpoint is not the completed 30000-update model")
    return ck


def _model_config():
    return dict(MODEL_CONFIG)


def _policy_contract():
    return dict(POLICY_CONTRACT, base_obs_dim=42, obstacle_obs_dim=45, obs_horizon=2,
                pred_horizon=16, act_horizon=8, act_dim=7, control_mode="pd_ee_delta_pose",
                robot="panda", observation_mode="state", named_state_fields=NAMED_FIELDS)


def equivalence_evidence(denoiser, base_checkpoint=BASE_CKPT, device="cpu", seed=1947,
                         control_enabled=True):
    """Independent same-noise epsilon and 100-step scheduler equivalence."""
    agent = build_agent(device)
    base_ck = _trusted_base(base_checkpoint)
    agent.load_state_dict(base_ck["weights"]["ema"], strict=True)
    agent.eval()
    independent_base = agent.noise_pred_net
    sample_gen = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn((2, 16, 7), generator=sample_gen).to(device)
    global_cond = torch.randn((2, 84), generator=sample_gen).to(device)
    timestep = torch.tensor([13, 71], device=device)
    with torch.no_grad():
        base_output = independent_base(sample, timestep, global_cond=global_cond)
        control_output = denoiser(sample, timestep, global_cond=global_cond,
                                  hint=torch.zeros(2, 16, 3, device=device), enabled=control_enabled)
    epsilon_diff = float((base_output - control_output).abs().max())
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                              clip_sample=True, prediction_type="epsilon")
    reference_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                        clip_sample=True, prediction_type="epsilon")
    initial = torch.randn((1, 16, 7), generator=sample_gen).to(device)
    ours, reference = initial.clone(), initial.clone()
    cond = global_cond[:1]
    scheduler.set_timesteps(100)
    reference_scheduler.set_timesteps(100)
    sample_generator = torch.Generator(device=device).manual_seed(seed + 1)
    reference_generator = torch.Generator(device=device).manual_seed(seed + 1)
    with torch.no_grad():
        for t, reference_t in zip(scheduler.timesteps, reference_scheduler.timesteps):
            ours = scheduler.step(denoiser(ours, t, global_cond=cond,
                hint=torch.zeros(1, 16, 3, device=device), enabled=control_enabled), t, ours,
                generator=sample_generator).prev_sample
            reference = reference_scheduler.step(independent_base(reference, reference_t, global_cond=cond),
                reference_t, reference, generator=reference_generator).prev_sample
    sample_diff = float((ours - reference).abs().max())
    return {"epsilon_max_abs_diff": epsilon_diff, "sampling_100step_max_abs_diff": sample_diff,
            "epsilon_matches": epsilon_diff <= 1e-6, "sampling_matches": sample_diff <= 1e-6,
            "noise_seed": seed, "scheduler_steps": 100}


def load_pickcube_constrain(checkpoint, device="cpu", weights="ema", base_checkpoint=None):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    expected_diffusion = {"scheduler": "DDPMScheduler", "beta_schedule": "squaredcos_cap_v2",
                          "clip_sample": True, "prediction_type": "epsilon", "num_train_timesteps": 100}
    data = cfg.get("data_identity", {})
    base = cfg.get("base_weights", {})
    training = cfg.get("training", {})
    if (ck.get("format") != FORMAT or cfg.get("model") != _model_config()
            or cfg.get("policy") != _policy_contract() or cfg.get("diffusion") != expected_diffusion
            or int(ck.get("step", -1)) != 30000
            or data.get("h5_sha256") != TRUSTED_H5_SHA256
            or data.get("manifest_sha256") != TRUSTED_MANIFEST_SHA256
            or data.get("selected_class") != "obstacle" or data.get("episodes") != 100
            or data.get("selected_episode_ids") != list(range(1000, 1100))
            or base.get("sha256") != TRUSTED_BASE_SHA256 or base.get("subkey") != "weights.ema"
            or base.get("state_sha256") != TRUSTED_BASE_STATE_SHA256
            or base.get("state_sha16") != TRUSTED_BASE_STATE_SHA256[:16]
            or base.get("updates") != 30000
            or training.get("seed") != 1 or training.get("total_iters") != 30000
            or training.get("batch_size") != 1024 or training.get("lr") != 1e-4
            or training.get("warmup") != 500 or training.get("ema_power") != 0.75
            or training.get("selected_weights") != "final EMA fixed before training"):
        raise ValueError("PickCube ConstrainNet checkpoint contract mismatch")
    PickCubeConditionNorm.from_dict(cfg.get("normalization", {}))
    if weights not in ("ema", "model"):
        raise ValueError("weights must be 'ema' or 'model'")
    denoiser = build_constrain_denoiser(cfg["model"], device=device)
    denoiser.load_state_dict(ck["model"], strict=True)
    verify_base_weights(denoiser, cfg)
    if weights == "ema":
        if ck.get("ema") is None:
            raise ValueError("checkpoint has no EMA weights")
        from diffusers.training_utils import EMAModel
        ema = EMAModel(parameters=denoiser.trainable_parameters(), power=0.75)
        ema.load_state_dict(ck["ema"])
        ema.copy_to(denoiser.trainable_parameters())
    if base_checkpoint is not None:
        base = _trusted_base(base_checkpoint)
        agent = build_agent("cpu")
        agent.load_state_dict(base["weights"]["ema"], strict=True)
        actual = _state_sha(agent.noise_pred_net.state_dict())
        if actual != cfg["base_weights"]["state_sha256"]:
            raise ValueError("embedded ConstrainNet base differs from independently loaded trusted base")
    return denoiser, ck


class PickCubeConstrainAdapter:
    def __init__(self, checkpoint, device="cpu", weights="ema", control_enabled=True, base_checkpoint=None):
        self.denoiser, self.checkpoint = load_pickcube_constrain(checkpoint, device, weights, base_checkpoint)
        self.denoiser.eval(); self.device = torch.device(device)
        self.config = self.checkpoint["config"]
        self.norm = PickCubeConditionNorm.from_dict(self.config["normalization"])
        self.control_enabled = control_enabled
        self.obs_horizon, self.pred_horizon, self.act_horizon = 2, 16, 8
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
        self.latest = None

    @staticmethod
    def map_action(action):
        action = torch.as_tensor(action)
        if action.shape[-1] != 7 or not torch.isfinite(action).all():
            raise ValueError("expected finite native seven-channel PickCube actions")
        if bool((action.abs() > 1.000001).any()):
            raise ValueError("native seven-channel PickCube actions must be within [-1,1]")
        return action

    def base_cond(self, queued):
        queued = torch.as_tensor(queued, dtype=torch.float32, device=self.device)
        if queued.shape[-2:] != (2, 45):
            raise ValueError(f"expected queued native state (2,45), got {tuple(queued.shape)}")
        self.latest = queued.detach().clone()
        return queued[..., :42].flatten(start_dim=queued.ndim - 2)

    def get_action(self, condition):
        if self.latest is None:
            raise RuntimeError("base_cond must capture the latest native observation first")
        condition = torch.as_tensor(condition, dtype=torch.float32, device=self.device)
        hint = pickcube_relative_condition(self.latest, self.norm)
        if hint.ndim == 1: hint = hint[None]
        hint = hint[:, None, :].expand(-1, 16, -1)
        sample = torch.randn(condition.shape[0], 16, 7, device=self.device)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                prediction = self.denoiser(sample, timestep, global_cond=condition, hint=hint, enabled=self.control_enabled)
                sample = self.scheduler.step(prediction, timestep, sample).prev_sample
        return sample[:, 1:9]


class PickCubeConstrainTrainer:
    def __init__(self, demo_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST, base_checkpoint=BASE_CKPT,
                 device="cuda", batch_size=1024, total_iters=30000, lr=1e-4, warmup=500,
                 ema_power=0.75, seed=1, run_dir=None, norm=None):
        self.demo_path, self.manifest_path, self.base_checkpoint = Path(demo_path), Path(manifest_path), Path(base_checkpoint)
        self.device, self.batch_size, self.total_iters = device, batch_size, total_iters
        self.lr, self.warmup, self.ema_power, self.seed = lr, warmup, ema_power, seed
        self.run_dir, self.norm = Path(run_dir) if run_dir else None, norm or PickCubeConditionNorm()
        self.model_config, self.policy_contract = _model_config(), _policy_contract()
        self.dataset = self.denoiser = self.ema = self.optimizer = self.lr_scheduler = None
        self.step, self.losses = 0, []

    def load_dataset(self):
        self.dataset = PickCubeObstacleDataset(self.demo_path, self.manifest_path)
        return self.dataset

    def setup(self):
        if self.denoiser is not None: return
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        if str(self.device).startswith("cuda"):
            if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.manual_seed_all(self.seed)
        base_ck = _trusted_base(self.base_checkpoint)
        self.dataset = self.load_dataset()
        agent = build_agent("cpu")
        agent.load_state_dict(base_ck["weights"]["ema"], strict=True)
        self.denoiser = build_constrain_denoiser(self.model_config, device="cpu")
        self.denoiser.base_net.load_state_dict(agent.noise_pred_net.state_dict(), strict=True)
        self.denoiser.to(self.device)
        params = self.denoiser.trainable_parameters()
        self.initial_control_parameters = [p.detach().clone() for p in params]
        self.base_state_sha256 = _state_sha(self.denoiser.base_net.state_dict())
        self.base_info = {"source": str(self.base_checkpoint), "sha256": sha256_file(self.base_checkpoint),
                          "subkey": "weights.ema", "state_sha256": self.base_state_sha256,
                          "updates": 30000, "identity": "Ticket 03 final EMA"}
        self.optimizer = torch.optim.AdamW(params, lr=self.lr, betas=(0.95, 0.999), weight_decay=1e-6)
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import EMAModel
        self.lr_scheduler = get_scheduler("cosine", self.optimizer, num_warmup_steps=self.warmup, num_training_steps=self.total_iters)
        self.ema = EMAModel(parameters=params, power=self.ema_power)
        self.loader = DataLoader(self.dataset, batch_sampler=BatchSampler(RandomSampler(self.dataset), self.batch_size, drop_last=True), num_workers=0)

    def checkpoint_config(self):
        if self.denoiser is None: raise RuntimeError("setup must run before creating checkpoint config")
        return {"model": self.model_config, "policy": self.policy_contract, "normalization": self.norm.to_dict(),
                "diffusion": {"scheduler": "DDPMScheduler", "beta_schedule": "squaredcos_cap_v2", "clip_sample": True, "prediction_type": "epsilon", "num_train_timesteps": 100},
                "base_weights": self.base_info,
                "data_identity": {"h5_sha256": self.dataset.h5_sha256, "manifest_sha256": self.dataset.manifest_sha256,
                                  "selected_class": "obstacle", "selected_episode_ids": list(range(1000,1100)), "episodes": 100},
                "training": {"seed": self.seed, "total_iters": self.total_iters, "batch_size": self.batch_size, "lr": self.lr,
                             "betas": [0.95,0.999], "weight_decay": 1e-6, "warmup": self.warmup, "scheduler": "cosine",
                             "ema_power": self.ema_power, "selected_weights": "final EMA fixed before training", "losses": self.losses}}

    def save_checkpoint(self, path):
        if _state_sha(self.denoiser.base_net.state_dict()) != self.base_state_sha256:
            raise ValueError("frozen PickCube base changed during training")
        config = self.checkpoint_config()
        config["base_weights"]["state_sha16"] = frozen_base_sha16(self.denoiser)
        return save_constrain_checkpoint(path, self.denoiser, self.ema, self.optimizer, self.lr_scheduler,
                                         self.step, {"torch": torch.get_rng_state()}, config,
                                         checkpoint_format=FORMAT)

    def train_batch(self, batch, scheduler):
        obs, actions = batch["observations"].to(self.device), batch["actions"].to(self.device)
        cond = obs[..., :42].flatten(1)
        hint_cond = pickcube_relative_condition(obs, self.norm)
        hint = hint_cond[:, None, :].expand(-1, actions.shape[1], -1)
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, 100, (len(actions),), device=self.device)
        noisy = scheduler.add_noise(actions, noise, timesteps)
        loss = torch.nn.functional.mse_loss(self.denoiser(noisy, timesteps, global_cond=cond, hint=hint), noise)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at update {self.step}")
        self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step(); self.lr_scheduler.step()
        self.ema.step(self.denoiser.trainable_parameters()); self.step += 1
        value = float(loss.item()); self.losses.append(value)
        return value

    def train_steps(self, count):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
        iterator = iter(self.loader)
        for _ in range(count):
            try: batch = next(iterator)
            except StopIteration:
                iterator = iter(self.loader); batch = next(iterator)
            self.train_batch(batch, scheduler)
        return self.losses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-path", type=Path, default=FORMAL_H5)
    parser.add_argument("--manifest", type=Path, default=FORMAL_MANIFEST)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CKPT)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--total-iters", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--ema-power", type=float, default=0.75)
    parser.add_argument("--probe-seconds", type=float, default=0,
                        help="separate non-formal throughput probe, capped at 60 seconds")
    args = parser.parse_args(argv)
    if args.probe_seconds < 0 or args.probe_seconds > 60:
        raise ValueError("throughput probe is bounded to 0..60 seconds")
    run, ckpt = args.run_dir, args.checkpoint_dir
    run.mkdir(parents=True, exist_ok=False)
    if not args.probe_seconds: ckpt.mkdir(parents=True, exist_ok=False)
    trainer = PickCubeConstrainTrainer(args.demo_path, args.manifest, args.base_checkpoint, args.device,
        args.batch_size, args.total_iters, args.lr, args.warmup, args.ema_power, args.seed, run)
    trainer.setup()
    started = time.monotonic()
    equivalence = equivalence_evidence(trainer.denoiser, args.base_checkpoint, args.device)
    if not equivalence["epsilon_matches"] or not equivalence["sampling_matches"]:
        raise RuntimeError(f"zero-control base equivalence failed before training: {equivalence}")
    # Persist complete resolved immutable config before optimizer updates.
    run_config = trainer.checkpoint_config()
    run_config["outputs"] = {"experiment": str(run), "checkpoint": str(ckpt)}
    run_config["pretraining_equivalence"] = equivalence
    run_config["initial_control_state_sha256"] = _state_sha({str(i): p for i, p in enumerate(trainer.initial_control_parameters)})
    run_config["execution_kind"] = "non-formal-throughput-probe" if args.probe_seconds else "formal-training"
    run_config["probe_seconds"] = args.probe_seconds
    (run / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    iterator = iter(trainer.loader)
    with (run / "loss.jsonl").open("w") as stream:
        step_limit = args.total_iters
        probe_started = time.monotonic()
        for _ in range(step_limit):
            if args.probe_seconds and time.monotonic() - probe_started >= args.probe_seconds:
                break
            try: batch = next(iterator)
            except StopIteration:
                iterator = iter(trainer.loader); batch = next(iterator)
            loss = trainer.train_batch(batch, scheduler)
            stream.write(json.dumps({"step": trainer.step, "loss": loss, "lr": trainer.lr_scheduler.get_last_lr()[0]}) + "\n"); stream.flush()
    if args.probe_seconds:
        elapsed = time.monotonic() - probe_started
        result = {"kind": "non-formal-throughput-probe", "updates": trainer.step,
                  "elapsed_seconds": elapsed, "updates_per_second": trainer.step / elapsed,
                  "loss_first": trainer.losses[0], "loss_last": trainer.losses[-1],
                  "all_losses_finite": bool(np.isfinite(trainer.losses).all()),
                  "batch_size": args.batch_size, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if str(args.device).startswith("cuda") else None}
        (run / "probe_summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        return result
    elapsed = time.monotonic() - started
    final = trainer.save_checkpoint(ckpt / "final.pt")
    trained_model, _ = load_pickcube_constrain(final, device=args.device, weights="ema")
    post_epsilon = equivalence_evidence(trained_model, args.base_checkpoint, args.device,
                                        control_enabled=False)
    independent_agent = build_agent(args.device)
    independent_agent.load_state_dict(_trusted_base(args.base_checkpoint)["weights"]["ema"], strict=True)
    independent_agent.eval()
    with torch.no_grad():
        probe = torch.randn(2, 16, 7, device=args.device)
        cond = torch.randn(2, 84, device=args.device)
        hint = torch.tensor([0.25, -0.4, 0.5], device=args.device).view(1, 1, 3).expand(2, 16, 3)
        base_out = independent_agent.noise_pred_net(probe, 23, global_cond=cond)
        disabled_out = trained_model(probe, 23, global_cond=cond, enabled=False)
        enabled_out = trained_model(probe, 23, global_cond=cond, hint=hint, enabled=True)
    enabled_out.square().mean().backward()
    nonzero_control_gradients = {
        prefix: any(p.grad is not None and bool(torch.any(p.grad != 0))
                    for name, p in trained_model.named_parameters() if name.startswith(prefix))
        for prefix in ("control_down_modules.", "control_mid_modules.", "hint_encoder.")
    }
    evidence = {"updates": trainer.step, "all_losses_finite": bool(trainer.losses) and bool(np.isfinite(trainer.losses).all()),
                "first_loss": trainer.losses[0], "last_loss": trainer.losses[-1], "minimum_loss": min(trainer.losses),
                "maximum_loss": max(trainer.losses), "elapsed_seconds": elapsed,
                "base_unchanged": _state_sha(trainer.denoiser.base_net.state_dict()) == trainer.base_state_sha256,
                "zero_residual_weights_moved": any(torch.count_nonzero(p).item() for n,p in trainer.denoiser.named_parameters() if n.startswith("zero_") and n.endswith("weight")),
                "control_parameter_changed": any(not torch.equal(p, initial) for p,initial in zip(trainer.denoiser.trainable_parameters(),trainer.initial_control_parameters)),
                "pretraining_equivalence": equivalence,
                "posttraining_disabled_max_abs_diff": float((base_out-disabled_out).abs().max()),
                "posttraining_disabled_matches_independent_base": bool(torch.allclose(base_out, disabled_out, atol=1e-6, rtol=1e-6)),
                "posttraining_independent_equivalence": post_epsilon,
                "posttraining_enabled_mean_abs_diff": float((enabled_out-disabled_out).abs().mean()),
                "control_gradients_nonzero": nonzero_control_gradients,
                "initial_control_state_sha256": run_config["initial_control_state_sha256"],
                "final_control_state_sha256": _state_sha({str(i): p for i,p in enumerate(trainer.denoiser.trainable_parameters())}),
                "frozen_base_state_sha256": _state_sha(trainer.denoiser.base_net.state_dict()),
                "checkpoint": str(final), "checkpoint_sha256": sha256_file(final)}
    if not all((evidence["all_losses_finite"], evidence["base_unchanged"],
                evidence["zero_residual_weights_moved"], evidence["control_parameter_changed"],
                evidence["posttraining_disabled_matches_independent_base"],
                evidence["posttraining_enabled_mean_abs_diff"] > 0,
                all(evidence["control_gradients_nonzero"].values()),
                evidence["posttraining_independent_equivalence"]["epsilon_matches"],
                evidence["posttraining_independent_equivalence"]["sampling_matches"])):
        raise RuntimeError(f"PickCube ConstrainNet training evidence failed: {evidence}")
    (run / "train_summary.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
