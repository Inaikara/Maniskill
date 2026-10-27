"""PlaceSphere-specific obstacle-conditioned ControlNet training and inference."""

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler

from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
from examples.baselines.constrain.constrain_ckpt import save_constrain_checkpoint
from examples.baselines.constrain.placesphere_inference_sampler import (
    build_inference_scheduler, sample_actions, sample_action_ensemble,
)
from examples.baselines.constrain.placesphere_base_dp import (
    FORMAL_DIR, FORMAL_H5, FORMAL_MANIFEST, TRUSTED_H5_SHA256,
    TRUSTED_MANIFEST_SHA256, TRUSTED_CONTRACT_SHA256, TRUSTED_REPORT_SHA256,
    MODEL_CONFIG as BASE_CONFIG, NAMED_FIELDS, PlaceSphereBaseDataset,
    PlaceSphereBaseDPAdapter, build_agent, sha256_file, R26_MATCHED_H5_SHA256,
    R26_MATCHED_MANIFEST_SHA256, R26_MATCHED_IDENTITY_SHA256,
    R26_DIRECT_IDS,
)
from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_H5_SHA256, MATCHED_MANIFEST_SHA256, MATCHED_VERSION, MATCHED_HOLD8_VERSION,
    MATCHED_HOLD8_OBSTACLE1000_VERSION, MATCHED_HOLD8_OBSTACLE500_VERSION,
    PlaceSphereManifestDataset, load_matched_training_identity,
)
from examples.baselines.constrain.placesphere_action_codec import (
    IDENTITY as IDENTITY_ACTION_ENCODING, CONTROLLER_RANGE,
    decode_to_native, encode_native, profile_sha256, validate_profile,
)

ROOT = Path(__file__).resolve().parents[3]
BASE_CKPT = ROOT / "outputs/ckpt/20261022_193900_placesphere_base_dp/final.pt"
TRUSTED_BASE_SHA256 = "07f93942cfcfd7d28120b0b210fd7c323098140f670bbfce4c9a83ff76b9fb30"
FORMAT = "placesphere_constrain_v1"
MODEL = {"input_dim": 7, "global_cond_dim": 78, "diffusion_step_embed_dim": 64,
         "down_dims": [64, 128, 256], "n_groups": 8, "cond_dim": 3,
         "num_diffusion_iters": 100}
POLICY = {"robot": "panda", "control_mode": "pd_ee_delta_pose", "observation_mode": "state",
          "obs_horizon": 2, "pred_horizon": 16, "act_horizon": 8,
          "base_obs_dim": 39, "raw_obs_dim": 42, "obstacle_obs_dim": 42,
          "act_dim": 7, "named_state_fields": NAMED_FIELDS}


def _state_sha(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def _offsets(fields):
    result, cursor = {}, 0
    for name, width in fields:
        result[name] = (cursor, cursor + width); cursor += width
    return result


_FIELDS = _offsets(NAMED_FIELDS)
_TCP = _FIELDS["extra/tcp_pose"][0]
_OBSTACLE = _FIELDS["extra/placesphere_obstacle"][0]
NORM = {"definition": "[obstacle_x-latest_tcp_x, obstacle_y-latest_tcp_y, cylinder_radius]",
        "axes": "world-parallel XY; relative offset anchored at latest TCP", "units": "meters",
        "scale": [0.5, 0.5, 0.08], "clip": [-1.0, 1.0],
        "tcp_field": "extra/tcp_pose[:2]", "obstacle_field": "extra/placesphere_obstacle[x,y,radius]",
        "named_state_fields": NAMED_FIELDS}


def placesphere_hint(queued, normalization=NORM):
    """Shared train/inference latest-frame hint extraction from schema offsets."""
    x = torch.as_tensor(queued)
    if x.shape[-2:] != (2, 42) or not torch.isfinite(x).all():
        raise ValueError(f"expected finite queued PlaceSphere state (2,42), got {tuple(x.shape)}")
    latest = x[..., -1, :]
    tcp = latest[..., _TCP:_TCP + 2]
    obstacle = latest[..., _OBSTACLE:_OBSTACLE + 3]
    scale = latest.new_tensor(normalization["scale"])
    return ((torch.cat((obstacle[..., :2] - tcp, obstacle[..., 2:3]), -1) / scale)
            .clamp(*normalization["clip"]))


class PlaceSphereObstacleDataset(Dataset):
    """Only trusted formal obstacle episodes 1000..1099; native successful T+1 data."""
    def __init__(self, data_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST, matched_r21b=False,
                 matched_identity=None):
        self.data_path, self.manifest_path = Path(data_path), Path(manifest_path)
        self.matched_r21b = matched_r21b
        self.is_matched = bool(matched_r21b or matched_identity is not None)
        if matched_r21b or matched_identity is not None:
            if matched_r21b and matched_identity is not None:
                raise ValueError("choose one matched dataset identity mode")
            identity = (load_matched_training_identity(
                matched_identity, data_path, manifest_path,
                expected_version=json.loads(Path(matched_identity).read_text()).get("dataset_version"))
                if matched_identity is not None else None)
            self.matched = PlaceSphereManifestDataset(
                data_path, manifest_path, "obstacle",
                expected_h5_sha256=identity["h5_sha256"] if identity else MATCHED_H5_SHA256,
                expected_manifest_sha256=identity["manifest_sha256"] if identity else MATCHED_MANIFEST_SHA256,
                max_release_hold_steps=8 if identity else None,
                expected_version=identity["dataset_version"] if identity else MATCHED_HOLD8_VERSION)
            self.trajectory_ids = self.matched.trajectory_ids
            self.h5_sha256, self.manifest_sha256 = self.matched.h5_sha256, self.matched.manifest_sha256
            self.dataset_version = identity["dataset_version"] if identity else MATCHED_VERSION
            self.release_hold_steps = 8 if identity else 0
            self.training_identity_sha256 = sha256_file(matched_identity) if identity else None
            return
        if sha256_file(self.data_path) != TRUSTED_H5_SHA256 or sha256_file(self.manifest_path) != TRUSTED_MANIFEST_SHA256:
            raise ValueError("formal PlaceSphere data identity is not trusted")
        # Base loader validates complete formal IDs/classes, named 39/42 state, native actions and terminal evidence.
        base = PlaceSphereBaseDataset(self.data_path, self.manifest_path)
        import h5py
        manifest = json.loads(self.manifest_path.read_text())
        records = {int(ep["episode_id"]): ep for ep in manifest["episodes"]}
        self.observations, self.actions, self.slices = [], [], []
        with h5py.File(self.data_path, "r") as h5:
            if len(h5) != 1100: raise ValueError("formal PlaceSphere HDF5 must contain exactly 1100 episodes")
            for eid in range(1000, 1100):
                group, record = h5[f"traj_{eid}"], records[eid]
                if group.attrs.get("kind") != "obstacle" or record.get("kind") != "obstacle":
                    raise ValueError(f"traj_{eid}: selected formal ID is not obstacle class")
                if not record.get("terminated", [0])[-1] or record.get("truncated", [1])[-1]:
                    raise ValueError(f"traj_{eid}: terminal success contract mismatch")
                infos = record.get("infos", [])
                if any(bool(i.get(k, [False])[-1]) for i in infos for k in ("obstacle_collision", "obstacle_crossing")):
                    raise ValueError(f"traj_{eid}: unsafe terminal evidence")
                raw, act = group["raw_obs"][:], group["actions"][:]
                if raw.shape != (len(act) + 1, 42) or act.ndim != 2 or act.shape[1] != 7:
                    raise ValueError(f"traj_{eid}: expected (T+1,42) native states and (T,7) actions")
                if not np.isfinite(raw).all() or not np.isfinite(act).all() or np.abs(act).max(initial=0) > 1.00001:
                    raise ValueError(f"traj_{eid}: non-finite or illegal native data")
                if not bool(group["success"][-1]) or not bool(group["terminated"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"traj_{eid}: missing safe successful terminal evidence")
                obs = torch.as_tensor(raw, dtype=torch.float32); actions = torch.as_tensor(act, dtype=torch.float32)
                ti = len(self.actions); self.observations.append(obs); self.actions.append(actions)
                length = len(actions)
                self.slices.extend((ti, s) for s in range(-1, length - 16 + 14))
        if len(base.trajectory_ids) != 1000: raise ValueError("formal direct dataset audit failed")
        self.trajectory_ids = list(range(1000, 1100))
        self.h5_sha256, self.manifest_sha256 = TRUSTED_H5_SHA256, TRUSTED_MANIFEST_SHA256

    def __len__(self): return len(self.matched) if self.is_matched else len(self.slices)

    def __getitem__(self, index):
        if self.is_matched:
            item = self.matched[index]
            return {"observations": item["raw_observations"], "actions": item["actions"]}
        trajectory, start = self.slices[index]
        obs, actions = self.observations[trajectory], self.actions[trajectory]
        length, end = len(actions), start + 16
        obs_seq = obs[max(0, start):start + 2]
        act_seq = actions[max(0, start):min(end, length)]
        if start < 0:
            obs_seq = torch.cat((obs_seq[:1].repeat(-start, 1), obs_seq))
            act_seq = torch.cat((act_seq[:1].repeat(-start, 1), act_seq))
        if end > length:
            act_seq = torch.cat((act_seq, torch.tensor([0, 0, 0, 0, 0, 0, 1.], dtype=act_seq.dtype).repeat(end - length, 1)))
        return {"observations": obs_seq, "actions": act_seq}


def _base_checkpoint(path):
    path = Path(path)
    if sha256_file(path) != TRUSTED_BASE_SHA256:
        ck = PlaceSphereBaseDPAdapter.validate_checkpoint(path)
        data = ck.get("data_identity", {})
        version = data.get("dataset_version")
        encoding = ck.get("action_encoding", IDENTITY_ACTION_ENCODING)
        validate_profile(encoding)
        if (version == MATCHED_VERSION and
                (data.get("h5_sha256") != MATCHED_H5_SHA256 or
                 data.get("manifest_sha256") != MATCHED_MANIFEST_SHA256)):
            raise ValueError("matched-r21b base dataset identity mismatch")
        if (version == MATCHED_HOLD8_VERSION and
                (data.get("release_hold_steps") != 8 or len(data.get("h5_sha256", "")) != 64
                 or len(data.get("manifest_sha256", "")) != 64)):
            raise ValueError("matched hold-8 base dataset identity mismatch")
        if (version not in (MATCHED_VERSION, MATCHED_HOLD8_VERSION,
                            MATCHED_HOLD8_OBSTACLE1000_VERSION, MATCHED_HOLD8_OBSTACLE500_VERSION) or ck.get("step") != 30000
                or data.get("selected_class") != "direct" or data.get("episodes") != 1000
                or len(data.get("selected_episode_ids", [])) != 1000):
            raise ValueError("base checkpoint is not a trusted matched 30000-step EMA")
        return ck
    ck = PlaceSphereBaseDPAdapter.validate_checkpoint(path, expected_episode_ids=range(1000), trusted_h5_sha256=TRUSTED_H5_SHA256)
    if ck.get("step") != 30000: raise ValueError("base checkpoint is not the completed 30000-update model")
    return ck


def _model_config(): return dict(MODEL)


def _diffusion():
    return {"scheduler": "DDPMScheduler", "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
            "prediction_type": "epsilon", "num_train_timesteps": 100}


def load_placesphere_constrain(checkpoint, device="cpu", weights="ema", base_checkpoint=None):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False); cfg = ck.get("config", {})
    data, base, training = cfg.get("data_identity", {}), cfg.get("base_weights", {}), cfg.get("training", {})
    action_encoding = cfg.get("action_encoding", IDENTITY_ACTION_ENCODING)
    validate_profile(action_encoding)
    if (action_encoding != IDENTITY_ACTION_ENCODING
            and cfg.get("base_weights", {}).get("action_encoding_sha256") != profile_sha256(action_encoding)):
        raise ValueError("PlaceSphere ConstrainNet action encoding metadata is missing or inconsistent")
    version = data.get("dataset_version")
    matched = version in (MATCHED_VERSION, MATCHED_HOLD8_VERSION,
                          MATCHED_HOLD8_OBSTACLE1000_VERSION, MATCHED_HOLD8_OBSTACLE500_VERSION)
    expected_updates = (150000 if version == MATCHED_HOLD8_OBSTACLE500_VERSION else
                        300000 if version == MATCHED_HOLD8_OBSTACLE1000_VERSION else
                        90000 if matched else 30000)
    model_invalid = (ck.get("format") != FORMAT or cfg.get("model") != MODEL
                     or cfg.get("policy") != POLICY or cfg.get("diffusion") != _diffusion()
                     or cfg.get("normalization") != NORM or ck.get("step") != expected_updates)
    if version == MATCHED_VERSION:
        data_invalid = (data.get("h5_sha256") != MATCHED_H5_SHA256
                        or data.get("manifest_sha256") != MATCHED_MANIFEST_SHA256)
    elif version == MATCHED_HOLD8_VERSION:
        data_invalid = (data.get("release_hold_steps") != 8
                        or len(data.get("h5_sha256", "")) != 64
                        or len(data.get("manifest_sha256", "")) != 64)
    elif version in (MATCHED_HOLD8_OBSTACLE1000_VERSION, MATCHED_HOLD8_OBSTACLE500_VERSION):
        data_invalid = (data.get("release_hold_steps") != 8
                        or len(data.get("h5_sha256", "")) != 64
                        or len(data.get("manifest_sha256", "")) != 64
                        or len(data.get("training_identity_sha256", "")) != 64)
        if version == MATCHED_HOLD8_OBSTACLE500_VERSION:
            data_invalid = (data_invalid
                            or data.get("h5_sha256") != R26_MATCHED_H5_SHA256
                            or data.get("manifest_sha256") != R26_MATCHED_MANIFEST_SHA256
                            or data.get("training_identity_sha256") != R26_MATCHED_IDENTITY_SHA256)
    else:
        data_invalid = (data.get("h5_sha256") != TRUSTED_H5_SHA256
                        or data.get("manifest_sha256") != TRUSTED_MANIFEST_SHA256
                        or data.get("frozen_contract_sha256") != TRUSTED_CONTRACT_SHA256
                        or data.get("collection_report_sha256") != TRUSTED_REPORT_SHA256
                        or data.get("selected_episode_ids") != list(range(1000, 1100))
                        or data.get("episodes") != 100)
    class_invalid = (data.get("selected_class") != "obstacle"
                      or data.get("episodes") != (1000 if version == MATCHED_HOLD8_OBSTACLE1000_VERSION else
                                                     500 if version == MATCHED_HOLD8_OBSTACLE500_VERSION else
                                                     300 if matched else 100)
                     or (version == MATCHED_VERSION and data.get("selected_episode_ids") != list(range(1, 600, 2)))
                     or (version == MATCHED_HOLD8_VERSION and len(data.get("selected_episode_ids", [])) != 300)
                       or (version == MATCHED_HOLD8_OBSTACLE1000_VERSION and len(data.get("selected_episode_ids", [])) != 1000)
                       or (version == MATCHED_HOLD8_OBSTACLE500_VERSION and data.get("selected_episode_ids") != [*range(1, 600, 2), *range(1300, 1500)])
                     or (not matched and data.get("selected_episode_ids") != list(range(1000, 1100))))
    base_invalid = (base.get("subkey") != "weights.ema" or base.get("updates") != 30000
                    or (not matched and base.get("sha256") != TRUSTED_BASE_SHA256))
    training_invalid = (training.get("seed") != 1 or training.get("total_iters") != expected_updates
                        or training.get("batch_size") != 1024 or training.get("lr") != 1e-4
                        or training.get("warmup") != 500 or training.get("ema_power") != .75
                        or training.get("selected_weights") != "final EMA fixed before training")
    if model_invalid or data_invalid or class_invalid or base_invalid or training_invalid:
        raise ValueError("PlaceSphere ConstrainNet checkpoint contract mismatch")
    denoiser = build_constrain_denoiser(cfg["model"], device=device); denoiser.load_state_dict(ck["model"], strict=True)
    if weights == "ema":
        from diffusers.training_utils import EMAModel
        ema = EMAModel(parameters=denoiser.trainable_parameters(), power=.75); ema.load_state_dict(ck["ema"]); ema.copy_to(denoiser.trainable_parameters())
    elif weights != "model": raise ValueError("weights must be ema or model")
    if base_checkpoint is not None:
        base_ck = _base_checkpoint(base_checkpoint); agent = build_agent("cpu"); agent.load_state_dict(base_ck["weights"]["ema"], strict=True)
        if base_ck.get("action_encoding", IDENTITY_ACTION_ENCODING) != action_encoding:
            raise ValueError("Base and Constrain action encoding profiles differ")
        base_identity = base_ck.get("data_identity", {})
        if matched and version not in (MATCHED_HOLD8_OBSTACLE1000_VERSION, MATCHED_HOLD8_OBSTACLE500_VERSION) and any(base_identity.get(key) != data.get(key)
                           for key in ("dataset_version", "h5_sha256", "manifest_sha256",
                                        "release_hold_steps", "training_identity_sha256")):
            raise ValueError("Base and Constrain matched dataset identities differ")
        if version == MATCHED_HOLD8_OBSTACLE1000_VERSION and (
                base_identity.get("dataset_version") != MATCHED_HOLD8_VERSION
                or base_identity.get("release_hold_steps") != 8
                or base_identity.get("selected_class") != "direct"
                or base_identity.get("episodes") != 1000):
            raise ValueError("expanded Constrain requires the qualified hold-8 matched direct Base EMA")
        if version == MATCHED_HOLD8_OBSTACLE500_VERSION:
            common_base_invalid = (base_identity.get("release_hold_steps") != 8
                                   or base_identity.get("selected_class") != "direct"
                                   or base_identity.get("episodes") != 1000
                                   or base_identity.get("selected_episode_ids") != R26_DIRECT_IDS)
            if action_encoding == IDENTITY_ACTION_ENCODING:
                # Historical R26 control weights were intentionally initialized
                # from the earlier native-space R23 Base EMA on the same direct IDs.
                base_invalid_for_version = base_identity.get("dataset_version") != MATCHED_HOLD8_VERSION
            else:
                base_invalid_for_version = (
                    base_identity.get("dataset_version") != MATCHED_HOLD8_OBSTACLE500_VERSION
                    or base_identity.get("h5_sha256") != data.get("h5_sha256")
                    or base_identity.get("manifest_sha256") != data.get("manifest_sha256")
                    or base_identity.get("training_identity_sha256") != data.get("training_identity_sha256"))
            if common_base_invalid or base_invalid_for_version:
                raise ValueError("frozen 500-obstacle Constrain requires its qualified matched direct Base EMA")
        if _state_sha(agent.noise_pred_net.state_dict()) != cfg["base_weights"]["state_sha256"]:
            raise ValueError("embedded base differs from independent trusted EMA")
    return denoiser, ck


def equivalence_evidence(denoiser, base_checkpoint=BASE_CKPT, device="cpu", seed=1947, control_enabled=True):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    ck = _base_checkpoint(base_checkpoint); agent = build_agent(device); agent.load_state_dict(ck["weights"]["ema"], strict=True); agent.eval()
    sample_gen = torch.Generator(device="cpu").manual_seed(seed)
    sample = torch.randn((2,16,7), generator=sample_gen).to(device); cond = torch.randn((2,78), generator=sample_gen).to(device)
    t = torch.tensor([13,71], device=device); hint = torch.zeros(2,16,3,device=device)
    with torch.no_grad():
        a = agent.noise_pred_net(sample,t,global_cond=cond)
        b = denoiser(sample,t,global_cond=cond,hint=hint,enabled=control_enabled)
    sched, ref = DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon"), DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    sched.set_timesteps(100); ref.set_timesteps(100)
    x = torch.randn((1,16,7),generator=sample_gen).to(device); y=x.clone(); c=cond[:1]
    g1=torch.Generator(device=device).manual_seed(seed+1); g2=torch.Generator(device=device).manual_seed(seed+1)
    with torch.no_grad():
        for i,j in zip(sched.timesteps,ref.timesteps):
            x=sched.step(denoiser(x,i,global_cond=c,hint=torch.zeros(1,16,3,device=device),enabled=control_enabled),i,x,generator=g1).prev_sample
            y=ref.step(agent.noise_pred_net(y,j,global_cond=c),j,y,generator=g2).prev_sample
    ed, sd=float((a-b).abs().max()),float((x-y).abs().max())
    return {"epsilon_max_abs_diff":ed,"sampling_100step_max_abs_diff":sd,"epsilon_matches":ed<=1e-6,"sampling_matches":sd<=1e-6,"seed":seed}


class PlaceSphereConstrainAdapter:
    def __init__(self, checkpoint, device="cpu", weights="ema", control_enabled=True,
                 base_checkpoint=None, sampler="ddpm", eta=None, ensemble_size=1):
        if ensemble_size not in (1, 4) or (ensemble_size == 4 and sampler != "ddim"):
            raise ValueError("ensemble_size must be 1, or 4 with DDIM")
        self.denoiser,self.checkpoint=load_placesphere_constrain(checkpoint,device,weights,base_checkpoint)
        self.denoiser.eval(); self.device=torch.device(device); self.control_enabled=control_enabled
        self.sampler,self.eta=sampler,eta
        self.ensemble_size=ensemble_size
        self.metadata={"checkpoint_sha256":sha256_file(checkpoint),
                       "sampler":sampler,"eta":eta,"ensemble_size":ensemble_size,
                       "action_encoding":self.checkpoint.get("config", {}).get(
                           "action_encoding", IDENTITY_ACTION_ENCODING)}
        self.obs_horizon,self.pred_horizon,self.act_horizon=2,16,8; self.latest=None
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        training_scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",
            clip_sample=True,prediction_type="epsilon")
        self.scheduler=(build_inference_scheduler(training_scheduler,sampler,eta)
                        if sampler == "ddim" or sampler != "ddpm" or eta is not None
                        else training_scheduler)
    def base_cond(self, queued):
        x=torch.as_tensor(queued,dtype=torch.float32,device=self.device)
        if x.shape[-2:] != (2,42): raise ValueError("expected queued PlaceSphere raw observations (2,42)")
        self.latest=x.detach().clone(); return x[...,:39].flatten(start_dim=x.ndim-2)
    def get_action(self, condition):
        if self.latest is None: raise RuntimeError("base_cond must capture latest observation first")
        c=torch.as_tensor(condition,dtype=torch.float32,device=self.device); hint=placesphere_hint(self.latest)
        if hint.ndim==1: hint=hint[None]
        hint=hint[:,None,:].expand(-1,16,-1)
        if self.sampler == "ddim":
            if self.ensemble_size == 4:
                if c.shape[0] != 1:
                    raise ValueError("four-plan ensemble requires batch size one")
                sample=torch.randn((1,4,16,7),device=self.device)
                return decode_to_native(sample_action_ensemble(self.scheduler,
                    lambda x,t,repeated: self.denoiser(x,t,global_cond=repeated,
                        hint=hint.expand(4,-1,-1),enabled=self.control_enabled),
                    sample,c,eta=self.eta), self.metadata["action_encoding"])
            sample=torch.randn(c.shape[0],16,7,device=self.device)
            return decode_to_native(sample_actions(self.scheduler,
                lambda x,t: self.denoiser(x,t,global_cond=c,hint=hint,enabled=self.control_enabled),
                sample,eta=self.eta)[:,1:9], self.metadata["action_encoding"])
        sample=torch.randn(c.shape[0],16,7,device=self.device)
        self.scheduler.set_timesteps(100)
        with torch.no_grad():
            for t in self.scheduler.timesteps:
                eps=self.denoiser(sample,t,global_cond=c,hint=hint,enabled=self.control_enabled)
                sample=self.scheduler.step(eps,t,sample).prev_sample
        return decode_to_native(sample[:,1:9], self.metadata["action_encoding"])
    @staticmethod
    def map_action(action): return PlaceSphereBaseDPAdapter.map_action_ee_delta_pose(action)


class PlaceSphereConstrainTrainer:
    def __init__(self,demo_path=FORMAL_H5,manifest_path=FORMAL_MANIFEST,base_checkpoint=BASE_CKPT,device="cuda",batch_size=1024,total_iters=30000,lr=1e-4,warmup=500,ema_power=.75,seed=1,run_dir=None,matched_r21b=False,matched_identity=None,action_encoding=IDENTITY_ACTION_ENCODING):
        self.demo_path,self.manifest_path,self.base_checkpoint=Path(demo_path),Path(manifest_path),Path(base_checkpoint)
        self.device,self.batch_size,self.total_iters=device,batch_size,total_iters; self.lr,self.warmup,self.ema_power,self.seed=lr,warmup,ema_power,seed
        self.run_dir=Path(run_dir) if run_dir else None; self.step=0; self.losses=[]; self.matched_r21b=matched_r21b
        self.matched_identity=Path(matched_identity) if matched_identity else None
        self.action_encoding=validate_profile(action_encoding)
    def setup(self):
        random.seed(self.seed);np.random.seed(self.seed);torch.manual_seed(self.seed)
        if str(self.device).startswith("cuda"):
            if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.manual_seed_all(self.seed)
        ck=_base_checkpoint(self.base_checkpoint)
        if ck.get("action_encoding", IDENTITY_ACTION_ENCODING) != self.action_encoding:
            raise ValueError("Constrain training requires a Base checkpoint with the exact same action encoding")
        self.dataset=PlaceSphereObstacleDataset(self.demo_path,self.manifest_path,self.matched_r21b,self.matched_identity)
        agent=build_agent("cpu");agent.load_state_dict(ck["weights"]["ema"],strict=True)
        self.denoiser=build_constrain_denoiser(MODEL,device="cpu");self.denoiser.base_net.load_state_dict(agent.noise_pred_net.state_dict(),strict=True);self.denoiser.to(self.device)
        self.base_state_sha=_state_sha(self.denoiser.base_net.state_dict());self.initial_control=[p.detach().clone() for p in self.denoiser.trainable_parameters()]
        self.base_info={"source":str(self.base_checkpoint),"sha256":sha256_file(self.base_checkpoint),"subkey":"weights.ema","state_sha256":self.base_state_sha,"updates":30000,"action_encoding_sha256":profile_sha256(self.action_encoding)}
        from diffusers.optimization import get_scheduler
        from diffusers.training_utils import EMAModel
        params=self.denoiser.trainable_parameters();self.optimizer=torch.optim.AdamW(params,lr=self.lr,betas=(.95,.999),weight_decay=1e-6)
        self.lr_scheduler=get_scheduler("cosine",self.optimizer,num_warmup_steps=self.warmup,num_training_steps=self.total_iters)
        self.ema=EMAModel(parameters=params,power=self.ema_power)
        self.loader=DataLoader(self.dataset,batch_sampler=BatchSampler(RandomSampler(self.dataset),self.batch_size,drop_last=True),num_workers=0)
    def train_batch(self,batch,scheduler):
        obs=batch["observations"].to(self.device)
        actions=encode_native(batch["actions"].to(self.device), self.action_encoding)
        cond=obs[...,:39].flatten(1);hint=placesphere_hint(obs)[:,None,:].expand(-1,actions.shape[1],-1)
        noise=torch.randn_like(actions);t=torch.randint(0,100,(len(actions),),device=self.device);noisy=scheduler.add_noise(actions,noise,t)
        loss=torch.nn.functional.mse_loss(self.denoiser(noisy,t,global_cond=cond,hint=hint),noise)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at update {self.step}")
        self.optimizer.zero_grad(set_to_none=True);loss.backward();self.optimizer.step();self.lr_scheduler.step();self.ema.step(self.denoiser.trainable_parameters())
        self.step+=1;self.losses.append(float(loss.item()));return self.losses[-1]
    def config(self):
        return {"model":dict(MODEL),"policy":dict(POLICY),"normalization":dict(NORM),"action_encoding":self.action_encoding,"diffusion":_diffusion(),"base_weights":dict(self.base_info),
          "data_identity":{"h5_sha256":self.dataset.h5_sha256,"manifest_sha256":self.dataset.manifest_sha256,"frozen_contract_sha256":None if (self.matched_r21b or self.matched_identity) else TRUSTED_CONTRACT_SHA256,"collection_report_sha256":None if (self.matched_r21b or self.matched_identity) else TRUSTED_REPORT_SHA256,"selected_class":"obstacle","selected_episode_ids":self.dataset.trajectory_ids,"episodes":len(self.dataset.trajectory_ids),**({"dataset_version":self.dataset.dataset_version} if (self.matched_r21b or self.matched_identity) else {}),**({"release_hold_steps":8,"training_identity_sha256":self.dataset.training_identity_sha256} if self.matched_identity else {})},
          "training":{"seed":self.seed,"total_iters":self.total_iters,"batch_size":self.batch_size,"lr":self.lr,"betas":[.95,.999],"weight_decay":1e-6,"warmup":self.warmup,"scheduler":"cosine","ema_power":self.ema_power,"selected_weights":"final EMA fixed before training"}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--demo-path",type=Path,default=FORMAL_H5);p.add_argument("--manifest",type=Path,default=FORMAL_MANIFEST);p.add_argument("--base-checkpoint",type=Path,default=BASE_CKPT);p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--checkpoint-dir",type=Path,required=True);p.add_argument("--device",default="cuda");p.add_argument("--total-iters",type=int,default=30000);p.add_argument("--batch-size",type=int,default=1024);p.add_argument("--seed",type=int,default=1);p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--warmup",type=int,default=500);p.add_argument("--ema-power",type=float,default=.75);p.add_argument("--matched-r21b",action="store_true");p.add_argument("--matched-data-identity",type=Path);p.add_argument("--action-encoding",choices=("identity_native_v1","placesphere_controller_range_v1"),default="identity_native_v1");p.add_argument("--preflight-only",action="store_true");a=p.parse_args(argv)
    a.run_dir.mkdir(parents=True,exist_ok=False)
    if not a.preflight_only: a.checkpoint_dir.mkdir(parents=True,exist_ok=False)
    if a.matched_r21b and a.matched_data_identity: raise ValueError("choose one matched dataset identity mode")
    if a.matched_r21b and (sha256_file(a.demo_path) != MATCHED_H5_SHA256 or sha256_file(a.manifest) != MATCHED_MANIFEST_SHA256): raise ValueError("matched PlaceSphere r21b data identity mismatch")
    if a.matched_data_identity:
        load_matched_training_identity(a.matched_data_identity,a.demo_path,a.manifest,
            expected_version=json.loads(a.matched_data_identity.read_text()).get("dataset_version"))
    action_encoding=CONTROLLER_RANGE if a.action_encoding==CONTROLLER_RANGE["name"] else IDENTITY_ACTION_ENCODING
    trainer=PlaceSphereConstrainTrainer(a.demo_path,a.manifest,a.base_checkpoint,a.device,a.batch_size,a.total_iters,a.lr,a.warmup,a.ema_power,a.seed,a.run_dir,a.matched_r21b,a.matched_data_identity,action_encoding);trainer.setup()
    if a.preflight_only:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
        iterator=iter(trainer.loader); losses=[]
        for _ in range(2):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(trainer.loader);batch=next(iterator)
            losses.append(trainer.train_batch(batch,scheduler))
        report={"status":"preflight_complete","updates":trainer.step,"all_losses_finite":bool(np.isfinite(losses).all()),"losses":losses,
                "base_unchanged":_state_sha(trainer.denoiser.base_net.state_dict())==trainer.base_state_sha,
                "control_parameters_changed":any(not torch.equal(p,q) for p,q in zip(trainer.denoiser.trainable_parameters(),trainer.initial_control)),
                "action_encoding":trainer.action_encoding,"data_h5_sha256":trainer.dataset.h5_sha256,
                "data_manifest_sha256":trainer.dataset.manifest_sha256}
        if not (report["all_losses_finite"] and report["base_unchanged"] and report["control_parameters_changed"]):
            raise RuntimeError(f"PlaceSphere Constrain preflight failed: {report}")
        (a.run_dir/"preflight.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2),flush=True)
        return report
    start=time.monotonic();pre=equivalence_evidence(trainer.denoiser,a.base_checkpoint,a.device)
    if not pre["epsilon_matches"] or not pre["sampling_matches"]: raise RuntimeError(f"pretraining equivalence failed: {pre}")
    cfg=trainer.config();cfg["outputs"]={"experiment":str(a.run_dir),"checkpoint":str(a.checkpoint_dir)};cfg["pretraining_equivalence"]=pre
    source_files=[Path(__file__),Path(__file__).with_name("placesphere_base_dp.py"),Path(__file__).with_name("placesphere_manifest_dataset.py"),Path(__file__).with_name("placesphere_action_codec.py"),Path(__file__).with_name("constrain_denoiser.py"),Path(__file__).with_name("constrain_ckpt.py"),Path(__file__).with_name("placesphere_obstacle.py"),ROOT/"examples/baselines/diffusion_policy/train.py",ROOT/"examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py"]
    snapshot=a.run_dir/"source_snapshot";snapshot.mkdir()
    cfg["source_hashes"]={}
    for path in source_files:
        rel=path.relative_to(ROOT);target=snapshot/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        cfg["source_hashes"][str(rel)]=sha256_file(path)
    (a.run_dir/"run_config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon");iterator=iter(trainer.loader)
    with (a.run_dir/"loss.jsonl").open("w") as stream:
        for _ in range(a.total_iters):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(trainer.loader);batch=next(iterator)
            value=trainer.train_batch(batch,scheduler);stream.write(json.dumps({"step":trainer.step,"loss":value,"lr":trainer.lr_scheduler.get_last_lr()[0]})+"\n");stream.flush()
    from examples.baselines.constrain.constrain_ckpt import frozen_base_sha16
    cfg=trainer.config();cfg["base_weights"]["state_sha16"]=frozen_base_sha16(trainer.denoiser)
    path=save_constrain_checkpoint(a.checkpoint_dir/"final.pt",trainer.denoiser,trainer.ema,trainer.optimizer,trainer.lr_scheduler,trainer.step,{"torch":torch.get_rng_state()},cfg,checkpoint_format=FORMAT)
    model,_=load_placesphere_constrain(path,a.device,"ema");post=equivalence_evidence(model,a.base_checkpoint,a.device,control_enabled=False)
    base_ck=_base_checkpoint(a.base_checkpoint);agent=build_agent(a.device);agent.load_state_dict(base_ck["weights"]["ema"],strict=True)
    x=torch.randn(2,16,7,device=a.device);c=torch.randn(2,78,device=a.device);h=torch.tensor([.25,-.4,.5],device=a.device).view(1,1,3).expand(2,16,3)
    with torch.no_grad(): base=agent.noise_pred_net(x,23,global_cond=c);disabled=model(x,23,global_cond=c,enabled=False);enabled=model(x,23,global_cond=c,hint=h)
    model.zero_grad(set_to_none=True);model(x,23,global_cond=c,hint=h).square().mean().backward()
    grads={prefix:any(q.grad is not None and bool(torch.any(q.grad!=0)) for n,q in model.named_parameters() if n.startswith(prefix)) for prefix in ("control_down_modules.","control_mid_modules.","hint_encoder.")}
    diff={"steps":trainer.step,"all_losses_finite":bool(trainer.losses) and bool(np.isfinite(trainer.losses).all()),"first_loss":trainer.losses[0],"last_loss":trainer.losses[-1],"min_loss":min(trainer.losses),"max_loss":max(trainer.losses),"elapsed_seconds":time.monotonic()-start,"base_unchanged":_state_sha(trainer.denoiser.base_net.state_dict())==trainer.base_state_sha,"pretraining_equivalence":pre,"disabled_posttraining_equivalence":post,"disabled_independent_max_abs_diff":float((base-disabled).abs().max()),"enabled_disabled_mean_abs_diff":float((enabled-disabled).abs().mean()),"control_gradients_nonzero":grads,"control_parameters_changed":any(not torch.equal(p,q) for p,q in zip(trainer.denoiser.trainable_parameters(),trainer.initial_control)),"checkpoint_sha256":sha256_file(path)}
    if not (diff["all_losses_finite"] and diff["base_unchanged"] and post["epsilon_matches"] and post["sampling_matches"] and diff["disabled_independent_max_abs_diff"]<=1e-6 and diff["enabled_disabled_mean_abs_diff"]>0 and all(grads.values()) and diff["control_parameters_changed"]): raise RuntimeError(f"PlaceSphere training evidence failed: {diff}")
    (a.run_dir/"train_summary.json").write_text(json.dumps(diff,indent=2)+"\n");print(json.dumps(diff,indent=2),flush=True)


if __name__ == "__main__": main()
