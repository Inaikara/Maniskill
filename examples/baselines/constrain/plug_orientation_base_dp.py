"""From-scratch official Diffusion Policy Base for 46-D fixed Plug scenes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from examples.baselines.constrain.base_dp import BasePolicyArgs, _PolicyEnvStub
from examples.baselines.constrain.plug_orientation_data import (
    _native_episode_ok, _pair_native_match, _term_as_timeout, sha256_file,
)
from examples.baselines.constrain.plug_orientation_scenes import (
    PlugSceneCatalog, SPLITS, restore_plug_scene,
)

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DP = ROOT / "examples/baselines/diffusion_policy"
OBS_DIM, ACT_DIM = 46, 7
OBS_HORIZON, ACT_HORIZON, PRED_HORIZON = 2, 8, 16
FORMAT = "plug_orientation_fixed_base_dp_v1"
CHECKPOINT_STEPS = (20_000, 40_000, 60_000, 80_000, 100_000)


def _canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _fixed_candidate_is_allowed(seed, target_yaw):
    return 2026121000 <= int(seed) <= 2026122499 and float(target_yaw) == 0.0


def build_agent(device="cpu"):
    if str(OFFICIAL_DP) not in sys.path: sys.path.insert(0, str(OFFICIAL_DP))
    import train as official_train
    official_train.device = torch.device(device)
    Agent = official_train.Agent
    args = BasePolicyArgs(OBS_DIM, ACT_DIM, OBS_HORIZON, ACT_HORIZON, PRED_HORIZON)
    args.unet_dims, args.diffusion_step_embed_dim, args.n_groups = [64, 128, 256], 64, 8
    return Agent(_PolicyEnvStub(OBS_DIM, ACT_DIM, OBS_HORIZON), args).to(device)


def resolved_config():
    return {"format": FORMAT, "seed": 1, "updates": 100000, "batch_size": 1024,
            "optimizer": "AdamW", "lr": 1e-4, "betas": [0.95, 0.999],
            "weight_decay": 1e-6, "warmup_updates": 500, "scheduler": "cosine",
            "ema_power": 0.75, "observation_dim": OBS_DIM, "action_dim": ACT_DIM,
            "action_bounds": [-1.0, 1.0],
            "obs_horizon": OBS_HORIZON, "act_horizon": ACT_HORIZON,
            "pred_horizon": PRED_HORIZON, "unet_dims": [64, 128, 256],
            "diffusion_step_embed_dim": 64, "n_groups": 8,
            "diffusion": {"type": "DDPMScheduler", "train_timesteps": 100,
                          "beta_schedule": "squaredcos_cap_v2", "prediction_type": "epsilon",
                          "clip_sample": True}, "normalization": "identity",
             "checkpoint_steps": list(CHECKPOINT_STEPS), "num_workers": 0}


def _validate_contract_config(contract):
    policy = contract["base_policy"]
    cfg = resolved_config()
    expected = {
        "seed": policy["training_seed"], "updates": policy["updates"],
        "batch_size": policy["batch_size"], "lr": policy["learning_rate"],
        "betas": policy["betas"], "weight_decay": policy["weight_decay"],
        "warmup_updates": policy["warmup_updates"], "ema_power": policy["ema_power"],
        "observation_dim": contract["data_collection"]["observation_dim"],
        "action_dim": contract["data_collection"]["action_dim"],
        "obs_horizon": policy["horizons"]["observation"],
        "act_horizon": policy["horizons"]["action"],
        "pred_horizon": policy["horizons"]["prediction"],
        "unet_dims": policy["architecture"]["unet_dims"],
        "diffusion_step_embed_dim": policy["architecture"]["diffusion_step_embed_dim"],
        "n_groups": policy["architecture"]["n_groups"],
        "checkpoint_steps": policy["checkpoints"],
    }
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError("Base DP implementation config differs from the frozen contract")
    if (cfg["optimizer"] != policy["optimizer"]
            or cfg["scheduler"] != policy["learning_rate_schedule"]):
        raise ValueError("optimizer or LR schedule differs from the frozen contract")
    diffusion = policy["diffusion"]
    if (cfg["diffusion"]["type"] != diffusion["scheduler"]
            or cfg["diffusion"]["train_timesteps"] != diffusion["train_timesteps"]
            or cfg["diffusion"]["beta_schedule"] != diffusion["beta_schedule"]
            or cfg["diffusion"]["prediction_type"] != diffusion["prediction_type"]
            or cfg["diffusion"]["clip_sample"] != diffusion["clip_sample"]
            or "identity" not in policy["normalization"].lower()
            or cfg["action_bounds"] != [-1.0, 1.0]):
        raise ValueError("Base DP diffusion or normalization differs from frozen contract")
    return cfg


def _flat_obs(node):
    """Flatten ManiSkill state dictionary in the official observation order."""
    from mani_skill.utils import common
    # ManiSkill's flatten_state_dict accepts batched dict tensors and preserves
    # registration-defined key order; convert HDF5 leaves to CPU tensors first.
    def tensors(value):
        if isinstance(value, dict): return {k: tensors(v) for k, v in value.items()}
        return torch.as_tensor(np.asarray(value))
    result = common.flatten_state_dict(tensors(node), use_torch=True)
    return result.detach().cpu().numpy().reshape(-1)


class PlugBaseDataset(Dataset):
    """Official 2/16 trajectory windows without crossing episode boundaries."""
    def __init__(self, data_path, manifest_path, expected_episodes=1000, *, _test_mode=False):
        self.obs, self.actions, self.windows = [], [], []
        metadata = json.loads(Path(manifest_path).read_text())
        rows = metadata["episodes"]
        ids = [int(row["episode_id"]) for row in rows]
        if len(ids) != expected_episodes or ids != list(range(expected_episodes)):
            raise ValueError("Plug training manifest must enumerate ordered 1000 episodes")
        self.selected_seeds = [int(row.get("source_candidate_seed", row.get("episode_seed", -1))) for row in rows]
        if not _test_mode and len(set(self.selected_seeds)) != expected_episodes:
            raise ValueError("Plug training source candidates must have unique identities")
        with h5py.File(data_path, "r") as h5:
            if len(h5) != expected_episodes: raise ValueError("Plug HDF5 episode count mismatch")
            for episode in ids:
                group = h5[f"traj_{episode}"]
                action = np.asarray(group["actions"][:], dtype=np.float32)
                if "obs" not in group: raise ValueError(f"traj_{episode} lacks observations")
                obs_group = group["obs"]
                if isinstance(obs_group, h5py.Dataset): obs = np.asarray(obs_group[:], dtype=np.float32)
                else:
                    obs = np.stack([_flat_obs({k: _read_h5_obs(v, i) for k, v in obs_group.items()})
                                    for i in range(len(action) + 1)])
                if obs.shape != (len(action) + 1, OBS_DIM) or action.ndim != 2 or action.shape[1] != ACT_DIM:
                    raise ValueError(f"traj_{episode}: expected finite T+1 46-D obs and T 7-D actions")
                if not np.isfinite(obs).all() or not np.isfinite(action).all() or np.max(np.abs(action), initial=0) > 1.00001:
                    raise ValueError(f"traj_{episode}: non-finite/out-of-range native data")
                if not _test_mode:
                    row = rows[episode]
                    seed = int(row.get("source_candidate_seed", -1))
                    if not _fixed_candidate_is_allowed(seed, row.get("target_yaw_rad", math.nan)):
                        raise ValueError(f"traj_{episode}: source is outside the frozen fixed-yaw candidate pool")
                    source_path, replay_path = Path(row["source_h5"]), Path(row["replay_h5"])
                    case_path = Path(row["case_json"])
                    if (sha256_file(source_path) != row["source_h5_sha256"]
                            or sha256_file(replay_path) != row["replay_h5_sha256"]
                            or sha256_file(case_path) != row["case_json_sha256"]):
                        raise ValueError(f"traj_{episode}: source/replay/case provenance hash mismatch")
                    case = json.loads(case_path.read_text())
                    if (not _native_episode_ok(source_path, case)[0]
                            or not _native_episode_ok(replay_path, case, replay=True)[0]
                            or not _pair_native_match(source_path, replay_path)):
                        raise ValueError(f"traj_{episode}: source/replay native eligibility no longer passes")
                    actual_actions_hash = hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()
                    if actual_actions_hash != row["source_actions_sha256"]:
                        raise ValueError(f"traj_{episode}: consolidated action bytes do not match source identity")
                    with h5py.File(source_path, "r") as source:
                        source_group = source["traj_0"]
                        if not _h5_group_equal(group, source_group):
                            raise ValueError(f"traj_{episode}: consolidated native trajectory differs from source")
                self.obs.append(torch.from_numpy(obs.copy()))
                self.actions.append(torch.from_numpy(action.copy()))
                length = len(action)
                if length < PRED_HORIZON:
                    raise ValueError(f"traj_{episode} is shorter than the official 16-action window")
                # Match the official episode-local window range exactly.
                for start in range(-(OBS_HORIZON - 1),
                                   length - PRED_HORIZON + (PRED_HORIZON - OBS_HORIZON)):
                    self.windows.append((episode, start))

    def __len__(self): return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        obs, action = self.obs[episode], self.actions[episode]
        obs_ids = [max(0, start), max(0, start + 1)]
        obs_window = obs[obs_ids]
        end = start + PRED_HORIZON
        action_window = action[max(0, start):end]
        if start < 0:
            action_window = torch.cat((action[0].repeat(-start, 1), action_window))
        if end > len(action):
            pad = torch.cat((torch.zeros(ACT_DIM - 1), action[-1, -1:]))
            action_window = torch.cat((action_window, pad.repeat(end - len(action), 1)))
        return {"observations": obs_window, "actions": action_window}


def _read_h5_obs(node, index):
    if isinstance(node, h5py.Group): return {k: _read_h5_obs(v, index) for k, v in node.items()}
    return np.asarray(node[index])


def _h5_group_equal(left, right):
    if set(left) != set(right): return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, h5py.Group):
            if not isinstance(b, h5py.Group) or not _h5_group_equal(a, b): return False
        elif isinstance(b, h5py.Group) or not np.array_equal(a[:], b[:]):
            return False
    return True


class PlugBaseAdapter:
    def __init__(self, checkpoint, device="cpu"):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = resolved_config()
        if (payload.get("format") != FORMAT or payload.get("observation_dim") != OBS_DIM
                or payload.get("action_dim") != ACT_DIM or payload.get("config") != cfg
                or payload.get("checkpoint_role") != "validation_candidate"
                or int(payload.get("update", -1)) not in CHECKPOINT_STEPS):
            raise ValueError("Plug BaseDP checkpoint contract mismatch")
        provenance = payload.get("dataset_provenance")
        if not isinstance(provenance, dict):
            raise ValueError("Plug BaseDP checkpoint lacks dataset provenance")
        contract_path = Path(__file__).with_name("plug_orientation_contract.json")
        if provenance.get("contract_sha256") != sha256_file(contract_path):
            raise ValueError("Plug BaseDP checkpoint was trained under a different contract")
        for path_key, hash_key in (("dataset_h5_path", "dataset_h5_sha256"),
                                   ("dataset_json_path", "dataset_json_sha256"),
                                   ("dataset_manifest_path", "dataset_manifest_sha256"),
                                   ("catalog_set_path", "catalog_set_sha256")):
            path, digest = provenance.get(path_key), provenance.get(hash_key)
            if not path or not digest or not Path(path).is_file() or sha256_file(path) != digest:
                raise ValueError(f"Plug BaseDP checkpoint {path_key} provenance mismatch")
        catalog_hashes = provenance.get("frozen_scene_catalog_sha256")
        catalog_set = json.loads(Path(provenance["catalog_set_path"]).read_text())
        for split, digest in (catalog_hashes or {}).items():
            declared = catalog_set.get("catalogs", {}).get(split, {})
            manifest_path = Path(provenance["catalog_set_path"]).parent / split / "manifest.json"
            if (not manifest_path.is_file() or sha256_file(manifest_path) != digest
                    or declared.get("manifest_sha256") != digest):
                raise ValueError(f"Plug BaseDP checkpoint catalog provenance mismatch: {split}")
        self.device = torch.device(device)
        self.agent = build_agent(device)
        self.agent.load_state_dict(payload["ema_agent"], strict=True)
        self.agent.eval()
        self.metadata = {"format": FORMAT, "checkpoint": str(checkpoint),
                         "checkpoint_sha256": sha256_file(checkpoint), "weights": "ema_agent",
                         "observation_dim": OBS_DIM, "action_dim": ACT_DIM,
                         "obs_horizon": OBS_HORIZON, "act_horizon": ACT_HORIZON,
                         "pred_horizon": PRED_HORIZON}

    def get_action(self, observations):
        observations = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if observations.shape != (1, OBS_HORIZON, OBS_DIM):
            raise ValueError(f"expected exactly (1, 2, 46), got {tuple(observations.shape)}")
        if not torch.isfinite(observations).all(): raise ValueError("policy input contains non-finite values")
        with torch.no_grad(): result = self.agent.get_action(observations)
        if (result.shape != (1, ACT_HORIZON, ACT_DIM) or not torch.isfinite(result).all()
                or torch.max(torch.abs(result)).item() > 1.00001):
            raise ValueError(f"official sampler must return finite (1, 8, 7), got {tuple(result.shape)}")
        return result


def _seed_everything(seed, device):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _save_checkpoint(path, agent, ema, optimizer, scheduler, update, config, provenance,
                     checkpoint_role="validation_candidate", format_id=FORMAT):
    raw = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    ema.store(agent.parameters())
    ema.copy_to(agent.parameters())
    try:
        ema_state = {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}
    finally:
        ema.restore(agent.parameters())
    payload = {"format": format_id, "observation_dim": OBS_DIM, "action_dim": ACT_DIM,
               "update": int(update), "checkpoint_role": checkpoint_role,
               "agent": raw, "ema_agent": ema_state,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "ema": ema.state_dict(), "torch_rng_state": torch.get_rng_state(),
               "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                "config": config, **provenance, "dataset_provenance": provenance}
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _archive_base_sources(exp_dir):
    relative_paths = [
        "examples/baselines/constrain/plug_orientation_base_dp.py",
        "examples/baselines/constrain/plug_orientation_scenes.py",
        "examples/baselines/constrain/plug_orientation_data.py",
        "examples/baselines/constrain/plug_orientation.py",
        "examples/baselines/constrain/plug_orientation_ee_expert.py",
        "examples/baselines/constrain/plug_orientation_contract.json",
        "examples/baselines/constrain/base_dp.py",
        "examples/baselines/diffusion_policy/train.py",
        "examples/baselines/diffusion_policy/diffusion_policy/evaluate.py",
        "examples/baselines/diffusion_policy/diffusion_policy/utils.py",
        "examples/baselines/diffusion_policy/diffusion_policy/make_env.py",
        "examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py",
    ]
    archive = Path(exp_dir) / "source_archive"
    archive.mkdir()
    hashes = {}
    for relative in relative_paths:
        source = ROOT / relative
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_file(source)
        destination.write_bytes(source.read_bytes())
        if sha256_file(destination) != digest or sha256_file(source) != digest:
            raise RuntimeError(f"Base source archive identity mismatch: {relative}")
        hashes[relative] = digest
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "torch", "diffusers", "gymnasium", "h5py")}
    return {"source_sha256": hashes, "runtime_versions": versions}


def train(data_path, manifest_path, exp_dir, ckpt_dir, *, _updates=100000, _batch_size=1024,
          device=None, agent_factory=build_agent, catalog_dirs=None, _test_mode=False):
    """Run only the frozen seed-1/100k production run; undersized args are test-only seam."""
    if _updates != 100000 or _batch_size != 1024:
        if not _test_mode:
            raise ValueError("production Plug Base budget is frozen at 100000 updates, batch 1024")
    if _updates < 1: raise ValueError("updates must be positive")
    contract_path = Path(__file__).with_name("plug_orientation_contract.json")
    contract = json.loads(contract_path.read_text())
    frozen_config = _validate_contract_config(contract)
    exp_dir, ckpt_dir = Path(exp_dir), Path(ckpt_dir)
    exp_dir.mkdir(parents=True, exist_ok=False); ckpt_dir.mkdir(parents=True, exist_ok=False)
    training_archive = _archive_base_sources(exp_dir)
    cfg = dict(frozen_config); cfg["updates"], cfg["batch_size"] = _updates, _batch_size
    _seed_everything(1, device or ("cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = PlugBaseDataset(data_path, manifest_path, _test_mode=_test_mode)
    loader = DataLoader(dataset, batch_size=_batch_size, shuffle=True, drop_last=True, num_workers=0)
    agent = agent_factory(device).to(device)
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=500, num_training_steps=_updates)
    ema = EMAModel(agent.parameters(), power=0.75)
    dataset_manifest = Path(manifest_path).with_name("dataset_manifest.json")
    if not dataset_manifest.is_file() and not _test_mode:
        raise ValueError("production training requires the collector's dataset_manifest.json")
    if dataset_manifest.is_file():
        dataset_identity = json.loads(dataset_manifest.read_text())
        if (dataset_identity.get("accepted_count") != 1000
                or len(dataset_identity.get("selected_seeds", [])) != 1000
                or dataset_identity.get("selected_seeds") != dataset.selected_seeds
                or dataset_identity.get("training_h5_sha256") != sha256_file(data_path)
                or dataset_identity.get("training_json_sha256") != sha256_file(manifest_path)):
            raise ValueError("dataset manifest does not bind this exact 1000-episode training package")
        attempts = dataset_identity.get("attempts", [])
        if (len(attempts) != dataset_identity.get("candidate_attempt_count")
                or len(attempts) > 1500
                or [int(row.get("seed", -1)) for row in attempts]
                   != list(range(2026121000, 2026121000 + len(attempts)))):
            raise ValueError("candidate attempts are not the frozen ordered seed denominator")
        eligible_seeds = [int(row["seed"]) for row in attempts if row.get("accepted") is True]
        if eligible_seeds != dataset.selected_seeds or len(eligible_seeds) != 1000:
            raise ValueError("training episodes are not the first 1000 ordered eligible candidates")
    if catalog_dirs is None and not _test_mode:
        raise ValueError("freeze all scene catalogs before training and pass catalog_dirs")
    catalog_hashes = {}
    catalog_set_hash = None
    if catalog_dirs is not None:
        if set(catalog_dirs) != set(SPLITS): raise ValueError("all four preregistered catalog splits are required")
        catalog_parents = {Path(value).resolve().parent for value in catalog_dirs.values()}
        if len(catalog_parents) != 1: raise ValueError("all frozen catalogs must share one catalog-set root")
        catalog_set_path = next(iter(catalog_parents)) / "catalog_set.json"
        catalog_set = json.loads(catalog_set_path.read_text())
        if catalog_set.get("format") != "plug-orientation-catalog-set-v1":
            raise ValueError("catalog set is not the frozen Plug catalog format")
        catalog_set_hash = sha256_file(catalog_set_path)
        for split, directory in catalog_dirs.items():
            catalog = PlugSceneCatalog(directory)
            expected_count = len(SPLITS[split])
            if catalog.manifest.get("split") != split or len(catalog) != expected_count:
                raise ValueError(f"invalid frozen catalog {split}")
            catalog_hashes[split] = sha256_file(Path(directory) / "manifest.json")
            declared = catalog_set.get("catalogs", {}).get(split, {})
            if (declared.get("manifest_sha256") != catalog_hashes[split]
                    or declared.get("scene_count") != expected_count):
                raise ValueError(f"catalog set identity mismatch for {split}")
    provenance = {"contract_sha256": sha256_file(Path(__file__).with_name("plug_orientation_contract.json")),
                  "dataset_h5_sha256": sha256_file(data_path), "dataset_json_sha256": sha256_file(manifest_path),
                  "dataset_h5_path": str(Path(data_path).resolve()),
                  "dataset_json_path": str(Path(manifest_path).resolve()),
                  "dataset_manifest_path": str(dataset_manifest.resolve()) if dataset_manifest.exists() else None,
                  "dataset_manifest_sha256": sha256_file(dataset_manifest) if dataset_manifest.exists() else None,
                  "dataset_episode_count": 1000, "teacher": "native_ee_teacher_v4",
                  "frozen_scene_catalog_sha256": catalog_hashes,
                  "catalog_set_sha256": catalog_set_hash,
                  "catalog_set_path": str(catalog_set_path.resolve()) if catalog_dirs is not None else None,
                  **training_archive}
    (exp_dir / "resolved_config.json").write_text(json.dumps({**cfg, **provenance}, indent=2, sort_keys=True) + "\n")
    start = time.monotonic(); update = 0
    try:
        with _term_as_timeout("SIGTERM received during Base training"):
            while update < _updates:
                saw_batch = False
                for batch in loader:
                    if update >= _updates: break
                    if time.monotonic() - start >= 21600:
                        raise TimeoutError("Base training exceeded the registered 21600-second wall cap")
                    saw_batch = True
                    observations, actions = batch["observations"].to(device), batch["actions"].to(device)
                    loss = agent.compute_loss(observations, actions)
                    if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at update {update + 1}")
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in agent.parameters()):
                        raise FloatingPointError(f"non-finite gradient at update {update + 1}")
                    optimizer.step(); scheduler.step(); ema.step(agent.parameters()); update += 1
                    if update % 100 == 0:
                        with (exp_dir / "training.jsonl").open("a") as log:
                            log.write(json.dumps({"update": update, "loss": float(loss.detach().cpu()),
                                                  "lr": optimizer.param_groups[0]["lr"],
                                                  "elapsed_seconds": time.monotonic() - start}, allow_nan=False) + "\n")
                    if update in CHECKPOINT_STEPS:
                        _save_checkpoint(ckpt_dir / f"checkpoint_{update:06d}.pt", agent, ema, optimizer, scheduler, update, cfg, provenance)
                if not saw_batch:
                    raise ValueError("training dataloader is empty")
    except BaseException as exc:
        failure_checkpoint = None
        try:
            failure_checkpoint = ckpt_dir / "failure_state.pt"
            _save_checkpoint(failure_checkpoint, agent, ema, optimizer, scheduler, update, cfg, provenance,
                             checkpoint_role="failure_snapshot")
        except BaseException as checkpoint_exc:
            failure_checkpoint = None
            checkpoint_error = f"{type(checkpoint_exc).__name__}: {checkpoint_exc}"
        else:
            checkpoint_error = None
        (exp_dir / "training_failure.json").write_text(json.dumps({"update": update,
            "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - start,
            "failure_state_checkpoint": str(failure_checkpoint) if failure_checkpoint else None,
            "failure_state_checkpoint_sha256": sha256_file(failure_checkpoint) if failure_checkpoint else None,
            "checkpoint_error": checkpoint_error}, indent=2) + "\n")
        raise
    return {"updates": update, "elapsed_seconds": time.monotonic() - start,
            "exp_dir": str(exp_dir), "ckpt_dir": str(ckpt_dir)}


def select_checkpoint(rows):
    if len(rows) != len(CHECKPOINT_STEPS) or sorted(int(r["update"]) for r in rows) != list(CHECKPOINT_STEPS):
        raise ValueError("selection only allows the five preregistered checkpoints")
    return max(rows, key=lambda row: (int(row["successes"]), int(row["update"])))


def validation_authorizes_test(successes, total=50):
    if total != 50: raise ValueError("validation gate denominator is frozen at 50")
    return int(successes) >= 40


def _validate_evaluation_identity(checkpoint, catalog_dir):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("format") != FORMAT or payload.get("config") != resolved_config()
            or payload.get("checkpoint_role") != "validation_candidate"
            or int(payload.get("update", -1)) not in CHECKPOINT_STEPS):
        raise ValueError("evaluation checkpoint format/config/update identity mismatch")
    provenance = payload.get("dataset_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("evaluation checkpoint lacks frozen dataset provenance")
    if provenance.get("contract_sha256") != sha256_file(Path(__file__).with_name("plug_orientation_contract.json")):
        raise ValueError("evaluation checkpoint contract identity mismatch")
    for path_key, hash_key in (("dataset_h5_path", "dataset_h5_sha256"),
                               ("dataset_json_path", "dataset_json_sha256"),
                               ("dataset_manifest_path", "dataset_manifest_sha256"),
                               ("catalog_set_path", "catalog_set_sha256")):
        path, digest = provenance.get(path_key), provenance.get(hash_key)
        if not path or not digest or not Path(path).is_file() or sha256_file(path) != digest:
            raise ValueError(f"evaluation checkpoint {path_key} identity mismatch")
    root = Path(provenance["catalog_set_path"]).parent
    catalog = PlugSceneCatalog(catalog_dir)
    split = catalog.manifest.get("split")
    manifest_sha = sha256_file(Path(catalog_dir) / "manifest.json")
    expected = provenance.get("frozen_scene_catalog_sha256", {}).get(split)
    if expected != manifest_sha:
        raise ValueError("evaluation catalog is not the checkpoint's frozen catalog split")
    catalog_set = json.loads(Path(provenance["catalog_set_path"]).read_text())
    if (catalog_set.get("catalogs", {}).get(split, {}).get("manifest_sha256") != manifest_sha
            or Path(catalog_dir).resolve() != (root / split).resolve()):
        raise ValueError("evaluation catalog bundle identity mismatch")
    return payload


def lock_validation_selection(rows, output_path):
    """Persist the preregistered five-checkpoint decision before any test opening."""
    if Path(output_path).exists(): raise FileExistsError("refusing to overwrite validation selection")
    if any(int(row.get("episodes", -1)) != 50 or not 0 <= int(row.get("successes", -1)) <= 50
           or not Path(row.get("checkpoint", "")).is_file()
           or row.get("catalog_split") != "fixed_validation"
           or not row.get("catalog_manifest_sha256")
           or not Path(row.get("evaluation_summary_path", "")).is_file()
           or sha256_file(row["evaluation_summary_path"]) != row.get("evaluation_summary_sha256")
           for row in rows):
        raise ValueError("every preregistered validation cohort must contain 50 scenes and a checkpoint")
    catalog_hashes = {row["catalog_manifest_sha256"] for row in rows}
    if len(catalog_hashes) != 1: raise ValueError("validation checkpoints used different frozen catalogs")
    checkpoint_hashes = [sha256_file(row["checkpoint"]) for row in rows]
    if len(set(checkpoint_hashes)) != len(rows):
        raise ValueError("five validation checkpoints must have distinct content identities")
    if any(Path(row["checkpoint"]).name != f"checkpoint_{int(row['update']):06d}.pt"
           for row in rows):
        raise ValueError("validation candidates must use the five registered checkpoint filenames")
    for row in rows:
        summary = json.loads(Path(row["evaluation_summary_path"]).read_text())
        if (summary.get("episodes") != 50 or summary.get("successes") != int(row["successes"])
                or summary.get("checkpoint_sha256") != sha256_file(row["checkpoint"])
                or summary.get("catalog_manifest_sha256") != row["catalog_manifest_sha256"]
                or summary.get("execution_stride", ACT_HORIZON) != ACT_HORIZON):
            raise ValueError("validation candidate counts are not bound to its complete evaluation summary")
    chosen = select_checkpoint(rows)
    selected_sha = sha256_file(chosen["checkpoint"])
    payload = {"format": "plug-orientation-validation-selection-v1",
               "selection_rule": "maximum fixed-validation native successes; ties latest update",
               "validation_successes": int(chosen["successes"]), "validation_episodes": 50,
               "selected_update": int(chosen["update"]), "selected_checkpoint": str(chosen["checkpoint"]),
               "selected_checkpoint_sha256": selected_sha,
               "validation_catalog_manifest_sha256": next(iter(catalog_hashes)),
               "candidate_results": [{"update": int(row["update"]),
                     "checkpoint": str(row["checkpoint"]), "checkpoint_sha256": sha256_file(row["checkpoint"]),
                     "successes": int(row["successes"]), "episodes": int(row.get("episodes", 50)),
                     "catalog_manifest_sha256": row["catalog_manifest_sha256"],
                     "catalog_split": row.get("catalog_split"),
                     "evaluation_summary_path": str(Path(row["evaluation_summary_path"]).resolve()),
                     "evaluation_summary_sha256": row["evaluation_summary_sha256"]}
                    for row in sorted(rows, key=lambda item: int(item["update"]))]}
    payload["selection_sha256"] = _canonical_sha256(payload)
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return payload


def evaluate_checkpoint(checkpoint, catalog_dir, output_dir, *, policy_factory=None, env_factory=None,
                        _authorized_test=False, _selection_artifact=None, _test_mode=False,
                        execution_stride=ACT_HORIZON, scene_indices=None,
                        _validated_payload=None, _prevalidated_selection=False):
    """Fixed-validation only. Full test evaluation is deliberately gated separately."""
    import gymnasium as gym
    catalog = PlugSceneCatalog(catalog_dir)
    expected_split = "fixed_base_test" if _authorized_test else "fixed_validation"
    expected_count = 100 if _authorized_test else 50
    if catalog.manifest["split"] != expected_split or len(catalog) != expected_count:
        raise ValueError("evaluation catalog does not match the authorized frozen split")
    if not isinstance(execution_stride, int) or not 1 <= execution_stride <= ACT_HORIZON:
        raise ValueError("execution stride must be an integer in [1, 8]")
    if _authorized_test and execution_stride != ACT_HORIZON:
        raise ValueError("fixed Base test authorization requires the original execution stride 8")
    indices = list(range(len(catalog))) if scene_indices is None else list(scene_indices)
    if (not indices or len(set(indices)) != len(indices)
            or any(not isinstance(index, int) or not 0 <= index < len(catalog) for index in indices)):
        raise ValueError("scene indices must be unique valid catalog indices")
    if _authorized_test and sorted(indices) != list(range(expected_count)):
        raise ValueError("fixed Base qualification requires the complete frozen 100-scene catalog")
    checkpoint_payload = (_validated_payload if _validated_payload is not None else
                          (None if _test_mode else _validate_evaluation_identity(checkpoint, catalog_dir)))
    if _authorized_test and _prevalidated_selection and _validated_payload is None:
        raise ValueError("prevalidated test selection requires a validated checkpoint payload")
    if _authorized_test and not _prevalidated_selection:
        if _selection_artifact is None:
            raise ValueError("fixed Base test requires a validation-selection artifact")
        selection = json.loads(Path(_selection_artifact).read_text())
        selection_payload = {key: value for key, value in selection.items() if key != "selection_sha256"}
        candidate_rows = selection.get("candidate_results", [])
        expected_choice = max(candidate_rows, key=lambda row: (int(row["successes"]), int(row["update"]))) if candidate_rows else {}
        if (selection.get("format") != "plug-orientation-validation-selection-v1"
                or int(selection.get("validation_successes", -1)) < 40
                or int(selection.get("validation_episodes", -1)) != 50
                or selection.get("selected_checkpoint_sha256") != sha256_file(checkpoint)
                or selection.get("selection_sha256") != _canonical_sha256(selection_payload)
                or sorted(int(row.get("update", -1)) for row in candidate_rows) != list(CHECKPOINT_STEPS)
                or expected_choice.get("update") != selection.get("selected_update")
                or expected_choice.get("checkpoint") != selection.get("selected_checkpoint")
                or expected_choice.get("checkpoint_sha256") != selection.get("selected_checkpoint_sha256")
                or any(sha256_file(row["checkpoint"]) != row["checkpoint_sha256"] for row in candidate_rows)):
            raise ValueError("fixed test checkpoint is not validation-selected and authorized")
        if any(row.get("catalog_split") != "fixed_validation"
               or Path(row.get("checkpoint", "")).name != f"checkpoint_{int(row['update']):06d}.pt"
               for row in candidate_rows):
            raise ValueError("selection does not identify the five frozen fixed-validation candidates")
        validation_sha = checkpoint_payload["dataset_provenance"]["frozen_scene_catalog_sha256"].get("fixed_validation")
        if (selection.get("validation_catalog_manifest_sha256") != validation_sha
                or len({row["checkpoint_sha256"] for row in candidate_rows}) != 5
                or any(row.get("catalog_manifest_sha256") != validation_sha for row in candidate_rows)):
            raise ValueError("selection does not bind the current frozen fixed-validation catalog and five distinct checkpoints")
        for row in candidate_rows:
            summary_path = Path(row.get("evaluation_summary_path", ""))
            if (not summary_path.is_file()
                    or sha256_file(summary_path) != row.get("evaluation_summary_sha256")):
                raise ValueError("validation evaluation summary identity mismatch")
            summary = json.loads(summary_path.read_text())
            if (summary.get("episodes") != 50 or summary.get("successes") != int(row["successes"])
                    or summary.get("checkpoint_sha256") != row["checkpoint_sha256"]
                    or summary.get("catalog_manifest_sha256") != validation_sha
                    or summary.get("execution_stride", ACT_HORIZON) != ACT_HORIZON):
                raise ValueError("validation candidate counts do not match the complete evaluation summary")
        for row in candidate_rows:
            candidate = (checkpoint_payload if row["checkpoint_sha256"] == selection["selected_checkpoint_sha256"]
                         else torch.load(row["checkpoint"], map_location="cpu", weights_only=False))
            if (candidate.get("format") != FORMAT
                    or candidate.get("checkpoint_role") != "validation_candidate"
                    or int(candidate.get("update", -1)) != int(row["update"])
                    or candidate.get("config") != resolved_config()
                    or candidate.get("contract_sha256") != sha256_file(Path(__file__).with_name("plug_orientation_contract.json"))):
                raise ValueError("validation selection includes a checkpoint outside the current Base contract")
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    if env_factory is None:
        from examples.baselines.constrain import plug_orientation  # register fixed task
        env_factory = lambda: gym.make(plug_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                                       sim_backend="physx_cpu", max_episode_steps=1000)
    policy = policy_factory(checkpoint) if policy_factory else PlugBaseAdapter(checkpoint)
    env = env_factory()
    records = []; started = time.monotonic()
    try:
        with _term_as_timeout("SIGTERM received during policy evaluation"):
            for index in indices:
                if time.monotonic() - started >= (18000 if _authorized_test else 3600):
                    raise TimeoutError("registered evaluation wall-clock cap exhausted")
                scene = catalog.load(index); obs = restore_plug_scene(env, scene).detach().cpu().numpy().reshape(-1)
                seed = 1 + index; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                history = [obs.copy(), obs.copy()]; n = 0; success = False; failure = "control_cap"
                states, observations, actions = [], [], []
                event_path = output_dir / f"episode_{index:03d}.events.jsonl"
                while n < 600:
                    if time.monotonic() - started >= (18000 if _authorized_test else 3600):
                        raise TimeoutError("registered evaluation wall-clock cap exhausted")
                    chunk = policy.get_action(torch.as_tensor(np.asarray(history)[None], dtype=torch.float32,
                                                              device=getattr(policy, "device", "cpu")))[0]
                    if chunk.shape != (8, 7) or not torch.isfinite(chunk).all(): raise ValueError("invalid action chunk")
                    for action in chunk[:min(execution_stride, 600 - n)]:
                        action_np = action.detach().cpu().numpy()
                        if not np.isfinite(action_np).all() or np.max(np.abs(action_np), initial=0) > 1.00001:
                            raise ValueError("policy action is non-finite or outside native [-1, 1] bounds")
                        step = env.step(action_np)
                        obs_next, _, terminated, truncated, info = step
                        observations.append(history[-1].tolist()); actions.append(action_np.tolist())
                        native_state = _jsonable(env.unwrapped.get_state_dict())
                        states.append(native_state)
                        flat_next = obs_next.detach().cpu().numpy().reshape(-1)
                        if flat_next.shape != (OBS_DIM,) or not np.isfinite(flat_next).all():
                            raise ValueError("environment returned an invalid 46-D policy observation")
                        event = {"control": n + 1, "policy_input_46d": history[-1].tolist(),
                                 "action_7d": action_np.tolist(), "observation_46d": flat_next.tolist(),
                                 "native_state": native_state, "native_info": _jsonable(info)}
                        with event_path.open("a") as events:
                            events.write(json.dumps(event, allow_nan=False) + "\n")
                        history = [history[-1], flat_next]
                        n += 1; success = bool(info.get("success", False))
                        if success: failure = "success"
                        if terminated or truncated or success:
                            failure = "success" if success else str(info.get("fail", "native_termination"))
                            break
                    if success or terminated or truncated: break
                rec = {"index": index, "seed": int(scene[0]["seed"]), "policy_seed": seed,
                       "success": success, "failure_type": failure, "executed_controls": n,
                       "actions": actions, "observations": observations, "states": states}
                records.append(rec)
                with (output_dir / "episodes.jsonl").open("a") as journal:
                    journal.write(json.dumps(rec, allow_nan=False) + "\n")
    except BaseException as exc:
        in_progress = None
        try:
            event_file = output_dir / f"episode_{index:03d}.events.jsonl"
            in_progress = {"scene_index": index, "controls_journaled": len(event_file.read_text().splitlines())
                           if event_file.exists() else 0}
        except UnboundLocalError:
            pass
        (output_dir / "partial_summary.json").write_text(json.dumps({
            "completed_episodes": len(records), "completed_records": records,
            "in_progress": in_progress, "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started}, indent=2, allow_nan=False) + "\n")
        raise
    finally: env.close()
    summary = {"episodes": len(records), "successes": sum(r["success"] for r in records),
               "failure_types": {kind: sum(row["failure_type"] == kind for row in records)
                                 for kind in sorted({row["failure_type"] for row in records})},
               "control_mode": "pd_ee_delta_pose", "observation_dim": OBS_DIM,
               "action_dim": ACT_DIM, "observation_horizon": OBS_HORIZON,
                "action_horizon": ACT_HORIZON, "prediction_horizon": PRED_HORIZON,
                "execution_stride": execution_stride, "scene_indices": indices,
               "episode_control_cap": 600, "environment_time_limit": 1000,
               "elapsed_seconds": time.monotonic() - started,
               "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
               "catalog_manifest_sha256": sha256_file(Path(catalog_dir) / "manifest.json"),
               "records": records}
    if _authorized_test:
        summary["base_qualification_gate"] = {"required_successes": 80, "denominator": 100,
                                               "passed": summary["successes"] >= 80}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def evaluate_fixed_base_test(checkpoint, catalog_dir, selection_artifact, output_dir,
                             *, policy_factory=None, env_factory=None):
    """Open only the fixed 100-scene Base test after validation selection is locked."""
    selection = json.loads(Path(selection_artifact).read_text())
    if int(selection.get("validation_successes", -1)) < 40 or int(selection.get("validation_episodes", -1)) != 50:
        raise ValueError("fixed test is gated on at least 40/50 frozen validation successes")
    checkpoint_sha = sha256_file(checkpoint)
    if selection.get("selected_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("selection artifact does not lock this checkpoint identity")
    catalog = PlugSceneCatalog(catalog_dir)
    if catalog.manifest["split"] != "fixed_base_test":
        raise ValueError("Ticket 08 Base qualification only opens the fixed_base_test catalog")
    return evaluate_checkpoint(checkpoint, catalog_dir, output_dir, policy_factory=policy_factory,
                               env_factory=env_factory, _authorized_test=True,
                               _selection_artifact=selection_artifact)


def evaluate_registered_checkpoints(checkpoint_dir, validation_catalog, output_root,
                                      *, policy_factory=None, env_factory=None):
    """Evaluate only the five checkpoint steps preregistered before training."""
    paths = [Path(checkpoint_dir) / f"checkpoint_{step:06d}.pt" for step in CHECKPOINT_STEPS]
    if any(not path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(f"missing preregistered checkpoint: {missing}")
    if len({sha256_file(path) for path in paths}) != len(paths):
        raise ValueError("the five preregistered checkpoints must have distinct content identities")
    for step, path in zip(CHECKPOINT_STEPS, paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("format") != FORMAT or payload.get("checkpoint_role") != "validation_candidate"
                or int(payload.get("update", -1)) != step or payload.get("config") != resolved_config()):
            raise ValueError(f"checkpoint is not the registered update {step}")
    Path(output_root).mkdir(parents=True, exist_ok=False)
    rows = []; started = time.monotonic()
    for step in CHECKPOINT_STEPS:
        if time.monotonic() - started >= 18000:
            raise TimeoutError("five-checkpoint validation stage exceeded 18000 seconds")
        path = Path(checkpoint_dir) / f"checkpoint_{step:06d}.pt"
        summary = evaluate_checkpoint(path, validation_catalog, Path(output_root) / f"validation_{step:06d}",
                                      policy_factory=policy_factory, env_factory=env_factory)
        summary_path = Path(output_root) / f"validation_{step:06d}" / "summary.json"
        rows.append({"update": step, "successes": summary["successes"],
                     "episodes": summary["episodes"], "checkpoint": str(path),
                     "catalog_manifest_sha256": summary["catalog_manifest_sha256"],
                     "catalog_split": "fixed_validation",
                     "evaluation_summary_path": str(summary_path),
                     "evaluation_summary_sha256": sha256_file(summary_path)})
    selection = lock_validation_selection(rows, Path(output_root) / "validation_selection.json")
    return {"candidates": rows, "selection": selection,
            "test_authorized": validation_authorizes_test(selection["validation_successes"])}


def _jsonable(value):
    if isinstance(value, dict): return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if hasattr(value, "detach"): return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect-fixed")
    collect.add_argument("--dataset-dir", required=True); collect.add_argument("--exp-dir", required=True)
    freeze = sub.add_parser("freeze-catalogs")
    freeze.add_argument("--output-root", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data", required=True); train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--catalog-root", required=True); train_parser.add_argument("--exp-dir", required=True)
    train_parser.add_argument("--ckpt-dir", required=True); train_parser.add_argument("--device")
    validation = sub.add_parser("evaluate-validation")
    validation.add_argument("--ckpt-dir", required=True); validation.add_argument("--catalog", required=True)
    validation.add_argument("--output-root", required=True)
    test = sub.add_parser("evaluate-fixed-test")
    test.add_argument("--checkpoint", required=True); test.add_argument("--catalog", required=True)
    test.add_argument("--selection", required=True); test.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "collect-fixed":
        from examples.baselines.constrain.plug_orientation_data import collect_fixed_dataset
        collect_fixed_dataset(args.dataset_dir, args.exp_dir)
    elif args.command == "freeze-catalogs":
        from examples.baselines.constrain.plug_orientation_scenes import freeze_evaluation_catalogs
        freeze_evaluation_catalogs(args.output_root)
    elif args.command == "train":
        catalog_root = Path(args.catalog_root)
        catalog_dirs = {name: catalog_root / name for name in SPLITS}
        train(args.data, args.manifest, args.exp_dir, args.ckpt_dir,
              device=args.device, catalog_dirs=catalog_dirs)
    elif args.command == "evaluate-validation":
        evaluate_registered_checkpoints(args.ckpt_dir, args.catalog, args.output_root)
    else:
        evaluate_fixed_base_test(args.checkpoint, args.catalog, args.selection, args.output_dir)


if __name__ == "__main__":
    main()
