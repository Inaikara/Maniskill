"""Forward randomized-yaw Constrain study using the frozen Plug Base v1 EMA."""
from __future__ import annotations

import hashlib
import argparse
import json
import math
import random
import os
import signal
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.constrain import plug_orientation_base_dp as base
from examples.baselines.constrain.constrain_denoiser import ConstrainDenoiser

ROOT = Path(__file__).resolve().parents[3]
BASE_CHECKPOINT = ROOT / "outputs/ckpt/20261119_194656_plug_orientation_base/checkpoint_080000.pt"
BASE_SHA256 = "ceeb68b4838ced6e6ffa43646bcc3cfc47538469e02264c4bc80a2542a33d130"
BASE_EMA_SHA256 = "ab97b69e3bf62d77de030dd745ed2f416691589a22b67c20efdefac791b2c083"
FORMAT = "plug_orientation_constrain_v1"
CONDITION_DIM = 2
CHECKPOINT_STEPS = (5000, 10000, 15000, 20000, 25000, 30000)
REGISTERED_RANDOM_EPISODE_COUNTS = {100, 200, 300, 500}
HELDOUT_CONTRACT = Path(__file__).with_name("plug_orientation_heldout_contract.json")
HELDOUT_CHECKPOINT = ROOT / "outputs/ckpt/20261122_042359_plug_orientation_constrain_random500/checkpoint_030000.pt"
HELDOUT_SELECTION = ROOT / "outputs/exp/20261122_045531_plug_constrain_random500_validation/validation_selection.json"
HELDOUT_CATALOG_ROOT = ROOT / "outputs/exp/20261119_184307_plug_orientation_catalogs"
HELDOUT_OUTPUT_ROOT = ROOT / "outputs/exp/20261123_heldout_plug_orientation"
RANDOM100_DATASET = ROOT / "demos/PlugCharger-orientation-random-100_20261120_041640"
PARENT_RANDOM100_MANIFEST_SHA256 = "5e0841998157fd383699fb87905c4d4869ac41ab9a00f222f6536afa2bf11198"
PARENT_RANDOM100_H5_SHA256 = "2df8ffa9b95c27e2ef9f7a471edefb68f73c8f6123769e9858eb9fa53f2fe26c"
PARENT_RANDOM100_JSON_SHA256 = "77e9fed3070ad326b0323ace70b6b7015ad4394bf374a71193487685db4f6c0f"
PARENT_RANDOM100_PREREGISTRATION_SHA256 = "4ad2bb78a3bb2966e90c47c2d17cbde2b6c213e4b5b47f8504fad2ce9777819b"
PARENT_RANDOM100_FORWARD_CONTRACT = ROOT / "outputs/exp/20261120_043914_plug_orientation_constrain_random100/plug_orientation_adaptation_contract.json"
PARENT_RANDOM100_FORWARD_CONTRACT_SHA256 = "60caac0271f9a26a68df5702fb8ba1ef6d104511543cf2b2223df5d2d039ea42"
RANDOM200_DATASET = ROOT / "demos/PlugCharger-orientation-random-200_20261120_121208"
PARENT_RANDOM200_MANIFEST_SHA256 = "3f96a3ff54072d64023dc057271614cd09225deb5356dafb8d51785562ac304b"
PARENT_RANDOM200_H5_SHA256 = "eeed1e0170702b89de1a5505f2ae863fccaf9c6c467eeec9a549ed14c510c210"
PARENT_RANDOM200_JSON_SHA256 = "a5245c8de32d103cffe7c60a1f9a409badfa42c87c0d6488931c6c951f1ff269"
PARENT_RANDOM200_PREREGISTRATION_SHA256 = "0ebde081173b847df5c67656cec22efb0577fe1893279ab31648240f61290200"
RANDOM300_DATASET = ROOT / "demos/PlugCharger-orientation-random-300_20261120_173932"
PARENT_RANDOM300_MANIFEST_SHA256 = "e32b6914a08d41adfb7cb2e87f0af458cd4d035880373a4b51cf3fdfc175b6dc"
PARENT_RANDOM300_H5_SHA256 = "95ede016098066b535208ec2569d4c092186a242e4627206306d44137dc2fa0f"
PARENT_RANDOM300_JSON_SHA256 = "129dcf74369eb95dad2e2960877024afe9f831a47bd36576dea4ce1411f1a42f"
MODEL_CONFIG = {"input_dim": 7, "global_cond_dim": 92, "diffusion_step_embed_dim": 64,
                "down_dims": [64, 128, 256], "n_groups": 8,
                "cond_dim": CONDITION_DIM, "num_diffusion_iters": 100}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _tensor_state_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_random_episode_count(payload):
    """Accept only the legacy 100 pool and the two sealed expansion lineages."""
    count = payload.get("random_episode_count", 100)
    training = payload.get("config", {}).get("training", {})
    configured = training.get("random_episode_count", count)
    if count != configured or count not in REGISTERED_RANDOM_EPISODE_COUNTS:
        raise ValueError("checkpoint has an unregistered or internally inconsistent random demo count")
    if count == 200:
        expected_contract = sha256_file(Path(__file__).with_name(
            "plug_orientation_adaptation_expansion200_contract.json"))
        digest_fields = (payload.get("random_h5_sha256"), payload.get("random_manifest_sha256"))
        if (payload.get("expansion_contract_sha256") != expected_contract
                or payload.get("parent_dataset_manifest_sha256") != PARENT_RANDOM100_MANIFEST_SHA256
                or training.get("random_episode_count") != 200
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in digest_fields)):
            raise ValueError("200-demo checkpoint lacks exact expansion/dataset lineage")
    if count == 300:
        expected_contract = sha256_file(Path(__file__).with_name(
            "plug_orientation_adaptation_expansion300_contract.json"))
        digest_fields = (payload.get("random_h5_sha256"), payload.get("random_manifest_sha256"))
        if (payload.get("expansion_contract_sha256") != expected_contract
                or payload.get("parent_dataset_manifest_sha256") != PARENT_RANDOM200_MANIFEST_SHA256
                or training.get("random_episode_count") != 300
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in digest_fields)):
            raise ValueError("300-demo checkpoint lacks exact expansion/dataset lineage")
    if count == 500:
        expected_contract = sha256_file(Path(__file__).with_name(
            "plug_orientation_adaptation_expansion500_contract.json"))
        digest_fields = (payload.get("random_h5_sha256"), payload.get("random_manifest_sha256"))
        if (payload.get("expansion_contract_sha256") != expected_contract
                or payload.get("parent_dataset_manifest_sha256") != PARENT_RANDOM300_MANIFEST_SHA256
                or training.get("random_episode_count") != 500
                or any(not isinstance(value, str) or len(value) != 64
                       or any(char not in "0123456789abcdef" for char in value)
                       for value in digest_fields)):
            raise ValueError("500-demo checkpoint lacks exact expansion/dataset lineage")
    return count


def _load_frozen_base(path=BASE_CHECKPOINT):
    path = Path(path)
    if sha256_file(path) != BASE_SHA256:
        raise ValueError("frozen Base checkpoint byte identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != base.FORMAT or payload.get("update") != 80000
            or payload.get("checkpoint_role") != "validation_candidate"
            or payload.get("config") != base.resolved_config()
            or _tensor_state_sha256(payload["ema_agent"]) != BASE_EMA_SHA256):
        raise ValueError("frozen Base v1 EMA identity mismatch")
    agent = base.build_agent("cpu")
    agent.load_state_dict(payload["ema_agent"], strict=True)
    if _tensor_state_sha256(agent.state_dict()) != BASE_EMA_SHA256:
        raise ValueError("loaded frozen Base EMA tensor identity mismatch")
    agent.eval()
    return agent, payload


def orientation_hint(observation, target_yaw):
    """Return live charger-vs-goal world-X yaw error as sin/cos."""
    from transforms3d.quaternions import quat2mat

    obs = np.asarray(observation, dtype=np.float64)
    target_yaw = float(target_yaw)
    if obs.shape != (46,) or not np.isfinite(obs).all() or not math.isfinite(target_yaw):
        raise ValueError("hint requires one finite 46-D observation and finite target yaw")
    quaternion = obs[28:32]
    if np.linalg.norm(quaternion) <= 1e-12:
        raise ValueError("charger quaternion is degenerate")
    axis = quat2mat(quaternion)[:, 0]
    charger_yaw = math.atan2(float(axis[1]), float(axis[0]))
    error = math.atan2(math.sin(charger_yaw - target_yaw), math.cos(charger_yaw - target_yaw))
    return np.asarray([math.sin(error), math.cos(error)], dtype=np.float32)


def build_adaptation_contract():
    """Build additive provenance; this never mutates the historical v1 contract."""
    _, payload = _load_frozen_base()
    return {
        "format": "plug_orientation_adaptation_contract_v1",
        "user_amendment": "不用严格限定base的成功率 只要是按照官方要求训练的就好 重点看我们的方法有没有优化 就按照第一版的base模型继续吧",
        "base_absolute_gate_waived": True,
        "base_qualification_claim": False,
        "historical_base_validation": {"successes": 30, "episodes": 50, "selected_update": 80000},
        "frozen_base": {"path": str(BASE_CHECKPOINT.relative_to(ROOT)), "sha256": BASE_SHA256,
                        "format": payload["format"], "update": 80000, "weights": "ema_agent",
                        "validation_successes": 30, "validation_episodes": 50,
                        "ema_agent_tensor_sha256": BASE_EMA_SHA256},
        "method_gates": {"fixed_retention_loss_pp_max": 5, "randomized_gain_pp_min": 20},
        "training": {"seed": 1, "updates": 30000, "batch_size": 1024,
                     "random_windows_per_batch": 512, "fixed_windows_per_batch": 512,
                     "optimizer": "AdamW", "lr": 1e-4, "betas": [0.95, 0.999],
                     "weight_decay": 1e-6, "warmup_updates": 500,
                     "scheduler": "cosine", "ema_power": 0.75,
                     "checkpoint_steps": list(CHECKPOINT_STEPS)},
        "data": {"initial_random_episodes": 100, "candidate_seed_range_inclusive": [2026123000, 2026123149],
                  "max_candidates": 150, "official_success": {"position_m_max": 0.005,
                  "full_quaternion_angle_rad_max": 0.2}, "yaw_plan": "default_rng(seed+1703), uniform[-pi/8,+pi/8]",
                  "hint_shuffle_seed": 1703, "expansion_reserved_seeds_inclusive": [2026123000, 2026123599],
                  "max_future_episodes": 300, "source_and_replay_control_cap_each": 90000,
                  "wall_clock_cap_seconds": 5400},
        "evaluation": {"validation_only_splits": ["random_validation", "fixed_validation"],
                       "final_splits": ["random_final_test", "fixed_base_test"],
                       "hint_modes": ["correct", "fixed_standard", "shuffled"],
                       "execution_stride": 8, "episode_control_cap": 600},
    }


def write_adaptation_contract(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    contract = build_adaptation_contract()
    path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return contract


class PlugAdaptationDataset(Dataset):
    """Unmodified native trajectories with hints derived from current observations."""
    def __init__(self, h5_path, manifest_path, expected_episodes):
        self.obs, self.actions, self.hints, self.windows = [], [], [], []
        registration = json.loads(Path(manifest_path).read_text())
        manifest = registration
        manifest_data_path = Path(manifest_path)
        if expected_episodes in {200, 300, 500} and "episodes" not in registration:
            manifest_data_path = Path(registration["training_json_path"])
            manifest = json.loads(manifest_data_path.read_text())
        rows = manifest["episodes"]
        if expected_episodes == 200:
            _validate_random200_lineage(h5_path, manifest_data_path, registration, manifest)
            if (registration.get("protocol") != "plug_orientation_random200_expansion_v1"
                    or registration.get("status") != "complete"
                    or registration.get("accepted_count") != 200
                    or registration.get("effective_episode_count") != 200
                    or registration.get("parent_dataset_manifest_sha256") != PARENT_RANDOM100_MANIFEST_SHA256
                    or registration.get("expansion_contract_sha256") != sha256_file(
                        Path(__file__).with_name("plug_orientation_adaptation_expansion200_contract.json"))
                    or [row.get("lineage", {}).get("kind") for row in rows[:100]] !=
                       ["immutable_parent_episode"] * 100):
                raise ValueError("200-episode dataset lacks the registered parent-first expansion lineage")
            if any(row["lineage"].get("parent_episode_id") != index
                   or row["lineage"].get("parent_manifest_sha256") != PARENT_RANDOM100_MANIFEST_SHA256
                   for index, row in enumerate(rows[:100])):
                raise ValueError("200-episode parent lineage is not the frozen ordered parent set")
            if (registration.get("training_h5_sha256") != sha256_file(h5_path)
                    or Path(registration.get("training_h5_path", "")).resolve() != Path(h5_path).resolve()
                    or registration.get("training_json_sha256") != sha256_file(manifest_data_path)):
                raise ValueError("200-episode dataset payload hashes disagree with its manifest")
        elif expected_episodes == 300:
            _validate_random300_lineage(h5_path, manifest_data_path, registration, manifest)
            if (registration.get("protocol") != "plug_orientation_random300_expansion_v1"
                    or registration.get("status") != "complete"
                    or registration.get("accepted_count") != 300
                    or registration.get("effective_episode_count") != 300
                    or registration.get("parent_dataset_manifest_sha256") != PARENT_RANDOM200_MANIFEST_SHA256
                    or registration.get("expansion_contract_sha256") != sha256_file(
                        Path(__file__).with_name("plug_orientation_adaptation_expansion300_contract.json"))
                    or [row.get("lineage", {}).get("kind") for row in rows[:200]] !=
                       ["immutable_parent_episode"] * 200):
                raise ValueError("300-episode dataset lacks the registered parent-first expansion lineage")
            if (registration.get("training_h5_sha256") != sha256_file(h5_path)
                    or Path(registration.get("training_h5_path", "")).resolve() != Path(h5_path).resolve()
                    or registration.get("training_json_sha256") != sha256_file(manifest_data_path)):
                raise ValueError("300-episode dataset payload hashes disagree with its manifest")
        elif expected_episodes == 500:
            _validate_random500_lineage(h5_path, manifest_data_path, registration, manifest)
            if (registration.get("training_h5_sha256") != sha256_file(h5_path)
                    or Path(registration.get("training_h5_path", "")).resolve() != Path(h5_path).resolve()
                    or registration.get("training_json_sha256") != sha256_file(manifest_data_path)):
                raise ValueError("500-episode dataset payload hashes disagree with its manifest")
        if len(rows) != expected_episodes:
            raise ValueError(f"expected exactly {expected_episodes} adaptation trajectories")
        with h5py.File(h5_path, "r") as h5:
            if len(h5) != expected_episodes:
                raise ValueError("HDF5 trajectory count differs from manifest")
            for episode, row in enumerate(rows):
                group = h5[f"traj_{episode}"]
                actions = np.asarray(group["actions"][:], dtype=np.float32)
                if isinstance(group["obs"], h5py.Dataset):
                    obs = np.asarray(group["obs"][:], dtype=np.float32)
                else:
                    obs = np.stack([base._flat_obs(base._read_h5_obs(group["obs"], i))
                                    for i in range(len(actions) + 1)])
                if (obs.shape != (len(actions) + 1, 46) or actions.ndim != 2
                        or actions.shape[1] != 7 or len(actions) < 16
                        or not np.isfinite(obs).all() or not np.isfinite(actions).all()
                        or np.max(np.abs(actions), initial=0) > 1.00001):
                    raise ValueError(f"traj_{episode} is not finite native T/T+1 46-D/7-D data")
                target_values = np.asarray(group["env_states/plug_orientation_target"][:])
                target = np.asarray(row.get("target_yaw_rad"), dtype=np.float64)
                if target.ndim != 0 or not np.isfinite(target) or target_values.shape != (len(actions) + 1, 4):
                    raise ValueError(f"traj_{episode} lacks aligned actual target-yaw state")
                if not np.allclose(target_values[:, 0], float(target), atol=1e-7, rtol=0):
                    raise ValueError(f"traj_{episode} target yaw disagrees with saved native state")
                if expected_episodes in {100, 200, 300, 1000}:
                    from examples.baselines.constrain import plug_orientation_data as native_data
                    for key in ("source_h5", "source_h5_sha256", "replay_h5", "replay_h5_sha256",
                                "case_json", "case_json_sha256"):
                        if key not in row:
                            raise ValueError(f"traj_{episode} missing registered {key}")
                    if any(sha256_file(row[key_path]) != row[key_hash] for key_path, key_hash in
                           (("source_h5", "source_h5_sha256"), ("replay_h5", "replay_h5_sha256"),
                            ("case_json", "case_json_sha256"))):
                        raise ValueError(f"traj_{episode} source/replay/case hash mismatch")
                    case = json.loads(Path(row["case_json"]).read_text())
                    if not math.isclose(float(case.get("target_yaw", math.nan)), float(target), abs_tol=1e-7, rel_tol=0):
                        raise ValueError(f"traj_{episode} registered yaw differs from source case")
                    if (not native_data._native_episode_ok(Path(row["source_h5"]), case)[0]
                            or not native_data._native_episode_ok(Path(row["replay_h5"]), case, replay=True)[0]
                            or not native_data._pair_native_match(row["source_h5"], row["replay_h5"])):
                        raise ValueError(f"traj_{episode} source/replay failed native admission")
                    action_digest = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
                    if row.get("source_actions_sha256") != action_digest:
                        raise ValueError(f"traj_{episode} native action digest mismatch")
                self.obs.append(torch.from_numpy(obs.copy()))
                self.actions.append(torch.from_numpy(actions.copy()))
                self.hints.append(torch.from_numpy(np.stack([orientation_hint(x, target) for x in obs])))
                # Match PlugBaseDataset: starts -1 .. T-3, including the
                # terminal windows padded with zero motion/last gripper.
                for start in range(-1, len(actions) - 2):
                    self.windows.append((episode, start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start = self.windows[index]
        obs, actions, hints = self.obs[episode], self.actions[episode], self.hints[episode]
        ids = [max(0, start), max(0, start + 1)]
        action_end = start + 16
        action = actions[max(0, start):action_end]
        if start < 0:
            action = torch.cat((actions[0:1], action))
        if action_end > len(actions):
            pad = torch.cat((torch.zeros(6), actions[-1, -1:])).reshape(1, 7)
            action = torch.cat((action, pad.repeat(action_end - len(actions), 1)))
        # Live control condition is evaluated at the newest input frame, never a future frame.
        return {"observations": obs[ids], "actions": action, "hint": hints[ids[-1]]}


class PlugConstrainAdapter:
    """EMA Constrain branch over the exact frozen Base; exposes native action chunks."""
    def __init__(self, checkpoint, device="cpu", control_enabled=True, hint_mode="correct"):
        if hint_mode not in {"correct", "fixed_standard", "shuffled"}:
            raise ValueError(f"unknown hint mode: {hint_mode}")
        self.device = torch.device(device)
        self.control_enabled, self.hint_mode = bool(control_enabled), hint_mode
        self.base_agent, _ = _load_frozen_base()
        self.denoiser = ConstrainDenoiser(self.base_agent.noise_pred_net, cond_dim=2).to(self.device)
        pinned_base = {key: value.detach().cpu().clone()
                       for key, value in self.denoiser.base_net.state_dict().items()}
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != FORMAT or payload.get("config", {}).get("model") != MODEL_CONFIG:
            raise ValueError("not a compatible Plug orientation Constrain checkpoint")
        if payload.get("frozen_base_sha256") != BASE_SHA256 or payload.get("frozen_base_ema_sha256") != BASE_EMA_SHA256:
            raise ValueError("Constrain checkpoint frozen Base provenance mismatch")
        random_count = _checkpoint_random_episode_count(payload)
        self.denoiser.load_state_dict(payload["model"], strict=True)
        if "ema_control" not in payload:
            raise ValueError("Constrain checkpoint lacks EMA control parameters")
        self._load_control_ema(payload["ema_control"])
        loaded = self.denoiser.base_net.state_dict()
        if set(pinned_base) != set(loaded) or not all(torch.equal(pinned_base[k], loaded[k].detach().cpu())
                                                       for k in pinned_base):
            raise ValueError("checkpoint overwrote the pinned frozen Base EMA")
        for value in payload["ema_control"]:
            if not torch.isfinite(value).all():
                raise ValueError("EMA control contains non-finite values")
        self.denoiser.eval()
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                       clip_sample=True, prediction_type="epsilon")
        self.target_yaw = 0.0
        self.shuffled_target_yaw = 0.0
        self.metadata = {"format": FORMAT, "checkpoint_sha256": sha256_file(checkpoint),
                         "frozen_base_sha256": BASE_SHA256, "weights": "EMA control + frozen Base",
                         "random_episode_count": random_count,
                         "control_enabled": self.control_enabled, "hint_mode": hint_mode}

    def _load_control_ema(self, values):
        params = self.denoiser.trainable_parameters()
        if len(values) != len(params):
            raise ValueError("EMA control parameter count mismatch")
        with torch.no_grad():
            for parameter, value in zip(params, values):
                if parameter.shape != value.shape:
                    raise ValueError("EMA control parameter shape mismatch")
                parameter.copy_(value.to(parameter.device, parameter.dtype))

    def set_episode_context(self, target_yaw, shuffled_target_yaw=None):
        self.target_yaw = float(target_yaw)
        self.shuffled_target_yaw = self.target_yaw if shuffled_target_yaw is None else float(shuffled_target_yaw)

    def get_action(self, observations):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if obs.shape != (1, 2, 46) or not torch.isfinite(obs).all():
            raise ValueError("expected finite (1,2,46) observation history")
        current = obs[0, -1].detach().cpu().numpy()
        target = self.target_yaw
        if self.hint_mode == "fixed_standard":
            target = 0.0
        elif self.hint_mode == "shuffled":
            target = self.shuffled_target_yaw
        hint = torch.as_tensor(orientation_hint(current, target), device=self.device)[None]
        repeated = hint[:, None, :].expand(-1, 16, -1)
        sample = torch.randn((1, 16, 7), device=self.device)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                epsilon = self.denoiser(sample, timestep, obs.flatten(1), hint=repeated,
                                        enabled=self.control_enabled)
                sample = self.scheduler.step(epsilon, timestep, sample).prev_sample
        result = sample[:, 1:9]
        if result.shape != (1, 8, 7) or not torch.isfinite(result).all():
            raise ValueError("sampler must produce finite native (1,8,7) actions")
        if result.abs().max().item() > 1.00001:
            raise ValueError("native action chunk exceeds [-1,1]")
        return result


def select_adaptation_checkpoint(rows):
    """Choose best random validation subject to <=5pp fixed-retention loss."""
    if len(rows) != len(CHECKPOINT_STEPS) or sorted(int(r["update"]) for r in rows) != list(CHECKPOINT_STEPS):
        raise ValueError("selection requires all six preregistered checkpoints")
    base_fixed = int(rows[0]["base_fixed_successes"])
    base_random = int(rows[0]["base_random_successes"])
    if (not 0 <= base_fixed <= 50 or not 0 <= base_random <= 50
            or any(int(row["base_fixed_successes"]) != base_fixed
                   or int(row["base_random_successes"]) != base_random for row in rows)):
        raise ValueError("every candidate must bind the same complete fresh-Base validation outcomes")
    minimum = math.ceil(base_fixed - 0.05 * 50)
    eligible = [row for row in rows if int(row["fixed_successes"]) >= minimum]
    if not eligible:
        return {"selected": None, "minimum_fixed_successes": minimum,
                "reason": "no candidate meets fixed-retention loss boundary"}
    chosen = max(eligible, key=lambda row: (int(row["random_successes"]), int(row["update"])))
    random_gain = int(chosen["random_successes"]) - base_random
    return {"selected": int(chosen["update"]), "minimum_fixed_successes": minimum,
            "base_fixed_successes": base_fixed, "base_random_successes": base_random,
            "selected_fixed_successes": int(chosen["fixed_successes"]),
            "selected_random_successes": int(chosen["random_successes"]),
            "random_gain_successes": random_gain,
            "random_gain_pp": 100.0 * random_gain / 50,
            "method_gates_met": random_gain >= 10,
            "selection_rule": "max random-validation successes among candidates within 5pp fixed retention; latest tie"}


def train(random_h5, random_manifest, fixed_h5, fixed_manifest, exp_dir, ckpt_dir,
          *, device="cuda", _updates=30000, _batch_size=1024, _test_mode=False,
          _random_episodes=100, _fixed_episodes=1000):
    """Fresh 30k control-branch training; production-sized overrides are forbidden."""
    if _test_mode and "PYTEST_CURRENT_TEST" not in os.environ:
        raise RuntimeError("test-only adaptation budget is available only inside pytest")
    if (_updates, _batch_size) != (30000, 1024) and not _test_mode:
        raise ValueError("Plug adaptation production budget is fixed at 30000/1024")
    if _updates < 1 or _batch_size < 2 or _batch_size % 2:
        raise ValueError("updates must be positive and batch size an even value")
    exp_dir, ckpt_dir = Path(exp_dir), Path(ckpt_dir)
    exp_dir.mkdir(parents=True, exist_ok=False)
    ckpt_dir.mkdir(parents=True, exist_ok=False)
    contract = build_adaptation_contract()
    if _random_episodes in {200, 300, 500}:
        contract_file = {200: "plug_orientation_adaptation_expansion200_contract.json",
                         300: "plug_orientation_adaptation_expansion300_contract.json",
                         500: "plug_orientation_adaptation_expansion500_contract.json"}[_random_episodes]
        contract["expansion"] = json.loads(Path(__file__).with_name(contract_file).read_text())
    if _random_episodes == 200:
        if sha256_file(PARENT_RANDOM100_FORWARD_CONTRACT) != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256:
            raise ValueError("frozen random-100 forward contract identity mismatch")
        random_registration = json.loads(Path(random_manifest).read_text())
        contract["expansion_run_lineage"] = {
            "parent_random100_forward_contract_sha256": PARENT_RANDOM100_FORWARD_CONTRACT_SHA256,
            "parent_dataset_manifest_sha256": random_registration.get("parent_dataset_manifest_sha256"),
            "random200_dataset_manifest_sha256": sha256_file(random_manifest),
            "random200_training_h5_sha256": sha256_file(random_h5),
            "random200_training_json_sha256": random_registration.get("training_json_sha256"),
            "random_episode_count": random_registration.get("effective_episode_count"),
        }
        if (contract["expansion_run_lineage"]["parent_dataset_manifest_sha256"] != PARENT_RANDOM100_MANIFEST_SHA256
                or contract["expansion_run_lineage"]["random_episode_count"] != 200
                or contract["expansion_run_lineage"]["random200_training_h5_sha256"]
                != random_registration.get("training_h5_sha256")):
            raise ValueError("random-200 run contract lineage does not bind the registered expansion")
    elif _random_episodes == 300:
        random_registration = json.loads(Path(random_manifest).read_text())
        contract["expansion_run_lineage"] = {
            "parent_random200_dataset_manifest_sha256": PARENT_RANDOM200_MANIFEST_SHA256,
            "random300_dataset_manifest_sha256": sha256_file(random_manifest),
            "random300_training_h5_sha256": sha256_file(random_h5),
            "random300_training_json_sha256": random_registration.get("training_json_sha256"),
            "random_episode_count": random_registration.get("effective_episode_count"),
        }
        if (contract["expansion_run_lineage"]["random_episode_count"] != 300
                or random_registration.get("parent_dataset_manifest_sha256") != PARENT_RANDOM200_MANIFEST_SHA256
                or random_registration.get("training_h5_sha256") != contract["expansion_run_lineage"]["random300_training_h5_sha256"]):
            raise ValueError("random-300 run contract lineage does not bind the registered expansion")
    elif _random_episodes == 500:
        if sha256_file(PARENT_RANDOM100_FORWARD_CONTRACT) != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256:
            raise ValueError("frozen original random-100 forward contract identity mismatch")
        random_registration = json.loads(Path(random_manifest).read_text())
        contract["expansion_run_lineage"] = {
            "random100_forward_contract_sha256": PARENT_RANDOM100_FORWARD_CONTRACT_SHA256,
            "parent_random300_dataset_manifest_sha256": PARENT_RANDOM300_MANIFEST_SHA256,
            "random500_dataset_manifest_sha256": sha256_file(random_manifest),
            "random500_training_h5_sha256": sha256_file(random_h5),
            "random500_training_json_sha256": random_registration.get("training_json_sha256"),
            "random_episode_count": random_registration.get("effective_episode_count"),
        }
        if (contract["expansion_run_lineage"]["random_episode_count"] != 500
                or random_registration.get("parent_dataset_manifest_sha256") != PARENT_RANDOM300_MANIFEST_SHA256
                or random_registration.get("training_h5_sha256") != contract["expansion_run_lineage"]["random500_training_h5_sha256"]):
            raise ValueError("random-500 run contract lineage does not bind the registered expansion")
    contract_path = exp_dir / "plug_orientation_adaptation_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n")
    archive = exp_dir / "source_archive"
    archive.mkdir()
    source_files = ["examples/baselines/constrain/plug_orientation_adaptation.py",
        "examples/baselines/constrain/plug_orientation_adaptation_contract.json",
        "examples/baselines/constrain/plug_orientation_adaptation_expansion200_contract.json",
        "examples/baselines/constrain/constrain_denoiser.py",
        "examples/baselines/constrain/plug_orientation_data.py",
        "examples/baselines/constrain/plug_orientation_base_dp.py",
        "examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py"]
    if _random_episodes == 300:
        source_files.append("examples/baselines/constrain/plug_orientation_adaptation_expansion300_contract.json")
    if _random_episodes == 500:
        source_files.extend(["examples/baselines/constrain/plug_orientation_adaptation_expansion300_contract.json",
            "examples/baselines/constrain/plug_orientation_adaptation_expansion500_contract.json"])
    archived = {}
    for rel in source_files:
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / rel).read_bytes())
        archived[rel] = sha256_file(ROOT / rel)
        if sha256_file(target) != archived[rel]:
            raise RuntimeError(f"training source archive mismatch: {rel}")
    (exp_dir / "source_closure.json").write_text(json.dumps(archived, indent=2, sort_keys=True) + "\n")
    if not _test_mode and (_random_episodes not in REGISTERED_RANDOM_EPISODE_COUNTS
                           or _fixed_episodes != 1000):
        raise ValueError("production adaptation requires a registered 100/200/300/500 random pool and 1000 fixed episodes")
    if not _test_mode and _random_episodes in {200, 300, 500}:
        expanded_manifest = json.loads(Path(random_manifest).read_text())
        expansion_file = {200: "plug_orientation_adaptation_expansion200_contract.json",
                          300: "plug_orientation_adaptation_expansion300_contract.json",
                          500: "plug_orientation_adaptation_expansion500_contract.json"}[_random_episodes]
        expected_expansion = sha256_file(Path(__file__).with_name(expansion_file))
        expected_protocol = f"plug_orientation_random{_random_episodes}_expansion_v1"
        expected_parent_manifest = {200: PARENT_RANDOM100_MANIFEST_SHA256,
                                    300: PARENT_RANDOM200_MANIFEST_SHA256,
                                    500: PARENT_RANDOM300_MANIFEST_SHA256}[_random_episodes]
        if (expanded_manifest.get("protocol") != expected_protocol
                or expanded_manifest.get("status") != "complete"
                or expanded_manifest.get("accepted_count") != _random_episodes
                or expanded_manifest.get("effective_episode_count") != _random_episodes
                or expanded_manifest.get("parent_dataset_manifest_sha256") != expected_parent_manifest
                or expanded_manifest.get("expansion_contract_sha256") != expected_expansion):
            raise ValueError(f"{_random_episodes}-episode training requires its registered expansion dataset")
    random_data = PlugAdaptationDataset(random_h5, random_manifest, _random_episodes)
    fixed_data = PlugAdaptationDataset(fixed_h5, fixed_manifest, _fixed_episodes)
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(1)
    device = torch.device(device)
    from diffusers.optimization import get_scheduler
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    base_agent, base_payload = _load_frozen_base()
    denoiser = ConstrainDenoiser(base_agent.noise_pred_net, cond_dim=2).to(device)
    base_state = {key: value.detach().cpu().clone() for key, value in denoiser.base_net.state_dict().items()}
    parameters = denoiser.trainable_parameters()
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=500, num_training_steps=_updates)
    ema = EMAModel(parameters=parameters, power=0.75)
    random_loader = iter(torch.utils.data.DataLoader(random_data, batch_size=_batch_size // 2,
                                                       shuffle=True, drop_last=True, num_workers=0))
    fixed_loader = iter(torch.utils.data.DataLoader(fixed_data, batch_size=_batch_size // 2,
                                                      shuffle=True, drop_last=True, num_workers=0))
    noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                    clip_sample=True, prediction_type="epsilon")
    provenance = {"random_episode_count": _random_episodes,
                  "random_h5_sha256": sha256_file(random_h5),
                  "random_manifest_sha256": sha256_file(random_manifest),
                  "fixed_h5_sha256": sha256_file(fixed_h5),
                  "fixed_manifest_sha256": sha256_file(fixed_manifest),
                  "contract_sha256": sha256_file(contract_path), "frozen_base_sha256": BASE_SHA256,
                  "frozen_base_ema_sha256": BASE_EMA_SHA256,
                  "source_closure_sha256": sha256_file(exp_dir / "source_closure.json")}
    if _random_episodes in {200, 300, 500}:
        expansion_file = {200: "plug_orientation_adaptation_expansion200_contract.json",
                          300: "plug_orientation_adaptation_expansion300_contract.json",
                          500: "plug_orientation_adaptation_expansion500_contract.json"}[_random_episodes]
        provenance["expansion_contract_sha256"] = sha256_file(Path(__file__).with_name(expansion_file))
        provenance["parent_dataset_manifest_sha256"] = json.loads(Path(random_manifest).read_text())["parent_dataset_manifest_sha256"]
    (exp_dir / "resolved_config.json").write_text(json.dumps({"format": FORMAT, "model": MODEL_CONFIG,
         "updates": _updates, "batch_size": _batch_size, "seed": 1,
         "random_episode_count": _random_episodes, "provenance": provenance},
        indent=2, sort_keys=True) + "\n")
    start = time.monotonic()
    log = (exp_dir / "training.jsonl").open("x")
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError("SIGTERM during adaptation training")))
    completed_update = 0
    try:
        for update in range(1, _updates + 1):
            if time.monotonic() - start >= 7200:
                raise TimeoutError("registered 7200-second training budget exhausted")
            batches = []
            for loader, dataset in ((random_loader, random_data), (fixed_loader, fixed_data)):
                try:
                    batches.append(next(loader))
                except StopIteration:
                    loader = iter(torch.utils.data.DataLoader(dataset, batch_size=_batch_size // 2,
                        shuffle=True, drop_last=True, num_workers=0))
                    batches.append(next(loader))
                    if dataset is random_data: random_loader = loader
                    else: fixed_loader = loader
            observations = torch.cat([b["observations"] for b in batches]).to(device)
            actions = torch.cat([b["actions"] for b in batches]).to(device)
            hints = torch.cat([b["hint"] for b in batches]).to(device)
            timesteps = torch.randint(0, 100, (len(actions),), device=device).long()
            noise = torch.randn_like(actions)
            noisy = noise_scheduler.add_noise(actions, noise, timesteps)
            repeated = hints[:, None, :].expand(-1, 16, -1)
            prediction = denoiser(noisy, timesteps, observations.flatten(1), hint=repeated)
            loss = torch.nn.functional.mse_loss(prediction, noise)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at update {update}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"non-finite gradient at update {update}")
            optimizer.step(); scheduler.step(); ema.step(parameters)
            completed_update = update
            if not all(torch.equal(base_state[k], v.detach().cpu()) for k, v in denoiser.base_net.state_dict().items()):
                raise RuntimeError("frozen Base changed during Constrain training")
            if update % 100 == 0:
                log.write(json.dumps({"update": update, "loss": float(loss.detach().cpu()),
                                      "lr": optimizer.param_groups[0]["lr"],
                                      "elapsed_seconds": time.monotonic() - start}) + "\n")
                log.flush()
            if update in CHECKPOINT_STEPS or update == _updates:
                raw_model = {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()}
                ema_controls = [p.detach().cpu().clone() for p in ema.shadow_params]
                payload = {"format": FORMAT, "checkpoint_role": "validation_candidate",
                    "update": update, "model": raw_model, "ema_control": ema_controls,
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "ema": ema.state_dict(), "model_config": MODEL_CONFIG,
                    "config": {"model": MODEL_CONFIG, "training": {"seed": 1,
                        "updates": _updates, "batch_size": _batch_size,
                        "random_episode_count": _random_episodes}},
                    **provenance}
                tmp = ckpt_dir / f"checkpoint_{update:06d}.pt.tmp"
                torch.save(payload, tmp); tmp.replace(ckpt_dir / f"checkpoint_{update:06d}.pt")
    except BaseException as exc:
        attempted_update = locals().get("update", 0)
        recovery = {"format": FORMAT, "checkpoint_role": "recovery", "last_update": completed_update,
            "failed_update": attempted_update,
            "model": {k: v.detach().cpu().clone() for k, v in denoiser.state_dict().items()},
            "ema_control": [p.detach().cpu().clone() for p in ema.shadow_params],
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "ema": ema.state_dict(),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "provenance": provenance}
        tmp_recovery = ckpt_dir / "recovery.pt.tmp"
        torch.save(recovery, tmp_recovery); tmp_recovery.replace(ckpt_dir / "recovery.pt")
        (exp_dir / "training_failure.json").write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - start,
            "last_checkpoint": max((int(p.stem.split("_")[-1]) for p in ckpt_dir.glob("checkpoint_*.pt")), default=0),
            "last_update": completed_update, "failed_update": attempted_update,
            "recovery_checkpoint": str(tmp_recovery.with_suffix("")),
            "source_provenance": provenance}, indent=2) + "\n")
        raise
    finally:
        log.close()
        signal.signal(signal.SIGTERM, previous_term)
    return {"updates": _updates, "elapsed_seconds": time.monotonic() - start,
            "exp_dir": str(exp_dir), "ckpt_dir": str(ckpt_dir)}


def evaluate_validation_candidates(ckpt_dir, random_catalog, fixed_catalog, output_root,
                                   *, base_checkpoint=BASE_CHECKPOINT, policy_factory=None,
                                   env_factory=None):
    """Evaluate six Constrain candidates against fresh frozen Base on both validation splits."""
    import gymnasium as gym
    from examples.baselines.constrain.plug_orientation_scenes import PlugSceneCatalog, restore_plug_scene

    catalogs = {"random_validation": PlugSceneCatalog(random_catalog),
                "fixed_validation": PlugSceneCatalog(fixed_catalog)}
    if (len(catalogs["random_validation"]) != 50 or len(catalogs["fixed_validation"]) != 50
            or any(catalog.manifest.get("split") != split for split, catalog in catalogs.items())):
        raise ValueError("candidate selection requires the complete frozen random/fixed validation catalogs")
    frozen_manifests = {
        "random_validation": "40eaa0958c1fa495b6b082ac8607d623c6963bd6c29094b184e395860eaaada9",
        "fixed_validation": "ec54ebe355459d98b1fae736648641b4734fcdee8e1f5af9b6de825edbfe54fc"}
    for split, path in (("random_validation", random_catalog), ("fixed_validation", fixed_catalog)):
        if sha256_file(Path(path) / "manifest.json") != frozen_manifests[split]:
            raise ValueError(f"{split} catalog manifest is not the frozen validation identity")
    checkpoint_paths = [Path(ckpt_dir) / f"checkpoint_{step:06d}.pt" for step in CHECKPOINT_STEPS]
    if any(not path.is_file() for path in checkpoint_paths):
        raise FileNotFoundError("all six registered Constrain checkpoints are required")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    if env_factory is None:
        from examples.baselines.constrain.plug_orientation import ENV_ID
        env_factory = lambda: gym.make(ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                                      sim_backend="physx_cpu", max_episode_steps=1000)
    base_policy = base.PlugBaseAdapter(base_checkpoint, device="cpu")
    rows = []
    fresh_base_counts = {}
    candidate_provenance = None
    candidate_training_config = None
    started = time.monotonic()
    for checkpoint, update in zip(checkpoint_paths, CHECKPOINT_STEPS):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("format") != FORMAT or payload.get("update") != update
                or payload.get("checkpoint_role") != "validation_candidate"
                or payload.get("frozen_base_sha256") != BASE_SHA256
                or payload.get("frozen_base_ema_sha256") != BASE_EMA_SHA256
                or payload.get("model_config") != MODEL_CONFIG
                or payload.get("config", {}).get("model") != MODEL_CONFIG
                or "ema_control" not in payload):
            raise ValueError(f"checkpoint is not registered frozen-Base candidate {update}")
        random_count = _checkpoint_random_episode_count(payload)
        training_config = payload.get("config", {}).get("training", {})
        expected_training_config = {"seed": 1, "updates": 30000, "batch_size": 1024,
                                    "random_episode_count": random_count}
        normalized_training_config = {key: training_config.get(key, random_count if key == "random_episode_count" else None)
                                      for key in expected_training_config}
        if normalized_training_config != expected_training_config:
            raise ValueError(f"candidate {update} does not match the registered fresh-training configuration")
        if candidate_training_config is None:
            candidate_training_config = normalized_training_config
        elif normalized_training_config != candidate_training_config:
            raise ValueError("validation candidates do not share one exact training configuration")
        provenance = {key: payload.get(key) for key in ("random_episode_count", "random_h5_sha256", "random_manifest_sha256",
            "fixed_h5_sha256", "fixed_manifest_sha256", "contract_sha256",
            "frozen_base_sha256", "frozen_base_ema_sha256", "source_closure_sha256",
            "expansion_contract_sha256", "parent_dataset_manifest_sha256")}
        if random_count == 100:
            provenance["random_episode_count"] = 100
            provenance["expansion_contract_sha256"] = "not_applicable_random100"
            provenance["parent_dataset_manifest_sha256"] = "not_applicable_random100"
        if any(value is None for value in provenance.values()):
            raise ValueError(f"candidate {update} lacks complete dataset/source provenance")
        if candidate_provenance is None:
            candidate_provenance = provenance
        elif provenance != candidate_provenance:
            raise ValueError("validation candidates do not share exact registered training provenance")
        frozen_state = base_policy.agent.noise_pred_net.state_dict()
        model_state = payload["model"]
        if any(f"base_net.{key}" not in model_state or not torch.equal(
                model_state[f"base_net.{key}"].detach().cpu(), value.detach().cpu())
               for key, value in frozen_state.items()):
            raise ValueError(f"candidate {update} changed frozen Base tensors")
        policy = policy_factory(checkpoint) if policy_factory else PlugConstrainAdapter(checkpoint, "cpu")
        pair = {"update": update, "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint), "random_episode_count": random_count}
        for split, catalog in catalogs.items():
            arm_records = {"base": [], "constrain_correct": []}
            env = env_factory()
            try:
                for index in range(50):
                    scene = catalog.load(index)
                    seed = 1 + index
                    actors = (("base", base_policy), ("constrain_correct", policy)) if update == CHECKPOINT_STEPS[0] else (("constrain_correct", policy),)
                    for role, actor in actors:
                        obs0 = restore_plug_scene(env, scene).detach().cpu().numpy().reshape(-1)
                        scene_meta = catalog.manifest["scenes"][index]
                        if role == "constrain_correct":
                            actor.set_episode_context(scene_meta["target_yaw_rad"])
                        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                        history = [obs0.copy(), obs0.copy()]
                        controls = 0; success = False; terminal = "control_cap"
                        event_path = output_root / f"{split}_{update:06d}_{role}_episode_{index:03d}.jsonl"
                        with event_path.open("w") as events:
                            events.write(json.dumps({"event": "episode_start", "index": index,
                                "seed": seed, "target_yaw_rad": scene_meta["target_yaw_rad"],
                                "initial_observation": obs0.tolist()}, allow_nan=False) + "\n")
                            while controls < 600:
                                chunk = actor.get_action(torch.as_tensor(np.asarray(history)[None], dtype=torch.float32))
                                if chunk.shape != (1, 8, 7) or not torch.isfinite(chunk).all():
                                    raise ValueError("candidate sampler returned invalid native action chunk")
                                for action in chunk[0]:
                                    action_np = action.detach().cpu().numpy()
                                    step = env.step(action_np)
                                    obs_next, _, terminated, truncated, info = step
                                    next_obs = obs_next.detach().cpu().numpy().reshape(-1)
                                    if next_obs.shape != (46,) or not np.isfinite(next_obs).all():
                                        raise ValueError("environment returned invalid 46-D observation")
                                    history = [history[-1], next_obs]
                                    controls += 1
                                    success = bool(info.get("success", False))
                                    events.write(json.dumps({"event": "control", "control": controls,
                                        "observation": next_obs.tolist(), "action": action_np.tolist(),
                                        "success": success, "terminated": bool(terminated),
                                        "truncated": bool(truncated), "info_success": success}, allow_nan=False) + "\n")
                                    events.flush()
                                    if success or terminated or truncated:
                                        terminal = "success" if success else str(info.get("fail", "native_termination"))
                                        break
                                if success or terminated or truncated:
                                    break
                        arm_records[role].append({"index": index, "seed": seed, "success": success,
                                                  "terminal": terminal, "controls": controls,
                                                  "target_yaw_rad": scene_meta["target_yaw_rad"]})
                for role, records in arm_records.items():
                    if role == "base" and update != CHECKPOINT_STEPS[0]:
                        continue
                    (output_root / f"{split}_{update:06d}_{role}.json").write_text(
                        json.dumps({"split": split, "update": update, "records": records},
                                   indent=2, allow_nan=False) + "\n")
            except BaseException as exc:
                (output_root / "partial_summary.json").write_text(json.dumps({"update": update,
                    "split": split, "completed": arm_records,
                    "error": f"{type(exc).__name__}: {exc}"}, indent=2, allow_nan=False) + "\n")
                raise
            finally:
                env.close()
            if update == CHECKPOINT_STEPS[0]:
                fresh_base_counts["base_fixed_successes" if split == "fixed_validation" else "base_random_successes"] = sum(
                    row["success"] for row in arm_records["base"])
            pair["fixed_successes" if split == "fixed_validation" else "random_successes"] = sum(
                row["success"] for row in arm_records["constrain_correct"])
        pair.update(fresh_base_counts)
        rows.append(pair)
        (output_root / "progress.json").write_text(json.dumps({"completed_candidates": rows,
            "elapsed_seconds": time.monotonic() - started}, indent=2) + "\n")
    selection = select_adaptation_checkpoint(rows)
    selection.update({"candidate_results": rows, "base_checkpoint_sha256": sha256_file(base_checkpoint),
                      "random_catalog_sha256": sha256_file(Path(random_catalog) / "manifest.json"),
                      "fixed_catalog_sha256": sha256_file(Path(fixed_catalog) / "manifest.json")})
    (output_root / "validation_selection.json").write_text(json.dumps(selection, indent=2,
        sort_keys=True, allow_nan=False) + "\n")
    return selection


def _heldout_preflight(checkpoint, selection_path, random_catalog, fixed_catalog,
                       contract_path=HELDOUT_CONTRACT):
    """Bind the independent roster to the already-selected validation artifact."""
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    if contract.get("post_validation_gate_heldout_authorized") is not True:
        raise ValueError("heldout requires explicit post-validation-gate authorization")
    if (sha256_file(contract_path) != "ed5cf1bddb73c056ca4aeacae211b5e4e0c465713a25b3170c83fa0e0362b29b"
            or sha256_file(Path(__file__).with_name("plug_orientation_adaptation_expansion500_contract.json"))
            != contract["adaptation_contract"]["sha256"]):
        raise ValueError("heldout contract or registered 500-pool contract identity mismatch")
    if sha256_file(BASE_CHECKPOINT) != contract["frozen_base"]["sha256"]:
        raise ValueError("heldout frozen Base checkpoint identity mismatch")
    selection_path, checkpoint = Path(selection_path), Path(checkpoint)
    if not selection_path.is_file() or not checkpoint.is_file():
        raise ValueError("heldout selection/checkpoint identity mismatch: registered file is missing")
    if (sha256_file(selection_path) != contract["validation_selection"]["sha256"]
            or sha256_file(checkpoint) != contract["checkpoint"]["sha256"]):
        raise ValueError("heldout selection/checkpoint identity mismatch")
    selection = json.loads(selection_path.read_text())
    if (selection.get("selected") != 30000 or selection.get("method_gates_met") is not False
            or selection.get("random_gain_pp") != 10.0
            or selection.get("selected_random_successes") != 23
            or selection.get("base_random_successes") != 18):
        raise ValueError("validation selection no longer records the original +10 pp gate failure")
    candidate = next((row for row in selection.get("candidate_results", [])
                      if row.get("update") == 30000), None)
    if (candidate is None or candidate.get("checkpoint_sha256") != contract["checkpoint"]["sha256"]
            or candidate.get("random_episode_count") != 500
            or candidate.get("random_successes") != 23
            or candidate.get("fixed_successes") != 37):
        raise ValueError("selected checkpoint does not match the registered 500-pool candidate")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("format") != FORMAT or payload.get("update") != 30000
            or payload.get("checkpoint_role") != "validation_candidate"
            or _checkpoint_random_episode_count(payload) != 500
            or payload.get("frozen_base_sha256") != BASE_SHA256
            or payload.get("frozen_base_ema_sha256") != BASE_EMA_SHA256
            or payload.get("config", {}).get("training", {}).get("seed") != 1
            or payload.get("config", {}).get("training", {}).get("updates") != 30000):
        raise ValueError("checkpoint is not the frozen-Base random-500 update-30000 candidate")
    for path, split, key in ((random_catalog, "random_final_test", "random_final_test"),
                             (fixed_catalog, "fixed_base_test", "fixed_base_test")):
        catalog_path = Path(path)
        manifest = catalog_path / "manifest.json"
        if sha256_file(manifest) != contract["catalogs"][key]["manifest_sha256"]:
            raise ValueError(f"{split} catalog identity mismatch")
        if sha256_file(catalog_path.parent / "catalog_set.json") != contract["catalogs"]["catalog_set_sha256"]:
            raise ValueError("heldout catalog-set identity mismatch")
        metadata = json.loads(manifest.read_text())
        if metadata.get("split") != split or metadata.get("scene_count") != 100:
            raise ValueError(f"expected complete registered {split} catalog")
    return contract


def _heldout_cohort_plan(random_catalog, fixed_catalog, *, shuffle_seed=1703):
    random_rows = random_catalog.manifest["scenes"]
    fixed_rows = fixed_catalog.manifest["scenes"]
    if len(random_rows) != 100 or len(fixed_rows) != 100:
        raise ValueError("heldout cohorts require both complete 100-scene catalogs")
    shuffled = np.random.default_rng(shuffle_seed).permutation(
        [float(row["target_yaw_rad"]) for row in random_rows])
    return [("random_final_test", "base", random_catalog, None),
            ("random_final_test", "constrain_correct", random_catalog, None),
            ("random_final_test", "constrain_fixed_standard", random_catalog, 0.0),
            ("random_final_test", "constrain_shuffled", random_catalog, shuffled),
            ("fixed_base_test", "base", fixed_catalog, None),
            ("fixed_base_test", "constrain_correct", fixed_catalog, None)]


def _heldout_yaw(observation):
    quat = np.asarray(observation, dtype=np.float64).reshape(-1)[28:32]
    if quat.shape != (4,) or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("heldout observation lacks finite charger quaternion at [28:32]")
    quat = quat / np.linalg.norm(quat)
    w, x, y, z = quat
    return float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _heldout_hint_target(role, actual_target, shuffled_target=None):
    if role == "constrain_correct":
        return float(actual_target)
    if role == "constrain_fixed_standard":
        return 0.0
    if role == "constrain_shuffled":
        if shuffled_target is None:
            raise ValueError("shuffled hint requires the frozen episode assignment")
        return float(shuffled_target)
    raise ValueError(f"role has no Constrain hint: {role}")


def _native_info_scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.size != 1 or not np.isfinite(array).all():
        return None
    return float(array.reshape(-1)[0])


def _run_heldout_rollouts(random_catalog, fixed_catalog, output_root, base_policy,
                          constrain_policy, env_factory, *, restore_fn=None,
                          _episode_limit=None, control_cap=600, _output_exists=False):
    """Native loop seam; test limits are private and absent from the public CLI."""
    from examples.baselines.constrain.plug_orientation_scenes import restore_plug_scene
    restore_fn = restore_plug_scene if restore_fn is None else restore_fn
    output_root = Path(output_root)
    if _output_exists:
        owned_contract = output_root / "heldout_contract.json"
        if not output_root.is_dir() or any(path != owned_contract for path in output_root.iterdir()):
            raise FileExistsError(f"heldout output directory is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    plan = _heldout_cohort_plan(random_catalog, fixed_catalog)
    completed, started = [], time.monotonic()
    env = env_factory()
    try:
        for split, role, catalog, assignment in plan:
            records, events_path = [], output_root / f"{split}_{role}.jsonl"
            count = len(catalog) if _episode_limit is None else min(len(catalog), _episode_limit)
            actor = base_policy if role == "base" else constrain_policy
            if role != "base":
                actor.hint_mode = {"constrain_correct": "correct",
                                   "constrain_fixed_standard": "fixed_standard",
                                   "constrain_shuffled": "shuffled"}[role]
            with events_path.open("w") as events:
                events.write(json.dumps({"event": "cohort_start", "split": split,
                    "cohort": role, "planned_episodes": count}, allow_nan=False) + "\n")
                for index in range(count):
                    scene = catalog.load(index)
                    metadata = scene[0]
                    actual_target = float(metadata["target_yaw_rad"])
                    shuffled_target = (float(assignment[index]) if role == "constrain_shuffled" else None)
                    observation = restore_fn(env, scene)
                    seed = 1 + index
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                    if role != "base":
                        actor.set_episode_context(_heldout_hint_target(role, actual_target, shuffled_target))
                    obs0 = observation.detach().cpu().numpy().reshape(-1) if hasattr(observation, "detach") else np.asarray(observation).reshape(-1)
                    history = [obs0.copy(), obs0.copy()]
                    initial_error = math.atan2(math.sin(actual_target - _heldout_yaw(obs0)),
                                               math.cos(actual_target - _heldout_yaw(obs0)))
                    controls, success, terminal, failure = 0, False, "control_cap", None
                    official_position_error, official_rotation_error = None, None
                    episode_path = output_root / f"{split}_{role}_episode_{index:03d}.jsonl"
                    with episode_path.open("w") as episode_events:
                        episode_events.write(json.dumps({"event": "episode_start", "index": index,
                            "seed": seed, "target_yaw_rad": actual_target,
                            "hint_target_yaw_rad": _heldout_hint_target(role, actual_target, shuffled_target)
                                if role != "base" else None,
                            "initial_observation": obs0.tolist()}, allow_nan=False) + "\n")
                        while controls < control_cap:
                            if time.monotonic() - started >= 21600:
                                raise TimeoutError("heldout stage exceeded its registered 21,600-second budget")
                            chunk = actor.get_action(torch.as_tensor(np.asarray(history)[None], dtype=torch.float32))
                            if tuple(chunk.shape) != (1, 8, 7) or not torch.isfinite(chunk).all():
                                raise ValueError("heldout sampler returned invalid native (1,8,7) action chunk")
                            for action in chunk[0]:
                                action_np = action.detach().cpu().numpy()
                                next_obs, _, terminated, truncated, info = env.step(action_np)
                                next_obs = next_obs.detach().cpu().numpy().reshape(-1) if hasattr(next_obs, "detach") else np.asarray(next_obs).reshape(-1)
                                if next_obs.shape != (46,) or not np.isfinite(next_obs).all():
                                    raise ValueError("heldout environment returned invalid 46-D observation")
                                controls += 1
                                success = bool(info.get("success", False))
                                official_position_error = _native_info_scalar(info.get("obj_to_goal_dist"))
                                official_rotation_error = _native_info_scalar(info.get("obj_to_goal_angle"))
                                failure = info.get("fail")
                                if failure is not None and not isinstance(failure, (str, int, float, bool)):
                                    failure = str(failure)
                                history = [history[-1], next_obs]
                                episode_events.write(json.dumps({"event": "control", "control": controls,
                                    "observation": next_obs.tolist(), "action": action_np.tolist(),
                                    "info_success": success, "terminated": bool(terminated),
                                    "truncated": bool(truncated),
                                    "official_position_error_m": official_position_error,
                                    "official_rotation_error_rad": official_rotation_error}, allow_nan=False) + "\n")
                                episode_events.flush()
                                if success or terminated or truncated or controls >= control_cap:
                                    terminal = "success" if success else str(failure or ("control_cap" if controls >= control_cap else "native_termination"))
                                    break
                            if success or terminated or truncated or controls >= control_cap:
                                break
                        final_yaw = _heldout_yaw(history[-1])
                        final_error = math.atan2(math.sin(actual_target - final_yaw),
                                                 math.cos(actual_target - final_yaw))
                        record = {"index": index, "seed": seed, "scene_id": metadata["scene_id"],
                            "target_yaw_rad": actual_target, "success": success, "terminal": terminal,
                            "failure_type": failure or (None if success else terminal), "controls": controls,
                            "official_position_error_m": official_position_error,
                            "official_rotation_error_rad": official_rotation_error,
                            "initial_alignment_error_rad": initial_error,
                            "final_alignment_error_rad": final_error}
                        episode_events.write(json.dumps({"event": "episode_end", **record}, allow_nan=False) + "\n")
                    records.append(record)
                    completed.append({"split": split, "cohort": role, **record})
                    events.write(json.dumps({"event": "episode_complete", **record}, allow_nan=False) + "\n")
                    events.flush()
                    _write_json(output_root / "progress.json", {"completed_episodes": completed,
                        "elapsed_seconds": time.monotonic() - started, "planned_episodes": 600})
            _write_json(output_root / f"{split}_{role}.json", {"split": split, "cohort": role,
                "records": records, "successes": sum(row["success"] for row in records),
                "episodes": len(records)})
    except BaseException as exc:
        _write_json(output_root / "partial_summary.json", {"completed_episodes": completed,
            "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        env.close()
    return {"completed_episodes": completed, "elapsed_seconds": time.monotonic() - started}


def evaluate_heldout(checkpoint, random_catalog, fixed_catalog, output_root, *,
                     selection_path=HELDOUT_SELECTION, contract_path=HELDOUT_CONTRACT,
                     policy_factory=None, base_policy_factory=None, env_factory=None):
    """Run the fixed independent 600-arm heldout roster; never select a model."""
    contract = _heldout_preflight(checkpoint, selection_path, random_catalog, fixed_catalog, contract_path)
    from examples.baselines.constrain.plug_orientation_scenes import PlugSceneCatalog
    catalogs = {"random": PlugSceneCatalog(random_catalog), "fixed": PlugSceneCatalog(fixed_catalog)}
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json(output_root / "heldout_contract.json", contract)
    base_policy = (base_policy_factory() if base_policy_factory else
                   base.PlugBaseAdapter(BASE_CHECKPOINT, device="cpu"))
    constrain_policy = (policy_factory(checkpoint) if policy_factory else
                        PlugConstrainAdapter(checkpoint, device="cpu", hint_mode="correct"))
    if env_factory is None:
        import gymnasium as gym
        from examples.baselines.constrain.plug_orientation import ENV_ID
        env_factory = lambda: gym.make(ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                                      sim_backend="physx_cpu", max_episode_steps=1000)
    result = _run_heldout_rollouts(catalogs["random"], catalogs["fixed"], output_root,
                                   base_policy, constrain_policy, env_factory, _output_exists=True)
    result.update({"contract_sha256": sha256_file(contract_path),
        "selection_sha256": sha256_file(selection_path), "checkpoint_sha256": sha256_file(checkpoint),
        "runner_source_sha256": sha256_file(__file__),
        "base_checkpoint_sha256": BASE_SHA256, "native_terminal_state_recomputation": "unavailable",
        "videos": "not_recorded_no_compatible_existing_per_arm_recorder"})
    _write_json(output_root / "summary.json", result)
    return result


def collect_random100(dataset_dir, exp_dir, *, run_case_fn=None, seed=1,
                      _candidate_count=150, _accepted_count=100, _wall_seconds=5400):
    """Collect the preregistered first 100 expert+replay successes at random yaw.

    Underscored count/deadline parameters are test-only seams; the CLI always
    uses the frozen 150-attempt/100-admission design.
    """
    from examples.baselines.constrain import plug_orientation_data as data
    from examples.baselines.constrain.run_plug_ee_lift_v4 import SOURCE_PATHS
    dataset_dir, exp_dir = Path(dataset_dir), Path(exp_dir)
    if _candidate_count != 150 or _accepted_count != 100 or _wall_seconds != 5400:
        # Unit fixtures may exercise fewer rows without exposing a CLI override.
        import inspect
        if not any("pytest" in frame.filename for frame in inspect.stack()):
            raise ValueError("random-100 production budgets are fixed")
    dataset_dir.mkdir(parents=True, exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(int(seed) + 1703)
    candidates = [{"seed": 2026123000 + i,
                   "target_yaw_rad": float(rng.uniform(-math.pi / 8, math.pi / 8))}
                  for i in range(_candidate_count)]
    bin_edges = np.linspace(-math.pi / 8, math.pi / 8, 6)
    planned_bins = np.histogram([row["target_yaw_rad"] for row in candidates], bins=bin_edges)[0].tolist()
    source_files = list(SOURCE_PATHS) + ["examples/baselines/constrain/plug_orientation_data.py",
        "examples/baselines/constrain/plug_orientation_adaptation.py",
        "examples/baselines/constrain/plug_orientation_base_dp.py",
        "examples/baselines/constrain/plug_orientation_adaptation_contract.json"]
    archive = exp_dir / "source_archive"
    archive.mkdir()
    source_hashes = {}
    for rel in source_files:
        dest = archive / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes())
        source_hashes[rel] = sha256_file(ROOT / rel)
    prereg = {"format": "plug_orientation_random100_preregistration_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_absolute_gate_waived": True, "frozen_base_sha256": BASE_SHA256,
        "frozen_base_ema_sha256": BASE_EMA_SHA256, "seed": seed,
        "candidates": candidates, "candidate_cap": _candidate_count,
        "yaw_bin_edges_rad": bin_edges.tolist(), "planned_yaw_bin_counts": planned_bins,
        "accepted_cap": _accepted_count, "wall_seconds": _wall_seconds,
        "per_candidate_control_cap": 600, "aggregate_source_replay_control_cap": 90000,
        "source_sha256": source_hashes}
    (exp_dir / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    prereg_hash = sha256_file(exp_dir / "preregistration.json")
    if run_case_fn is None:
        from examples.baselines.constrain.plug_orientation_ee_expert import run_case
        run_case_fn = run_case
    started = time.monotonic(); attempts = []; accepted = []
    journal = (dataset_dir / "attempts.jsonl").open("x")
    try:
        for index, candidate in enumerate(candidates):
            if time.monotonic() - started >= _wall_seconds:
                break
            if sha256_file(exp_dir / "preregistration.json") != prereg_hash:
                raise RuntimeError("random collection registration changed")
            if any(sha256_file(archive / name) != digest or sha256_file(ROOT / name) != digest
                   for name, digest in source_hashes.items()):
                raise RuntimeError("random collection source identity changed")
            seed_i, yaw = candidate["seed"], candidate["target_yaw_rad"]
            out = dataset_dir / f"candidate_{index + 1:03d}_seed_{seed_i}"
            row = {**candidate, "status": "failed", "accepted": False,
                   "source_h5": str(out / "expert.h5"), "replay_h5": str(out / "replay/replay.h5")}
            try:
                remaining = _wall_seconds - (time.monotonic() - started)
                with data._term_as_timeout("SIGTERM received during random native candidate"), data._collection_deadline(remaining):
                    case = run_case_fn(seed_i, out, target_yaw=yaw, deadline=started + _wall_seconds)
                case_path = out / "case.json"
                if not case_path.is_file():
                    case_path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
                row.update(case_json=str(case_path), expert_controls=int(case.get("expert_controls", 0)),
                           replay_controls=int(case.get("replay_controls", 0)))
                if max(row["expert_controls"], row["replay_controls"]) > 600:
                    raise RuntimeError("native teacher exceeded 600-control episode cap")
                row["source_h5_sha256"] = sha256_file(row["source_h5"]) if Path(row["source_h5"]).is_file() else None
                row["replay_h5_sha256"] = sha256_file(row["replay_h5"]) if Path(row["replay_h5"]).is_file() else None
                ok = case.get("expert", {}).get("official_success") is True and case.get("replay", {}).get("success") is True
                if row["source_h5_sha256"] and row["replay_h5_sha256"]:
                    src_ok = data._native_episode_ok(Path(row["source_h5"]), case)
                    rep_ok = data._native_episode_ok(Path(row["replay_h5"]), case, replay=True)
                    ok = ok and src_ok[0] and rep_ok[0] and data._pair_native_match(row["source_h5"], row["replay_h5"])
                row["accepted"] = bool(ok)
                row["status"] = "accepted" if ok else "failed"
                row["rejection_reason"] = None if ok else "official_success_or_native_pair_validation_failed"
                if ok:
                    row.update(source_sha256=row["source_h5_sha256"], replay_sha256=row["replay_h5_sha256"])
                    accepted.append(row)
            except BaseException as exc:
                known = type(exc) is RuntimeError and str(exc) in {"MPLib screw planning failed", "native teacher exceeded 600-control episode cap"}
                row.update(error=f"{type(exc).__name__}: {exc}", failure_class="known_planning_failure" if known else "unexpected_exception")
                if not known:
                    attempts.append(row); journal.write(json.dumps(row, allow_nan=False) + "\n"); journal.flush()
                    journal.flush(); os.fsync(journal.fileno())
                    (dataset_dir / "partial_summary.json").write_text(json.dumps({
                        "attempts": attempts, "accepted_count": len(accepted),
                        "source_controls_known": sum(r.get("expert_controls", 0) or 0 for r in attempts),
                        "replay_controls_known": sum(r.get("replay_controls", 0) or 0 for r in attempts),
                        "error": row.get("error")}, indent=2, allow_nan=False) + "\n")
                    raise
            attempts.append(row); journal.write(json.dumps(row, allow_nan=False) + "\n"); journal.flush(); os.fsync(journal.fileno())
            (dataset_dir / "progress.json").write_text(json.dumps({"attempts": attempts, "accepted": len(accepted)}, indent=2) + "\n")
            if len(accepted) == _accepted_count:
                break
    finally:
        journal.close()
    accepted_bins = np.histogram([r["target_yaw_rad"] for r in accepted], bins=bin_edges)[0].tolist()
    coverage_ok = _accepted_count != 100 or all(count > 0 for count in accepted_bins)
    collection_complete = len(accepted) == _accepted_count and coverage_ok
    if collection_complete and all(Path(r["source_h5"]).is_file() and Path(r["replay_h5"]).is_file() for r in accepted):
        data.consolidate_attempts(dataset_dir, accepted, _accepted_count)
    result = {"status": "complete" if collection_complete else "incomplete",
              "attempts": attempts, "accepted_count": len(accepted),
              "source_controls": sum(r.get("expert_controls", 0) for r in attempts),
              "replay_controls": sum(r.get("replay_controls", 0) for r in attempts),
              "preregistration_sha256": prereg_hash, "planned_yaw_bin_counts": planned_bins,
              "accepted_yaw_bin_counts": accepted_bins,
              "source_archive_sha256": source_hashes}
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest.update(result)
    manifest.update({"protocol": "plug_orientation_random100_v1",
                     "preregistration_sha256": prereg_hash,
                     "collection_experiment_dir": str(exp_dir.resolve()),
                     "collection_source_archive": str(archive.resolve())})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return result


def random200_candidate_plan(seed=1):
    """Actual-run-seeded sequential yaw stream, skipping the original 150 plans."""
    rng = np.random.default_rng(int(seed) + 1703)
    yaws = rng.uniform(-math.pi / 8, math.pi / 8, size=300)
    return [{"seed": 2026123150 + i, "target_yaw_rad": float(yaw)}
            for i, yaw in enumerate(yaws[150:])]


def random300_candidate_plan(seed=1):
    """Reserve the first fresh 150 candidates after all 300 registered yaw draws."""
    rng = np.random.default_rng(int(seed) + 1703)
    yaws = rng.uniform(-math.pi / 8, math.pi / 8, size=450)
    return [{"seed": 2026123300 + i, "target_yaw_rad": float(yaw)}
            for i, yaw in enumerate(yaws[300:])]


def random500_candidate_plan(seed=1):
    """Reserve only fresh seeds and yaw draws after the sealed 300-demo plan."""
    rng = np.random.default_rng(int(seed) + 1703)
    yaws = rng.uniform(-math.pi / 8, math.pi / 8, size=750)
    return [{"seed": 2026123450 + i, "target_yaw_rad": float(yaw)}
            for i, yaw in enumerate(yaws[450:])]


def _tree_hashes(root):
    root = Path(root)
    return {str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _dataset_ancestry_identity(dataset_dir):
    """Capture producer-bound identities without comparing historical code to HEAD."""
    dataset_dir = Path(dataset_dir)
    from examples.baselines.constrain import plug_orientation_data as data
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    collection = Path(manifest["collection_experiment_dir"])
    prereg_path = collection / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    archive = Path(manifest.get("collection_source_archive") or (collection / "source_archive"))
    source_hashes = prereg.get("source_sha256", {})
    for relative, digest in source_hashes.items():
        if not (archive / relative).is_file() or sha256_file(archive / relative) != digest:
            raise ValueError(f"historical ancestor source archive mismatch: {relative}")
    return {"dataset": str(dataset_dir.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "h5_sha256": sha256_file(dataset_dir / data.DATA_H5),
        "json_sha256": sha256_file(dataset_dir / data.DATA_JSON),
        "collection_preregistration_sha256": sha256_file(prereg_path),
        "source_archive_path": str(archive.resolve()),
        "producer_source_archive_sha256": source_hashes,
        "tree_sha256": _tree_hashes(dataset_dir)}


def _same_h5_node(left, right):
    if isinstance(left, h5py.Group):
        return (isinstance(right, h5py.Group) and set(left) == set(right)
                and all(_same_h5_node(left[key], right[key]) for key in left))
    return isinstance(right, h5py.Dataset) and np.array_equal(left[:], right[:])


def _validate_random200_lineage(h5_path, json_path, registration, manifest):
    """Validate expanded rows against frozen parent bytes and sealed collection ledger."""
    if os.environ.get("PYTEST_CURRENT_TEST") and getattr(_validate_random200_lineage, "_fixture_override", False):
        return
    from examples.baselines.constrain import plug_orientation_data as data
    parent = RANDOM100_DATASET
    parent_manifest_path = parent / "dataset_manifest.json"
    parent_json_path, parent_h5_path = parent / data.DATA_JSON, parent / data.DATA_H5
    parent_manifest = json.loads(parent_manifest_path.read_text())
    prereg_path = Path(parent_manifest["collection_experiment_dir"]) / "preregistration.json"
    hashes = (sha256_file(parent_manifest_path), sha256_file(parent_h5_path),
              sha256_file(parent_json_path), sha256_file(prereg_path))
    frozen_hashes = (PARENT_RANDOM100_MANIFEST_SHA256, PARENT_RANDOM100_H5_SHA256,
        PARENT_RANDOM100_JSON_SHA256, PARENT_RANDOM100_PREREGISTRATION_SHA256)
    if hashes != frozen_hashes:
        raise ValueError("actual random-100 parent/preregistration identity mismatch")
    parent_prereg = json.loads(prereg_path.read_text())
    parent_archive = Path(parent_manifest["collection_source_archive"])
    for relative, digest in parent_prereg.get("source_sha256", {}).items():
        if sha256_file(parent_archive / relative) != digest:
            raise ValueError(f"actual frozen parent source closure mismatch: {relative}")
    for key, digest in zip(("parent_dataset_manifest_sha256", "parent_h5_sha256",
                            "parent_json_sha256", "parent_preregistration_sha256"), frozen_hashes):
        if registration.get(key) != digest:
            raise ValueError(f"expanded registration does not bind {key}")
    rows = json.loads(Path(json_path).read_text())["episodes"]
    parent_rows = json.loads(parent_json_path.read_text())["episodes"]
    for i, row in enumerate(rows[:100]):
        frozen = parent_rows[i]
        for key in ("source_h5", "replay_h5", "case_json"):
            if row.get(key) != frozen.get(key) or row.get(f"{key}_sha256") != frozen.get(f"{key}_sha256"):
                raise ValueError(f"parent episode {i} {key} reference/hash differs from frozen metadata")
            path = Path(row[key])
            if not path.is_file() or sha256_file(path) != frozen.get(f"{key}_sha256"):
                raise ValueError(f"parent episode {i} actual {key} hash mismatch")
    with h5py.File(parent_h5_path, "r") as old, h5py.File(h5_path, "r") as new:
        for i in range(100):
            if not _same_h5_node(old[f"traj_{i}"], new[f"traj_{i}"]):
                raise ValueError(f"expanded trajectory {i} differs from immutable parent")
    collection = Path(registration["collection_experiment_dir"])
    prereg = collection / "preregistration.json"
    if sha256_file(prereg) != registration.get("preregistration_sha256"):
        raise ValueError("expanded collection preregistration hash mismatch")
    plan = json.loads(prereg.read_text()).get("candidate_plan", [])
    expansion_archive = collection / "source_archive"
    prereg_payload = json.loads(prereg.read_text())
    for relative, digest in prereg_payload.get("source_sha256", {}).items():
        # This is a historical collection: validate its sealed archive, not
        # today's live implementation, which may contain an offline reader fix.
        if sha256_file(expansion_archive / relative) != digest:
            raise ValueError(f"expanded archived source closure mismatch: {relative}")
    journal_path = Path(json_path).parent / "attempts.jsonl"
    if not journal_path.is_file() or sha256_file(journal_path) != registration.get("attempts_sha256"):
        raise ValueError("expanded attempts journal hash mismatch")
    try:
        journal_rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line]
    except (OSError, ValueError) as exc:
        raise ValueError("expanded attempts journal is invalid") from exc
    if journal_rows != registration.get("attempts"):
        raise ValueError("expanded attempts journal differs from manifest ledger")
    accepted = [row for row in registration.get("attempts", []) if row.get("accepted")]
    expected_yaws = np.random.default_rng(1704).uniform(-math.pi / 8, math.pi / 8, 300)[150:]
    if len(plan) != 150 or len(accepted) != 100:
        raise ValueError("expanded candidate plan/accepted ledger is incomplete")
    for i, candidate in enumerate(plan):
        if (candidate.get("seed") != 2026123150 + i
                or not math.isclose(float(candidate.get("target_yaw_rad", math.nan)),
                                    float(expected_yaws[i]), abs_tol=1e-7, rel_tol=0)):
            raise ValueError("expanded candidate plan differs from the sealed seed/yaw stream")
    previous_candidate_index = -1
    for j, row in enumerate(rows[100:]):
        attempt = accepted[j]
        candidate_index = int(attempt.get("candidate_index", -1)) - 1
        if (candidate_index < 0 or candidate_index >= len(plan)
                or candidate_index <= previous_candidate_index):
            raise ValueError("accepted candidate ledger is not in original candidate-plan order")
        previous_candidate_index = candidate_index
        candidate = plan[candidate_index]
        yaw = float(expected_yaws[candidate_index])
        if (row.get("lineage", {}).get("kind") != "new_candidate"
                or int(row.get("episode_seed", -1)) != int(attempt["seed"])
                or candidate.get("seed") != int(attempt["seed"])
                or attempt.get("seed") != candidate["seed"]
                or not math.isclose(float(candidate["target_yaw_rad"]), yaw, abs_tol=1e-12, rel_tol=0)
                # Consolidated metadata is sourced from the actual native target
                # state, whose yaw is stored as float32. Compare that row at its
                # storage precision; keep the sealed plan and ledger checks at
                # full float64 precision above/below.
                or np.float32(row["target_yaw_rad"]) != np.float32(yaw)
                or not math.isclose(float(attempt["target_yaw_rad"]), yaw, abs_tol=1e-12, rel_tol=0)):
            raise ValueError("new rows differ from the ordered first-eligible registered candidate ledger")
        source = Path(row["source_h5"])
        if not source.is_file() or sha256_file(source) != row.get("source_h5_sha256"):
            raise ValueError("expanded candidate source hash mismatch")
        replay, case_path = Path(row["replay_h5"]), Path(row["case_json"])
        if (not replay.is_file() or sha256_file(replay) != row.get("replay_h5_sha256")
                or not case_path.is_file() or sha256_file(case_path) != row.get("case_json_sha256")
                or row.get("source_h5_sha256") != attempt.get("source_h5_sha256")
                or row.get("replay_h5_sha256") != attempt.get("replay_h5_sha256")
                or Path(attempt.get("source_h5", "")).resolve() != source.resolve()
                or Path(attempt.get("replay_h5", "")).resolve() != replay.resolve()
                or Path(attempt.get("case_json", "")).resolve() != case_path.resolve()):
            raise ValueError("expanded candidate source/replay/case references do not match accepted ledger hashes")
        case_data = json.loads(case_path.read_text())
        if not math.isclose(float(case_data.get("target_yaw", math.nan)), yaw, abs_tol=1e-7, rel_tol=0):
            raise ValueError("expanded candidate case yaw differs from sealed plan")
        if (not data._native_episode_ok(source, case_data)[0]
                or not data._native_episode_ok(replay, case_data, replay=True)[0]
                or not data._pair_native_match(source, replay)):
            raise ValueError("expanded candidate fails native source/replay admission audit")
        with h5py.File(source, "r") as src, h5py.File(h5_path, "r") as combined:
            if not _same_h5_node(src["traj_0"], combined[f"traj_{100+j}"]):
                raise ValueError("expanded candidate differs from actual source trajectory")


def _validate_random300_lineage(h5_path, json_path, registration, manifest):
    """Bind the 300-row pool to the frozen 200 parent and first eligible additions."""
    from examples.baselines.constrain import plug_orientation_data as data
    parent = RANDOM200_DATASET
    parent_manifest_path = parent / "dataset_manifest.json"
    parent_h5, parent_json_path = parent / data.DATA_H5, parent / data.DATA_JSON
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_json = json.loads(parent_json_path.read_text())
    if (sha256_file(parent_manifest_path) != PARENT_RANDOM200_MANIFEST_SHA256
            or sha256_file(parent_h5) != PARENT_RANDOM200_H5_SHA256
            or sha256_file(parent_json_path) != PARENT_RANDOM200_JSON_SHA256
            or len(parent_json.get("episodes", [])) != 200
            or parent_manifest.get("status") != "complete"
            or parent_manifest.get("effective_episode_count") != 200):
        raise ValueError("actual frozen random-200 parent identity mismatch")
    parent_registration = parent_manifest
    _validate_random200_lineage(parent_h5, parent_json_path, parent_registration, parent_json)
    if (registration.get("protocol") != "plug_orientation_random300_expansion_v1"
            or registration.get("status") != "complete"
            or registration.get("accepted_count") != 300
            or registration.get("effective_episode_count") != 300
            or registration.get("new_accepted_count") != 100
            or registration.get("parent_dataset_manifest_sha256") != PARENT_RANDOM200_MANIFEST_SHA256
            or registration.get("parent_h5_sha256") != PARENT_RANDOM200_H5_SHA256
            or registration.get("parent_json_sha256") != PARENT_RANDOM200_JSON_SHA256
            or registration.get("expansion_contract_sha256") != sha256_file(
                Path(__file__).with_name("plug_orientation_adaptation_expansion300_contract.json"))):
        raise ValueError("300-episode registration lacks the exact frozen parent/contract lineage")
    prereg = Path(registration["collection_experiment_dir"]) / "preregistration.json"
    prereg_payload = json.loads(prereg.read_text())
    if sha256_file(prereg) != registration.get("preregistration_sha256"):
        raise ValueError("300-expansion preregistration hash mismatch")
    source_archive = prereg.parent / "source_archive"
    for relative, digest in prereg_payload.get("source_sha256", {}).items():
        if sha256_file(source_archive / relative) != digest:
            raise ValueError(f"300-expansion archived source mismatch: {relative}")
    plan = prereg_payload.get("candidate_plan", [])
    expected_yaws = np.random.default_rng(1704).uniform(-math.pi / 8, math.pi / 8, 450)[300:]
    if len(plan) != 150:
        raise ValueError("300-expansion candidate plan is incomplete")
    for i, row in enumerate(plan):
        if (row.get("seed") != 2026123300 + i
                or not math.isclose(float(row.get("target_yaw_rad", math.nan)),
                                    float(expected_yaws[i]), abs_tol=1e-12, rel_tol=0)):
            raise ValueError("300-expansion candidate plan differs from the frozen 450-draw stream")
    journal_path = Path(json_path).parent / "attempts.jsonl"
    if (not journal_path.is_file()
            or sha256_file(journal_path) != registration.get("attempts_sha256")):
        raise ValueError("300-expansion attempts journal hash mismatch")
    journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line]
    if journal != registration.get("attempts"):
        raise ValueError("300-expansion journal differs from manifest ledger")
    if not 100 <= len(journal) <= 150:
        raise ValueError("300-expansion attempt journal is outside its 100..150 bounded plan")
    for i, attempt in enumerate(journal):
        if (attempt.get("candidate_index") != i + 1
                or attempt.get("seed") != plan[i]["seed"]
                or not math.isclose(float(attempt.get("target_yaw_rad", math.nan)),
                                    float(expected_yaws[i]), abs_tol=1e-12, rel_tol=0)
                or attempt.get("expert_controls") is None or attempt.get("replay_controls") is None
                or max(int(attempt["expert_controls"]), int(attempt["replay_controls"])) > 600):
            raise ValueError("300-expansion attempt journal differs from its ordered, bounded plan")
    if (sum(int(row["expert_controls"]) for row in journal) > 90000
            or sum(int(row["replay_controls"]) for row in journal) > 90000
            or registration.get("new_source_controls") != sum(int(row["expert_controls"]) for row in journal)
            or registration.get("new_replay_controls") != sum(int(row["replay_controls"]) for row in journal)):
        raise ValueError("300-expansion current-attempt control totals exceed or differ from the sealed ledger")
    accepted = [row for row in journal if row.get("accepted")]
    if len(accepted) != 100:
        raise ValueError("300-expansion does not contain exactly 100 admitted additions")
    rows = json.loads(Path(json_path).read_text())["episodes"]
    if len(rows) != 300:
        raise ValueError("300-expansion training JSON row count mismatch")
    with h5py.File(parent_h5, "r") as original, h5py.File(h5_path, "r") as combined:
        for i in range(200):
            if not _same_h5_node(original[f"traj_{i}"], combined[f"traj_{i}"]):
                raise ValueError(f"300-expansion parent trajectory {i} changed")
            row, frozen = rows[i], parent_json["episodes"][i]
            lineage = row.get("lineage", {})
            if (lineage.get("kind") != "immutable_parent_episode"
                    or lineage.get("parent_episode_id") != i
                    or lineage.get("parent_manifest_sha256") != PARENT_RANDOM200_MANIFEST_SHA256
                    or lineage.get("parent_h5_sha256") != PARENT_RANDOM200_H5_SHA256
                    or lineage.get("parent_json_sha256") != PARENT_RANDOM200_JSON_SHA256
                    or lineage.get("parent_episode_metadata") != frozen
                    or any(row.get(key) != frozen.get(key) for key in
                           ("source_h5", "source_h5_sha256", "replay_h5", "replay_h5_sha256",
                            "case_json", "case_json_sha256", "source_actions_sha256"))
                    or np.float32(row.get("target_yaw_rad")) != np.float32(frozen["target_yaw_rad"])):
                raise ValueError(f"300-expansion parent row {i} differs from immutable random-200 values")
    previous = -1
    new_bin_counts = np.histogram([float(row["target_yaw_rad"]) for row in rows[200:]],
                                  bins=np.linspace(-math.pi / 8, math.pi / 8, 6))[0].tolist()
    effective_bin_counts = np.histogram([float(row["target_yaw_rad"]) for row in rows],
                                        bins=np.linspace(-math.pi / 8, math.pi / 8, 6))[0].tolist()
    if (registration.get("new_yaw_bin_counts") != new_bin_counts
            or registration.get("effective_yaw_bin_counts") != effective_bin_counts
            or not all(effective_bin_counts)):
        raise ValueError("300-expansion yaw-bin ledger differs from actual accepted rows")
    for j, attempt in enumerate(accepted):
        plan_index = int(attempt.get("candidate_index", -1)) - 1
        row = rows[200 + j]
        if (plan_index <= previous or plan_index < 0 or plan_index >= len(plan)
                or attempt.get("seed") != plan[plan_index]["seed"]
                or not math.isclose(float(attempt.get("target_yaw_rad", math.nan)),
                                    float(expected_yaws[plan_index]), abs_tol=1e-12, rel_tol=0)
                or row.get("lineage", {}).get("kind") != "new_candidate"
                or row.get("episode_seed") != attempt.get("seed")
                or np.float32(row.get("target_yaw_rad")) != np.float32(expected_yaws[plan_index])):
            raise ValueError("300-expansion rows differ from ordered first-eligible candidates")
        previous = plan_index
        for key in ("source_h5", "replay_h5", "case_json"):
            path = Path(row[key])
            if (not path.is_file() or sha256_file(path) != row.get(f"{key}_sha256")
                    or Path(attempt.get(key, "")).resolve() != path.resolve()
                    or attempt.get(f"{key}_sha256") != row.get(f"{key}_sha256")):
                raise ValueError("300-expansion candidate raw-file lineage mismatch")
        case = json.loads(Path(row["case_json"]).read_text())
        if (not math.isclose(float(case.get("target_yaw", math.nan)),
                             float(expected_yaws[plan_index]), abs_tol=1e-7, rel_tol=0)
                or row.get("source_actions_sha256") != attempt.get("source_actions_sha256")
                or not data._native_episode_ok(Path(row["source_h5"]), case)[0]
                or not data._native_episode_ok(Path(row["replay_h5"]), case, replay=True)[0]
                or not data._pair_native_match(row["source_h5"], row["replay_h5"])):
            raise ValueError("300-expansion candidate failed native source/replay admission audit")
        source = Path(row["source_h5"])
        with h5py.File(source, "r") as raw, h5py.File(h5_path, "r") as combined:
            if not _same_h5_node(raw["traj_0"], combined[f"traj_{200 + j}"]):
                raise ValueError("300-expansion candidate differs from admitted native source")


def _validate_random500_lineage(h5_path, json_path, registration, manifest):
    """Validate the actual immutable 300 pool and exact ordered 200 new rows."""
    from examples.baselines.constrain import plug_orientation_data as data
    parent = RANDOM300_DATASET
    parent_h5, parent_json_path = parent / data.DATA_H5, parent / data.DATA_JSON
    parent_manifest_path = parent / "dataset_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_json = json.loads(parent_json_path.read_text())
    if (sha256_file(parent_manifest_path) != PARENT_RANDOM300_MANIFEST_SHA256
            or sha256_file(parent_h5) != PARENT_RANDOM300_H5_SHA256
            or sha256_file(parent_json_path) != PARENT_RANDOM300_JSON_SHA256
            or len(parent_json.get("episodes", [])) != 300):
        raise ValueError("actual frozen random-300 parent identity mismatch")
    _validate_random300_lineage(parent_h5, parent_json_path, parent_manifest, parent_json)
    contract_sha = sha256_file(Path(__file__).with_name("plug_orientation_adaptation_expansion500_contract.json"))
    if (registration.get("protocol") != "plug_orientation_random500_expansion_v1"
            or registration.get("status") != "complete" or registration.get("accepted_count") != 500
            or registration.get("effective_episode_count") != 500
            or registration.get("new_accepted_count") != 200
            or registration.get("parent_dataset_manifest_sha256") != PARENT_RANDOM300_MANIFEST_SHA256
            or registration.get("parent_h5_sha256") != PARENT_RANDOM300_H5_SHA256
            or registration.get("parent_json_sha256") != PARENT_RANDOM300_JSON_SHA256
            or registration.get("expansion_contract_sha256") != contract_sha):
        raise ValueError("500 registration lacks exact immutable parent/contract lineage")
    prereg_path = Path(registration["collection_experiment_dir"]) / "preregistration.json"
    prereg = json.loads(prereg_path.read_text())
    if sha256_file(prereg_path) != registration.get("preregistration_sha256"):
        raise ValueError("500 preregistration hash mismatch")
    archive = prereg_path.parent / "source_archive"
    if prereg.get("source_sha256") != registration.get("source_archive_sha256"):
        raise ValueError("500 source archive identities differ between registration and preregistration")
    for name, digest in prereg.get("source_sha256", {}).items():
        if sha256_file(archive / name) != digest:
            raise ValueError(f"500 archived source mismatch: {name}")
    plan = random500_candidate_plan()
    if prereg.get("candidate_plan") != plan or prereg.get("candidate_cap") != 300:
        raise ValueError("500 candidate plan differs from sealed seed/yaw stream")
    journal_path = Path(json_path).parent / "attempts.jsonl"
    if sha256_file(journal_path) != registration.get("attempts_sha256"):
        raise ValueError("500 attempts journal hash mismatch")
    journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line]
    if journal != registration.get("attempts") or not 200 <= len(journal) <= 300:
        raise ValueError("500 journal differs from embedded ordered attempt ledger")
    for i, attempt in enumerate(journal):
        if (attempt.get("candidate_index") != i + 1 or attempt.get("seed") != plan[i]["seed"]
                or not math.isclose(float(attempt.get("target_yaw_rad", math.nan)),
                                    plan[i]["target_yaw_rad"], abs_tol=1e-12, rel_tol=0)
                or attempt.get("expert_controls") is None or attempt.get("replay_controls") is None
                or max(attempt["expert_controls"], attempt["replay_controls"]) > 600):
            raise ValueError("500 attempt is not a fully counted row in the sealed plan")
    if (sum(row["expert_controls"] for row in journal) > 180000
            or sum(row["replay_controls"] for row in journal) > 180000
            or registration.get("new_source_controls") != sum(row["expert_controls"] for row in journal)
            or registration.get("new_replay_controls") != sum(row["replay_controls"] for row in journal)):
        raise ValueError("500 new-stage native control totals exceed or differ from journal")
    accepted = [row for row in journal if row.get("accepted")]
    rows = json.loads(Path(json_path).read_text())["episodes"]
    if len(accepted) != 200 or len(rows) != 500:
        raise ValueError("500 dataset must contain exactly 300 parent and 200 admitted new rows")
    new_bins = np.histogram([float(row["target_yaw_rad"]) for row in rows[300:]],
                            bins=np.linspace(-math.pi / 8, math.pi / 8, 6))[0].tolist()
    effective_bins = np.histogram([float(row["target_yaw_rad"]) for row in rows],
                                  bins=np.linspace(-math.pi / 8, math.pi / 8, 6))[0].tolist()
    if (not all(new_bins) or not all(effective_bins)
            or registration.get("new_yaw_bin_counts") != new_bins
            or registration.get("effective_yaw_bin_counts") != effective_bins):
        raise ValueError("500 accepted new/effective yaw bins are missing or differ from the manifest")
    lineage_path = Path(registration.get("parent_lineage_path", ""))
    if (not lineage_path.is_file()
            or sha256_file(lineage_path) != registration.get("parent_lineage_sha256")):
        raise ValueError("500 parent-lineage record hash mismatch")
    lineage_record = json.loads(lineage_path.read_text())
    expected_ancestry = {
        "random200": _dataset_ancestry_identity(RANDOM200_DATASET),
        "random100": _dataset_ancestry_identity(RANDOM100_DATASET),
    }
    if (lineage_record.get("parent_manifest_sha256") != PARENT_RANDOM300_MANIFEST_SHA256
            or lineage_record.get("parent_h5_sha256") != PARENT_RANDOM300_H5_SHA256
            or lineage_record.get("parent_json_sha256") != PARENT_RANDOM300_JSON_SHA256
            or lineage_record.get("parent_tree_sha256") != registration.get("parent_tree_sha256")
            or lineage_record.get("parent_preregistration_sha256") != registration.get("parent_preregistration_sha256")
            or lineage_record.get("parent_prereg_source_sha256") != parent_manifest.get("source_archive_sha256")
            or lineage_record.get("parent_lineage_sha256") != sha256_file(parent / "parent_lineage.json")
            or lineage_record.get("random100_forward_contract_sha256") != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256
            or sha256_file(PARENT_RANDOM100_FORWARD_CONTRACT) != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256
            or lineage_record.get("ancestor_dataset_identities") != expected_ancestry):
        raise ValueError("500 parent-lineage record does not bind the 300/200/100 ancestry")
    with h5py.File(parent_h5, "r") as old, h5py.File(h5_path, "r") as combined:
        for i, frozen in enumerate(parent_json["episodes"]):
            row = rows[i]
            lineage = row.get("lineage", {})
            if (not _same_h5_node(old[f"traj_{i}"], combined[f"traj_{i}"])
                    or lineage.get("kind") != "immutable_parent_episode"
                    or lineage.get("parent_episode_id") != i
                    or lineage.get("parent_manifest_sha256") != PARENT_RANDOM300_MANIFEST_SHA256
                    or lineage.get("parent_episode_metadata") != frozen
                    or any(row.get(key) != frozen.get(key) for key in
                           ("source_h5", "source_h5_sha256", "replay_h5", "replay_h5_sha256",
                            "case_json", "case_json_sha256", "source_actions_sha256"))):
                raise ValueError(f"500 parent row {i} differs from immutable 300-pool values")
    prior = -1
    for j, attempt in enumerate(accepted):
        plan_i = attempt["candidate_index"] - 1
        row = rows[300 + j]
        yaw = plan[plan_i]["target_yaw_rad"]
        if (plan_i <= prior or attempt["seed"] != plan[plan_i]["seed"]
                or not math.isclose(attempt["target_yaw_rad"], yaw, abs_tol=1e-12, rel_tol=0)
                or row.get("lineage", {}).get("kind") != "new_candidate"
                or row.get("episode_seed") != attempt["seed"]
                or np.float32(row["target_yaw_rad"]) != np.float32(yaw)):
            raise ValueError("500 rows do not match first-eligible ordered candidates")
        prior = plan_i
        for key in ("source_h5", "replay_h5", "case_json"):
            path = Path(row[key])
            if (not path.is_file() or sha256_file(path) != row.get(f"{key}_sha256")
                    or path.resolve() != Path(attempt[key]).resolve()
                    or attempt.get(f"{key}_sha256") != row.get(f"{key}_sha256")):
                raise ValueError("500 candidate raw source/replay/case identity mismatch")
        case = json.loads(Path(row["case_json"]).read_text())
        if (not math.isclose(case["target_yaw"], yaw, abs_tol=1e-7, rel_tol=0)
                or not data._native_episode_ok(Path(row["source_h5"]), case)[0]
                or not data._native_episode_ok(Path(row["replay_h5"]), case, replay=True)[0]
                or not data._pair_native_match(row["source_h5"], row["replay_h5"])):
            raise ValueError("500 candidate fails official source/replay admission")
        with h5py.File(row["source_h5"], "r") as source, h5py.File(h5_path, "r") as combined:
            if not _same_h5_node(source["traj_0"], combined[f"traj_{300+j}"]):
                raise ValueError("500 consolidated candidate differs from source trajectory")


def collect_random200(dataset_dir, exp_dir, *, parent_dataset=RANDOM100_DATASET,
                      run_case_fn=None, seed=1, _candidate_count=150,
                      _accepted_count=100, _wall_seconds=5400,
                      _parent_count=100):
    """Create a 200-effective-demo pool: immutable parent 100 + first fresh 100.

    Private underscored parameters are reserved for local fixture tests; the CLI
    exposes no production budget or parent override.
    """
    from examples.baselines.constrain import plug_orientation_data as data
    from examples.baselines.constrain.run_plug_ee_lift_v4 import SOURCE_PATHS

    dataset_dir, exp_dir, parent_dataset = map(Path, (dataset_dir, exp_dir, parent_dataset))
    expected_limits = ((150, 100, 5400, 100) if _parent_count == 100 else
                       (150, 100, 5400, 200) if _parent_count == 200 else
                       (300, 200, 10800, 300))
    if (_candidate_count, _accepted_count, _wall_seconds, _parent_count) != expected_limits:
        if "PYTEST_CURRENT_TEST" not in os.environ:
            raise ValueError("random expansion production budgets are frozen")
    parent_manifest_path = parent_dataset / "dataset_manifest.json"
    parent_json_path = parent_dataset / data.DATA_JSON
    parent_h5_path = parent_dataset / data.DATA_H5
    parent_manifest = json.loads(parent_manifest_path.read_text())
    parent_json = json.loads(parent_json_path.read_text())
    expected_parent_protocol = ({100: "plug_orientation_random100_v1",
        200: "plug_orientation_random200_expansion_v1",
        300: "plug_orientation_random300_expansion_v1"}.get(_parent_count, "plug_orientation_random100_v1"))
    expected_parent_manifest_sha = {100: PARENT_RANDOM100_MANIFEST_SHA256,
        200: PARENT_RANDOM200_MANIFEST_SHA256, 300: PARENT_RANDOM300_MANIFEST_SHA256}.get(
            _parent_count, PARENT_RANDOM100_MANIFEST_SHA256)
    expected_parent_h5_sha = {100: PARENT_RANDOM100_H5_SHA256,
        200: PARENT_RANDOM200_H5_SHA256, 300: PARENT_RANDOM300_H5_SHA256}.get(
            _parent_count, PARENT_RANDOM100_H5_SHA256)
    expected_parent_json_sha = {100: PARENT_RANDOM100_JSON_SHA256,
        200: PARENT_RANDOM200_JSON_SHA256, 300: PARENT_RANDOM300_JSON_SHA256}.get(
            _parent_count, PARENT_RANDOM100_JSON_SHA256)
    if (parent_manifest.get("protocol") != expected_parent_protocol
            or parent_manifest.get("accepted_count") != _parent_count
            or len(parent_json.get("episodes", [])) != _parent_count
            or (_parent_count in {100, 200} and (sha256_file(parent_h5_path) != expected_parent_h5_sha
                or sha256_file(parent_json_path) != expected_parent_json_sha
                or sha256_file(parent_manifest_path) != expected_parent_manifest_sha))):
        raise ValueError(("parent random-100 dataset does not match its original random-100 registration"
                          if _parent_count == 100 else f"parent random-{_parent_count} dataset does not match its frozen identity"))
    if ((parent_manifest.get("training_h5_sha256") is not None
            and parent_manifest["training_h5_sha256"] != sha256_file(parent_h5_path))
            or (parent_manifest.get("training_json_sha256") is not None
                and parent_manifest["training_json_sha256"] != sha256_file(parent_json_path))):
        raise ValueError("parent consolidated HDF5/JSON does not match its manifest hash")
    parent_tree = _tree_hashes(parent_dataset)
    parent_manifest_sha = sha256_file(parent_manifest_path)
    parent_h5_sha, parent_json_sha = sha256_file(parent_h5_path), sha256_file(parent_json_path)
    parent_collection_dir = Path(parent_manifest.get("collection_experiment_dir", ""))
    parent_prereg_path = parent_collection_dir / "preregistration.json"
    parent_prereg_sha = parent_manifest.get("preregistration_sha256")
    if (not parent_prereg_path.is_file() or not parent_prereg_sha
            or sha256_file(parent_prereg_path) != parent_prereg_sha):
        raise ValueError("parent collection preregistration is missing or does not match the frozen hash")
    expected_parent_prereg_sha = (PARENT_RANDOM200_PREREGISTRATION_SHA256 if _parent_count == 200
                                  else PARENT_RANDOM100_PREREGISTRATION_SHA256)
    if _parent_count in {100, 200} and parent_prereg_sha != expected_parent_prereg_sha:
        raise ValueError(f"parent preregistration is not the frozen random-{_parent_count} registration")
    if _parent_count == 100 and not all(parent_manifest.get("accepted_yaw_bin_counts", [])):
        raise ValueError("frozen parent does not satisfy the effective five-bin yaw coverage gate")
    parent_source_archive = Path(parent_manifest.get("collection_source_archive") or
                                 (parent_collection_dir / "source_archive"))
    parent_prereg = json.loads(parent_prereg_path.read_text())
    for relative, digest in parent_prereg.get("source_sha256", {}).items():
        archived_file = parent_source_archive / relative
        if not archived_file.is_file() or sha256_file(archived_file) != digest:
            raise ValueError(f"parent source archive mismatch: {relative}")
    is_random300 = _parent_count == 200
    is_random500 = _parent_count == 300
    ancestor_dataset_identities = ({
        "random200": _dataset_ancestry_identity(RANDOM200_DATASET),
        "random100": _dataset_ancestry_identity(RANDOM100_DATASET),
    } if is_random500 else None)
    candidates = ((random500_candidate_plan(seed) if is_random500 else
                   random300_candidate_plan(seed) if is_random300 else
                   random200_candidate_plan(seed))[:_candidate_count])
    bin_edges = np.linspace(-math.pi / 8, math.pi / 8, 6)
    dataset_dir.mkdir(parents=True, exist_ok=False); exp_dir.mkdir(parents=True, exist_ok=False)
    if run_case_fn is None:
        from examples.baselines.constrain.plug_orientation_ee_expert import run_case
        run_case_fn = run_case
    source_files = list(SOURCE_PATHS) + [
        "examples/baselines/constrain/plug_orientation_data.py",
        "examples/baselines/constrain/plug_orientation_adaptation.py",
        "examples/baselines/constrain/plug_orientation_base_dp.py",
        "examples/baselines/constrain/plug_orientation_adaptation_contract.json",
        "examples/baselines/constrain/plug_orientation_adaptation_expansion200_contract.json"]
    if is_random300:
        source_files.append("examples/baselines/constrain/plug_orientation_adaptation_expansion300_contract.json")
    if is_random500:
        source_files.extend(["examples/baselines/constrain/plug_orientation_adaptation_expansion300_contract.json",
            "examples/baselines/constrain/plug_orientation_adaptation_expansion500_contract.json"])
    archive = exp_dir / "source_archive"; archive.mkdir()
    source_hashes = {}
    for rel in source_files:
        dest = archive / rel; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / rel).read_bytes()); source_hashes[rel] = sha256_file(ROOT / rel)
        if sha256_file(dest) != source_hashes[rel]: raise RuntimeError(f"source archive mismatch: {rel}")
    expansion_contract_path = Path(__file__).with_name(
        "plug_orientation_adaptation_expansion500_contract.json" if is_random500 else
        "plug_orientation_adaptation_expansion300_contract.json" if is_random300
        else "plug_orientation_adaptation_expansion200_contract.json")
    expansion_sha = sha256_file(expansion_contract_path)
    prereg = {"format": f"plug_orientation_random{_parent_count + _accepted_count}_expansion_preregistration_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": f"plug_orientation_random{_parent_count + _accepted_count}_expansion_v1", "seed": seed,
        "candidate_plan": candidates, "candidate_cap": _candidate_count,
        "accepted_new_cap": _accepted_count, "effective_episode_count": _parent_count + _accepted_count,
        "yaw_rng_policy": ("np.default_rng(1704): sequential uniform draws; skip draws 0..449; use draws 450..749"
                            if is_random500 else "np.default_rng(1704): sequential uniform draws; skip draws 0..299; use draws 300..449"
                            if is_random300 else "np.default_rng(1704): sequential uniform draws; skip draws 0..149 from the same algorithm used by actual random-100 run; use draws 150..299"),
        "yaw_rng_draw_start": 450 if is_random500 else 300 if is_random300 else 150, "yaw_bin_edges_rad": bin_edges.tolist(),
        "parent_dataset": str(parent_dataset.resolve()), "parent_tree_sha256": parent_tree,
        "parent_manifest_sha256": parent_manifest_sha, "parent_h5_sha256": parent_h5_sha,
        "parent_json_sha256": parent_json_sha, "parent_preregistration_sha256": parent_prereg_sha,
        "parent_preregistration_path": str(parent_prereg_path.resolve()),
        "parent_source_archive": str(parent_source_archive.resolve()),
        "expansion_contract_sha256": expansion_sha,
        "frozen_base_sha256": BASE_SHA256, "source_sha256": source_hashes,
        "per_candidate_control_cap": 600, "aggregate_source_control_cap": 180000 if is_random500 else 90000,
        "aggregate_replay_control_cap": 180000 if is_random500 else 90000, "wall_clock_cap_seconds": _wall_seconds,
        "whole_stage_wall_clock_cap_seconds": 12000 if is_random500 else 6000}
    prereg_path = exp_dir / "preregistration.json"
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True, allow_nan=False) + "\n")
    prereg_sha = sha256_file(prereg_path)
    original_attempts = [row for row in parent_manifest.get("attempts", []) if row.get("accepted")]
    if is_random500:
        _validate_random300_lineage(parent_h5_path, parent_json_path, parent_manifest, parent_json)
        if sha256_file(PARENT_RANDOM100_FORWARD_CONTRACT) != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256:
            raise ValueError("frozen original random-100 forward contract identity mismatch")
        selected_parent = []
        for i, metadata in enumerate(parent_json["episodes"]):
            selected_parent.append({"accepted": True, "seed": int(metadata["episode_seed"]),
                "target_yaw_rad": float(metadata["target_yaw_rad"]),
                "source_h5": metadata["source_h5"], "replay_h5": metadata["replay_h5"],
                "case_json": metadata["case_json"], "parent_episode_id": i})
    elif is_random300:
        _validate_random200_lineage(parent_h5_path, parent_json_path, parent_manifest, parent_json)
        selected_parent = []
        for i, metadata in enumerate(parent_json["episodes"]):
            selected_parent.append({"accepted": True, "seed": int(metadata["episode_seed"]),
                "target_yaw_rad": float(metadata["target_yaw_rad"]),
                "source_h5": metadata["source_h5"], "replay_h5": metadata["replay_h5"],
                "case_json": metadata["case_json"], "parent_episode_id": i})
    else:
        if len(original_attempts) < _parent_count:
            raise ValueError("parent manifest lacks its accepted lineage rows")
        selected_parent = original_attempts[:_parent_count]
    if _parent_count == 100:
        with h5py.File(parent_h5_path, "r") as consolidated_parent:
            for episode_id, (attempt, metadata) in enumerate(zip(selected_parent, parent_json["episodes"])):
                row = dict(attempt)
                for key in ("source_h5", "replay_h5", "case_json"):
                    path = Path(row[key])
                    row[key] = str((path if path.is_absolute() else ROOT / path).resolve())
                case = json.loads(Path(row["case_json"]).read_text())
                if (int(row["seed"]) != int(metadata["episode_seed"])
                            or not math.isclose(float(row["target_yaw_rad"]), float(metadata["target_yaw_rad"]), abs_tol=1e-6)
                        or sha256_file(row["source_h5"]) != metadata["source_h5_sha256"]
                        or sha256_file(row["replay_h5"]) != metadata["replay_h5_sha256"]
                        or sha256_file(row["case_json"]) != metadata["case_json_sha256"]
                        or not math.isclose(float(case.get("target_yaw", math.nan)),
                                            float(metadata["target_yaw_rad"]), abs_tol=1e-7, rel_tol=0)):
                    raise ValueError(f"parent episode {episode_id} raw source/replay/case identity mismatch")
                with h5py.File(row["source_h5"], "r") as source:
                    actions = np.asarray(source["traj_0/actions"][:])
                    if (hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
                            != metadata["source_actions_sha256"]
                            or not _same_h5_node(source["traj_0"], consolidated_parent[f"traj_{episode_id}"])):
                        raise ValueError(f"parent episode {episode_id} source differs from frozen consolidated values")
                if (not data._native_episode_ok(Path(row["source_h5"]), case)[0]
                        or not data._native_episode_ok(Path(row["replay_h5"]), case, replay=True)[0]
                        or not data._pair_native_match(row["source_h5"], row["replay_h5"])):
                    raise ValueError(f"parent episode {episode_id} fails the frozen native source/replay audit")
                row["parent_episode_id"] = episode_id
                selected_parent[episode_id] = row
    lineage = {"parent_dataset": str(parent_dataset.resolve()), "parent_manifest_sha256": parent_manifest_sha,
        "parent_h5_sha256": parent_h5_sha, "parent_json_sha256": parent_json_sha,
        "parent_preregistration_sha256": parent_prereg_sha,
        "parent_source_archive": str(parent_source_archive.resolve()),
        "parent_prereg_source_sha256": parent_prereg.get("source_sha256", {}),
        "parent_tree_sha256": parent_tree, "parent_episode_rows": parent_json["episodes"],
        "parent_candidate_attempt_count": len(parent_manifest.get("attempts", [])),
        "parent_accepted_count": _parent_count,
        "parent_attempt_ledger": parent_manifest.get("attempts", [])}
    if is_random500:
        lineage["parent_lineage_sha256"] = sha256_file(parent_dataset / "parent_lineage.json")
        lineage["random100_forward_contract_sha256"] = PARENT_RANDOM100_FORWARD_CONTRACT_SHA256
        lineage["ancestor_dataset_identities"] = ancestor_dataset_identities
    (dataset_dir / "parent_lineage.json").write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n")
    attempts, accepted_new, source_controls, replay_controls = [], [], 0, 0
    started = time.monotonic()
    journal_path = dataset_dir / "attempts.jsonl"
    with journal_path.open("x") as journal:
        for index, candidate in enumerate(candidates):
            if time.monotonic() - started >= _wall_seconds: break
            if sha256_file(prereg_path) != prereg_sha: raise RuntimeError("expansion preregistration changed")
            if _tree_hashes(parent_dataset) != parent_tree:
                raise RuntimeError("immutable parent data changed during expansion")
            if sha256_file(parent_prereg_path) != parent_prereg_sha:
                raise RuntimeError("parent collection preregistration changed during expansion")
            if any(sha256_file(parent_source_archive / relative) != digest
                   for relative, digest in parent_prereg.get("source_sha256", {}).items()):
                raise RuntimeError("parent source archive changed during expansion")
            if any(sha256_file(archive / name) != digest or sha256_file(ROOT / name) != digest
                   for name, digest in source_hashes.items()):
                raise RuntimeError("expansion source identity changed")
            seed_i, yaw = candidate["seed"], candidate["target_yaw_rad"]
            out = dataset_dir / f"candidate_{index + 1:03d}_seed_{seed_i}"
            row = {**candidate, "candidate_index": index + 1, "status": "failed", "accepted": False,
                   "source_h5": str(out / "expert.h5"), "replay_h5": str(out / "replay/replay.h5")}
            try:
                remaining = _wall_seconds - (time.monotonic() - started)
                with data._term_as_timeout("SIGTERM during random-200 candidate"), data._collection_deadline(remaining):
                    case = run_case_fn(seed_i, out, target_yaw=yaw, deadline=started + _wall_seconds)
                case_path = out / "case.json"
                if not case_path.is_file(): case_path.write_text(json.dumps(case, indent=2, allow_nan=False) + "\n")
                row.update(case_json=str(case_path), expert_controls=int(case["expert_controls"]),
                           replay_controls=int(case["replay_controls"]))
                if is_random300 or is_random500:
                    row["case_json_sha256"] = sha256_file(case_path)
                if row["expert_controls"] < 0 or row["replay_controls"] < 0:
                    raise ValueError("teacher/replay control counts must be nonnegative")
                if max(row["expert_controls"], row["replay_controls"]) > 600:
                    raise RuntimeError("native teacher exceeded 600-control episode cap")
                row["source_h5_sha256"] = sha256_file(row["source_h5"]) if Path(row["source_h5"]).is_file() else None
                row["replay_h5_sha256"] = sha256_file(row["replay_h5"]) if Path(row["replay_h5"]).is_file() else None
                if (is_random300 or is_random500) and row["source_h5_sha256"]:
                    with h5py.File(row["source_h5"], "r") as source_h5:
                        row["source_actions_sha256"] = hashlib.sha256(
                            np.ascontiguousarray(source_h5["traj_0/actions"][:]).tobytes()).hexdigest()
                source_ok = data._native_episode_ok(Path(row["source_h5"]), case)[0] if row["source_h5_sha256"] else False
                replay_ok = data._native_episode_ok(Path(row["replay_h5"]), case, replay=True)[0] if row["replay_h5_sha256"] else False
                matched = source_ok and replay_ok and data._pair_native_match(row["source_h5"], row["replay_h5"])
                if time.monotonic() - started >= _wall_seconds:
                    matched = False
                    row["rejection_reason"] = "collection_deadline_expired_after_validation"
                row.update(accepted=bool(matched), status="accepted" if matched else "rejected",
                           rejection_reason=None if matched else row.get("rejection_reason",
                               "official_success_or_native_pair_validation_failed"))
                if matched:
                    row.update(source_sha256=row["source_h5_sha256"], replay_sha256=row["replay_h5_sha256"])
                    accepted_new.append(row)
                source_controls += row["expert_controls"]; replay_controls += row["replay_controls"]
            except BaseException as exc:
                known = type(exc) is RuntimeError and str(exc) in {
                    "MPLib screw planning failed", "native teacher exceeded 600-control episode cap"}
                row.update(error=f"{type(exc).__name__}: {exc}",
                           failure_class="known_planning_failure" if known else "unexpected_exception")
                partial = out / "partial_failure.json"
                row["partial_failure"] = str(partial) if partial.is_file() else None
                case_path = out / "case.json"
                if case_path.is_file():
                    row["case_json"] = str(case_path)
                    row["case_json_sha256"] = sha256_file(case_path)
                for key, path in (("source_h5_sha256", row["source_h5"]),
                                  ("replay_h5_sha256", row["replay_h5"])):
                    row[key] = sha256_file(path) if Path(path).is_file() else None
                partial_data = {}
                try:
                    partial_data = json.loads(partial.read_text()) if partial.is_file() else {}
                except (OSError, ValueError, TypeError):
                    partial_data = {}
                replay_not_started = (known and partial_data.get("stage") == "expert_failed_before_replay"
                    and partial_data.get("replay_status") == "not_started"
                    and partial_data.get("replay_controls") == 0)
                replay_count = (0 if replay_not_started else
                    row.get("replay_controls", partial_data.get("replay_controls")))
                row["replay_controls"] = int(replay_count) if replay_count is not None else None
                if row["replay_controls"] is not None:
                    replay_controls += row["replay_controls"]
                try:
                    control_count = row.get("expert_controls", partial_data.get("expert_controls"))
                    row["expert_controls"] = int(control_count)
                    source_controls += row["expert_controls"]
                except (OSError, ValueError, KeyError, TypeError):
                    row["expert_controls"] = None
                    if known:
                        attempts.append(row)
                        journal.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                        journal.flush(); os.fsync(journal.fileno())
                        summary = {"attempts": attempts, "accepted_new_count": len(accepted_new),
                            "source_controls_known": source_controls, "replay_controls_known": replay_controls,
                            "unknown_source_control_attempts": sum(r.get("expert_controls") is None for r in attempts),
                            "unknown_replay_control_attempts": sum(r.get("replay_controls") is None for r in attempts),
                            "error": row["error"]}
                        (dataset_dir / "partial_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
                        raise RuntimeError("known teacher failure lacks auditable control count") from exc
                if not known:
                    attempts.append(row); journal.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                    journal.flush(); os.fsync(journal.fileno())
                    (dataset_dir / "partial_summary.json").write_text(json.dumps({"attempts": attempts,
                        "accepted_new_count": len(accepted_new), "source_controls_known": source_controls,
                        "replay_controls_known": replay_controls, "error": row["error"]}, indent=2)+"\n")
                    raise
                if row["replay_controls"] is None:
                    attempts.append(row); journal.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
                    journal.flush(); os.fsync(journal.fileno())
                    _write_json(dataset_dir / "partial_summary.json", {"attempts": attempts,
                        "accepted_new_count": len(accepted_new), "source_controls_known": source_controls,
                        "replay_controls_known": replay_controls,
                        "unknown_source_control_attempts": sum(r.get("expert_controls") is None for r in attempts),
                        "unknown_replay_control_attempts": sum(r.get("replay_controls") is None for r in attempts),
                        "error": row["error"]})
                    raise RuntimeError("known teacher failure lacks auditable replay control count") from exc
            attempts.append(row); journal.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
            journal.flush(); os.fsync(journal.fileno())
            _write_json(dataset_dir / "progress.json", {"attempts": attempts,
                "accepted_new_count": len(accepted_new), "source_controls": source_controls,
                "replay_controls": replay_controls})
            aggregate_cap = 180000 if is_random500 else 90000
            if source_controls > aggregate_cap or replay_controls > aggregate_cap:
                (dataset_dir / "partial_summary.json").write_text(json.dumps({"attempts": attempts,
                    "accepted_new_count": len(accepted_new), "source_controls_known": source_controls,
                    "replay_controls_known": replay_controls,
                    "error": "aggregate source/replay native-control cap exceeded"}, indent=2)+"\n")
                raise RuntimeError("aggregate source/replay native-control cap exceeded")
            if len(accepted_new) == _accepted_count: break
    new_bins = np.histogram([row["target_yaw_rad"] for row in accepted_new], bins=bin_edges)[0].tolist()
    admission_complete = (len(accepted_new) == _accepted_count
                          and (_parent_count != 300 or all(new_bins)))
    if len(accepted_new) == _accepted_count and time.monotonic() - started >= _wall_seconds:
        _write_json(dataset_dir / "partial_summary.json", {"attempts": attempts,
            "accepted_new_count": len(accepted_new), "source_controls_known": source_controls,
            "replay_controls_known": replay_controls,
            "unknown_source_control_attempts": sum(r.get("expert_controls") is None for r in attempts),
            "unknown_replay_control_attempts": sum(r.get("replay_controls") is None for r in attempts),
            "error": "collection deadline expired before expansion consolidation"})
        raise TimeoutError("random-200 collection deadline expired before expansion consolidation")
    if (admission_complete
            and (sha256_file(prereg_path) != prereg_sha
                 or _tree_hashes(parent_dataset) != parent_tree
                 or sha256_file(parent_prereg_path) != parent_prereg_sha
                 or any(sha256_file(archive / name) != digest or sha256_file(ROOT / name) != digest
                        for name, digest in source_hashes.items())
                  or any(sha256_file(parent_source_archive / relative) != digest
                         for relative, digest in parent_prereg.get("source_sha256", {}).items())
                  or (is_random500 and ancestor_dataset_identities != {
                      "random200": _dataset_ancestry_identity(RANDOM200_DATASET),
                      "random100": _dataset_ancestry_identity(RANDOM100_DATASET)})
                  or (is_random500 and sha256_file(PARENT_RANDOM100_FORWARD_CONTRACT)
                      != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256))):
        raise RuntimeError("sealed inputs changed before expansion consolidation")
    if admission_complete and time.monotonic() - started < _wall_seconds:
        combined_rows = selected_parent + accepted_new
        effective_count = _parent_count + _accepted_count
        combined = data.consolidate_attempts(dataset_dir, combined_rows, effective_count)
        for i, row in enumerate(combined["episodes"][:_parent_count]):
            row["lineage"] = {"kind": "immutable_parent_episode", "parent_episode_id": i,
                "parent_manifest_sha256": parent_manifest_sha, "parent_h5_sha256": parent_h5_sha,
                "parent_json_sha256": parent_json_sha, "parent_episode_metadata": parent_json["episodes"][i]}
        for row in combined["episodes"][_parent_count:]: row["lineage"] = {"kind": "new_candidate"}
        _write_json(dataset_dir / data.DATA_JSON, combined)
        coverage = [row["target_yaw_rad"] for row in combined["episodes"]]
        bins = np.histogram(coverage, bins=bin_edges)[0].tolist()
        manifest = {"dataset": dataset_dir.name,
            "protocol": f"plug_orientation_random{effective_count}_expansion_v1",
            "status": "complete", "accepted_count": effective_count, "effective_episode_count": effective_count,
            "new_accepted_count": _accepted_count, "new_candidate_attempt_count": len(attempts),
            "parent_candidate_attempt_count": len(parent_manifest.get("attempts", [])),
            "parent_accepted_count": _parent_count,
            "new_source_controls": source_controls, "new_replay_controls": replay_controls,
            "attempts_sha256": sha256_file(journal_path), "attempts": attempts,
            "parent_lineage_path": str((dataset_dir / "parent_lineage.json").resolve()),
            "parent_lineage_sha256": sha256_file(dataset_dir / "parent_lineage.json"),
            "parent_dataset_manifest_sha256": parent_manifest_sha, "parent_h5_sha256": parent_h5_sha,
            "parent_json_sha256": parent_json_sha, "parent_tree_sha256": parent_tree,
            "parent_preregistration_sha256": parent_prereg_sha,
            "expansion_contract_sha256": expansion_sha, "preregistration_sha256": prereg_sha,
            "collection_experiment_dir": str(exp_dir.resolve()), "new_yaw_bin_counts": new_bins,
            "effective_yaw_bin_counts": bins, "training_h5_sha256": sha256_file(dataset_dir / data.DATA_H5),
            "training_json_sha256": sha256_file(dataset_dir / data.DATA_JSON),
            "training_h5_path": str((dataset_dir / data.DATA_H5).resolve()),
            "training_json_path": str((dataset_dir / data.DATA_JSON).resolve()),
            "training_coverage": coverage, "source_archive_sha256": source_hashes}
    else:
        manifest = {"dataset": dataset_dir.name,
            "protocol": f"plug_orientation_random{_parent_count + len(accepted_new)}_expansion_v1",
            "status": "incomplete", "accepted_count": _parent_count + len(accepted_new),
            "new_accepted_count": len(accepted_new), "new_candidate_attempt_count": len(attempts),
            "parent_candidate_attempt_count": len(parent_manifest.get("attempts", [])),
            "parent_accepted_count": _parent_count,
            "attempts_sha256": sha256_file(journal_path), "attempts": attempts,
            "parent_dataset_manifest_sha256": parent_manifest_sha, "expansion_contract_sha256": expansion_sha,
            "unknown_source_control_attempts": sum(r.get("expert_controls") is None for r in attempts),
            "unknown_replay_control_attempts": sum(r.get("replay_controls") is None for r in attempts)}
    if time.monotonic() - started >= _wall_seconds:
        _write_json(dataset_dir / "partial_summary.json", {"attempts": attempts,
            "accepted_new_count": len(accepted_new), "source_controls_known": source_controls,
            "replay_controls_known": replay_controls,
            "unknown_source_control_attempts": sum(r.get("expert_controls") is None for r in attempts),
            "unknown_replay_control_attempts": sum(r.get("replay_controls") is None for r in attempts),
            "error": "collection deadline expired before manifest publication"})
        raise TimeoutError("random-200 collection deadline expired before manifest publication")
    if (sha256_file(prereg_path) != prereg_sha or _tree_hashes(parent_dataset) != parent_tree
            or sha256_file(parent_prereg_path) != parent_prereg_sha
            or any(sha256_file(archive / name) != digest or sha256_file(ROOT / name) != digest
                   for name, digest in source_hashes.items())
            or any(sha256_file(parent_source_archive / relative) != digest
                   for relative, digest in parent_prereg.get("source_sha256", {}).items())
            or (is_random500 and ancestor_dataset_identities != {
                "random200": _dataset_ancestry_identity(RANDOM200_DATASET),
                "random100": _dataset_ancestry_identity(RANDOM100_DATASET)})
            or (is_random500 and sha256_file(PARENT_RANDOM100_FORWARD_CONTRACT)
                != PARENT_RANDOM100_FORWARD_CONTRACT_SHA256)):
        raise RuntimeError("immutable parent data changed during expansion")
    _write_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest


def collect_random300(dataset_dir, exp_dir, *, parent_dataset=RANDOM200_DATASET,
                      run_case_fn=None, seed=1, _candidate_count=150,
                      _accepted_count=100, _wall_seconds=5400):
    """Append the first 100 eligible fresh episodes to the immutable random-200 pool."""
    if Path(parent_dataset).resolve() != RANDOM200_DATASET.resolve():
        raise ValueError("production random-300 collection is pinned to the immutable random-200 parent")
    return collect_random200(dataset_dir, exp_dir, parent_dataset=RANDOM200_DATASET,
        run_case_fn=run_case_fn, seed=seed, _candidate_count=_candidate_count,
        _accepted_count=_accepted_count, _wall_seconds=_wall_seconds, _parent_count=200)


def collect_random500(dataset_dir, exp_dir, *, parent_dataset=RANDOM300_DATASET,
                      run_case_fn=None, seed=1, _candidate_count=300,
                      _accepted_count=200, _wall_seconds=10800):
    """Append first 200 eligible candidates to the immutable random-300 pool."""
    if Path(parent_dataset).resolve() != RANDOM300_DATASET.resolve():
        raise ValueError("production random-500 collection is pinned to the immutable random-300 parent")
    return collect_random200(dataset_dir, exp_dir, parent_dataset=RANDOM300_DATASET,
        run_case_fn=run_case_fn, seed=seed, _candidate_count=_candidate_count,
        _accepted_count=_accepted_count, _wall_seconds=_wall_seconds, _parent_count=300)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    contract_parser = commands.add_parser("write-contract")
    contract_parser.add_argument("--output", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--random-h5", required=True); train_parser.add_argument("--random-manifest", required=True)
    train_parser.add_argument("--fixed-h5", required=True); train_parser.add_argument("--fixed-manifest", required=True)
    train_parser.add_argument("--exp-dir", required=True); train_parser.add_argument("--ckpt-dir", required=True)
    train_parser.add_argument("--device", default="cuda")
    expanded_train = commands.add_parser("train-expanded")
    expanded_train.add_argument("--random-h5", required=True); expanded_train.add_argument("--random-manifest", required=True)
    expanded_train.add_argument("--fixed-h5", required=True); expanded_train.add_argument("--fixed-manifest", required=True)
    expanded_train.add_argument("--exp-dir", required=True); expanded_train.add_argument("--ckpt-dir", required=True)
    expanded_train.add_argument("--device", default="cuda")
    expanded300_train = commands.add_parser("train-expanded300")
    expanded300_train.add_argument("--random-h5", required=True); expanded300_train.add_argument("--random-manifest", required=True)
    expanded300_train.add_argument("--fixed-h5", required=True); expanded300_train.add_argument("--fixed-manifest", required=True)
    expanded300_train.add_argument("--exp-dir", required=True); expanded300_train.add_argument("--ckpt-dir", required=True)
    expanded300_train.add_argument("--device", default="cuda")
    expanded500_train = commands.add_parser("train-expanded500")
    expanded500_train.add_argument("--random-h5", required=True); expanded500_train.add_argument("--random-manifest", required=True)
    expanded500_train.add_argument("--fixed-h5", required=True); expanded500_train.add_argument("--fixed-manifest", required=True)
    expanded500_train.add_argument("--exp-dir", required=True); expanded500_train.add_argument("--ckpt-dir", required=True)
    expanded500_train.add_argument("--device", default="cuda")
    collector = commands.add_parser("collect-random100")
    collector.add_argument("--dataset-dir", required=True); collector.add_argument("--exp-dir", required=True)
    collector.add_argument("--seed", type=int, default=1)
    collector200 = commands.add_parser("collect-random200")
    collector200.add_argument("--parent-dataset", required=True)
    collector200.add_argument("--dataset-dir", required=True); collector200.add_argument("--exp-dir", required=True)
    collector300 = commands.add_parser("collect-random300")
    collector300.add_argument("--parent-dataset", required=True)
    collector300.add_argument("--dataset-dir", required=True); collector300.add_argument("--exp-dir", required=True)
    collector500 = commands.add_parser("collect-random500")
    collector500.add_argument("--parent-dataset", required=True)
    collector500.add_argument("--dataset-dir", required=True); collector500.add_argument("--exp-dir", required=True)
    validation = commands.add_parser("evaluate-validation")
    validation.add_argument("--ckpt-dir", required=True); validation.add_argument("--random-catalog", required=True)
    validation.add_argument("--fixed-catalog", required=True); validation.add_argument("--output-root", required=True)
    heldout = commands.add_parser("evaluate-heldout",
        help="run the immutable 600-arm, CPU paired heldout roster (maximum 25,200 s outer budget)")
    heldout.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "write-contract":
        write_adaptation_contract(args.output)
    elif args.command == "collect-random100":
        collect_random100(args.dataset_dir, args.exp_dir, seed=args.seed)
    elif args.command == "collect-random200":
        if Path(args.parent_dataset).resolve() != RANDOM100_DATASET.resolve():
            raise ValueError("production random-200 collection is pinned to the immutable random-100 parent")
        collect_random200(args.dataset_dir, args.exp_dir, parent_dataset=RANDOM100_DATASET)
    elif args.command == "collect-random300":
        if Path(args.parent_dataset).resolve() != RANDOM200_DATASET.resolve():
            raise ValueError("production random-300 collection is pinned to the immutable random-200 parent")
        collect_random300(args.dataset_dir, args.exp_dir, parent_dataset=RANDOM200_DATASET)
    elif args.command == "collect-random500":
        if Path(args.parent_dataset).resolve() != RANDOM300_DATASET.resolve():
            raise ValueError("production random-500 collection is pinned to the immutable random-300 parent")
        collect_random500(args.dataset_dir, args.exp_dir, parent_dataset=RANDOM300_DATASET)
    elif args.command == "train":
        train(args.random_h5, args.random_manifest, args.fixed_h5, args.fixed_manifest,
              args.exp_dir, args.ckpt_dir, device=args.device)
    elif args.command == "train-expanded":
        train(args.random_h5, args.random_manifest, args.fixed_h5, args.fixed_manifest,
              args.exp_dir, args.ckpt_dir, device=args.device, _random_episodes=200)
    elif args.command == "train-expanded300":
        train(args.random_h5, args.random_manifest, args.fixed_h5, args.fixed_manifest,
              args.exp_dir, args.ckpt_dir, device=args.device, _random_episodes=300)
    elif args.command == "train-expanded500":
        train(args.random_h5, args.random_manifest, args.fixed_h5, args.fixed_manifest,
              args.exp_dir, args.ckpt_dir, device=args.device, _random_episodes=500)
    elif args.command == "evaluate-heldout":
        evaluate_heldout(HELDOUT_CHECKPOINT, HELDOUT_CATALOG_ROOT / "random_final_test",
                         HELDOUT_CATALOG_ROOT / "fixed_base_test", args.output_root)
    else:
        evaluate_validation_candidates(args.ckpt_dir, args.random_catalog, args.fixed_catalog,
                                       args.output_root)


if __name__ == "__main__":
    main()
