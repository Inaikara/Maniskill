"""R41: one matched-R40-Base PickCube ControlNet retrain with calibrated hint-9."""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import pickcube_constrain as pc
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain import pickcube_xy_gauge_round40 as r40
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_body_hint import (
    BodyHintNorm, HINT_DIM, PROFILE, T_TCP_HAND, T_TCP_LINK7,
    PANDA_MODEL_SOURCE_SHA256, PANDA_URDF_SHA256,
    body_hint_from_raw, body_hint_metadata,
)

EXP = Path("outputs/exp/20261029_031500_pickcube_bodyhint_r41")
CKPT = Path("outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_constrain")
R40_PRIMARY_EXP = Path("outputs/exp/20261029_004500_pickcube_xy_gauge_r40")
BASE = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt")
BASE_DIR = BASE.parent
UNUSED_BASE_DIR = Path("outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_base_not_retrained")
URDF = Path("mani_skill/assets/robots/panda/panda_v2.urdf")
ROBOT_SOURCE = Path("mani_skill/agents/robots/panda/panda.py")
CONTROL_FORMAT = "pickcube_body_hint9_constrain_v1"
CONTROL_CONFIG = dict(engine.CONSTRAIN_CONFIG, cond_dim=HINT_DIM)
SOURCE_FILES = tuple(r40.SOURCE_FILES) + (
    "examples/baselines/constrain/pickcube_body_hint.py",
    "examples/baselines/constrain/pickcube_body_hint_round41.py",
    str(URDF), str(ROBOT_SOURCE),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_identity():
    files = {name: sha256_file(name) for name in SOURCE_FILES}
    return {"files": files, "files_sha256": _digest(files)}


def _install_profile():
    r40._install_profile()
    r40.EXP, r40.CONTROL_CKPT = EXP, CKPT
    r40.SOURCE_FILES = SOURCE_FILES
    r40.source_identity = source_identity
    engine.SOURCE_FILES = SOURCE_FILES
    engine.source_identity = source_identity
    engine.ROUND37_EXP = EXP
    engine.ROUND37_BASE_CKPT = r40.BASE_CKPT
    engine.ROUND37_CONTROL_CKPT = CKPT
    engine.CONSTRAIN_CONFIG = CONTROL_CONFIG
    engine.CONSTRAIN_FORMAT = CONTROL_FORMAT
    pc.MODEL_CONFIG = dict(pc.MODEL_CONFIG, cond_dim=HINT_DIM)
    pc.CONSTRAIN_FORMAT = CONTROL_FORMAT
    pc.PickCubeConditionNorm = BodyHintNorm
    pc.pickcube_relative_condition = body_hint_from_raw


def calibrate_existing_states():
    """Read-only native FK validation against 36 existing T+1 HDF5 observations."""
    import h5py
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    from scipy.spatial.transform import Rotation

    data = engine.DATA_H5
    if sha256_file(URDF) != PANDA_URDF_SHA256 or sha256_file(ROBOT_SOURCE) != PANDA_MODEL_SOURCE_SHA256:
        raise ValueError("Panda FK asset/source hash differs from the hint-9 calibrated profile")
    ids = [0, 123, 499, 999, 1000, 1023, 1100, 1250, 1499]
    env = make_pickcube_env(obstacle_radius=0.03, obstacle_height=0.18)
    maximum_tcp_error = 0.0
    relation_spread = {"hand": 0.0, "link7": 0.0}
    relations = {"hand": [], "link7": []}
    count = 0
    try:
        base = env.unwrapped
        env.reset(seed=0, options={"obstacle_enabled": True, "obstacle_center": [0.1, 0.2],
                                  "cube_xy": [-0.1, 0.2], "goal_pos": [0.1, 0.2, 0.12]})
        links = base.agent.robot.links_map
        with h5py.File(data, "r") as source:
            for tid in ids:
                obs = np.asarray(source[f"traj_{tid}/obs"])
                for frame in np.linspace(0, len(obs)-1, 4, dtype=int):
                    raw = obs[frame]
                    qpos = torch.as_tensor(raw[:9], dtype=torch.float32, device=base.device)[None]
                    base.agent.robot.set_qpos(qpos)
                    base.agent.robot.set_qvel(torch.zeros_like(qpos))
                    base.scene.update_render()
                    def matrix(link):
                        return link.pose.to_transformation_matrix()[0].detach().cpu().numpy()
                    tcp, hand, link7 = matrix(base.agent.tcp), matrix(links["panda_hand"]), matrix(links["panda_link7"])
                    for key, value in (("hand", np.linalg.inv(tcp) @ hand),
                                       ("link7", np.linalg.inv(tcp) @ link7)):
                        relations[key].append(value)
                    saved = np.eye(4)
                    saved[:3, 3] = raw[19:22]
                    saved[:3, :3] = Rotation.from_quat(raw[22:26][[1, 2, 3, 0]]).as_matrix()
                    maximum_tcp_error = max(maximum_tcp_error, float(np.max(np.abs(tcp-saved))))
                    count += 1
    finally:
        env.close()
    for key, values in relations.items():
        relation_spread[key] = float(np.max(np.abs(np.asarray(values) - np.asarray(values[0]))))
    if count != 36 or maximum_tcp_error > 1e-5 or max(relation_spread.values()) > 1e-5:
        raise RuntimeError(f"fixed Panda frame calibration failed: {count=}, {maximum_tcp_error=}, {relation_spread=}")
    measured = {"hand": relations["hand"][0], "link7": relations["link7"][0]}
    expected = {"hand": np.asarray(T_TCP_HAND), "link7": np.asarray(T_TCP_LINK7)}
    constant_error = {key: float(np.max(np.abs(measured[key] - expected[key]))) for key in measured}
    if max(constant_error.values()) > 1e-5:
        raise RuntimeError(f"fixed constants differ from native FK: {constant_error}")
    return {"method": "native read-only FK at existing saved qpos; no env.step or state labels",
            "sampled_episode_ids": ids, "frames_per_episode": 4, "state_count": count,
            "tcp_vs_saved_raw45_max_abs": maximum_tcp_error,
            "T_tcp_hand": np.asarray(T_TCP_HAND).tolist(),
            "T_tcp_link7": np.asarray(T_TCP_LINK7).tolist(),
            "fixed_relation_max_spread_m_or_rotation": relation_spread,
            "constant_vs_native_max_abs": constant_error,
            "urdf_path": str(URDF), "urdf_sha256": sha256_file(URDF),
            "robot_model_source": str(ROBOT_SOURCE), "robot_model_source_sha256": sha256_file(ROBOT_SOURCE),
            "tcp_link": "panda_hand_tcp", "derived_links": ["panda_hand", "panda_link7"],
            "raw_quaternion_order": "wxyz", "tolerance": 1e-5}


def register():
    _install_profile()
    if EXP.exists() or CKPT.exists():
        raise FileExistsError("R41 output paths must be new and are never overwritten")
    if sha256_file(BASE) != "9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719":
        raise ValueError("R41 requires the exact R40 gauge-trained Base final EMA")
    calibration = calibrate_existing_states()
    retained_base_dir = r40.BASE_CKPT
    r40.BASE_CKPT = UNUSED_BASE_DIR
    try:
        reg = r40.register()
    finally:
        r40.BASE_CKPT = retained_base_dir
    reg["round"] = 41
    reg["status"] = "registered_after_fixed_frame_calibration_before_control_training"
    reg["paths"]["constrain_checkpoint_dir"] = str(CKPT)
    reg["paths"]["constrain_checkpoint"] = str(CKPT / "final.pt")
    reg["paths"]["constrain_run"] = str(EXP / "train_constrain")
    reg["paths"]["base_checkpoint_dir"] = str(BASE_DIR)
    reg["paths"]["base_checkpoint"] = str(BASE)
    reg["model"]["constrain"] = CONTROL_CONFIG
    reg["model"]["condition_dim"] = 84
    reg["observation_encoding"]["relative_obstacle_hint"] = "latest raw 45-D frame; fixed body hint 9"
    reg["body_hint"] = {"metadata": body_hint_metadata(), "calibration": calibration,
                        "calibration_sha256": _digest(calibration),
                        "runtime_input": "raw 45-D observation and fixed matrices only",
                        "no_new_observations_or_privileged_runtime_state": True,
                        "height_semantics": "world Z feature, not cylinder-surface clearance"}
    reg["training"]["base_updates"] = 0
    reg["training"]["base_policy"] = "frozen R40 final EMA; direct Base not retrained"
    reg["training"]["constrain_updates"] = 150000
    reg["training"]["budgets_seconds"] = {"control": 9000}
    reg["native_evaluation"]["episodes"] = 30
    reg["native_evaluation"]["wall_timeout_seconds_per_episode"] = 300
    reg["native_evaluation"]["outer_timeout_seconds"] = 36000
    reg["native_evaluation"]["primary_reference"] = {
        "r40_base": {"task": 27, "safe": 0, "contact": 30, "xy": 21},
        "r40_control": {"task": 29, "safe": 15, "contact": 15, "xy": 5}}
    reg["native_evaluation"]["gate"] = {"task_gte": 27, "safe_gte": 26,
                                          "contact_lte": 3, "xy_lte": 3}
    reg["hypothesis"] = "one representation-only intervention: add fixed Panda hand/link7 world points to the existing relative obstacle hint"
    reg["formal_performance_claim"] = False
    reg.pop("registration_sha256", None)
    reg["source_identity"] = source_identity()
    reg["registration_sha256"] = _digest(reg)
    (EXP / "preregistration.json").write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    (EXP / "frame_calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    return reg


def verify_registration():
    _install_profile()
    reg = json.loads((EXP / "preregistration.json").read_text())
    body = dict(reg); expected = body.pop("registration_sha256", None)
    if expected != _digest(body) or reg.get("source_identity") != source_identity():
        raise ValueError("R41 registration or source identity mismatch")
    if reg["body_hint"]["metadata"] != body_hint_metadata():
        raise ValueError("R41 hint frames/profile mismatch")
    if sha256_file(BASE) != "9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719":
        raise ValueError("R41 frozen matched Base changed")
    return reg


def record_pretraining_source_correction():
    """Preserve preregistered bytes and pin a pre-training implementation fix."""
    reg_path = EXP / "preregistration.json"
    reg = json.loads(reg_path.read_text())
    if (Path(reg["paths"]["constrain_checkpoint_dir"]).exists()
            or Path(reg["paths"]["constrain_run"]).exists()):
        raise RuntimeError("R41 source correction is only allowed before Control training")
    old = reg["source_identity"]
    current = source_identity()
    changed = {name: {"registered_sha256": old["files"].get(name),
                      "current_sha256": current["files"].get(name)}
               for name in set(old["files"]) | set(current["files"])
               if old["files"].get(name) != current["files"].get(name)}
    expected_changes = {"examples/baselines/constrain/pickcube_body_hint.py",
                        "examples/baselines/constrain/pickcube_body_hint_round41.py"}
    if not changed or not set(changed).issubset(expected_changes):
        raise ValueError(f"unexpected R41 pre-training source delta: {sorted(changed)}")
    prior_events = sorted(EXP.glob("pretraining_source_correction*.json"))
    attempt = len(prior_events) + 1
    original_path = (EXP / "preregistration_initial.json" if attempt == 1 else
                     EXP / f"preregistration_pre_correction_{attempt:02d}.json")
    if not original_path.exists():
        original_path.write_bytes(reg_path.read_bytes())
    previous = reg["registration_sha256"]
    reg.setdefault("source_identity_at_initial_registration", old)
    correction = {
        "reason": "R41 implementation, preflight evidence, and native-report provenance code was finalized after the initial preregistration; the registered per-file hashes identify each source delta.",
        "changed_files": changed, "control_training_started": False,
        "native_evaluation_started": False, "weights_preserved": True,
    }
    history = reg.get("pretraining_source_correction_history")
    if history is None:
        history = [reg["pretraining_source_correction"]] if reg.get("pretraining_source_correction") else []
    history.append(correction)
    reg["pretraining_source_correction_history"] = history
    reg["pretraining_source_correction"] = correction
    reg["source_identity"] = current
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = _digest(reg)
    event = {"event": "pretraining_feature_implementation_correction",
             "previous_registration_sha256": previous,
             "next_registration_sha256": reg["registration_sha256"],
             "initial_registration_snapshot": str(original_path),
             "initial_registration_snapshot_sha256": sha256_file(original_path),
             "source_identity_before": old, "source_identity_after": current,
             "changed_files": changed, "training_started": False}
    event_path = EXP / ("pretraining_source_correction.json" if attempt == 1 else
                        f"pretraining_source_correction_{attempt:02d}.json")
    if event_path.exists():
        raise FileExistsError(event_path)
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    return event


def record_post_training_source_correction():
    """Pin an evaluation-loader fix without changing or retraining completed weights."""
    reg_path = EXP / "preregistration.json"
    reg = json.loads(reg_path.read_text())
    checkpoint = CKPT / "final.pt"
    summary = json.loads((EXP / "train_constrain" / "train_summary.json").read_text())
    if not checkpoint.is_file() or summary.get("status") != "complete":
        raise RuntimeError("post-training correction requires the completed final EMA and training summary")
    paired = EXP / "paired"
    if paired.exists() and any(paired.rglob("episode_*.json")):
        raise RuntimeError("post-training source correction is forbidden after any native episode has run")
    old = reg["source_identity"]
    current = source_identity()
    changed = {name: {"training_sha256": old["files"].get(name),
                      "current_sha256": current["files"].get(name)}
               for name in set(old["files"]) | set(current["files"])
               if old["files"].get(name) != current["files"].get(name)}
    if set(changed) != {"examples/baselines/constrain/pickcube_body_hint_round41.py"}:
        raise ValueError(f"unexpected R41 post-training source delta: {sorted(changed)}")
    frozen = EXP / "preregistration_control_training_frozen.json"
    current_registration = reg_path.read_bytes()
    if not frozen.exists():
        frozen.write_bytes(current_registration)
    attempts = len(list(EXP.glob("post_training_source_correction*.json"))) + 1
    phase_snapshot = (frozen if attempts == 1 else
                      EXP / f"preregistration_pre_native_correction_{attempts:02d}.json")
    if phase_snapshot != frozen and not phase_snapshot.exists():
        phase_snapshot.write_bytes(current_registration)
    previous = reg["registration_sha256"]
    correction = {"phase": "after 150000-update Control completion, before native trial",
        "reason": "Ensure automatic strict loading selects the registered global-XY gauge profile for R40 Base and disabled-Control preflight; no model, sampler, feature, or environment behavior changed.",
        "changed_files": changed, "checkpoint_sha256": sha256_file(checkpoint),
        "training_completed": True, "weights_retrained": False,
        "native_evaluation_attempted": paired.exists(), "native_rollout_started": False}
    reg.setdefault("source_identity_at_control_training", old)
    history = reg.get("post_training_source_correction_history")
    if history is None:
        history = [reg["post_training_source_correction"]] if reg.get("post_training_source_correction") else []
    history.append(correction)
    reg["post_training_source_correction_history"] = history
    reg["post_training_source_correction"] = correction
    reg["source_identity"] = current
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = _digest(reg)
    event = {"event": "post_training_evaluation_loader_correction_before_native_trial",
        "previous_registration_sha256": previous, "next_registration_sha256": reg["registration_sha256"],
        "frozen_control_training_registration": str(frozen),
        "frozen_control_training_registration_sha256": sha256_file(frozen),
        "checkpoint_sha256": sha256_file(checkpoint), "source_identity_before": old,
        "source_identity_after": current, "changed_files": changed,
        "weights_retrained": False, "native_evaluation_attempted": paired.exists(),
        "native_rollout_started": False}
    event_path = (EXP / "post_training_source_correction.json" if attempts == 1 else
                  EXP / f"post_training_source_correction_{attempts:02d}.json")
    if event_path.exists(): raise FileExistsError(event_path)
    reg_path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    return event


def load_body_control(checkpoint, stats, device="cpu", expected_base_sha=None,
                      expected_augmentation=None):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    data = cfg.get("data_identity", {})
    base = cfg.get("base_weights", {})
    normalization = cfg.get("normalization", {})
    if (ck.get("format") != CONTROL_FORMAT or cfg.get("model") != CONTROL_CONFIG
            or normalization.get("hint") != body_hint_metadata()
            or normalization.get("observation_profile") != engine.PROFILE
            or normalization.get("observation_stats") != stats
            or normalization.get("observation_stats_sha256") != stats.get("sha256")
            or normalization.get("action") != "identity native seven-channel"
            or cfg.get("policy") != engine.POLICY
            or cfg.get("diffusion") != engine.DIFFUSION
            or data.get("h5_sha256") != engine.DATA_H5_SHA
            or data.get("manifest_sha256") != engine.DATA_MANIFEST_SHA
            or data.get("selected_class") != "obstacle"
            or data.get("selected_episode_ids") != list(range(1000, 1500))
            or data.get("episodes") != 500 or data.get("windows") != 45840
            or base.get("sha256") != sha256_file(BASE)
            or (expected_base_sha is not None and base.get("sha256") != expected_base_sha)
            or base.get("subkey") != "weights.ema" or base.get("updates") != 30000
            or base.get("path") != str(BASE)
            or cfg.get("training") != dict(engine.TRAINING, updates=engine.CONSTRAIN_UPDATES)
            or cfg.get("training_augmentation") != expected_augmentation
            or cfg.get("selected_weights") != "ema"
            or cfg.get("model_action_space") != "native identity coordinates"
            or cfg.get("action_encoding") is not None
            or cfg.get("action_encoding_identity") is not True
            or ck.get("ema") is None or ck.get("optimizer") is None
            or ck.get("lr_scheduler") is None or ck.get("rng") is None
            or ck.get("step") != engine.CONSTRAIN_UPDATES):
        raise ValueError("R41 typed body-hint checkpoint contract mismatch")
    base_ck = torch.load(BASE, map_location="cpu", weights_only=False)
    independent = engine.build_agent("cpu")
    independent.load_state_dict(base_ck["weights"]["ema"], strict=True)
    if engine.state_sha(independent.noise_pred_net.state_dict()) != base.get("state_sha256"):
        raise ValueError("R41 embedded frozen Base state differs from the exact R40 Base EMA")
    denoiser = engine.build_constrain_denoiser(CONTROL_CONFIG, device=device)
    denoiser.load_state_dict(ck["model"], strict=True)
    from diffusers.training_utils import EMAModel
    ema = EMAModel(parameters=denoiser.trainable_parameters(), power=0.75)
    ema.load_state_dict(ck["ema"]); ema.copy_to(denoiser.trainable_parameters())
    denoiser.eval()
    return denoiser, ck


def _install_evaluation_loader():
    parent = engine.ZScoreConstrainAdapter

    class BodyHintAdapter(parent):
        def __init__(self, checkpoint, stats_path, base_checkpoint, device="cpu", control_enabled=True,
                     expected_augmentation=None):
            if expected_augmentation is None:
                expected_augmentation = r40.METADATA
            self.stats = engine.load_stats(stats_path)
            self.denoiser, self.ck = load_body_control(checkpoint, self.stats, device,
                expected_base_sha=sha256_file(base_checkpoint),
                expected_augmentation=expected_augmentation)
            self.device = torch.device(device); self.control_enabled = control_enabled
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            self.scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                clip_sample=True, prediction_type="epsilon")
            self.scheduler.set_timesteps(100)
            self.metadata = {"format": CONTROL_FORMAT, "profile": PROFILE,
                "normalizer_sha256": self.stats["sha256"],
                "checkpoint_sha256": sha256_file(checkpoint), "weights": "ema",
                "control_enabled": control_enabled, "execution_horizon": 8,
                "hint_profile": PROFILE, "hint_dim": HINT_DIM}

        def get_action(self, raw_queued):
            raw = torch.as_tensor(raw_queued, dtype=torch.float32, device=self.device)
            if raw.ndim == 2: raw = raw[None]
            if raw.shape[-2:] != (2, 45):
                raise ValueError("R41 requires raw native (batch,2,45) observations")
            condition = engine.normalize_observation(raw[..., :42], self.stats).flatten(1)
            hint = body_hint_from_raw(raw)[:, None, :].expand(-1, 16, -1)
            sample = torch.randn((raw.shape[0], 16, 7), device=self.device)
            with torch.no_grad():
                for timestep in self.scheduler.timesteps:
                    epsilon = self.denoiser(sample, timestep, global_cond=condition,
                        hint=hint, enabled=self.control_enabled)
                    sample = self.scheduler.step(epsilon, timestep, sample).prev_sample
            return sample[:, 1:9]

    engine.ZScoreConstrainAdapter = BodyHintAdapter

    from examples.baselines.constrain import pickcube_learning_pilot as pilot
    if not hasattr(pilot, "_r41_original_run_pilot_episode"):
        pilot._r41_original_run_pilot_episode = pilot.run_pilot_episode

    def run_episode_with_protocol(adapter, scene_set, index, device, max_steps=500):
        row = pilot._r41_original_run_pilot_episode(adapter, scene_set, index, device, max_steps)
        protocol = {"policy_condition_dim": 84, "action_dim": 7,
            "sampler": dict(engine.DIFFUSION),
            "native_action_space": "identity pd_ee_delta_pose; no codec/decode/action clamp",
            "checkpoint_sha256": adapter.metadata["checkpoint_sha256"],
            "source_identity_sha256": source_identity()["files_sha256"]}
        if adapter.metadata.get("hint_dim") == HINT_DIM:
            initial = scene_set.load(index)["observation"]
            hint = body_hint_from_raw(torch.as_tensor(np.asarray(initial, dtype=np.float32)))
            protocol.update({"cond_dim": HINT_DIM, "hint_profile": PROFILE,
                "hint_feature_order": body_hint_metadata()["feature_order"],
                "hint_profile_sha256": _digest(body_hint_metadata()),
                "hint_raw_initial_value": hint.tolist(),
                "hint_raw_initial_sha256": hashlib.sha256(hint.numpy().tobytes()).hexdigest(),
                "hint_source": "latest raw 45-D observation and fixed matrices only"})
        else:
            protocol.update({"cond_dim": 0, "hint_profile": None})
        row["runtime_protocol"] = protocol
        return row

    pilot.run_pilot_episode = run_episode_with_protocol

    if not hasattr(engine, "_r41_original_atomic_json"):
        engine._r41_original_atomic_json = engine._atomic_json

    def atomic_json_with_hint_preregistration(path, value):
        if Path(path).name == "preregistration.json" and value.get("status") == "frozen_before_native_evaluation":
            value = dict(value)
            value["body_hint_profile"] = {"cond_dim": HINT_DIM,
                "metadata": body_hint_metadata(), "metadata_sha256": _digest(body_hint_metadata()),
                "control_model_config": CONTROL_CONFIG,
                "control_model_config_sha256": _digest(CONTROL_CONFIG),
                "runtime_input": "latest raw 45-D observation and fixed matrices only",
                "source_identity": source_identity()}
        return engine._r41_original_atomic_json(path, value)

    engine._atomic_json = atomic_json_with_hint_preregistration


def train_control(device="cuda", resume=None):
    reg = verify_registration()
    base_run = json.loads((r40.EXP / "preregistration.json").read_text())["paths"]["base_checkpoint"]
    if sha256_file(base_run) != sha256_file(BASE):
        raise ValueError("R41 Base path does not resolve to the preregistered R40 EMA")
    snapshot = r40._phase_snapshot(reg, "control_training")
    runtime_path = EXP / "phase_runtime_control_training.json"
    attempt = 1
    while runtime_path.exists():
        attempt += 1
        runtime_path = EXP / f"phase_runtime_control_training_attempt{attempt:02d}.json"
    runtime = {"phase": "control_training", "attempt": attempt,
        "argv": [sys.executable, "-m", "examples.baselines.constrain.pickcube_body_hint_round41",
                 "train-control", "--device", device] + (["--resume", str(resume)] if resume else []),
        "started_at_unix": time.time(), "budget_seconds": 9000,
        "resume_checkpoint": str(resume) if resume else None,
        "registration_sha256": reg["registration_sha256"],
        "phase_preregistration_sha256": snapshot["phase_preregistration_sha256"],
        "source_identity": source_identity(), "base_sha256": sha256_file(BASE),
        "data_h5_sha256": sha256_file(engine.DATA_H5),
        "manifest_sha256": sha256_file(engine.DATA_MANIFEST),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_free_total_bytes": torch.cuda.mem_get_info(0) if torch.cuda.is_available() else None,
        "status": "running"}
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    prior_term_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(143)))
    try:
        result = engine.train_constrain(reg["paths"]["stats"], str(BASE), reg["paths"]["constrain_run"],
            str(CKPT), device=device, resume=resume,
            provenance={"registration_sha256": reg["registration_sha256"],
            "phase_preregistration_sha256": snapshot["phase_preregistration_sha256"]},
            training_augmentation=r40.METADATA)
    except BaseException as exc:
        runtime.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_term_handler)
        runtime.update(elapsed_seconds=time.monotonic()-started,
            finished_at_unix=time.time(),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else None,
            source_identity_after=source_identity(),
            data_h5_sha256_after=sha256_file(engine.DATA_H5),
            manifest_sha256_after=sha256_file(engine.DATA_MANIFEST))
        if runtime.get("status") == "running": runtime["status"] = "completed"
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    result["status"] = "complete" if result.get("step") == 150000 and result.get("all_losses_finite") else "pending_audit"
    result["source_identity_after_training"] = source_identity()
    result["source_identity_matches_registration"] = result["source_identity_after_training"] == reg["source_identity"]
    if not result["source_identity_matches_registration"]:
        result["status"] = "pending_audit"
    final_ck = torch.load(CKPT / "final.pt", map_location="cpu", weights_only=False)
    model = final_ck["model"]
    base_tensors = {name[len("base_net."):]: value for name, value in model.items()
                    if name.startswith("base_net.")}
    expected_base = engine.build_agent("cpu")
    base_payload = torch.load(BASE, map_location="cpu", weights_only=False)
    expected_base.load_state_dict(base_payload["weights"]["ema"], strict=True)
    zero_moved = any(bool(torch.count_nonzero(value))
        for name, value in model.items()
        if name.startswith(("zero_skip_convs.", "zero_mid_conv."))
        and name.endswith("weight"))
    copy_moved = False
    for name, value in model.items():
        if name.startswith("control_down_modules."):
            base_name = name.replace("control_down_modules.", "base_net.down_modules.", 1)
        elif name.startswith("control_mid_modules."):
            base_name = name.replace("control_mid_modules.", "base_net.mid_modules.", 1)
        else:
            continue
        if base_name in model and not torch.equal(value, model[base_name]):
            copy_moved = True
            break
    source_base = engine.state_sha(expected_base.noise_pred_net.state_dict())
    embedded_base = engine.state_sha(base_tensors)
    result["frozen_base_exact_r40_ema"] = source_base == embedded_base == final_ck["config"]["base_weights"]["state_sha256"]
    result["control_zero_residual_weights_moved"] = zero_moved
    result["control_copied_encoder_weights_moved"] = copy_moved
    if not all((result["frozen_base_exact_r40_ema"], zero_moved, copy_moved)):
        result["status"] = "pending_audit"
    (Path(reg["paths"]["constrain_run"]) / "train_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    runtime.update(status=result["status"], training_result=result,
                   source_identity_matches_registration=result["source_identity_matches_registration"])
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    return result


def preflight():
    _install_profile()
    from examples.baselines.constrain import pickcube_xy_gauge as gauge
    dataset = engine.native_dataset("obstacle", include_obstacle=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
    first = next(iter(loader))
    raw = first["observations"]
    hint = body_hint_from_raw(raw)
    if raw.shape != (1024, 2, 45) or hint.shape != (1024, 9) or not torch.isfinite(hint).all():
        raise RuntimeError("R41 real frozen-window hint preflight failed")
    shifted, _ = gauge.augment_window_observations(raw, gauge.make_generator(424243))
    shifted_hint = body_hint_from_raw(shifted)
    gauge_error = float((hint-shifted_hint).abs().max())
    if gauge_error > 1e-6:
        raise RuntimeError(f"R41 body hint XY-origin invariance failed: {gauge_error}")
    chunks = [hint]
    for batch_index, batch in enumerate(loader):
        if batch_index:
            chunks.append(body_hint_from_raw(batch["observations"]))
    values = torch.cat(chunks).numpy()

    # Real-data zero-control equivalence against an independently loaded R40 Base.
    base_ck = torch.load(BASE, map_location="cpu", weights_only=False)
    base_agent = engine.build_agent("cpu")
    base_agent.load_state_dict(base_ck["weights"]["ema"], strict=True); base_agent.eval()
    denoiser = engine.build_constrain_denoiser(CONTROL_CONFIG, device="cpu")
    denoiser.base_net.load_state_dict(base_agent.noise_pred_net.state_dict(), strict=True)
    global_cond = engine.normalize_observation(raw[..., :42], engine.load_stats(r40.EXP / "position_encoding.json")).flatten(1)
    repeated_hint = hint[:, None, :].expand(-1, 16, -1)
    torch.manual_seed(424243)
    sample = torch.randn(2, 16, 7); timestep = torch.tensor([13, 71])
    with torch.no_grad():
        base_eps = base_agent.noise_pred_net(sample, timestep, global_cond=global_cond[:2])
        disabled_eps = denoiser(sample, timestep, global_cond=global_cond[:2], hint=repeated_hint[:2], enabled=False)
        zero_control_eps = denoiser(sample, timestep, global_cond=global_cond[:2], hint=repeated_hint[:2], enabled=True)
    eps_error = float((base_eps-disabled_eps).abs().max())
    zero_error = float((base_eps-zero_control_eps).abs().max())
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    sched_a = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    sched_b = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
    sched_a.set_timesteps(100); sched_b.set_timesteps(100)
    state_a = torch.randn(1,16,7,generator=torch.Generator().manual_seed(41)); state_b = state_a.clone()
    gen_a = torch.Generator().manual_seed(42); gen_b = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for t in sched_a.timesteps:
            a = base_agent.noise_pred_net(state_a, t, global_cond=global_cond[:1])
            b = denoiser(state_b, t, global_cond=global_cond[:1], hint=repeated_hint[:1], enabled=False)
            state_a = sched_a.step(a, t, state_a, generator=gen_a).prev_sample
            state_b = sched_b.step(b, t, state_b, generator=gen_b).prev_sample
    sample_error = float((state_a-state_b).abs().max())
    if max(eps_error, zero_error, sample_error) > 1e-6:
        raise RuntimeError(f"R41 zero-control preflight failed: {eps_error=}, {zero_error=}, {sample_error=}")
    report_fixture = r40.validate_report_contract()
    evidence = {"raw_shape": list(raw.shape), "hint_shape": list(hint.shape),
        "frozen_windows_scanned": len(values), "finite": bool(np.isfinite(values).all()),
        "gauge_invariance_max_abs": gauge_error,
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "rms": np.sqrt(np.mean(values**2, axis=0)).tolist(),
        "zero_control_epsilon_max_abs": eps_error,
        "zero_residual_enabled_epsilon_max_abs": zero_error,
        "independent_base_DDPM100_sample_max_abs": sample_error,
        "report_fixture": report_fixture,
        "data_h5_sha256": sha256_file(engine.DATA_H5),
        "data_manifest_sha256": sha256_file(engine.DATA_MANIFEST),
        "base_checkpoint_sha256": sha256_file(BASE)}
    path = EXP / "preflight"; path.mkdir(exist_ok=True)
    (path / "hint9_frozen_batch.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def evaluate():
    reg = verify_registration()
    if not (CKPT / "final.pt").is_file():
        raise FileNotFoundError("R41 Control final checkpoint required")
    primary_pair = R40_PRIMARY_EXP / "paired"
    primary_tree = r40._file_tree_identity(primary_pair)
    primary_reg = json.loads((R40_PRIMARY_EXP / "preregistration.json").read_text())
    if primary_reg.get("registration_sha256") is None:
        raise ValueError("R40 primary registration is missing")
    primary_control = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_constrain/final.pt")
    if sha256_file(primary_control) != "2215f9f7d8bbec933bec53b2880758206af304a171297c5369ad85dcadfe1374":
        raise ValueError("R41 primary comparator is not the retained R40 Control EMA")
    for policy in ("base_dp", "constrain_dp"):
        if len([name for name in primary_tree["files"] if name.startswith(policy + "/episode_")]) != 30:
            raise ValueError(f"R40 primary episode cohort is incomplete for {policy}")

    original_install = r40._install_evaluation_loaders
    original_base_adapter = engine.ZScoreBaseAdapter
    original_control_adapter = engine.ZScoreConstrainAdapter
    original_install()
    _install_evaluation_loader()

    # Freeze a compact R41 native protocol before any environment step.
    from examples.baselines.constrain.pickcube_xy_gauge_round40 import _digest as digest_r40
    dev = Path(reg["paths"]["dev_scenes"])
    base_control = engine.preflight_action_equivalence(
        str(BASE), reg["paths"]["constrain_checkpoint"], reg["paths"]["stats"], "cuda")
    # Let the single R40 installer run once inside evaluate(); repeated wrapping
    # of its already-gauge-wrapped Base adapter breaks the constructor seam.
    engine.ZScoreBaseAdapter = original_base_adapter
    engine.ZScoreConstrainAdapter = original_control_adapter
    prereg = {
        "status": "frozen_before_native_trial", "registration_sha256": reg["registration_sha256"],
        "source_identity": source_identity(), "base_checkpoint_sha256": sha256_file(BASE),
        "control_checkpoint_sha256": sha256_file(CKPT / "final.pt"),
        "data_h5_sha256": reg["dataset"]["h5_sha256"],
        "manifest_sha256": reg["dataset"]["manifest_sha256"],
        "validation_sha256": reg["dataset"]["validation_sha256"],
        "normalizer_file_sha256": sha256_file(reg["paths"]["stats"]),
        "scene_manifest_sha256": sha256_file(dev / "manifest.json"),
        "scene_h5_sha256": sha256_file(dev / "scenes.h5"),
        "episodes_per_policy": 30, "max_control_steps": 500,
        "episode_wall_timeout_seconds": 300, "overall_wall_budget_seconds": 36000,
        "sampler": dict(engine.DIFFUSION), "inference_augmentation": "off",
        "control_hint": {"metadata": body_hint_metadata(), "metadata_sha256": _digest(body_hint_metadata()),
                         "model_config": CONTROL_CONFIG, "model_config_sha256": _digest(CONTROL_CONFIG)},
        "disabled_control_base_action_preflight": base_control,
        "r40_primary_registration_sha256": primary_reg["registration_sha256"],
        "r40_primary_tree_sha256": primary_tree["files_sha256"],
        "r40_primary_tree_files_sha256": primary_tree["files"],
        "transition_semantics": {
            "positive": "improved_new_only=old_false_new_true; regressed_old_only=old_true_new_false",
            "violation": "cleared_old_only=old_true_new_false; introduced_new_only=old_false_new_true"},
    }
    prereg_attempt = len(list(EXP.glob("native_evaluation_preregistration*.json"))) + 1
    prereg["attempt"] = prereg_attempt
    prereg["evaluation_preregistration_sha256"] = digest_r40(prereg)
    prereg_path = (EXP / "native_evaluation_preregistration.json" if prereg_attempt == 1 else
                   EXP / f"native_evaluation_preregistration_attempt{prereg_attempt:02d}.json")
    if prereg_path.exists():
        raise FileExistsError("R41 native trial already preregistered; do not repeat it")
    prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")

    def install_r41_loaders():
        original_install()
        _install_evaluation_loader()

    r40._install_evaluation_loaders = install_r41_loaders
    original_write_eval_inputs = r40._write_native_eval_preregistration

    def write_native_eval_inputs_for_attempt(registration):
        old_path = EXP / "preregistration_native_eval_inputs.json"
        if old_path.exists():
            if (Path(reg["paths"]["paired"]).exists()
                    and any(Path(reg["paths"]["paired"]).rglob("episode_*.json"))):
                raise RuntimeError("cannot replace native preregistration after an episode row exists")
            archived = EXP / "preregistration_native_eval_inputs_attempt01.json"
            if archived.exists():
                raise FileExistsError(archived)
            old_path.replace(archived)
        return original_write_eval_inputs(registration)

    r40._write_native_eval_preregistration = write_native_eval_inputs_for_attempt
    engine.EPISODE_TIMEOUT_SECONDS = 300
    runtime_attempt = len(list(EXP.glob("phase_runtime_native_evaluation*.json"))) + 1
    runtime_path = (EXP / "phase_runtime_native_evaluation.json" if runtime_attempt == 1 else
                    EXP / f"phase_runtime_native_evaluation_attempt{runtime_attempt:02d}.json")
    runtime = {"status": "running", "argv": [sys.executable, "-m",
        "examples.baselines.constrain.pickcube_body_hint_round41", "evaluate"],
        "attempt": runtime_attempt,
        "started_at_unix": time.time(), "budget_seconds": 36000,
        "episode_wall_timeout_seconds": 300,
        "evaluation_preregistration_sha256": prereg["evaluation_preregistration_sha256"],
        "source_identity": source_identity(), "gpu_name": torch.cuda.get_device_name(0),
        "gpu_free_total_bytes": torch.cuda.mem_get_info(0)}
    runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
    started = time.monotonic()
    try:
        try:
            r40.evaluate()
        except KeyError as exc:
            if exc.args != ("phase_counts",):
                raise
            # R40's known report-key mismatch is repaired from retained rows only.
        paired = Path(reg["paths"]["paired"])
        policies = ("base_dp", "constrain_dp")
        rows = {policy: [json.loads((paired / policy / f"episode_{i:03d}.json").read_text())
                         for i in range(30)] for policy in policies}
        primary = {policy: [json.loads((primary_pair / policy / f"episode_{i:03d}.json").read_text())
                            for i in range(30)] for policy in policies}
        if any(row.get("scene_index") != i or row.get("status") != "evaluated"
               for policy in policies for i, row in enumerate(rows[policy])):
            raise ValueError("R41 native trial is missing or has invalid episode rows")
        if any(row.get("status") != "evaluated" for policy in policies for row in primary[policy]):
            raise ValueError("R40 primary rows are incomplete")
        analysis = r40._analyze_native_rows(rows, primary)
        base_trace_identical = all(
            current.get("step_trace") == old.get("step_trace")
            and current.get("result") == old.get("result")
            for current, old in zip(rows["base_dp"], primary["base_dp"]))
        primary_metrics = {policy: engine.summarize_pilot(primary[policy], 30) for policy in policies}
        analysis["primary_comparator"] = {
            "name": "R40 matched Base and Control on the identical 30 restored development scenes",
            "registration_sha256": prereg["r40_primary_registration_sha256"],
            "paired_tree_sha256": primary_tree["files_sha256"],
            "paired_file_sha256": primary_tree["files"],
            "metrics": {policy: engine.summarize_pilot(rows[policy], 30) for policy in policies},
            "r40_metrics": primary_metrics,
            "transitions": {policy: {field: engine._transitions(primary[policy], rows[policy], field)
                                      for field in ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")}
                            for policy in policies},
            "base_native_action_trace_and_outcomes_exact": base_trace_identical,
            "interpretation": "Base is the exact retained R40 checkpoint; Control weights are new and only the hint representation changed",
        }
        analysis["vs_r40"] = analysis["primary_comparator"]["transitions"]
        analysis.pop("vs_r38", None)
        analysis["base_native_action_trace_and_outcomes_exact"] = base_trace_identical
        analysis["native_evaluation_preregistration_sha256"] = prereg["evaluation_preregistration_sha256"]
        analysis["base_disabled_control_preflight"] = base_control
        analysis["native_evaluation_rerun"] = False
        row_hashes = {f"{policy}/episode_{i:03d}.json": sha256_file(
            paired / policy / f"episode_{i:03d}.json") for policy in policies for i in range(30)}
        snapshot = {"evaluation_preregistration_sha256": prereg["evaluation_preregistration_sha256"],
            "source_identity": source_identity(), "r40_primary_tree_sha256": primary_tree["files_sha256"],
            "r40_primary_files_sha256": primary_tree["files"],
            "current_row_sha256": row_hashes, "native_evaluation_rerun": False}
        snapshot["analysis_inputs_sha256"] = _digest(snapshot)
        input_path = EXP / "analysis_report_inputs.json"
        if input_path.exists(): raise FileExistsError(input_path)
        input_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        analysis["analysis_input_snapshot_sha256"] = snapshot["analysis_inputs_sha256"]
        analysis["source_identity_at_native"] = source_identity()
        analysis["analysis_source_identity"] = source_identity()
        analysis["analysis_sha256"] = _digest({k: v for k, v in analysis.items() if k != "analysis_sha256"})
        analysis_path = paired / "analysis.json"
        analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        status = {"status": "complete", "retained_rows": {policy: 30 for policy in policies},
            "runtime_errors": {policy: 0 for policy in policies}, "native_evaluation_rerun": False,
            "gate_passed": analysis["gate"]["passed"],
            "base_native_action_trace_and_outcomes_exact": base_trace_identical,
            "analysis_input_snapshot_sha256": snapshot["analysis_inputs_sha256"]}
        (paired / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        runtime.update(status="complete", retained_rows={policy: 30 for policy in policies},
                       analysis_sha256=sha256_file(analysis_path),
                       base_native_action_trace_and_outcomes_exact=base_trace_identical)
        return analysis
    except BaseException as exc:
        runtime.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runtime.update(elapsed_seconds=time.monotonic()-started, finished_at_unix=time.time(),
                       source_identity_after=source_identity())
        runtime_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("calibrate", "register", "record-pretraining-source-correction",
        "record-post-training-source-correction",
        "verify", "preflight", "train-control", "evaluate"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    _install_profile()
    result = {"calibrate": calibrate_existing_states, "register": register,
              "record-pretraining-source-correction": record_pretraining_source_correction,
              "record-post-training-source-correction": record_post_training_source_correction,
              "verify": verify_registration, "preflight": preflight,
              "train-control": lambda: train_control(args.device, args.resume), "evaluate": evaluate}[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
