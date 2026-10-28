"""Matched PickCube R40 training with one global-XY gauge augmentation."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain import pickcube_position_units_round38 as r38
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_xy_gauge import METADATA

EXP = Path("outputs/exp/20261029_004500_pickcube_xy_gauge_r40")
BASE_CKPT = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base")
CONTROL_CKPT = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_constrain")
R38_EXP = Path("outputs/exp/20261028_221700_pickcube_position_units_r38")
R38_PAIRED = R38_EXP / "paired"
SOURCE_FILES = (
    "examples/baselines/constrain/pickcube_xy_gauge.py",
    "examples/baselines/constrain/pickcube_xy_gauge_round40.py",
    "examples/baselines/constrain/pickcube_position_units.py",
    "examples/baselines/constrain/pickcube_position_units_round38.py",
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/pickcube_observation_norm.py",
    "examples/baselines/constrain/materialize_pickcube_fullscene500.py",
    "examples/baselines/constrain/pickcube_base_dp.py",
    "examples/baselines/constrain/pickcube_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/constrain_ckpt.py",
    "examples/baselines/constrain/pickcube_learning_pilot.py",
    "examples/baselines/constrain/pickcube_transport_detour.py",
    "mani_skill/envs/tasks/tabletop/pick_cube.py",
)


def source_identity():
    files = {name: sha256_file(name) for name in SOURCE_FILES}
    return {"files": files, "files_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def _install_profile():
    r38._install_profile()
    engine.ROUND37_EXP = EXP
    engine.ROUND37_BASE_CKPT = BASE_CKPT
    engine.ROUND37_CONTROL_CKPT = CONTROL_CKPT
    engine.R36_DEV = Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes")
    engine.SOURCE_FILES = SOURCE_FILES
    engine.source_identity = source_identity


def _paths():
    r36 = json.loads(engine.R36_REGISTRATION.read_text())
    scenes = Path(r36["paths"]["dev_scenes"])
    return {
        "experiment": str(EXP), "registration": str(EXP / "preregistration.json"),
        "stats": str(EXP / "position_encoding.json"),
        "base_run": str(EXP / "train_base"), "base_checkpoint_dir": str(BASE_CKPT),
        "base_checkpoint": str(BASE_CKPT / "final.pt"),
        "constrain_run": str(EXP / "train_constrain"),
        "constrain_checkpoint_dir": str(CONTROL_CKPT),
        "constrain_checkpoint": str(CONTROL_CKPT / "final.pt"),
        "preflight": str(EXP / "preflight"), "paired": str(EXP / "paired"),
        "dev_scenes": str(scenes), "r36_paired": str(engine.R36_PAIRED),
    }, r36, scenes


def _digest(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def register():
    _install_profile()
    paths, r36, scenes = _paths()
    if BASE_CKPT.exists() or CONTROL_CKPT.exists():
        raise FileExistsError("R40 output path already exists; registration never overwrites artifacts")
    for path, expected in (
        (engine.DATA_H5, engine.DATA_H5_SHA),
        (engine.DATA_MANIFEST, engine.DATA_MANIFEST_SHA),
        (Path(engine.DATA_MANIFEST).with_name("validation.json"),
         "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f"),
        (scenes / "manifest.json", r36["validation"]["dev_manifest"]["sha256"]),
        (scenes / "scenes.h5", r36["validation"]["dev_scenes_h5"]["sha256"]),
        (Path("mani_skill/envs/tasks/tabletop/pick_cube.py"),
         r36["validation"]["maniskill_pickcube_source"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"pinned R40 source identity mismatch: {path}")
    dataset_direct = engine.native_dataset("direct")
    dataset_obstacle = engine.native_dataset("obstacle", include_obstacle=True)
    if len(dataset_direct) != 83004 or len(dataset_obstacle) != 45840:
        raise ValueError("frozen window cohort counts differ from the 1000/500 contract")

    profile = r38.profile_metadata()
    stats = dict(profile)
    stats["source_identity"] = {
        "selected_class": "fixed_physical_reference_not_fitted",
        "selected_episode_ids": list(range(1000)),
        "h5_sha256": engine.DATA_H5_SHA,
        "manifest_sha256": engine.DATA_MANIFEST_SHA,
    }
    stats_path = EXP / "position_encoding.json"
    if EXP.exists():
        if ({item.name for item in EXP.iterdir()} != {"position_encoding.json"}
                or json.loads(stats_path.read_text()) != stats):
            raise FileExistsError("R40 output path contains unexpected existing artifacts")
    else:
        EXP.mkdir(parents=True, exist_ok=False)
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    r36_scenes = r36["validation"]
    body = {
        "round": 40, "status": "registered_before_training_and_native_evaluation",
        "profile": profile, "profile_sha256": profile["sha256"],
        "paths": paths,
        "dataset": {
            "h5_path": engine.DATA_H5, "h5_sha256": engine.DATA_H5_SHA,
            "manifest_path": engine.DATA_MANIFEST, "manifest_sha256": engine.DATA_MANIFEST_SHA,
            "validation_sha256": "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f",
            "direct_ids": list(range(1000)), "obstacle_ids": list(range(1000, 1500)),
            "episodes": 1500, "direct_windows": 83004, "obstacle_windows": 45840,
            "actions": "native identity seven-channel; unchanged", "closed_terminal_grip": -1,
        },
        "observation_encoding": {
            **profile, "fit_from_data": False, "controller_translation_reference_m": 0.1,
            "apply_once_to_42d_state": True,
            "relative_obstacle_hint": "original raw native 45-D latest frame; unchanged",
        },
        "normalizer": {
            "file_sha256": sha256_file(EXP / "position_encoding.json"),
            "statistics_sha256": profile["sha256"], "fit_from_data": False,
            "fit_frame_count": 0, "fit_source_ids": [], "floor": None, "floored_fields": [],
        },
        "training_augmentation": {
            **METADATA,
            "distribution_change": "training-only coordinate-origin gauge covariance over same frozen physical episodes",
            "per_window": True, "shared_history_frames": True,
            "delta_statistics": "augmentation_delta_stats in each phase train_summary.json; min/max/mean and window count",
            "world_fields": ["extra/tcp_pose.xy", "extra/goal_pos.xy", "extra/obj_pose.xy",
                             "extra/pickcube_obstacle.center_xy (Control only)"],
            "unchanged": ["qpos", "qvel", "grasp flags", "z", "quaternions",
                          "relative vectors", "obstacle radius", "all native action labels"],
            "dedicated_rng": METADATA["rng_algorithm"],
            "inference": "off; unshifted native world frame",
        },
        "training": {
            "base_updates": 30000, "constrain_updates": 150000, **engine.TRAINING,
            "base_class": "direct", "constrain_class": "obstacle",
            "base_frozen_in_constrain": True, "loss": "unweighted epsilon MSE",
            "recovery_every_updates": 10000,
            "budgets_seconds": {"base": 3000, "control": 9000},
            "matched_end_to_end_training": True,
        },
        "model": {"base": dict(engine.BASE_CONFIG), "constrain": dict(engine.CONSTRAIN_CONFIG),
                  "condition_dim": 84, "action_dim": 7},
        "sampler": dict(engine.DIFFUSION),
        "native_evaluation": {
            "episodes": 30, "max_control_steps": 500,
            "wall_timeout_seconds_per_episode": 300, "outer_timeout_seconds": 36000,
            "scene_manifest_sha256": r36_scenes["dev_manifest"]["sha256"],
            "scene_h5_sha256": r36_scenes["dev_scenes_h5"]["sha256"],
            "noise_seed_rule": "torch.manual_seed(scene seed) independently per policy",
            "augmentation": "off",
            "primary_reference": {"r38_base": {"task": 22, "safe": 0, "contact": 30, "xy": 23},
                                  "r38_control": {"task": 29, "safe": 13, "contact": 16, "xy": 6}},
            "r39_negative_guidance": {"task": 29, "safe": 11, "contact": 17, "xy": 6,
                                      "excluded_from_primary": True},
            "gate": {"task_gte": 27, "safe_gte": 26, "contact_lte": 3, "xy_lte": 3},
        },
        "formal_performance_claim": False,
        "interpretation": "exploratory matched end-to-end retraining; not an isolated Control-causality estimate",
        "hypothesis": "small training-only world-coordinate-origin randomization may reduce absolute global-position memorization across 500 demonstrations and 333 task variants; exploratory, not causally established by R38/R39",
        "source_identity": source_identity(),
        "r38_registration_sha256": sha256_file(r38.EXP / "preregistration.json"),
        "r36_registration_sha256": sha256_file(engine.R36_REGISTRATION),
    }
    body["registration_sha256"] = _digest(body)
    (EXP / "preregistration.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def verify(path=None):
    _install_profile()
    path = Path(path or EXP / "preregistration.json")
    reg = json.loads(path.read_text())
    body = dict(reg)
    expected = body.pop("registration_sha256", None)
    if expected != _digest(body):
        raise ValueError("R40 registration checksum mismatch")
    if reg.get("source_identity") != source_identity():
        raise ValueError("R40 source identity differs from frozen preregistration")
    if reg.get("training_augmentation", {}).get("kind") != METADATA["kind"]:
        raise ValueError("R40 training augmentation metadata missing")
    for name, expected_hash in (
        (engine.DATA_H5, engine.DATA_H5_SHA),
        (engine.DATA_MANIFEST, engine.DATA_MANIFEST_SHA),
        (Path(engine.DATA_MANIFEST).with_name("validation.json"), reg["dataset"]["validation_sha256"]),
        (Path(reg["paths"]["stats"]), reg["normalizer"]["file_sha256"]),
    ):
        if sha256_file(name) != expected_hash:
            raise ValueError(f"R40 pinned artifact changed: {name}")
    r38.validate_profile(json.loads(Path(reg["paths"]["stats"]).read_text()))
    return reg


def refresh_registration():
    """Freeze corrected preflight protocol details before any training starts."""
    path = EXP / "preregistration.json"
    reg = json.loads(path.read_text())
    if (Path(reg["paths"]["base_checkpoint"]).exists()
            or Path(reg["paths"]["constrain_checkpoint"]).exists()):
        raise RuntimeError("R40 preregistration cannot change after training checkpoints exist")
    draft = EXP / "preregistration_preflight_draft.json"
    if not draft.exists():
        draft.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    reg["training_augmentation"].update(METADATA)
    reg["training_augmentation"]["delta_statistics"] = (
        "augmentation_delta_stats in each phase train_summary.json; min/max/mean and window count")
    reg["training_augmentation"]["dedicated_rng"] = METADATA["rng_algorithm"]
    reg["source_identity"] = source_identity()
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = _digest(reg)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    return reg


def record_post_base_source_correction():
    """Pin a required loader fix after Base, without changing its weights."""
    path = EXP / "preregistration.json"
    reg = json.loads(path.read_text())
    ckpt_path = Path(reg["paths"]["base_checkpoint"])
    summary_path = Path(reg["paths"]["base_run"]) / "train_summary.json"
    if not ckpt_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("source correction requires the durable completed Base checkpoint and summary")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or not summary.get("source_identity_matches_registration"):
        raise ValueError("Base checkpoint is not a completed model from the frozen initial preregistration")
    old_identity = reg["source_identity"]
    current = source_identity()
    changed = {name: {"registered_sha256": old_identity["files"].get(name),
                      "current_sha256": current["files"].get(name)}
               for name in set(old_identity["files"]) | set(current["files"])
               if old_identity["files"].get(name) != current["files"].get(name)}
    expected_changed = {
        "examples/baselines/constrain/pickcube_zscore_round37.py",
        "examples/baselines/constrain/pickcube_xy_gauge_round40.py",
    }
    if set(changed) != expected_changed:
        raise ValueError(f"unexpected source delta after Base training: {sorted(changed)}")
    frozen_path = EXP / "preregistration_base_training_frozen.json"
    original_bytes = path.read_bytes()
    if frozen_path.exists():
        if frozen_path.read_bytes() != original_bytes:
            raise ValueError("immutable Base preregistration snapshot already differs")
    else:
        frozen_path.write_bytes(original_bytes)
    original_registration_sha = reg["registration_sha256"]
    reg["source_identity_at_base_training"] = old_identity
    reg["source_correction_after_base"] = {
        "phase": "Control preflight only; before Control training",
        "reason": "The real-data Control preflight exposed a NameError because the new explicit original-raw-hint invariance check referenced pickcube_relative_condition without importing it. Add the missing import in both Control preflight and training paths; no algorithm, tensor, dataset, or checkpoint weight changes.",
        "changed_files": changed,
        "base_checkpoint_sha256": sha256_file(ckpt_path),
        "base_training_source_identity": old_identity,
        "base_weights_preserved_without_retraining": True,
        "preflight_attempt_exit_status": 1,
    }
    reg["source_identity"] = current
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = _digest(reg)
    event = {
        "event": "post_base_source_correction_before_control_training",
        "previous_registration_sha256": original_registration_sha,
        "next_registration_sha256": reg["registration_sha256"],
        "frozen_base_registration_snapshot": str(frozen_path),
        "base_checkpoint_sha256": sha256_file(ckpt_path),
        "source_identity_before": old_identity,
        "source_identity_after": current,
        "changed_files": changed,
        "base_retrained": False,
        "base_weights_unchanged": True,
    }
    event_path = EXP / "post_base_source_correction.json"
    if event_path.exists():
        raise FileExistsError(f"R40 source-correction record already exists: {event_path}")
    path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    return event


def record_post_training_analysis_correction():
    """Freeze the R38-primary report adapter after both trained weights exist."""
    path = EXP / "preregistration.json"
    reg = json.loads(path.read_text())
    base_path, control_path = Path(reg["paths"]["base_checkpoint"]), Path(reg["paths"]["constrain_checkpoint"])
    if not base_path.is_file() or not control_path.is_file():
        raise FileNotFoundError("analysis source correction requires both durable trained checkpoints")
    for summary_path in (Path(reg["paths"]["base_run"]) / "train_summary.json",
                         Path(reg["paths"]["constrain_run"]) / "train_summary.json"):
        if json.loads(summary_path.read_text()).get("status") != "complete":
            raise ValueError("both R40 training phases must be complete before report-only correction")
    old_identity = reg["source_identity"]
    current = source_identity()
    changed = {name: {"registered_sha256": old_identity["files"].get(name),
                      "current_sha256": current["files"].get(name)}
               for name in set(old_identity["files"]) | set(current["files"])
               if old_identity["files"].get(name) != current["files"].get(name)}
    if set(changed) != {"examples/baselines/constrain/pickcube_xy_gauge_round40.py"}:
        raise ValueError(f"unexpected source delta after Control training: {sorted(changed)}")
    frozen_path = EXP / "preregistration_control_training_frozen.json"
    original_bytes = path.read_bytes()
    if frozen_path.exists():
        if frozen_path.read_bytes() != original_bytes:
            raise ValueError("immutable Control preregistration snapshot already differs")
    else:
        frozen_path.write_bytes(original_bytes)
    previous_sha = reg["registration_sha256"]
    reg["source_identity_at_control_training"] = old_identity
    reg["post_training_analysis_correction"] = {
        "phase": "post-training report adapter, before any native evaluation",
        "reason": "The shared historical R37 evaluator labels its prior-trial comparison as R36. R40 requires the preregistered R38 matched-training run as primary, so R40 adds a pinned offline adapter that hashes and compares the retained R38 episode rows. No policy, sampler, environment, training, or model tensor changes.",
        "changed_files": changed,
        "base_checkpoint_sha256": sha256_file(base_path),
        "control_checkpoint_sha256": sha256_file(control_path),
        "trained_weights_preserved_without_retraining": True,
        "native_evaluation_started": False,
    }
    reg["source_identity"] = current
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = _digest(reg)
    event = {
        "event": "post_training_analysis_correction_before_native_evaluation",
        "previous_registration_sha256": previous_sha,
        "next_registration_sha256": reg["registration_sha256"],
        "frozen_control_training_snapshot": str(frozen_path),
        "base_checkpoint_sha256": sha256_file(base_path),
        "control_checkpoint_sha256": sha256_file(control_path),
        "source_identity_before": old_identity,
        "source_identity_after": current,
        "changed_files": changed,
        "base_retrained": False,
        "control_retrained": False,
    }
    event_path = EXP / "post_training_analysis_correction.json"
    if event_path.exists():
        raise FileExistsError(f"R40 post-training correction record already exists: {event_path}")
    path.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    return event


def _phase_snapshot(reg, phase):
    attempt = 1
    path = EXP / f"preregistration_{phase}.json"
    if path.exists():
        attempt = 2
        while (EXP / f"preregistration_{phase}_attempt{attempt:02d}.json").exists():
            attempt += 1
        path = EXP / f"preregistration_{phase}_attempt{attempt:02d}.json"
    body = {"phase": phase, "attempt": attempt,
            "registration_sha256": reg["registration_sha256"],
            "source_identity_before": source_identity(),
            "dataset": reg["dataset"], "training_augmentation": reg["training_augmentation"]}
    body["phase_preregistration_sha256"] = _digest(body)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def _prepare_phase_dirs(run_dir, checkpoint_dir, phase):
    """Preserve and clear only the empty directories left by a pre-training abort."""
    run_dir, checkpoint_dir = Path(run_dir), Path(checkpoint_dir)
    if run_dir.exists():
        entries = list(run_dir.iterdir())
        if entries:
            if len(entries) != 1 or entries[0].name != "phase_runtime.json":
                raise FileExistsError(f"R40 phase output already contains data: {run_dir}")
            preserved = EXP / f"phase_runtime_{phase}_attempt01.json"
            if preserved.exists():
                raise FileExistsError(f"preserved R40 phase record already exists: {preserved}")
            entries[0].replace(preserved)
        run_dir.rmdir()
    if checkpoint_dir.exists():
        if any(checkpoint_dir.iterdir()):
            raise FileExistsError(f"R40 checkpoint output already contains data: {checkpoint_dir}")
        checkpoint_dir.rmdir()


def _train(kind, device):
    reg = verify()
    phase = "base_training" if kind == "base" else "control_training"
    snapshot = _phase_snapshot(reg, phase)
    paths = reg["paths"]
    if kind == "base":
        _prepare_phase_dirs(paths["base_run"], paths["base_checkpoint_dir"], phase)
        result = engine.train_base(paths["stats"], paths["base_run"], paths["base_checkpoint_dir"],
                                   device=device, provenance={"registration": paths["registration"],
                                                              "phase_preregistration_sha256": snapshot["phase_preregistration_sha256"]},
                                   training_augmentation=METADATA)
        summary_path = Path(paths["base_run"]) / "train_summary.json"
    else:
        if not Path(paths["base_checkpoint"]).is_file():
            raise FileNotFoundError("R40 Control training requires the completed R40 gauge-trained Base EMA")
        _prepare_phase_dirs(paths["constrain_run"], paths["constrain_checkpoint_dir"], phase)
        result = engine.train_constrain(paths["stats"], paths["base_checkpoint"], paths["constrain_run"],
                                        paths["constrain_checkpoint_dir"], device=device,
                                        provenance={"registration": paths["registration"],
                                                    "phase_preregistration_sha256": snapshot["phase_preregistration_sha256"]},
                                        training_augmentation=METADATA)
        summary_path = Path(paths["constrain_run"]) / "train_summary.json"
    result["phase_preregistration_sha256"] = snapshot["phase_preregistration_sha256"]
    result["source_identity_after_training"] = source_identity()
    result["source_identity_matches_registration"] = result["source_identity_after_training"] == reg["source_identity"]
    result["dataset_identity_after_training"] = {
        "h5_sha256": sha256_file(engine.DATA_H5), "manifest_sha256": sha256_file(engine.DATA_MANIFEST),
        "validation_sha256": sha256_file(Path(engine.DATA_MANIFEST).with_name("validation.json")),
    }
    result["dataset_identity_matches_registration"] = (
        result["dataset_identity_after_training"]["h5_sha256"] == reg["dataset"]["h5_sha256"]
        and result["dataset_identity_after_training"]["manifest_sha256"] == reg["dataset"]["manifest_sha256"]
        and result["dataset_identity_after_training"]["validation_sha256"] == reg["dataset"]["validation_sha256"])
    result["status"] = "complete" if (result.get("all_losses_finite")
                                       and result["source_identity_matches_registration"]
                                       and result["dataset_identity_matches_registration"]
                                       and result.get("step") == (30000 if kind == "base" else 150000)) else "pending_audit"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _install_evaluation_loaders():
    base_class, control_class = engine.ZScoreBaseAdapter, engine.ZScoreConstrainAdapter

    class GaugeBaseAdapter(base_class):
        def __init__(self, checkpoint, stats_path, device="cpu"):
            super().__init__(checkpoint, stats_path, device, expected_augmentation=METADATA)

    class GaugeControlAdapter(control_class):
        def __init__(self, checkpoint, stats_path, base_checkpoint, device="cpu", control_enabled=True):
            super().__init__(checkpoint, stats_path, base_checkpoint, device, control_enabled,
                             expected_augmentation=METADATA)

    engine.ZScoreBaseAdapter = GaugeBaseAdapter
    engine.ZScoreConstrainAdapter = GaugeControlAdapter


def validate_report_contract():
    """Exercise the actual evaluator aggregation shape before native stepping."""
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    primary, current = {}, {}
    for index in range(30):
        identity = {"scene_index": index, "scene_id": f"pickcube-{index:03d}", "seed": index + 100,
                    "restored_state_sha256": f"state-{index}",
                    "restored_controller_state_sha256": f"controller-{index}",
                    "restored_observation_sha256": f"observation-{index}",
                    "normalizer_condition_sha256": f"condition-{index}",
                    "status": "evaluated", "wall_seconds": float(index + 1),
                    "first_native_success_step": index + 20,
                    "final_cube_xyz": [0.0, 0.0, 0.1], "goal_xyz": [0.0, 0.0, 0.1],
                    "final_goal_distance_3d": 0.0, "first_contact": {"phase": "transport", "link": "hand"}}
        old_result = {name: index % 2 == 0 for name in fields}
        new_result = {name: index % 3 == 0 for name in fields}
        old_result.update({"phase": "task_success", "terminated": True, "truncated": False})
        new_result.update({"phase": "task_success", "terminated": True, "truncated": False})
        primary.setdefault("base_dp", []).append({**identity, "result": old_result})
        current.setdefault("base_dp", []).append({**identity, "result": new_result})
        primary.setdefault("constrain_dp", []).append({**identity, "result": old_result})
        current.setdefault("constrain_dp", []).append({**identity, "result": new_result})
    report = _analyze_native_rows(current, primary)
    required = ("policy_metrics", "terminal_phase_counts", "contact_first_phase_counts",
                "paired_base_vs_control", "vs_r38", "gate", "first_contact_link_counts",
                "final_cube_xyz_and_goal_xyz_present")
    if any(name not in report for name in required) or not all(
            metric["complete"] for metric in report["policy_metrics"].values()):
        raise RuntimeError("paired report contract failed actual-shaped row aggregation")
    return {"actual_shaped_row_count": 30, "summary_complete": True,
            "transition_fields": list(fields), "aggregated_fields": list(required),
            "reconciled": True}


def _file_tree_identity(root):
    root = Path(root)
    files = {str(path.relative_to(root)): sha256_file(path)
             for path in sorted(root.rglob("*")) if path.is_file()}
    return {"files": files, "files_sha256": _digest(files)}


def _load_r38_primary():
    reg = json.loads((R38_EXP / "preregistration.json").read_text())
    body = dict(reg)
    registration_sha = body.pop("registration_sha256", None)
    if registration_sha != _digest(body):
        raise ValueError("R38 primary preregistration checksum mismatch")
    if reg["paths"]["paired"] != str(R38_PAIRED):
        raise ValueError("R38 primary paired path differs from retained R38 registration")
    tree = _file_tree_identity(R38_PAIRED)
    for policy in ("base_dp", "constrain_dp"):
        rows = [json.loads((R38_PAIRED / policy / f"episode_{index:03d}.json").read_text())
                for index in range(30)]
        if any(row.get("scene_index") != index or row.get("status") != "evaluated"
               for index, row in enumerate(rows)):
            raise ValueError(f"R38 primary rows are incomplete or unordered for {policy}")
        reg_checkpoint = Path(reg["paths"]["base_checkpoint" if policy == "base_dp" else "constrain_checkpoint"])
        expected = reg.get("matched_base_checkpoint" if policy == "base_dp" else "matched_constrain_checkpoint", {}).get("sha256")
        if expected and sha256_file(reg_checkpoint) != expected:
            raise ValueError(f"R38 primary checkpoint hash mismatch for {policy}")
    return reg, tree


def _write_native_eval_preregistration(reg):
    r38_reg, r38_tree = _load_r38_primary()
    body = {
        "status": "frozen_before_native_evaluation",
        "r40_registration_sha256": reg["registration_sha256"],
        "r40_source_identity": source_identity(),
        "r40_base_checkpoint_sha256": sha256_file(reg["paths"]["base_checkpoint"]),
        "r40_control_checkpoint_sha256": sha256_file(reg["paths"]["constrain_checkpoint"]),
        "training_augmentation": reg["training_augmentation"],
        "inference_augmentation": "off",
        "r38_registration_sha256": r38_reg["registration_sha256"],
        "r38_primary_paired_tree": r38_tree,
        "primary_comparison": "R40 matched gauge-augmented Base/Control versus R38 matched position-unit Base/Control on the same 30 R36 restored development scenes and per-policy scene seeds",
        "r39_guided_control": {"task": 29, "safe": 11, "contact": 17, "xy": 6,
                               "role": "negative guidance result; excluded from primary"},
        "evaluation_protocol": reg["native_evaluation"],
    }
    body["evaluation_preregistration_sha256"] = _digest(body)
    path = EXP / "preregistration_native_eval_inputs.json"
    if path.exists():
        raise FileExistsError(f"R40 native evaluation preregistration already exists: {path}")
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body, r38_reg, r38_tree


def _enrich_with_r38_primary(analysis, r38_reg, r38_tree):
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot

    rows = {}
    for policy in ("base_dp", "constrain_dp"):
        rows[policy] = {
            "r38": [json.loads((R38_PAIRED / policy / f"episode_{index:03d}.json").read_text())
                    for index in range(30)],
            "r40": [json.loads((EXP / "paired" / policy / f"episode_{index:03d}.json").read_text())
                    for index in range(30)],
        }
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    vs_r38, state_identity_match = {}, {}
    for policy, cohorts in rows.items():
        old, new = cohorts["r38"], cohorts["r40"]
        if any(row.get("status") != "evaluated" or row.get("scene_index") != i
               for group in (old, new) for i, row in enumerate(group)):
            raise ValueError(f"R38-primary transition comparison has incomplete {policy} rows")
        state_identity_match[policy] = all(
            a.get(key) == b.get(key)
            for a, b in zip(old, new)
            for key in ("scene_id", "restored_state_sha256", "restored_controller_state_sha256",
                        "restored_observation_sha256"))
        vs_r38[policy] = {field: engine._transitions(old, new, field) for field in fields}
        if not summarize_pilot(new, 30)["complete"]:
            raise ValueError(f"R40 primary comparison has an incomplete {policy} denominator")
    if not all(state_identity_match.values()):
        raise ValueError("R40/R38 primary rows do not use identical restored development states")
    analysis["primary_comparator"] = {
        "name": "R38 matched position/controller-unit pipeline",
        "registration_sha256": r38_reg["registration_sha256"],
        "paired_tree_sha256": r38_tree["files_sha256"],
        "paired_file_sha256": r38_tree["files"],
        "metrics": {policy: summarize_pilot(rows[policy]["r40"], 30) for policy in rows},
        "r38_metrics": {policy: summarize_pilot(rows[policy]["r38"], 30) for policy in rows},
        "restored_state_identity_matches_scene_by_scene": state_identity_match,
        "transitions_r38_to_r40": vs_r38,
        "interpretation": "matched end-to-end retraining comparison; not isolated Control-factor causality",
    }
    analysis["vs_r38"] = vs_r38
    analysis["r39_guidance_is_not_primary"] = {"task": 29, "safe": 11, "contact": 17, "xy": 6}
    return analysis


def _distribution(values):
    import numpy as np

    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return {"min": min(values), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "max": max(values)}


def _analyze_native_rows(rows_by_policy, primary_rows):
    """Build every R40 report field from retained row JSON only."""
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot

    policies = ("base_dp", "constrain_dp")
    fields = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    summaries = {policy: summarize_pilot(rows_by_policy[policy], 30) for policy in policies}
    if any(not summary["complete"] for summary in summaries.values()):
        raise ValueError("R40 analysis requires exactly 30 evaluated rows per policy")
    pairing = all(
        left.get(key) == right.get(key)
        for left, right in zip(rows_by_policy["base_dp"], rows_by_policy["constrain_dp"])
        for key in ("scene_id", "restored_state_sha256", "restored_controller_state_sha256",
                    "restored_observation_sha256"))
    normalized_pairing = all(
        left.get("normalizer_condition_sha256") == right.get("normalizer_condition_sha256")
        for left, right in zip(rows_by_policy["base_dp"], rows_by_policy["constrain_dp"]))
    if not pairing or not normalized_pairing:
        raise ValueError("R40 Base/Control paired native observation identity mismatch")
    vs_primary = {}
    state_matches = {}
    for policy in policies:
        old, new = primary_rows[policy], rows_by_policy[policy]
        state_matches[policy] = all(
            left.get(key) == right.get(key)
            for left, right in zip(old, new)
            for key in ("scene_id", "seed", "restored_state_sha256",
                        "restored_controller_state_sha256", "restored_observation_sha256",
                        "normalizer_condition_sha256"))
        if not state_matches[policy]:
            raise ValueError(f"R40 {policy} rows do not match the primary R38 restored state/condition hashes")
        vs_primary[policy] = {field: engine._transitions(old, new, field) for field in fields}
    if any(not state_matches[p] for p in policies):
        raise ValueError("R40 to R38 primary state identity mismatch")
    paired_metrics = {field: engine._transitions(rows_by_policy["base_dp"],
                                                  rows_by_policy["constrain_dp"], field)
                      for field in fields}
    gate_criteria = {
        "task": summaries["constrain_dp"]["task_success_count"] >= 27,
        "safe": summaries["constrain_dp"]["safe_success_count"] >= 26,
        "contact": summaries["constrain_dp"]["obstacle_contact_count"] <= 3,
        "xy": summaries["constrain_dp"]["xy_crossing_count"] <= 3,
        "complete": all(value["complete"] for value in summaries.values()),
        "paired": pairing,
        "same_r38_restored_states_and_conditions": all(state_matches.values()),
    }
    terminal_phase = {
        policy: engine._count_values([row.get("result", {}).get("phase") for row in rows])
        for policy, rows in rows_by_policy.items()
    }
    contact_phase = {policy: engine._phase_counts(rows) for policy, rows in rows_by_policy.items()}
    contact_link = {}
    for policy, rows in rows_by_policy.items():
        counts = {}
        for row in rows:
            link = (row.get("first_contact") or {}).get("link")
            if link:
                counts[link] = counts.get(link, 0) + 1
        contact_link[policy] = counts
    return {
        "complete": True,
        "runtime_errors": {policy: summaries[policy]["runtime_errors"] for policy in policies},
        "policy_metrics": summaries,
        "pairing_identity_matches_scene_by_scene": pairing,
        "normalizer_condition_hashes_match_between_policies": normalized_pairing,
        "paired_base_vs_control": paired_metrics,
        "vs_r38": vs_primary,
        "primary_comparator": {
            "name": "R38 matched position/controller-unit pipeline",
            "metrics": {policy: summarize_pilot(primary_rows[policy], 30) for policy in policies},
            "r40_metrics": summaries,
            "restored_state_condition_identity_matches_scene_by_scene": state_matches,
            "transitions_r38_to_r40": vs_primary,
            "interpretation": "matched end-to-end retraining comparison; not isolated Control-factor causality",
        },
        "gate": {"criteria": gate_criteria, "passed": all(gate_criteria.values())},
        "wall_seconds": {policy: [row.get("wall_seconds") for row in rows]
                         for policy, rows in rows_by_policy.items()},
        "wall_seconds_distribution": {
            policy: _distribution([row.get("wall_seconds") for row in rows])
            for policy, rows in rows_by_policy.items()},
        "terminal_phase_counts": terminal_phase,
        "contact_first_phase_counts": contact_phase,
        "first_contact_link_counts": contact_link,
        "first_contact_entity_pair_positions": "unknown; recorder does not expose contact entity-pair positions",
        "termination_flags": {policy: {"terminated": sum(bool(row["result"].get("terminated")) for row in rows),
                                        "truncated": sum(bool(row["result"].get("truncated")) for row in rows)}
                              for policy, rows in rows_by_policy.items()},
        "first_success_step_distribution": {
            policy: _distribution([row.get("first_native_success_step") for row in rows])
            for policy, rows in rows_by_policy.items()},
        "final_goal_distance_3d_distribution": {
            policy: _distribution([row.get("final_goal_distance_3d") for row in rows])
            for policy, rows in rows_by_policy.items()},
        "final_cube_xyz_and_goal_xyz_present": all(
            "final_cube_xyz" in row and "goal_xyz" in row for rows in rows_by_policy.values() for row in rows),
        "r39_guidance_is_not_primary": {"task": 29, "safe": 11, "contact": 17, "xy": 6},
        "formal_performance_claim": False,
        "comparison_interpretation": "matched end-to-end gauge-augmented retraining; exploratory, not isolated causal attribution",
    }


def analyze_retained_native_rows():
    """Repair report output offline; never creates or retries an environment."""
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot

    registration_path = EXP / "preregistration.json"
    registration = json.loads(registration_path.read_text())
    body = dict(registration)
    registration_sha = body.pop("registration_sha256", None)
    if registration_sha != _digest(body):
        raise ValueError("R40 preregistration checksum mismatch during offline report repair")
    pair = EXP / "paired"
    native_prereg = json.loads((pair / "preregistration.json").read_text())
    if native_prereg.get("registration_sha256") != registration_sha:
        raise ValueError("retained native preregistration does not match R40 registration")
    source_at_native = native_prereg["source_identity_before_native"]
    current_source = source_identity()
    changed = {name: {"native_sha256": source_at_native["files"].get(name),
                      "analysis_sha256": current_source["files"].get(name)}
               for name in set(source_at_native["files"]) | set(current_source["files"])
               if source_at_native["files"].get(name) != current_source["files"].get(name)}
    if set(changed) != {"examples/baselines/constrain/pickcube_xy_gauge_round40.py"}:
        raise ValueError(f"unexpected report-only source delta: {sorted(changed)}")
    for name, expected in ((engine.DATA_H5, registration["dataset"]["h5_sha256"]),
                           (engine.DATA_MANIFEST, registration["dataset"]["manifest_sha256"]),
                           (Path(engine.DATA_MANIFEST).with_name("validation.json"),
                            registration["dataset"]["validation_sha256"]),
                           (registration["paths"]["base_checkpoint"], native_prereg["base_checkpoint_sha256"]),
                           (registration["paths"]["constrain_checkpoint"], native_prereg["constrain_checkpoint_sha256"])):
        if sha256_file(name) != expected:
            raise ValueError(f"R40 pinned native input changed before report repair: {name}")
    eval_inputs = json.loads((EXP / "preregistration_native_eval_inputs.json").read_text())
    r38_reg, r38_tree = _load_r38_primary()
    if (eval_inputs["r38_primary_paired_tree"] != r38_tree
            or eval_inputs["r38_registration_sha256"] != r38_reg["registration_sha256"]):
        raise ValueError("R38 primary comparison inputs differ from pre-native snapshot")
    rows_by_policy, primary_rows = {}, {}
    input_hashes = {}
    for policy in ("base_dp", "constrain_dp"):
        rows_by_policy[policy], primary_rows[policy] = [], []
        for index in range(30):
            current_path = pair / policy / f"episode_{index:03d}.json"
            primary_path = R38_PAIRED / policy / f"episode_{index:03d}.json"
            current_row, primary_row = json.loads(current_path.read_text()), json.loads(primary_path.read_text())
            if current_row.get("checkpoint_sha256") != native_prereg[
                    "base_checkpoint_sha256" if policy == "base_dp" else "constrain_checkpoint_sha256"]:
                raise ValueError(f"R40 retained row checkpoint mismatch: {current_path}")
            rows_by_policy[policy].append(current_row)
            primary_rows[policy].append(primary_row)
            input_hashes[str(current_path.relative_to(EXP))] = sha256_file(current_path)
    analysis_inputs = {
        "native_evaluation_preregistration_sha256": native_prereg["evaluation_registration_sha256"],
        "r40_registration_sha256": registration_sha,
        "native_source_identity": source_at_native,
        "analysis_source_identity": current_source,
        "analysis_only_source_changes": changed,
        "r40_episode_row_sha256": input_hashes,
        "r38_primary_paired_tree_sha256": r38_tree["files_sha256"],
        "r38_primary_files_sha256": r38_tree["files"],
        "native_evaluation_rerun": False,
    }
    snapshot_path = EXP / "analysis_report_inputs.json"
    if snapshot_path.exists():
        raise FileExistsError(f"R40 report input snapshot already exists: {snapshot_path}")
    analysis_inputs["analysis_inputs_sha256"] = _digest(analysis_inputs)
    snapshot_path.write_text(json.dumps(analysis_inputs, indent=2, sort_keys=True) + "\n")
    analysis = _analyze_native_rows(rows_by_policy, primary_rows)
    analysis.update({
        "analysis_input_snapshot_sha256": analysis_inputs["analysis_inputs_sha256"],
        "native_evaluation_registration_sha256": native_prereg["evaluation_registration_sha256"],
        "base_checkpoint_sha256": native_prereg["base_checkpoint_sha256"],
        "constrain_checkpoint_sha256": native_prereg["constrain_checkpoint_sha256"],
        "normalizer_sha256": native_prereg["normalizer_statistics_sha256"],
        "scene_manifest_sha256": native_prereg["scene_manifest_sha256"],
        "scene_h5_sha256": native_prereg["scene_h5_sha256"],
        "source_identity_at_native_evaluation": source_at_native,
        "analysis_source_identity_after_report_fix": current_source,
        "analysis_only_source_changes": changed,
        "native_evaluation_rerun": False,
        "augmentation_inference": "off; raw unshifted native world-frame observations",
    })
    analysis_path = pair / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    (pair / "r38_primary_comparison.json").write_text(
        json.dumps(analysis["primary_comparator"], indent=2, sort_keys=True) + "\n")
    status = {"status": "complete", "analysis_repaired_from_retained_rows": True,
              "native_evaluation_rerun": False, "retained_rows": {p: 30 for p in rows_by_policy},
              "runtime_errors": analysis["runtime_errors"], "gate_passed": analysis["gate"]["passed"],
              "analysis_input_snapshot_sha256": analysis_inputs["analysis_inputs_sha256"]}
    (pair / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    correction = {
        "event": "analysis_only_report_repair_from_completed_native_rows",
        "failure": "legacy aggregator indexed phase_counts, absent from summarize_pilot rows; raised KeyError after all 60 rows had been saved",
        "source_changes": changed,
        "analysis_input_snapshot_sha256": analysis_inputs["analysis_inputs_sha256"],
        "retained_native_row_counts": {p: len(rows) for p, rows in rows_by_policy.items()},
        "runtime_errors": analysis["runtime_errors"],
        "native_evaluation_rerun": False,
        "additional_trial": False,
        "analysis_sha256": sha256_file(analysis_path),
        "status_sha256": sha256_file(pair / "status.json"),
    }
    (EXP / "analysis_report_correction.json").write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")
    return analysis


def evaluate():
    reg = verify()
    base_path, control_path = Path(reg["paths"]["base_checkpoint"]), Path(reg["paths"]["constrain_checkpoint"])
    if not base_path.is_file() or not control_path.is_file():
        raise FileNotFoundError("R40 native evaluation requires both completed final EMA checkpoints")
    validate_report_contract()
    eval_prereg, r38_reg, r38_tree = _write_native_eval_preregistration(reg)
    _install_evaluation_loaders()
    analysis = engine.evaluate_round37(reg["paths"]["registration"])
    analysis["native_evaluation_preregistration_sha256"] = eval_prereg["evaluation_preregistration_sha256"]
    analysis = _enrich_with_r38_primary(analysis, r38_reg, r38_tree)
    engine._atomic_json(Path(reg["paths"]["paired"]) / "analysis.json", analysis)
    engine._atomic_json(Path(reg["paths"]["paired"]) / "r38_primary_comparison.json",
                        analysis["primary_comparator"])
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "refresh-registration", "record-post-training-analysis-correction", "verify", "preflight-base", "preflight-control",
                                               "train-base", "train-control", "evaluate", "analyze-retained"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "register":
        result = register()
    elif args.command == "refresh-registration":
        result = refresh_registration()
    elif args.command == "record-post-training-analysis-correction":
        result = record_post_training_analysis_correction()
    elif args.command == "verify":
        result = verify()
    elif args.command == "preflight-base":
        reg = verify(); Path(reg["paths"]["preflight"]).mkdir(parents=True, exist_ok=True)
        result = engine.preflight_base(reg["paths"]["stats"],
                                       Path(reg["paths"]["preflight"]) / "base.json", args.device,
                                       training_augmentation=METADATA)
    elif args.command == "preflight-control":
        reg = verify(); Path(reg["paths"]["preflight"]).mkdir(parents=True, exist_ok=True)
        result = engine.preflight_control(reg["paths"]["stats"], reg["paths"]["base_checkpoint"],
                                          Path(reg["paths"]["preflight"]) / "control.json", args.device,
                                          training_augmentation=METADATA)
    elif args.command == "train-base":
        result = _train("base", args.device)
    elif args.command == "train-control":
        result = _train("control", args.device)
    elif args.command == "analyze-retained":
        result = analyze_retained_native_rows()
    else:
        result = evaluate()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
