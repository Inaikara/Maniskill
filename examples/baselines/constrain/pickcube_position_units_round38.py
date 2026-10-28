"""Matched PickCube Round 38 fixed physical-position-unit experiment."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_position_units import (
    PROFILE, encode_observation, profile_metadata, validate_profile,
)

EXP = Path("outputs/exp/20261028_221700_pickcube_position_units_r38")
BASE_CKPT = Path("outputs/ckpt/20261028_221700_pickcube_position_units_r38_base")
CONTROL_CKPT = Path("outputs/ckpt/20261028_221700_pickcube_position_units_r38_constrain")
FORMAT_BASE = "pickcube_position_controller_units_base_v1"
FORMAT_CONTROL = "pickcube_position_controller_units_constrain_v1"


def _stats_path():
    return EXP / "position_encoding.json"


def _install_profile():
    """Bind the unchanged matched-training engine to this disjoint typed profile."""
    engine.PROFILE = PROFILE
    engine.BASE_FORMAT = FORMAT_BASE
    engine.CONSTRAIN_FORMAT = FORMAT_CONTROL
    engine.normalize_observation = lambda values, _stats=None: encode_observation(values)
    engine.validate_stats = validate_profile
    engine.EPISODE_TIMEOUT_SECONDS = 300

    def load_stats(path, expected_sha=None):
        path = Path(path)
        if expected_sha and sha256_file(path) != expected_sha:
            raise ValueError("fixed observation profile file hash differs from registration")
        return validate_profile(json.loads(path.read_text()))

    def source_identity():
        files = {
            name: sha256_file(name) for name in (
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
                "examples/baselines/constrain/pickcube_expert.py",
                "mani_skill/envs/tasks/tabletop/pick_cube.py",
            )
        }
        return {"files": files, "files_sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}

    engine.load_stats = load_stats
    engine.source_identity = source_identity


def register():
    _install_profile()
    exp = EXP
    exp.mkdir(parents=True, exist_ok=False)
    stats = profile_metadata()
    stats["source_identity"] = {
        "selected_class": "fixed_physical_reference_not_fitted",
        "selected_episode_ids": list(range(1000)),
        "h5_sha256": engine.DATA_H5_SHA,
        "manifest_sha256": engine.DATA_MANIFEST_SHA,
    }
    (exp / "position_encoding.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    r36 = json.loads(engine.R36_REGISTRATION.read_text())
    scenes = Path(r36["paths"]["dev_scenes"])
    if (sha256_file(scenes / "manifest.json") != r36["validation"]["dev_manifest"]["sha256"]
            or sha256_file(scenes / "scenes.h5") != r36["validation"]["dev_scenes_h5"]["sha256"]
            or sha256_file("mani_skill/envs/tasks/tabletop/pick_cube.py")
            != r36["validation"]["maniskill_pickcube_source"]["sha256"]):
        raise ValueError("R36 scene or native PickCube source pin mismatch")
    engine.ROUND37_EXP = exp
    engine.ROUND37_BASE_CKPT = BASE_CKPT
    engine.ROUND37_CONTROL_CKPT = CONTROL_CKPT
    engine.R36_DEV = scenes
    engine.SOURCE_FILES = ()
    reg = {
        "round": 38, "status": "registered_before_training_and_native_evaluation",
        "profile": profile_metadata(), "profile_sha256": profile_metadata()["sha256"],
        "paths": {"experiment": str(exp), "registration": str(exp / "preregistration.json"),
                  "stats": str(exp / "position_encoding.json"), "base_run": str(exp / "train_base"),
                  "base_checkpoint_dir": str(BASE_CKPT), "base_checkpoint": str(BASE_CKPT / "final.pt"),
                  "constrain_run": str(exp / "train_constrain"),
                  "constrain_checkpoint_dir": str(CONTROL_CKPT),
                  "constrain_checkpoint": str(CONTROL_CKPT / "final.pt"),
                  "preflight": str(exp / "preflight"), "paired": str(exp / "paired"),
                  "dev_scenes": str(scenes), "r36_paired": str(engine.R36_PAIRED)},
        "dataset": {"h5_path": engine.DATA_H5, "h5_sha256": engine.DATA_H5_SHA,
                    "manifest_path": engine.DATA_MANIFEST, "manifest_sha256": engine.DATA_MANIFEST_SHA,
                    "validation_sha256": "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f",
                    "direct_ids": list(range(1000)), "obstacle_ids": list(range(1000, 1500)),
                    "direct_windows": 83004, "obstacle_windows": 45840,
                    "actions": "native identity seven-channel; unchanged", "closed_terminal_grip": -1},
        "observation_encoding": {**profile_metadata(), "fit_from_data": False,
                                 "controller_translation_reference_m": 0.1,
                                 "apply_once_to_42d_state": True,
                                 "relative_obstacle_hint": "raw native 45-D latest frame; unchanged"},
        "normalizer": {"file_sha256": sha256_file(exp / "position_encoding.json"),
                       "statistics_sha256": profile_metadata()["sha256"],
                       "fit_from_data": False, "fit_frame_count": 0,
                       "fit_source_ids": [], "floor": None, "floored_fields": []},
        "training": {"base_updates": 30000, "constrain_updates": 150000,
                     **engine.TRAINING, "base_class": "direct", "constrain_class": "obstacle",
                     "base_frozen_in_constrain": True, "loss": "unweighted epsilon MSE",
                     "recovery_every_updates": 10000, "budgets_seconds": {"base": 3000, "control": 9000}},
        "model": {"base": dict(engine.BASE_CONFIG), "constrain": dict(engine.CONSTRAIN_CONFIG),
                  "condition_dim": 84, "action_dim": 7},
        "sampler": dict(engine.DIFFUSION),
        "native_evaluation": {"episodes": 30, "max_control_steps": 500,
                              "wall_timeout_seconds_per_episode": 300,
                              "outer_timeout_seconds": 36000,
                              "scene_manifest_sha256": sha256_file(scenes / "manifest.json"),
                              "scene_h5_sha256": sha256_file(scenes / "scenes.h5"),
                              "noise_seed_rule": "torch.manual_seed(scene seed) independently per policy",
                              "primary_reference": {"r36_base": {"task": 10, "safe": 0, "contact": 30, "xy": 15},
                                                     "r36_constrain": {"task": 25, "safe": 16, "contact": 13, "xy": 4}},
                              "gate": {"task_gte": 27, "safe_gte": 26, "contact_lte": 3, "xy_lte": 3}},
        "formal_performance_claim": False,
        "interpretation": "matched end-to-end observation representation retraining; not isolated control attribution",
        "source_identity": engine.source_identity(),
        "r36_registration_sha256": sha256_file(engine.R36_REGISTRATION),
    }
    reg["registration_sha256"] = hashlib.sha256(
        json.dumps(reg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (exp / "preregistration.json").write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    return reg


def refresh_registration():
    """Record source corrections discovered during preflight, before training."""
    path = EXP / "preregistration.json"
    _install_profile()
    old = json.loads(path.read_text())
    snapshot = EXP / "preregistration_preflight_draft.json"
    if not snapshot.exists():
        snapshot.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")
    old["source_identity"] = engine.source_identity()
    old.pop("registration_sha256", None)
    old["registration_sha256"] = hashlib.sha256(
        json.dumps(old, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")


def verify(path):
    _install_profile()
    engine.ROUND37_EXP = EXP
    engine.ROUND37_BASE_CKPT = BASE_CKPT
    engine.ROUND37_CONTROL_CKPT = CONTROL_CKPT
    reg = json.loads(Path(path).read_text())
    body = dict(reg)
    digest = body.pop("registration_sha256", None)
    if digest != hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise ValueError("Round 38 registration checksum mismatch")
    if reg["source_identity"] != engine.source_identity():
        raise ValueError("Round 38 source identity differs from preregistration")
    for name, expected in ((engine.DATA_H5, engine.DATA_H5_SHA),
                           (engine.DATA_MANIFEST, engine.DATA_MANIFEST_SHA)):
        if sha256_file(name) != expected:
            raise ValueError(f"frozen data hash mismatch: {name}")
    if sha256_file(reg["paths"]["stats"]) != reg["normalizer"]["file_sha256"]:
        raise ValueError("Round 38 position profile artifact identity mismatch")
    validate_profile(json.loads(Path(reg["paths"]["stats"]).read_text()))
    return reg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "refresh-registration", "preflight-base", "train-base",
                         "preflight-control", "train-control", "evaluate", "analyze-existing"))
    parser.add_argument("--registration", type=Path, default=EXP / "preregistration.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "register":
        print(json.dumps(register(), indent=2, sort_keys=True))
        return
    if args.command == "refresh-registration":
        refresh_registration()
        return
    reg = verify(args.registration)
    engine.R36_DEV = Path(reg["paths"]["dev_scenes"])
    if args.command == "analyze-existing":
        result = engine.analyze_existing_round37(args.registration)
    elif args.command == "preflight-base":
        Path(reg["paths"]["preflight"]).mkdir(parents=True, exist_ok=True)
        result = engine.preflight_base(reg["paths"]["stats"], Path(reg["paths"]["preflight"]) / "base.json", args.device)
    elif args.command == "train-base":
        Path(reg["paths"]["base_run"]).parent.mkdir(parents=True, exist_ok=True)
        result = engine.train_base(reg["paths"]["stats"], reg["paths"]["base_run"],
                                  reg["paths"]["base_checkpoint_dir"], args.device,
                                  provenance={"registration": str(args.registration), "profile": PROFILE})
        result["source_identity_after_training"] = engine.source_identity()
        result["dataset_identity_after_training"] = {
            "h5_sha256": sha256_file(engine.DATA_H5),
            "manifest_sha256": sha256_file(engine.DATA_MANIFEST),
        }
        result["source_identity_matches_registration"] = result["source_identity_after_training"] == reg["source_identity"]
        result["dataset_identity_matches_registration"] = (
            result["dataset_identity_after_training"]["h5_sha256"] == engine.DATA_H5_SHA
            and result["dataset_identity_after_training"]["manifest_sha256"] == engine.DATA_MANIFEST_SHA)
        result["status"] = "complete" if (result["source_identity_matches_registration"]
                                           and result["dataset_identity_matches_registration"]
                                           and result["all_losses_finite"]) else "pending_audit"
        Path(reg["paths"]["base_run"], "train_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif args.command == "preflight-control":
        Path(reg["paths"]["preflight"]).mkdir(parents=True, exist_ok=True)
        result = engine.preflight_control(reg["paths"]["stats"], reg["paths"]["base_checkpoint"],
                                          Path(reg["paths"]["preflight"]) / "control.json", args.device)
    elif args.command == "train-control":
        Path(reg["paths"]["constrain_run"]).parent.mkdir(parents=True, exist_ok=True)
        result = engine.train_constrain(reg["paths"]["stats"], reg["paths"]["base_checkpoint"],
                                       reg["paths"]["constrain_run"], reg["paths"]["constrain_checkpoint_dir"],
                                       args.device, provenance={"registration": str(args.registration), "profile": PROFILE})
        result["source_identity_after_training"] = engine.source_identity()
        result["dataset_identity_after_training"] = {
            "h5_sha256": sha256_file(engine.DATA_H5),
            "manifest_sha256": sha256_file(engine.DATA_MANIFEST),
        }
        result["source_identity_matches_registration"] = result["source_identity_after_training"] == reg["source_identity"]
        result["dataset_identity_matches_registration"] = (
            result["dataset_identity_after_training"]["h5_sha256"] == engine.DATA_H5_SHA
            and result["dataset_identity_after_training"]["manifest_sha256"] == engine.DATA_MANIFEST_SHA)
        result["status"] = "complete" if (result["source_identity_matches_registration"]
                                           and result["dataset_identity_matches_registration"]
                                           and result["all_losses_finite"]) else "pending_audit"
        Path(reg["paths"]["constrain_run"], "train_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        result = engine.evaluate_round37(args.registration)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
