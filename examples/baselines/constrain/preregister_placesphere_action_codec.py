"""Freeze R30 action-codec source/data/checkpoint/evaluation identities."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.baselines.constrain.placesphere_action_codec import CONTROLLER_RANGE
from examples.baselines.constrain.placesphere_base_dp import sha256_file


ROOT = Path(__file__).resolve().parents[3]
DATA = Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3")
SCENES = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
PRIMARY = Path("outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval")
EVAL_SOURCES = {
    "evaluate_placesphere_matched.py": "evaluate_placesphere_matched.py",
    "placesphere_base_dp.py": "placesphere_base_dp.py",
    "placesphere_constrain.py": "placesphere_constrain.py",
    "placesphere_inference_sampler.py": "placesphere_inference_sampler.py",
    "placesphere_learning_pilot.py": "placesphere_learning_pilot.py",
    "placesphere_pilot.py": "placesphere_pilot.py",
    "placesphere_transport_detour.py": "placesphere_transport_detour.py",
    "placesphere_scenes.py": "placesphere_scenes.py",
    "placesphere_obstacle.py": "placesphere_obstacle.py",
    "diagnose_placesphere_failure.py": "diagnose_placesphere_failure.py",
    "constrain_denoiser.py": "constrain_denoiser.py",
    "constrain_ckpt.py": "constrain_ckpt.py",
    "placesphere_action_codec.py": "placesphere_action_codec.py",
}
TRAIN_SOURCES = [
    "examples/baselines/constrain/train_placesphere_base_dp.py",
    "examples/baselines/constrain/placesphere_base_dp.py",
    "examples/baselines/constrain/placesphere_manifest_dataset.py",
    "examples/baselines/constrain/placesphere_action_codec.py",
    "examples/baselines/constrain/audit_placesphere_action_encoding.py",
    "examples/baselines/constrain/preregister_placesphere_action_codec.py",
    "examples/baselines/constrain/train_logging.py",
    "examples/baselines/constrain/placesphere_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/constrain_ckpt.py",
    "examples/baselines/constrain/placesphere_obstacle.py",
    "examples/baselines/diffusion_policy/train.py",
    "examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py",
]


def _tree_sha(directory):
    root = ROOT / directory
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    return {"path": str(directory), "files": len(entries),
            "sha256": hashlib.sha256("".join(
                name + "\0" + digest + "\n" for name, digest in entries).encode()).hexdigest()}


def registration(stamp, audit_path):
    audit_path = Path(audit_path)
    audit = json.loads((ROOT / audit_path).read_text())
    if (audit.get("format") != "placesphere_native_action_codec_audit_v1"
            or audit.get("audit_passed") is not True
            or audit.get("codec_profile") != CONTROLLER_RANGE
            or audit.get("dataset", {}).get("class_counts") != {"direct": 1000, "obstacle": 500}
            or audit.get("bounds", {}).get("actual_actions_clipped_or_modified") is not False):
        raise ValueError("the frozen full-dataset action-range audit did not pass")
    exp = f"outputs/exp/{stamp}_placesphere_action_codec_r30"
    base_ckpt = f"outputs/ckpt/{stamp}_placesphere_action_codec_r30_base/final.pt"
    control_ckpt = f"outputs/ckpt/{stamp}_placesphere_action_codec_r30_constrain_150k/final.pt"
    eval_sources = {name: sha256_file(ROOT / "examples/baselines/constrain" / filename)
                    for name, filename in EVAL_SOURCES.items()}
    training_sources = {name: sha256_file(ROOT / name) for name in TRAIN_SOURCES}
    data = {
        "dataset": DATA.as_posix(),
        "data_h5_sha256": audit["dataset"]["h5_sha256"],
        "data_manifest_sha256": audit["dataset"]["manifest_sha256"],
        "training_identity_sha256": audit["dataset"]["training_identity_sha256"],
        "direct_episodes": 1000, "obstacle_episodes": 500,
        "direct_ids": [*range(0, 600, 2), *range(600, 1300)],
        "obstacle_ids": list(range(1300, 1500)),
        "direct_windows": 135417, "obstacle_windows": 80151,
        "action_audit": audit_path.as_posix(),
        "action_audit_sha256": sha256_file(ROOT / audit_path),
    }
    return {
        "format": "placesphere_r30_action_codec_training_preregistration_v1",
        "created_before_long_training": True,
        "branch": "goal/transport-detour-optimization",
        "base_head": "60ccdf0572016d9d1090bb1382d54de9c80e11d9",
        "integration_branch": "dev",
        "user_contract": "fixed 1000 DIRECT / 500 OBSTACLE; no new data, demos, layouts, or seeds; frozen-data representation optimization",
        "action_encoding": CONTROLLER_RANGE,
        "action_semantics": "native controller-normalized coordinates; pose scale is not meters/radians; native HDF5 remains unchanged; encode by fixed known controller cap and decode before native action map/clamp",
        "audit": {"all_1500_selected_episodes_read": True,
                  "pose_bounds_native_normalized": [-0.35, 0.35],
                  "gripper_bounds_native": [-1.0, 1.0],
                  "no_clipping_or_data_mutation": True,
                  "effective_snr_timesteps": [0, 10, 25, 50, 75, 99],
                  "interpretation": "descriptive per-channel standard-noising SNR only"},
        "data": data,
        "initial_models": {
            "base": {"updates": 30000, "selected_class": "direct", "seed": 1,
                     "batch_size": 1024, "lr": 1e-4, "warmup": 500,
                     "schedule": "cosine", "ema_power": 0.75,
                     "selected_weights": "final EMA fixed before training",
                     "run_dir": f"{exp}/training_base", "checkpoint": base_ckpt},
            "constrain": {"updates": 150000, "selected_class": "obstacle", "seed": 1,
                          "batch_size": 1024, "lr": 1e-4, "warmup": 500,
                          "schedule": "cosine", "ema_power": 0.75,
                          "base_checkpoint": base_ckpt,
                          "selected_weights": "final EMA fixed before training",
                          "run_dir": f"{exp}/training_constrain_150k", "checkpoint": control_ckpt},
            "objective": "uniform official windows; unchanged epsilon MSE/DDPM100 betas; no phase/proximity weighting"},
        "preflight": {"base_two_finite_updates": f"{exp}/preflight_base",
                      "constrain_two_finite_updates_and_frozen_base": f"{exp}/preflight_constrain",
                      "strict_matched_profile_and_independent_base_equivalence": True},
        "evaluation": {
            "sampler": "DDIMScheduler eta=0; 100 steps",
            "scene_dir": SCENES.as_posix(), "scene_manifest_sha256": sha256_file(ROOT / SCENES / "manifest.json"),
            "scene_h5_sha256": sha256_file(ROOT / SCENES / "scenes.h5"),
            "primary_baseline": PRIMARY.as_posix(),
            "primary_summary_sha256": sha256_file(ROOT / PRIMARY / "paired_summary.json"),
            "output_dir": f"{exp}/paired_eval", "stride": 2,
            "prediction_horizon": 16, "executed_slice": [1, 9],
            "max_control_steps": 200, "episode_wall_seconds": 300,
            "fixed_denominator_per_policy": 30, "runtime_errors_retained": True,
            "ensemble_size": 1, "gate": {"task_min": 27, "safe_min": 26,
                                         "contact_max": 3, "xy_max": 3},
            "comparison_note": "new codec changes both policies' weights; this is an end-to-end pipeline result, not control-only attribution",
        },
        "inputs": {
            "base_checkpoint_sha256": None, "constrain_checkpoint_sha256": None,
            "data_h5_sha256": data["data_h5_sha256"],
            "data_manifest_sha256": data["data_manifest_sha256"],
            "training_identity_sha256": data["training_identity_sha256"],
            "scene_manifest_sha256": sha256_file(ROOT / SCENES / "manifest.json"),
            "scene_h5_sha256": sha256_file(ROOT / SCENES / "scenes.h5"),
            "primary_summary_sha256": sha256_file(ROOT / PRIMARY / "paired_summary.json"),
        },
        "training_source_sha256": training_sources,
        "evaluation_source_sha256": eval_sources,
        "evaluation_primary_tree": _tree_sha(PRIMARY),
        "budgets": {"base_training_seconds": 2400, "constrain_training_seconds": 9000,
                    "evaluation_seconds": 18000, "evaluation_episodes": 60},
        "gate_failure_policy": "stop formal08 preparation; ordinary optimization remains authorized on exactly these frozen data and dev scenes",
    }


def evaluation_registration(training_registration_path, base_checkpoint, constrain_checkpoint):
    training_path = ROOT / training_registration_path
    training = json.loads(training_path.read_text())
    base_path, control_path = ROOT / base_checkpoint, ROOT / constrain_checkpoint
    base = __import__("torch").load(base_path, map_location="cpu", weights_only=False)
    control = __import__("torch").load(control_path, map_location="cpu", weights_only=False)
    correction_path = ROOT / "outputs/exp/20261027_224727_placesphere_action_codec_r30/checkpoint_metadata_correction.json"
    correction = json.loads(correction_path.read_text())
    expected_profile = training["action_encoding"]
    if (base.get("action_encoding") != expected_profile
            or control.get("config", {}).get("action_encoding") != expected_profile
            or base.get("step") != 30000 or control.get("step") != 150000
            or base.get("data_identity", {}).get("dataset_version") != "placesphere_matched_terminal_r26_hold8_obstacle500_v1"
            or control.get("config", {}).get("base_weights", {}).get("sha256") != sha256_file(base_path)
            or correction.get("corrected_base", {}).get("sha256") != sha256_file(base_path)
            or correction.get("corrected_constrain", {}).get("sha256") != sha256_file(control_path)
            or correction.get("tensor_identity", {}).get("constrain_ema_shadow_params_unchanged") is not True):
        raise ValueError("final checkpoints do not match the registered step counts and codec profile")
    stamp = training["initial_models"]["base"]["run_dir"].split("/")[2].split("_placesphere")[0]
    exp = f"outputs/exp/{stamp}_placesphere_action_codec_r30"
    return {
        "format": "placesphere_r30_action_codec_ddim100_preregistration_v1",
        "training_registration": training_registration_path.as_posix(),
        "training_registration_sha256": sha256_file(training_path),
        "checkpoint_metadata_correction": {"path": correction_path.relative_to(ROOT).as_posix(),
                                            "sha256": sha256_file(correction_path)},
        "action_encoding": expected_profile,
        "source_sha256": {name: sha256_file(ROOT / "examples/baselines/constrain" / filename)
                          for name, filename in EVAL_SOURCES.items()},
        "data": {"dataset": training["data"]["dataset"],
                 "direct_episodes": 1000, "obstacle_episodes": 500,
                 "h5_sha256": training["data"]["data_h5_sha256"],
                 "manifest_sha256": training["data"]["data_manifest_sha256"],
                 "training_identity_sha256": training["data"]["training_identity_sha256"]},
        "scenes": {"path": training["evaluation"]["scene_dir"],
                   "manifest_sha256": training["evaluation"]["scene_manifest_sha256"],
                   "scenes_h5_sha256": training["evaluation"]["scene_h5_sha256"]},
        "checkpoints": {"base_dp": {"path": base_checkpoint.as_posix(), "sha256": sha256_file(base_path)},
                        "constrain_dp": {"path": constrain_checkpoint.as_posix(), "sha256": sha256_file(control_path)}},
        "inputs": {"base_checkpoint_sha256": sha256_file(base_path),
                   "constrain_checkpoint_sha256": sha256_file(control_path),
                   "data_h5_sha256": training["data"]["data_h5_sha256"],
                   "data_manifest_sha256": training["data"]["data_manifest_sha256"],
                   "training_identity_sha256": training["data"]["training_identity_sha256"],
                   "scene_manifest_sha256": training["evaluation"]["scene_manifest_sha256"],
                   "scene_h5_sha256": training["evaluation"]["scene_h5_sha256"],
                   "primary_summary_sha256": training["evaluation"]["primary_summary_sha256"]},
        "sampler": {"name": "DDIMScheduler", "inference_steps": 100, "eta": 0.0,
                    "training_scheduler_unchanged": True},
        "protocol": {"sampler": "DDIMScheduler eta=0; 100 steps", "ensemble_size": 1,
                     "primary_gate": {"task_min": 27, "safe_min": 26, "contact_max": 3, "xy_max": 3},
                     "stride": 2, "prediction_horizon": 16,
                     "executed_action_slice": [1, 9], "max_control_steps": 200,
                     "episode_wall_seconds": 300, "fixed_denominator_per_policy": 30,
                     "primary_baseline": "outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval",
                     "primary_baseline_summary_sha256": training["evaluation"]["primary_summary_sha256"],
                     "output_dir": f"{exp}/paired_eval"},
        "budgets": {"outer_wall_seconds": 18000, "per_policy_episodes": 30,
                    "total_policy_episodes": 60},
        "evaluation_primary_tree": training["evaluation_primary_tree"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-after-training", action="store_true")
    parser.add_argument("--training-registration", type=Path)
    parser.add_argument("--base-checkpoint", type=Path)
    parser.add_argument("--constrain-checkpoint", type=Path)
    args = parser.parse_args(argv)
    if args.evaluation_after_training:
        result = evaluation_registration(args.training_registration,
                                        args.base_checkpoint, args.constrain_checkpoint)
    else:
        result = registration(args.stamp, args.audit)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": args.output.as_posix(), "sha256": sha256_file(path),
                      "registered_sources": len(result.get("training_source_sha256", {}))
                                             + len(result.get("evaluation_source_sha256", {}))}, indent=2))


if __name__ == "__main__":
    main()
