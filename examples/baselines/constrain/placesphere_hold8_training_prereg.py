"""Freeze the single matched hold-8 training/evaluation trial before updates."""

import argparse
import hashlib
import json
from pathlib import Path

from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_HOLD8_VERSION,
    load_matched_training_identity,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23")
VALIDATION = Path("outputs/exp/20261026_184500_placesphere_hold8_collection_r23")
BASELINE = Path("outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval")
SCENES = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
BASE_RUN = "outputs/exp/20261026_195000_placesphere_hold8_base_dp_r23"
BASE_CKPT = "outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23"
CONTROL_RUN = "outputs/exp/20261026_195000_placesphere_hold8_constrain_90k_r23"
CONTROL_CKPT = "outputs/ckpt/20261026_195000_placesphere_hold8_constrain_90k_r23"
EVAL_OUT = "outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23/paired_eval"

SOURCE_FILES = (
    "examples/baselines/constrain/placesphere_hold8_training_prereg.py",
    "examples/baselines/constrain/placesphere_manifest_dataset.py",
    "examples/baselines/constrain/placesphere_base_dp.py",
    "examples/baselines/constrain/train_placesphere_base_dp.py",
    "examples/baselines/constrain/placesphere_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/constrain_ckpt.py",
    "examples/baselines/constrain/placesphere_learning_pilot.py",
    "examples/baselines/constrain/placesphere_matched_collection.py",
    "examples/baselines/constrain/placesphere_matched_terminal.py",
    "examples/baselines/constrain/validate_placesphere_matched_collection.py",
    "examples/baselines/constrain/placesphere_pilot.py",
    "examples/baselines/constrain/placesphere_obstacle.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/placesphere_transport_detour.py",
    "examples/baselines/constrain/evaluate_placesphere_matched.py",
    "examples/baselines/constrain/diagnose_placesphere_failure.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preregister(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    identity_path = DATA / "training_identity.json"
    identity = load_matched_training_identity(
        identity_path, DATA / "trajectory.h5", DATA / "trajectory.json",
        expected_version=MATCHED_HOLD8_VERSION)
    validation = json.loads((VALIDATION / "dataset_validation.json").read_text())
    collection = json.loads((DATA / "report.json").read_text())
    if (not validation.get("complete") or validation.get("class_counts") != {"direct": 1000, "obstacle": 300}
            or validation.get("t_plus_one_restore_audits_passed") != 1300
            or validation.get("obstacle_task_identity", {}).get("greedy_unique") != 300
            or collection.get("accepted") != {"direct": 1000, "obstacle": 300}
            or identity.get("release_hold_steps") != 8):
        raise ValueError("hold-8 package is not the complete independently validated 1,000/300 cohort")
    scene_manifest = SCENES / "manifest.json"
    primary_summary = BASELINE / "paired_summary.json"
    if not scene_manifest.is_file() or not primary_summary.is_file():
        raise FileNotFoundError("frozen matched dev scenes or Round-22 primary comparison is missing")
    files = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    output_paths = {"base_run": BASE_RUN, "base_checkpoint": BASE_CKPT,
                    "constrain_run": CONTROL_RUN, "constrain_checkpoint": CONTROL_CKPT,
                    "paired_evaluation": EVAL_OUT}
    payload = {
        "kind": "placesphere_matched_hold8_single_training_trial_r23",
        "factor": "expert-only 8-step open-neutral hold after release; selected on eight frozen matched training OBSTACLE episodes by preregistered native physics rule",
        "data": {"dataset_dir": str(DATA), "h5_sha256": identity["h5_sha256"],
                 "manifest_sha256": identity["manifest_sha256"],
                 "training_identity_sha256": sha256(identity_path),
                 "dataset_validation_sha256": sha256(VALIDATION / "dataset_validation.json"),
                 "registration_sha256": identity["registration_sha256"],
                 "class_counts": identity["class_counts"], "release_hold_steps": 8,
                 "old_trajectory_bytes_copied": False},
        "source_hashes": files,
        "source_identity_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "base_training": {"updates": 30000, "seed": 1, "batch_size": 1024, "lr": 1e-4,
                          "warmup": 500, "ema_power": .75, "selected_weights": "final EMA",
                          "device": "cuda", "budget_seconds": 2400},
        "constrain_training": {"updates": 90000, "seed": 1, "batch_size": 1024, "lr": 1e-4,
                               "warmup": 500, "ema_power": .75, "base_source": "new Base final EMA",
                               "selected_weights": "final EMA", "device": "cuda", "budget_seconds": 6000},
        "evaluation": {"scene_dir": str(SCENES), "scene_manifest_sha256": sha256(scene_manifest),
                       "scene_count": 30, "seeds": "frozen manifest unchanged",
                       "policy_step_limit": 200, "episode_wall_seconds": 120,
                       "overall_wall_seconds": 8000,
                       "primary_comparison": str(BASELINE),
                       "primary_paired_summary_sha256": sha256(primary_summary),
                       "secondary_comparison": "outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired",
                       "selected_weights": "final EMA, chosen before opening dev results"},
        "gate": {"task_success_min": 27, "safe_success_min": 26,
                 "contact_max": 3, "xy_crossing_max": 3,
                 "runtime_errors_remain_in_denominator": True,
                 "main_comparison": "paired hold-8 trial vs current 26/25/1/0 matched EMA",
                 "transitions_required": ["task/safe improved-regressed-both-neither",
                                          "contact/XY cleared-introduced-both-neither"]},
        "outputs": output_paths,
        "training_data_selection": {"direct_ids": "all manifest direct class IDs (1,000)",
                                    "obstacle_ids": "all manifest obstacle class IDs (300)",
                                    "window_sampling": "uniform official full-state 2/16 window sampling",
                                    "ordinary_data_not_copied": True,
                                    "native_thresholds_or_geometry_changed": False},
        "budgets_seconds": {"base": 2400, "constrain": 6000, "evaluation": 8000,
                            "evaluation_episode": 120, "train_run_cleanup": "timeout TERM then KILL after 30 s"},
        "training_not_started_at_registration": True,
        "argv": __import__("sys").argv,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = preregister(args.output)
    print(json.dumps({"output": args.output, "source_identity_sha256": result["source_identity_sha256"],
                      "h5_sha256": result["data"]["h5_sha256"], "hold_steps": 8}, sort_keys=True))


if __name__ == "__main__":
    main()
