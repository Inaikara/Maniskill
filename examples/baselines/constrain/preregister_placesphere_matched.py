"""Freeze the matched-r21b PlaceSphere training and 30-scene comparison plan."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from examples.baselines.constrain.placesphere_base_dp import sha256_file
from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_H5_SHA256, MATCHED_MANIFEST_SHA256, MATCHED_VERSION,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = Path("demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b")
DEV = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
PRIMARY = Path("outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired")
SECONDARY = Path("outputs/exp/20261024_155244_placesphere_data_expansion/paired")

SOURCES = (
    "examples/baselines/constrain/preregister_placesphere_matched.py",
    "examples/baselines/constrain/placesphere_manifest_dataset.py",
    "examples/baselines/constrain/placesphere_base_dp.py",
    "examples/baselines/constrain/train_placesphere_base_dp.py",
    "examples/baselines/constrain/placesphere_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/constrain_ckpt.py",
    "examples/baselines/constrain/evaluate_placesphere_matched.py",
    "examples/baselines/constrain/placesphere_learning_pilot.py",
    "examples/baselines/constrain/placesphere_transport_detour.py",
    "examples/baselines/constrain/placesphere_pilot.py",
    "examples/baselines/constrain/placesphere_scenes.py",
    "examples/baselines/constrain/diagnose_placesphere_failure.py",
    "examples/baselines/diffusion_policy/train.py",
    "examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py",
    "mani_skill/envs/tasks/tabletop/place_sphere.py",
    "mani_skill/examples/motionplanning/panda/solutions/place_sphere.py",
)


def _tree_hash(directory):
    digest = hashlib.sha256()
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(directory).as_posix().encode())
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_registration():
    data_path, manifest_path = DATA / "trajectory.h5", DATA / "trajectory.json"
    if sha256_file(data_path) != MATCHED_H5_SHA256 or sha256_file(manifest_path) != MATCHED_MANIFEST_SHA256:
        raise ValueError("qualifying matched-r21b dataset identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    by_kind = {kind: [int(ep["episode_id"]) for ep in manifest["episodes"] if ep.get("kind") == kind]
               for kind in ("direct", "obstacle")}
    if (len(by_kind["direct"]) != 1000 or len(by_kind["obstacle"]) != 300 or
            set(by_kind["direct"]) & set(by_kind["obstacle"]) or
            set(by_kind["direct"]) | set(by_kind["obstacle"]) != set(range(1300))):
        raise ValueError("matched manifest role IDs/counts do not cover 1,300 unique episode IDs")
    source_hashes = {rel: sha256_file(ROOT / rel) for rel in SOURCES}
    primary_hashes = {role: _tree_hash(PRIMARY / role) for role in ("base_dp", "constrain_dp")}
    secondary_hashes = {role: _tree_hash(SECONDARY / role) for role in ("base_dp", "constrain_dp")}
    return {
        "format": "placesphere_matched_training_preregistration_v1",
        "dataset_version": MATCHED_VERSION,
        "worktree_git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_head_is_worktree_source_identity": False,
        "source_files_sha256": source_hashes,
        "source_identity_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "dataset": {"directory": str(DATA), "h5_sha256": MATCHED_H5_SHA256,
                    "manifest_sha256": MATCHED_MANIFEST_SHA256,
                    "manifest_ids_by_class": by_kind,
                    "nonqualifying_r21_package_excluded": "20261026_123000_matched_terminal_r21",
                    "direct_layout_audit": {"unique": 472, "repeats": 528},
                    "obstacle_layout_audit": {"unique": 300, "repeats": 0},
                    "full_trajectory_tails_preserved": True},
        "action_window_audit": {"official_dp_training_windows": {"direct": 132080, "obstacle": 50077},
            "updates_batch_window_coverage_epochs": {"base_direct_30000": 232.59,
                "constrain_obstacle_90000": 1841.70},
            "stride_8_diagnostic_windows": {"direct": 16189, "obstacle": 6175},
            "direct": {"episodes_over_200": 60,
            "length_quantiles_median_p90_p95_max": [105, 147, 321, 909],
            "actions_beyond_200_fraction": 0.15830327622482718,
            "open_near_zero_actions": 28723, "open_near_zero_fraction": 0.21583258190562069,
            "windows_by_current_phase": {"grasp_approach": 5700, "gripper_close": 1000,
                "lift": 1006, "transport": 3620, "stabilize": 177, "retreat": 158,
                "settle": 3311, "grasp_pose": 622, "clearance_approach": 534, "release": 61}},
            "obstacle": {"episodes_over_200": 29,
            "length_quantiles_median_p90_p95_max": [126, 185, 544, 889],
            "actions_beyond_200_fraction": 0.20707862715127937,
            "open_near_zero_actions": 12855, "open_near_zero_fraction": 0.25517597316235585,
            "windows_by_current_phase": {"grasp_approach": 1705, "gripper_close": 300,
                "lift": 301, "transport": 1881, "stabilize": 70, "grasp_pose": 181,
                "clearance_approach": 145, "retreat": 63, "settle": 1513, "release": 16}},
            "native_static_8_action_windows_before_first_success": 0,
            "sampling": "unchanged uniform official DP trajectory-window sampler; no reweight/truncation"},
        "training": {"base": {"selected_class": "direct", "updates": 30000},
            "constrain": {"selected_class": "obstacle", "updates": 90000,
                "initialization": "new matched Base final EMA", "trainable": "control branch only",
                "epsilon_loss": "unweighted", "phase_or_proximity_weight": None},
            "seed": 1, "batch_size": 1024, "learning_rate": 1e-4, "warmup": 500,
            "optimizer": "AdamW betas=(0.95,0.999), weight_decay=1e-6",
            "scheduler": "cosine with exact declared update budget", "ema_power": 0.75,
            "selected_checkpoint": "final EMA fixed before training; no dev selection",
            "timeouts_seconds": {"base": 2400, "constrain": 6000}},
        "development_evaluation": {"scene_directory": str(DEV), "scene_manifest_sha256": sha256_file(DEV / "manifest.json"),
            "scene_h5_sha256": sha256_file(DEV / "scenes.h5"), "scene_count": 30,
            "paired_baselines": {"primary_300_90k": {"directory": str(PRIMARY), "tree_sha256": primary_hashes},
                                 "secondary_100_obstacle": {"directory": str(SECONDARY), "tree_sha256": secondary_hashes}},
            "max_policy_steps": 200, "episode_wall_seconds": 120, "overall_timeout_seconds": 8000,
            "seed_rule": "same frozen scene seed independently reset before each policy",
            "gate": {"task_success_min": .90, "safe_success_min": .85, "contact_max": .10, "xy_crossing_max": .10},
            "runtime_errors": "retained in requested denominator; no silent drop"},
        "outputs": {"base_exp": "outputs/exp/20261026_153000_placesphere_matched_base_dp",
            "base_ckpt": "outputs/ckpt/20261026_153000_placesphere_matched_base_dp",
            "constrain_exp": "outputs/exp/20261026_153000_placesphere_matched_constrain_90k",
            "constrain_ckpt": "outputs/ckpt/20261026_153000_placesphere_matched_constrain_90k",
            "evaluation": "outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval"},
        "claims": {"expert_tails_do_not_prove_200_step_policy_sufficiency": True,
            "matched_dataset_and_replacement_layout_effects_not_isolated_pure_expert_cause": True,
            "development_set_is_not_formal_or_generalization_evidence": True},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_registration()
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output),
                      "source_identity_sha256": payload["source_identity_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
