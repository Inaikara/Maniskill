"""Read-only provenance reconciliation for the 2026-11-16 Peg pilot."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "outputs/exp/20261116_164019_peg_orientation_constrain_exploratory"
CHECKPOINTS = ROOT / "outputs/ckpt/20261116_164019_peg_orientation_constrain_exploratory"
BASE = ROOT / "outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt"
CATALOG_ROOT = ROOT / "outputs/exp/20261111_172604_peg_orientation_adaptation"
DATA_ROOT = ROOT / "demos/PegInsertionSide-orientation-random-100"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    prereg = json.loads((RUN / "preregistration.json").read_text())
    results = json.loads((RUN / "results.json").read_text())
    cohort_paths = {
        "base_random": "base_validation_random_50",
        "base_fixed": "base_validation_fixed_50",
        "constrain_5000": "constrain_random_update_05000_50_retry_after_hint_fix",
        "constrain_10000": "constrain_random_update_10000_50",
        "constrain_15000": "constrain_random_update_15000_50",
        "constrain_20000": "constrain_random_update_20000_50",
        "constrain_25000": "constrain_random_update_25000_50",
        "constrain_30000": "constrain_random_update_30000_50",
        "constrain_selected_fixed": "constrain_fixed_selected_25000_50",
    }
    cohorts = {}
    for name, relative in cohort_paths.items():
        directory = RUN / relative
        assert len(list(directory.glob("episode_*.json"))) == 50
        records = [json.loads((directory / f"episode_{i:03d}.json").read_text()) for i in range(50)]
        assert [row["scene_index"] for row in records] == list(range(50))
        assert [row["policy_sampling_seed"] for row in records] == list(range(1, 51))
        split = "fixed" if "fixed" in relative else "random"
        catalog = json.loads((CATALOG_ROOT / f"validation_{split}_scenes/manifest.json").read_text())
        assert [row["seed"] for row in records] == [int(scene["seed"]) for scene in catalog["scenes"]]
        assert all(0 < row["executed_controls"] <= 300 for row in records)
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["episodes"] == len(records)
        assert summary["evaluation_settings"]["max_episode_steps"] == 300
        assert summary["evaluation_settings"]["env_time_limit"] == 300
        assert summary["successes"] == sum(bool(row["success"]) for row in records)
        assert summary["records"] == records
        native_successes = sum(bool(row["success"]) for row in records)
        cohorts[name] = {
            "path": relative,
            "episodes": len(records),
            "scene_indices": [0, 49],
            "catalog_seed_matches_manifest": True,
            "catalog_scene_identity": "episode rows retain catalog index and seed; current ordered manifest maps index to scene ID",
            "hint_and_profile_provenance": "control_enabled appears in summaries; live-hint mode/profile was not recorded in episode headers",
            "policy_sampling_seed_range": [1, 50],
            "cap": 300,
            "executed_controls_min_max": [min(r["executed_controls"] for r in records), max(r["executed_controls"] for r in records)],
            "native_successes": native_successes,
            "summary_outcomes_match_episode_records": True,
            "summary_checkpoint": summary.get("checkpoint"),
            "summary_checkpoint_hash_matches_current_file": (
                summary.get("checkpoint", {}).get("checkpoint_sha256")
                == sha256(ROOT / summary["checkpoint"]["checkpoint"])
            ),
            "summary_catalog": summary.get("scene_catalog"),
            "summary_evaluation_settings": summary.get("evaluation_settings"),
            "historical_checkpoint_source_and_hash_recorded_at_rollout": False,
        }

    current = {
        "base_checkpoint": {"path": str(BASE.relative_to(ROOT)), "sha256_now": sha256(BASE)},
        "adapter_checkpoints": {
            f"update_{step:05d}": {
                "path": str((CHECKPOINTS / f"update_{step:05d}.pt").relative_to(ROOT)),
                "sha256_now": sha256(CHECKPOINTS / f"update_{step:05d}.pt"),
            } for step in (5000, 10000, 15000, 20000, 25000, 30000)
        },
        "training_data": {},
        "catalog_manifests": {},
        "source_hashes_now": {
            relative: sha256(ROOT / relative) for relative in (
                "examples/baselines/constrain/peg_orientation_base_dp.py",
                "examples/baselines/constrain/peg_orientation_constrain.py",
                "examples/baselines/constrain/peg_orientation_data.py",
            )
        },
        "recovery_auditor_source_sha256_now": sha256(__file__),
        "frozen_base_embedding_state_sha16_in_training_and_checkpoint_metadata": results["training"]["frozen_base_state_sha16"],
    }
    for field, key in (("hdf5", "hdf5"), ("companion_json", "companion_json"), ("training_coverage", "training_coverage")):
        path = ROOT / prereg["training_data"][field]
        current["training_data"][key] = {"path": str(path.relative_to(ROOT)), "sha256_now": sha256(path),
                                          "preregistered_sha256": prereg["training_data"][f"{field}_sha256"]}
        assert current["training_data"][key]["sha256_now"] == current["training_data"][key]["preregistered_sha256"]
    for split in ("random", "fixed"):
        path = CATALOG_ROOT / f"validation_{split}_scenes/manifest.json"
        current["catalog_manifests"][split] = {"path": str(path.relative_to(ROOT)), "sha256_now": sha256(path)}

    assert cohorts["base_random"]["native_successes"] == results["base_validation"]["random_50"]
    assert cohorts["base_fixed"]["native_successes"] == results["base_validation"]["fixed_50"]
    assert cohorts["constrain_25000"]["native_successes"] == results["training"]["selected_successes"]
    assert cohorts["constrain_selected_fixed"]["native_successes"] == results["paired"]["fixed_validation"]["constrain_successes"]
    for name, count in (("constrain_5000", 5), ("constrain_10000", 8), ("constrain_15000", 13),
                        ("constrain_20000", 14), ("constrain_25000", 16), ("constrain_30000", 14)):
        assert cohorts[name]["native_successes"] == count

    pair_counts = {}
    for label, base_name, constrain_name, expected in (
        ("random", "base_random", "constrain_25000", results["paired"]["random_validation"]),
        ("fixed", "base_fixed", "constrain_selected_fixed", results["paired"]["fixed_validation"]),
    ):
        base_records = [json.loads((RUN / cohort_paths[base_name] / f"episode_{i:03d}.json").read_text()) for i in range(50)]
        constrain_records = [json.loads((RUN / cohort_paths[constrain_name] / f"episode_{i:03d}.json").read_text()) for i in range(50)]
        assert [(r["scene_index"], r["seed"]) for r in base_records] == [
            (r["scene_index"], r["seed"]) for r in constrain_records
        ]
        base_only = sum(b["success"] and not c["success"] for b, c in zip(base_records, constrain_records))
        constrain_only = sum(c["success"] and not b["success"] for b, c in zip(base_records, constrain_records))
        both = sum(b["success"] and c["success"] for b, c in zip(base_records, constrain_records))
        neither = 50 - base_only - constrain_only - both
        assert (base_only, constrain_only, both, neither) == tuple(
            expected[key] for key in ("base_only", "constrain_only", "both_success", "neither")
        )
        pair_counts[label] = {"base_only": base_only, "constrain_only": constrain_only,
                              "both_success": both, "neither": neither,
                              "matches_existing_results": True}

    return {
        "report_type": "read-only historical recovery verification; not a historical provenance reconstruction",
        "generated_from_current_files_only": True,
        "historical_preregistration_and_results_modified": False,
        "cohorts": cohorts,
        "paired_native_outcome_reconciliation": pair_counts,
        "identities_verifiable_now": current,
        "historical_identity_limits": [
            "Episode headers did not record checkpoint SHA-256, adapter source hash, evaluator source hash, or catalog-manifest hash at execution time.",
            "The present files' hashes and summary metadata establish current identity only; they do not authenticate which bytes the process loaded then.",
            "This limitation applies to every cohort. The 5,000-update summary metadata was added after its 50 rollouts, so that cohort's checkpoint/source association is specifically unverified and its contribution to checkpoint selection is provisional.",
            "Native outcomes and selected 25,000/random 16/50 and fixed 21/50 are independently present in surviving episode records and summaries; selection is verified among retained cohorts only, subject to the provisional 5,000 cohort limitation.",
            "Original preregistration has no pre-execution source hashes for any cohort. Current source hashes are recorded here without claiming historical authentication.",
        ],
        "native_outcome_counts_preserved": True,
    }


if __name__ == "__main__":
    report = audit()
    path = RUN / "recovery_verification_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(path.relative_to(ROOT)), "cohorts": len(report["cohorts"]),
                      "report_sha256": sha256(path)}, sort_keys=True))
