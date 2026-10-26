"""Write the final command, budget, identity, and outcome ledger for Round 23."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from examples.baselines.constrain.placesphere_matched_diagnosis import sha256

ROOT = Path(__file__).resolve().parents[3]
EXP = Path("outputs/exp/20261026_195000_placesphere_hold8_paired_eval_r23")
DATA = Path("demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23")
REGISTRATION = Path("outputs/exp/20261026_183000_placesphere_hold8_registration_r23.json")
TRAIN_PREREG = Path("outputs/exp/20261026_195500_placesphere_hold8_training_prereg_r23b.json")
DIAG = Path("outputs/exp/20261026_210000_placesphere_matched_diagnosis_r23d")
PROBE = Path("outputs/exp/20261026_181000_placesphere_matched_hold_probe_r23f")
BASE_RUN = Path("outputs/exp/20261026_195000_placesphere_hold8_base_dp_r23")
CONTROL_RUN = Path("outputs/exp/20261026_195000_placesphere_hold8_constrain_90k_r23")
EVAL = EXP / "paired_eval"


def _episode_rows(policy):
    return [json.loads((EVAL / policy / f"episode_{index:03d}.json").read_text())
            for index in range(30)]


def build_record():
    prereg = json.loads(TRAIN_PREREG.read_text())
    probe = json.loads((PROBE / "result.json").read_text())
    diagnosis = json.loads((DIAG / "diagnosis.json").read_text())
    base_summary = json.loads((BASE_RUN / "train_summary.json").read_text())
    control_summary = json.loads((CONTROL_RUN / "train_summary.json").read_text())
    paired = json.loads((EVAL / "paired_summary.json").read_text())
    full_collection = json.loads((DATA / "report.json").read_text())
    pilot_report = json.loads(Path("demos/PlaceSphere-transport-detour/20261026_183500_hold8_pilot_r23/report.json").read_text())
    base_rows, control_rows = _episode_rows("base_dp"), _episode_rows("constrain_dp")
    commands = [
        {
            "name": "hold8_full_collection_preregistration",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.placesphere_matched_collection",
                     "preregister", "--registration", str(REGISTRATION), "--release-hold-steps", "8"],
            "outer_timeout_seconds": 2400, "exit_code": 0,
        },
        {
            "name": "hold8_native_collector_pilot",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.placesphere_matched_collection",
                     "pilot", "--registration", str(REGISTRATION), "--dataset-dir",
                     "demos/PlaceSphere-transport-detour/20261026_183500_hold8_pilot_r23",
                     "--experiment-dir", "outputs/exp/20261026_183500_placesphere_hold8_pilot_r23",
                     "--pilot-layouts", "20", "--direct", "20", "--obstacle", "20",
                     "--pair-layouts", "20", "--outer-wall-s", "4800", "--max-attempts", "200"],
            "outer_timeout_seconds": 5400, "exit_code": 0,
            "elapsed_seconds": pilot_report["elapsed_seconds"],
            "accepted": pilot_report["accepted"], "attempted_layouts": pilot_report["attempted_layouts"],
        },
        {
            "name": "hold8_pilot_independent_validation",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.validate_placesphere_matched_collection",
                     "--dataset-dir", "demos/PlaceSphere-transport-detour/20261026_183500_hold8_pilot_r23",
                     "--experiment-dir", "outputs/exp/20261026_183500_placesphere_hold8_pilot_r23",
                     "--registration", str(REGISTRATION)],
            "outer_timeout_seconds": 1200, "exit_code": 0, "restore_passed": 40,
        },
        {
            "name": "hold8_train_layout_physics_probe",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.placesphere_matched_hold_probe",
                     "--output-dir", str(PROBE), "--episode-wall-s", "180", "--phase-wall-s", "2400"],
            "outer_timeout_seconds": 2700, "episode_wall_seconds": 180,
            "exit_code": 0, "elapsed_seconds": sum(row["elapsed_seconds"] for row in probe["rows"]),
        },
        {
            "name": "full_matched_hold8_collection",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.placesphere_matched_collection",
                     "collect", "--registration", str(REGISTRATION), "--dataset-dir", str(DATA),
                     "--experiment-dir", "outputs/exp/20261026_184500_placesphere_hold8_collection_r23",
                     "--direct", "1000", "--obstacle", "300", "--pair-layouts", "300",
                     "--outer-wall-s", "14400", "--max-attempts", "5000", "--release-hold-steps", "8"],
            "outer_timeout_seconds": 15000, "inner_wall_seconds": 14400,
            "attempt_budget": 5000, "exit_code": 0,
            "elapsed_seconds": full_collection["elapsed_seconds"],
        },
        {
            "name": "matched_hold8_independent_dataset_validation",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.validate_placesphere_matched_collection",
                     "--dataset-dir", str(DATA), "--experiment-dir",
                     "outputs/exp/20261026_184500_placesphere_hold8_collection_r23",
                     "--registration", str(REGISTRATION)],
            "outer_timeout_seconds": 1200, "exit_code": 0,
            "episode_denominator": 1300, "restore_passed": 1300,
        },
        {
            "name": "matched_hold8_base_training",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.train_placesphere_base_dp",
                     "--demo-path", str(DATA / "trajectory.h5"), "--manifest", str(DATA / "trajectory.json"),
                     "--matched-data-identity", str(DATA / "training_identity.json"),
                     "--run-dir", "outputs/exp/20261026_195000_placesphere_hold8_base_dp_r23",
                     "--checkpoint-dir", "outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23",
                     "--device", "cuda", "--seed", "1", "--total-iters", "30000",
                     "--batch-size", "1024", "--lr", "1e-4", "--warmup", "500", "--ema-power", "0.75"],
            "outer_timeout_seconds": 2400, "exit_code": 0,
            "elapsed_seconds": base_summary["elapsed_seconds"],
            "updates": base_summary["actual_updates"], "losses_finite": base_summary["all_losses_finite"],
            "checkpoint_sha256": base_summary["checkpoint_sha256"],
        },
        {
            "name": "matched_hold8_constrain_training",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.placesphere_constrain",
                     "--demo-path", str(DATA / "trajectory.h5"), "--manifest", str(DATA / "trajectory.json"),
                     "--matched-data-identity", str(DATA / "training_identity.json"),
                     "--base-checkpoint", "outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt",
                     "--run-dir", "outputs/exp/20261026_195000_placesphere_hold8_constrain_90k_r23",
                     "--checkpoint-dir", "outputs/ckpt/20261026_195000_placesphere_hold8_constrain_90k_r23",
                     "--device", "cuda", "--total-iters", "90000", "--batch-size", "1024", "--seed", "1",
                     "--lr", "1e-4", "--warmup", "500", "--ema-power", "0.75"],
            "outer_timeout_seconds": 6000, "exit_code": 0,
            "elapsed_seconds": control_summary["elapsed_seconds"],
            "updates": control_summary["steps"], "losses_finite": control_summary["all_losses_finite"],
            "base_unchanged": control_summary["base_unchanged"],
            "disabled_control_equivalence": control_summary["disabled_posttraining_equivalence"],
            "checkpoint_sha256": control_summary["checkpoint_sha256"],
        },
        {
            "name": "matched_hold8_paired_dev_evaluation",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.evaluate_placesphere_matched",
                     "--scene-dir", "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes",
                     "--base-checkpoint", "outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt",
                     "--constrain-checkpoint", "outputs/ckpt/20261026_195000_placesphere_hold8_constrain_90k_r23/final.pt",
                     "--output-dir", str(EVAL), "--primary-baseline",
                     "outputs/exp/20261026_153000_placesphere_matched_paired_eval/paired_eval",
                     "--secondary-baseline", "outputs/exp/20261025_085101_placesphere_obstacle300_90k/paired",
                     "--device", "cuda"],
            "outer_timeout_seconds": 8000, "per_episode_wall_seconds": 120,
            "policy_step_limit": 200, "requested_per_policy": 30,
            "exit_code": 0, "base_episode_wall_sum_seconds": sum(row["wall_seconds"] for row in base_rows),
            "constrain_episode_wall_sum_seconds": sum(row["wall_seconds"] for row in control_rows),
            "runtime_errors": sum(row.get("status") != "evaluated" for row in base_rows + control_rows),
            "gate_passed": paired["gate"]["passed"],
            "hold8_metrics": paired["summaries"]["constrain_dp"],
            "primary_transitions": paired["comparisons"]["primary_constrain_vs_300_90k"],
        },
        {
            "name": "corrected_current_matched_failure_diagnosis",
            "argv": ["python", "-u", "-m", "examples.baselines.constrain.placesphere_matched_diagnosis",
                     "--output-dir", str(DIAG), "--device", "cuda", "--episode-wall-s", "120",
                     "--phase-wall-s", "7200"],
            "outer_timeout_seconds": 7800, "episode_wall_seconds": 120,
            "selected_per_policy": 10, "reproduction_mismatches": sum(
                row["reproduction"]["mismatch"] for row in diagnosis["episodes"]),
            "runtime_errors": sum(row["status"] != "evaluated" for row in diagnosis["episodes"]),
        },
    ]
    return {
        "format": "placesphere_round23_execution_record_v1",
        "branch": "goal/transport-detour-optimization",
        "starting_head": "a402e6c6b01bd2622468e8f827fb13a076795d2a",
        "no_checkout_commit_or_push": True,
        "environment_activation": "source /home/xuwj/.miniconda3/etc/profile.d/conda.sh && conda activate maniskill",
        "training_preregistration": str(TRAIN_PREREG),
        "training_preregistration_sha256": sha256(TRAIN_PREREG),
        "registered_source_identity_sha256": prereg["source_identity_sha256"],
        "registered_source_hashes_current": all(
            sha256(ROOT / name) == digest for name, digest in prereg["source_hashes"].items()),
        "dataset_identity": {name: sha256(DATA / name) for name in
                             ("trajectory.h5", "trajectory.json", "training_identity.json")},
        "final_outputs": {"base_ema": "outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt",
                          "base_ema_sha256": base_summary["checkpoint_sha256"],
                          "constrain_ema": "outputs/ckpt/20261026_195000_placesphere_hold8_constrain_90k_r23/final.pt",
                          "constrain_ema_sha256": control_summary["checkpoint_sha256"],
                          "paired_summary": str(EVAL / "paired_summary.json"),
                          "paired_summary_sha256": sha256(EVAL / "paired_summary.json")},
        "commands": commands,
        "prior_incomplete_attempts": [
            {"path": "outputs/exp/20261026_171000_placesphere_matched_diagnosis_r23",
             "outcome": "20 policy episode records completed; expert reader rejected older nested metadata layout; superseded, not used"},
            {"path": "outputs/exp/20261026_172000_placesphere_matched_diagnosis_r23b",
             "outcome": "reproduced policy outcomes; expert action velocity was aligned to pre-action state; superseded by corrected T+1 join"},
            {"path": "outputs/exp/20261026_205500_placesphere_matched_diagnosis_r23c",
             "outcome": "correct policy action/release event instrumentation; expert velocity was still pre-action; superseded by corrected T+1 join"},
            {"paths": ["outputs/exp/20261026_174000_placesphere_hold_probe_r23",
                       "outputs/exp/20261026_174500_placesphere_hold_probe_r23b",
                       "outputs/exp/20261026_175000_placesphere_hold_probe_r23c",
                       "outputs/exp/20261026_175500_placesphere_hold_probe_r23d"],
             "outcome": "initial source-ID/metadata lookups or result-label serialization failed; retained separately; only complete corrected r23f entered selection"},
            {"path": "outputs/exp/20261026_183500_placesphere_hold8_pilot_r23",
             "outcome": "pilot collection complete; first validator check incorrectly required full registration target counts, second compared requested hold to actual early-success executed prefix; corrected validation passed before full collection"},
        ],
        "cleanup": {"timeouts_use_term_then_kill": True, "final_model_processes_expected": False,
                    "historical_artifacts_overwritten": False},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(EXP / "round23_execution.json"))
    args = parser.parse_args(argv)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = build_record()
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(path), "sha256": sha256(path),
                      "source_hashes_current": record["registered_source_hashes_current"]}, sort_keys=True))


if __name__ == "__main__":
    main()
