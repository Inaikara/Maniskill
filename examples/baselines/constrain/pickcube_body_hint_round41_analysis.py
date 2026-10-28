"""Offline R41 report builder; reads saved rows only and never constructs an env."""

import hashlib
import json
from pathlib import Path

from examples.baselines.constrain import pickcube_body_hint_round41 as r41
from examples.baselines.constrain import pickcube_xy_gauge_round40 as r40
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot

EXP = r41.EXP
PAIRED = EXP / "paired"
R40_PAIRED = r41.R40_PRIMARY_EXP / "paired"
POLICIES = ("base_dp", "constrain_dp")
FIELDS = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_rows(root):
    return {policy: [json.loads((root / policy / f"episode_{i:03d}.json").read_text())
                     for i in range(30)] for policy in POLICIES}


def analyze():
    r41._install_profile()
    current_reg = json.loads((EXP / "preregistration.json").read_text())
    body = dict(current_reg); reg_hash = body.pop("registration_sha256", None)
    if reg_hash != _digest(body):
        raise ValueError("R41 registration checksum mismatch")
    native = json.loads((PAIRED / "preregistration.json").read_text())
    native_source = native["source_identity_before_native"]
    current_source = r41.source_identity()
    if native_source != current_source:
        raise ValueError("native runtime source differs from the pinned R41 source")
    prereg_path = EXP / "native_evaluation_preregistration_attempt02.json"
    prereg = json.loads(prereg_path.read_text())
    prereg_body = dict(prereg); prereg_sha = prereg_body.pop("evaluation_preregistration_sha256")
    if prereg_sha != _digest(prereg_body):
        raise ValueError("R41 native preregistration checksum mismatch")

    rows = _read_rows(PAIRED)
    primary = _read_rows(R40_PAIRED)
    for policy in POLICIES:
        if any(row.get("scene_index") != index or row.get("status") != "evaluated"
               for index, row in enumerate(rows[policy])):
            raise ValueError(f"saved R41 rows incomplete or invalid for {policy}")
        if any(row.get("checkpoint_sha256") != prereg[
                "base_checkpoint_sha256" if policy == "base_dp" else "control_checkpoint_sha256"]
               for row in rows[policy]):
            raise ValueError(f"saved R41 checkpoint identity mismatch for {policy}")

    report = r40._analyze_native_rows(rows, primary)
    base_trace_identical = all(
        new.get("step_trace") == old.get("step_trace") and new.get("result") == old.get("result")
        for new, old in zip(rows["base_dp"], primary["base_dp"]))
    report["primary_comparator"] = {
        "name": "R40 matched Base and Control on the identical 30 restored development scenes",
        "registration_sha256": prereg["r40_primary_registration_sha256"],
        "paired_tree_sha256": prereg["r40_primary_tree_sha256"],
        "metrics": {policy: summarize_pilot(rows[policy], 30) for policy in POLICIES},
        "r40_metrics": {policy: summarize_pilot(primary[policy], 30) for policy in POLICIES},
        "transitions": {policy: {field: engine._transitions(primary[policy], rows[policy], field)
                                  for field in FIELDS} for policy in POLICIES},
        "base_native_action_trace_and_outcomes_exact": base_trace_identical,
        "interpretation": "same frozen R40 Base; only Control hint representation and Control weights differ",
    }
    report["vs_r40"] = report["primary_comparator"]["transitions"]
    report.pop("vs_r38", None)
    report["base_native_action_trace_and_outcomes_exact"] = base_trace_identical
    same_state_criterion = report["gate"]["criteria"].pop(
        "same_r38_restored_states_and_conditions", None)
    if same_state_criterion is not None:
        report["gate"]["criteria"]["same_r40_restored_states_and_conditions"] = same_state_criterion
    report["comparison_interpretation"] = (
        "one new Control hint representation and Control retraining, with the exact R40 Base, global condition, "
        "frozen demonstrations, coordinate gauge augmentation, sampler, actions, and scenes held fixed")
    report["native_evaluation_preregistration_sha256"] = prereg_sha
    report["base_disabled_control_preflight"] = prereg["disabled_control_base_action_preflight"]
    report["native_evaluation_rerun"] = False
    report["source_identity_at_native"] = native_source
    report["analysis_source_identity"] = current_source
    report["source_changes_after_native"] = {}
    report["hint_profile"] = prereg["control_hint"]

    row_hashes = {f"{policy}/episode_{i:03d}.json": sha256_file(
        PAIRED / policy / f"episode_{i:03d}.json") for policy in POLICIES for i in range(30)}
    primary_tree = r40._file_tree_identity(R40_PAIRED)
    inputs = {"native_evaluation_preregistration_sha256": prereg_sha,
        "paired_native_preregistration_sha256": native["evaluation_registration_sha256"],
        "source_identity_at_native": native_source,
        "r40_primary_tree_sha256": primary_tree["files_sha256"],
        "r40_primary_files_sha256": primary_tree["files"],
        "r41_episode_rows_sha256": row_hashes,
        "native_evaluation_rerun": False}
    inputs["analysis_inputs_sha256"] = _digest(inputs)
    input_path = EXP / "analysis_report_inputs.json"
    if input_path.exists():
        if json.loads(input_path.read_text()) != inputs:
            raise ValueError("immutable R41 report inputs changed")
    else:
        input_path.write_text(json.dumps(inputs, indent=2, sort_keys=True) + "\n")
    report["analysis_input_snapshot_sha256"] = inputs["analysis_inputs_sha256"]
    report["analysis_source_identity_after_report"] = current_source
    report["analysis_sha256"] = _digest({k: v for k, v in report.items() if k != "analysis_sha256"})
    analysis_path = PAIRED / "analysis.json"
    previous_analysis = EXP / "analysis_pre_report_label_correction.json"
    if analysis_path.exists() and not previous_analysis.exists():
        previous_analysis.write_bytes(analysis_path.read_bytes())
    analysis_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    status = {"status": "complete", "analysis_repaired_from_retained_rows": True,
        "native_evaluation_rerun": False, "retained_rows": {policy: 30 for policy in POLICIES},
        "runtime_errors": {policy: 0 for policy in POLICIES},
        "gate_passed": report["gate"]["passed"],
        "base_native_action_trace_and_outcomes_exact": base_trace_identical,
        "analysis_input_snapshot_sha256": inputs["analysis_inputs_sha256"]}
    (PAIRED / "status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    correction = {"event": "offline_r41_report_rebuilt_from_complete_immutable_rows",
        "failure": "post-trial report adapter indexed missing summarize_pilot on the training module; all 60 native episode rows had already completed and been atomically saved",
        "retained_rows": {policy: 30 for policy in POLICIES}, "native_evaluation_rerun": False,
        "analysis_inputs_sha256": inputs["analysis_inputs_sha256"],
        "analysis_sha256": sha256_file(analysis_path), "status_sha256": sha256_file(PAIRED / "status.json")}
    correction_path = EXP / ("analysis_report_correction.json" if not
        (EXP / "analysis_report_correction.json").exists() else "analysis_report_correction_02.json")
    correction_path.write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, sort_keys=True))
