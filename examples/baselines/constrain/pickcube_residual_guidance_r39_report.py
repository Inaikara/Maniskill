"""Report-only integrity and paired analysis from immutable R39 episode rows."""

import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_residual_guidance_r39 import (
    BASE_CKPT, CONTROL_CKPT, EXP, R38, SCENES, STATS,
)

DATA_H5 = Path("demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.h5")
DATA_MANIFEST = DATA_H5.with_name("trajectory.json")
VALIDATION = DATA_H5.with_name("validation.json")


def _load_rows(root, policy):
    return [json.loads((Path(root) / policy / f"episode_{i:03d}.json").read_text())
            for i in range(30)]


def transitions(old_rows, new_rows, field, violation=False):
    old = [bool(row.get("result", {}).get(field, False)) for row in old_rows]
    new = [bool(row.get("result", {}).get(field, False)) for row in new_rows]
    result = {"both": sum(a and b for a, b in zip(old, new)),
              "neither": sum(not a and not b for a, b in zip(old, new)),
              "old_marginal": sum(old), "new_marginal": sum(new)}
    if violation:
        result.update(cleared_old_only=sum(a and not b for a, b in zip(old, new)),
                      introduced_new_only=sum(not a and b for a, b in zip(old, new)))
        cells = ("cleared_old_only", "introduced_new_only", "both", "neither")
        old_reconciled = result["cleared_old_only"] + result["both"]
        new_reconciled = result["introduced_new_only"] + result["both"]
    else:
        result.update(improved_new_only=sum(not a and b for a, b in zip(old, new)),
                      regressed_old_only=sum(a and not b for a, b in zip(old, new)))
        cells = ("improved_new_only", "regressed_old_only", "both", "neither")
        old_reconciled = result["regressed_old_only"] + result["both"]
        new_reconciled = result["improved_new_only"] + result["both"]
    if (any(type(result[key]) is not int for key in result)
            or sum(result[key] for key in cells) != len(old)
            or result["old_marginal"] != old_reconciled
            or result["new_marginal"] != new_reconciled):
        raise ValueError(f"paired transition marginal reconciliation failed for {field}")
    return result


def _distribution(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "p90": float(np.percentile(values, 90)), "max": max(values)}


def _policy_report(rows):
    evaluated = [row for row in rows if row.get("status") == "evaluated"]
    result_keys = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
    metrics = {key: sum(bool(row["result"][key]) for row in evaluated) for key in result_keys}
    metrics.update(requested_denominator=30, evaluated=len(evaluated), runtime_errors=30-len(evaluated))
    phases, links = {}, {}
    for row in evaluated:
        contact = row.get("first_contact") or {}
        phase = contact.get("phase", "unknown") if row["result"]["obstacle_contact"] else None
        if phase:
            phases[phase] = phases.get(phase, 0) + 1
            link = contact.get("link") or "unknown"
            links[link] = links.get(link, 0) + 1
    return {
        "metrics": metrics, "first_contact_phase_counts": phases,
        "first_contact_link_counts": links,
        "success_step": _distribution([row.get("first_native_success_step") for row in evaluated]),
        "wall_seconds": _distribution([row.get("wall_seconds") for row in rows]),
        "final_3d_goal_error_m": _distribution([row.get("final_goal_distance_3d") for row in evaluated]),
        "termination_counts": {name: sum(row["result"]["termination_reason"] == name for row in evaluated)
                               for name in sorted({row["result"]["termination_reason"] for row in evaluated})},
        "contact_entity_pair_positions": "unknown; native recorder exposes first-contact link/phase only",
    }


def assemble():
    paired_root = EXP / "paired"
    old_base = _load_rows(EXP / "r38_snapshot", "base_dp")
    old_control = _load_rows(EXP / "r38_snapshot", "constrain_dp")
    base = _load_rows(paired_root, "base_dp")
    guided = _load_rows(paired_root, "guided_control")
    expected_hashes = json.loads((EXP / "preregistration.json").read_text())["inputs"]
    snapshot_hash_checks = {
        "base_dp": all(sha256_file(EXP / "r38_snapshot/base_dp" / f"episode_{i:03d}.json") == expected_hashes["r38_base_episode_sha256"][i] for i in range(30)),
        "constrain_dp": all(sha256_file(EXP / "r38_snapshot/constrain_dp" / f"episode_{i:03d}.json") == expected_hashes["r38_control_episode_sha256"][i] for i in range(30)),
    }
    identity_fields = ("scene_id", "seed", "restored_state_sha256",
                       "restored_controller_state_sha256", "restored_observation_sha256")
    paired = all(all(a.get(key) == b.get(key) for key in identity_fields)
                 for a, b in zip(base, guided))
    exact_base = all(new.get("status") == "evaluated"
                     and all(new.get(key) == old.get(key) for key in (
                         *identity_fields, "first_grasp_step", "first_lift_step", "first_transport_step",
                         "first_native_success_step", "first_contact", "step_trace", "result"))
                     for new, old in zip(base, old_base))
    comparisons = {}
    fields = {"task_success": ("native_success", False), "safe_success": ("safe_success", False),
              "obstacle_contact": ("obstacle_contact", True),
              "xy_forbidden_crossing": ("xy_forbidden_crossing", True)}
    for label, left, right in (("r38_control_vs_r39_guided", old_control, guided),
                               ("r39_base_vs_guided", base, guided)):
        comparisons[label] = {name: transitions(left, right, field, violation)
                              for name, (field, violation) in fields.items()}
    if not all(snapshot_hash_checks.values()):
        raise ValueError("pinned R38 analysis snapshots differ from R39 registration")
    if any(row.get("guidance_kind") is not None or row.get("guidance_scale") is not None
           for row in base):
        raise ValueError("Base records must declare guidance as not applicable")
    if any(row.get("guidance_kind") != "learned_residual_guidance"
           or row.get("guidance_scale") != 1.5 for row in guided):
        raise ValueError("guided Control records do not carry the preregistered guidance identity")
    control = _policy_report(guided)
    prereg = json.loads((EXP / "preregistration.json").read_text())
    counts = control["metrics"]
    gate = {"task": counts["native_success"] >= 27,
            "safe": counts["safe_success"] >= 26,
            "contact": counts["obstacle_contact"] <= 3,
            "xy": counts["xy_forbidden_crossing"] <= 3,
            "complete": counts["evaluated"] == 30 and counts["runtime_errors"] == 0,
            "paired": paired, "base_exactly_matches_r38": exact_base}
    audit_files = {"frozen_h5": (DATA_H5, expected_hashes["frozen_h5_sha256"]),
                   "frozen_manifest": (DATA_MANIFEST, expected_hashes["frozen_manifest_sha256"]),
                   "validation": (VALIDATION, expected_hashes["validation_sha256"]),
                   "base_checkpoint": (BASE_CKPT, expected_hashes["base_checkpoint_sha256"]),
                   "control_checkpoint": (CONTROL_CKPT, expected_hashes["control_checkpoint_sha256"]),
                   "position_profile": (STATS, expected_hashes["normalizer_file_sha256"]),
                   "scene_manifest": (SCENES / "manifest.json", expected_hashes["scene_manifest_sha256"]),
                   "scene_h5": (SCENES / "scenes.h5", expected_hashes["scene_h5_sha256"])}
    integrity = {name: {"path": str(path), "expected_sha256": expected,
                        "actual_sha256": sha256_file(path), "matches": sha256_file(path) == expected}
                 for name, (path, expected) in audit_files.items()}
    report = {
        "status": "report_only_reassembled_from_retained_episode_rows",
        "native_evaluation_rerun": False,
        "registration_sha256": prereg["registration_sha256"],
        "observed_runtime": {"device": "cuda", "gpu": "NVIDIA GeForce RTX 5070",
                             "python": "3.11.16", "torch": "2.14.0+cu130",
                             "sim_backend": "physx_cpu", "max_control_steps": 500,
                             "episode_wall_seconds": 600, "overall_wall_seconds": 36000},
        "runtime_source_identity_at_execution": prereg["source_identity_before_native"],
        "report_source_sha256": sha256_file(__file__),
        "policy_reports": {"base_dp": _policy_report(base), "guided_control": _policy_report(guided),
                           "r38_base_reference": _policy_report(old_base),
                           "r38_control_reference": _policy_report(old_control)},
        "policy_guidance": {
            "base_dp": {"guidance_kind": "not_applicable", "guidance_scale": None},
            "guided_control": {"guidance_kind": "learned_residual_guidance", "guidance_scale": 1.5,
                               "formula": "eps_base + 1.5 * (eps_control - eps_base)"},
        },
        "transitions": comparisons,
        "paired_scene_identity_matches": paired,
        "base_exactly_matches_r38_scene_by_scene": exact_base,
        "pinned_r38_snapshot_hashes_match": snapshot_hash_checks,
        "post_run_input_integrity": integrity,
        "contact_entity_pair_positions": "unknown in native contact recorder; first contact link and classified phase retained",
        "gate": {"criteria": gate, "passed": all(gate.values()), "registered_criteria": prereg["protocol"]["gate"]},
        "formal_performance_claim": False,
        "interpretation": "One exploratory scale-1.5 learned conditional residual extrapolation. No classifier-free unconditional branch, safety guarantee, or route-preservation claim.",
    }
    if not all(item["matches"] for item in integrity.values()):
        report["integrity_status"] = "failed"
    else:
        report["integrity_status"] = "all_pinned_inputs_unchanged"
    return report


def main():
    output = EXP / "paired"
    original = output / "analysis.json"
    snapshot = output / "native_analysis_before_report_enrichment.json"
    if original.exists() and not snapshot.exists():
        snapshot.write_bytes(original.read_bytes())
    report = assemble()
    (output / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output / "analysis_report_fix.json").write_text(json.dumps({
        "native_evaluation_rerun": False,
        "original_analysis_sha256": sha256_file(snapshot),
        "enriched_analysis_sha256": sha256_file(output / "analysis.json"),
        "report_source_sha256": sha256_file(__file__),
        "reason": "add exact marginal reconciliation, phase/timing summaries and post-run frozen-input audit from retained rows",
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
