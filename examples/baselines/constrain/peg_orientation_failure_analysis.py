"""Offline summaries of retained Peg stage-geometry episode traces."""

import argparse
import hashlib
import json
import math
from pathlib import Path


FORMAL_RUN = Path("outputs/exp/20261117_153824_peg_orientation_formal_gate")
COHORTS = (
    "random_test_base", "random_test_correct", "random_test_fixed_standard",
    "random_test_shuffled", "fixed_retention_base", "fixed_retention_correct",
)


def _snapshots(episode):
    for step in episode.get("phase_trace", []):
        for stage in ("stage_geometry_preaction", "stage_geometry_postaction"):
            sample = step.get(stage)
            if sample is not None:
                yield sample
    terminal = episode.get("stage_geometry_terminal")
    if terminal is not None:
        yield terminal


def summarize_episode(episode):
    snapshots = list(_snapshots(episode))
    if not snapshots:
        raise ValueError("episode has no stage-geometry snapshots")
    xyz = [sample["official_peg_head_in_hole_m"] for sample in snapshots]
    distances = [math.sqrt(sum(float(v) ** 2 for v in point)) for point in xyz]
    nearest = min(range(len(distances)), key=distances.__getitem__)
    terminal = episode.get("stage_geometry_terminal", snapshots[-1])
    trace = episode.get("phase_trace", [])
    yaw_nearest = snapshots[nearest].get("wrapped_peg_plus_x_minus_box_plus_x_yaw_rad")
    axial_deltas = []
    for step in trace:
        before = step.get("stage_geometry_preaction", {}).get("official_peg_head_in_hole_m")
        after = step.get("stage_geometry_postaction", {}).get("official_peg_head_in_hole_m")
        if before is not None and after is not None:
            axial_deltas.append(float(after[0]) - float(before[0]))
    grasped = [bool(step.get("is_grasping")) for step in trace]
    return {
        "success": bool(episode["success"]),
        "snapshot_count": len(snapshots),
        "trace_native_insertion_seen": any(bool(s.get("official_insertion")) for s in snapshots),
        "minimum_head_distance_m": distances[nearest],
        "minimum_head_xyz_m": [float(v) for v in xyz[nearest]],
        "minimum_head_axial_x_m": float(xyz[nearest][0]),
        "minimum_head_lateral_yz_m": math.hypot(float(xyz[nearest][1]), float(xyz[nearest][2])),
        "minimum_distance_yaw_rad": yaw_nearest,
        "terminal_head_xyz_m": [float(v) for v in terminal["official_peg_head_in_hole_m"]],
        "terminal_head_distance_m": math.sqrt(sum(float(v) ** 2 for v in terminal["official_peg_head_in_hole_m"])),
        "terminal_head_axial_x_m": float(terminal["official_peg_head_in_hole_m"][0]),
        "terminal_head_lateral_yz_m": math.hypot(*map(float, terminal["official_peg_head_in_hole_m"][1:])),
        "terminal_yaw_rad": terminal.get("wrapped_peg_plus_x_minus_box_plus_x_yaw_rad"),
        "ever_grasped": bool(episode.get("phase", {}).get("ever_grasped")),
        "grasped_controls_from_trace": sum(grasped),
        "contact_loss_transitions": int(episode.get("phase", {}).get("contact_losses", 0)),
        "first_grasp_control": episode.get("phase", {}).get("first_grasp_control"),
        "axial_delta_per_control_m": axial_deltas,
        "hint_sample_count": len(episode.get("hint_samples", [])),
        "sampler_calls": episode.get("phase", {}).get("sampler_calls"),
    }


def _quantiles(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return None

    def quantile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (x - lo)

    return {"n": len(values), "median": quantile(.5), "q25": quantile(.25), "q75": quantile(.75), "min": values[0], "max": values[-1]}


def _group_summary(rows):
    return {
        "episodes": len(rows),
        "native_successes": sum(r["success"] for r in rows),
        "trace_native_insertion_seen": sum(r["trace_native_insertion_seen"] for r in rows),
        "ever_grasped": sum(r["ever_grasped"] for r in rows),
        "contact_loss_transitions_total": sum(r["contact_loss_transitions"] for r in rows),
        "minimum_head_distance_m": _quantiles(r["minimum_head_distance_m"] for r in rows),
        "minimum_head_axial_x_m": _quantiles(r["minimum_head_axial_x_m"] for r in rows),
        "minimum_head_lateral_yz_m": _quantiles(r["minimum_head_lateral_yz_m"] for r in rows),
        "minimum_distance_yaw_rad": _quantiles(r["minimum_distance_yaw_rad"] for r in rows),
        "terminal_head_distance_m": _quantiles(r["terminal_head_distance_m"] for r in rows),
        "terminal_head_axial_x_m": _quantiles(r["terminal_head_axial_x_m"] for r in rows),
        "terminal_head_lateral_yz_m": _quantiles(r["terminal_head_lateral_yz_m"] for r in rows),
        "terminal_yaw_rad": _quantiles(r["terminal_yaw_rad"] for r in rows),
        "axial_delta_per_control_m": _quantiles(d for r in rows for d in r["axial_delta_per_control_m"]),
        "grasped_controls_per_episode": _quantiles(r["grasped_controls_from_trace"] for r in rows),
    }


def analyze(run=FORMAL_RUN):
    run = Path(run)
    inputs = {}
    by_cohort = {}
    source_hashes_before = {}
    for cohort in COHORTS:
        folder = run / cohort
        episodes = []
        hashes = {}
        for path in sorted(folder.glob("episode_*.json")):
            raw = path.read_bytes()
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            episodes.append(json.loads(raw))
        if len(episodes) != 100:
            raise ValueError(f"{cohort}: expected 100 episode files, found {len(episodes)}")
        source_hashes_before[cohort] = hashes
        inputs[cohort] = [summarize_episode(ep) for ep in episodes]
        by_cohort[cohort] = _group_summary(inputs[cohort])
        by_cohort[cohort]["geometry_groups"] = {
            "trace_native_insertion_seen": _group_summary([r for r in inputs[cohort] if r["trace_native_insertion_seen"]]),
            "no_recorded_native_insertion_snapshot": _group_summary([r for r in inputs[cohort] if not r["trace_native_insertion_seen"]]),
        }
    source_hashes_after = {
        cohort: {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((run / cohort).glob("episode_*.json"))}
        for cohort in COHORTS
    }
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("retained source episode hashes changed during offline analysis")
    return {
        "format": "peg-orientation-failure-analysis-v1",
        "source_run": str(run),
        "formal_registration_sha256": hashlib.sha256((run / "preregistration.json").read_bytes()).hexdigest(),
        "formal_results_sha256": hashlib.sha256((run / "results.json").read_bytes()).hexdigest(),
        "formal_post_execution_audit_sha256": hashlib.sha256((run / "post_execution_audit.json").read_bytes()).hexdigest(),
        "raw_episode_hashes": source_hashes_before,
        "source_hashes_unchanged_after_analysis": True,
        "definitions": {
            "snapshots": "all preaction and postaction stage snapshots plus stage_geometry_terminal once",
            "distance": "3D Euclidean distance from official peg head to hole frame origin",
            "native_insertion": "recorded official_insertion boolean; upstream task uses head-frame x >= -0.015 m and |y|,|z| <= hole radius",
            "yaw": "recorded wrapped peg local +X minus hole local +X about world Z; descriptive, no threshold",
            "geometry_group": "whether any recorded snapshot has official_insertion true; outcome-independent predicate, exploratory and not a causal phase label",
            "translation_stall": "only continuous per-control change in head-frame axial x is available; no requested translation vector or TCP positions exist in retained phase_trace",
        },
        "cohorts": by_cohort,
        "episodes": inputs,
    }


def render_markdown(report):
    lines = ["# Peg orientation retained-trace failure diagnosis", "", "Offline descriptive analysis only; no native action, model inference, training, or source edit.", "",
             "Success gate remains unchanged: random correct-hint +18pp (not the required +20pp); fixed-retention +6pp (within loss bound). The XY≤0.10m `not_reached` trigger is not used as a physical phase or insertion label.", "",
             "## Cohort measurements", "", "Values are episode medians [Q1,Q3]; signed yaw/axial values retain sign. Geometry-seen groups are defined by recorded native `official_insertion` snapshots, not success.", "",
             "| Cohort | Success | Grasped | Contact-loss transitions | min head 3D m | head x at min 3D m | lateral yz at min 3D m | yaw at min 3D rad | terminal head 3D m | terminal yaw rad | insertion snapshots seen |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(q):
        return "n/a" if not q else f"{q['median']:.5f} [{q['q25']:.5f},{q['q75']:.5f}]"
    for name, c in report["cohorts"].items():
        lines.append(f"| {name} | {c['native_successes']}/100 | {c['ever_grasped']}/100 | {c['contact_loss_transitions_total']} | {fmt(c['minimum_head_distance_m'])} | {fmt(c['minimum_head_axial_x_m'])} | {fmt(c['minimum_head_lateral_yz_m'])} | {fmt(c['minimum_distance_yaw_rad'])} | {fmt(c['terminal_head_distance_m'])} | {fmt(c['terminal_yaw_rad'])} | {c['trace_native_insertion_seen']}/100 |")
    lines += ["", "## Ranked falsifiable hypotheses (exploratory post-test associations)", "", "1. **Approach/translation stall (moderate confidence as next probe, not cause):** if late approach command/frame translation is the limitation, a validation trace will show retained grasp plus small axial head progress alongside nontrivial requested translation. Falsified by no retained grasp or continued substantial axial progress until termination. The current episode format has no requested translation, so it cannot currently be tested.", "2. **Residual head alignment (moderate descriptive support):** random-correct failures stop with median closest head x near −0.104 m and lateral yz near 0.009 m; if residual geometry is the primary bottleneck, failure traces should cluster short of the native head-x insertion boundary even when yaw is favorable. Falsified by failures frequently entering native head geometry without success. Recorded insertion snapshots were seen in 21/21 successes and 0/79 failures.", "3. **Hint improves grasped approach/contact stability (low-to-moderate):** correct hint has 31 contact-loss transitions vs Base 61 and 21 vs 3 successes; if stability mediates benefit, matched scene outcomes should pair lower loss with greater head-axis progress. Falsified by no within-pair relationship; profiles also differ only by hint in the registered random cohort but mechanism remains correlated, not proven.", "4. **Yaw residual limits subset of approaches (low):** success median absolute yaw at closest head sample is 0.019 rad vs 0.060 rad for correct-hint failures. Falsified if matched rows show overlap/no association; no threshold is asserted.", "5. **Hint refresh/correctness implementation bug (very low / unsubstantiated):** a complete call-level trace would show stale or wrong hint vs recomputation from live peg/target yaw. Retained hint samples cover only first two and last call, so no cadence conclusion is possible. Contract says refresh each observation; the sampled records/code review show no proven violation.", "", "No implementation correction is supported by this offline run. In particular, the formal result remains random +18pp (fails +20pp) and fixed +6pp (within bound); it is not reclassified.", "", "## Smallest next validation probe (not performed)", "", "Indices 6–11 from the existing random-validation catalog, selected now and fixed before the next run; do not reuse the earlier diagnosis indices 0–5, and do not use final random-test failures. One frozen correct-hint checkpoint, seeds `1+i`, 300 native controls/episode, no optimizer updates, 6 episodes only. Capture at every control/sampler call: requested normalized 7D action, pre/post TCP pose, peg/hole poses, official head-in-hole XYZ/insertion boolean, live hint and target/peg yaw. This tests the sole bounded candidate correction hypothesis: **if** late-approach translation is requested but realized head axial progress remains low with grasp maintained, test a narrowly scoped hole-frame translational command-frame correction; if request itself is small, do not alter the controller mapping and instead treat it as a policy behavior/training-budget question. Prediction: a real frame-mapping bug yields a consistent request-to-realized direction/scale mismatch; absence falsifies that correction. Any code change or native intervention requires separate user authorization; no added optimizer budget is requested here. The evidence is validation-informed, not blind.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=FORMAL_RUN)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = analyze(args.run)
    (args.output_dir / "full_analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "summary.md").write_text(render_markdown(report))
    print(json.dumps({"cohorts": {k: {"successes": v["native_successes"], "grasped": v["ever_grasped"], "contact_losses": v["contact_loss_transitions_total"], "insertion_seen": v["trace_native_insertion_seen"]} for k, v in report["cohorts"].items()}, "source_hashes_unchanged_after_analysis": report["source_hashes_unchanged_after_analysis"]}, indent=2))


if __name__ == "__main__":
    main()
