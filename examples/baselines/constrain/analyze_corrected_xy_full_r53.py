"""Bounded offline audit of the fixed R53 corrected-XY cohorts."""

import hashlib
import json
import math
from pathlib import Path

from examples.baselines.constrain.render_only_xy_eval import validate_registration


ROOT = Path("outputs/exp/20261102_135852_corrected_xy_full_r53")
CATALOG_ROOT = Path("outputs/exp/20261102_133924_initial_clear_contract_r52")
TASKS = ("pickcube", "placesphere")
INDICES = list(range(30))
RADIUS_MARGIN_M = 0.005


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def point_gap(point, center, radius):
    return math.hypot(point[0] - center[0], point[1] - center[1]) - radius


def chord_gap(start, end, center, radius):
    dx, dy = end[0] - start[0], end[1] - start[1]
    denom = dx * dx + dy * dy
    t = 0.0 if denom == 0 else max(0.0, min(1.0,
        ((center[0] - start[0]) * dx + (center[1] - start[1]) * dy) / denom))
    return point_gap([start[0] + t * dx, start[1] + t * dy], center, radius)


def validate_trace_geometry(trace, expected_center, expected_radius):
    """Reject trace geometry that disagrees with the catalog-derived contract."""
    for step in trace:
        center = step["obstacle_center_xy"]
        radius = float(step["forbidden_radius_m"])
        if (not all(abs(float(a) - float(b)) <= 1e-6
                    for a, b in zip(center, expected_center))
                or abs(radius - float(expected_radius)) > 1e-7):
            raise ValueError("trace geometry differs from frozen catalog geometry")


def analyze_task(task):
    registration_path = ROOT / f"{task}_registration.json"
    registration = validate_registration(registration_path)
    assert registration["indices"] == INDICES and registration["task"] == task
    assert registration["method"] == "ConstrainNet+XY constraint"
    profile = registration["constraint_profile"]
    assert profile["name"] == "tcp_xy_cylinder_radius_plus_5mm_v1"
    assert profile["working_clearance_profile"] == "tcp_xy_working_clearance_3mm_v2"
    assert profile["working_clearance_m"] == 0.003
    assert profile["xy_action_bounds"] == ([-1.0, 1.0] if task == "pickcube" else [-0.35, 0.35])

    rows_path = ROOT / f"{task}_rows/rows.json"
    summary_path = ROOT / f"{task}_rows/summary.json"
    rows = json.loads(rows_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert len(rows) == 30 and [row["scene_index"] for row in rows] == INDICES
    assert summary["denominator"] == 30 and summary["evaluated"] == 30
    assert summary["runtime_errors"] == 0
    assert all(row["status"] == "evaluated" for row in rows)
    assert sum(bool(row["native_success"]) for row in rows) == summary["native_success_count"]
    assert sum(bool(row["xy_zero"]) for row in rows) == summary["xy_zero_count"]

    phases = {}
    failures = []
    all_sample_gaps, all_chord_gaps = [], []
    intervention_count = native_clip_count = executed_count = 0
    interventions_by_index = {}
    physical_crossing_count = body_contact_count = 0
    per_episode = []
    for row in rows:
        trace = row["actual_tcp_trace"]
        assert trace, f"{task} index {row['scene_index']} has no captured controls"
        assert trace[0]["control_index"] == 1
        center = trace[0]["obstacle_center_xy"]
        radius = float(trace[0]["forbidden_radius_m"])
        assert abs(radius - 0.035) < 1e-7
        assert all(step["obstacle_center_xy"] == center and
                   abs(float(step["forbidden_radius_m"]) - radius) < 1e-9 for step in trace)
        assert all(step["control_index"] == i and step["env_step_executed"]
                   for i, step in enumerate(trace, 1))
        points = [trace[0]["tcp_xy_before"]] + [step["tcp_xy_after"] for step in trace]
        sample_gaps = [point_gap(point, center, radius) for point in points]
        chord_gaps = [chord_gap(a, b, center, radius) for a, b in zip(points, points[1:])]
        all_sample_gaps.extend(sample_gaps)
        all_chord_gaps.extend(chord_gaps)
        assert min(sample_gaps) >= -1e-12 and min(chord_gaps) >= -1e-12
        assert row["xy_validation"]["zero_xy_violations"] and row["xy_zero"]
        native_steps = [i for i, step in enumerate(trace, 1) if step.get("native_success")]
        assert bool(native_steps) == bool(row["native_success"])
        if native_steps:
            first_success = native_steps[0]
            assert first_success == row["native_success_step"] == len(trace)
            assert row["xy_validation"]["first_success_step"] == first_success
            assert row["xy_validation"]["first_success_step_valid"]
        else:
            phase = row["result"].get("phase", "unknown")
            failures.append({"scene_index": row["scene_index"], "phase": phase,
                             "elapsed_steps": row["result"].get("elapsed_steps"),
                             "termination_reason": row["result"].get("termination_reason")})
        phase = row["result"].get("phase", "unknown")
        phases[phase] = phases.get(phase, 0) + 1
        physical_crossing_count += int(bool(row["result"].get("xy_forbidden_crossing")))
        body_contact_count += int(bool(row["result"].get("obstacle_contact")))
        episode_interventions = 0
        episode_native_clips = 0
        for step in trace:
            executed_count += 1
            requested, filtered = step["requested_action"], step["filtered_action"]
            lower, upper = profile["xy_action_bounds"]
            bounded = [max(lower, min(upper, float(requested[i]))) for i in (0, 1)]
            native_clipped = any(abs(float(requested[i]) - bounded[i]) > 1e-12 for i in (0, 1))
            changed = any(abs(bounded[i] - float(filtered[i])) > 1e-12 for i in (0, 1))
            intervention_count += int(changed)
            episode_interventions += int(changed)
            native_clip_count += int(native_clipped)
            episode_native_clips += int(native_clipped)
        interventions_by_index[str(row["scene_index"])] = episode_interventions
        per_episode.append({"scene_index": row["scene_index"], "native_success": bool(row["native_success"]),
            "native_success_step": row.get("native_success_step"), "failure_phase": None if row["native_success"] else phase,
            "control_count": len(trace), "initial_sample_gap_m": sample_gaps[0],
            "minimum_sample_gap_m": min(sample_gaps), "minimum_chord_gap_m": min(chord_gaps),
            "interventions": episode_interventions, "native_xy_clipping_controls": episode_native_clips,
            "physical_object_crossing_latch": bool(row["result"].get("xy_forbidden_crossing")),
            "body_contact_diagnostic": bool(row["result"].get("obstacle_contact"))})

    catalog = CATALOG_ROOT / task
    loader_module = __import__(
        f"examples.baselines.constrain.{task}_transport_detour", fromlist=["TransportDetourSceneSet"])
    scene_catalog = loader_module.TransportDetourSceneSet(catalog)
    for row in rows:
        scene = scene_catalog.load(row["scene_index"])
        expected_center = scene["metadata"]["cylinder"]["center_xy"]
        expected_radius = float(scene["metadata"]["cylinder"]["radius"]) + RADIUS_MARGIN_M
        validate_trace_geometry(row["actual_tcp_trace"], expected_center, expected_radius)
        actual_center = row["xy_constraint"]["center_xy"]
        assert all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(actual_center, expected_center))
        assert abs(float(row["xy_constraint"]["forbidden_radius_m"]) - expected_radius) <= 1e-7
        assert row["xy_constraint"]["visual_only_obstacle"] and row["xy_constraint"]["collision_shapes"] == 0
        assert row["restored_state_sha256"] == row["scene_state_sha256"]
        if task == "placesphere" and row["native_success"]:
            assert row["result"]["terminal_on_bin"] and row["result"]["terminal_static"]
            assert not row["result"]["terminal_is_grasped"]
    preflight = json.loads((catalog / "preflight_r52.json").read_text())
    assert preflight["all_passed"] and preflight["checked"] == 30
    assert all(scene["signed_initial_gap_m"] > 0 and scene["layout_bounds_and_goal_height_match"]
               for scene in preflight["scenes"])
    assert all(row["scene_state_sha256"] == row["restored_state_sha256"] for row in rows)
    return {"denominator": 30, "evaluated": summary["evaluated"], "runtime_errors": summary["runtime_errors"],
        "native_successes": summary["native_success_count"], "native_success_gate_27": summary["native_success_count"] >= 27,
        "xy_zero_count": summary["xy_zero_count"], "xy_gate_30": summary["xy_zero_count"] == 30,
        "minimum_initial_gap_m": preflight["minimum_initial_profile_gap_m"],
        "minimum_actual_sample_gap_m": min(all_sample_gaps), "minimum_actual_chord_gap_m": min(all_chord_gaps),
        "actual_geometry_sample_and_chord_violations": 0,
        "trace_points_include_initial_and_all_executed_controls": True,
        "first_native_success_control_included": True,
        "failed_task_episodes_remain_in_denominator": True,
        "failure_phases": failures, "terminal_phase_counts": phases,
        "filter_interventions": intervention_count, "executed_controls": executed_count,
        "filter_intervention_frequency": intervention_count / executed_count if executed_count else 0.0,
        "native_xy_action_bound_clipping_controls": native_clip_count,
        "native_xy_action_bound_clipping_frequency": native_clip_count / executed_count if executed_count else 0.0,
        "interventions_by_scene_index": interventions_by_index,
        "historical_physical_object_crossing_latch_count": physical_crossing_count,
        "body_contact_diagnostic_count_non_gating": body_contact_count,
        "native_task_completion_semantics": ("cube held at task goal" if task == "pickcube"
            else "sphere on bin, static, and not grasped"),
        "catalog_manifest_sha256": sha256(catalog / "manifest.json"),
        "catalog_h5_sha256": sha256(catalog / "scenes.h5"),
        "registration_sha256": registration["registration_sha256"],
        "registration_file_sha256": sha256(registration_path),
        "rows_sha256": sha256(rows_path), "summary_sha256": sha256(summary_path),
        "source_entries": len(registration["sources"]), "input_entries": len(registration["inputs"]),
        "checkpoint_sha256": registration["inputs"]["checkpoint"]["sha256"],
        "training_h5_sha256": registration["inputs"]["data_h5"]["sha256"],
        "training_manifest_sha256": registration["inputs"]["data_manifest"]["sha256"],
        "episodes": per_episode}


def main():
    result = {"schema": "corrected_xy_full_r53_offline_analysis_v1",
        "scope": "two fixed, preregistered 30-scene ConstrainNet+XY engineering cohorts; no formal/generalization claim",
        "acceptance_profile": "tcp_xy_cylinder_radius_plus_5mm_v1; actual restored cylinder radius + 0.005 m",
        "working_profile": "tcp_xy_working_clearance_3mm_v2; 0.003 m",
        "tasks": {task: analyze_task(task) for task in TASKS}}
    result["analysis_source_sha256"] = sha256(Path(__file__))
    path = ROOT / "offline_analysis_r53.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({task: {key: result["tasks"][task][key] for key in (
        "native_successes", "runtime_errors", "xy_zero_count", "minimum_actual_sample_gap_m",
        "minimum_actual_chord_gap_m", "filter_interventions", "executed_controls", "failure_phases",
        "terminal_phase_counts", "historical_physical_object_crossing_latch_count")}
        for task in TASKS}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
