import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.baselines.constrain.diagnose_placesphere_failure import (
    DEFAULT_SAMPLE_TARGET,
    adjustment_suggestions,
    audit_retained,
    classify_contact_location,
    classify_contact_phase,
    classify_crossing,
    classify_terminal_mechanism,
    compare_scene,
    diagnostic_lifted,
    diagnostic_near_goal,
    main,
    measured_conclusions,
    read_obstacle_contact,
    replay_saved_actions,
    render_findings,
    render_measured_report,
    select_representative,
    summarize_failure_evidence,
)

EVAL_DIR = Path("outputs/exp/PlaceSphere/BaseDP/evaluation")
SCENE_DIR = Path("outputs/exp/PlaceSphere/BaseDP/scenes")
BASE_CHECKPOINT = "07f93942cfcfd7d28120b0b210fd7c323098140f670bbfce4c9a83ff76b9fb30"
CONSTRAIN_CHECKPOINT = "0d20daee00ed90863ca4e68d1a7c1093774e4b7c04d20c4d470c3470dd715284"
SCENE_MANIFEST_SHA = "1606a307fb117f82ecf3e798d170b394a099b8c4c653eee96c9ee2d6b2649be3"
SCENE_HDF5_SHA = "ed4e09429a29884cbcc2d77cd5cd1d8ec01c49d74a00a64f02dc7b6929937ece"
FROZEN_CONFIG_SHA = "174625a2cd49363ef16059c4d9acf9d6e5ee41bb4236bcfd58e422d535633f98"
PAIRED_SUMMARY_SHA = "bca66ea0efa1e5f8f3282510a56f8c912ab0dc3e5bb1beac8aa31f4f6ee4b5b0"
EXPECTED_INDEXES = [
    2, 6, 61, 78, 81, 99, 11, 19, 31, 35, 51, 52, 60, 65, 80, 96,
    8, 14, 23, 27, 30, 36, 39, 43, 0, 1, 3, 4,
]


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(index, *, contact=False, release=False, xy=False, safe=False):
    phase = "release_or_stability_failed" if release else "task_success"
    return {
        "scene_index": index,
        "base": {
            "obstacle_contact": contact,
            "xy_forbidden_crossing": xy,
            "safe_success": safe,
            "native_success": not release,
            "phase": phase,
        },
        "constrain": {
            "obstacle_contact": False,
            "xy_forbidden_crossing": False,
            "safe_success": safe,
            "native_success": True,
            "phase": "task_success",
        },
    }


def test_audit_reports_full_counts_and_missing_crossing_detail():
    episode = EVAL_DIR / "base_dp" / "episode_000.json"
    before = _sha(episode)
    listing = sorted(path.name for path in EVAL_DIR.iterdir())
    audit = audit_retained(EVAL_DIR, SCENE_DIR)
    full = audit.get("full_set", {})
    assert full.get("base_dp", {}).get("obstacle_contact_count") == 2
    assert full.get("constrain_dp", {}).get("obstacle_contact_count") == 6
    assert full.get("base_dp", {}).get("xy_crossing_count") == 25
    assert full.get("constrain_dp", {}).get("xy_crossing_count") == 6
    assert full.get("base_dp", {}).get("task_success_count") == 97
    assert full.get("constrain_dp", {}).get("task_success_count") == 87
    assert full.get("base_dp", {}).get("safe_success_count") == 73
    assert full.get("constrain_dp", {}).get("safe_success_count") == 86
    assert full.get("base_dp", {}).get("release_or_stability_failed_count") == 2
    assert full.get("constrain_dp", {}).get("release_or_stability_failed_count") == 12
    assert full.get("base_dp", {}).get("timeout_count") == 3
    assert full.get("constrain_dp", {}).get("timeout_count") == 13
    assert full.get("paired_task_success") == {
        "both": 85, "base_only": 12, "constrain_only": 2, "neither": 1}
    assert full.get("paired_safe_success") == {
        "both": 68, "base_only": 5, "constrain_only": 18, "neither": 9}
    assert full.get("paired_xy_forbidden_crossing") == {
        "both": 3, "base_only": 22, "constrain_only": 3, "neither": 72}
    identities = audit.get("identities", {})
    assert identities.get("checkpoint_sha256", {}).get("base_dp") == BASE_CHECKPOINT
    assert identities.get("checkpoint_sha256", {}).get("constrain_dp") == CONSTRAIN_CHECKPOINT
    assert identities.get("scene_manifest_file_sha256") == SCENE_MANIFEST_SHA
    assert identities.get("scene_hdf5_sha256") == SCENE_HDF5_SHA
    assert identities.get("frozen_config_sha256") == FROZEN_CONFIG_SHA
    assert identities.get("paired_summary_sha256") == PAIRED_SUMMARY_SHA
    detail = audit.get("retained_detail", {})
    assert detail.get("has_crossing_height") is False
    assert detail.get("has_contact_step") is False
    assert detail.get("has_release_step") is False
    assert detail.get("has_terminal_pose") is False
    assert detail.get("has_terminal_velocity") is False
    assert audit.get("retained_records_sufficient_for_mechanism") is False
    assert audit.get("contactless_crossing_preclassified_as_over_top") is False
    assert _sha(episode) == before
    assert sorted(path.name for path in EVAL_DIR.iterdir()) == listing


def test_representative_selection_covers_required_strata_and_is_not_a_full_set_rate():
    assert DEFAULT_SAMPLE_TARGET == 28
    with pytest.raises(ValueError):
        select_representative([_row(0, safe=True)], target=19)
    rows = [_row(index, contact=True) for index in range(3)]
    rows += [_row(index, release=True) for index in range(3, 5)]
    rows += [_row(index, xy=True) for index in range(5, 15)]
    rows += [_row(index, safe=True) for index in range(15, 40)]
    selected = select_representative(rows, target=24)
    assert selected.get("indexes") == list(range(13)) + list(range(15, 26))
    assert selected.get("sample_counts", {}).get("xy_violation") == 8
    assert selected.get("full_set_counts", {}).get("xy_violation") == 10
    assert selected.get("sample_rates_are_not_full_set_rates") is True
    exceeded = select_representative([_row(index, contact=True) for index in range(25)], target=24)
    assert exceeded.get("indexes") == list(range(25))
    assert exceeded.get("exceeded_target_for_mandatory_strata") is True


def test_retained_placesphere_sample_covers_crossing_contact_release_and_safe_success():
    audit = audit_retained(EVAL_DIR, SCENE_DIR)
    selected = select_representative(audit.get("rows", []), target=28)
    assert selected.get("indexes") == EXPECTED_INDEXES
    labels = {item["scene_index"]: item["stratum"] for item in selected.get("assignments", [])}
    assert labels.get(2) == "contact"
    assert labels.get(11) == "release_or_stability_failure"
    assert labels.get(8) == "xy_violation"
    assert labels.get(0) == "both_safe_success"
    assert selected.get("sample_count") == 28
    assert selected.get("full_set_count") == 100
    assert selected.get("sample_counts", {}).get("xy_violation") == 8
    assert selected.get("full_set_counts", {}).get("xy_violation") == 16
    assert selected.get("sample_rates_are_not_full_set_rates") is True
    assert selected.get("compares_both_policies_on_same_scene") is True


def test_contactless_crossing_is_not_preclassified_as_over_top():
    overlap = classify_crossing(
        sphere_center_z=0.05, previous_sphere_center_z=0.05, sphere_radius=0.02,
        obstacle_height=0.08, start_xy=[-0.08, 0.0], end_xy=[0.08, 0.0],
        center_xy=[0.0, 0.0], obstacle_radius=0.03, forbidden_radius=0.055,
        contact_on_step=False)
    assert overlap.get("center_distance_m") == pytest.approx(0.0)
    assert overlap.get("sphere_bottom_z_m") == pytest.approx(0.03)
    assert overlap.get("bottom_minus_obstacle_top_m") == pytest.approx(-0.05)
    assert overlap.get("relation") == "xy_overlap_without_top_clearance"
    assert overlap.get("over_top") is False
    assert overlap.get("contactless_crossing_preclassified_as_over_top") is False

    above = classify_crossing(
        sphere_center_z=0.12, previous_sphere_center_z=0.12, sphere_radius=0.02,
        obstacle_height=0.08, start_xy=[-0.08, 0.0], end_xy=[0.08, 0.0],
        center_xy=[0.0, 0.0], obstacle_radius=0.03, forbidden_radius=0.055,
        contact_on_step=False)
    assert above.get("relation") == "geometric_above_top"
    assert above.get("over_top") is True
    assert above.get("contactless_crossing_preclassified_as_over_top") is False

    dipped = classify_crossing(
        sphere_center_z=0.05, previous_sphere_center_z=0.12, sphere_radius=0.02,
        obstacle_height=0.08, start_xy=[-0.08, 0.0], end_xy=[0.08, 0.0],
        center_xy=[0.0, 0.0], obstacle_radius=0.03, forbidden_radius=0.055,
        contact_on_step=False)
    assert dipped.get("over_top") is False
    assert dipped.get("relation") == "xy_overlap_without_top_clearance"

    margin = classify_crossing(
        sphere_center_z=0.04, previous_sphere_center_z=0.04, sphere_radius=0.02,
        obstacle_height=0.08, start_xy=[-0.08, 0.052], end_xy=[0.08, 0.052],
        center_xy=[0.0, 0.0], obstacle_radius=0.03, forbidden_radius=0.055,
        contact_on_step=False)
    assert margin.get("center_distance_m") == pytest.approx(0.052)
    assert margin.get("relation") == "forbidden_margin_only"
    assert margin.get("over_top") is False

    contacted = classify_crossing(
        sphere_center_z=0.12, previous_sphere_center_z=0.12, sphere_radius=0.02,
        obstacle_height=0.08, start_xy=[-0.08, 0.0], end_xy=[0.08, 0.0],
        center_xy=[0.0, 0.0], obstacle_radius=0.03, forbidden_radius=0.055,
        contact_on_step=True)
    assert contacted.get("relation") == "contact"
    assert contacted.get("over_top") is False


def test_terminal_mechanism_does_not_replace_native_success():
    success = classify_terminal_mechanism(
        native_success=True, ever_released=True, release_on_bin=True,
        terminal_on_bin=True, terminal_static=True, terminal_grasped=False,
        termination_reason="native_success", elapsed_steps=80, max_steps=200)
    assert success.get("mechanism") is None
    assert success.get("replaces_native_success") is False

    held = classify_terminal_mechanism(
        native_success=False, ever_released=False, release_on_bin=None,
        terminal_on_bin=False, terminal_static=False, terminal_grasped=True,
        termination_reason="timeout", elapsed_steps=200, max_steps=200)
    assert held.get("mechanism") == "incorrect_release"
    assert held.get("insufficient_time") is False

    off_bin = classify_terminal_mechanism(
        native_success=False, ever_released=True, release_on_bin=False,
        terminal_on_bin=False, terminal_static=False, terminal_grasped=False,
        termination_reason="timeout", elapsed_steps=200, max_steps=200)
    assert off_bin.get("mechanism") == "incorrect_release"
    assert off_bin.get("insufficient_time") is False

    left = classify_terminal_mechanism(
        native_success=False, ever_released=True, release_on_bin=True,
        terminal_on_bin=False, terminal_static=False, terminal_grasped=False,
        termination_reason="timeout", elapsed_steps=200, max_steps=200)
    assert left.get("mechanism") == "post_release_deviation"
    assert left.get("insufficient_time") is False

    settling = classify_terminal_mechanism(
        native_success=False, ever_released=True, release_on_bin=True,
        terminal_on_bin=True, terminal_static=False, terminal_grasped=False,
        termination_reason="timeout", elapsed_steps=200, max_steps=200)
    assert settling.get("mechanism") == "not_stabilized"
    assert settling.get("insufficient_time") is True
    assert settling.get("replaces_native_success") is False

    early = classify_terminal_mechanism(
        native_success=False, ever_released=True, release_on_bin=True,
        terminal_on_bin=True, terminal_static=False, terminal_grasped=False,
        termination_reason="environment_terminal", elapsed_steps=40, max_steps=200)
    assert early.get("mechanism") == "not_stabilized"
    assert early.get("insufficient_time") is False


def test_contact_phase_uses_diagnostic_thresholds_not_native_placement():
    assert diagnostic_lifted(0.05, 0.0) is True
    assert diagnostic_lifted(0.04, 0.0) is False
    assert diagnostic_near_goal([0.0, 0.0], [0.09, 0.0]) is True
    assert diagnostic_near_goal([0.0, 0.0], [0.10, 0.0]) is False
    assert classify_contact_phase(
        grasped=False, lifted=False, near_goal=False, released=False,
        native_success=False) == "grasp_approach"
    assert classify_contact_phase(
        grasped=True, lifted=False, near_goal=False, released=False,
        native_success=False) == "lift"
    assert classify_contact_phase(
        grasped=True, lifted=True, near_goal=False, released=False,
        native_success=False) == "transport"
    assert classify_contact_phase(
        grasped=True, lifted=True, near_goal=True, released=False,
        native_success=False) == "release_or_stabilization"
    assert classify_contact_phase(
        grasped=False, lifted=True, near_goal=False, released=True,
        native_success=False) == "release_or_stabilization"
    assert classify_contact_location(
        [0.01, 0.0, 0.08], center_xy=[0.0, 0.0], radius=0.03, height=0.08) == "top_cap"
    assert classify_contact_location(
        [0.03, 0.0, 0.04], center_xy=[0.0, 0.0], radius=0.03, height=0.08) == "side"
    assert classify_contact_location(
        [0.10, 0.0, 0.04], center_xy=[0.0, 0.0], radius=0.03, height=0.08) == "other"


def test_reader_reports_sphere_contact_and_leaves_scoring_unchanged():
    import gymnasium as gym
    from examples.baselines.constrain.placesphere_obstacle import ENV_ID

    env = gym.make(
        ENV_ID, num_envs=1, obs_mode="state", sim_backend="physx_cpu",
        control_mode="pd_ee_delta_pose", max_episode_steps=200,
        obstacle_radius=0.03, obstacle_height=0.08, safety_margin=0.005)
    try:
        env.reset(seed=3, options={
            "obstacle_center": [0.4, 0.4], "obstacle_enabled": True,
            "sphere_xy": [0.2, 0.2], "container_xy": [0.1, 0.1]})
        _, _, _, _, info = env.step(torch.zeros(1, 7))
        assert not bool(info["obstacle_collision"].item())
        assert read_obstacle_contact(env) is None

        env.reset(seed=3, options={
            "obstacle_center": [0.0, 0.0], "obstacle_enabled": True,
            "sphere_xy": [0.2, 0.2], "container_xy": [0.1, 0.1]})
        base = env.unwrapped
        state = base.get_state_dict()
        state["actors"]["sphere"] = state["actors"]["sphere"].clone()
        state["actors"]["sphere"][0, :3] = torch.tensor([0.06, 0.0, 0.04])
        state["actors"]["sphere"][0, 7:10] = torch.tensor([-1.5, 0.0, 0.0])
        base.set_state_dict(state)
        _, _, _, _, info = env.step(torch.zeros(1, 7))
        assert bool(info["obstacle_collision"].item())
        before = {
            "collision": bool(info["obstacle_collision"].item()),
            "success": bool(info["success"].item()),
            "native": bool(info["native_success"].item()),
            "first": dict(base._first_obstacle_contact),
            "flag": bool(base._obstacle_collision[0].item()),
        }
        hit = read_obstacle_contact(env)
        assert hit.get("entity") == "sphere"
        assert len(hit.get("position", [])) == 3
        assert all(np.isfinite(hit["position"]))
        assert hit.get("surface", {}).get("is_surface_clearance") is False
        assert hit.get("position_source") == "post_step_contact_cache_impulse_cleared"
        assert dict(base._first_obstacle_contact) == before["first"]
        evaluated = base.evaluate()
        assert bool(evaluated["obstacle_collision"].item()) is before["collision"]
        assert bool(evaluated["success"].item()) is before["success"]
        assert bool(evaluated["native_success"].item()) is before["native"]
    finally:
        env.close()


def test_saved_action_replay_records_crossing_and_matches_retained_scoring():
    episode_path = EVAL_DIR / "base_dp" / "episode_008.json"
    before = _sha(episode_path)
    record = json.loads(episode_path.read_text())
    from examples.baselines.constrain.placesphere_scenes import PlaceSphereSceneSet

    catalog = PlaceSphereSceneSet(SCENE_DIR)
    trace = replay_saved_actions(
        catalog, 8, record["actions_native_pd_ee_delta_pose"], wall_seconds=60)
    scoring = trace.get("scoring", {})
    assert scoring.get("native_success") is True
    assert scoring.get("obstacle_contact") is False
    assert scoring.get("xy_forbidden_crossing") is True
    assert scoring.get("safe_success") is False
    assert scoring.get("elapsed_steps") == 82
    crossing = trace.get("crossing") or {}
    assert crossing.get("sphere_radius_m") == pytest.approx(0.02)
    assert crossing.get("obstacle_height_m") == pytest.approx(0.08, abs=1e-6)
    assert "sphere_center_z_m" in crossing
    assert "contact_on_step" in crossing
    assert crossing.get("contactless_crossing_preclassified_as_over_top") is False
    assert crossing.get("over_top") is (crossing.get("relation") == "geometric_above_top")
    terminal = trace.get("terminal") or {}
    assert len(terminal.get("relative_xyz", [])) == 3
    assert len(terminal.get("linear_velocity", [])) == 3
    assert terminal.get("mechanism", {}).get("replaces_native_success") is False
    assert trace.get("scoring_source", "").startswith("env.step info")
    assert _sha(episode_path) == before


def test_adjustment_suggestions_stay_hypotheses_without_evidence():
    suggestions = adjustment_suggestions([{
        "base": {
            "crossing": {"over_top": False, "relation": "forbidden_margin_only", "at_end": False,
                         "contactless_crossing_preclassified_as_over_top": False},
            "first_contact": None,
            "terminal": {"mechanism": {"mechanism": "incorrect_release", "insufficient_time": False}},
        },
        "constrain": {"crossing": None, "first_contact": None, "terminal": {"mechanism": {"mechanism": None}}},
        "expert": {"replay_matches_retained_safety": True},
    }])
    assert suggestions.get("height", {}).get("status") == "hypothesis"
    assert suggestions.get("end_clearance", {}).get("status") == "hypothesis"
    assert suggestions.get("shared_expert_terminal", {}).get("status") == "hypothesis"
    evidenced = adjustment_suggestions([{
        "base": {
            "crossing": {"over_top": True, "relation": "geometric_above_top", "at_end": False,
                         "bottom_minus_obstacle_top_m": 0.02},
            "first_contact": None,
            "terminal": {"mechanism": {"mechanism": None}},
        },
        "constrain": {
            "crossing": None,
            "first_contact": {"phase": "release_or_stabilization", "location": "side", "at_end": True},
            "terminal": {"mechanism": {"mechanism": "not_stabilized", "insufficient_time": True}},
        },
        "expert": {"replay_matches_retained_safety": True,
                   "replay": {"terminal": {"on_bin": True, "static": True, "grasped": False}}},
    }])
    assert evidenced.get("height", {}).get("status") == "evidenced"
    assert evidenced.get("end_clearance", {}).get("status") == "evidenced"
    assert evidenced.get("shared_expert_terminal", {}).get("status") == "evidenced"


def test_compare_and_summary_keep_sample_separate_from_full_set():
    compared = compare_scene(
        {"crossing": {"relation": "forbidden_margin_only", "over_top": False,
                      "contactless_crossing_preclassified_as_over_top": False},
         "first_contact": None,
         "terminal": {"mechanism": {"mechanism": None, "replaces_native_success": False}}},
        {"crossing": None,
         "first_contact": {"phase": "transport", "entity": "panda_hand"},
         "terminal": {"mechanism": {"mechanism": "not_stabilized", "insufficient_time": True,
                                    "replaces_native_success": False}}},
        {"sphere_clearance_m": 0.09, "robot_sweep_clearance_m": 0.04,
         "release_control_step": 129, "post_release_control_steps": 17})
    assert compared.get("contactless_crossing_preclassified_as_over_top") is False
    assert compared.get("expert_clearance", {}).get("sphere", {}).get("is_surface_clearance") is False
    assert compared.get("expert_clearance", {}).get("sphere", {}).get("clearance_m") == pytest.approx(0.09)
    assert compared.get("base_crossing_relation") == "forbidden_margin_only"
    assert compared.get("constrain_contact_phase") == "transport"
    findings = summarize_failure_evidence(
        [{"scene_index": 8, "base": {"crossing": {"relation": "forbidden_margin_only"}},
          "constrain": {"crossing": None, "first_contact": None,
                        "terminal": {"mechanism": {"mechanism": "not_stabilized"}}}}],
        {"full_set": {"base_dp": {"xy_crossing_count": 25, "requested": 100}}})
    assert findings.get("full_set", {}).get("base_dp", {}).get("xy_crossing_count") == 25
    assert findings.get("sample", {}).get("scene_count") == 1
    assert findings.get("sample_counts_are_full_set_rates") is False
    assert findings.get("contactless_crossing_preclassified_as_over_top") is False
    text = render_findings(findings)
    assert "Sample counts are not full-set rates." in text
    assert "not preclassified as over-top" in text
    assert "forbidden_margin_only" in text


def test_measured_report_keeps_margin_crossing_separate_from_over_top():
    scenes = [{
        "scene_index": 8,
        "base": {
            "crossing": {"relation": "geometric_above_top", "over_top": True, "contact_on_step": False,
                         "bottom_minus_obstacle_top_m": 0.05, "at_end": False, "phase": "transport"},
            "first_contact": None,
            "terminal": {"mechanism": {"mechanism": None}},
        },
        "constrain": {
            "crossing": {"relation": "forbidden_margin_only", "over_top": False, "contact_on_step": False,
                         "bottom_minus_obstacle_top_m": -0.08, "at_end": True,
                         "phase": "release_or_stabilization"},
            "first_contact": {"phase": "grasp_approach", "location": "top_cap"},
            "terminal": {"mechanism": {"mechanism": "incorrect_release", "insufficient_time": False},
                         "release_state": {"on_bin": False}},
        },
    }]
    text, conclusions = render_measured_report(
        scenes, {"full_set": {"base_dp": {"xy_crossing_count": 25, "requested": 100}}})
    assert conclusions.get("contactless_crossing_preclassified_as_over_top") is False
    assert conclusions.get("geometric_above_top_count") == 1
    assert conclusions.get("not_over_top_relations") == ["forbidden_margin_only"]
    assert conclusions.get("post_release_deviation", {}).get("status") == "hypothesis"
    assert conclusions.get("end_clearance", {}).get("status") == "evidenced"
    assert "not preclassified as over-top" in text
    assert "forbidden_margin_only" in text
    assert measured_conclusions(scenes).get("sample_counts_are_full_set_rates") is False


def test_audit_only_entry_writes_index_and_leaves_sources_unchanged(tmp_path):
    source = EVAL_DIR / "paired_summary.json"
    digest = _sha(source)
    listing = sorted(path.name for path in EVAL_DIR.iterdir())
    main(["--eval-dir", str(EVAL_DIR), "--scene-dir", str(SCENE_DIR),
          "--output-dir", str(tmp_path / "diag"), "--audit-only"])
    index = json.loads((tmp_path / "diag" / "index.json").read_text())
    assert index.get("retained_records_sufficient_for_mechanism") is False
    assert index.get("replay_performed") is False
    assert index.get("contactless_crossing_preclassified_as_over_top") is False
    assert _sha(source) == digest
    assert sorted(path.name for path in EVAL_DIR.iterdir()) == listing
