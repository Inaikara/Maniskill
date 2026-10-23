import hashlib
import json
import math
from pathlib import Path

import numpy as np

import pytest
import torch

from examples.baselines.constrain.diagnose_pickcube_contact import (
    DEFAULT_SAMPLE_TARGET,
    audit_retained,
    classify_contact_location,
    classify_contact_phase,
    compare_scene,
    diagnostic_lifted,
    diagnostic_near_goal,
    geometry_xy_clearance,
    link_group,
    load_expert_actions,
    main,
    observe_obstacle_contacts,
    replay_saved_actions,
    select_representative,
    summarize_contact_evidence,
)

EVAL_DIR = Path("outputs/exp/20261022_162002_pickcube_paired_eval")
SCENE_DIR = Path("outputs/exp/20261022_152157_pickcube_scenes")
BASE_CHECKPOINT = "b379573970712bcc17279ed237602a7be823acce9f7134ba125041ba5478389c"
CONSTRAIN_CHECKPOINT = "da94d795f9cc5622b9ad87c7c19178e3e05ec1e93837bfe673400a71208d0ef4"
SCENE_MANIFEST_SHA = "ea1ae25961ad4af673be865399d000717032452a5fb24fc7c77443f42558c346"
SCENE_HDF5_SHA = "0a9740c5882335ed02eebd58cb4b27f61f0bd9ff8e75eac292825d1a1c7ca6e7"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row(index, *, b_safe=False, c_safe=False, b_succ=True, c_succ=True,
         b_contact=True, c_contact=True):
    return {
        "scene_index": index,
        "base": {"safe_success": b_safe, "native_success": b_succ,
                 "obstacle_contact": b_contact, "xy_forbidden_crossing": False},
        "constrain": {"safe_success": c_safe, "native_success": c_succ,
                      "obstacle_contact": c_contact, "xy_forbidden_crossing": False},
    }


def _selection_fixture():
    rows = [_row(index) for index in range(40)]
    rows[1] = _row(1, b_safe=True, b_contact=False, c_contact=False)
    rows[7] = _row(7, b_safe=True, c_safe=True, b_contact=False, c_contact=False)
    for index in (2, 4, 6, 8, 10, 12, 14):
        rows[index] = _row(index, b_succ=False, c_succ=False)
    for index in (3, 5, 9, 11, 13, 15, 17):
        rows[index] = _row(index, b_succ=False, c_succ=True, c_safe=True, c_contact=False)
    for index in (16, 18, 20, 22, 24):
        rows[index] = _row(index, c_succ=False)
    for index in (19, 21, 23, 25, 27, 29, 31):
        rows[index] = _row(index, c_safe=True, c_contact=False)
    return rows


def _clone(value):
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    return value.clone()


def test_audit_reports_full_counts_missing_contact_detail_and_does_not_mutate():
    episode = EVAL_DIR / "base_dp" / "episode_000.json"
    before = _sha(episode)
    listing = sorted(path.name for path in EVAL_DIR.iterdir())
    audit = audit_retained(EVAL_DIR, SCENE_DIR)
    full = audit.get("full_set", {})
    assert full.get("base_dp", {}).get("obstacle_contact_count") == 97
    assert full.get("constrain_dp", {}).get("obstacle_contact_count") == 53
    assert full.get("base_dp", {}).get("xy_crossing_count") == 0
    assert full.get("constrain_dp", {}).get("xy_crossing_count") == 0
    assert full.get("base_dp", {}).get("task_success_count") == 82
    assert full.get("constrain_dp", {}).get("task_success_count") == 83
    assert full.get("base_dp", {}).get("safe_success_count") == 2
    assert full.get("constrain_dp", {}).get("safe_success_count") == 38
    assert full.get("base_dp", {}).get("corrected_control_step_limit_count") == 18
    assert full.get("constrain_dp", {}).get("corrected_control_step_limit_count") == 17
    assert full.get("paired_task_success") == {
        "both": 71, "base_only": 11, "constrain_only": 12, "neither": 6}
    assert full.get("paired_safe_success") == {
        "both": 1, "base_only": 1, "constrain_only": 37, "neither": 61}
    identities = audit.get("identities", {})
    assert identities.get("checkpoint_sha256", {}).get("base_dp") == BASE_CHECKPOINT
    assert identities.get("checkpoint_sha256", {}).get("constrain_dp") == CONSTRAIN_CHECKPOINT
    assert identities.get("scene_manifest_file_sha256") == SCENE_MANIFEST_SHA
    assert identities.get("scene_hdf5_sha256") == SCENE_HDF5_SHA
    detail = audit.get("retained_contact_detail", {})
    assert detail.get("has_contact_step") is False
    assert detail.get("has_contact_entity") is False
    assert detail.get("has_contact_position") is False
    assert detail.get("has_contact_phase") is False
    assert audit.get("retained_records_sufficient_for_mechanism") is False
    assert audit.get("outcome_phase_is_contact_phase") is False
    assert audit.get("zero_xy_crossing_is_robot_safety_proof") is False
    assert _sha(episode) == before
    assert sorted(path.name for path in EVAL_DIR.iterdir()) == listing


def test_representative_selection_is_stratified_and_not_a_full_set_rate():
    assert DEFAULT_SAMPLE_TARGET == 28
    with pytest.raises(ValueError):
        select_representative(_selection_fixture(), target=19)
    selected = select_representative(_selection_fixture(), target=28)
    assert selected.get("indexes") == [
        1, 7, 2, 4, 6, 8, 10, 12, 14, 3, 5, 9, 11, 16, 18, 20, 22,
        19, 21, 23, 25, 27, 29, 0, 26, 28, 30, 32,
    ]
    labels = {item["scene_index"]: item["stratum"] for item in selected.get("assignments", [])}
    assert labels.get(1) == "base_safe_success"
    assert labels.get(2) == "both_task_failure"
    assert labels.get(3) == "base_only_task_failure"
    assert labels.get(16) == "constrain_only_task_failure"
    assert labels.get(19) == "base_contact_constrain_safe"
    assert labels.get(0) == "both_contact_task_success"
    assert selected.get("sample_rates_are_not_full_set_rates") is True
    assert selected.get("full_set_counts", {}).get("base_only_task_failure") == 7
    assert selected.get("sample_counts", {}).get("base_only_task_failure") == 4
    mandatory = select_representative(
        [_row(index, b_safe=True, b_contact=False, c_contact=False) for index in range(25)],
        target=24,
    )
    assert mandatory.get("indexes") == list(range(25))
    assert mandatory.get("exceeded_target_for_mandatory_strata") is True


def test_retained_pickcube_sample_covers_contact_failure_and_safe_success():
    audit = audit_retained(EVAL_DIR, SCENE_DIR)
    selected = select_representative(audit.get("rows", []), target=28)
    assert selected.get("indexes") == [
        23, 81, 5, 41, 58, 77, 91, 96, 2, 19, 25, 26, 9, 13, 14, 16,
        0, 1, 3, 7, 8, 10, 4, 6, 11, 12, 17, 20,
    ]
    assert selected.get("sample_rates_are_not_full_set_rates") is True
    assert selected.get("sample_count") == 28
    assert selected.get("full_set_count") == 100
    assert selected.get("sample_counts", {}).get("base_contact", selected.get("sample_contact_count")) != 97


def test_contact_phase_uses_diagnostic_thresholds_not_xy_crossing():
    assert diagnostic_lifted(0.05, 0.0) is True
    assert diagnostic_lifted(0.04, 0.0) is False
    assert diagnostic_near_goal([0.0, 0.0], [0.09, 0.0]) is True
    assert diagnostic_near_goal([0.0, 0.0], [0.10, 0.0]) is False
    assert classify_contact_phase(grasped=False, lifted=False, near_goal=False,
                                  native_success=False) == "grasp_approach"
    assert classify_contact_phase(grasped=False, lifted=False, near_goal=True,
                                  native_success=False) == "grasp_approach"
    assert classify_contact_phase(grasped=True, lifted=False, near_goal=False,
                                  native_success=False) == "lift"
    assert classify_contact_phase(grasped=True, lifted=True, near_goal=False,
                                  native_success=False) == "transport"
    assert classify_contact_phase(grasped=True, lifted=True, near_goal=True,
                                  native_success=False) == "goal_stabilization"
    assert classify_contact_phase(grasped=True, lifted=True, near_goal=False,
                                  native_success=True) == "goal_stabilization"


def test_geometry_clearance_and_contact_location_are_not_surface_proof():
    clearance = geometry_xy_clearance([0.05, 0.0], [0.0, 0.0], 0.03)
    assert clearance.get("clearance_m") == pytest.approx(0.02)
    assert clearance.get("center_distance_m") == pytest.approx(0.05)
    assert clearance.get("is_surface_clearance") is False
    assert clearance.get("method") == "geometry_approximation_xy_distance_minus_radius"
    assert classify_contact_location(
        [0.01, 0.0, 0.25], center_xy=[0.0, 0.0], radius=0.03, height=0.25) == "top_cap"
    assert classify_contact_location(
        [0.03, 0.0, 0.10], center_xy=[0.0, 0.0], radius=0.03, height=0.25) == "side"
    assert classify_contact_location(
        [0.10, 0.0, 0.10], center_xy=[0.0, 0.0], radius=0.03, height=0.25) == "other"
    assert link_group("cube") == "cube"
    assert link_group("panda_hand") == "gripper"
    assert link_group("panda_leftfinger") == "gripper"
    assert link_group("panda_link4") == "arm"


def test_observer_reports_real_contact_position_and_leaves_scoring_unchanged():
    import gymnasium as gym
    from examples.baselines.constrain.pickcube_obstacle import ENV_ID

    env = gym.make(ENV_ID, num_envs=1, obs_mode="state_dict", sim_backend="physx_cpu",
                   control_mode="pd_ee_delta_pose", max_episode_steps=500)
    try:
        env.reset(seed=3, options={"obstacle_center": [0.4, 0.4], "obstacle_enabled": True})
        _, _, _, _, info = env.step(torch.zeros(7))
        assert not bool(info["obstacle_collision"].item())
        assert observe_obstacle_contacts(env) == []

        env.reset(seed=3, options={"obstacle_center": [0.0, 0.0], "obstacle_enabled": True})
        base = env.unwrapped
        state = _clone(base.get_state_dict())
        state["actors"]["cube"][0, :3] = torch.tensor([0.045, 0.0, 0.02])
        base.set_state_dict(state)
        _, _, _, _, info = env.step(torch.zeros(7))
        assert bool(info["obstacle_collision"].item())
        before = {
            "collision": bool(info["obstacle_collision"].item()),
            "success": bool(info["success"].item()),
            "native": bool(info["native_success"].item()),
            "first": dict(base._first_obstacle_contact),
            "flag": bool(base._obstacle_collision[0].item()),
        }
        hits = observe_obstacle_contacts(env)
        assert hits
        assert any(hit["entity"] == before["first"]["other"] for hit in hits)
        primary = max(hits, key=lambda hit: hit["impulse"])
        assert len(primary["position"]) == 3
        assert all(math.isfinite(value) for value in primary["position"])
        assert primary["impulse"] > 1e-6
        assert primary["surface"]["is_surface_clearance"] is True
        assert primary["surface"]["method"] == "physx_contact_point_separation"
        labelled = geometry_xy_clearance(primary["position"][:2], [0.0, 0.0], 0.03)
        assert labelled["is_surface_clearance"] is False
        assert dict(base._first_obstacle_contact) == before["first"]
        assert bool(base._obstacle_collision[0].item()) is before["flag"]
        evaluated = base.evaluate()
        assert bool(evaluated["obstacle_collision"].item()) is before["collision"]
        assert bool(evaluated["success"].item()) is before["success"]
        assert bool(evaluated["native_success"].item()) is before["native"]
    finally:
        env.close()


def test_cleared_impulse_cache_still_records_link_contact_position():
    from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet

    episode_path = EVAL_DIR / "base_dp" / "episode_017.json"
    record = json.loads(episode_path.read_text())
    catalog = PickCubeSceneSet(SCENE_DIR)
    try:
        trace = replay_saved_actions(
            catalog, 17, record["actions_native_pd_ee_delta_pose"], wall_seconds=60)
    except RuntimeError as exc:
        pytest.fail(f"replay aborted instead of recording cleared-cache contact: {exc}")
    scoring = trace.get("scoring", {})
    assert scoring.get("native_success") is True
    assert scoring.get("obstacle_contact") is True
    assert scoring.get("xy_forbidden_crossing") is False
    assert scoring.get("safe_success") is False
    assert scoring.get("elapsed_steps") == 82
    contact = trace.get("first_contact") or {}
    assert contact.get("entity") == "panda_link7"
    assert contact.get("position_source") == "post_step_contact_cache_impulse_cleared"
    assert contact.get("impulse_source") == "env_in_step_max_impulse"
    assert contact.get("surface", {}).get("is_surface_clearance") is False
    position = np.asarray(contact.get("position"), dtype=float)
    center = np.asarray(catalog.load(17)["metadata"]["cylinder"]["center_xy"], dtype=float)
    assert position.shape == (3,)
    assert abs(np.linalg.norm(position[:2] - center) - 0.03) < 0.01


def test_saved_action_replay_matches_retained_scoring_and_records_contact():
    from examples.baselines.constrain.pickcube_scenes import PickCubeSceneSet

    episode_path = EVAL_DIR / "base_dp" / "episode_000.json"
    before = _sha(episode_path)
    record = json.loads(episode_path.read_text())
    catalog = PickCubeSceneSet(SCENE_DIR)
    prefix = replay_saved_actions(catalog, 0, record["actions_native_pd_ee_delta_pose"][:1],
                                  wall_seconds=60)
    assert prefix.get("first_contact") is None
    assert prefix.get("scoring", {}).get("obstacle_contact") is False
    trace = replay_saved_actions(catalog, 0, record["actions_native_pd_ee_delta_pose"],
                                 wall_seconds=60)
    scoring = trace.get("scoring", {})
    assert scoring.get("native_success") is True
    assert scoring.get("obstacle_contact") is True
    assert scoring.get("xy_forbidden_crossing") is False
    assert scoring.get("safe_success") is False
    assert scoring.get("elapsed_steps") == 88
    assert scoring.get("outcome_phase") == "task_success"
    contact = trace.get("first_contact") or {}
    assert 1 <= contact.get("control_step", 0) <= 88
    assert contact.get("entity")
    assert len(contact.get("position", [])) == 3
    assert contact.get("phase") in {"grasp_approach", "lift", "transport", "goal_stabilization"}
    assert contact.get("surface", {}).get("is_surface_clearance") is True
    relation = trace.get("relation", {})
    assert len(relation.get("cube_xyz", [])) == 3
    assert len(relation.get("tcp_xyz", [])) == 3
    assert relation.get("obstacle", {}).get("radius") == pytest.approx(0.03)
    assert relation.get("geometry_clearance", {}).get("is_surface_clearance") is False
    assert trace.get("zero_xy_crossing_is_robot_safety_proof") is False
    assert _sha(episode_path) == before
    expert = load_expert_actions(catalog, 0)
    assert getattr(expert, "shape", None) == (116, 7)


def test_compare_labels_expert_clearance_and_refuses_xy_crossing_as_safety():
    base = {
        "scoring": {"obstacle_contact": True, "xy_forbidden_crossing": False,
                    "native_success": True, "safe_success": False},
        "paths": {"cube_xy": [[0.0, 0.0], [1.0, 0.0]], "tcp_xy": [[0.0, 0.1], [1.0, 0.1]]},
        "first_contact": {"entity": "panda_hand", "phase": "transport"},
        "min_geometry_clearance_m": {
            "method": "geometry_approximation_xy_distance_minus_radius",
            "is_surface_clearance": False, "cube": 0.08, "tcp": 0.01},
    }
    constrain = {
        "scoring": {"obstacle_contact": False, "xy_forbidden_crossing": False,
                    "native_success": True, "safe_success": True},
        "paths": {"cube_xy": [[0.0, 1.0], [1.0, 1.0]], "tcp_xy": [[0.0, 1.1], [1.0, 1.1]]},
        "first_contact": None,
        "min_geometry_clearance_m": {
            "method": "geometry_approximation_xy_distance_minus_radius",
            "is_surface_clearance": False, "cube": 0.12, "tcp": 0.05},
    }
    expert = {
        "waypoint_xy": [0.0, 1.0], "cube_clearance_m": 0.118,
        "robot_sweep_clearance_m": 0.034, "first_obstacle_contact": None,
        "safe_success": True,
        "paths": {"cube_xy": [[0.0, 1.0], [1.0, 1.0]], "tcp_xy": [[0.0, 1.2], [1.0, 1.2]]},
    }
    compared = compare_scene(base, constrain, expert)
    assert compared.get("zero_xy_crossing_is_robot_safety_proof") is False
    assert compared.get("expert_clearance", {}).get("cube", {}).get("is_surface_clearance") is False
    assert compared.get("expert_clearance", {}).get("cube", {}).get("method") == (
        "retained_cylinder_center_to_cube_xy_path_distance")
    assert compared.get("expert_clearance", {}).get("cube", {}).get("clearance_m") == pytest.approx(0.118)
    assert compared.get("expert_clearance", {}).get("robot_sweep", {}).get("method") == (
        "retained_cylinder_center_to_collision_vertex_xy_distance")
    assert compared.get("expert_clearance", {}).get("robot_sweep", {}).get("is_surface_clearance") is False
    deviation = compared.get("path_deviation", {})
    assert deviation.get("base_to_expert_cube_xy_mean_nearest_m") == pytest.approx(1.0)
    assert deviation.get("constrain_to_expert_cube_xy_mean_nearest_m") == pytest.approx(0.0)
    assert deviation.get("is_surface_clearance") is False


def test_findings_keep_sample_counts_separate_from_full_set():
    findings = summarize_contact_evidence([
        {"scene_index": 0, "base": {"first_contact": {
            "entity": "panda_hand", "phase": "transport", "location": "side"}},
         "constrain": {"first_contact": None}},
        {"scene_index": 1, "base": {"first_contact": {
            "entity": "cube", "phase": "grasp_approach", "location": "side"}},
         "constrain": {"first_contact": {"entity": "panda_link4", "phase": "lift",
                                         "location": "top_cap"}}},
    ], {"full_set": {"base_dp": {"obstacle_contact_count": 97, "xy_crossing_count": 0,
                                 "requested": 100}}})
    assert findings.get("full_set", {}).get("base_dp", {}).get("obstacle_contact_count") == 97
    assert findings.get("sample", {}).get("scene_count") == 2
    assert findings.get("sample", {}).get("base_dp", {}).get(
        "contact_entity_counts", {}).get("panda_hand") == 1
    assert findings.get("sample", {}).get("base_dp", {}).get(
        "contact_phase_counts", {}).get("transport") == 1
    assert findings.get("sample_counts_are_full_set_rates") is False
    assert findings.get("zero_xy_crossing_is_robot_safety_proof") is False
    assert findings.get("confirmed_mechanism") is None


def test_audit_only_entry_writes_index_and_leaves_sources_unchanged(tmp_path):
    source = EVAL_DIR / "paired_summary.json"
    digest = _sha(source)
    listing = sorted(path.name for path in EVAL_DIR.iterdir())
    main(["--eval-dir", str(EVAL_DIR), "--scene-dir", str(SCENE_DIR),
          "--output-dir", str(tmp_path / "diag"), "--audit-only"])
    index_path = tmp_path / "diag" / "index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text())
    assert index.get("retained_records_sufficient_for_mechanism") is False
    assert index.get("replay_performed") is False
    assert _sha(source) == digest
    assert sorted(path.name for path in EVAL_DIR.iterdir()) == listing
