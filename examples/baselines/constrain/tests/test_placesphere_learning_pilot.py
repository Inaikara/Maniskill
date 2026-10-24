"""Seams for the PlaceSphere transport-detour learnability pilot."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain.placesphere_learning_pilot import (
    LAYOUT_TOLERANCE_M,
    NATIVE_SUCCESS,
    PAIR_ACCEPTANCE_CHECKS,
    batched_observation,
    build_expansion_preregistration,
    build_episode_index,
    build_preregistration,
    coerce_action_sequence,
    compare_ticket02,
    dataset_coverage,
    decide_gate,
    dev_premises_hold,
    direct_candidate_accepted,
    expansion_id_plan,
    expansion_transitions,
    layouts_conflict,
    load_expansion_addendum,
    load_preregistration,
    package_records,
    require_baseline_counts,
    resolve_running_identity,
    split_is_independent,
    task_duplicate_report,
    trajectory_has_release_retreat_settle,
    validate_expansion_ids,
    write_coverage,
    write_preregistration,
    assert_external_independence,
    assert_retained_arrays,
)
from examples.baselines.constrain.placesphere_learning_pilot import PilotTrajectoryDataset


def _layout(seed, start, goal, center):
    return {
        "seed": seed, "start_xy": list(start), "container_xy": list(goal),
        "cylinder": {"center_xy": list(center)},
    }


def test_preregistration_freezes_subset_task_dedup_training_and_gate(tmp_path):
    payload = build_preregistration(round_index=1, version="r1", paths={"exp": "outputs/exp/x"})
    path = tmp_path / "preregistration.json"
    write_preregistration(path, payload)
    loaded = load_preregistration(path)

    assert loaded["train_obstacle_count"] == 20
    assert loaded["train_direct_count"] == 20
    assert loaded["train_obstacle_count"] <= loaded["obstacle_plan_cap"] == 100
    assert loaded["dev_scene_count"] == 30
    assert loaded["layout_dedup"]["mode"] == "task"
    assert loaded["layout_dedup"]["cylinder_proximity_is_conflict"] is False
    assert loaded["layout_dedup"]["tolerance_m"] == LAYOUT_TOLERANCE_M == 0.02
    assert loaded["training"]["selected_weights"] == "final EMA fixed before training; no task evaluation"
    assert loaded["training"]["total_iters"] == 30000
    assert loaded["training"]["batch_size"] == 1024
    assert loaded["training"]["lr"] == 1e-4
    assert loaded["eval"]["max_steps"] == 200
    assert loaded["native_success"] == "on_bin and static and not grasped"
    assert loaded["gate"]["task_success_rate_gte"] == 0.90
    assert loaded["gate"]["safe_success_rate_gte"] == 0.85
    assert loaded["gate"]["contact_rate_lte"] == 0.10
    assert loaded["gate"]["xy_crossing_rate_lte"] == 0.10
    assert loaded["gate"]["formal_performance_claim"] is False
    assert any("release" in item for item in loaded["training"]["differences_from_formal"])
    with pytest.raises(FileExistsError):
        write_preregistration(path, payload)
    with pytest.raises(ValueError):
        build_preregistration(round_index=3, version="r3", paths={})


def test_task_dedup_keeps_distinct_tasks_with_nearby_cylinders():
    near = _layout(2, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0))
    other = _layout(3, (0.08, 0.05), (0.16, -0.08), (0.11, 0.01))
    same = _layout(4, (0.01, 0.0), (0.21, 0.0), (0.4, 0.4))
    assert not layouts_conflict(near, other, cylinder=False)
    assert layouts_conflict(near, same, cylinder=False)
    assert layouts_conflict(near, _layout(2, (-0.2, 0.2), (0.3, -0.2), (0.4, 0.4)), cylinder=False)


def test_split_must_be_independent_of_train_dev_and_prior():
    train = [_layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0))]
    prior = [_layout(8, (-0.2, 0.1), (0.15, 0.1), (-0.1, 0.2))]
    dev_ok = [_layout(3, (0.05, -0.15), (0.18, -0.12), (0.12, -0.16))]
    dev_bad = [_layout(4, (0.0, 0.01), (0.2, 0.0), (0.3, 0.3))]
    assert split_is_independent(train, dev_ok, prior, cylinder=False)
    assert not split_is_independent(train, dev_bad, prior, cylinder=False)


def test_release_retreat_settle_requires_open_gripper_and_phase_flags():
    actions = np.zeros((6, 7), np.float32)
    actions[:3, 6] = -1
    actions[3:, 6] = 1
    phases = ["retreat", "release", "settle"]
    assert trajectory_has_release_retreat_settle(actions, phases, ungrasped=True, static=True, on_bin=True)
    assert not trajectory_has_release_retreat_settle(actions, ["retreat", "release"], ungrasped=True, static=True, on_bin=True)
    closed = actions.copy()
    closed[:, 6] = -1
    assert not trajectory_has_release_retreat_settle(closed, phases, ungrasped=True, static=True, on_bin=True)
    assert not trajectory_has_release_retreat_settle(actions, phases, ungrasped=False, static=True, on_bin=True)


def _record(kind, episode_id, length=12):
    width = 39 if kind == "direct" else 42
    obs = np.zeros((length + 1, width), np.float32)
    actions = np.zeros((length, 7), np.float32)
    actions[:4, 6] = -1
    actions[4:, 6] = 1
    return {
        "kind": kind, "episode_id": episode_id, "scene_id": f"s{episode_id}",
        "obs": obs, "actions": actions,
        "terminated": np.zeros(length + 1, np.uint8),
        "truncated": np.zeros(length + 1, np.uint8),
        "native_success": True, "on_bin": True, "static": True, "ungrasped": True,
        "release_reached": True, "retreat_reached": True, "settle_reached": True,
        "obstacle_collision": False, "obstacle_crossing": False,
        "layout": _layout(episode_id, (episode_id, 0.0), (episode_id + 0.2, 0.0), (episode_id + 0.1, 0.5)),
    }


def test_package_keeps_only_the_predeclared_subset_and_refuses_incomplete_release(tmp_path):
    registration = build_preregistration(round_index=1, version="r1", paths={"exp": "x"})
    registration["train_direct_count"] = 1
    registration["train_obstacle_count"] = 1
    registration["selected_direct_ids"] = [0]
    registration["selected_obstacle_ids"] = [1000]
    records = [_record("direct", 0), _record("obstacle", 1000)]
    dest = package_records(records, tmp_path / "demos", registration)
    with h5py.File(dest / "trajectory.h5", "r") as handle:
        assert set(handle.keys()) == {"traj_0", "traj_1000"}
        assert handle["traj_0"]["actions"].shape == (12, 7)
        assert int(handle["traj_0"].attrs["ungrasped"]) == 1
        assert int(handle["traj_1000"].attrs["settle_reached"]) == 1
    grasped = [_record("direct", 0), _record("obstacle", 1000)]
    grasped[1]["ungrasped"] = False
    with pytest.raises(ValueError, match="ungrasped"):
        package_records(grasped, tmp_path / "grasped", registration)
    extra = records + [_record("obstacle", 1001)]
    with pytest.raises(ValueError, match="undeclared"):
        package_records(extra, tmp_path / "extra", registration)


def test_dataset_uses_39d_direct_and_42d_obstacle_and_refuses_extras(tmp_path):
    registration = build_preregistration(round_index=1, version="r1", paths={"exp": "x"})
    registration["train_direct_count"] = 1
    registration["train_obstacle_count"] = 1
    registration["selected_direct_ids"] = [0]
    registration["selected_obstacle_ids"] = [1000]
    dest = package_records([_record("direct", 0), _record("obstacle", 1000)], tmp_path / "ok", registration)
    direct = PilotTrajectoryDataset(dest / "trajectory.h5", dest / "trajectory.json", "direct", [0])
    obstacle = PilotTrajectoryDataset(dest / "trajectory.h5", dest / "trajectory.json", "obstacle", [1000])
    assert tuple(direct[0]["observations"].shape) == (2, 39)
    assert tuple(direct[0]["actions"].shape) == (16, 7)
    assert tuple(obstacle[0]["observations"].shape) == (2, 42)
    assert float(direct[0]["actions"][-1, 6]) == 1.0
    with h5py.File(dest / "trajectory.h5", "a") as handle:
        handle.create_group("traj_7")
    with pytest.raises(ValueError, match="undeclared"):
        PilotTrajectoryDataset(dest / "trajectory.h5", dest / "trajectory.json", "direct", [0])


def test_action_coercion_accepts_unbatched_horizon_and_rejects_wrong_width():
    horizon = torch.zeros(8, 7)
    batched = coerce_action_sequence(horizon)
    assert tuple(batched.shape) == (1, 8, 7)
    assert tuple(coerce_action_sequence(torch.zeros(1, 8, 7)).shape) == (1, 8, 7)
    queued = batched_observation(torch.zeros(2, 42))
    assert tuple(queued.shape) == (1, 2, 42)
    with pytest.raises(ValueError, match="width 7"):
        coerce_action_sequence(torch.zeros(2, 106))
    with pytest.raises(ValueError, match="42"):
        batched_observation(torch.zeros(2, 45))


def test_gate_pauses_unless_every_threshold_holds_and_blocks_ticket_08():
    passing = {
        "task_success_rate": 0.90, "safe_success_rate": 0.90,
        "obstacle_contact_rate": 0.10, "xy_crossing_rate": 0.0,
        "complete": True, "runtime_errors": 0,
    }
    assert decide_gate(passing)["decision"] == "pass"
    assert decide_gate(passing)["formal_performance_claim"] is False
    assert decide_gate(passing)["ticket08"] == "eligible"
    paused = decide_gate(dict(passing, obstacle_contact_rate=0.2))
    assert paused["decision"] == "pause"
    assert "obstacle_contact_rate" in paused["failures"]
    assert paused["ticket08"] == "blocked"
    assert decide_gate(dict(passing, complete=False, runtime_errors=1))["decision"] == "pause"


def test_ticket02_comparison_is_not_a_same_distribution_delta():
    rows = [{
        "first_contact": {"phase": "transport", "location": "side", "link_group": "gripper"},
        "crossing": {"relation": "geometric_above_top"},
        "terminal_mechanism": {"mechanism": "not_stabilized", "replaces_native_success": False},
    }]
    report = compare_ticket02(rows)
    assert report["same_distribution_delta"] is False
    assert report["formal_performance_claim"] is False
    assert report["ticket02"]["distribution"] != "this pilot"
    assert report["pilot_terminal_mechanism_counts"]["not_stabilized"] == 1
    assert report["pilot_crossing_relation_counts"]["geometric_above_top"] == 1


def test_dev_premises_require_ticket04_block_clearance_and_release():
    good = {
        "cylinder": {"radius": 0.03},
        "expert": {
            "direct_transport_blocked": True, "native_success": True, "safe_success": True,
            "on_bin": True, "static": True, "ungrasped": True,
            "release_phase_reached": True, "retreat_phase_reached": True, "settle_phase_reached": True,
            "obstacle_collision": False, "obstacle_crossing": False,
            "direct_block_contact": {"phase": "transport"},
            "endpoint_clearance": {"method": "full_robot_phase_vertices", "paths": ["direct", "side"]},
            "alternate_side_fallback": False,
        },
    }
    assert dev_premises_hold([good])
    missing_release = json.loads(json.dumps(good))
    missing_release["expert"]["settle_phase_reached"] = False
    assert not dev_premises_hold([missing_release])
    endpoint = json.loads(json.dumps(good))
    endpoint["expert"]["direct_block_contact"]["phase"] = "grasp_approach"
    assert not dev_premises_hold([endpoint])


def test_attempt_scene_persists_matching_direct_release(tmp_path):
    from examples.baselines.constrain.placesphere_transport_detour import _write_attempt_scene

    length = 4
    actions = np.zeros((length, 7), np.float32)
    actions[:2, 6] = -1
    actions[-1, 6] = 1
    direct = {
        "actions": actions,
        "policy_observations": np.zeros((length + 1, 39), np.float32),
        "raw_observations": np.zeros((length + 1, 42), np.float32),
        "terminated": np.zeros(length + 1, np.uint8),
        "truncated": np.zeros(length + 1, np.uint8),
        "terminal": {"native_success": True, "on_bin": True, "static": True, "ungrasped": True},
        "trace": {"phase_status": [("retreat", "reached"), ("release", "reached"), ("settle", "reached")]},
    }
    safe = {
        "actions": actions,
        "policy_observations": np.zeros((length + 1, 39), np.float32),
        "raw_observations": np.zeros((length + 1, 42), np.float32),
        "terminated": np.zeros(length + 1, np.uint8),
        "truncated": np.zeros(length + 1, np.uint8),
        "infos": [{} for _ in range(length + 1)],
        "initial_state": {"q": np.array([1.0])},
        "initial_controller": {"grip": np.array([1.0])},
        "states": [{"q": np.array([1.0])}],
        "controller_states": [{"grip": np.array([1.0])}],
    }
    _write_attempt_scene(tmp_path, safe, {"actions": np.zeros((2, 7), np.float32)}, {"seed": 1}, direct=direct)
    with h5py.File(tmp_path / "scene.h5", "r") as handle:
        assert handle["scene"]["direct_actions"].shape == (length, 7)
        assert handle["scene"]["direct_policy_observations"].shape == (length + 1, 39)
        assert float(handle["scene"]["direct_actions"][-1, 6]) == 1.0
        assert int(handle["scene"].attrs["direct_ungrasped"]) == 1
        assert int(handle["scene"].attrs["direct_settle_reached"]) == 1


def test_generate_rejects_a_duplicate_layout_before_ingest(tmp_path):
    from examples.baselines.constrain.placesphere_transport_detour import generate

    script = tmp_path / "worker.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "result = Path(sys.argv[1])\n"
        "result.joinpath('result.json').write_text(json.dumps({\n"
        "  'accepted': True, 'failure': None, 'counters': {}, 'phase_timings': {},\n"
        "  'metadata': {'seed': 7, 'start_xy': [0.0, 0.0], 'container_xy': [0.2, 0.0],\n"
        "               'cylinder': {'center_xy': [0.1, 0.0]}}}))\n"
    )

    def argv(seed, result_dir):
        return ["python", str(script), str(result_dir)]

    report = generate(
        tmp_path / "out", seed=1, target=1, max_attempts=1, max_wall_seconds=30,
        per_attempt_deadline=10, worker_argv_fn=argv,
        reject_metadata=lambda metadata: "layout_duplicate",
    )
    assert report["accepted"] == 0
    assert report["failures"].get("layout_duplicate") == 1
    assert not (tmp_path / "out" / "scenes.h5").exists()


def _used_seeds(spec):
    seeds = []
    for block in spec["blocks"]:
        seeds.extend(range(int(block["seed"]) + 1, int(block["seed"]) + int(block["max_attempts"]) + 1))
    continuation = spec["continuation"]
    seeds.extend(range(
        int(continuation["seed"]) + 1,
        int(continuation["seed"]) + int(continuation["max_attempts"]) + 1))
    return seeds


def test_expansion_preregistration_predeclares_duplicate_policy_and_place_sphere_contract(tmp_path):
    payload = build_expansion_preregistration(
        version="20261024_expansion",
        paths={"exp": "outputs/exp/x"},
        reason="user authorized one PlaceSphere data-expansion round",
        prior_version="outputs/exp/20261024_014459_placesphere_learning_pilot",
    )
    assert payload["kind"] == "authorized_data_expansion"
    assert payload["train_direct_count"] == 1000
    assert payload["train_obstacle_count"] == 100
    assert payload["obstacle_plan_cap"] == 100
    assert payload["sweeps"] == "none"
    assert payload["selected_direct_ids"] == list(range(1000))
    assert payload["selected_obstacle_ids"] == list(range(1000, 1100))
    assert payload["dev_scene_count"] == 30
    assert payload["dev_reused"] is True
    assert payload["collection"]["dev_collected"] is False
    assert payload["native_success"] == NATIVE_SUCCESS
    assert payload["eval"]["max_steps"] == 200
    duplicates = payload["within_train_duplicates"]
    assert duplicates["pairs_within_run_mutual_exclusion"] is True
    assert duplicates["direct_within_run_mutual_exclusion"] is False
    assert duplicates["direct_may_repeat_train_task_layouts"] is True
    assert duplicates["packaged_count_is_unique_count"] is False
    assert duplicates["log"] == "greedy_unique_and_conflicts"
    assert duplicates["external_excludes"] == ["ticket04", "dev30", "retained20"]
    assert duplicates["exclude_new_pairs"] is False
    assert "PickCube" in duplicates["why_predeclared"]
    assert payload["collection"]["pairs"]["within_run_mutual_exclusion"] is True
    assert payload["collection"]["pairs"]["target"] == 80
    assert payload["collection"]["pairs"]["workers"] == 2
    assert payload["collection"]["pairs"]["per_attempt_deadline"] == 160
    assert payload["collection"]["pairs"]["blocks"][0]["seed"] == 2026128001
    assert payload["collection"]["direct"]["within_run_mutual_exclusion"] is False
    assert payload["collection"]["direct"]["drop_conflicts"] is False
    assert payload["collection"]["direct"]["target"] == 900
    assert payload["collection"]["direct"]["workers"] == 4
    assert payload["collection"]["direct"]["per_attempt_deadline"] == 160
    assert [block["target"] for block in payload["collection"]["direct"]["blocks"]] == [225, 225, 225, 225]
    assert [block["max_attempts"] for block in payload["collection"]["direct"]["blocks"]] == [700, 700, 700, 700]
    selection = payload["collection_selection"]
    assert selection["retained_direct_ids"] == list(range(20))
    assert selection["retained_obstacle_ids"] == list(range(1000, 1020))
    assert selection["additional_direct_count"] == 900
    assert selection["additional_direct_obstacle_packaged"] is False
    assert selection["additional_direct_obstacle_execution"] == "retained_as_acceptance_predicate"
    assert "distribution" in selection["obstacle_execution_not_skipped_because"]
    assert payload["training"]["total_iters"] == 30000
    assert payload["training"]["batch_size"] == 1024
    assert payload["training"]["lr"] == 1e-4
    assert payload["training"]["seed"] == 1
    assert payload["gate"]["task_success_rate_gte"] == 0.90
    assert payload["gate"]["safe_success_rate_gte"] == 0.85
    assert payload["gate"]["contact_rate_lte"] == 0.10
    assert payload["gate"]["xy_crossing_rate_lte"] == 0.10
    assert payload["gate"]["formal_performance_claim"] is False
    assert "Ticket 08" in payload["gate"]["on_fail"]
    assert "Ticket 07" not in payload["gate"]["on_fail"]
    assert payload["baseline"]["base_counts"] == {"task": 0, "safe": 0, "contact": 29, "xy": 23}
    assert payload["baseline"]["constrain_counts"] == {"task": 16, "safe": 11, "contact": 8, "xy": 3}
    assert payload["baseline"]["authoritative_base"] == "base_dp"
    assert payload["baseline"]["failed_base_attempt"] is None
    occupied = set(range(2026108002, 2026108048)) | set(range(2026109002, 2026109069))
    occupied.update((20261026, 20261028, 20261029, 20261031))
    for spec in (payload["collection"]["pairs"], payload["collection"]["direct"]):
        seeds = _used_seeds(spec)
        assert len(seeds) == len(set(seeds))
        assert not (set(seeds) & occupied)
        spans = []
        for block in spec["blocks"]:
            spans.append((int(block["seed"]) + 1, int(block["seed"]) + int(block["max_attempts"])))
        continuation = spec["continuation"]
        spans.append((int(continuation["seed"]) + 1, int(continuation["seed"]) + int(continuation["max_attempts"])))
        spans.sort()
        assert all(left[1] < right[0] for left, right in zip(spans, spans[1:]))
    path = tmp_path / "preregistration.json"
    write_preregistration(path, payload)
    assert load_preregistration(path)["train_direct_count"] == 1000
    with pytest.raises(ValueError):
        build_expansion_preregistration(version="x", paths={}, reason="", prior_version="prior")


def test_direct_candidate_keeps_release_and_obstacle_predicate():
    checks = {name: True for name in PAIR_ACCEPTANCE_CHECKS}
    assert "direct_release_retreat_settle" in checks
    assert "no_obstacle_contact" in checks
    assert direct_candidate_accepted(checks)
    missing_release = dict(checks, direct_release_retreat_settle=False)
    assert not direct_candidate_accepted(missing_release)
    contacted = dict(checks, no_obstacle_contact=False)
    assert not direct_candidate_accepted(contacted)
    assert not direct_candidate_accepted({})


def test_expansion_ids_keep_original_20_and_refuse_300_obstacle():
    plan = expansion_id_plan()
    assert plan["direct"] == list(range(1000))
    assert plan["obstacle"] == list(range(1000, 1100))
    assert set(range(20)).issubset(plan["direct"])
    assert set(range(1000, 1020)).issubset(plan["obstacle"])
    validate_expansion_ids(plan["direct"], plan["obstacle"])
    with pytest.raises(ValueError, match="original"):
        validate_expansion_ids(list(range(1, 1001)), plan["obstacle"])
    with pytest.raises(ValueError, match="100"):
        validate_expansion_ids(plan["direct"], list(range(1000, 1300)))


def test_duplicate_report_does_not_call_a_repeated_set_unique():
    rows = [
        _layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0)),
        _layout(2, (0.08, 0.05), (0.16, -0.08), (0.11, 0.01)),
        _layout(3, (0.01, 0.0), (0.21, 0.0), (0.4, 0.4)),
    ]
    report = task_duplicate_report(rows)
    assert report["packaged_count"] == 3
    assert report["greedy_unique"] == 2
    assert report["conflicts"] == 1
    assert report["packaged_count_is_unique_count"] is False


def test_expansion_transitions_label_exact_place_sphere_ids():
    baseline = [
        {"scene_id": "placesphere-000", "status": "evaluated", "result": {
            "native_success": False, "safe_success": False,
            "obstacle_contact": True, "xy_forbidden_crossing": True}},
        {"scene_id": "placesphere-001", "status": "evaluated", "result": {
            "native_success": True, "safe_success": True,
            "obstacle_contact": False, "xy_forbidden_crossing": False}},
    ]
    expansion = [
        {"scene_id": "placesphere-001", "status": "evaluated", "result": {
            "native_success": False, "safe_success": False,
            "obstacle_contact": True, "xy_forbidden_crossing": False}},
        {"scene_id": "placesphere-000", "status": "evaluated", "result": {
            "native_success": True, "safe_success": False,
            "obstacle_contact": False, "xy_forbidden_crossing": True}},
    ]
    report = expansion_transitions(baseline, expansion, (
        "native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing"))
    assert report["matched_scene_ids"] == ["placesphere-000", "placesphere-001"]
    assert report["native_success"] == {"improved": 1, "regressed": 1, "both": 0, "neither": 0}
    assert report["safe_success"] == {"improved": 0, "regressed": 1, "both": 0, "neither": 1}
    assert report["obstacle_contact"] == {"cleared": 1, "introduced": 1, "both": 0, "neither": 0}
    assert report["xy_forbidden_crossing"] == {"cleared": 0, "introduced": 0, "both": 1, "neither": 1}


def test_episode_index_stores_the_checkpoint_sha():
    rows = [{"scene_index": 0, "scene_id": "placesphere-000", "status": "evaluated",
             "checkpoint_sha256": "abc"}]
    index = build_episode_index(rows, "abc", checkpoint="outputs/ckpt/x/final.pt")
    assert index["checkpoint_sha256"] == "abc"
    assert index["episodes"][0]["checkpoint_sha256"] == "abc"
    assert index["checkpoint"] == "outputs/ckpt/x/final.pt"
    with pytest.raises(ValueError, match="checkpoint"):
        build_episode_index(rows, "def", checkpoint="outputs/ckpt/x/final.pt")


def test_coverage_uses_the_frozen_30000_updates():
    coverage = dataset_coverage(windows=4096, episodes=1000, updates=30000, batch_size=1024)
    assert coverage["samples_drawn"] == 30000 * 1024
    assert coverage["epochs"] == (30000 * 1024) / 4096
    assert coverage["updates"] == 30000
    with pytest.raises(ValueError):
        dataset_coverage(windows=100, episodes=1, updates=30000, batch_size=1024)


def test_recovery_is_dated_before_rerun_and_does_not_rewrite_preregistration(tmp_path):
    registration = {
        "paths": {"exp": str(tmp_path)},
        "source_identity": {"files_sha256": "frozen"},
        "generator_identity": {"files_sha256": "generator"},
    }
    prereg = tmp_path / "preregistration.json"
    prereg.write_text(json.dumps(registration) + "\n")
    before = prereg.read_bytes()
    assert resolve_running_identity(registration, {"files_sha256": "frozen"})["matched"] == "preregistration"
    with pytest.raises(RuntimeError, match="source identity"):
        resolve_running_identity(registration, {"files_sha256": "changed"})
    recovery = {
        "files_sha256": "changed",
        "generator_files_sha256": "generator-edited",
        "recorded_before_rerun": True,
        "recorded_at": "2026-10-24T12:00:00",
    }
    (tmp_path / "recovery_identity.json").write_text(json.dumps(recovery))
    with pytest.raises(RuntimeError, match="generator"):
        resolve_running_identity(registration, {"files_sha256": "changed"})
    recovery["generator_files_sha256"] = "generator"
    recovery["recorded_before_rerun"] = False
    (tmp_path / "recovery_identity.json").write_text(json.dumps(recovery))
    with pytest.raises(RuntimeError, match="dated|before"):
        resolve_running_identity(registration, {"files_sha256": "changed"})
    recovery["recorded_before_rerun"] = True
    (tmp_path / "recovery_identity.json").write_text(json.dumps(recovery))
    matched = resolve_running_identity(registration, {"files_sha256": "changed"})
    assert matched["matched"] == "recovery_identity"
    assert prereg.read_bytes() == before


def test_external_independence_rejects_a_dev_or_prior_hit():
    train = [_layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0))]
    dev = [_layout(2, (0.08, 0.0), (0.28, 0.0), (0.2, 0.0))]
    prior = [_layout(3, (0.0, 0.2), (0.2, 0.2), (0.1, 0.2))]
    assert assert_external_independence(train, dev, prior) is True
    hit = [_layout(4, (0.01, 0.0), (0.21, 0.0), (0.4, 0.4))]
    with pytest.raises(ValueError, match="dev"):
        assert_external_independence(train, hit, prior)


def test_budget_addendum_cannot_loosen_duplicate_policy_or_acceptance(tmp_path):
    from examples.baselines.constrain.placesphere_learning_pilot import ADDENDUM_DOES_NOT_CHANGE

    payload = {
        "reason": "measured_yield_below_preregistered_attempt_budget",
        "does_not_change": list(ADDENDUM_DOES_NOT_CHANGE),
        "recorded_before_rerun": True,
        "recorded_at": "2026-10-24T12:00:00",
        "scene_contract_unchanged": True,
        "formal_performance_claim": False,
        "direct_rule": {
            "within_run_mutual_exclusion": False,
            "acceptance_predicate": "full_run_attempt",
            "obstacle_execution": "retained_as_acceptance_predicate",
            "obstacle_packaged": False,
            "target": 900,
            "external_excludes": ["ticket04", "dev30", "retained20"],
            "exclude_new_pairs": False,
            "workers": 4,
            "per_attempt_deadline": 160,
        },
    }
    path = tmp_path / "collection_budget_addendum.json"
    path.write_text(json.dumps(payload))
    assert load_expansion_addendum(path)["direct_rule"]["target"] == 900
    payload["direct_rule"]["acceptance_predicate"] = "geometry_only"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="acceptance"):
        load_expansion_addendum(path)


def test_retained_arrays_match_original_bytes(tmp_path):
    path = tmp_path / "trajectory.h5"
    obs = np.array([[1.0, 2.0]], np.float32)
    actions = np.array([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]], np.float32)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("obs", data=obs)
        group.create_dataset("actions", data=actions)
    before = path.read_bytes()
    records = [{
        "episode_id": 0,
        "provenance": {"source": "retained_r1"},
        "obs": obs.copy(),
        "actions": actions.copy(),
    }]
    assert assert_retained_arrays(path, records) == __import__("hashlib").sha256(before).hexdigest()
    assert path.read_bytes() == before
    records[0]["obs"] = np.array([[9.0, 9.0]], np.float32)
    with pytest.raises(ValueError, match="retained"):
        assert_retained_arrays(path, records)
    assert path.read_bytes() == before


def test_baseline_counts_must_match_the_frozen_rows():
    rows = [{
        "status": "evaluated",
        "result": {"native_success": False, "safe_success": False,
                   "obstacle_contact": True, "xy_forbidden_crossing": True},
    }]
    counts = require_baseline_counts(rows, {"task": 0, "safe": 0, "contact": 1, "xy": 1}, denominator=1)
    assert counts["task"] == 0
    with pytest.raises(ValueError, match="baseline"):
        require_baseline_counts(rows, {"task": 1, "safe": 0, "contact": 1, "xy": 1}, denominator=1)


REPO_ROOT = Path(__file__).resolve().parents[4]
EXPANSION_EXP = REPO_ROOT / "outputs/exp/20261024_155244_placesphere_data_expansion"


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _expansion_registration():
    registration = json.loads((EXPANSION_EXP / "preregistration.json").read_text())
    registration["paths"] = dict(registration["paths"])
    registration["paths"]["demos"] = str((REPO_ROOT / registration["paths"]["demos"]).resolve())
    return registration


def _run_dir_from(source, dest):
    dest.mkdir()
    (dest / "run_config.json").write_bytes(Path(source).read_bytes())
    return dest


def _nested_registration(demo_dir, episode_ids, updates=8, batch_size=4):
    return {
        "selected_direct_ids": [],
        "selected_obstacle_ids": list(episode_ids),
        "obstacle_plan_cap": 100,
        "training": {"total_iters": updates, "batch_size": batch_size},
        "paths": {"demos": str(demo_dir)},
    }


def _nested_config(demo_dir, episode_ids, *, episodes=None, windows=None, updates=8, batch_size=4):
    registration = _nested_registration(demo_dir, episode_ids, updates, batch_size)
    identity = {
        "selected_class": "obstacle",
        "episodes": len(episode_ids) if episodes is None else episodes,
        "selected_episode_ids": list(episode_ids),
        "h5_sha256": _file_sha256(Path(demo_dir) / "trajectory.h5"),
        "manifest_sha256": _file_sha256(Path(demo_dir) / "trajectory.json"),
    }
    if windows is not None:
        identity["windows"] = windows
    return {
        "model": {"input_dim": 7},
        "base_weights": {"updates": updates},
        "data_identity": identity,
        "training": {"total_iters": updates, "batch_size": batch_size},
        "selected_weights": "final EMA fixed before training; no task evaluation",
    }, registration


def _write_nested(tmp_path, config):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps(config))
    return run_dir


def test_write_coverage_counts_nested_obstacle_windows_not_episode_ids(tmp_path):
    demo_dir = tmp_path / "demos"
    episode_ids = [1000]
    registration = _nested_registration(demo_dir, episode_ids)
    package_records([_record("obstacle", 1000, length=20)], demo_dir, registration)
    config, registration = _nested_config(demo_dir, episode_ids)
    assert "windows" not in config
    assert "windows" not in config["data_identity"]
    coverage = write_coverage(_write_nested(tmp_path, config), registration)
    assert coverage["windows"] == 19
    assert coverage["episodes"] == 1
    assert coverage["epochs"] == (8 * 4) / 19
    assert coverage["formal_performance_claim"] is False
    assert coverage["generalization_claim"] is False
    written = json.loads((tmp_path / "run" / "coverage.json").read_text())
    assert written["windows"] == 19


def test_write_coverage_reports_retained_constrain_windows(tmp_path):
    source = EXPANSION_EXP / "train_constrain" / "run_config.json"
    historical = {
        "config": source.read_bytes(),
        "coverage": (source.parent / "coverage.json").read_bytes(),
        "summary": (source.parent / "train_summary.json").read_bytes(),
    }
    config = json.loads(historical["config"])
    assert "windows" not in config
    assert "windows" not in config["data_identity"]
    assert config["data_identity"]["episodes"] == 100
    assert config["data_identity"]["selected_class"] == "obstacle"
    coverage = write_coverage(_run_dir_from(source, tmp_path / "run"), _expansion_registration())
    assert coverage["windows"] == 13533
    assert coverage["episodes"] == 100
    assert coverage["updates"] == 30000
    assert coverage["batch_size"] == 1024
    assert coverage["samples_drawn"] == 30000 * 1024
    assert coverage["epochs"] == (30000 * 1024) / 13533
    assert f"{coverage['epochs']:.2f}" == "2270.01"
    assert coverage["formal_performance_claim"] is False
    assert source.read_bytes() == historical["config"]
    assert (source.parent / "coverage.json").read_bytes() == historical["coverage"]
    assert (source.parent / "train_summary.json").read_bytes() == historical["summary"]


def test_write_coverage_keeps_base_top_level_window_count(tmp_path):
    source = EXPANSION_EXP / "train_base" / "run_config.json"
    before = source.read_bytes()
    before_coverage = (source.parent / "coverage.json").read_bytes()
    coverage = write_coverage(_run_dir_from(source, tmp_path / "run"), _expansion_registration())
    assert coverage["windows"] == 113188
    assert coverage["episodes"] == 1000
    assert coverage["epochs"] == (30000 * 1024) / 113188
    assert f"{coverage['epochs']:.2f}" == "271.41"
    assert source.read_bytes() == before
    assert (source.parent / "coverage.json").read_bytes() == before_coverage


def test_write_coverage_rejects_missing_zero_and_nonpositive_counts(tmp_path):
    demo_dir = tmp_path / "demos"
    episode_ids = [1000]
    package_records([_record("obstacle", 1000, length=20)], demo_dir, _nested_registration(demo_dir, episode_ids))
    config, registration = _nested_config(demo_dir, episode_ids)

    zero_episodes = json.loads(json.dumps(config))
    zero_episodes["data_identity"]["episodes"] = 0
    run_dir = _write_nested(tmp_path / "zero-episodes", zero_episodes)
    with pytest.raises(ValueError, match="episodes"):
        write_coverage(run_dir, registration)
    assert not (run_dir / "coverage.json").exists()

    zero_windows = json.loads(json.dumps(config))
    zero_windows["data_identity"]["windows"] = 0
    with pytest.raises(ValueError, match="windows"):
        write_coverage(_write_nested(tmp_path / "zero-windows", zero_windows), registration)

    negative = json.loads(json.dumps(config))
    negative["data_identity"]["windows"] = -19
    with pytest.raises(ValueError, match="windows"):
        write_coverage(_write_nested(tmp_path / "negative-windows", negative), registration)

    fractional = json.loads(json.dumps(config))
    fractional["data_identity"]["windows"] = 19.5
    with pytest.raises(ValueError, match="windows"):
        write_coverage(_write_nested(tmp_path / "fractional-windows", fractional), registration)

    missing_demos = json.loads(json.dumps(config))
    blind = json.loads(json.dumps(registration))
    blind["paths"] = {"demos": str(tmp_path / "absent")}
    with pytest.raises(ValueError, match="windows|demo"):
        write_coverage(_write_nested(tmp_path / "missing-demos", missing_demos), blind)

    mismatched = json.loads(json.dumps(config))
    mismatched["data_identity"]["h5_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        write_coverage(_write_nested(tmp_path / "bad-hash", mismatched), registration)

    budget = json.loads(json.dumps(registration))
    budget["training"] = {"total_iters": 1, "batch_size": 4}
    with pytest.raises(ValueError, match="budget"):
        write_coverage(_write_nested(tmp_path / "bad-budget", config), budget)

    base = {
        "windows": 0,
        "episodes": 1,
        "training": {"total_iters": 8, "batch_size": 4},
    }
    with pytest.raises(ValueError, match="windows"):
        write_coverage(_write_nested(tmp_path / "base-zero", base), registration)
