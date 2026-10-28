"""Seams for the PickCube transport-detour learnability pilot."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import torch

from examples.baselines.constrain.pickcube_learning_pilot import (
    LAYOUT_TOLERANCE_M,
    TRAIN_OBSTACLE_COUNT,
    PilotTrajectoryDataset,
    obstacle_proximity_weight,
    weighted_epsilon_mse,
    build_window_gap_index,
    build_window_gap_preregistration,
    classify_window_gap,
    draw_sha256,
    fixed_window_draw,
    longer_budget_start,
    materialize_window,
    rotation_gap,
    assert_exposure_budget,
    assert_frozen_source_identity,
    build_exposure_preregistration,
    expected_base_updates,
    assert_obstacle_increment_independence,
    base_replay_reusable,
    build_episode_index,
    build_expansion_preregistration,
    build_obstacle_increment_preregistration,
    build_preregistration,
    build_fullscene500_preregistration,
    dataset_coverage,
    decide_gate,
    direct_candidate_accepted,
    enforce_runtime_source_identity,
    load_obstacle_increment_addendum,
    expansion_id_plan,
    freeze_pilot_identity,
    expansion_transitions,
    layouts_conflict,
    load_preregistration,
    layouts_from_episodes,
    load_preregistration,
    main,
    obstacle_increment_id_plan,
    package_records,
    partition_task_independent,
    retained_expansion_records,
    separate_failed_attempts,
    split_is_independent,
    validate_expansion_ids,
    validate_obstacle_increment_ids,
    write_preregistration,
    _pickcube_ddim_scheduler,
)
from examples.baselines.constrain.pickcube_transport_detour import (
    _write_attempt_scene,
    generate,
)


def _layout(seed, start, goal, center):
    return {"seed": seed, "start_xy": list(start), "goal_xy": list(goal), "center_xy": list(center)}


def test_preregistration_freezes_subset_dedup_training_and_gate(tmp_path):
    payload = build_preregistration(round_index=1, version="r1", paths={"exp": "outputs/exp/x"})
    path = tmp_path / "preregistration.json"
    write_preregistration(path, payload)
    loaded = load_preregistration(path)

    assert loaded["train_obstacle_count"] == 20
    assert loaded["train_direct_count"] == 20
    assert loaded["train_obstacle_count"] <= loaded["obstacle_plan_cap"] == 100
    assert loaded["dev_scene_count"] == 30
    assert loaded["layout_dedup"]["tolerance_m"] == LAYOUT_TOLERANCE_M == 0.02
    assert loaded["training"]["selected_weights"] == "final EMA fixed before training; no task evaluation"
    assert loaded["training"]["total_iters"] == 30000
    assert loaded["training"]["batch_size"] == 1024
    assert loaded["gate"]["task_success_rate_gte"] == 0.90
    assert loaded["gate"]["safe_success_rate_gte"] == 0.85
    assert loaded["gate"]["contact_rate_lte"] == 0.10
    assert loaded["gate"]["xy_crossing_rate_lte"] == 0.10
    assert loaded["gate"]["formal_performance_claim"] is False
    assert any("1000" in item for item in loaded["training"]["differences_from_formal"])
    with pytest.raises(FileExistsError):
        write_preregistration(path, payload)
    with pytest.raises(ValueError):
        build_preregistration(round_index=3, version="r3", paths={})


def test_round2_task_dedup_keeps_distinct_tasks_with_nearby_cylinders():
    near_cylinder = _layout(2, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0))
    other_task = _layout(3, (0.08, 0.05), (0.16, -0.08), (0.11, 0.01))
    same_task = _layout(4, (0.01, 0.0), (0.21, 0.0), (0.4, 0.4))
    assert layouts_conflict(near_cylinder, other_task)
    assert not layouts_conflict(near_cylinder, other_task, cylinder=False)
    assert layouts_conflict(near_cylinder, same_task, cylinder=False)
    payload = build_preregistration(
        round_index=2, version="r2", paths={"exp": "x"}, reason="cylinder clause infeasible",
        prior_version="r1", dedup_mode="task",
    )
    assert payload["layout_dedup"]["cylinder_proximity_is_conflict"] is False
    assert payload["layout_dedup"]["within_run_enforced"] is True


def test_layout_dedup_rejects_shared_task_shared_cylinder_or_seed():
    base = _layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0))
    assert layouts_conflict(base, _layout(2, (0.01, 0.0), (0.21, 0.0), (0.4, 0.4)))
    assert layouts_conflict(base, _layout(2, (-0.2, 0.2), (0.3, -0.2), (0.11, 0.01)))
    assert layouts_conflict(base, _layout(1, (-0.2, 0.2), (0.3, -0.2), (0.4, 0.4)))
    assert not layouts_conflict(base, _layout(9, (-0.2, 0.2), (0.3, -0.2), (0.4, 0.4)))


def test_split_must_be_independent_of_train_dev_and_prior():
    train = [_layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0))]
    prior = [_layout(8, (-0.2, 0.1), (0.15, 0.1), (-0.1, 0.2))]
    dev_ok = [_layout(3, (0.05, -0.15), (0.18, -0.12), (0.12, -0.16))]
    dev_bad = [_layout(4, (0.0, 0.01), (0.2, 0.0), (0.3, 0.3))]
    assert split_is_independent(train, dev_ok, prior)
    assert not split_is_independent(train, dev_bad, prior)


def test_gate_pauses_unless_every_threshold_holds_and_is_not_formal_performance():
    passing = {
        "task_success_rate": 0.90, "safe_success_rate": 0.90,
        "obstacle_contact_rate": 0.10, "xy_crossing_rate": 0.0,
        "complete": True, "runtime_errors": 0,
    }
    assert decide_gate(passing)["decision"] == "pass"
    assert decide_gate(passing)["formal_performance_claim"] is False
    dirty = dict(passing, obstacle_contact_rate=0.2)
    paused = decide_gate(dirty)
    assert paused["decision"] == "pause"
    assert "obstacle_contact_rate" in paused["failures"]
    assert decide_gate(dict(passing, complete=False, runtime_errors=1))["decision"] == "pause"


def _record(kind, episode_id, length=20):
    obs = np.zeros((length + 1, 45), np.float32)
    obs[:, -1] = 0.03
    actions = np.zeros((length, 7), np.float32)
    actions[:, 6] = -1
    return {
        "kind": kind, "episode_id": episode_id, "scene_id": f"s{episode_id}",
        "obs": obs, "actions": actions,
        "terminated": np.zeros(length + 1, np.uint8),
        "truncated": np.zeros(length + 1, np.uint8),
        "native_success": True, "held_at_success": True,
        "obstacle_collision": False, "obstacle_crossing": False,
        "layout": _layout(episode_id, (episode_id, 0.0), (episode_id + 0.2, 0.0), (episode_id + 0.1, 0.5)),
    }


def test_package_keeps_only_the_predeclared_subset_and_refuses_extras(tmp_path):
    registration = build_preregistration(round_index=1, version="r1", paths={"exp": "x"})
    registration["train_direct_count"] = 1
    registration["train_obstacle_count"] = 1
    registration["selected_direct_ids"] = [0]
    registration["selected_obstacle_ids"] = [1000]
    records = [_record("direct", 0), _record("obstacle", 1000)]
    dest = package_records(records, tmp_path / "demos", registration)
    with h5py.File(dest / "trajectory.h5", "r") as handle:
        assert set(handle.keys()) == {"traj_0", "traj_1000"}
        assert handle["traj_0"].attrs["kind"] == "direct"
        assert handle["traj_1000"].attrs["kind"] == "obstacle"
    extra = records + [_record("obstacle", 1001)]
    with pytest.raises(ValueError, match="undeclared"):
        package_records(extra, tmp_path / "extra", registration)


def test_dataset_refuses_an_undeclared_trajectory_and_trains_direct_without_obstacle(tmp_path):
    registration = build_preregistration(round_index=1, version="r1", paths={"exp": "x"})
    registration["train_direct_count"] = 1
    registration["train_obstacle_count"] = 1
    registration["selected_direct_ids"] = [0]
    registration["selected_obstacle_ids"] = [1000]
    dest = package_records([_record("direct", 0), _record("obstacle", 1000)], tmp_path / "ok", registration)
    dataset = PilotTrajectoryDataset(dest / "trajectory.h5", dest / "trajectory.json", "direct", [0])
    window = dataset[0]
    assert tuple(window["observations"].shape) == (2, 42)
    assert tuple(window["actions"].shape) == (16, 7)
    with h5py.File(dest / "trajectory.h5", "a") as handle:
        handle.create_group("traj_7")
    with pytest.raises(ValueError, match="undeclared"):
        PilotTrajectoryDataset(dest / "trajectory.h5", dest / "trajectory.json", "direct", [0])


def test_fullscene500_dataset_encodes_native_actions_and_preserves_closed_terminal_padding(tmp_path):
    from examples.baselines.constrain.pickcube_action_codec import CONTROLLER_RANGE

    direct = _record("direct", 0, length=20)
    direct["actions"][:, 0] = 0.175
    direct["actions"][-1, 6] = -1.0
    obstacle = _record("obstacle", 1000, length=20)
    dest = package_records([direct, obstacle], tmp_path / "encoded-source", {
        "selected_direct_ids": [0], "selected_obstacle_ids": [1000],
        "train_direct_count": 1, "train_obstacle_count": 1, "obstacle_plan_cap": 100,
    })
    dataset = PilotTrajectoryDataset(
        dest / "trajectory.h5", dest / "trajectory.json", "direct", [0], CONTROLLER_RANGE)
    last = dataset[-1]["actions"]
    assert torch.allclose(last[:3, 0], torch.full((3,), 0.5))
    assert torch.equal(last[3:, :6], torch.zeros((13, 6)))
    assert torch.equal(last[:, 6], torch.full((16,), -1.0))


def test_fullscene500_preregistration_freezes_matched_training_and_codec(tmp_path):
    from examples.baselines.constrain.pickcube_action_codec import profile_sha256

    registration = build_fullscene500_preregistration(
        {"dev_scenes": "outputs/exp/frozen/dev"},
        {"datasets": {"h5": {"sha256": "h"}, "manifest": {"sha256": "m"}},
         "baseline": {}, "validation": {}},
        {"files_sha256": "source"},
    )
    assert registration["selected_direct_ids"] == list(range(1000))
    assert registration["selected_obstacle_ids"] == list(range(1000, 1500))
    assert registration["training"]["base_updates"] == 30000
    assert registration["training"]["constrain_updates"] == 150000
    assert registration["action_encoding_sha256"] == profile_sha256()
    assert registration["training"]["inference"]["execution_horizon"] == 2
    registration["gate"] = {"formal_performance_claim": False}
    path = tmp_path / "fullscene500-preregistration.json"
    path.write_text(json.dumps(registration))
    assert load_preregistration(path)["train_obstacle_count"] == 500
    registration["action_encoding_sha256"] = "0" * 64
    path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="contract mismatch"):
        load_preregistration(path)


def test_pickcube_ddim_runtime_option_preserves_50_default_and_uses_full_endpoint_sequence():
    legacy = _pickcube_ddim_scheduler()
    full = _pickcube_ddim_scheduler(100)
    assert len(legacy.timesteps) == 50
    assert len(full.timesteps) == 100
    assert int(legacy.timesteps[0]) == 98
    assert int(full.timesteps[0]) == 99
    assert int(full.timesteps[-1]) == 0
    assert legacy.config.num_train_timesteps == full.config.num_train_timesteps == 100
    assert legacy.config.beta_schedule == full.config.beta_schedule == "squaredcos_cap_v2"
    assert legacy.config.prediction_type == full.config.prediction_type == "epsilon"
    with pytest.raises(ValueError, match="exactly 50 or 100"):
        _pickcube_ddim_scheduler(75)


def test_attempt_scene_persists_matching_direct_terminals(tmp_path):
    length = 4
    direct = {
        "actions": np.ones((length, 7), np.float32),
        "observations": [np.full(45, 0.2, np.float32) for _ in range(length + 1)],
        "terminated": np.zeros(length + 1, np.uint8),
        "truncated": np.zeros(length + 1, np.uint8),
        "safety": {"native_success": True, "held_at_success": True},
    }
    safe = {
        "states": [{"q": np.array([1.0])}],
        "observations": [np.full(45, 0.3, np.float32) for _ in range(length + 1)],
        "actions": np.full((length, 7), 0.4, np.float32),
        "initial_snapshot": (None, {"grip": 1}),
        "terminated": np.array([0, 0, 0, 0, 1], np.uint8),
        "truncated": np.zeros(length + 1, np.uint8),
    }
    block = {"actions": np.zeros((2, 7), np.float32)}
    _write_attempt_scene(tmp_path, safe, block, {"seed": 1}, direct)
    with h5py.File(tmp_path / "scene.h5", "r") as handle:
        assert handle["scene"]["direct_actions"].shape == (length, 7)
        assert handle["scene"]["direct_obs"].shape == (length + 1, 45)
        assert int(handle["scene"].attrs["direct_native_success"]) == 1


def test_merge_refuses_layout_conflict_and_keeps_independent_ids(tmp_path):
    from examples.baselines.constrain.pickcube_learning_pilot import merge_scene_catalogs

    def catalog(path, scene_id, seed, start):
        path.mkdir()
        (path / "source_identity.json").write_text(json.dumps({"files_sha256": "abc"}) + "\n")
        with h5py.File(path / "scenes.h5", "w") as handle:
            handle.create_dataset(f"{scene_id}/marker", data=np.array([seed]))
        (path / "manifest.json").write_text(json.dumps({
            "scenes": [{
                "scene_id": scene_id, "seed": seed,
                "start_xy": [start, 0.0], "goal_xy": [start + 0.3, 0.2],
                "cylinder": {"center_xy": [start + 0.15, 0.4]},
            }],
        }) + "\n")

    left, right, clash = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    catalog(left, "pickcube-000", 1, 0.0)
    catalog(right, "pickcube-004", 2, 0.5)
    catalog(clash, "pickcube-000", 3, 0.01)
    assert merge_scene_catalogs([left, right], tmp_path / "ok", seed=9) == 2
    merged = json.loads((tmp_path / "ok" / "manifest.json").read_text())
    assert [row["scene_id"] for row in merged["scenes"]] == ["pickcube-000", "pickcube-001"]
    with pytest.raises(ValueError, match="layout-independent"):
        merge_scene_catalogs([left, clash], tmp_path / "bad", seed=9)


def test_generate_rejects_a_duplicate_layout_before_ingest(tmp_path):
    def worker(seed, result_dir):
        payload = {"accepted": True, "failure": None, "counters": {}, "phase_timings": {},
                   "metadata": {"seed": 7, "start_xy": [0.0, 0.0], "goal_xy": [0.2, 0.0],
                                "cylinder": {"center_xy": [0.1, 0.0]}}}
        Path(result_dir, "result.json").write_text(json.dumps(payload))
        return ["python", "-c", "raise SystemExit(0)"]

    # The parent reads result.json written by the worker process. Use a worker
    # command that writes the duplicate metadata, matching generate's contract.
    script = tmp_path / "worker.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "result = Path(sys.argv[1])\n"
        "result.joinpath('result.json').write_text(json.dumps({\n"
        "  'accepted': True, 'failure': None, 'counters': {}, 'phase_timings': {},\n"
        "  'metadata': {'seed': 7, 'start_xy': [0.0, 0.0], 'goal_xy': [0.2, 0.0],\n"
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


def test_expansion_preregistration_freezes_counts_reuse_and_full_acceptance():
    payload = build_expansion_preregistration(
        version="20261024_expansion",
        paths={"exp": "outputs/exp/x"},
        reason="user authorized one PickCube data-expansion round",
        prior_version="outputs/exp/20261023_235744_pickcube_learning_pilot_r2",
    )
    assert payload["kind"] == "authorized_data_expansion"
    assert payload["train_direct_count"] == 1000
    assert payload["train_obstacle_count"] == 100
    assert payload["obstacle_plan_cap"] == 100
    assert payload["selected_direct_ids"] == list(range(1000))
    assert payload["selected_obstacle_ids"] == list(range(1000, 1100))
    assert payload["dev_scene_count"] == 30
    assert payload["collection"]["dev_collected"] is False
    selection = payload["collection_selection"]
    assert selection["retained_direct_ids"] == list(range(20))
    assert selection["retained_obstacle_ids"] == list(range(1000, 1020))
    assert selection["new_pair_count"] == 80
    assert selection["additional_direct_count"] == 900
    assert selection["additional_direct_obstacle_packaged"] is False
    assert selection["additional_direct_obstacle_execution"] == "retained_as_acceptance_predicate"
    assert "distribution" in selection["obstacle_execution_not_skipped_because"]
    assert payload["layout_dedup"]["mode"] == "task"
    assert payload["layout_dedup"]["cylinder_proximity_is_conflict"] is False
    assert payload["training"]["total_iters"] == 30000
    assert payload["training"]["batch_size"] == 1024
    assert payload["training"]["lr"] == 1e-4
    assert payload["training"]["seed"] == 1
    assert payload["gate"]["task_success_rate_gte"] == 0.90
    assert payload["gate"]["safe_success_rate_gte"] == 0.85
    assert payload["gate"]["contact_rate_lte"] == 0.10
    assert payload["gate"]["xy_crossing_rate_lte"] == 0.10
    assert payload["gate"]["formal_performance_claim"] is False
    assert payload["baseline"]["base_counts"] == {"task": 6, "safe": 0, "contact": 30, "xy": 21}
    assert payload["baseline"]["constrain_counts"] == {"task": 11, "safe": 7, "contact": 21, "xy": 14}
    assert payload["baseline"]["authoritative_base"] == "base_dp_recovery"
    assert payload["baseline"]["failed_base_is_not_authoritative"] is True
    assert payload["collection"]["pairs"]["workers"] == 2
    assert payload["collection"]["direct"]["workers"] == 4
    assert payload["collection"]["pairs"]["max_attempts"] == 400
    assert payload["collection"]["direct"]["max_attempts"] == 3600
    assert payload["collection"]["direct"]["wall_seconds"] > 0
    assert payload["collection"]["direct"]["per_attempt_deadline"] == 150
    with pytest.raises(ValueError):
        build_expansion_preregistration(
            version="x", paths={}, reason="", prior_version="prior")


def test_direct_candidate_keeps_the_pair_acceptance_predicate():
    checks = {
        "direct_native_success": True,
        "direct_held": True,
        "direct_transport_blocked": True,
        "obstacle_native_success": True,
        "obstacle_held": True,
        "no_obstacle_contact": True,
        "no_obstacle_crossing": True,
        "join_continuous": True,
        "action_observation_contract": True,
    }
    assert direct_candidate_accepted(checks)
    blocked = dict(checks, direct_transport_blocked=False)
    assert not direct_candidate_accepted(blocked)
    assert not direct_candidate_accepted({})


def test_expansion_ids_include_original_20_and_refuse_extra_obstacle():
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


def test_partition_uses_task_identity_and_keeps_nearby_cylinders():
    distinct = [
        _layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0)),
        _layout(2, (0.08, 0.05), (0.16, -0.08), (0.11, 0.01)),
    ]
    kept, rejected = partition_task_independent(distinct)
    assert len(kept) == 2 and rejected == []
    clash = [
        _layout(1, (0.0, 0.0), (0.2, 0.0), (0.1, 0.0)),
        _layout(3, (0.01, 0.0), (0.21, 0.0), (0.4, 0.4)),
    ]
    kept, rejected = partition_task_independent(clash)
    assert len(kept) == 1 and len(rejected) == 1


def test_expansion_transitions_match_scene_ids_and_separate_failed_attempts():
    baseline = [
        {"scene_id": "pickcube-000", "status": "evaluated", "result": {
            "native_success": False, "safe_success": False,
            "obstacle_contact": True, "xy_forbidden_crossing": True}},
        {"scene_id": "pickcube-001", "status": "evaluated", "result": {
            "native_success": True, "safe_success": False,
            "obstacle_contact": True, "xy_forbidden_crossing": False}},
    ]
    expansion = [
        {"scene_id": "pickcube-001", "status": "evaluated", "result": {
            "native_success": True, "safe_success": True,
            "obstacle_contact": False, "xy_forbidden_crossing": False}},
        {"scene_id": "pickcube-000", "status": "evaluated", "result": {
            "native_success": True, "safe_success": False,
            "obstacle_contact": False, "xy_forbidden_crossing": True}},
    ]
    report = expansion_transitions(baseline, expansion, ("native_success", "safe_success",
                                                         "obstacle_contact", "xy_forbidden_crossing"))
    assert report["matched_scene_ids"] == ["pickcube-000", "pickcube-001"]
    assert report["native_success"] == {"improved": 1, "regressed": 0, "both": 1, "neither": 0}
    assert report["safe_success"]["improved"] == 1
    assert report["obstacle_contact"] == {"cleared": 2, "introduced": 0, "both": 0, "neither": 0}
    assert report["xy_forbidden_crossing"] == {"cleared": 0, "introduced": 0, "both": 1, "neither": 1}
    failed = [{"scene_id": "pickcube-000", "status": "runtime_error"}]
    separated = separate_failed_attempts(failed, baseline)
    assert separated["failed_attempt_errors"] == 1
    assert separated["failed_attempt_is_authoritative"] is False
    assert separated["recovery_authoritative"] is True
    assert separated["recovery_evaluated"] == 2


def test_episode_index_stores_the_checkpoint_sha():
    rows = [{"scene_index": 0, "scene_id": "pickcube-000", "status": "evaluated",
             "checkpoint_sha256": "abc"}]
    index = build_episode_index(rows, "abc", checkpoint="outputs/ckpt/x/final.pt")
    assert index["checkpoint_sha256"] == "abc"
    assert index["episodes"][0]["checkpoint_sha256"] == "abc"
    assert index["checkpoint"] == "outputs/ckpt/x/final.pt"
    with pytest.raises(ValueError, match="checkpoint"):
        build_episode_index(rows, "def", checkpoint="outputs/ckpt/x/final.pt")


def test_dataset_coverage_uses_the_frozen_update_budget():
    coverage = dataset_coverage(windows=2048, episodes=1000, updates=30000, batch_size=1024)
    assert coverage["samples_drawn"] == 30000 * 1024
    assert coverage["epochs"] == (30000 * 1024) / 2048
    assert coverage["updates"] == 30000
    with pytest.raises(ValueError):
        dataset_coverage(windows=100, episodes=1, updates=30000, batch_size=1024)


def test_expansion_addendum_keeps_counts_and_acceptance_and_refuses_a_rule_change(tmp_path):
    from examples.baselines.constrain.pickcube_learning_pilot import (
        ADDENDUM_DOES_NOT_CHANGE, load_expansion_addendum,
    )

    payload = {
        "reason": "measured_yield_below_preregistered_attempt_budget",
        "does_not_change": list(ADDENDUM_DOES_NOT_CHANGE),
        "scene_contract_unchanged": True,
        "formal_performance_claim": False,
        "direct_rule": {
            "within_run_mutual_exclusion": False,
            "acceptance_predicate": "full_run_attempt",
            "obstacle_execution": "retained_as_acceptance_predicate",
            "obstacle_packaged": False,
            "target": 900,
            "external_excludes": ["ticket03", "dev30", "retained20"],
            "exclude_new_pairs": False,
            "workers": 4,
            "per_attempt_deadline": 150,
        },
    }
    path = tmp_path / "collection_budget_addendum.json"
    path.write_text(json.dumps(payload))
    assert load_expansion_addendum(path)["direct_rule"]["target"] == 900
    payload["direct_rule"]["target"] = 300
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="counts"):
        load_expansion_addendum(path)


def test_proximity_weight_peaks_at_the_obstacle_and_is_one_beyond_8cm():
    observations = torch.zeros(4, 2, 45)
    observations[:, -1, 42] = torch.tensor([0.0, 0.04, 0.08, 0.16])
    weight = obstacle_proximity_weight(observations, near_distance_m=0.08, max_extra=1.5)
    assert weight.tolist() == [2.5, 1.75, 1.0, 1.0]


def test_weighted_epsilon_mse_normalizes_by_weight_sum():
    prediction = torch.zeros(2, 1, 1)
    noise = torch.tensor([[[2.0]], [[4.0]]])
    assert weighted_epsilon_mse(prediction, noise, torch.tensor([1.0, 3.0])).item() == 13.0
    assert weighted_epsilon_mse(prediction, noise, None).item() == 10.0


def test_source_identity_mismatch_is_not_masked():
    registration = {"source_identity": {"files_sha256": "frozen"}}
    assert assert_frozen_source_identity(registration, {"files_sha256": "frozen"})["files_sha256"] == "frozen"
    with pytest.raises(RuntimeError, match="source identity"):
        assert_frozen_source_identity(registration, {"files_sha256": "changed"})


def test_training_optimization_kind_cannot_skip_source_identity(tmp_path):
    registration = {
        "kind": "authorized_training_optimization",
        "source_identity": {"files_sha256": "frozen"},
        "paths": {"exp": str(tmp_path)},
    }
    with pytest.raises(RuntimeError, match="source identity"):
        enforce_runtime_source_identity(
            registration, {"files_sha256": "changed"}, "train-constrain")
    matched = enforce_runtime_source_identity(
        registration, {"files_sha256": "frozen"}, "evaluate")
    assert matched["matched"] == "preregistration"
    recovery = {
        "files_sha256": "changed",
        "preregistration_files_sha256": "frozen",
        "reason": "appended before rerun",
        "exempt": True,
    }
    (tmp_path / "recovery_identity.json").write_text(json.dumps(recovery))
    with pytest.raises(RuntimeError, match="bypass"):
        enforce_runtime_source_identity(
            registration, {"files_sha256": "changed"}, "train-constrain")


def test_train_constrain_refuses_optimization_identity_mismatch(tmp_path):
    payload = build_preregistration(1, "opt", {"exp": str(tmp_path)})
    payload["kind"] = "authorized_training_optimization"
    payload["source_identity"] = {"files_sha256": "not-the-running-tree"}
    payload["paths"]["exp"] = str(tmp_path)
    path = tmp_path / "preregistration.json"
    write_preregistration(path, payload)
    with pytest.raises(RuntimeError, match="source identity"):
        main([
            "train-constrain", "--registration", str(path),
            "--demo-dir", str(tmp_path), "--base-checkpoint", str(tmp_path / "missing.pt"),
            "--run-dir", str(tmp_path / "run"), "--checkpoint-dir", str(tmp_path / "ckpt"),
        ])


def test_obstacle_increment_locks_300_and_rejects_proximity_baseline():
    payload = build_obstacle_increment_preregistration(
        version="obstacle300",
        paths={"exp": "outputs/exp/x"},
        reason="authorized 100 to 300 obstacle increment; direct stays the existing 1000",
        prior_version="outputs/exp/20261024_104348_pickcube_data_expansion",
    )
    assert payload["kind"] == "authorized_obstacle_increment"
    assert payload["train_direct_count"] == 1000
    assert payload["train_obstacle_count"] == 300
    assert payload["obstacle_plan_cap"] == 300
    assert payload["authorized_obstacle_cap"] == 1000
    assert payload["slice_is_not_the_authorized_cap"] is True
    assert payload["selected_direct_ids"] == list(range(1000))
    assert payload["selected_obstacle_ids"] == list(range(1000, 1300))
    assert payload["training"]["loss_weight"] is None
    assert payload["training"]["proximity_weight"] == "off"
    assert payload["training"]["base_retrained"] is False
    assert payload["direct_unique_claim"] is False
    assert payload["layout_dedup"]["within_run_enforced"] is True
    assert payload["layout_dedup"]["external_excludes"] == ["ticket03", "dev30", "existing100"]
    assert payload["comparison"]["constrain_counts"] == {"task": 26, "safe": 13, "contact": 17, "xy": 3}
    assert payload["comparison"]["rejected_proximity_is_not_baseline"] is True
    assert payload["comparison"]["rejected_proximity_counts"] == {
        "task": 22, "safe": 11, "contact": 18, "xy": 4,
    }
    assert payload["collection"]["obstacle"]["target"] == 200
    assert payload["collection"]["obstacle"]["per_attempt_deadline"] == 150
    assert payload["collection"]["obstacle"]["max_attempts"] == 3000
    assert payload["gate"]["formal_performance_claim"] is False
    with pytest.raises(ValueError):
        build_obstacle_increment_preregistration(
            version="x", paths={}, reason="", prior_version="prior")


def test_obstacle_increment_ids_keep_original_100_and_refuse_a_larger_slice():
    plan = obstacle_increment_id_plan()
    assert plan["direct"] == list(range(1000))
    assert plan["retained_obstacle"] == list(range(1000, 1100))
    assert plan["new_obstacle"] == list(range(1100, 1300))
    validate_obstacle_increment_ids(plan["direct"], plan["obstacle"])
    with pytest.raises(ValueError, match="original"):
        validate_obstacle_increment_ids(plan["direct"], list(range(1100, 1400)))
    with pytest.raises(ValueError, match="300"):
        validate_obstacle_increment_ids(plan["direct"], list(range(1000, 2000)))


def test_new_obstacle_layouts_must_be_unique_against_existing_100_dev_and_prior():
    existing = [_layout(index, (index * 0.05, 0.0), (index * 0.05, 0.2), (0.0, 0.1)) for index in range(3)]
    new = [_layout(100 + index, (0.4 + index * 0.05, -0.2), (0.4, 0.2), (0.5, 0.0)) for index in range(2)]
    dev = [_layout(8, (-0.2, 0.2), (0.2, 0.2), (0.0, 0.3))]
    prior = [_layout(9, (-0.3, -0.2), (0.3, -0.2), (0.0, -0.3))]
    assert_obstacle_increment_independence(existing, new, dev, prior)
    clash = list(new)
    clash.append(_layout(50, (0.01, 0.0), (0.01, 0.2), (0.4, 0.4)))
    with pytest.raises(ValueError, match="unique"):
        assert_obstacle_increment_independence(existing, clash, dev, prior)


def test_base_replay_requires_checkpoint_state_and_seed():
    rows = [{
        "scene_index": 0, "scene_id": "pickcube-000", "status": "evaluated",
        "checkpoint_sha256": "abc", "seed": 7, "restored_state_sha256": "state",
    }]
    scenes = [{"seed": 7, "state_sha256": "state"}]
    assert base_replay_reusable(rows, scenes, "abc", {"files_sha256": "same"}, {"files_sha256": "same"})
    assert not base_replay_reusable(rows, scenes, "abc", {"files_sha256": "same"}, {"files_sha256": "other"})
    assert not base_replay_reusable(rows, scenes, "other", {"files_sha256": "same"}, {"files_sha256": "same"})
    assert not base_replay_reusable(rows, [{"seed": 8, "state_sha256": "state"}], "abc",
                                    {"files_sha256": "same"}, {"files_sha256": "same"})


def test_obstacle_increment_addendum_keeps_identity_and_refuses_a_rule_change(tmp_path):
    registration = build_obstacle_increment_preregistration(
        version="obstacle300", paths={"exp": "outputs/exp/x"},
        reason="authorized 100 to 300", prior_version="prior",
    )
    payload = {
        "reason": "measured_yield_below_preregistered_attempt_budget",
        "does_not_change": [
            "counts", "task_identity_definition", "gate", "training", "geometry",
            "expert", "external_excludes", "within_run_dedup", "acceptance_predicate",
        ],
        "within_run_mutual_exclusion": True,
        "target": 200,
        "measured_kept": 169,
        "continuation": {
            "name": "deficit", "seed": 2026135001, "max_attempts": 1600,
            "wall_seconds": 18000, "per_attempt_deadline": 150,
        },
    }
    path = tmp_path / "obstacle_increment_addendum.json"
    path.write_text(json.dumps(payload))
    loaded = load_obstacle_increment_addendum(path, registration)
    assert loaded["continuation"]["seed"] == 2026135001
    assert loaded["target"] == 200
    payload["within_run_mutual_exclusion"] = False
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="within-run"):
        load_obstacle_increment_addendum(path, registration)


def test_episode_layouts_read_the_nested_layout(tmp_path):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps({"episodes": [{
        "kind": "obstacle",
        "layout": {"seed": 3, "start_xy": [0.1, 0.2], "goal_xy": [0.3, 0.4], "center_xy": [0.2, 0.3]},
    }]}))
    rows = layouts_from_episodes(path, "obstacle")
    assert rows == [{
        "seed": 3, "start_xy": [0.1, 0.2], "goal_xy": [0.3, 0.4], "center_xy": [0.2, 0.3],
    }]
    assert not layouts_conflict(rows[0], _layout(9, (0.4, 0.4), (0.5, 0.5), (0.0, 0.0)), cylinder=False)


def test_retained_expansion_records_copy_arrays_and_do_not_drop_direct_duplicates(tmp_path):
    source = tmp_path / "trajectory.h5"
    manifest = {"episodes": [], "expansion": {"within_train_task_conflicts": 1}}
    with h5py.File(source, "w") as handle:
        for episode_id, kind, start in (
            (0, "direct", (0.0, 0.0)),
            (1, "direct", (0.01, 0.0)),
            (1000, "obstacle", (0.2, 0.0)),
        ):
            group = handle.create_group(f"traj_{episode_id}")
            group.attrs["kind"] = kind
            group.create_dataset("obs", data=np.full((3, 45), episode_id, np.float32))
            group.create_dataset("actions", data=np.full((2, 7), 0.1, np.float32))
            group.create_dataset("terminated", data=np.array([0, 1], np.uint8))
            group.create_dataset("truncated", data=np.zeros(2, np.uint8))
            group.create_dataset("success", data=np.array([0, 0, 1], np.uint8))
            manifest["episodes"].append({
                "episode_id": episode_id, "kind": kind,
                "scene_id": f"pickcube-{episode_id:03d}",
                "layout": _layout(episode_id, start, (start[0] + 0.2, 0.0), (0.1, 0.0)),
                "provenance": {"source": "expansion", "seed": episode_id},
                "native_success": True, "held_at_success": True,
                "obstacle_collision": False, "obstacle_crossing": False,
            })
    before = source.read_bytes()
    records, quality = retained_expansion_records(source, manifest, expected_direct=2, expected_obstacle=1)
    assert source.read_bytes() == before
    assert [record["episode_id"] for record in records] == [0, 1, 1000]
    assert quality["direct_unique_claim"] is False
    assert quality["within_direct_task_conflicts"] == 1
    assert np.array_equal(records[0]["obs"], np.full((3, 45), 0, np.float32))


def _gap_metrics(train, dev, shared, near, far):
    return {
        "train_epsilon": train, "dev_epsilon": dev, "shared_epsilon": shared,
        "near_rotation_mae": near, "far_rotation_mae": far,
    }


def test_fixed_window_draw_repeats_noise_and_timesteps():
    left_noise, left_steps = fixed_window_draw(4, 2026102501)
    right_noise, right_steps = fixed_window_draw(4, 2026102501)
    other_noise, other_steps = fixed_window_draw(4, 2026102502)
    assert torch.equal(left_noise, right_noise)
    assert torch.equal(left_steps, right_steps)
    assert draw_sha256(left_noise, left_steps) == draw_sha256(right_noise, right_steps)
    assert not torch.equal(left_noise, other_noise) or not torch.equal(left_steps, other_steps)
    assert int(left_steps.min()) >= 0 and int(left_steps.max()) <= 99
    assert tuple(left_noise.shape) == (4, 16, 7)


def test_materialize_window_pads_edges_with_the_training_rule():
    obs = torch.arange(4 * 45, dtype=torch.float32).reshape(4, 45)
    actions = torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7)
    obs_seq, act_seq = materialize_window(obs, actions, -1, 15)
    assert tuple(obs_seq.shape) == (2, 45)
    assert tuple(act_seq.shape) == (16, 7)
    assert torch.equal(obs_seq[0], obs[0]) and torch.equal(obs_seq[1], obs[0])
    assert torch.equal(act_seq[0], actions[0]) and torch.equal(act_seq[1], actions[0])
    assert torch.equal(act_seq[2], actions[1])
    assert torch.equal(act_seq[-1, :6], torch.zeros(6))
    assert act_seq[-1, 6] == actions[-1, 6]


def test_rotation_gap_averages_the_executed_slice_inside_12cm():
    predicted = torch.zeros(2, 8, 7)
    expert = torch.zeros(2, 8, 7)
    predicted[0, 0, 3] = 0.08
    predicted[1, :, 3:6] = 0.02
    report = rotation_gap(predicted, expert, torch.tensor([0.10, 0.20]), near_m=0.12)
    assert report["near_count"] == 1 and report["far_count"] == 1
    assert report["near_rotation_mae"] == pytest.approx(0.08 / 24)
    assert report["far_rotation_mae"] == pytest.approx(0.02)


def test_remaining_gap_is_longer_budget_and_a_closed_gap_is_closed_loop():
    remaining = classify_window_gap(
        _gap_metrics(0.0025, 0.0100, 0.0025, 0.090, 0.005),
        _gap_metrics(0.0024, 0.0090, 0.0024, 0.090, 0.005),
    )
    assert remaining["next"] == "longer_budget"
    assert remaining["underfit"] is True
    assert remaining["gap_closed"] is False
    closed = classify_window_gap(
        _gap_metrics(0.0020, 0.0030, 0.0021, 0.015, 0.005),
        _gap_metrics(0.0022, 0.0090, 0.0022, 0.090, 0.005),
    )
    assert closed["next"] == "closed_loop"
    assert closed["underfit"] is False
    worse_fit = classify_window_gap(
        _gap_metrics(0.0020, 0.0030, 0.0040, 0.010, 0.005),
        _gap_metrics(0.0020, 0.0080, 0.0020, 0.090, 0.005),
    )
    assert worse_fit["underfit"] is True and worse_fit["next"] == "longer_budget"
    with pytest.raises(ValueError, match="missing"):
        classify_window_gap(_gap_metrics(0.002, 0.003, 0.002, None, 0.005), _gap_metrics(0.002, 0.003, 0.002, 0.01, 0.005))


def test_exhausted_cosine_is_scratch_not_a_zero_lr_resume():
    exhausted = longer_budget_start({"_last_lr": [0.0], "last_epoch": 30000}, True)
    assert exhausted["start"] == "scratch"
    assert exhausted["chosen_updates"] == 90000
    assert "0" in exhausted["reason"]
    assert longer_budget_start({"_last_lr": [1e-5]}, True)["start"] == "resume"
    assert longer_budget_start({"_last_lr": [1e-4]}, False)["start"] == "scratch"


def test_window_gap_index_requires_both_checkpoints_and_the_same_draw():
    protocol = {"sample_seed": 2026102501}
    identity = {"files_sha256": "abc"}
    decision = {"rule": "pre-registered window gap", "next": "longer_budget"}
    entry = {"epsilon": 0.002, "draw_sha256": "draw", "sample_sha256": "sample"}
    splits = {
        "shared100": {"draw_sha256": "draw", "sample_sha256": "sample", "obstacle100": dict(entry), "obstacle300": dict(entry)},
        "dev30": {"draw_sha256": "devdraw", "sample_sha256": "devsample", "obstacle100": dict(entry, draw_sha256="devdraw"), "obstacle300": dict(entry, draw_sha256="devdraw")},
        "train300": {"draw_sha256": "train", "sample_sha256": "trainsample", "obstacle300": dict(entry)},
    }
    index = build_window_gap_index(splits, protocol, identity, decision)
    assert index["prose_only"] is False
    assert index["source_identity"]["files_sha256"] == "abc"
    broken = json.loads(json.dumps(splits))
    broken["shared100"]["obstacle300"]["draw_sha256"] = "other"
    with pytest.raises(ValueError, match="timesteps"):
        build_window_gap_index(broken, protocol, identity, decision)
    missing = json.loads(json.dumps(splits))
    del missing["dev30"]["obstacle100"]
    with pytest.raises(ValueError, match="obstacle100"):
        build_window_gap_index(missing, protocol, identity, decision)


def test_window_gap_preregistration_freezes_both_checkpoints_and_forbids_training():
    payload = build_window_gap_preregistration("window-gap", {"exp": "outputs/exp/x", "diagnosis": "outputs/exp/x/diagnosis"})
    assert payload["kind"] == "authorized_window_gap_diagnosis"
    assert payload["no_collection"] is True and payload["no_training"] is True
    assert payload["formal_performance_claim"] is False
    assert payload["checkpoints"]["obstacle100"]["sha256"] == "9ca9022aee201ae7f9ffe702d941863d6ffd4e4c6056f220fcdb2d2fe23c830d"
    assert payload["checkpoints"]["obstacle300"]["sha256"] == "bd7757122b75ad0bcc9cc5bd88a4b5174d48dfc520d1c8459332eea157732801"
    assert payload["protocol"]["sample_seed"] == 2026102501
    assert payload["protocol"]["diffusion_timesteps"] == 100
    assert payload["base_checkpoint_sha256"] == "dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d"


def test_diagnose_window_gap_refuses_identity_mismatch_before_checkpoints(tmp_path):
    payload = build_window_gap_preregistration("window-gap", {"exp": str(tmp_path), "diagnosis": str(tmp_path / "diagnosis")})
    payload["source_identity"] = {"files_sha256": "not-the-running-tree"}
    path = tmp_path / "preregistration.json"
    write_preregistration(path, payload)
    with pytest.raises(RuntimeError, match="source identity"):
        main(["diagnose-window-gap", "--registration", str(path), "--device", "cpu"])
    assert not (tmp_path / "diagnosis").exists()


def _diagnosis_index(next_slice="longer_budget", start="scratch", last_lr=0.0):
    return {
        "decision": {
            "rule": "pre-registered window gap",
            "next": next_slice,
            "underfit": next_slice == "longer_budget",
            "gap_closed": False,
        },
        "resume_decision": {"start": start, "last_lr": last_lr, "chosen_updates": 90000},
    }


def test_expected_base_updates_stay_30000_when_the_constrain_budget_is_90000():
    assert expected_base_updates({"training": {"total_iters": 30000}}) == 30000
    assert expected_base_updates({"training": {"total_iters": 90000, "base_updates": 30000}}) == 30000


def test_exposure_preregistration_is_scratch_90000_from_the_underfit_diagnosis():
    payload = build_exposure_preregistration(
        "exposure", {"exp": "outputs/exp/x"}, _diagnosis_index(), "outputs/exp/diag/index.json")
    assert payload["kind"] == "authorized_training_optimization"
    assert payload["changed_factor"] == "constrain_updates"
    assert payload["training"]["total_iters"] == 90000
    assert payload["training"]["base_updates"] == 30000
    assert payload["training"]["start"] == "scratch"
    assert payload["training"]["loss_weight"] is None
    assert payload["training"]["proximity_weight"] == "off"
    assert payload["training"]["base_retrained"] is False
    assert payload["no_collection"] is True
    assert payload["base_checkpoint_sha256"] == "dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d"
    assert payload["comparison"]["obstacle100"] == {"task": 26, "safe": 13, "contact": 17, "xy": 3}
    assert payload["comparison"]["obstacle300"] == {"task": 25, "safe": 12, "contact": 18, "xy": 2}
    assert payload["selected_obstacle_ids"] == list(range(1000, 1300))
    with pytest.raises(ValueError, match="underfit"):
        build_exposure_preregistration("x", {"exp": "y"}, _diagnosis_index("closed_loop"), "diag")
    with pytest.raises(ValueError, match="scratch"):
        build_exposure_preregistration("x", {"exp": "y"}, _diagnosis_index(start="resume", last_lr=1e-5), "diag")


def test_exposure_train_refuses_resume_before_any_update(tmp_path):
    payload = build_preregistration(1, "exposure", {"exp": str(tmp_path)})
    payload["kind"] = "authorized_training_optimization"
    payload["changed_factor"] = "constrain_updates"
    payload["source_identity"] = freeze_pilot_identity()
    payload["training"]["total_iters"] = 90000
    payload["training"]["base_updates"] = 30000
    payload["training"]["start"] = "resume"
    payload["training"]["loss_weight"] = None
    payload["train_direct_count"] = 1000
    payload["train_obstacle_count"] = 300
    payload["obstacle_plan_cap"] = 300
    payload["selected_direct_ids"] = list(range(1000))
    payload["selected_obstacle_ids"] = list(range(1000, 1300))
    path = tmp_path / "preregistration.json"
    write_preregistration(path, payload)
    with pytest.raises(RuntimeError, match="scratch"):
        main([
            "train-constrain", "--registration", str(path),
            "--demo-dir", str(tmp_path), "--base-checkpoint", str(tmp_path / "missing.pt"),
            "--run-dir", str(tmp_path / "run"), "--checkpoint-dir", str(tmp_path / "ckpt"),
        ])
