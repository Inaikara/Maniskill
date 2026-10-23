"""Seams for the PickCube transport-detour learnability pilot."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from examples.baselines.constrain.pickcube_learning_pilot import (
    LAYOUT_TOLERANCE_M,
    TRAIN_OBSTACLE_COUNT,
    PilotTrajectoryDataset,
    build_preregistration,
    decide_gate,
    layouts_conflict,
    load_preregistration,
    package_records,
    split_is_independent,
    write_preregistration,
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
