"""Seams for the PlaceSphere transport-detour learnability pilot."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain.placesphere_learning_pilot import (
    LAYOUT_TOLERANCE_M,
    batched_observation,
    build_preregistration,
    coerce_action_sequence,
    compare_ticket02,
    decide_gate,
    dev_premises_hold,
    layouts_conflict,
    load_preregistration,
    package_records,
    split_is_independent,
    trajectory_has_release_retreat_settle,
    write_preregistration,
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
