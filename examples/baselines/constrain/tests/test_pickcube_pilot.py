import h5py
import numpy as np
import pytest

from examples.baselines.constrain.pickcube_pilot import (
    PickCubePilotConfig,
    PickCubePilotDataset,
    _observation_matches_except_obstacle,
    _same_initial_except_obstacle,
    _value_hash,
    _safe_cylinder_candidates,
    collect_pilot,
)
from examples.baselines.constrain.pickcube_provenance import base_line_distance, canonical_json_hash


def test_provenance_canonical_hash_is_order_independent_and_mutation_sensitive():
    assert canonical_json_hash({"a": 1, "b": [2]}) == canonical_json_hash({"b": [2], "a": 1})
    assert canonical_json_hash({"a": 1}) != canonical_json_hash({"a": 2})


def test_base_line_distance_handles_degenerate_segment_and_ordinary_perpendicular():
    assert base_line_distance([0, 0], [0, 0], [3, 4]) == 5
    assert base_line_distance([0, 0], [2, 0], [1, 3]) == 3


def test_dataset_keeps_native_seven_dimensional_actions_and_validates_t_plus_one(tmp_path):
    path = tmp_path / "trajectory.h5"
    with h5py.File(path, "w") as handle:
        traj = handle.create_group("traj_0")
        traj.create_dataset("obs", data=np.arange(5 * 9, dtype=np.float32).reshape(5, 9))
        traj.create_dataset("actions", data=np.arange(4 * 7, dtype=np.float32).reshape(4, 7))
        traj.create_dataset("success", data=np.ones(5, dtype=np.uint8))
        traj.create_dataset("terminated", data=np.array([0, 0, 0, 0, 1], dtype=np.uint8))
        traj.create_dataset("truncated", data=np.zeros(5, dtype=np.uint8))
        states = traj.create_group("env_states")
        states.create_dataset("native", data=np.zeros((5, 3), dtype=np.float32))
    dataset = PickCubePilotDataset(path, obs_horizon=2, pred_horizon=3)
    sample = dataset[0]
    assert sample["observations"].shape == (2, 9)
    assert sample["actions"].shape == (3, 7)
    np.testing.assert_array_equal(sample["actions"][2], np.arange(7, 14, dtype=np.float32))


def test_collector_bounds_complete_attempts_and_records_failed_attempt_reason(tmp_path):
    calls = []

    def fake_attempt(kind, seed, rng, config):
        calls.append((kind, seed))
        return None, "native_timeout"

    report = collect_pilot(
        tmp_path / "pilot",
        PickCubePilotConfig(target_per_class=1, max_attempts_per_class=2),
        attempt_runner=fake_attempt,
    )
    assert len(calls) == 4
    assert report["classes"]["direct"]["complete_attempts"] == 2
    assert report["classes"]["direct"]["failures"] == {"native_timeout": 2}
    assert not report["complete"]
    assert not (tmp_path / "pilot" / "trajectory.h5").exists()


def test_collector_refuses_existing_output_before_running_attempt(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "report.json").write_text("{}")
    with pytest.raises(FileExistsError):
        collect_pilot(output, PickCubePilotConfig(), attempt_runner=lambda *args: pytest.fail())


@pytest.mark.parametrize("config", [
    PickCubePilotConfig(max_attempts_per_class=0),
    PickCubePilotConfig(target_per_class=1.5),
    PickCubePilotConfig(max_wall_seconds=float("inf")),
    PickCubePilotConfig(max_wall_seconds=0),
])
def test_collector_rejects_invalid_or_unbounded_budgets_before_creating_output(tmp_path, config):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        collect_pilot(output, config, attempt_runner=lambda *args: pytest.fail())
    assert not output.exists()


def test_restore_identity_ignores_only_obstacle_condition_and_rejects_robot_perturbation():
    baseline = {
        "actors": {"robot": np.array([[1.0, 2.0]]), "cube": np.array([[3.0]])},
        "pickcube_obstacle": np.array([[0.0, 0.2, 0.03, 1, 0, 0, 0, 0, 0]])
    }
    restored = {
        "actors": {"robot": np.array([[1.0, 2.0]]), "cube": np.array([[3.0]])},
        "pickcube_obstacle": np.array([[0.05, 0.2, 0.03, 1, 0, 0, 0, 0, 0]])
    }
    assert _same_initial_except_obstacle(baseline, restored)
    assert _value_hash(baseline, ignore_obstacle_pose=True) == _value_hash(
        restored, ignore_obstacle_pose=True
    )
    restored["actors"]["robot"][0, 0] += 0.01
    assert not _same_initial_except_obstacle(baseline, restored)
    assert _value_hash(baseline, ignore_obstacle_pose=True) != _value_hash(
        restored, ignore_obstacle_pose=True
    )


def test_cylinder_candidates_require_measured_clearance_from_entire_robot_sweep():
    candidates = _safe_cylinder_candidates(
        np.array([True, True, True]),
        np.array([0.08, 0.09, 0.10]),
        np.array([0.017, 0.029, 0.042]),
        required_robot_clearance=0.03,
        required_cube_clearance=0.063,
    )
    np.testing.assert_array_equal(candidates, [2])


def test_observation_restore_compares_named_native_fields_except_obstacle_condition():
    template = {"agent": {"qpos": np.array([1.0]), "is_grasped": np.array([0])},
                "extra": {"goal": np.array([0.2]), "pickcube_obstacle": np.array([0.0, 0.2, 0.03])}}
    restored = {"agent": {"qpos": np.array([1.0]), "is_grasped": np.array([0])},
                "extra": {"goal": np.array([0.2]), "pickcube_obstacle": np.array([0.1, 0.2, 0.03])}}
    assert _observation_matches_except_obstacle(template, restored)
    restored["agent"]["is_grasped"][0] = 1
    assert not _observation_matches_except_obstacle(template, restored)


def test_failure_result_preserves_phase_counters(tmp_path):
    def failed(kind, seed, rng, config):
        return {"failure": "restore_mismatch", "diagnostics": {"first_difference": "actors/cube"},
                "counters": {"geometry_tries": 2, "waypoint_tries": 1,
                             "template_executions": 1, "cylinder_candidates": 525}}, None

    report = collect_pilot(
        tmp_path / "diagnostic", PickCubePilotConfig(target_per_class=1, max_attempts_per_class=1), failed
    )
    attempt = report["attempts"][0]
    assert attempt["failure"] == "restore_mismatch"
    assert attempt["diagnostics"]["first_difference"] == "actors/cube"
    assert attempt["counters"] == {
        "geometry_tries": 2, "waypoint_tries": 1,
        "template_executions": 1, "cylinder_candidates": 525
    }


def test_attempt_records_preserve_phase_timings_and_separate_executions(tmp_path):
    def failed(kind, seed, rng, config):
        return {"failure": "contact", "counters": {
                    "template_executions": 1, "obstacle_executions": 1,
                }, "phase_timings": {
                    "geometry": 0.2, "template": 1.5,
                    "cylinder_sampling": 0.1, "obstacle_execution": 2.0,
                }}, None

    report = collect_pilot(
        tmp_path / "timed", PickCubePilotConfig(target_per_class=1, max_attempts_per_class=1), failed
    )
    attempt = report["attempts"][0]
    assert attempt["phase_timings"] == {
        "geometry": 0.2, "template": 1.5,
        "cylinder_sampling": 0.1, "obstacle_execution": 2.0,
    }
    assert attempt["template_executions"] == 1
    assert attempt["obstacle_executions"] == 1
    assert report["classes"]["direct"]["phase_elapsed_seconds"]["obstacle_execution"] == 2.0


def test_accepted_direct_and_obstacle_are_saved_with_full_state_and_hashes(tmp_path):
    def successful(kind, seed, rng, config):
        steps = 4
        return {
            "actions": np.zeros((steps, 7), dtype=np.float32),
            "obs": np.zeros((steps + 1, 9), dtype=np.float32),
            "states": [{"native": np.full((1, 3), i, dtype=np.float32)} for i in range(steps + 1)],
            "terminated": np.array([0, 0, 0, 0, 1], dtype=np.uint8),
            "truncated": np.zeros(steps + 1, dtype=np.uint8),
            "infos": [{"native_success": False}] * steps + [{"native_success": True}],
            "success": True,
            "info": {"native_success": True, "safe_success": True},
        }, None

    output = tmp_path / "accepted"
    report = collect_pilot(output, PickCubePilotConfig(target_per_class=1), successful)
    assert report["complete"]
    dataset = PickCubePilotDataset(output / "trajectory.h5")
    assert len(dataset) == 10
    assert dataset[0]["actions"].shape == (16, 7)
    assert len(report["artifact_sha256"]["trajectory.h5"]) == 64
    assert (output / "artifact_hashes.json").exists()
