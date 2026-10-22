import h5py
import numpy as np
import pytest

from examples.baselines.constrain.pickcube_pilot import (
    PickCubePilotConfig,
    PickCubePilotDataset,
    collect_pilot,
)


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
