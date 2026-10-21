"""Validate and retain machine-readable evidence for a wristcam pilot."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.collect_constructive import validate_aggregate
from examples.baselines.constrain.constrain_dataset import ConstrainDemoDataset


def validate(dataset_dir):
    dataset_dir = Path(dataset_dir)
    report = json.loads((dataset_dir / "report.json").read_text())
    metadata = json.loads((dataset_dir / "trajectory.json").read_text())
    h5_path = dataset_dir / "trajectory.h5"
    assert report["complete"] and report["accepted"] == 5
    assert report["distinct_templates"] == 5
    assert len({episode["template_index"] for episode in report["episodes"]}) == 5
    assert all(
        episode["trace"]["stack_success"]
        and not episode["trace"]["obstacle_collision"]
        and not episode["trace"]["obstacle_crossing"]
        for episode in report["episodes"]
    )
    assert all(
        episode["stack_success"]
        and not episode["obstacle_collision"]
        and not episode["obstacle_crossing"]
        for episode in metadata["episodes"]
    )
    validate_aggregate(h5_path, dataset_dir / "trajectory.json", 5)

    episode_evidence = []
    with h5py.File(h5_path, "r") as handle:
        for index, episode in enumerate(metadata["episodes"]):
            trajectory = handle[f"traj_{index}"]
            actions = trajectory["actions"][:]
            observations = trajectory["obs"][:]
            assert actions.shape[1] == 7
            assert observations.shape == (len(actions) + 1, 51)
            assert np.count_nonzero(actions[:, 3:6]) == 0
            episode_evidence.append(
                dict(
                    trajectory=index,
                    steps=len(actions),
                    action_shape=list(actions.shape),
                    observation_shape=list(observations.shape),
                    rotation_max_abs=float(np.abs(actions[:, 3:6]).max()),
                    stack_success=episode["stack_success"],
                    obstacle_collision=episode["obstacle_collision"],
                    obstacle_crossing=episode["obstacle_crossing"],
                    template_index=episode["template_index"],
                )
            )

    dataset = ConstrainDemoDataset(h5_path)
    assert len(dataset.trajectory_lengths) == 5
    assert dataset[0]["observations"].shape == (2, 51)
    assert dataset[0]["actions"].shape == (16, 4)
    evidence = dict(
        status="PASS",
        episodes=episode_evidence,
        aggregate_state_validation="PASS: every env_state leaf has T+1 rows",
        dataset_trajectory_lengths=dataset.trajectory_lengths,
        dataset_windows=len(dataset),
        dataset_observation_shape=list(dataset[0]["observations"].shape),
        dataset_action_shape=list(dataset[0]["actions"].shape),
    )
    (dataset_dir / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_dir), indent=2))


if __name__ == "__main__":
    main()
