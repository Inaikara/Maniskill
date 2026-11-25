import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest

from examples.baselines.constrain.peg_orientation_data import (
    assign_hint_targets,
    build_hint_for_mode,
    combine_random_training_datasets,
    fixed_training_reservation_supplement,
    freeze_split_plan,
    preinsertion_yaw_diagnostic,
    write_training_coverage,
)


def test_frozen_splits_use_disjoint_seed_ranges_and_keep_test_cohort_fixed():
    plan = freeze_split_plan()

    assert plan["random_test"]["seeds"] == list(range(2026112300, 2026112400))
    assert len(plan["fixed_retention"]["seeds"]) == 100
    assert len(plan["validation_random"]["seeds"]) == 50
    assert len(plan["validation_fixed"]["seeds"]) == 50
    assert len(plan["train_fixed_future_seeds"]) == 1000
    assert plan["train_fixed_future_seeds"] == list(range(2026112500, 2026113500))
    all_seeds = [seed for split in plan.values() if isinstance(split, dict)
                 for seed in split["seeds"]]
    assert len(all_seeds) == len(set(all_seeds))
    assert min(all_seeds) > 2026112099
    assert not set(plan["train_fixed_future_seeds"]) & set(all_seeds)


def test_fixed_training_reservation_is_additive_and_disjoint():
    supplement = fixed_training_reservation_supplement()

    assert supplement["supersedes"]["prior_fixed_training_seeds"] == list(
        range(2026112500, 2026113500)
    )
    assert supplement["fixed_training_candidate_seeds"] == list(
        range(2026112500, 2026116000)
    )
    assert supplement["supersedes"]["sha256"] == (
        "295a4d87114358ac8f5583422e9ef4914761dc2e875c5de64040a50614aea4fb"
    )
    reserved = set(supplement["fixed_training_candidate_seeds"])
    for seeds in supplement["disjoint_from"].values():
        assert not reserved & set(seeds)


def test_hint_target_assignment_is_episode_consistent_and_precomputable():
    target_yaws = np.linspace(1.2, 1.9, 8)
    assignment = assign_hint_targets(target_yaws, seed=1703)

    assert assignment["rule"] == "seeded_permutation_of_test_targets_v1"
    assert assignment["seed"] == 1703
    assert sorted(assignment["target_yaws"]) == pytest.approx(sorted(target_yaws))
    assert assign_hint_targets(target_yaws, seed=1703) == assignment
    assert any(not np.isclose(a, b) for a, b in zip(target_yaws, assignment["target_yaws"]))


def test_correct_standard_and_shuffled_hints_recompute_live_relative_yaw():
    peg_yaw = np.deg2rad(110)

    np.testing.assert_allclose(
        build_hint_for_mode("correct", peg_yaw, np.deg2rad(100)),
        [np.sin(np.deg2rad(10)), np.cos(np.deg2rad(10))], atol=1e-7,
    )
    np.testing.assert_allclose(
        build_hint_for_mode("fixed_standard", peg_yaw, np.deg2rad(100)),
        [np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20))], atol=1e-7,
    )
    shuffled = build_hint_for_mode("shuffled", peg_yaw, np.deg2rad(100), np.deg2rad(80))
    np.testing.assert_allclose(shuffled, [np.sin(np.deg2rad(30)), np.cos(np.deg2rad(30))], atol=1e-7)


def test_preinsertion_yaw_diagnostic_records_trigger_and_never_reached():
    before = preinsertion_yaw_diagnostic(
        peg_position=np.array([0.2, 0.0, 0.1]), hole_center=np.zeros(3),
        peg_yaw=0.8, target_yaw=1.2, trigger_distance_m=0.1,
        step_index=7, terminal=True, native_success=False, failure_type="no_grasp",
    )
    reached = preinsertion_yaw_diagnostic(
        peg_position=np.array([0.05, 0.0, 0.1]), hole_center=np.zeros(3),
        peg_yaw=0.8, target_yaw=1.2, trigger_distance_m=0.1,
    )

    assert before["status"] == "not_reached"
    assert before["sample_step"] is None and before["terminal"]
    assert before["native_success"] is False and before["failure_type"] == "no_grasp"
    assert reached["status"] == "sampled"
    assert reached["yaw_error_rad"] == pytest.approx(-0.4)


def test_combined_training_package_keeps_source_bytes_and_rejects_duplicate_seeds(tmp_path):
    sources = []
    for name, seeds in (("original", range(50)), ("additional", range(50, 99)),
                        ("recovery", [99])):
        directory = tmp_path / name
        directory.mkdir()
        episodes = [{"episode_id": index, "episode_seed": seed} for index, seed in enumerate(seeds)]
        with h5py.File(directory / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", "w") as handle:
            for index, seed in enumerate(seeds):
                group = handle.create_group(f"traj_{index}")
                group.create_dataset("seed", data=seed)
                group.create_dataset("actions", data=np.asarray([[seed]], dtype=np.float32))
        (directory / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text(json.dumps({
            "env_info": {"env_id": "PegInsertionSideRandomOrientation-v1"}, "episodes": episodes,
        }))
        (directory / "dataset_manifest.json").write_text("{}")
        sources.append(directory)
    source_hash = hashlib.sha256(
        (sources[0] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5").read_bytes()
    ).hexdigest()

    manifest = combine_random_training_datasets(sources, tmp_path / "combined")

    assert manifest["episode_count"] == 100
    assert manifest["all_episode_seeds_unique"]
    assert manifest["provenance_counts"] == {"retained": 50, "new": 50}
    assert manifest["source_datasets"][str(sources[0])]["trajectory_h5_sha256"] == source_hash
    assert hashlib.sha256(
        (sources[0] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5").read_bytes()
    ).hexdigest() == source_hash
    with h5py.File(tmp_path / "combined" / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5") as handle:
        assert len(handle) == 100
        assert handle["traj_99/seed"][()] == 99

    duplicate_sources = [tmp_path / f"duplicate_{index}" for index in range(3)]
    for source, duplicate in zip(sources, duplicate_sources):
        shutil.copytree(source, duplicate)
    with h5py.File(duplicate_sources[2] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", "a") as handle:
        handle["traj_0/actions"][...] = np.asarray([[0]], dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate action trajectory"):
        combine_random_training_datasets(duplicate_sources, tmp_path / "duplicate_combined")


def test_combined_training_package_accepts_fixed_base_count_of_1000(tmp_path):
    sources = []
    for name, seeds in (("original", range(50)), ("additional", range(50, 1000))):
        directory = tmp_path / name
        directory.mkdir()
        episodes = [{"episode_id": index, "episode_seed": seed}
                    for index, seed in enumerate(seeds)]
        with h5py.File(directory / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", "w") as handle:
            for index, seed in enumerate(seeds):
                group = handle.create_group(f"traj_{index}")
                group.create_dataset("actions", data=np.asarray([[seed]], dtype=np.float32))
        (directory / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text(json.dumps({
            "env_info": {"env_id": "PegInsertionSideOrientation-v1"}, "episodes": episodes,
        }))
        (directory / "dataset_manifest.json").write_text("{}")
        sources.append(directory)

    manifest = combine_random_training_datasets(
        sources, tmp_path / "fixed_combined", expected_count=1000,
    )

    assert manifest["episode_count"] == 1000
    assert len(manifest["selected_seeds"]) == len(set(manifest["selected_seeds"])) == 1000
    assert manifest["provenance_counts"] == {"retained": 50, "new": 950}
    with h5py.File(tmp_path / "fixed_combined" / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5") as handle:
        assert len(handle) == 1000


def test_training_coverage_counts_retained_composite_rows_not_first_primitive(tmp_path):
    sources = []
    for name, seeds in (("old-composite", range(100)), ("new", range(100, 200))):
        directory = tmp_path / name
        directory.mkdir()
        episodes = [{"episode_id": i, "episode_seed": seed} for i, seed in enumerate(seeds)]
        metadata = {"episodes": episodes}
        (directory / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text(json.dumps(metadata))
        (directory / "dataset_manifest.json").write_text(json.dumps({
            "selected_seeds": list(seeds), "episode_count": 100,
            "converted_replay_successes": 100, "collection_attempts": 100,
            "expert_successes": 100, "replay_failures": 0,
            "all_observation_action_hint_alignment": True,
            "all_selected_episodes_have_rotation_control": True,
            "all_selected_episodes_pass_state_restore": True,
            "selected_target_yaws": [1.57] * 100,
            "selected_peg_initial_yaws": [1.0] * 100,
        }))
        (directory / "collection_report.json").write_text(json.dumps({
            "attempts": [{"seed": seed, "hole_center": [0, 0, 0],
                          "peg_half_size": [0.1, 0.1, 0.1], "hole_radius": 0.1}
                         for seed in seeds]
        }))
        sources.append(directory)
    output = tmp_path / "package"
    output.mkdir()
    with h5py.File(output / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", "w") as h5:
        for i in range(200):
            group = h5.create_group(f"traj_{i}")
            group.create_dataset("actions", data=np.asarray([[i]], dtype=np.float32))
    (output / "dataset_manifest.json").write_text("{}")

    coverage = write_training_coverage(sources, output, expected_count=200, retained_count=100)

    assert coverage["newly_collected_episode_count"] == 100
    assert len(coverage["episodes"]) == 200


def test_training_coverage_composes_retained_training_package_with_new_source(tmp_path):
    retained = tmp_path / "retained-200"
    retained.mkdir()
    retained_records = [
        {"source_dataset": "old-source", "seed": seed, "target_yaw_rad": 1.4,
         "peg_initial_yaw_rad": 0.8, "hole_center_m": [0, 0, 0],
         "peg_half_size_m": [0.1, 0.1, 0.1], "hole_radius_m": 0.1}
        for seed in range(200)
    ]
    (retained / "training_coverage.json").write_text(json.dumps({
        "episode_count": 200, "episodes": retained_records,
        "sources": {"old-source": {"episodes": 200}},
    }))
    (retained / "dataset_manifest.json").write_text(json.dumps({
        "episode_count": 200, "selected_seeds": list(range(200)),
    }))
    new = tmp_path / "new-350"
    new.mkdir()
    new_seeds = list(range(1000, 1350))
    episodes = [{"episode_id": i, "episode_seed": seed}
                for i, seed in enumerate(new_seeds)]
    (new / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text(
        json.dumps({"episodes": episodes})
    )
    (new / "dataset_manifest.json").write_text(json.dumps({
        "selected_seeds": new_seeds, "episode_count": 350,
        "converted_replay_successes": 325, "collection_attempts": 544,
        "expert_successes": 350, "replay_failures": 25,
        "all_observation_action_hint_alignment": True,
        "all_selected_episodes_have_rotation_control": True,
        "all_selected_episodes_pass_state_restore": True,
        "selected_target_yaws": [1.5] * 350,
        "selected_peg_initial_yaws": [0.9] * 350,
    }))
    (new / "collection_report.json").write_text(json.dumps({
        "attempts": [{"seed": seed, "hole_center": [0, 0, 0],
                      "peg_half_size": [0.1, 0.1, 0.1], "hole_radius": 0.1}
                     for seed in new_seeds],
    }))
    sources = [retained, new]
    output = tmp_path / "package-500"
    output.mkdir()
    with h5py.File(output / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", "w") as h5:
        for i in range(500):
            group = h5.create_group(f"traj_{i}")
            group.create_dataset("actions", data=np.asarray([[i]], dtype=np.float32))
    (output / "dataset_manifest.json").write_text("{}")

    coverage = write_training_coverage(sources, output, expected_count=500, retained_count=200)

    assert coverage["episode_count"] == 500
    assert coverage["newly_collected_episode_count"] == 300
    assert [row["seed"] for row in coverage["episodes"]] == list(range(200)) + new_seeds[:300]
    assert [row["source_dataset"] for row in coverage["episodes"][:200]] == ["old-source"] * 200
    assert coverage["sources"][new.name]["episodes"] == 300
