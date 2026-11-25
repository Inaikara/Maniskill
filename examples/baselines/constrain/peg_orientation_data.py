"""Frozen train/validation/test data and model-independent Peg scene catalogs."""

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from transforms3d.quaternions import quat2mat

from examples.baselines.constrain import peg_orientation
from examples.baselines.constrain.scene_set import _read_state, _stable_hash


SPLIT_PLAN = {
    "format": "peg-orientation-splits-v1",
    "train_random_existing_seeds": list(range(2026112000, 2026112100)),
    "train_random_additional_seeds": list(range(2026112100, 2026112200)),
    "train_fixed_future_seeds": list(range(2026112500, 2026113500)),
    "validation_random_seeds": list(range(2026112200, 2026112250)),
    "validation_fixed_seeds": list(range(2026112250, 2026112300)),
    "random_test_seeds": list(range(2026112300, 2026112400)),
    "fixed_retention_seeds": list(range(2026112400, 2026112500)),
}
SPLITS = {
    "random_test": SPLIT_PLAN["random_test_seeds"],
    "fixed_retention": SPLIT_PLAN["fixed_retention_seeds"],
    "validation_random": SPLIT_PLAN["validation_random_seeds"],
    "validation_fixed": SPLIT_PLAN["validation_fixed_seeds"],
}
FIXED_TRAINING_RESERVATION_V2 = {
    "format": "peg-fixed-training-reservation-v2",
    "supersedes": {
        "path": "outputs/exp/20261111_172604_peg_orientation_adaptation/frozen_splits.json",
        "sha256": "295a4d87114358ac8f5583422e9ef4914761dc2e875c5de64040a50614aea4fb",
        "prior_fixed_training_seeds": list(range(2026112500, 2026113500)),
    },
    "retained_scene_catalogs": {
        "random_test": {
            "path": "outputs/exp/20261111_172604_peg_orientation_adaptation/random_test_scenes",
            "scenes_sha256": "ee659401e16a9533b83843a5147877d2881d0b5eae15f0654c1542a844eef9c7",
            "manifest_sha256": "ecdaa4fe3977132740bf0141cb64d633c264887dd03fc767e9afbe78e05043b6",
        },
        "validation_random": {
            "path": "outputs/exp/20261111_172604_peg_orientation_adaptation/validation_random_scenes",
            "scenes_sha256": "cdc3f1ee5cc7fc36f4884158c21c86fd238037e87e07b64bcafd427af3ae5d89",
            "manifest_sha256": "a2d2b7c8a0f5981f9174bfe9c3ef1ef56469ca1edc61738fce70dcfb43d4f3bb",
        },
        "validation_fixed": {
            "path": "outputs/exp/20261111_172604_peg_orientation_adaptation/validation_fixed_scenes",
            "scenes_sha256": "bfd31085fb9ba2cc283c75b3c9d89342defab6ed06fb42452b493db10b894a37",
            "manifest_sha256": "7d616d19e2ad91113ede63d9d43a3ca8d712d41369cdf5ba7263c86c4ccc3204",
        },
        "fixed_retention": {
            "path": "outputs/exp/20261111_172604_peg_orientation_adaptation/fixed_retention_scenes",
            "scenes_sha256": "81d9ae6d15a4aa0e87cb61cc9e40b35a27d41e2f772ded34d669a0b84759ffb3",
            "manifest_sha256": "164dd035c616b4be2b454c3cbbbf7b4e95c96b89aec34cf49766b24ee8984c06",
        },
    },
    "fixed_training_candidate_seeds": list(range(2026112500, 2026116000)),
    "disjoint_from": {
        "initial_fixed_candidates": list(range(2026112000, 2026112100)),
        "random_training_candidates": list(range(2026112000, 2026112200)),
        "validation_random_seeds": SPLIT_PLAN["validation_random_seeds"],
        "validation_fixed_seeds": SPLIT_PLAN["validation_fixed_seeds"],
        "random_test_seeds": SPLIT_PLAN["random_test_seeds"],
        "fixed_retention_seeds": SPLIT_PLAN["fixed_retention_seeds"],
    },
    "chronology": "Versioned supplement created before fixed-base collection; the original frozen split file and its scene catalogs are retained unchanged.",
}


def freeze_split_plan():
    """Return the preregistered disjoint seed catalog, excluding training rows."""
    splits = {name: {"seeds": list(seeds)} for name, seeds in SPLITS.items()}
    flattened = [seed for split in splits.values() for seed in split["seeds"]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("frozen evaluation split seeds overlap")
    return {**SPLIT_PLAN, **splits}


def fixed_training_reservation_supplement():
    """Return the additive fixed-training reserve without rewriting frozen splits."""
    payload = dict(FIXED_TRAINING_RESERVATION_V2)
    reserved = payload["fixed_training_candidate_seeds"]
    excluded = [seed for values in payload["disjoint_from"].values() for seed in values]
    if set(reserved) & set(excluded):
        raise ValueError("fixed training candidate reservation overlaps a retained split")
    return payload


def assign_hint_targets(target_yaws, seed):
    """Freeze a deterministic, episode-level permutation for shuffled hints."""
    target_yaws = np.asarray(target_yaws, dtype=np.float64)
    permutation = np.random.default_rng(seed).permutation(len(target_yaws))
    return {
        "rule": "seeded_permutation_of_test_targets_v1", "seed": int(seed),
        "permutation": permutation.tolist(),
        "target_yaws": target_yaws[permutation].tolist(),
    }


def preinsertion_yaw_diagnostic(peg_position, hole_center, peg_yaw, target_yaw,
                                 trigger_distance_m=0.1, step_index=None, terminal=False,
                                 native_success=None, failure_type=None):
    """Sample the first pre-insertion proximity and finalize terminal non-reaches."""
    distance = float(np.linalg.norm(np.asarray(peg_position)[:2] - np.asarray(hole_center)[:2]))
    if distance > trigger_distance_m:
        return {"status": "not_reached", "distance_m": distance,
                "trigger_distance_m": trigger_distance_m, "yaw_error_rad": None,
                "sample_step": None, "terminal": bool(terminal),
                "native_success": native_success, "failure_type": failure_type}
    error = float(np.arctan2(np.sin(peg_yaw - target_yaw), np.cos(peg_yaw - target_yaw)))
    return {"status": "sampled", "distance_m": distance,
            "trigger_distance_m": trigger_distance_m, "yaw_error_rad": error,
            "sample_step": step_index, "terminal": bool(terminal),
            "native_success": native_success, "failure_type": failure_type}


def _pose_yaw(pose):
    axis = quat2mat(np.asarray(pose.q[0].detach().cpu()))[:, 0]
    return float(np.arctan2(axis[1], axis[0]))


def _state_to_numpy(value):
    if isinstance(value, dict):
        return {key: _state_to_numpy(child) for key, child in value.items()}
    return np.asarray(value[0].detach().cpu())


def build_scene_catalog(output_dir, split_name, reset_options=None):
    """Capture complete initial physical states for a preregistered scene split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(peg_orientation.RANDOM_ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", reconfiguration_freq=1)
    manifest = {"format": "peg-orientation-scenes-v1", "split": split_name,
                "scene_count": len(SPLITS[split_name]), "scenes_file": "scenes.h5", "scenes": []}
    try:
        with h5py.File(output_dir / "scenes.h5", "w") as handle:
            for index, seed in enumerate(SPLITS[split_name]):
                options = dict(reset_options or {})
                if split_name in ("fixed_retention", "validation_fixed"):
                    options["target_yaw"] = peg_orientation.TARGET_YAW
                observation, _ = env.reset(seed=seed, options=options)
                task = env.unwrapped
                state = _state_to_numpy(task.get_state_dict())
                scene_id = f"peg-{split_name}-{index:03d}"
                group = handle.create_group(scene_id)
                _write_state(group.create_group("initial_state"), state)
                group.create_dataset("initial_observation", data=observation.detach().cpu().numpy()[0])
                hole = task.box_hole_pose.p[0].detach().cpu().numpy()
                peg = task.peg.pose.p[0].detach().cpu().numpy()
                half_size = task.peg_half_sizes[0].detach().cpu().numpy()
                manifest["scenes"].append({
                    "scene_id": scene_id, "seed": seed, "target_yaw_rad": float(task.target_yaw),
                    "peg_initial_yaw_rad": _pose_yaw(task.peg.pose),
                    "peg_position_m": peg.tolist(), "hole_center_m": hole.tolist(),
                    "peg_half_size_m": half_size.tolist(),
                    "hole_radius_m": float(task.box_hole_radii[0].detach().cpu()),
                    "initial_state_sha256": _stable_hash(state),
                })
    finally:
        env.close()
    manifest["scenes_file_sha256"] = _sha256(output_dir / "scenes.h5")
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _write_state(group, state):
    for key, value in state.items():
        if isinstance(value, dict):
            _write_state(group.create_group(key), value)
        else:
            group.create_dataset(key, data=np.asarray(value)[None])


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PegSceneCatalog:
    """Read-only, checksum-verified initial states loadable without a model."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.path = self.directory / self.manifest["scenes_file"]
        if _sha256(self.path) != self.manifest["scenes_file_sha256"]:
            raise ValueError("scene data checksum mismatch")
        recorded = self.manifest["manifest_sha256"]
        payload = {key: value for key, value in self.manifest.items() if key != "manifest_sha256"}
        if _canonical_hash(payload) != recorded:
            raise ValueError("scene manifest checksum mismatch")

    def __len__(self):
        return self.manifest["scene_count"]

    def load(self, index):
        metadata = self.manifest["scenes"][index]
        with h5py.File(self.path, "r") as handle:
            group = handle[metadata["scene_id"]]
            state = _read_state(group["initial_state"])
            observation = np.asarray(group["initial_observation"])
        if _stable_hash(state) != metadata["initial_state_sha256"]:
            raise ValueError(f"scene state checksum mismatch: {metadata['scene_id']}")
        return metadata, state, observation


def restore_peg_scene(env, scene):
    """Reset the native task, restore one catalog state, and return its observation."""
    metadata, state, expected = scene
    env.reset(seed=int(metadata["seed"]))
    tensor_state = _to_torch_state(state, env.unwrapped.device)
    env.unwrapped.set_state_dict(tensor_state)
    observation = env.unwrapped.get_obs()
    actual = observation.detach().cpu().numpy().reshape(-1)
    if not np.allclose(actual, expected.reshape(-1), atol=1e-6, rtol=0):
        raise ValueError(f"restored observation mismatch: {metadata['scene_id']}")
    return observation


def _to_torch_state(state, device):
    if isinstance(state, dict):
        return {key: _to_torch_state(value, device) for key, value in state.items()}
    return torch.as_tensor(state, device=device).unsqueeze(0)


def build_hint_for_mode(mode, peg_yaw, actual_target_yaw, shuffled_target_yaw=None):
    """Build live relative-yaw hint with one episode-constant target assignment."""
    if mode == "correct":
        target_yaw = actual_target_yaw
    elif mode == "fixed_standard":
        target_yaw = peg_orientation.TARGET_YAW
    elif mode == "shuffled" and shuffled_target_yaw is not None:
        target_yaw = shuffled_target_yaw
    elif mode == "shuffled":
        raise ValueError("shuffled mode requires the scene's preregistered pseudo-target yaw")
    else:
        raise ValueError(f"unknown hint mode: {mode}")
    return peg_orientation.build_orientation_hint(peg_yaw, target_yaw)


def combine_random_training_datasets(source_dirs, output_dir, expected_count=100):
    """Create a new fixed-size package without changing any source dataset."""
    source_dirs = [Path(directory) for directory in source_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    filename = "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    json_name = "trajectory.state.pd_ee_delta_pose.physx_cpu.json"
    source_metadata = [json.loads((directory / json_name).read_text()) for directory in source_dirs]
    source_hashes = {str(directory): {
        "trajectory_h5_sha256": _sha256(directory / filename),
        "trajectory_json_sha256": _sha256(directory / json_name),
        "dataset_manifest_sha256": _sha256(directory / "dataset_manifest.json"),
    } for directory in source_dirs}
    selected_rows = []
    retained_count = 0
    for metadata in source_metadata:
        remaining = expected_count - len(selected_rows)
        selected = metadata["episodes"][:remaining]
        if not selected_rows:
            retained_count = len(selected)
        selected_rows.extend(selected)
        if len(selected_rows) == expected_count:
            break
    seeds = [int(row["episode_seed"]) for row in selected_rows]
    if len(seeds) != expected_count or len(seeds) != len(set(seeds)):
        raise ValueError(f"source datasets must contain {expected_count} distinct episode seeds")
    combined = dict(source_metadata[0])
    combined["episodes"] = []
    trajectory_hashes = set()
    with h5py.File(output_dir / filename, "w") as target:
        for directory, metadata in zip(source_dirs, source_metadata):
            if len(combined["episodes"]) == expected_count:
                break
            with h5py.File(directory / filename, "r") as source:
                remaining = expected_count - len(combined["episodes"])
                for row in metadata["episodes"][:remaining]:
                    source_key = f"traj_{row['episode_id']}"
                    actions = np.ascontiguousarray(source[source_key]["actions"][:])
                    trajectory_hash = hashlib.sha256(actions.tobytes()).hexdigest()
                    if trajectory_hash in trajectory_hashes:
                        raise ValueError(f"duplicate action trajectory in {directory}")
                    trajectory_hashes.add(trajectory_hash)
                    target_key = f"traj_{len(combined['episodes'])}"
                    source.copy(source_key, target, name=target_key)
                    copied = dict(row)
                    copied["episode_id"] = len(combined["episodes"])
                    copied["source_dataset"] = directory.name
                    combined["episodes"].append(copied)
    json_path = output_dir / json_name
    json_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    manifest = {
        "dataset": output_dir.name, "episode_count": len(seeds),
        "control_mode": "pd_ee_delta_pose", "observation_mode": "state",
        "sim_backend": "physx_cpu", "robot_uids": "panda_wristcam",
        "source_datasets": source_hashes, "selected_seeds": seeds,
        "all_episode_seeds_unique": True,
        "all_action_trajectories_unique": True,
        "trajectory_h5_sha256": _sha256(output_dir / filename),
        "trajectory_json_sha256": _sha256(json_path),
        "provenance_counts": {"retained": retained_count, "new": len(seeds) - retained_count},
        "provenance": (
            f"first {retained_count} retained unchanged by identity, followed by "
            f"{len(seeds) - retained_count} new officially replay-verified episodes"
        ),
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_training_coverage(source_dirs, output_dir, expected_count=100, retained_count=None):
    """Record per-episode angle and geometry coverage for replay-verified rows."""
    source_dirs, output_dir = [Path(path) for path in source_dirs], Path(output_dir)
    records = []
    source_counts = {}
    for directory in source_dirs:
        if len(records) == expected_count:
            break
        manifest = json.loads((directory / "dataset_manifest.json").read_text())
        remaining = expected_count - len(records)
        existing_coverage = directory / "training_coverage.json"
        if existing_coverage.exists():
            coverage = json.loads(existing_coverage.read_text())
            selected = coverage["episodes"][:remaining]
            selected_seeds = [int(row["seed"]) for row in selected]
            if (coverage["episode_count"] != len(coverage["episodes"])
                    or manifest["selected_seeds"][:len(selected)] != selected_seeds):
                raise ValueError(f"composite training coverage does not match {directory}")
            records.extend(selected)
            source_counts[directory.name] = {
                "episodes": len(selected), "composed_from_coverage": True,
                "sources": coverage["sources"],
            }
            continue
        collection = json.loads((directory / "collection_report.json").read_text())
        attempts = {int(row["seed"]): row for row in collection["attempts"]}
        selected = manifest["selected_seeds"][:remaining]
        expected_selected = min(manifest["episode_count"], remaining)
        if len(selected) != expected_selected or not (
            manifest["all_observation_action_hint_alignment"]
            and manifest["all_selected_episodes_have_rotation_control"]
            and manifest["all_selected_episodes_pass_state_restore"]
        ):
            raise ValueError(f"selected replay audit incomplete for {directory}")
        source_counts[directory.name] = {
            "episodes": len(selected), "official_replay_successes": manifest["converted_replay_successes"],
            "collection_attempts": manifest["collection_attempts"],
            "expert_successes": manifest["expert_successes"],
            "replay_failures": manifest["replay_failures"],
        }
        for index, seed in enumerate(selected):
            attempt = attempts[int(seed)]
            records.append({
                "source_dataset": directory.name, "seed": int(seed),
                "target_yaw_rad": manifest["selected_target_yaws"][index],
                "peg_initial_yaw_rad": manifest["selected_peg_initial_yaws"][index],
                "hole_center_m": attempt["hole_center"],
                "peg_half_size_m": attempt["peg_half_size"], "hole_radius_m": attempt["hole_radius"],
            })
    if (len(records) != expected_count
            or len({row["seed"] for row in records}) != expected_count):
        raise ValueError(f"training sources must resolve to {expected_count} unique valid episodes")
    combined_path = output_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    trajectory_hashes = set()
    with h5py.File(combined_path, "r") as handle:
        if len(handle) != expected_count:
            raise ValueError(f"combined training HDF5 must contain exactly {expected_count} trajectories")
        for key in sorted(handle, key=lambda value: int(value.split("_")[1])):
            actions = np.ascontiguousarray(handle[key]["actions"][:])
            identity = hashlib.sha256(actions.tobytes()).hexdigest()
            if identity in trajectory_hashes:
                raise ValueError(f"duplicate action trajectory: {key}")
            trajectory_hashes.add(identity)
    yaws = np.asarray([row["target_yaw_rad"] for row in records])
    if np.any(yaws < peg_orientation.TARGET_YAW - peg_orientation.YAW_HALF_RANGE) or np.any(
        yaws > peg_orientation.TARGET_YAW + peg_orientation.YAW_HALF_RANGE
    ):
        raise ValueError("a selected training target yaw falls outside the adaptation range")

    def ranges(values):
        values = np.asarray(values)
        return [[float(values[:, axis].min()), float(values[:, axis].max())]
                for axis in range(values.shape[1])]

    if retained_count is None:
        retained_count = source_counts[next(iter(source_counts))]["episodes"]
    if not 0 <= retained_count <= expected_count:
        raise ValueError("retained_count must be between zero and expected_count")
    coverage = {
        "episode_count": len(records),
        "newly_collected_episode_count": len(records) - retained_count,
        "angle_units": "radians", "geometry_units": "meters",
        "target_yaw_range_rad": [float(yaws.min()), float(yaws.max())],
        "target_yaw_histogram_5_bins": np.histogram(
            yaws, bins=np.linspace(peg_orientation.TARGET_YAW - peg_orientation.YAW_HALF_RANGE,
                                   peg_orientation.TARGET_YAW + peg_orientation.YAW_HALF_RANGE, 6)
        )[0].tolist(),
        "peg_initial_yaw_range_rad": [
            min(row["peg_initial_yaw_rad"] for row in records),
            max(row["peg_initial_yaw_rad"] for row in records),
        ],
        "hole_center_axis_ranges_m": ranges([row["hole_center_m"] for row in records]),
        "peg_half_size_axis_ranges_m": ranges([row["peg_half_size_m"] for row in records]),
        "hole_radius_range_m": [min(row["hole_radius_m"] for row in records),
                                max(row["hole_radius_m"] for row in records)],
        "sources": source_counts, "episodes": records,
    }
    path = output_dir / "training_coverage.json"
    path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    dataset_manifest_path = output_dir / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    dataset_manifest["training_coverage_sha256"] = _sha256(path)
    dataset_manifest["all_action_trajectories_unique"] = True
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n")
    return coverage


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-splits")
    freeze.add_argument("--output", required=True)
    fixed_reserve = sub.add_parser("freeze-fixed-reservation")
    fixed_reserve.add_argument("--output", required=True)
    scenes = sub.add_parser("build-scenes")
    scenes.add_argument("--split", choices=SPLITS, required=True)
    scenes.add_argument("--output-dir", required=True)
    combine = sub.add_parser("combine-training")
    combine.add_argument("--source-dirs", nargs="+", required=True)
    combine.add_argument("--output-dir", required=True)
    combine.add_argument("--expected-count", type=int, default=100)
    hints = sub.add_parser("freeze-hints")
    hints.add_argument("--scenes", required=True)
    hints.add_argument("--output", required=True)
    hints.add_argument("--seed", type=int, default=1703)
    audit = sub.add_parser("audit-training")
    audit.add_argument("--source-dirs", nargs="+", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--expected-count", type=int, default=100)
    audit.add_argument("--retained-count", type=int)
    args = parser.parse_args()
    if args.command == "freeze-splits":
        Path(args.output).write_text(json.dumps(freeze_split_plan(), indent=2, sort_keys=True) + "\n")
    elif args.command == "freeze-fixed-reservation":
        Path(args.output).write_text(json.dumps(
            fixed_training_reservation_supplement(), indent=2, sort_keys=True,
        ) + "\n")
    elif args.command == "build-scenes":
        build_scene_catalog(args.output_dir, args.split)
    elif args.command == "combine-training":
        combine_random_training_datasets(args.source_dirs, args.output_dir, args.expected_count)
    elif args.command == "freeze-hints":
        scenes_manifest = json.loads((Path(args.scenes) / "manifest.json").read_text())
        yaw_values = [row["target_yaw_rad"] for row in scenes_manifest["scenes"]]
        assignment = assign_hint_targets(yaw_values, args.seed)
        payload = {
            "format": "peg-orientation-hint-interventions-v1",
            "correct": "read actual target yaw per episode and compute wrapped current peg-minus-target yaw at every observation",
            "fixed_standard": "substitute 90 degrees as target yaw and recompute current relative yaw at every observation",
            "shuffled": "use the frozen per-episode pseudo-target yaw permutation; hold it constant for the episode",
            "shuffled_assignment": assignment,
            "main_metric": "native info.success; orientation values are secondary diagnostics",
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        write_training_coverage(args.source_dirs, args.output_dir, args.expected_count, args.retained_count)


if __name__ == "__main__":
    main()
