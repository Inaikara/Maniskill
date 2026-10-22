"""Full-trajectory structural, training-window and fresh-scene restore audit."""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.pickcube_expert import (
    _set_state_dict_exact,
    make_pickcube_env,
)
from examples.baselines.constrain.pickcube_pilot import (
    PickCubePilotDataset,
    _value_hash,
)


def _restore_like(value, template):
    if isinstance(template, dict):
        return {key: _restore_like(value[key], template[key]) for key in template}
    if torch.is_tensor(template):
        return torch.as_tensor(value, dtype=template.dtype, device=template.device).reshape(template.shape)
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype).reshape(template.shape)
    if isinstance(template, tuple):
        return tuple(value)
    if isinstance(template, list):
        return list(value)
    return type(template)(value)


def validate(dataset_dir, output_dir):
    root, output = Path(dataset_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    episodes = json.loads((root / "trajectory.json").read_text())["episodes"]
    classes = {kind: 0 for kind in ("direct", "obstacle")}
    restore_evidence = []
    side_counts = {kind: {"-1": 0, "1": 0, "waypoints": 0,
                          "minimum_base_line_distance_m": math.inf} for kind in classes}
    observation_dimensions = set()
    env = make_pickcube_env()
    try:
        with h5py.File(root / "trajectory.h5", "r") as handle:
            for episode in episodes:
                index, kind = episode["episode_id"], episode["kind"]
                group = handle[f"traj_{index}"]
                actions, obs = group["actions"][:], group["obs"][:]
                assert actions.ndim == 2 and actions.shape[1] == 7
                assert (np.isfinite(actions).all()
                        and np.all(np.abs(actions[:, :6]) <= 0.350001)
                        and np.all(np.abs(actions[:, 6]) <= 1.000001))
                assert obs.ndim == 2 and len(obs) == len(actions) + 1
                assert np.isfinite(obs).all()
                observation_dimensions.add(obs.shape[1])
                assert len(group["success"]) == len(actions) + 1 and group["success"][-1]
                assert len(group["terminated"]) == len(actions) + 1
                assert group["terminated"][-1] and not group["truncated"][:].any()
                def check_rows(node):
                    if isinstance(node, h5py.Dataset):
                        assert len(node) == len(actions) + 1
                        assert np.isfinite(node[:]).all()
                    else:
                        for child in node.values():
                            check_rows(child)
                check_rows(group["env_states"])
                identity = episode["trace"]["runtime_initial_identity"]
                info = episode["infos"][-1]
                assert bool(info["native_success"][0]) and bool(info["success"][0])
                if kind == "obstacle":
                    assert bool(info["obstacle_enabled"][0])
                    assert not bool(info["obstacle_collision"][0])
                    assert not bool(info["obstacle_crossing"][0])
                layout = episode["layout"]
                start, goal, base = map(lambda key: np.asarray(layout[key], dtype=float),
                                        ("start_xy", "goal_xy", "base_xy"))
                direction = goal - start
                distance = float(np.linalg.norm(direction))
                signed = float(np.cross(direction, base - start))
                side = int(np.sign(signed))
                assert distance >= 0.14 and side != 0
                line_distance = abs(signed) / distance
                assert line_distance >= 0.06
                side_counts[kind][str(side)] += 1
                side_counts[kind]["minimum_base_line_distance_m"] = min(
                    side_counts[kind]["minimum_base_line_distance_m"], line_distance)
                waypoint = episode["trace"].get("waypoint_xy")
                if kind == "obstacle":
                    assert waypoint is not None
                    assert np.sign(np.cross(direction, np.asarray(waypoint) - start)) == side
                    side_counts[kind]["waypoints"] += 1
                # HDF5 nested groups are recursively materialized as the native state tree.
                def read_tree(node):
                    return ({key: read_tree(value) for key, value in node.items()}
                            if isinstance(node, h5py.Group) else node[0])
                state = read_tree(group["env_states"])
                env.reset(seed=episode["seed"], options={"obstacle_enabled": kind == "obstacle"})
                state = _restore_like(state, env.unwrapped.get_state_dict())
                _set_state_dict_exact(env.unwrapped, state)
                native_obs = env.unwrapped.get_obs()
                flattened = np.asarray(native_obs).reshape(-1)
                np.testing.assert_array_equal(flattened, obs[0])
                assert _value_hash(state) == identity["native_state_sha256"]
                controller = env.unwrapped.agent.controller
                template = controller.get_state()
                controller.set_state(_restore_like(identity["controller_state_snapshot"], template))
                assert _value_hash(controller.get_state()) == identity["controller_state_sha256"]
                restore_evidence.append({"episode_id": index, "kind": kind,
                    "native_state_sha256": identity["native_state_sha256"],
                    "controller_state_sha256": identity["controller_state_sha256"],
                    "observation_equal": True})
                classes[kind] += 1
    finally:
        env.close()
    dataset = PickCubePilotDataset(root / "trajectory.h5")
    windows = len(dataset)
    evidence = {"episodes": len(episodes), "class_counts": classes,
                "observation_dimensions": sorted(observation_dimensions),
                "geometry_and_waypoint_checks": side_counts,
                "training_windows": windows, "restore_checks": restore_evidence,
                "trajectory_sha256": hashlib.sha256((root / "trajectory.h5").read_bytes()).hexdigest()}
    (output / "validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    write_manifest(root, output, evidence)
    if classes != {"direct": 1000, "obstacle": 100}:
        raise RuntimeError(f"formal targets not met: {classes}")
    return evidence


def write_manifest(root, output, evidence=None):
    root, output = Path(root), Path(output)
    evidence = evidence or json.loads((output / "validation.json").read_text())
    episodes = json.loads((root / "trajectory.json").read_text())["episodes"]
    classes = evidence["class_counts"]
    report = json.loads((root / "report.json").read_text())
    manifest = {
        "dataset": str(root), "complete": report["complete"],
        "targets": {kind: report["classes"][kind]["target"] for kind in classes},
        "accepted_counts": classes,
        "attempts": {kind: {"complete": report["classes"][kind]["complete_attempts"],
                            "elapsed_seconds_sum": sum(a["elapsed_seconds"] for a in report["attempts"]
                                                        if a["class"] == kind),
                            "failures": report["classes"][kind]["failures"],
                            "stage_counts": report["classes"][kind]["stage_counts"],
                            "phase_elapsed_seconds": report["classes"][kind]["phase_elapsed_seconds"]}
                     for kind in classes},
        "episode_seeds": [{"episode_id": e["episode_id"], "kind": e["kind"], "seed": e["seed"]}
                          for e in episodes],
        "source_fingerprints": json.loads((root / "frozen_contract.json").read_text())[
            "source_fingerprints"],
        "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("trajectory.h5", "trajectory.json", "frozen_contract.json",
                                "collection_config.json", "contract.json", "report.json",
                                "artifact_hashes.json", "attempts.jsonl", "episodes.jsonl")},
        "frozen_contract_sha256": report["frozen_contract_sha256"],
        "validation": {key: value for key, value in evidence.items() if key != "restore_checks"},
    }
    replay_path = output / "replay.json"
    if replay_path.exists():
        manifest["replay_sha256"] = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    runtime_config = output / "runtime_contract_resolution.json"
    if runtime_config.exists():
        manifest["runtime_contract_resolution_sha256"] = hashlib.sha256(
            runtime_config.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_runtime_contract_resolution(output_dir):
    output = Path(output_dir)
    env = make_pickcube_env()
    try:
        base = env.unwrapped
        controller = base.agent.controller.controllers["arm"]
        config = controller.config
        pose_controller = importlib.import_module("mani_skill.agents.controllers.pd_ee_pose")
        panda_config = importlib.import_module("mani_skill.agents.robots.panda.panda")
        sources = (Path(pose_controller.__file__), Path(panda_config.__file__))
        resolved = {
            "capture": "live PickCubeObstacle Panda arm controller after formal run; runtime values read directly",
            "control_mode": base.control_mode,
            "pos_lower": np.broadcast_to(config.pos_lower, 3).astype(float).tolist(),
            "pos_upper": np.broadcast_to(config.pos_upper, 3).astype(float).tolist(),
            "rot_lower": np.broadcast_to(config.rot_lower, 3).astype(float).tolist(),
            "rot_upper": np.broadcast_to(config.rot_upper, 3).astype(float).tolist(),
            "expert_native_rotation_input": "XYZ Euler delta / signed rot_lower; clipped to [-0.35, 0.35]",
            "expert_translation_input": "delta position / 0.1 m; clipped to [-0.35, 0.35]",
            "controller_state_snapshot_observed": base.agent.controller.get_state(),
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
        }
    finally:
        env.close()
    destination = output / "runtime_contract_resolution.json"
    destination.write_text(json.dumps(resolved, indent=2) + "\n")
    return resolved


def replay_representatives(dataset_dir, output_dir, per_class=5):
    root, output = Path(dataset_dir), Path(output_dir)
    episodes = json.loads((root / "trajectory.json").read_text())["episodes"]
    selected = {kind: [episode for episode in episodes if episode["kind"] == kind][:per_class]
                for kind in ("direct", "obstacle")}
    if any(len(rows) != per_class for rows in selected.values()):
        raise ValueError("not enough accepted episodes for representative replay")
    env = make_pickcube_env()
    results = []
    try:
        with h5py.File(root / "trajectory.h5", "r") as handle:
            for kind in ("direct", "obstacle"):
                for episode in selected[kind]:
                    group = handle[f"traj_{episode['episode_id']}"]
                    def read_tree(node):
                        return ({key: read_tree(value) for key, value in node.items()}
                                if isinstance(node, h5py.Group) else node[0])
                    base = env.unwrapped
                    env.reset(seed=episode["seed"], options={"obstacle_enabled": kind == "obstacle"})
                    state = _restore_like(read_tree(group["env_states"]), base.get_state_dict())
                    _set_state_dict_exact(base, state)
                    identity = episode["trace"]["runtime_initial_identity"]
                    controller = base.agent.controller
                    controller.set_state(_restore_like(identity["controller_state_snapshot"],
                                                        controller.get_state()))
                    final_info = None
                    for action in group["actions"][:]:
                        _, _, terminated, truncated, final_info = env.step(
                            torch.as_tensor(action, dtype=torch.float32, device=base.device))
                        if terminated or truncated:
                            break
                    native_success = bool(final_info["native_success"][0])
                    collision = bool(final_info["obstacle_collision"][0])
                    crossing = bool(final_info["obstacle_crossing"][0])
                    safe = native_success and not collision and not crossing
                    if not native_success or (kind == "obstacle" and not safe):
                        raise RuntimeError(f"representative replay failed: {kind} episode {episode['episode_id']}")
                    results.append({"episode_id": episode["episode_id"], "kind": kind,
                                    "seed": episode["seed"], "replayed_actions": len(group["actions"]),
                                    "native_success": native_success,
                                    "obstacle_collision": collision,
                                    "obstacle_crossing": crossing})
    finally:
        env.close()
    destination = output / "replay.json"
    destination.write_text(json.dumps({"per_class": per_class, "episodes": results}, indent=2) + "\n")
    write_manifest(root, output)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--replay-representatives", action="store_true")
    parser.add_argument("--resolve-runtime-contract", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        write_manifest(args.dataset_dir, args.output_dir)
        return
    if args.replay_representatives:
        replay_representatives(args.dataset_dir, args.output_dir)
        return
    if args.resolve_runtime_contract:
        write_runtime_contract_resolution(args.output_dir)
        write_manifest(args.dataset_dir, args.output_dir)
        return
    validate(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
