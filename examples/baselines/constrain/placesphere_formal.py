"""Formal PlaceSphere dataset collection and saved-artifact validation."""

import argparse
import contextlib
import hashlib
import json
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.placesphere_pilot import (
    PlaceSpherePilotConfig,
    PlaceSpherePilotDataset,
    _h5_state,
    _actions_are_legal,
    _tree_sha256,
    _policy_observation,
    make_env,
    collect_pilot,
)


def _tree_close(left, right, *, atol=1e-7):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _tree_close(left[key], right[key], atol=atol) for key in left)
    if isinstance(left, dict) or isinstance(right, dict):
        return False
    a, b = np.asarray(left), np.asarray(right)
    return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=0, equal_nan=False)


def _tree_max_abs_difference(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return float("inf")
        return max((_tree_max_abs_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, dict) or isinstance(right, dict):
        return float("inf")
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))


def validate_formal(dataset_dir, experiment_dir, *, replay_per_class=5):
    """Validate every saved trajectory, restore every initial scene, and replay examples."""
    validation_started = time.monotonic()
    root, experiment = Path(dataset_dir), Path(experiment_dir)
    contract = json.loads((root / "frozen_contract.json").read_text())
    config = PlaceSpherePilotConfig(**json.loads((root / "collection_config.json").read_text()))
    episodes = json.loads((root / "trajectory.json").read_text())["episodes"]
    expected = {"direct": config.target("direct"), "obstacle": config.target("obstacle")}
    if contract["limits"]["targets_per_class"] != expected:
        raise ValueError("collection config targets differ from the frozen contract")
    obs_dimension = contract["policy_observation"]["dimension"]
    raw_dimension = contract["raw_observation"]["dimension"]
    raw_fields = contract["raw_observation"]["flatten_order"]
    if sum(field["flattened_dimension"] for field in raw_fields) != raw_dimension:
        raise ValueError("full raw observation schema does not account for its runtime dimension")
    policy_columns = []
    offset = 0
    for field in raw_fields:
        width = field["flattened_dimension"]
        if field["path"] != "/extra/placesphere_obstacle":
            policy_columns.extend(range(offset, offset + width))
        offset += width
    if len(policy_columns) != obs_dimension:
        raise ValueError("task-specific raw-to-policy observation map differs from the frozen policy dimension")
    table_bounds = contract["sampling"]["table_bounds_xy_m"]
    container_extent = contract["sampling"]["full_container_xy_half_extent_m"]
    layout_contract = contract["sampling"]
    if len(episodes) != sum(expected.values()):
        raise ValueError("formal manifest count differs from frozen targets")
    counts = {kind: sum(ep["kind"] == kind for ep in episodes) for kind in expected}
    if counts != expected:
        raise ValueError(f"formal class counts differ from frozen targets: {counts}")

    episode_checks = []
    restore_checks = []
    max_state_restore_delta = 0.0
    max_controller_restore_delta = 0.0
    max_observation_restore_delta = 0.0

    def batched(node):
        if isinstance(node, dict):
            return {key: batched(value) for key, value in node.items()}
        return np.expand_dims(np.asarray(node), axis=0)

    def stored_controller_state(group, index):
        value = _h5_state(group, index)
        if isinstance(value, dict):
            value.pop("snapshot_sha256", None)
        return value

    with h5py.File(root / "trajectory.h5", "r") as h5:
        if len(h5) != len(episodes):
            raise ValueError("HDF5 and JSON trajectory counts differ")
        for index, episode in enumerate(episodes):
            key = f"traj_{index}"
            group = h5[key]
            actions, obs = group["actions"][:], group["obs"][:]
            raw_obs = group["raw_obs"][:]
            count = len(actions)
            if (not _actions_are_legal(actions) or obs.shape != (count + 1, obs_dimension)
                    or raw_obs.shape != (count + 1, raw_dimension)
                    or not np.isfinite(obs).all() or not np.isfinite(raw_obs).all()):
                raise ValueError(f"{key}: invalid action/observation contract")
            if not np.array_equal(raw_obs[:, policy_columns], obs):
                raise ValueError(f"{key}: saved 39-D policy observations do not match full raw observations")
            if (len(group["success"]) != count + 1 or len(group["terminated"]) != count + 1
                    or len(group["truncated"]) != count + 1
                    or not group["success"][-1] or not group["terminated"][-1]
                    or group["truncated"][:].any()):
                raise ValueError(f"{key}: invalid T/T+1 terminal evidence")
            for state_name in ("env_states", "controller_states"):
                def check_lengths(node):
                    for value in node.values():
                        if isinstance(value, h5py.Group):
                            check_lengths(value)
                        else:
                            if value.shape[0] != count + 1:
                                raise ValueError(f"{key}: {state_name} is not T+1")
                            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value[:]).all():
                                raise ValueError(f"{key}: non-finite values in {state_name}")
                check_lengths(group[state_name])
            for state_index in range(count + 1):
                controller_group = group["controller_states"]
                expected_hash = controller_group["snapshot_sha256"][state_index].decode()
                if _tree_sha256(stored_controller_state(controller_group, state_index)) != expected_hash:
                    raise ValueError(f"{key}: controller snapshot identity mismatch at {state_index}")
            infos = episode["infos"]
            if len(infos) != count + 1:
                raise ValueError(f"{key}: missing T+1 info")
            final = infos[-1]
            for field in ("native_success", "is_obj_on_bin", "is_obj_static"):
                if not bool(np.asarray(final[field]).reshape(-1)[0]):
                    raise ValueError(f"{key}: native terminal info lacks {field}")
            if bool(np.asarray(final["is_obj_grasped"]).reshape(-1)[0]):
                raise ValueError(f"{key}: terminal sphere remains grasped")
            if episode["kind"] == "obstacle":
                for info in infos:
                    if any(bool(np.asarray(info[k]).reshape(-1)[0])
                           for k in ("obstacle_collision", "obstacle_crossing")):
                        raise ValueError(f"{key}: obstacle safety flag was set")
            trace = episode["trace"]
            base = np.asarray(trace["base_xy"], dtype=np.float64)
            start = np.asarray(episode["layout"]["start_xy"], dtype=np.float64)
            goal = np.asarray(episode["layout"]["container_xy"], dtype=np.float64)
            xmin, ymin, xmax, ymax = table_bounds
            if not (-0.19 <= start[0] <= -0.08 and -0.13 <= start[1] <= 0.13
                    and 0.04 <= goal[0] <= 0.15 and -0.13 <= goal[1] <= 0.13
                    and xmin + container_extent <= goal[0] <= xmax - container_extent
                    and ymin + container_extent <= goal[1] <= ymax - container_extent
                    and np.linalg.norm(goal - start) >= layout_contract["minimum_transport_distance_m"]):
                raise ValueError(f"{key}: layout violates the frozen sampling bounds")
            line = goal - start
            base_line_distance = abs(float(np.cross(line, base - start))) / np.linalg.norm(line)
            if base_line_distance < layout_contract["minimum_base_line_distance_m"]:
                raise ValueError(f"{key}: base is too close to the transport line")
            if episode["kind"] == "obstacle":
                waypoint = np.asarray(trace["waypoint_xy"])
                if np.cross(goal-start, waypoint-start) * np.cross(goal-start, base-start) <= 0:
                    raise ValueError(f"{key}: obstacle waypoint is not on base side")
            episode_checks.append({"episode_id": index, "kind": episode["kind"],
                                   "steps": count, "T_plus_1": count + 1,
                                   "native_terminal_success": True,
                                   "safety_clear": episode["kind"] != "obstacle" or all(
                                       not bool(np.asarray(info[k]).reshape(-1)[0])
                                       for info in infos
                                       for k in ("obstacle_collision", "obstacle_crossing"))})

    ds = PlaceSpherePilotDataset(root / "trajectory.h5", config.obs_horizon, config.pred_horizon)
    if len(ds.class_windows["direct"]) == 0 or len(ds.class_windows["obstacle"]) == 0:
        raise ValueError("formal training windows missing a data class")
    env = make_env(config=config)
    try:
        for index, episode in enumerate(episodes):
            with h5py.File(root / "trajectory.h5", "r") as h5:
                group = h5[f"traj_{index}"]
                state = _h5_state(group["env_states"], 0)
                controller = stored_controller_state(group["controller_states"], 0)
            env.reset(seed=episode["seed"])
            base = env.unwrapped
            base.set_state_dict(state)
            base.agent.controller.set_state(controller)
            restored = base.get_state_dict()
            if not _tree_close(batched(state), restored):
                raise ValueError(f"traj_{index}: native initial state restore mismatch")
            max_state_restore_delta = max(
                max_state_restore_delta, _tree_max_abs_difference(batched(state), restored))
            if not _tree_close(batched(controller), base.agent.controller.get_state()):
                raise ValueError(f"traj_{index}: initial controller restore mismatch")
            max_controller_restore_delta = max(max_controller_restore_delta,
                _tree_max_abs_difference(batched(controller), base.agent.controller.get_state()))
            _, info = env.unwrapped.get_obs(), base.evaluate()
            restored_obs = _policy_observation(base, info)
            with h5py.File(root / "trajectory.h5", "r") as h5:
                expected_obs = h5[f"traj_{index}"]["obs"][0]
            max_observation_restore_delta = max(max_observation_restore_delta,
                float(np.max(np.abs(restored_obs - expected_obs), initial=0.0)))
            if not np.allclose(restored_obs, expected_obs, atol=1e-6, rtol=0):
                raise ValueError(f"traj_{index}: initial policy observation restore mismatch")
            start = np.asarray(episode["layout"]["start_xy"])
            goal = np.asarray(episode["layout"]["container_xy"])
            if not np.allclose(base.obj.pose.p[0, :2].detach().cpu().numpy(), start, atol=1e-7):
                raise ValueError(f"traj_{index}: restored sphere placement differs from layout")
            if not np.allclose(base.bin.pose.p[0, :2].detach().cpu().numpy(), goal, atol=1e-7):
                raise ValueError(f"traj_{index}: restored container placement differs from layout")
            restore_checks.append(index)

        selected = {kind: [i for i, ep in enumerate(episodes) if ep["kind"] == kind]
                    for kind in expected}
        env.close()
        replay_results = []
        for kind, indices in selected.items():
            representative_successes = 0
            for index in indices:
                with h5py.File(root / "trajectory.h5", "r") as h5:
                    group = h5[f"traj_{index}"]
                    state = _h5_state(group["env_states"], 0)
                    controller = stored_controller_state(group["controller_states"], 0)
                    actions = group["actions"][:]
                replay_env = make_env(config=config)
                try:
                    replay_env.reset(seed=episodes[index]["seed"])
                    base = replay_env.unwrapped
                    base.set_state_dict(state)
                    base.agent.controller.set_state(controller)
                    term = trunc = False
                    for action in actions:
                        _, _, term, trunc, info = replay_env.step(torch.as_tensor(action, dtype=torch.float32))
                        if term or trunc:
                            break
                    native = bool(base.evaluate()["native_success"][0].item())
                    safe = bool(base.evaluate()["success"][0].item())
                    sphere_velocity = base.obj.linear_velocity[0].detach().cpu().numpy()
                finally:
                    replay_env.close()
                representative = bool(native and (kind != "obstacle" or safe)
                                      and representative_successes < replay_per_class)
                if representative:
                    representative_successes += 1
                replay_results.append({"episode_id": index, "kind": kind,
                                       "native_success": native, "safe_success": safe,
                                       "terminated": bool(term), "truncated": bool(trunc),
                                       "sphere_linear_velocity_m_s": sphere_velocity.tolist(),
                                       "selected_representative": representative})
                if representative_successes == replay_per_class:
                    break
    finally:
        env.close()

    replay_complete = all(sum(row["kind"] == kind and row["selected_representative"]
                              for row in replay_results) == replay_per_class
                          for kind in expected)
    result = {"complete": replay_complete, "frozen_contract_sha256": hashlib.sha256(
        (root / "frozen_contract.json").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((root / "trajectory.h5").read_bytes()).hexdigest(),
        "class_counts": counts, "training_windows": {k: len(v) for k, v in ds.class_windows.items()},
        "episode_validation": episode_checks, "restored_episode_ids": restore_checks,
        "restore_numeric_tolerance": {"state_and_controller_absolute": 1e-7,
                                      "flattened_observation_absolute": 1e-6,
                                      "relative": 0},
        "observed_max_abs_restore_delta": {
            "native_state": max_state_restore_delta,
            "controller_state": max_controller_restore_delta,
            "policy_observation": max_observation_restore_delta},
        "replays": replay_results,
        "elapsed_validation_seconds": time.monotonic() - validation_started}
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "formal_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    if not replay_complete:
        raise RuntimeError("formal validation replay shortfall; actual outcomes are recorded")
    source_paths = ("placesphere_formal.py", "placesphere_pilot.py",
                    "placesphere_obstacle.py", "pickcube_expert.py", "layout_geometry.py")
    code_root = Path(__file__).resolve().parent
    identity = {
        "dataset_artifacts_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.iterdir()) if path.is_file()},
        "formal_validation_sha256": hashlib.sha256(
            (experiment / "formal_validation.json").read_bytes()).hexdigest(),
        "execution_local_source_sha256": contract["local_source_sha256_before_collection"],
        "current_local_source_sha256": {
            f"examples/baselines/constrain/{name}": hashlib.sha256(
                (code_root / name).read_bytes()).hexdigest() for name in source_paths},
        "native_source_sha256": contract["native_worktree_source_sha256"],
    }
    (experiment / "formal_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-attempts-per-class", type=int, default=2000)
    parser.add_argument("--max-wall-seconds", type=float, default=6000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    dataset, experiment = Path(args.dataset_dir), Path(args.experiment_dir)
    if args.validate_only:
        return validate_formal(dataset, experiment)
    config = PlaceSpherePilotConfig(seed=args.seed, target_per_class=1000,
        direct_target=1000, obstacle_target=100,
        max_attempts_per_class=args.max_attempts_per_class,
        max_wall_seconds=args.max_wall_seconds, max_steps=args.max_steps)
    experiment.mkdir(parents=True, exist_ok=False)
    try:
        with (experiment / "collection.log").open("w") as log, \
                contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report = collect_pilot(dataset, config)
    except BaseException:
        (experiment / "collection_failure.txt").write_text(traceback.format_exc())
        raise
    (experiment / "collection_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise RuntimeError(f"formal collection incomplete; retained report at {dataset}")
    validate_formal(dataset, experiment)


if __name__ == "__main__":
    main()
