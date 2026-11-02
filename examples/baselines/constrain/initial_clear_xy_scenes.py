"""Initial-TCP geometric admission for render-only evaluation scene sampling."""

import numpy as np
import hashlib
import json
import time
import signal
import shutil
from pathlib import Path


PROFILE_NAME = "tcp_xy_cylinder_radius_plus_5mm_v1"


def _point(value, name):
    if value is None:
        raise LookupError("missing_measured_tcp")
    point = np.asarray(value, dtype=np.float64)
    if point.size != 2 or not np.all(np.isfinite(point)):
        raise ValueError(f"non_finite {name}")
    return point.reshape(2)


def _segment_gap(start, end, center):
    delta = end - start
    fraction = np.clip(np.dot(center - start, delta) / max(float(np.dot(delta, delta)), 1e-24), 0, 1)
    return float(np.linalg.norm(start + fraction * delta - center))


def _coordinate_records(document, task):
    records = []
    containers = document.get("episodes", document.get("scenes", []))
    for item in containers:
        metadata = item.get("metadata", item)
        start = metadata.get("start_xy")
        goal = metadata.get("goal_xy", metadata.get("container_xy"))
        if start is None or goal is None:
            continue
        records.append((metadata.get("seed", item.get("seed")), start, goal))
    return records


def _duplicate_layout(seed, start, goal, records):
    for old_seed, old_start, old_goal in records:
        if old_seed is not None and int(old_seed) == int(seed):
            return True
        if (np.max(np.abs(np.asarray(start) - np.asarray(old_start))) <= .02
                and np.max(np.abs(np.asarray(goal) - np.asarray(old_goal))) <= .02):
            return True
    return False


def initial_tcp_admission(tcp_xy, center_xy, cylinder_radius_m, margin_m, chord_start=None):
    """Fail closed unless measured TCP clears the strict radius+margin disk."""
    radius, margin = float(cylinder_radius_m), float(margin_m)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("cylinder radius must be finite and positive")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and non-negative")
    try:
        tcp = _point(tcp_xy, "tcp")
        center = _point(center_xy, "center")
        start = _point(chord_start, "chord_start") if chord_start is not None else None
    except LookupError as exc:
        return {"accepted": False, "reason": str(exc)}
    except ValueError as exc:
        return {"accepted": False, "reason": "non_finite" if "non_finite" in str(exc) else str(exc)}
    profile_radius = radius + margin
    gap = float(np.linalg.norm(tcp - center)) - profile_radius
    outside = gap > 1e-12
    result = {"accepted": outside, "reason": "accepted" if outside else
              ("boundary" if gap >= -1e-12 else "inside"), "tcp_xy": tcp.tolist(),
              "center_xy": center.tolist(), "cylinder_radius_m": radius,
              "margin_m": margin, "profile": PROFILE_NAME,
              "signed_initial_gap_m": gap}
    if start is not None and _segment_gap(start, tcp, center) <= profile_radius:
        result.update(accepted=False, reason="grazing_chord")
    return result


def _hash_tree(tree):
    from examples.baselines.constrain.pickcube_scenes import _tree_hash
    return _tree_hash(tree)


def sample_task(task, output_dir, *, seed, target=30, max_attempts=1000, max_wall_seconds=600):
    """Sample native reset snapshots only; never steps, loads a policy, or writes demos."""
    import gymnasium as gym
    import torch

    previous_make = gym.make
    gym.make = lambda *args, **kwargs: previous_make(*args, obstacle_mode="render_only", **kwargs)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if task == "pickcube":
        from examples.baselines.constrain.pickcube_expert import make_pickcube_env
        from examples.baselines.constrain.pickcube_transport_detour import FORMAT
        from examples.baselines.constrain.pickcube_scenes import _write_tree
        from examples.baselines.constrain import pickcube_obstacle  # register task
        env_factory = lambda: make_pickcube_env(obstacle_radius=.03, obstacle_height=.14)
        key, object_actor, goal_actor, obs_kind = "pickcube", "cube", "goal_site", "flat"
    elif task == "placesphere":
        from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
        from examples.baselines.constrain.placesphere_transport_detour import FORMAT
        from examples.baselines.constrain.placesphere_scenes import _write_tree
        from examples.baselines.constrain import placesphere_obstacle
        from examples.baselines.constrain.placesphere_pilot import _obs_vector, _policy_observation
        env_factory = lambda: make_env(config=PlaceSpherePilotConfig(
            max_steps=320, obstacle_radius=.03, obstacle_height=.12))
        key, object_actor, goal_actor, obs_kind = "placesphere", "obj", "bin", "place"
    else:
        raise ValueError("task must be pickcube or placesphere")

    root = Path.cwd()
    source_rel = ["examples/baselines/constrain/initial_clear_xy_scenes.py",
                  "examples/baselines/constrain/render_only_xy_eval.py",
                  f"examples/baselines/constrain/{task}_obstacle.py",
                  f"examples/baselines/constrain/{task}_transport_detour.py",
                  "examples/baselines/constrain/pickcube_scenes.py",
                  "examples/baselines/constrain/pickcube_expert.py",
                  "examples/baselines/constrain/placesphere_pilot.py",
                  "mani_skill/agents/robots/panda/panda.py",
                  "mani_skill/envs/tasks/tabletop/pick_cube.py",
                  "mani_skill/envs/tasks/tabletop/place_sphere.py"]
    source_rel = [name for name in source_rel if (root / name).is_file()]
    snapshot_dir = output / "source_snapshot"
    snapshot_dir.mkdir()
    source_hashes = {}
    for rel in source_rel:
        src = root / rel
        snapshot_target = snapshot_dir / (rel.replace("/", "__"))
        shutil.copyfile(src, snapshot_target)
        source_hashes[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
    if task == "pickcube":
        input_candidates = {
            "checkpoint": Path("outputs/ckpt/20261029_160000_pickcube_horizon_clearance_r44_control/final.pt"),
            "base_checkpoint": Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt"),
            "training_h5": Path("demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.h5"),
            "training_manifest": Path("demos/PickCube-transport-detour/20261028_164500_frozen500/trajectory.json"),
            "old_eval_manifest": Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes/manifest.json"),
            "old_eval_h5": Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes/scenes.h5")}
    else:
        input_candidates = {
            "checkpoint": Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt"),
            "base_checkpoint": Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base_verified/final.pt"),
            "training_h5": Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.h5"),
            "training_manifest": Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.json"),
            "old_eval_manifest": Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes/manifest.json"),
            "old_eval_h5": Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes/scenes.h5")}
    inputs = {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
              for key, path in input_candidates.items() if path.is_file()}
    dedup_records = []
    for key in ("training_manifest", "old_eval_manifest"):
        path = Path(inputs[key]["path"])
        dedup_records.extend(_coordinate_records(json.loads(path.read_text()), task))
    prior_path = (Path("outputs/exp/20261102_111155_render_only_xy_r46") /
                  ("pickcube_full_registration.json" if task == "pickcube" else "placesphere_full_registration.json"))
    # The R46 execution rows preserve the historic task/seed/start/goal cohort.
    # This source is recorded as a deduplication input, not an evaluation result.
    if prior_path.is_file():
        inputs["old_r46_registration"] = {"path": str(prior_path),
            "sha256": hashlib.sha256(prior_path.read_bytes()).hexdigest()}
        try:
            prior = json.loads(prior_path.read_text())
            for row in prior.get("inputs", {}).values():
                candidate = Path(row.get("path", ""))
                if candidate.name == "rows.json" and candidate.is_file():
                    dedup_records.extend(_coordinate_records(json.loads(candidate.read_text()), task))
        except (OSError, ValueError, TypeError):
            pass
    registration = {"format": "initial_clear_xy_sampling_registration_v1", "task": task,
        "seed": int(seed), "target": int(target), "max_attempts": int(max_attempts),
        "max_wall_seconds": float(max_wall_seconds), "per_reset_alarm_seconds": 300,
        "policy_rollouts": 0, "demonstration_outputs": 0, "checkpoint_identity_for_future_trial":
            inputs.get("checkpoint"), "sources": source_hashes, "inputs": inputs}
    registration["registration_sha256"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True).encode()).hexdigest()
    (output / "sampling_registration.json").write_text(json.dumps(registration, indent=2, sort_keys=True)+"\n")

    import h5py
    started, rows, journal = time.monotonic(), [], []
    env = None
    try:
        env = env_factory()
        for attempt in range(max_attempts):
            if len(rows) >= target or time.monotonic() - started >= max_wall_seconds:
                break
            attempt_seed = int(seed + attempt)
            try:
                def reset_timeout(_signum, _frame):
                    raise TimeoutError("native reset exceeded 300 seconds")
                previous_alarm = signal.signal(signal.SIGALRM, reset_timeout)
                signal.alarm(300)
                _obs, _info = env.reset(seed=attempt_seed, options={"obstacle_enabled": False})
                base = env.unwrapped
                obj = getattr(base, object_actor)
                goal = getattr(base, goal_actor)
                start_xy = np.asarray(obj.pose.p).reshape(-1, 3)[0, :2].astype(float)
                goal_xy = np.asarray(goal.pose.p).reshape(-1, 3)[0, :2].astype(float)
                vector = goal_xy - start_xy
                length = float(np.linalg.norm(vector))
                if not np.isfinite(length) or length < .12:
                    raise ValueError("transport_too_short")
                fraction = .4 + .2 * float(np.random.default_rng(attempt_seed).random())
                center = start_xy + fraction * vector
                radius, margin, height = .03, .005, (.14 if task == "pickcube" else .08)
                options = {"obstacle_enabled": True, "obstacle_center": center.tolist()}
                options["cube_xy" if task == "pickcube" else "sphere_xy"] = start_xy.tolist()
                options["goal_pos" if task == "pickcube" else "container_xy"] = (
                    [*goal_xy, float(goal.pose.p.reshape(-1, 3)[0, 2])] if task == "pickcube" else goal_xy.tolist())
                obs, info = env.reset(seed=attempt_seed, options=options)
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous_alarm)
                base = env.unwrapped
                tcp_actor = getattr(base.agent, "tcp", None) or base.agent.robot.links_map["panda_hand_tcp"]
                tcp_xy = np.asarray(tcp_actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
                admission = initial_tcp_admission(tcp_xy, center, radius, margin)
                if not admission["accepted"]:
                    raise ValueError(admission["reason"])
                if _duplicate_layout(attempt_seed, start_xy, goal_xy, dedup_records):
                    raise ValueError("duplicate_training_or_historical_eval_layout")
                # The center lies on the middle 40–60% of the direct object-goal chord;
                # both endpoints are outside the physical cylinder disk.
                if min(np.linalg.norm(start_xy-center), np.linalg.norm(goal_xy-center)) <= radius:
                    raise ValueError("endpoint_inside_cylinder")
                state = base.get_state_dict()
                controller = base.agent.controller.get_state()
                if obs_kind == "flat":
                    observation = np.asarray(base.get_obs().detach().cpu()).reshape(-1)
                    controller_json = json.loads(json.dumps(controller, default=lambda x: x.detach().cpu().tolist()))
                else:
                    raw = _obs_vector(base.get_obs(info=base.evaluate(), unflattened=True))
                    policy = _policy_observation(base, base.evaluate())
                    observation = (policy, raw)
                    from examples.baselines.constrain.placesphere_scenes import _json
                    controller_json = _json(controller)
                index, scene_id = len(rows), f"{task}-initial-clear-{len(rows):03d}"
                metadata = {"scene_id": scene_id, "seed": attempt_seed, "start_xy": start_xy.tolist(),
                    ("goal_xy" if task == "pickcube" else "container_xy"): goal_xy.tolist(),
                    "cylinder": {"center_xy": center.tolist(), "radius": radius, "height": height,
                                 "height_source": "frozen_render_only_profile_height"},
                    "sampling_profile": "initial_clear_xy_v1", "sampling_fraction": fraction,
                    "initial_tcp_admission": admission,
                    "state_sha256": _hash_tree(state), "controller_sha256": _hash_tree(controller),
                    "expert": {}}
                if obs_kind == "flat":
                    metadata["observation_sha256"] = hashlib.sha256(observation.tobytes()).hexdigest()
                    metadata["observation_shape"] = list(observation.shape)
                    metadata["observation_dtype"] = str(observation.dtype)
                else:
                    metadata["raw_observation_sha256"] = hashlib.sha256(observation[1].tobytes()).hexdigest()
                    metadata["policy_observation_sha256"] = hashlib.sha256(observation[0].tobytes()).hexdigest()
                with h5py.File(output / "scenes.h5", "a") as handle:
                    group = handle.create_group(scene_id)
                    _write_tree(group.create_group("state"), state)
                    if obs_kind == "flat":
                        group.create_dataset("initial_observation", data=observation)
                    else:
                        _write_tree(group.create_group("controller"), controller)
                        group.create_dataset("initial_policy_observation", data=observation[0])
                        group.create_dataset("initial_raw_observation", data=observation[1])
                    group.attrs["controller_json"] = json.dumps(controller_json, sort_keys=True)
                if obs_kind == "flat":
                    metadata["controller"] = controller_json
                rows.append(metadata)
                dedup_records.append((attempt_seed, start_xy.tolist(), goal_xy.tolist()))
                journal.append({"attempt": attempt, "seed": attempt_seed, "accepted": True,
                                "scene_index": index, "elapsed_seconds": time.monotonic()-started,
                                "admission": admission})
            except Exception as exc:
                signal.alarm(0)
                if "previous_alarm" in locals():
                    signal.signal(signal.SIGALRM, previous_alarm)
                journal.append({"attempt": attempt, "seed": attempt_seed, "accepted": False,
                                "reason": str(exc), "elapsed_seconds": time.monotonic()-started})
        manifest = {"format": FORMAT, "scene_count": len(rows), "scenes_file": "scenes.h5",
                    "scenes": rows, "generation_seed": seed,
                    "sampling_profile": "initial_clear_xy_v1", "sampling_attempts": len(journal)}
        encoded = json.dumps(manifest, sort_keys=True, indent=2).encode()
        manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        (output / "attempts.jsonl").write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in journal))
        report = {"task": task, "target": target, "accepted": len(rows), "attempts": len(journal),
                  "elapsed_seconds": time.monotonic()-started, "complete": len(rows) == target,
                  "profile": PROFILE_NAME, "policy_rollouts": 0, "demo_outputs": 0}
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
        if not report["complete"]:
            raise RuntimeError(f"sampling incomplete: {report}")
        return report
    finally:
        if env is not None:
            env.close()
        gym.make = previous_make


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pickcube", "placesphere"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target", type=int, default=30)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--max-wall-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    report = sample_task(args.task, args.output_dir, seed=args.seed, target=args.target,
                         max_attempts=args.max_attempts, max_wall_seconds=args.max_wall_seconds)
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


if __name__ == "__main__":
    main()
