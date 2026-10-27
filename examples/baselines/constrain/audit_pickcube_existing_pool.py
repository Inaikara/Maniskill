"""Read-only inventory of retained PickCube transport-detour artifacts.

This deliberately does not construct environments, sample layouts, or replay
actions. It reports which already-packaged obstacle episodes are actually
available and their task-layout overlaps.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SOURCES = {
    "expansion_1000_100": Path("demos/PickCube-transport-detour/20261024_104348_expansion"),
    "obstacle300": Path("demos/PickCube-transport-detour/20261024_213325_obstacle300"),
}
RETAINED_CATALOGS = (
    Path("outputs/exp/20261024_104348_pickcube_data_expansion/direct_scenes/assembled"),
    Path("outputs/exp/20261024_104348_pickcube_data_expansion/pair_scenes/assembled_final"),
    Path("outputs/exp/20261024_213325_pickcube_obstacle300/obstacle_scenes/assembled"),
    Path("outputs/exp/20261024_213325_pickcube_obstacle300/obstacle_scenes/assembled_deficit"),
)


def layout_key(episode):
    layout = episode.get("layout", {})
    start, goal = layout.get("start_xy"), layout.get("goal_xy")
    if start is None or goal is None:
        return None
    obstacle = layout.get("obstacle_center_xy", layout.get("cylinder_center_xy", layout.get("center_xy")))
    fields = [*start, *goal]
    if episode.get("kind") == "obstacle":
        if obstacle is None:
            obstacle = episode.get("obstacle", {}).get("center_xy")
        if obstacle is None:
            return None
        fields.extend(obstacle)
    return tuple(int(round(float(value) / 0.02)) for value in fields)


def task_duplicate(left, right, tolerance=0.02):
    """Ticket identity: same seed, or componentwise-near start and goal."""
    left_layout, right_layout = left.get("layout", {}), right.get("layout", {})
    left_seed = left_layout.get("seed", left.get("provenance", {}).get("seed"))
    right_seed = right_layout.get("seed", right.get("provenance", {}).get("seed"))
    if left_seed is not None and right_seed is not None and int(left_seed) == int(right_seed):
        return True
    for field in ("start_xy", "goal_xy"):
        a, b = left_layout.get(field), right_layout.get(field)
        if a is None or b is None or np.asarray(a).shape != np.asarray(b).shape:
            return False
        if np.any(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) > tolerance):
            return False
    return True


def count_task_unique(episodes):
    unique = []
    duplicates = []
    for episode in episodes:
        if any(task_duplicate(episode, previous) for previous in unique):
            duplicates.append(int(episode["episode_id"]))
        else:
            unique.append(episode)
    return len(unique), duplicates


def _action_summary(handle, episode):
    group = handle[f"traj_{episode['episode_id']}"]
    actions = np.asarray(group["actions"], dtype=np.float64)
    return {
        "steps": int(len(actions)),
        "shape": list(actions.shape),
        "finite": bool(np.isfinite(actions).all()),
        "min": np.min(actions, axis=0).tolist(),
        "max": np.max(actions, axis=0).tolist(),
        "rms": np.sqrt(np.mean(np.square(actions), axis=0)).tolist(),
        "obs_shape": list(group["obs"].shape),
        "terminated_count": int(np.asarray(group["terminated"]).sum()),
        "truncated_count": int(np.asarray(group["truncated"]).sum()),
    }


def inventory_source(name, root):
    manifest_path, h5_path = root / "trajectory.json", root / "trajectory.h5"
    if not manifest_path.is_file() or not h5_path.is_file():
        return {"name": name, "path": root.as_posix(), "available": False}
    episodes = json.loads(manifest_path.read_text())["episodes"]
    with h5py.File(h5_path, "r") as handle:
        rows = []
        for episode in episodes:
            row = {
                "episode_id": int(episode["episode_id"]),
                "kind": episode.get("kind", "unknown"),
                "layout_key_2cm": layout_key(episode),
            "expert": episode.get("expert", {
                "native_success": episode.get("native_success"),
                "held_at_success": episode.get("held_at_success"),
                "obstacle_collision": episode.get("obstacle_collision"),
                "obstacle_crossing": episode.get("obstacle_crossing"),
            }),
                "config": episode.get("config", {}),
                "actions": _action_summary(handle, episode),
            }
            rows.append(row)
    kinds = {kind: [row for row in rows if row["kind"] == kind] for kind in ("direct", "obstacle")}
    return {
        "name": name,
        "path": root.as_posix(),
        "available": True,
        "trajectory_sha256": hashlib.sha256(h5_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "counts": {kind: len(selected) for kind, selected in kinds.items()},
        "task_unique_componentwise_2cm": {
            kind: count_task_unique([episode for episode in episodes if episode.get("kind") == kind])[0]
            for kind in ("direct", "obstacle")
        },
        "action_summary": {
            kind: _summarize_action_rows([row["actions"] for row in selected])
            for kind, selected in kinds.items()
        },
        "episodes": rows,
    }


def _summarize_action_rows(rows):
    if not rows:
        return {"episodes_with_actions": 0}
    lengths = np.asarray([row["steps"] for row in rows], dtype=np.int64)
    mins = np.asarray([row["min"] for row in rows], dtype=np.float64)
    maxs = np.asarray([row["max"] for row in rows], dtype=np.float64)
    rms = np.asarray([row["rms"] for row in rows], dtype=np.float64)
    return {
        "episodes_with_actions": len(rows),
        "length_min_median_p90_p95_max": [int(lengths.min()), float(np.median(lengths)),
                                           float(np.percentile(lengths, 90)),
                                           float(np.percentile(lengths, 95)), int(lengths.max())],
        "channel_min": mins.min(axis=0).tolist(),
        "channel_max": maxs.max(axis=0).tolist(),
        "episode_rms_min_median_max": [rms.min(axis=0).tolist(), np.median(rms, axis=0).tolist(),
                                        rms.max(axis=0).tolist()],
        "all_finite": all(row["finite"] for row in rows),
        "action_shapes": sorted({tuple(row["shape"]) for row in rows}),
        "observation_shapes": sorted({tuple(row["obs_shape"]) for row in rows}),
    }


def audit(repo_root=Path(".")):
    repo = Path(repo_root)
    sources = {name: inventory_source(name, repo / path) for name, path in SOURCES.items()}
    raw_episodes = {}
    for name, source in sources.items():
        raw_episodes[name] = {
            role: [episode for episode in json.loads((repo / SOURCES[name] / "trajectory.json").read_text())["episodes"]
                   if episode.get("kind") == role] if source.get("available") else []
            for role in ("direct", "obstacle")
        }
    obstacle300 = raw_episodes.get("obstacle300", {}).get("obstacle", [])
    expansion_obstacle = raw_episodes.get("expansion_1000_100", {}).get("obstacle", [])
    expansion_extra = [episode for episode in expansion_obstacle
                       if not any(task_duplicate(episode, prior) for prior in obstacle300)]
    dev_path = repo / Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes/manifest.json")
    prior_path = repo / Path("outputs/exp/20261023_165956_pickcube_transport_detour/manifest.json")
    external = []
    external_sources = []
    for label, path in (("frozen_dev30", dev_path), ("ticket03", prior_path)):
        if path.is_file():
            payload = json.loads(path.read_text())
            rows = payload.get("scenes", payload.get("episodes", []))
            normalized = []
            for index, row in enumerate(rows):
                layout = row.get("layout", row)
                normalized.append({"episode_id": row.get("episode_id", row.get("scene_id", index)),
                                   "kind": "obstacle",
                                   "layout": {"seed": layout.get("seed"),
                                              "start_xy": layout.get("start_xy"),
                                              "goal_xy": layout.get("goal_xy")}})
            external.extend(normalized)
            external_sources.append({"name": label, "path": path.relative_to(repo).as_posix(),
                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                     "records": len(normalized)})
        else:
            external_sources.append({"name": label, "path": path.relative_to(repo).as_posix(),
                                     "available": False})
    retained = []
    catalog_candidates = []
    for path in RETAINED_CATALOGS:
        absolute = repo / path
        files = sorted(p for p in absolute.rglob("*") if p.is_file()) if absolute.exists() else []
        retained.append({"path": path.as_posix(), "exists": absolute.exists(),
                         "files": [{"path": p.relative_to(repo).as_posix(),
                                    "size": p.stat().st_size,
                                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                    for p in files]})
        manifest_path, scenes_path = absolute / "manifest.json", absolute / "scenes.h5"
        if not manifest_path.is_file() or not scenes_path.is_file():
            continue
        rows = json.loads(manifest_path.read_text()).get("scenes", [])
        per_catalog = []
        with h5py.File(scenes_path, "r") as h5:
            for row in rows:
                expert = row.get("expert", {})
                candidate = {"episode_id": row.get("scene_id"), "kind": "obstacle",
                             "layout": {"seed": row.get("seed"), "start_xy": row.get("start_xy"),
                                        "goal_xy": row.get("goal_xy")}}
                group = h5[str(row["scene_id"])]
                actions, observations = np.asarray(group["expert_actions"]), np.asarray(group["observations"])
                certified = (
                    expert.get("native_success") is True and expert.get("held_at_success") is True
                    and expert.get("safe_success") is True and expert.get("obstacle_collision") is False
                    and expert.get("obstacle_crossing") is False and expert.get("direct_transport_blocked") is True
                    and float(row.get("cylinder", {}).get("radius", -1)) == 0.03
                    and actions.ndim == 2 and actions.shape[1] == 7 and np.isfinite(actions).all()
                    and observations.ndim == 2 and observations.shape == (len(actions) + 1, 45)
                    and bool(np.asarray(group["obstacle_terminated"]).any())
                    and not bool(np.asarray(group["obstacle_truncated"]).any())
                )
                if not certified or any(task_duplicate(candidate, old) for old in obstacle300):
                    continue
                if any(task_duplicate(candidate, prior) for prior in external):
                    continue
                if any(task_duplicate(candidate, prior) for prior in catalog_candidates):
                    continue
                catalog_candidates.append(candidate)
                per_catalog.append(row["scene_id"])
        retained[-1]["catalog_identity"] = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "scenes_h5_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
            "source_identity": json.loads((absolute / "source_identity.json").read_text()),
            "scene_count": len(rows),
            "new_eligible_ids_after_old300_dev30_ticket03_and_prior_catalog_order": per_catalog,
        }
    return {
        "audit_type": "read_only_existing_artifact_inventory_no_sampling_no_replay",
        "dedup_rule": "same seed OR componentwise start_xy and goal_xy within 0.02 m; greedy source order",
        "sources": sources,
        "overlap": {
            "expansion_obstacle_unique": count_task_unique(expansion_obstacle)[0],
            "obstacle300_obstacle_unique": count_task_unique(obstacle300)[0],
            "expansion_obstacle_also_in_obstacle300": len(expansion_obstacle) - len(expansion_extra),
            "expansion_obstacle_not_in_obstacle300": len(expansion_extra),
            "combined_obstacle_unique": count_task_unique([*obstacle300, *expansion_obstacle])[0],
        },
        "additional_existing_source_catalogs": retained,
        "external_exclusion_sources": external_sources,
        "catalog_new_eligible_count": len(catalog_candidates),
        "materialization": {
            "goal_obstacle_count": 500,
            "already_qualified_obstacle_count": len(obstacle300),
            "additional_unique_obstacle_required": max(0, 500 - len(obstacle300)),
            "available_packaged_obstacle_trajectories": len(expansion_extra) + len(catalog_candidates),
            "existing_pool_max_unique": len(obstacle300) + len(expansion_extra) + len(catalog_candidates),
            "can_materialize_500_without_replay": len(obstacle300) + len(expansion_extra) + len(catalog_candidates) >= 500,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": args.output.as_posix(), "overlap": result["overlap"],
                      "materialization": result["materialization"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
