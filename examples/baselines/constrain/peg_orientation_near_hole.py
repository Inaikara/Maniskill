"""Build an expert-reachable, validation-only near-hole branch bank."""

import hashlib
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from examples.baselines.constrain import peg_orientation
from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog, restore_peg_scene
from examples.baselines.constrain.peg_orientation_takeover import capture_physical_state, native_grasped


TOLERANCE = 1e-5
PREFIX_CAP = 300


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    return value


def _head_in_hole(task):
    pose = task.box_hole_pose.inv() * task.peg_head_pose
    return np.asarray(pose.p[0].detach().cpu(), dtype=np.float64)


def near_hole_eligible(task):
    head = _head_in_hole(task)
    inserted = bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any())
    return near_hole_geometry_eligible(head, native_grasped(task), inserted)


def near_hole_geometry_eligible(head_xyz, grasped, inserted):
    head = np.asarray(head_xyz, dtype=np.float64)
    return (bool(grasped) and not bool(inserted) and -0.05 <= head[0] <= -0.02
            and np.linalg.norm(head) <= 0.05)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finalize_saved_bank(journal_path, output_path, *, split, catalog_manifest_sha256,
                        source_h5_sha256, elapsed_seconds):
    """Finalize a complete per-branch journal without running or replaying env steps."""
    rows = [json.loads(line) for line in Path(journal_path).read_text().splitlines() if line]
    if len(rows) != 20 or [row["catalog_index"] for row in rows] != list(range(20)):
        raise ValueError("branch journal must contain the 20 frozen catalog indices in order")
    result = {"split": split, "planned_catalog_indices": list(range(20)), "branches": rows,
              "eligible_count": sum(bool(row["eligible"]) for row in rows),
              "controls": sum(int(row["prefix_controls"]) for row in rows),
              "prefix_cap_per_state": PREFIX_CAP,
              "source_h5_sha256": source_h5_sha256,
              "catalog_manifest_sha256": catalog_manifest_sha256,
              "elapsed_seconds": float(elapsed_seconds)}
    Path(output_path).write_text(json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n")
    return result


def add_no_candidate_rows(rows, scene_metadata):
    """Include frozen catalog scenes that had no expert-success source trajectory."""
    by_index = {int(row["catalog_index"]): row for row in rows}
    for index in range(20):
        if index not in by_index:
            metadata = scene_metadata[index]
            by_index[index] = {"split": metadata["split"], "catalog_index": index,
                               "scene_id": metadata["scene_id"], "seed": int(metadata["seed"]),
                               "source_episode_id": None, "eligible": False,
                               "reason": "no_expert_replay_candidate", "prefix_controls": 0}
    if set(by_index) != set(range(20)):
        raise ValueError("journal contains an out-of-range frozen catalog index")
    return [by_index[index] for index in range(20)]


def build_bank(catalog_dir, source_dir, output_dir, *, split, budget, started=None):
    """Replay converted native EE expert prefixes from catalog-matched starts."""
    catalog = PegSceneCatalog(catalog_dir)
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((source_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").read_text())
    h5_path = source_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    seed_to_row = {int(row["episode_seed"]): row for row in metadata["episodes"]}
    env_id = peg_orientation.RANDOM_ENV_ID if split == "validation_random" else peg_orientation.ENV_ID
    env = gym.make(env_id, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=600)
    rows = []
    started = time.monotonic() if started is None else started
    controls = 0
    try:
        for index in range(20):
            scene = catalog.load(index)
            scene_meta, _, expected_obs = scene
            seed = int(scene_meta["seed"])
            source_row = seed_to_row.get(seed)
            row = {"split": split, "catalog_index": index, "scene_id": scene_meta["scene_id"],
                   "seed": seed, "source_episode_id": None if source_row is None else int(source_row["episode_id"]),
                   "eligible": False, "reason": "no_expert_replay_candidate", "prefix_controls": 0}
            if source_row is None:
                rows.append(row)
                with (output_dir / "branches.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                continue
            obs = restore_peg_scene(env, scene)
            initial_obs = _array(obs).reshape(-1)
            with h5py.File(h5_path, "r") as source:
                trajectory = source[f"traj_{int(source_row['episode_id'])}"]
                recorded_obs = np.asarray(trajectory["obs"])
                actions = np.asarray(trajectory["actions"])
            if not np.allclose(initial_obs, expected_obs.reshape(-1), atol=TOLERANCE, rtol=0):
                raise ValueError(f"catalog initial observation mismatch for {scene_meta['scene_id']}")
            if not np.allclose(initial_obs, recorded_obs[0].reshape(-1), atol=TOLERANCE, rtol=0):
                raise ValueError(f"expert replay initial observation differs from catalog for seed {seed}")
            history = [initial_obs]
            action_hash = hashlib.sha256()
            for action in actions[:PREFIX_CAP]:
                if controls >= budget:
                    raise RuntimeError("aggregate near-prefix control budget exhausted")
                if time.monotonic() - started >= 1800:
                    raise TimeoutError("aggregate near-prefix stage deadline exceeded")
                action_hash.update(np.asarray(action).tobytes())
                obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
                controls += 1
                row["prefix_controls"] += 1
                history.append(_array(obs).reshape(-1))
                if near_hole_eligible(env.unwrapped):
                    state = capture_physical_state(env.unwrapped)
                    row.update({"eligible": True, "reason": "near_hole_branch",
                                "head_in_hole_xyz_m": _head_in_hole(env.unwrapped).tolist(),
                                "peg_to_tcp_pose": _array((env.unwrapped.agent.tcp.pose.inv()
                                                           * env.unwrapped.peg.pose).raw_pose).reshape(-1).tolist(),
                                "physical_state": state,
                                "observation_history": [x.tolist() for x in history[-2:]],
                                "prefix_actions_sha256": action_hash.hexdigest(),
                                "controller_mode": env.unwrapped.control_mode,
                                "native_success": bool(np.asarray(info.get("success", False)).any())})
                    break
                if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                    row["reason"] = "terminal_before_near_hole"
                    break
            if row["prefix_controls"] == PREFIX_CAP and not row["eligible"]:
                row["reason"] = "near_hole_window_not_reached_within_prefix_cap"
            rows.append(row)
            journal = output_dir / "branches.jsonl"
            with journal.open("a") as stream:
                stream.write(json.dumps(row, default=lambda x: _array(x).tolist(), sort_keys=True) + "\n")
    finally:
        env.close()
    result = finalize_saved_bank(
        output_dir / "branches.jsonl", output_dir / "bank.json", split=split,
        catalog_manifest_sha256=_sha256(Path(catalog_dir) / "manifest.json"),
        source_h5_sha256=_sha256(h5_path), elapsed_seconds=time.monotonic() - started)
    return result
