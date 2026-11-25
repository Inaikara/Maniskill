"""Frozen reset-only scene catalogs for Plug Base evaluation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from examples.baselines.constrain import plug_orientation
from examples.baselines.constrain.plug_orientation_data import _write_json

SPLITS = {
    "fixed_validation": tuple(range(2026124100, 2026124150)),
    "random_validation": tuple(range(2026124000, 2026124050)),
    "random_final_test": tuple(range(2026125000, 2026125100)),
    "fixed_base_test": tuple(range(2026125100, 2026125200)),
}
RESERVED_ADAPTATION = tuple(range(2026123000, 2026123300))


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def _state_hash(state):
    def walk(value):
        if isinstance(value, dict): return {k: walk(v) for k, v in sorted(value.items())}
        array = np.ascontiguousarray(value)
        return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    return _canonical(walk(state))


def _write_state(group, state):
    for key, value in state.items():
        if isinstance(value, dict): _write_state(group.create_group(str(key)), value)
        else: group.create_dataset(str(key), data=np.asarray(value)[None])


def _state_numpy(value):
    if isinstance(value, dict): return {k: _state_numpy(v) for k, v in value.items()}
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim and array.shape[0] == 1: array = array[0]
    return np.asarray(array)


def _read_state(group):
    return {key: (_read_state(value) if isinstance(value, h5py.Group) else np.asarray(value[0]))
            for key, value in group.items()}


def _torch_state(state, device):
    if isinstance(state, dict): return {k: _torch_state(v, device) for k, v in state.items()}
    return torch.as_tensor(state, device=device).unsqueeze(0)


def build_scene_catalog(output_dir, split_name, *, env_factory=None):
    """Reset only; saves state/observation before any policy action or optimization."""
    if split_name not in SPLITS: raise ValueError(f"unknown fixed Plug split {split_name}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if env_factory is None:
        import examples.baselines.constrain.plug_orientation  # noqa: F401 register experiment env
        env_factory = lambda: gym.make(plug_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                                       sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=1000)
    manifest = {"format": "plug-orientation-scenes-v1", "split": split_name,
                "scene_count": len(SPLITS[split_name]), "scenes_file": "scenes.h5", "scenes": [],
                "reserved_adaptation_seed_range": [RESERVED_ADAPTATION[0], RESERVED_ADAPTATION[-1]]}
    env = env_factory()
    try:
        with h5py.File(output_dir / "scenes.h5", "w") as handle:
            for index, seed in enumerate(SPLITS[split_name]):
                yaw_rng = np.random.default_rng(int(seed) + 1703)
                yaw = 0.0 if split_name in ("fixed_validation", "fixed_base_test") else float(yaw_rng.uniform(-np.pi / 8, np.pi / 8))
                # The random yaw stream is separately deterministic and is not a
                # function of native environment RNG or data-collection targets.
                obs, _ = env.reset(seed=int(seed), options={"target_yaw": yaw})
                task = env.unwrapped
                state = _state_numpy(task.get_state_dict())
                flat = obs.detach().cpu().numpy().reshape(-1) if hasattr(obs, "detach") else np.asarray(obs).reshape(-1)
                if flat.shape != (46,) or not np.isfinite(flat).all():
                    raise ValueError(f"scene {seed} initial observation must be finite 46-D")
                sid = f"plug-{split_name}-{index:03d}"
                group = handle.create_group(sid)
                _write_state(group.create_group("initial_state"), state)
                group.create_dataset("initial_observation", data=flat)
                charger_pose = task.charger.pose.raw_pose[0].detach().cpu().numpy().astype(float).tolist()
                goal_position = task.goal_pose.p[0].detach().cpu().numpy().astype(float).tolist()
                manifest["scenes"].append({"scene_id": sid, "seed": int(seed), "target_yaw_rad": yaw,
                    "charger_pose_wxyz": charger_pose, "goal_position_m": goal_position,
                    "initial_state_sha256": _state_hash(state),
                    "initial_observation_sha256": hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest()})
    finally:
        env.close()
    manifest["scenes_file_sha256"] = _sha(output_dir / "scenes.h5")
    manifest["manifest_sha256"] = _canonical(manifest)
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def freeze_evaluation_catalogs(output_root, *, env_factory=None):
    """Create all immutable splits together before the first optimizer update."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    result = {}
    for split in SPLITS:
        result[split] = build_scene_catalog(root / split, split, env_factory=env_factory)
    (root / "catalog_set.json").write_text(json.dumps({
        "format": "plug-orientation-catalog-set-v1",
        "catalogs": {name: {"manifest_sha256": hashlib.sha256(
            (root / name / "manifest.json").read_bytes()).hexdigest(),
            "scene_count": len(SPLITS[name]), "split": name} for name in SPLITS}},
        indent=2, sort_keys=True) + "\n")
    return result


class PlugSceneCatalog:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.path = self.directory / self.manifest["scenes_file"]
        if _sha(self.path) != self.manifest["scenes_file_sha256"]: raise ValueError("scene HDF5 checksum mismatch")
        payload = {k: v for k, v in self.manifest.items() if k != "manifest_sha256"}
        if _canonical(payload) != self.manifest["manifest_sha256"]: raise ValueError("scene manifest checksum mismatch")
        if self.manifest["scene_count"] != len(self.manifest["scenes"]): raise ValueError("scene count mismatch")
        split = self.manifest.get("split")
        if split not in SPLITS or len(self) != len(SPLITS[split]): raise ValueError("catalog split/size is not registered")
        seeds = [int(row["seed"]) for row in self.manifest["scenes"]]
        if seeds != list(SPLITS[split]): raise ValueError("catalog seeds differ from the frozen split")
        for row in self.manifest["scenes"]:
            yaw = float(row["target_yaw_rad"])
            if split in ("fixed_validation", "fixed_base_test"):
                if yaw != 0.0: raise ValueError("fixed catalog contains a nonzero target yaw")
            else:
                expected_yaw = float(np.random.default_rng(int(row["seed"]) + 1703).uniform(-np.pi / 8, np.pi / 8))
                if (yaw < -np.pi / 8 or yaw > np.pi / 8
                        or not math.isclose(yaw, expected_yaw, rel_tol=0, abs_tol=0)):
                    raise ValueError("random catalog yaw differs from its frozen per-seed generator")

    def __len__(self): return int(self.manifest["scene_count"])

    def load(self, index):
        row = self.manifest["scenes"][index]
        with h5py.File(self.path, "r") as h5:
            group = h5[row["scene_id"]]
            state, obs = _read_state(group["initial_state"]), np.asarray(group["initial_observation"][:])
        if _state_hash(state) != row["initial_state_sha256"]: raise ValueError("scene state checksum mismatch")
        if obs.shape != (46,) or not np.isfinite(obs).all(): raise ValueError("invalid scene observation")
        if hashlib.sha256(np.ascontiguousarray(obs).tobytes()).hexdigest() != row["initial_observation_sha256"]:
            raise ValueError("scene observation checksum mismatch")
        return row, state, obs


def restore_plug_scene(env, scene):
    row, state, expected = scene
    # One reset initializes the same task class; then restore exactly one captured state.
    env.reset(seed=int(row["seed"]), options={"target_yaw": float(row["target_yaw_rad"])})
    env.unwrapped.set_state_dict(_torch_state(state, env.unwrapped.device))
    obs = env.unwrapped.get_obs()
    actual = obs.detach().cpu().numpy().reshape(-1)
    if actual.shape != (46,) or not np.allclose(actual, expected, atol=1e-6, rtol=0):
        raise ValueError(f"restored Plug scene mismatch: {row['scene_id']}")
    return obs
