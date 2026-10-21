"""Self-contained storage and restoration for expert-verified wristcam scenes."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.collect_constructive import initial_state_hash


@dataclass(frozen=True)
class Scene:
    scene_id: str
    metadata: dict
    initial_state: dict
    initial_observation: np.ndarray


def _read_state(group):
    result = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            result[key] = _read_state(value)
        else:
            result[key] = np.asarray(value[0])
    return result


def _state_leaves(state, prefix=""):
    for key, value in state.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            yield from _state_leaves(value, path)
        else:
            yield path, np.asarray(value)


def _stable_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(_state_leaves(state)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def export_scene_set(witness_dir, output_dir, identity):
    """Copy each accepted witness's complete row-zero physical state locally."""
    witness_dir, output_dir = Path(witness_dir), Path(output_dir)
    report = json.loads((witness_dir / "report.json").read_text())
    episode_data = json.loads((witness_dir / "trajectory.json").read_text())["episodes"]
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes_path = output_dir / "scenes.h5"
    manifest = {
        "format": "maniskill-wristcam-scenes-v1",
        "identity": identity,
        "scene_count": len(episode_data),
        "scenes_file": scenes_path.name,
        "scenes": [],
        "collection": {
            key: report[key] for key in (
                "candidate_templates", "no_obstacle_successful_templates",
                "cylinder_sampling_calls", "failures", "physics_attempts",
                "accepted", "elapsed_seconds", "stage_seconds",
            )
        },
    }
    with h5py.File(witness_dir / "trajectory.h5", "r") as source, h5py.File(scenes_path, "w") as target:
        for index, episode in enumerate(episode_data):
            key = f"traj_{index}"
            state = _read_state(source[key]["env_states"])
            state_hash = _stable_hash(state)
            selected = {
                name: value for name, value in _state_leaves(state)
                if name in ("actors/cubeA", "actors/cubeB", "articulations/panda_wristcam")
            }
            if initial_state_hash(selected) != episode["initial_state_hash"]:
                raise ValueError(f"initial robot/cube state mismatch for {key}")
            scene_id = f"wristcam-{index:03d}"
            scene_group = target.create_group(scene_id)
            source.copy(f"{key}/env_states", scene_group, name="env_states")
            scene_group.create_dataset("initial_observation", data=source[f"{key}/obs"][0])
            cylinder = episode["cylinder"]
            reset_options = episode_data[index]["reset_kwargs"]["options"]
            expert = report["episodes"][index]["trace"]
            manifest["scenes"].append({
                "scene_id": scene_id,
                "template_index": episode["template_index"],
                "seed": episode["seed"],
                "cubes": reset_options["cube_layout"],
                "cylinder": cylinder,
                "initial_state_sha256": state_hash,
                "initial_state_robot_cube_sha256": episode["initial_state_hash"],
                "expert": {
                    "stack_success": expert["stack_success"],
                    "obstacle_collision": expert["obstacle_collision"],
                    "obstacle_crossing": expert["obstacle_crossing"],
                    "steps": expert["steps"],
                    "stop": "first stack_success",
                    "control_mode": episode["control_mode"],
                },
            })
    manifest["scenes_file_sha256"] = hashlib.sha256(scenes_path.read_bytes()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class SceneSet:
    """Read-only scene catalog; loading never invokes generation or an expert."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest["format"] != "maniskill-wristcam-scenes-v1":
            raise ValueError("unsupported scene-set format")
        self._path = self.directory / self.manifest["scenes_file"]
        if "manifest_sha256" in self.manifest:
            recorded = self.manifest["manifest_sha256"]
            payload = {key: value for key, value in self.manifest.items() if key != "manifest_sha256"}
            actual = hashlib.sha256(
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if actual != recorded:
                raise ValueError("scene manifest checksum mismatch")
        if "scenes_file_sha256" in self.manifest:
            actual = hashlib.sha256(self._path.read_bytes()).hexdigest()
            if actual != self.manifest["scenes_file_sha256"]:
                raise ValueError("scene data checksum mismatch")

    def __len__(self):
        return self.manifest["scene_count"]

    def load(self, index):
        metadata = self.manifest["scenes"][index]
        with h5py.File(self._path, "r") as handle:
            group = handle[metadata["scene_id"]]
            state = _read_state(group["env_states"])
            observation = np.asarray(group["initial_observation"])
        if _stable_hash(state) != metadata["initial_state_sha256"]:
            raise ValueError(f"scene state checksum mismatch: {metadata['scene_id']}")
        return Scene(metadata["scene_id"], metadata, state, observation)


def restore_scene(env, scene):
    """Reset native controller/task state, then restore the stored physical state.

    The environment must be constructed with this scene's recorded cylinder
    radius and height; the reset clears sticky safety state and resets the
    controller before the full state dictionary is applied.
    """
    import torch

    base = env.unwrapped
    cylinder = scene.metadata["cylinder"]
    env.reset(seed=int(scene.metadata["seed"]), options={
        "reconfigure": False,
        "obstacle_center": cylinder["center_xy"],
        "obstacle_radius": cylinder["radius"],
        "cube_layout": scene.metadata["cubes"],
    })
    state = {}
    for name, value in _state_leaves(scene.initial_state):
        cursor = state
        parts = name.split("/")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = torch.as_tensor(value, device=base.device).unsqueeze(0)
    base.set_state_dict(state)
    observed = base.get_obs()
    actual = np.asarray(observed.cpu() if hasattr(observed, "cpu") else observed)
    if not np.allclose(actual.reshape(-1), scene.initial_observation.reshape(-1), atol=1e-6, rtol=0):
        raise ValueError(f"restored observation mismatch: {scene.scene_id}")
    return observed


def current_state_snapshot(env):
    """Return one environment's current state in the scene-store representation."""
    def convert(value):
        if isinstance(value, dict):
            return {key: convert(child) for key, child in value.items()}
        array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        return np.array(array[0], copy=True)

    return convert(env.unwrapped.get_state_dict())
