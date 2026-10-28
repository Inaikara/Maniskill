"""Typed PickCube direct-only observation z-score profile."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import NAMED_FIELDS

PROFILE = "pickcube_direct_zscore_v1"
FLOOR = 0.01
OBS_DIM = 42
DIRECT_IDS = list(range(1000))
FIT_ALGORITHM = "float64_population_moments_all_native_T+1_frames_direct_ids_0_999_v1"
FIELDS = [[name, width] for name, width in NAMED_FIELDS if name != "extra/pickcube_obstacle"]


def fit_direct_observations(episodes, episode_ids=DIRECT_IDS, source_identity=None):
    """Fit once over native T+1 observations for exactly direct IDs 0..999.

    episodes maps episode ID to its raw (T+1,45) native observation array.
    """
    if list(episode_ids) != DIRECT_IDS or set(episodes) != set(DIRECT_IDS):
        raise ValueError("normalizer fitting requires exactly direct episode IDs 0..999")
    count, total, squares = 0, np.zeros(OBS_DIM, np.float64), np.zeros(OBS_DIM, np.float64)
    minima = np.full(OBS_DIM, np.inf, np.float64)
    for episode_id in DIRECT_IDS:
        values = np.asarray(episodes[episode_id])
        if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
            raise ValueError(f"direct episode {episode_id} must contain finite (T+1,45) raw observations")
        raw = values[:, :OBS_DIM].astype(np.float64, copy=False)
        count += len(raw)
        total += raw.sum(axis=0, dtype=np.float64)
        squares += np.square(raw).sum(axis=0, dtype=np.float64)
        minima = np.minimum(minima, raw.min(axis=0))
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
    floored = std < FLOOR
    body = {
        "profile": PROFILE, "obs_dim": OBS_DIM, "fields": FIELDS,
        "direct_episode_ids": DIRECT_IDS, "fit_algorithm": FIT_ALGORITHM,
        "fit_frame_count": count, "floor": FLOOR,
        "source_identity": source_identity,
        "count": [count] * OBS_DIM,
        "mean": mean.tolist(), "population_std": std.tolist(),
        "minimum": minima.tolist(), "rms": np.sqrt(squares / count).tolist(),
        "floored_fields": [_field_name(i) for i in np.flatnonzero(floored)],
    }
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def _field_name(index):
    for name, width in FIELDS:
        start = sum(w for _, w in FIELDS[:FIELDS.index([name, width])])
        if start <= index < start + width:
            return f"{name}[{index-start}]"
    raise IndexError(index)


def validate_stats(stats):
    if not isinstance(stats, dict) or stats.get("profile") != PROFILE or stats.get("obs_dim") != OBS_DIM:
        raise ValueError("PickCube observation normalizer profile/dimension mismatch")
    if stats.get("fields") != FIELDS or stats.get("direct_episode_ids") != DIRECT_IDS or stats.get("fit_algorithm") != FIT_ALGORITHM:
        raise ValueError("PickCube observation normalizer source contract mismatch")
    source = stats.get("source_identity")
    if source is not None and (source.get("selected_class") != "direct"
                               or source.get("selected_episode_ids") != DIRECT_IDS
                               or not source.get("h5_sha256") or not source.get("manifest_sha256")):
        raise ValueError("PickCube observation normalizer source identity mismatch")
    if stats.get("floor") != FLOOR or stats.get("fit_frame_count", 0) <= 0:
        raise ValueError("PickCube observation normalizer floor/count mismatch")
    for key in ("mean", "population_std", "minimum", "rms", "count"):
        values = np.asarray(stats.get(key), dtype=np.float64)
        if values.shape != (OBS_DIM,) or not np.isfinite(values).all():
            raise ValueError(f"PickCube observation normalizer {key} must contain 42 finite values")
    if not np.all(np.asarray(stats["count"]) == stats["fit_frame_count"]):
        raise ValueError("PickCube per-feature counts must include every selected direct T+1 frame")
    if np.any(np.asarray(stats["population_std"]) < 0):
        raise ValueError("PickCube population standard deviations must be nonnegative")
    expected = dict(stats)
    digest = expected.pop("sha256", None)
    actual = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != actual:
        raise ValueError("PickCube observation normalizer hash mismatch")
    return stats


def normalize_observation(values, stats):
    validate_stats(stats)
    x = torch.as_tensor(values, dtype=torch.float32)
    if x.shape[-1] != OBS_DIM:
        raise ValueError("PickCube normalized observations must end in 42 features")
    mean = torch.as_tensor(stats["mean"], dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(stats["population_std"], dtype=x.dtype, device=x.device).clamp_min(FLOOR)
    return (x - mean) / scale


def fit_frozen_direct_dataset(data_path, manifest_path, expected_h5_sha256,
                              expected_manifest_sha256):
    """Read and fit the frozen 1,000 direct trajectories, rejecting other sources."""
    data_path, manifest_path = Path(data_path), Path(manifest_path)
    h5_hash, manifest_hash = sha256_path(data_path), sha256_path(manifest_path)
    if h5_hash != expected_h5_sha256 or manifest_hash != expected_manifest_sha256:
        raise ValueError("frozen PickCube HDF5/manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("episodes", [])
    by_id = {int(row["episode_id"]): row for row in rows}
    if (manifest.get("format") != "pickcube-fullscene500-v1" or len(rows) != 1500
            or set(by_id) != set(range(1500))
            or manifest.get("class_ids", {}).get("direct") != DIRECT_IDS
            or manifest.get("class_ids", {}).get("obstacle") != list(range(1000, 1500))
            or any(by_id[i].get("kind") != "direct" for i in DIRECT_IDS)):
        raise ValueError("normalizer source must have exactly frozen direct IDs 0..999")
    episodes = {}
    with h5py.File(data_path, "r") as handle:
        if len(handle) != 1500:
            raise ValueError("normalizer source HDF5 must contain exactly 1,500 episodes")
        for episode_id in DIRECT_IDS:
            group = handle[f"traj_{episode_id}"]
            if group.attrs.get("kind") != "direct":
                raise ValueError(f"traj_{episode_id}: HDF5 kind is not direct")
            obs = np.asarray(group["obs"][:])
            if obs.shape != (len(group["actions"]) + 1, 45):
                raise ValueError(f"traj_{episode_id}: expected native T+1 observations with 45 features")
            episodes[episode_id] = obs
    return fit_direct_observations(episodes, source_identity={
        "selected_class": "direct", "selected_episode_ids": DIRECT_IDS,
        "h5_sha256": h5_hash, "manifest_sha256": manifest_hash,
        "data_path": str(data_path), "manifest_path": str(manifest_path),
    })


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
