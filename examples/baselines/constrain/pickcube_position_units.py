"""Fixed PickCube position-to-controller-unit observation encoding."""

import hashlib
import json

import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import NAMED_FIELDS

PROFILE = "pickcube_position_controller_units_v1"
POSITION_REFERENCE_M = 0.1
OBS_DIM = 42
FLAT_DIM = 45


def _named_offsets(fields):
    offsets = {}
    start = 0
    for name, width in fields:
        if name in offsets or int(width) < 1:
            raise ValueError("observation field schema contains duplicate or invalid fields")
        offsets[name] = (start, start + int(width))
        start += int(width)
    if start != FLAT_DIM:
        raise ValueError("PickCube named observation schema must total 45 fields")
    return offsets


def position_indices(fields=NAMED_FIELDS):
    """Return position channel indices derived only from the named field schema."""
    offsets = _named_offsets(fields)
    selections = {
        "extra/tcp_pose": (0, 3),
        "extra/goal_pos": (0, 3),
        "extra/obj_pose": (0, 3),
        "extra/tcp_to_obj_pos": (0, 3),
        "extra/obj_to_goal_pos": (0, 3),
    }
    indices = []
    for field, (begin, end) in selections.items():
        if field not in offsets:
            raise ValueError(f"PickCube observation schema is missing {field}")
        field_start, field_end = offsets[field]
        pose_field = field in ("extra/tcp_pose", "extra/obj_pose")
        if pose_field and field_end - field_start != 7:
            raise ValueError(f"PickCube pose schema width mismatch for {field}")
        if not pose_field and field_end - field_start != 3:
            raise ValueError(f"PickCube position schema width mismatch for {field}")
        if field_start >= OBS_DIM:
            continue
        indices.extend(range(field_start + begin, min(field_start + end, OBS_DIM)))
    if len(indices) != 15:
        raise ValueError("PickCube state schema must expose exactly 15 meter-position values")
    return tuple(indices)


POSITION_INDICES = position_indices()
SCALE_VECTOR = tuple(10.0 if i in POSITION_INDICES else 1.0 for i in range(OBS_DIM))
PROFILE_BODY = {
    "profile": PROFILE,
    "version": 1,
    "obs_dim": OBS_DIM,
    "fields": [[name, width] for name, width in NAMED_FIELDS if name != "extra/pickcube_obstacle"],
    "scale": list(SCALE_VECTOR),
    "offset": [0.0] * OBS_DIM,
    "position_reference_m": POSITION_REFERENCE_M,
    "position_fields": ["extra/tcp_pose[0:3]", "extra/goal_pos[0:3]", "extra/obj_pose[0:3]",
                        "extra/tcp_to_obj_pos[0:3]", "extra/obj_to_goal_pos[0:3]"],
}
PROFILE_SHA256 = hashlib.sha256(
    json.dumps(PROFILE_BODY, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def profile_metadata():
    return {**PROFILE_BODY, "sha256": PROFILE_SHA256}


def validate_profile(metadata):
    expected = profile_metadata()
    candidate = dict(metadata) if isinstance(metadata, dict) else metadata
    source = candidate.pop("source_identity", None) if isinstance(candidate, dict) else None
    if candidate != expected:
        raise ValueError("PickCube fixed position-unit profile/schema/hash mismatch")
    if source is not None and (source.get("h5_sha256") is None
                               or source.get("manifest_sha256") is None):
        raise ValueError("PickCube fixed position-unit source identity is incomplete")
    return metadata


def encode_observation(values, fields=NAMED_FIELDS):
    if fields != NAMED_FIELDS:
        raise ValueError("PickCube fixed position-unit profile requires the frozen named schema")
    x = torch.as_tensor(values)
    if x.shape[-1] != OBS_DIM:
        raise ValueError("PickCube position-unit input must end in exactly 42 features")
    scale = torch.as_tensor(SCALE_VECTOR, dtype=x.dtype, device=x.device)
    return x * scale


def source_identity():
    return {"profile": PROFILE, "profile_sha256": PROFILE_SHA256,
            "fields": PROFILE_BODY["fields"], "scale": list(SCALE_VECTOR),
            "offset": [0.0] * OBS_DIM, "fit_from_data": False}
