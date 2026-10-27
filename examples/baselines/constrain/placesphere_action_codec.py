"""Versioned PlaceSphere action-space encodings shared by training and rollout."""

import hashlib
import json

import torch


IDENTITY = {
    "name": "identity_native_v1",
    "version": 1,
    "scale": [1.0] * 7,
    "offset": [0.0] * 7,
    "coordinates": "native pd_ee_delta_pose controller coordinates",
}

CONTROLLER_RANGE = {
    "name": "placesphere_controller_range_v1",
    "version": 1,
    "scale": [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 1.0],
    "offset": [0.0] * 7,
    "coordinates": "native normalized pd_ee_delta_pose controller coordinates; not meters or radians",
}


def validate_profile(profile):
    if not isinstance(profile, dict) or profile.get("name") not in (
            IDENTITY["name"], CONTROLLER_RANGE["name"]):
        raise ValueError("unknown or missing PlaceSphere action encoding profile")
    expected = IDENTITY if profile["name"] == IDENTITY["name"] else CONTROLLER_RANGE
    if profile != expected:
        raise ValueError("PlaceSphere action encoding profile parameters mismatch")
    return profile


def profile_sha256(profile):
    validate_profile(profile)
    blob = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _parameters(actions, profile):
    validate_profile(profile)
    actions = torch.as_tensor(actions)
    if actions.shape[-1] != 7 or not torch.isfinite(actions).all():
        raise ValueError("PlaceSphere action codec expects finite seven-channel actions")
    scale = actions.new_tensor(profile["scale"])
    offset = actions.new_tensor(profile["offset"])
    return actions, scale, offset


def encode_native(actions, profile):
    actions, scale, offset = _parameters(actions, profile)
    if profile["name"] == IDENTITY["name"]:
        return actions
    if profile["name"] == CONTROLLER_RANGE["name"]:
        pose_over = actions[..., :6].abs() > 0.350001
        grip_over = actions[..., 6].abs() > 1.000001
        if bool(pose_over.any()) or bool(grip_over.any()):
            raise ValueError("native expert action exceeds the frozen controller-range codec bounds")
    return (actions - offset) / scale


def decode_to_native(actions, profile):
    actions, scale, offset = _parameters(actions, profile)
    if profile["name"] == IDENTITY["name"]:
        return actions
    return actions * scale + offset
