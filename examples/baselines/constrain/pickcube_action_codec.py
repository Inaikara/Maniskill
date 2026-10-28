"""Versioned PickCube action encoding for the frozen controller-range policy."""

import hashlib
import json

import torch


CONTROLLER_RANGE = {
    "name": "pickcube_controller_range_v1",
    "version": 1,
    "scale": [0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 1.0],
    "offset": [0.0] * 7,
    "coordinates": "native normalized pd_ee_delta_pose controller coordinates; not meters or radians",
}


def validate_profile(profile):
    if profile != CONTROLLER_RANGE:
        raise ValueError("unknown or mismatched PickCube action encoding profile")
    return profile


def profile_sha256(profile=CONTROLLER_RANGE):
    validate_profile(profile)
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate(actions, profile):
    validate_profile(profile)
    actions = torch.as_tensor(actions)
    if actions.shape[-1] != 7 or not torch.isfinite(actions).all():
        raise ValueError("PickCube action codec expects finite seven-channel actions")
    return actions, actions.new_tensor(profile["scale"])


def encode_native(actions, profile=CONTROLLER_RANGE):
    actions, scale = _validate(actions, profile)
    if bool((actions[..., :6].abs() > 0.350001).any()) or bool((actions[..., 6].abs() > 1.000001).any()):
        raise ValueError("native action exceeds frozen PickCube controller-range bounds")
    return actions / scale


def decode_to_native(actions, profile=CONTROLLER_RANGE):
    actions, scale = _validate(actions, profile)
    return actions * scale


def decode_prediction_to_native(actions, profile=CONTROLLER_RANGE):
    """Decode model output without clipping and enforce the native controller bound."""
    native = decode_to_native(actions, profile)
    if bool((native.abs() > 1.000001).any()):
        raise ValueError("decoded PickCube prediction exceeds native controller bounds")
    return native
