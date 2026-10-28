"""Fixed-frame Panda body-point obstacle hints derived from raw PickCube state."""

import torch

from examples.baselines.constrain.pickcube_constrain import (
    NAMED_FIELDS,
    PickCubeConditionNorm,
)

PROFILE = "pickcube_panda_body_hint9_v1"
RAW_OBSERVATION_DIM = 45
HINT_DIM = 9
PANDA_URDF_SHA256 = "524c6986dbbabe244e05801f262862205e880834d01459b06ee273016d0579ba"
PANDA_MODEL_SOURCE_SHA256 = "bf85577be75b7513802e676decb5543e3e65f2a3dfbc26175dd8b77ae57691f3"
TCP_POSITION = slice(19, 22)
TCP_QUATERNION_WXYZ = slice(22, 26)
OBSTACLE_XY = slice(42, 44)
OBSTACLE_RADIUS = 44

# Native Panda FK constants for T_tcp_hand and T_tcp_link7, meters, row-major.
T_TCP_HAND = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, -0.1034),
    (0.0, 0.0, 0.0, 1.0),
)
T_TCP_LINK7 = (
    (0.70710678, -0.70710678, 0.0, 0.0),
    (0.70710678, 0.70710678, 0.0, 0.0),
    (0.0, 0.0, 1.0, -0.2104),
    (0.0, 0.0, 0.0, 1.0),
)


class BodyHintNorm:
    """Fixed feature scaling; appended values remain meter-valued until /0.5."""

    def to_dict(self):
        return {
            "profile": PROFILE,
            "feature_order": [
                "obstacle_x-tcp_x", "obstacle_y-tcp_y", "cylinder_radius",
                "obstacle_x-hand_x", "obstacle_y-hand_y", "hand_world_z",
                "obstacle_x-link7_x", "obstacle_y-link7_y", "link7_world_z",
            ],
            "units": "meters; world-parallel XY; Z is world height, not surface clearance",
            "original_feature_normalization": PickCubeConditionNorm().to_dict(),
            "added_feature_formula": "divide by 0.5 m then clip to [-1,1]",
            "tcp_position_slice": [19, 22], "tcp_quaternion_slice": [22, 26],
            "quaternion_order": "wxyz",
            "obstacle_xy_slice": [42, 44], "radius_index": 44,
            "T_tcp_hand": [list(row) for row in T_TCP_HAND],
            "T_tcp_link7": [list(row) for row in T_TCP_LINK7],
            "urdf_path": "mani_skill/assets/robots/panda/panda_v2.urdf",
            "urdf_sha256": PANDA_URDF_SHA256,
            "robot_model_source": "mani_skill/agents/robots/panda/panda.py",
            "robot_model_source_sha256": PANDA_MODEL_SOURCE_SHA256,
            "fk_validation": {"existing_frozen_states": 36, "episodes": 9,
                               "frames_per_episode": 4, "max_abs_tolerance": 1e-5},
            "named_fields": NAMED_FIELDS,
            "runtime_geometry_source": "raw 45-D observation only plus fixed Panda frame constants",
        }

    @classmethod
    def from_dict(cls, value):
        if value != cls().to_dict():
            raise ValueError("PickCube body-hint profile or calibrated frames mismatch")
        return cls()


def _rotation_wxyz(q):
    if not torch.isfinite(q).all():
        raise ValueError("TCP quaternion contains non-finite values")
    norm = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    if bool((norm < 1e-8).any()):
        raise ValueError("TCP quaternion has zero norm")
    w, x, y, z = (q / norm).unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(q.shape[:-1] + (3, 3))


def body_hint_from_raw(raw_observation, norm=None):
    """Return normalized 9-D hint from latest raw (45,) or (...,2,45) input."""
    raw = torch.as_tensor(raw_observation)
    if raw.shape[-1] != RAW_OBSERVATION_DIM:
        raise ValueError(f"expected raw PickCube 45-D observation, got {tuple(raw.shape)}")
    if raw.ndim >= 2 and raw.shape[-2] == 2:
        raw = raw[..., -1, :]
    if not torch.isfinite(raw).all():
        raise ValueError("raw PickCube observation contains non-finite values")
    original = torch.cat((raw[..., 42:44] - raw[..., 19:21],
                          raw[..., 44:45]), dim=-1)
    original = original / raw.new_tensor([0.5, 0.5, 0.08])
    original = original.clamp(-1, 1)
    position = raw[..., TCP_POSITION]
    rotation = _rotation_wxyz(raw[..., TCP_QUATERNION_WXYZ])
    obstacle = raw[..., OBSTACLE_XY]
    hand_offset = torch.as_tensor(T_TCP_HAND, dtype=raw.dtype, device=raw.device)
    link7_offset = torch.as_tensor(T_TCP_LINK7, dtype=raw.dtype, device=raw.device)

    def world_point(offset):
        local = offset[:3, 3].expand(position.shape).unsqueeze(-1)
        xyz = position + torch.matmul(rotation, local).squeeze(-1)
        return xyz

    hand = world_point(hand_offset)
    link7 = world_point(link7_offset)
    additions = torch.stack((obstacle[..., 0] - hand[..., 0],
                             obstacle[..., 1] - hand[..., 1], hand[..., 2],
                             obstacle[..., 0] - link7[..., 0],
                             obstacle[..., 1] - link7[..., 1], link7[..., 2]), dim=-1)
    additions = (additions / 0.5).clamp(-1, 1)
    return torch.cat((original, additions), dim=-1)


def body_hint_metadata():
    return BodyHintNorm().to_dict()
