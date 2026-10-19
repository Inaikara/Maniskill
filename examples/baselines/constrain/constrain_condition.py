"""Relative obstacle condition for the constrain policy (Ticket 03).

The condition of one observation frame is the world-parallel relative geometry
between the vertical cylinder obstacle and the TCP:

    [obstacle_x - tcp_x, obstacle_y - tcp_y, obstacle_radius]   (meters)

It is anchored at the TCP position but keeps the world XY axis directions —
the TCP orientation never rotates the condition axes. The proven flattened
``StackCubeObstacle-v1`` state layout (verified against the real environment,
see ``tests/test_constrain_condition.py``) is:

    qpos [0, 9) | qvel [9, 18) | tcp raw_pose [18, 25) | cubeA pose [25, 32)
    | cubeB pose [32, 39) | tcp->cubeA [39, 42) | tcp->cubeB [42, 45)
    | cubeA->cubeB [45, 48) | obstacle (center_x, center_y, radius) [48, 51)

so the TCP world xy lives at indices 18/19 and the appended obstacle features
at 48/49/50.

Normalization is one explicit, fixed and configurable affine map of each
feature to [-1, 1] followed by clipping — never per-batch statistics. The
same implementation instance (same ``ConditionNormConfig``) is used by the
training loss and by the closed-loop inference adapter, and the exact config
is stored in every checkpoint.

The current 3d condition is repeated over the 16-step prediction horizon: it
encodes the *current* relative geometry, never a future TCP/obstacle path.
"""

from dataclasses import dataclass

import torch

TCP_XY_INDICES = (18, 19)
OBSTACLE_INDICES = (48, 49, 50)
FLAT_OBS_DIM = 51

_TRANSFORM_DOC = (
    "per-feature affine map [lo, hi] -> [-1, 1], then clip to [-1, 1]"
)


@dataclass(frozen=True)
class ConditionNormConfig:
    """Fixed, configurable normalization shared by training and inference.

    ``rel_xy_range`` normalizes obstacle-minus-TCP world XY offsets;
    ``radius_range`` keeps the real cylinder radius semantics (linear in the
    actual radius, e.g. the default maps 0.02 m -> -0.5 and 0.04 m -> 0.0).
    Out-of-range values clip to [-1, 1]. Neither range reuses the cylinder
    world-coordinate sampling ranges, and no batch statistics are involved.
    """

    rel_xy_range: tuple = (-1.0, 1.0)
    radius_range: tuple = (0.0, 0.08)
    clip: bool = True

    def __post_init__(self):
        for name in ("rel_xy_range", "radius_range"):
            lo, hi = getattr(self, name)
            if not (hi > lo):
                raise ValueError(f"{name} must be (lo, hi) with hi > lo, got {name}={lo, hi}")

    def to_dict(self):
        return dict(
            transform=_TRANSFORM_DOC,
            rel_xy_range=[float(v) for v in self.rel_xy_range],
            radius_range=[float(v) for v in self.radius_range],
            clip=bool(self.clip),
            tcp_xy_indices=[int(v) for v in TCP_XY_INDICES],
            obstacle_indices=[int(v) for v in OBSTACLE_INDICES],
            flat_obs_dim=FLAT_OBS_DIM,
            units="meters",
        )

    @classmethod
    def from_dict(cls, d):
        if d["tcp_xy_indices"] != list(TCP_XY_INDICES) or d["obstacle_indices"] != list(OBSTACLE_INDICES):
            raise ValueError(
                f"Recorded obs indices {d['tcp_xy_indices']}/{d['obstacle_indices']} "
                f"do not match this module's proven schema {TCP_XY_INDICES}/{OBSTACLE_INDICES}"
            )
        return cls(
            rel_xy_range=tuple(d["rel_xy_range"]),
            radius_range=tuple(d["radius_range"]),
            clip=bool(d["clip"]),
        )


def _normalize(values, lo, hi, clip):
    scaled = 2.0 * (values - lo) / (hi - lo) - 1.0
    return scaled.clamp(-1.0, 1.0) if clip else scaled


def _check_layout(stacked: torch.Tensor):
    if stacked.shape[-1] != FLAT_OBS_DIM:
        raise ValueError(
            f"Expected the proven {FLAT_OBS_DIM}d obstacle flat obs layout, "
            f"got shape {tuple(stacked.shape)}"
        )
    if stacked.shape[-2] != 2:
        raise ValueError(
            f"Expected the queued 2-frame observation history, got "
            f"shape {tuple(stacked.shape)}"
        )


def latest_relative_condition(stacked, norm_cfg: ConditionNormConfig = None):
    """Normalized ``[obstacle_x - tcp_x, obstacle_y - tcp_y, radius]`` read
    from the most recent frame of the queued 2-frame history.

    Accepts ``(2, 51)`` or ``(batch, 2, 51)`` flat obstacle observations and
    returns ``(3,)`` or ``(batch, 3)``. The latest frame (index -1) is always
    the source: never the initial position or a stale condition.
    """
    norm_cfg = norm_cfg if norm_cfg is not None else ConditionNormConfig()
    stacked = torch.as_tensor(stacked)
    _check_layout(stacked)
    latest = stacked[..., -1, :]
    rel_xy = latest[..., list(OBSTACLE_INDICES[:2])] - latest[..., list(TCP_XY_INDICES)]
    radius = latest[..., OBSTACLE_INDICES[2]:OBSTACLE_INDICES[2] + 1]
    cond = torch.cat(
        [
            _normalize(rel_xy, *norm_cfg.rel_xy_range, norm_cfg.clip),
            _normalize(radius, *norm_cfg.radius_range, norm_cfg.clip),
        ],
        dim=-1,
    )
    return cond


def build_hint(condition, pred_horizon: int):
    """Repeat the current condition over the prediction horizon.

    ``(3,)``/``(batch, 3)`` -> ``(1, pred_horizon, 3)``/``(batch, pred_horizon,
    3)``: the condition represents current relative geometry and is constant
    across the predicted steps.
    """
    condition = torch.as_tensor(condition)
    if condition.dim() == 1:
        condition = condition[None]
    return condition[:, None, :].expand(-1, pred_horizon, -1)
