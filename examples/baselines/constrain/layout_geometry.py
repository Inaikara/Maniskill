"""Pure XY geometry for task-specific obstacle layout sampling."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LayoutGeometryConfig:
    table_bounds: tuple = (-0.3, -0.3, 0.3, 0.3)
    min_transport_distance: float = 0.1
    min_base_line_distance: float = 0.05
    waypoint_offset_range: tuple = (0.1, 0.2)
    waypoint_lead_fraction: float = 0.25
    max_attempts: int = 100

    def __post_init__(self):
        bounds = np.asarray(self.table_bounds, dtype=np.float64)
        offsets = np.asarray(self.waypoint_offset_range, dtype=np.float64)
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError("table_bounds must contain four finite values")
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("table_bounds must have positive width and height")
        if offsets.shape != (2,) or not np.isfinite(offsets).all():
            raise ValueError("waypoint_offset_range must contain two finite values")
        if offsets[0] <= 0 or offsets[0] > offsets[1]:
            raise ValueError("waypoint_offset_range must be positive and ordered")
        if not math.isfinite(self.min_transport_distance) or self.min_transport_distance < 0:
            raise ValueError("min_transport_distance must be finite and non-negative")
        if not math.isfinite(self.min_base_line_distance) or self.min_base_line_distance <= 0:
            raise ValueError("min_base_line_distance must be finite and positive")
        if not math.isfinite(self.waypoint_lead_fraction) or self.waypoint_lead_fraction < 0:
            raise ValueError("waypoint_lead_fraction must be finite and non-negative")
        if not isinstance(self.max_attempts, int) or self.max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")


@dataclass(frozen=True)
class LayoutValidation:
    valid: bool
    reason: str | None
    transport_distance: float
    base_line_distance: float


@dataclass(frozen=True)
class WaypointSample:
    waypoint_xy: np.ndarray | None
    attempts: int
    reason: str | None


def validate_layout(start, goal, base, config):
    start = _xy(start, "start")
    goal = _xy(goal, "goal")
    base = _xy(base, "base")
    direction = goal - start
    distance = float(np.linalg.norm(direction))
    base_distance = (
        abs(float(_cross(direction, base - start))) / distance
        if distance > 0
        else float(np.linalg.norm(base - start))
    )
    xmin, ymin, xmax, ymax = config.table_bounds
    if not all(
        xmin <= point[0] <= xmax and ymin <= point[1] <= ymax
        for point in (start, goal)
    ):
        reason = "table_bounds"
    elif distance < config.min_transport_distance:
        reason = "transport_distance"
    elif distance == 0 or base_distance < config.min_base_line_distance:
        reason = "base_line_distance"
    else:
        reason = None
    return LayoutValidation(reason is None, reason, distance, base_distance)


def sample_same_side_waypoint(start, goal, base, rng, config):
    start = _xy(start, "start")
    goal = _xy(goal, "goal")
    base = _xy(base, "base")
    direction = goal - start
    distance = float(np.linalg.norm(direction))
    if distance == 0:
        return WaypointSample(None, 0, "degenerate_segment")
    along = direction / distance
    base_side = float(_cross(direction, base - start))
    if base_side == 0:
        return WaypointSample(None, 0, "base_on_transport_line")
    side_sign = math.copysign(1.0, base_side)
    perpendicular = np.array([-along[1], along[0]]) * side_sign
    midpoint = 0.5 * (start + goal)
    xmin, ymin, xmax, ymax = config.table_bounds
    for attempt in range(1, config.max_attempts + 1):
        offset = float(rng.uniform(*config.waypoint_offset_range))
        lead = config.waypoint_lead_fraction * distance
        candidate = midpoint + perpendicular * offset + along * lead
        if xmin <= candidate[0] <= xmax and ymin <= candidate[1] <= ymax:
            return WaypointSample(candidate, attempt, None)
    return WaypointSample(None, config.max_attempts, "same_side_waypoint_space_exhausted")


def _xy(value, name):
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must contain two finite XY values")
    return point


def _cross(left, right):
    return left[0] * right[1] - left[1] * right[0]
