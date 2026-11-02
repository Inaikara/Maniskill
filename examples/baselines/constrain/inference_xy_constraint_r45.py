"""Observation-driven, per-control-step XY detour constraint (R45)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class XYConstraint:
    """Filter native ``pd_ee_delta_pose`` actions; translation is controller units.

    Panda translation channels are normalized by ``translation_scale_m`` (0.1 m
    per action unit); action values are bounded to ``[-action_limit, action_limit]``.
    Orientation, z, and gripper channels pass through untouched.
    """

    translation_scale_m: float = 0.1
    action_limit: float = 0.35
    boundary_epsilon_m: float = 1e-6
    angular_candidates: int = 1440
    working_clearance_m: float = 0.003
    xy_action_bounds: tuple = (-0.35, 0.35)

    @staticmethod
    def segment_violates(start, end, center, radius):
        start, end, center = (np.asarray(v, dtype=np.float64).reshape(2)
                              for v in (start, end, center))
        direction = end - start
        t = np.clip(np.dot(center - start, direction) /
                    max(float(np.dot(direction, direction)), 1e-24), 0.0, 1.0)
        return bool(np.linalg.norm(start + t * direction - center) < radius)

    def filter_action(self, action, tcp_xy, center_xy, forbidden_radius):
        action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
        tcp = np.asarray(tcp_xy, dtype=np.float64).reshape(2)
        center = np.asarray(center_xy, dtype=np.float64).reshape(2)
        radius = float(forbidden_radius)
        if action.size != 7 or not np.isfinite(action).all() or not np.isfinite(tcp).all() \
                or not np.isfinite(center).all() or not np.isfinite(radius) or radius <= 0:
            raise ValueError("action and XY constraint inputs must be finite with positive radius")
        bounds = np.asarray(self.xy_action_bounds, dtype=np.float64)
        if not np.isfinite(self.translation_scale_m) or self.translation_scale_m <= 0 \
                or not np.isfinite(self.action_limit) or self.action_limit <= 0 \
                or bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1] \
                or not np.isfinite(self.working_clearance_m) or self.working_clearance_m < 0:
            raise ValueError("controller scale and action bound must be finite and positive")
        lower, upper = bounds
        requested_xy = np.clip(action[:2], lower, upper)
        target = tcp + requested_xy * self.translation_scale_m
        radial = tcp - center
        distance = float(np.linalg.norm(radial))
        if distance < radius - 1e-12:
            raise ValueError("TCP starts inside the forbidden disk; XY constraint cannot recover")
        safe_radius = radius + self.working_clearance_m
        if distance > safe_radius + 1e-10 and not self.segment_violates(
                tcp, target, center, safe_radius):
            action[:2] = requested_xy
            return action
        if distance <= safe_radius + 1e-10:
            # Already in the working buffer: make an outward radial command.
            direction = radial / distance
        else:
            # The shortest safe straight segments are tangents to the working disk.
            direction = radial / distance
            perpendicular = np.array([-direction[1], direction[0]])
            cosine = safe_radius / distance
            sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
            candidates = [center + (safe_radius + 1e-9) *
                          (cosine * direction + sign * sine * perpendicular)
                          for sign in (-1.0, 1.0)]
            endpoint = min(candidates, key=lambda point: float(np.sum((point-target) ** 2)))
            delta = endpoint - tcp
            proposed = delta / self.translation_scale_m
            scale = 1.0
            for value in proposed:
                if value > upper:
                    scale = min(scale, upper / value)
                elif value < lower:
                    scale = min(scale, lower / value)
            action[:2] = proposed * scale
            executed = tcp + action[:2] * self.translation_scale_m
            if self.segment_violates(tcp, executed, center, safe_radius):
                raise ValueError("controller bounds cannot preserve XY working clearance")
            return action
        proposed = direction * min(self.working_clearance_m + max(safe_radius-distance, 0.0),
                                  self.translation_scale_m * max(abs(lower), abs(upper)))
        proposed /= self.translation_scale_m
        scale = 1.0
        for value in proposed:
            if value > upper:
                scale = min(scale, upper / value)
            elif value < lower:
                scale = min(scale, lower / value)
        action[:2] = proposed * scale
        return action


def validate_tcp_trace(tcp_xy, center_xy, forbidden_radius, first_success_step=None):
    points = np.asarray(tcp_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    radius = float(forbidden_radius)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 \
            or not np.isfinite(points).all() or not np.isfinite(center).all() \
            or not np.isfinite(radius) or radius <= 0:
        raise ValueError("trace, center, and radius must be finite valid XY geometry")
    samples = np.linalg.norm(points - center, axis=1) < radius
    segments = [i for i, (a, b) in enumerate(zip(points[:-1], points[1:]))
                if XYConstraint.segment_violates(a, b, center, radius)]
    first_valid = (None if first_success_step is None else
                   0 <= int(first_success_step) < len(points)
                   and not bool(samples[int(first_success_step)])
                   and int(first_success_step) not in segments
                   and int(first_success_step) - 1 not in segments)
    return {"violating_samples": np.flatnonzero(samples).astype(int).tolist(),
            "violating_segments": segments,
            "first_success_step": first_success_step,
            "first_success_step_valid": first_valid,
            "zero_xy_violations": not bool(samples.any()) and not bool(segments)}
