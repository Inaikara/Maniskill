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
        if not np.isfinite(self.translation_scale_m) or self.translation_scale_m <= 0 \
                or not np.isfinite(self.action_limit) or self.action_limit <= 0:
            raise ValueError("controller scale and action bound must be finite and positive")
        target = tcp + np.clip(action[:2], -self.action_limit, self.action_limit) * self.translation_scale_m
        safe_radius = radius + self.boundary_epsilon_m
        if np.linalg.norm(tcp - center) >= safe_radius and not self.segment_violates(
                tcp, target, center, radius):
            action[:2] = np.clip(action[:2], -self.action_limit, self.action_limit)
            return action
        # Score boundary endpoints by displacement from the requested endpoint;
        # reject any candidate whose executed chord still crosses the disk.
        angles = np.linspace(0.0, 2 * np.pi, self.angular_candidates, endpoint=False)
        candidates = center + safe_radius * np.stack((np.cos(angles), np.sin(angles)), axis=1)
        safe = [point for point in candidates
                if not self.segment_violates(tcp, point, center, radius)]
        if not safe:
            raise ValueError("TCP starts inside the forbidden disk; XY constraint cannot recover")
        endpoint = min(safe, key=lambda point: float(np.sum((point - target) ** 2)))
        action[:2] = np.clip((endpoint - tcp) / self.translation_scale_m,
                             -self.action_limit, self.action_limit)
        executed = tcp + action[:2] * self.translation_scale_m
        if self.segment_violates(tcp, executed, center, radius):
            # Bounds may make the nearest boundary endpoint unreachable this step.
            # Select the furthest safe point along its action chord by bisection.
            delta = executed - tcp
            lo, hi = 0.0, 1.0
            for _ in range(48):
                mid = (lo + hi) / 2
                if self.segment_violates(tcp, tcp + mid * delta, center, radius):
                    hi = mid
                else:
                    lo = mid
            action[:2] *= lo
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
