"""R43 frozen-asset clearance geometry and fail-closed admission primitives."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_body_hint import T_TCP_HAND, T_TCP_LINK7

BODY_MARGIN_M = 0.02
CUBE_XY_MARGIN_M = 0.01
SAMPLE_SPHERE_PADDING_M = 0.005
MAX_POINTS_PER_LINK = 256
GEOMETRY_LINKS = ("panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger")
PRECHECK_SCHEMA = "pickcube_clearance_precheck_r43_v1"
TRAINING_SCHEMA = "pickcube_clearance_training_admission_r43_v1"
GEOMETRY_TOLERANCE_M = 1e-5
BODY_PREDICTION_P95_M = 0.01
RESPONSE_IMPROVEMENT = 0.10
SWEPT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def cylinder_sdf(points, center_xy, radius, height):
    """Exact finite vertical cylinder SDF, in meters."""
    radial = torch.linalg.vector_norm(points[..., :2] - torch.as_tensor(center_xy, device=points.device,
                                      dtype=points.dtype), dim=-1) - radius
    vertical = torch.abs(points[..., 2] - height / 2) - height / 2
    outside = torch.linalg.vector_norm(torch.stack((radial.clamp_min(0), vertical.clamp_min(0)), dim=-1), dim=-1)
    return outside + torch.maximum(radial, vertical).clamp_max(0)


def padded_cylinder_sdf(points, center_xy, radius, height):
    return cylinder_sdf(points, center_xy, radius, height) - SAMPLE_SPHERE_PADDING_M


def transform_link_points_to_world(points_link, tcp_world, tcp_to_link):
    """Apply calibrated T_tcp_link to link-local row vectors, then world TCP."""
    points = np.asarray(points_link, dtype=np.float64)
    tcp = np.asarray(tcp_world, dtype=np.float64)
    relation = np.asarray(tcp_to_link, dtype=np.float64)
    link = tcp @ relation
    return points @ link[:3, :3].T + link[:3, 3]


def transform_link_points_to_world_inverse_bug(points_link, tcp_world, tcp_to_link):
    """Retained only as a test witness; never used by the R43 pipeline."""
    points = np.asarray(points_link, dtype=np.float64)
    tcp = np.asarray(tcp_world, dtype=np.float64)
    local = np.linalg.inv(np.asarray(tcp_to_link, dtype=np.float64))
    return points @ (tcp @ local)[:3, :3].T + (tcp @ local)[:3, 3]


def carried_cube_xy_gap(raw, center_xy, forbidden_radius):
    """Native obstacle forbidden-disk gap for carried cube center, always XY-only."""
    obs = torch.as_tensor(raw)
    cube_xy = obs[..., 29:31]
    center = torch.as_tensor(center_xy, dtype=obs.dtype, device=obs.device)
    gap = torch.linalg.vector_norm(cube_xy - center, dim=-1) - forbidden_radius
    grasped = obs[..., 18] > 0.5
    return torch.where(grasped, gap, torch.full_like(gap, float("inf")))


def native_forbidden_radius(cylinder_radius, cube_half_size, safety_margin):
    """Same formula as PickCubeObstacleEnv._forbidden_radius."""
    return float(cylinder_radius) + np.sqrt(2.0) * float(cube_half_size) + float(safety_margin)


def clearance_hinge(gap, margin):
    return torch.relu(torch.as_tensor(margin, dtype=gap.dtype, device=gap.device) - gap).square()


def clearance_proxy_loss(body_points_world, cylinder_center_xy, cylinder_radius,
                         cylinder_height, raw_observation, cube_forbidden_radius,
                         body_margin=BODY_MARGIN_M, cube_margin=CUBE_XY_MARGIN_M):
    """Differentiable training-only body SDF and carried-cube XY margin costs.

    Inputs are one predicted state per batch row. Empty component masks return a
    graph-connected zero, and cylinder height affects the body SDF only.
    """
    sdf = padded_cylinder_sdf(body_points_world, cylinder_center_xy,
                              cylinder_radius, cylinder_height)
    body_cost = clearance_hinge(sdf, body_margin).mean()
    cube_gap = carried_cube_xy_gap(raw_observation, cylinder_center_xy,
                                   cube_forbidden_radius)
    active = torch.isfinite(cube_gap)
    if bool(active.any()):
        cube_cost = clearance_hinge(cube_gap[active], cube_margin).mean()
    else:
        cube_cost = body_points_world.sum() * 0.0
    return body_cost, cube_cost


def swept_clearance_components(body_points_current, tcp_translation_delta,
                               cube_center_xy_current, cylinder_center_xy,
                               cylinder_radius, cylinder_height, grasped,
                               cube_forbidden_radius, *, fractions=SWEPT_FRACTIONS):
    """Per-row clearance costs on four swept translation samples, excluding t=0."""
    if tuple(fractions) != SWEPT_FRACTIONS:
        raise ValueError("R43 swept fractions are frozen at [0.25, 0.5, 0.75, 1.0]")
    if body_points_current.ndim == 3:
        body_points_current = body_points_current[:, None, :, :]
    f = torch.as_tensor(fractions, dtype=body_points_current.dtype,
                        device=body_points_current.device)
    delta = (tcp_translation_delta[:, None, None, None, :]
             * f[None, :, None, None, None])
    points = body_points_current[:, None, :, :, :] + delta
    sdf = cylinder_sdf(points, cylinder_center_xy[:, None, None, None, :],
                       cylinder_radius[:, None, None, None],
                       cylinder_height[:, None, None, None]) - SAMPLE_SPHERE_PADDING_M
    body_hinge = clearance_hinge(sdf, BODY_MARGIN_M)
    body_by_row = body_hinge.mean(dim=(1, 2, 3))
    body_active = (body_hinge > 0).any(dim=(1, 2, 3))
    body_cost = body_by_row[body_active].mean() if bool(body_active.any()) else body_points_current.sum() * 0.0

    cube_xy = cube_center_xy_current[:, None, :] + tcp_translation_delta[:, None, :2] * f[None, :, None]
    radius = torch.as_tensor(cube_forbidden_radius, dtype=cube_xy.dtype, device=cube_xy.device)
    gap = torch.linalg.vector_norm(cube_xy - cylinder_center_xy[:, None, :], dim=-1) - radius[:, None]
    cube_hinge = clearance_hinge(gap, CUBE_XY_MARGIN_M)
    grasped = torch.as_tensor(grasped, dtype=torch.bool, device=cube_xy.device)
    cube_active = grasped & (cube_hinge > 0).any(dim=1)
    cube_by_row = cube_hinge.mean(dim=1)
    cube_cost = cube_by_row[cube_active].mean() if bool(cube_active.any()) else cube_xy.sum() * 0.0
    return body_cost, cube_cost, {
        "body_active_rows": body_active.sum(), "cube_active_rows": cube_active.sum(),
        "body_min_gap_m": sdf.detach().amin(), "cube_min_gap_m": gap.detach().amin(),
    }


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_training_admission(registration_path, report_path, expected_source_files,
                              expected_data, expected_asset_hashes):
    """Verify immutable registration/report identities and semantic gate criteria."""
    registration_path, report_path = Path(registration_path), Path(report_path)
    registration = json.loads(registration_path.read_text())
    report = json.loads(report_path.read_text())
    reg_digest = registration.get("registration_sha256")
    reg_body = dict(registration)
    reg_body.pop("registration_sha256", None)
    if reg_digest != _canonical_digest(reg_body):
        raise ValueError("training registration checksum mismatch")
    if registration.get("schema") != TRAINING_SCHEMA:
        raise ValueError("wrong clearance training registration schema")
    if registration.get("source_files") != expected_source_files:
        raise ValueError("training registration source identity mismatch")
    if report.get("registration_sha256") != registration.get("precheck_registration_sha256"):
        raise ValueError("precheck belongs to a different frozen registration")
    if registration.get("precheck_sha256") != sha256_file(report_path):
        raise ValueError("training registration does not pin this precheck report")
    if report.get("schema") != PRECHECK_SCHEMA or report.get("status") != "passed":
        raise ValueError("precheck is not a typed passed R43 report")
    if report.get("source_files") != expected_source_files:
        raise ValueError("precheck source identity differs from current source files")
    if any(sha256_file(path) != digest for path, digest in expected_source_files.items()):
        raise ValueError("current source file hash mismatch")
    if report.get("data") != expected_data or registration.get("data") != expected_data:
        raise ValueError("frozen dataset identity mismatch")
    if report.get("asset_hashes") != expected_asset_hashes or registration.get("asset_hashes") != expected_asset_hashes:
        raise ValueError("geometry/dynamics asset identity mismatch")
    metrics = report.get("metrics", {})
    # Acceptance constants are code-level protocol, not editable report fields.
    criteria = (metrics.get("native_point_max_abs_m", float("inf")) <= GEOMETRY_TOLERANCE_M
                and metrics.get("heldout_body_prediction_p95_m", float("inf")) <= BODY_PREDICTION_P95_M
                and metrics.get("heldout_body_mean_improvement_fraction", -float("inf")) >= RESPONSE_IMPROVEMENT
                and metrics.get("response_improvement_fraction", -float("inf")) >= RESPONSE_IMPROVEMENT
                and metrics.get("manifest_radius_max_abs_error_m", float("inf")) <= GEOMETRY_TOLERANCE_M
                and metrics.get("manifest_center_max_abs_error_m", float("inf")) <= GEOMETRY_TOLERANCE_M
                and metrics.get("manifest_height_max_abs_error_m", float("inf")) <= GEOMETRY_TOLERANCE_M
                and report.get("native_fk_validation", {}).get("states") == 36
                and metrics.get("all_four_links_covered") is True
                and metrics.get("cube_xy_kernel_exercised") is True
                and metrics.get("padding_applied") is True)
    if not criteria:
        raise ValueError("R43 semantic precheck criteria failed")
    if report.get("training_permitted") is not True:
        raise ValueError("precheck admission was not granted")
    return {"registration_sha256": reg_digest, "precheck_sha256": sha256_file(report_path),
            "report_schema": PRECHECK_SCHEMA, "verified": True}
