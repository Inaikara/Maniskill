"""Action-first constructive StackCubeObstacle demonstration collection.

Builds obstacle demonstrations by direct ``pd_ee_delta_pose`` feedback waypoint
control — no RRT, no MPLib, no joint-trajectory search, no replay conversion.
The policy drives the TCP through explicit Cartesian waypoints and reads the
live TCP pose every step, so approximate tracking is inherent and recorded
rather than gated.

Pipeline (single aggregate ``trajectory.h5``/``trajectory.json`` plus
``report.json``), obstacle-last by construction:

1. **Sanity episode** (no real obstacle): one full pick/carry/place with the
   cylinder parked far outside the workspace, proving the waypoint policy
   achieves a real stack before any template work.
2. **Action templates**: for each template the cube pair (A = pick, B = stack
   target) and a side-detour waypoint W are sampled *independently of any
   obstacle* (W = AB midpoint + perpendicular offset + forward lead). The
   template is then really executed without an obstacle (cylinder parked far
   away) from a fixed reset seed, and the actual swept robot/cube footprint is
   recorded per physics step (collision-geometry vertices projected below the
   cylinder top) together with the TCP/cube traces of the horizontal carry
   phases. Only stacking templates are kept.
3. **Cylinder sampling around the fixed action**: for each reuse slot a
   cylinder is sampled inside the direct A->B blocking strip (so the direct
   path is genuinely blocked) but outside every recorded footprint point by
   the requested clearance (``roomy``/``tight``). Cheap geometry is rejected
   first; accepted candidates are then re-executed with real physics from the
   template's exact reset seed, cube layout and identical waypoint plan.
   ``reuse_count`` therefore means that many DIFFERENT cylinder placements for
   the same path and the same initial robot/cube state.
4. **Acceptance**: physics flags only — ``stack_success``, no
   ``obstacle_collision``, no ``obstacle_crossing`` (the post-grasp forbidden
   disk). The initial robot/cube state of every reuse episode is verified
   bit-identical against the recorded aggregate HDF5 rows.

Commanded carry height is constant (grasp z + ``lift_height``) over the two
horizontal carry phases; grasp, lift and place are vertical; orientation is
the reset (downward) orientation held by zero rotation deltas — tracking
errors, actual carry height and orientation drift are recorded, never gated.
Failed recordings discard their temporary artifacts and never touch the
aggregate output.
"""

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

import examples.baselines.constrain  # noqa: F401  (registers StackCubeObstacle-v1)
from examples.baselines.constrain.stackcube_obstacle import ENV_ID
from mani_skill.utils import common
from mani_skill.utils.io_utils import dump_json

TARGET_CONTROL_MODE = "pd_ee_delta_pose"
CUBE_HALF_SIZE = 0.02
CUBE_XY_RADIUS = float(np.hypot(CUBE_HALF_SIZE, CUBE_HALF_SIZE))
# Measured panda ``pd_ee_delta_pose`` controller facts: position action unit is
# 0.1 m and rotation action unit is 0.1 rad (compact axis-angle, norm-clipped).
# This policy always commands zero rotation, so the reset (downward) TCP
# orientation is held by the controller itself; cube yaw variety in the sampled
# layouts does not change the TCP orientation because no rotation is commanded.
POS_ACTION_SCALE = 0.1
GRIPPER_CLOSE = -1.0
GRIPPER_OPEN = 1.0
GRIPPER_AXIS = 6

# Detour waypoint: sideways offset from the AB midpoint plus a forward lead
# along AB (toward B), so the return carry leg crosses the blocking strip's
# y-band already past the obstacle. The sideways offset must exceed the
# measured palm-horn reach (0.10 m, see README) plus the cylinder radius.
WAYPOINT_OFFSET_RANGE = (0.13, 0.17)
WAYPOINT_LEAD_FRACTION = 0.25
# Reachable envelope for the detour waypoint (panda base at (-0.615, 0)).
WAYPOINT_XY_BOX = (-0.30, -0.10, 0.30, 0.45)  # xmin, ymin, xmax, ymax
WAYPOINT_MAX_BASE_DIST = 0.66

# Template footprint model. The cylinder must clear the *actual* recorded
# robot collision geometry (vertices below the cylinder top), not just the TCP
# path: the measured panda palm hull carries horn rails 0.10 m out in the
# finger-separation axis at TCP z + 0.037..0.049, which dip below a 0.18 m
# cylinder top and are what actually contacts the rim.
FOOTPRINT_Z_SLACK = 0.005
FOOTPRINT_MARGIN = 0.01
# Cylinder candidates are sampled along the direct AB segment (the blocking
# strip) with a small lateral jitter; the strip constraint itself enforces
# blocking, so only the endpoints and the footprint reject candidates.
CYLINDER_STRIP_FRACTION = (0.30, 0.70)
CYLINDER_ENDPOINT_SLACK = 0.005
CLEARANCE_TARGETS = ("roomy", "tight")

STATE_LEAVES = ("actors/cubeA", "actors/cubeB")
PANDA_BASE_XY = (-0.615, 0.0)


@dataclass(frozen=True)
class LayoutConfig:
    """Sampling ranges for cube pairs and cylinders around successful actions."""

    cube_xy_range: tuple = (-0.15, 0.05, 0.18, 0.30)
    obstacle_xy_range: tuple = (-0.15, 0.15, 0.18, 0.30)
    radius_range: tuple = (0.02, 0.04)
    min_segment_clearance: float = 0.01
    max_separation: float = 0.30
    max_reach_alignment: float = 0.5
    base_xy: tuple = PANDA_BASE_XY
    max_attempts: int = 5000


@dataclass(frozen=True)
class Config:
    """Recordable configuration for constructive collection."""

    target_successes: int = 10
    reuse_count: int = 2
    seed: int = 0
    max_templates: int = 200
    max_episodes: int = 400
    # Physics attempts per reuse slot (half preferred clearance label, half
    # fallback label).
    max_cylinder_attempts: int = 40
    # Waypoint policy.
    waypoint_tolerance: float = 0.004
    max_steps_per_waypoint: int = 60
    # The grasp descent is contact-rich: it is *not* convergence-gated (the
    # grasp check gates it); this looser tolerance only ends it early.
    descend_tolerance: float = 0.012
    grasp_steps: int = 10
    settle_steps: int = 5
    lift_height: float = 0.12
    # Cap every waypoint phase's position action and hold the hover briefly:
    # fast TCP swings make the fingers/forearm transiently sweep into the
    # cylinder even when the commanded path clears it.
    waypoint_action_cap: float = 0.35
    hover_hold_steps: int = 8
    # Scene/geometry. The cylinder is tall enough that the constant-height
    # carry (TCP at grasp z + lift_height, carried cube top ~0.16 m) cannot
    # pass over the top, so the side detour is forced.
    obstacle_height: float = 0.18
    obstacle_radius_default: float = 0.03
    safety_margin: float = 0.005
    # Cube-pair separation: large enough that a blocking cylinder fits between
    # the measured palm-horn reach at both endpoints (see FOOTPRINT_* above).
    min_cube_separation: float = 0.25
    # Clearance label thresholds: achieved footprint clearance
    # (min distance to recorded robot geometry minus cylinder radius).
    roomy_clearance: float = 0.03
    tight_clearance: float = 0.015
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    sanity_obstacle_center: tuple = (0.8, 0.8)
    # Recording. The registry caps StackCubeObstacle-v1 at 50 steps; the full
    # task needs the explicit override or episodes truncate early. The flat
    # "state" observation keeps the standard aggregate T+1 obs row layout.
    max_episode_steps: int = 500
    obs_mode: str = "state"
    robot_uids: str = "panda_wristcam"
    robot_init_qpos_noise: float = 0.02

    def env_kwargs(self) -> dict:
        return dict(
            num_envs=1,
            obs_mode=self.obs_mode,
            sim_backend="physx_cpu",
            robot_uids=self.robot_uids,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
            control_mode=TARGET_CONTROL_MODE,
            obstacle_radius=self.obstacle_radius_default,
            obstacle_height=self.obstacle_height,
            max_episode_steps=self.max_episode_steps,
        )

    def as_dict(self) -> dict:
        payload = {key: _jsonable(value) for key, value in asdict(self).items()}
        payload["target_control_mode"] = TARGET_CONTROL_MODE
        payload["pos_action_scale"] = POS_ACTION_SCALE
        payload["clearance_targets"] = list(CLEARANCE_TARGETS)
        payload["footprint_margin"] = FOOTPRINT_MARGIN
        payload["footprint_z_slack"] = FOOTPRINT_Z_SLACK
        payload["waypoint_offset_range"] = list(WAYPOINT_OFFSET_RANGE)
        payload["waypoint_lead_fraction"] = WAYPOINT_LEAD_FRACTION
        payload["cylinder_strip_fraction"] = list(CYLINDER_STRIP_FRACTION)
        payload["cylinder_sampling_prior"] = {
            "side": "least-occupied footprint projection at 0.04 m lateral",
            "magnitude_fraction": [0.5, 1.0],
            "clearance": "exact cKDTree nearest-point query against full recorded footprint",
        }
        payload["panda_base_xy"] = list(PANDA_BASE_XY)
        return payload

    def fingerprint(self) -> str:
        canonical = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.reshape(-1)]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (list, dict, str, float, int, bool)) or value is None:
        return value
    raise TypeError(f"not JSON serialisable: {type(value)}")


# --------------------------------------------------------------------------- #
# Template geometry (public seams, pure numpy, obstacle-independent)
# --------------------------------------------------------------------------- #


def segment_point_distance(point, start, end) -> float:
    """Shortest 2D distance from ``point`` to the segment ``start``-``end``."""
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length_sq = float(direction @ direction)
    if length_sq <= 0.0:
        return float(np.linalg.norm(point - start))
    t = float((point - start) @ direction) / length_sq
    t = min(1.0, max(0.0, t))
    projection = start + t * direction
    return float(np.linalg.norm(point - projection))


def _inside_box(xy, box) -> bool:
    xmin, ymin, xmax, ymax = box
    return xmin <= xy[0] <= xmax and ymin <= xy[1] <= ymax


def sample_cube_pair(rng: np.random.Generator, cfg: Config) -> Optional[dict]:
    """Sample the template cube pair (pick A, stack B) independent of any obstacle.

    Separation stays inside ``[min_cube_separation, layout.max_separation]``
    and the cube axis stays roughly perpendicular to the reaching direction so
    the arm can approach both cubes. Returns ``None`` within the attempt
    budget.
    """
    box = cfg.layout.cube_xy_range
    base = np.asarray(cfg.layout.base_xy, dtype=np.float64)
    for _ in range(cfg.layout.max_attempts):
        cube_a = np.array(
            [rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])]
        )
        separation = rng.uniform(cfg.min_cube_separation, cfg.layout.max_separation)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cube_b = cube_a + separation * np.array([np.cos(angle), np.sin(angle)])
        if not _inside_box(cube_b, box):
            continue
        direction = (cube_b - cube_a) / separation
        reach = 0.5 * (cube_a + cube_b) - base
        alignment = abs(float(direction @ reach)) / float(np.linalg.norm(reach))
        if alignment > cfg.layout.max_reach_alignment:
            continue
        return dict(
            cubeA_xy=cube_a,
            cubeB_xy=cube_b,
            cubeA_yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
            cubeB_yaw=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    return None


def choose_waypoint(cubes: dict, rng: np.random.Generator) -> Optional[dict]:
    """Sample the detour waypoint W independent of any obstacle.

    W is the AB midpoint shifted sideways (perpendicular to AB, either side)
    by ``offset`` and led forward along AB toward B by
    ``WAYPOINT_LEAD_FRACTION * separation`` so the return carry leg crosses
    the blocking strip's band already past the future obstacle. Returns
    ``None`` when no side lands inside the reachable envelope.
    """
    a = np.asarray(cubes["cubeA_xy"], dtype=np.float64)
    b = np.asarray(cubes["cubeB_xy"], dtype=np.float64)
    midpoint = 0.5 * (a + b)
    ab = b - a
    norm = float(np.linalg.norm(ab))
    if norm <= 0.0:
        return None
    along = ab / norm
    perpendicular = np.array([-along[1], along[0]])
    offset = rng.uniform(*WAYPOINT_OFFSET_RANGE)
    lead = WAYPOINT_LEAD_FRACTION * norm
    side = 1.0 if rng.uniform() < 0.5 else -1.0
    for chosen_side in (side, -side):
        waypoint = midpoint + perpendicular * chosen_side * offset + along * lead
        if not _inside_box(waypoint, WAYPOINT_XY_BOX):
            continue
        if float(np.linalg.norm(waypoint - np.asarray(PANDA_BASE_XY))) > WAYPOINT_MAX_BASE_DIST:
            continue
        return dict(
            side=float(chosen_side),
            offset=float(offset),
            lead=float(lead),
            waypoint_xy=waypoint,
        )
    return None


def sample_cylinder(
    rng: np.random.Generator,
    template: dict,
    cfg: Config,
    min_clearance: float,
) -> tuple:
    """Sample a blocking cylinder around a fixed no-obstacle action template.

    The cylinder centre is drawn inside the direct A->B blocking strip (so the
    direct path is genuinely blocked) but must clear every recorded template
    footprint point by ``radius + min_clearance`` — the actual swept robot and
    carried-cube geometry, not a nominal path. Candidates are evaluated in
    vectorized batches; cheap geometry is rejected first and physics
    acceptance happens later in the real episode. Returns ``(cylinder,
    None)`` or ``(None, reason)``.
    """
    carry_top = float(template["carry_z"]) + CUBE_HALF_SIZE
    if cfg.obstacle_height <= carry_top:
        return None, "cylinder_not_tall_enough_for_side_detour"
    cubes = template["cubes"]
    cube_a = np.asarray(cubes["cubeA_xy"], dtype=np.float64)
    cube_b = np.asarray(cubes["cubeB_xy"], dtype=np.float64)
    footprint = np.asarray(template["footprint"], dtype=np.float64).reshape(-1, 2)
    ab = cube_b - cube_a
    separation = float(np.linalg.norm(ab))
    if separation <= 0.0:
        return None, "degenerate_segment"
    along = ab / separation
    perpendicular = np.array([-along[1], along[0]])
    footprint_side = (footprint - cube_a) @ perpendicular
    # Prefer the side whose footprint projection is least occupied. This is a
    # prior only: every sampled centre still passes the exact 2D clearance test.
    side_scores = [
        float(np.min(np.abs(footprint_side - sign * 0.04)))
        for sign in (-1.0, 1.0)
    ]
    preferred_side = (-1.0, 1.0)[int(side_scores[1] > side_scores[0])]
    footprint_tree = cKDTree(footprint)
    min_clearance = max(float(min_clearance), FOOTPRINT_MARGIN)
    box = cfg.layout.obstacle_xy_range
    batch = 512
    for _ in range(max(1, cfg.layout.max_attempts // batch)):
        radius = rng.uniform(*cfg.layout.radius_range, size=batch)
        forbidden = radius + CUBE_XY_RADIUS + cfg.safety_margin
        max_lateral = forbidden - cfg.layout.min_segment_clearance - 1e-6
        t = rng.uniform(*CYLINDER_STRIP_FRACTION, size=batch)
        lateral = preferred_side * rng.uniform(0.5, 1.0, size=batch) * np.maximum(max_lateral, 0.0)
        centers = cube_a + t[:, None] * ab + lateral[:, None] * perpendicular
        keep = (
            (max_lateral > 0.0)
            & (centers[:, 0] >= box[0]) & (centers[:, 0] <= box[2])
            & (centers[:, 1] >= box[1]) & (centers[:, 1] <= box[3])
            & (np.linalg.norm(centers - cube_a, axis=1) > forbidden + CYLINDER_ENDPOINT_SLACK)
            & (np.linalg.norm(centers - cube_b, axis=1) > forbidden + CYLINDER_ENDPOINT_SLACK)
        )
        for start in range(0, batch, 128):
            index = np.arange(start, min(start + 128, batch))[keep[start:start + 128]]
            if not len(index):
                continue
            chunk = centers[index]
            distances, _ = footprint_tree.query(chunk, k=1, workers=1)
            achieved = distances - radius[index]
            feasible = np.nonzero(achieved >= min_clearance)[0]
            if len(feasible):
                pick = index[feasible[0]]
                center = centers[pick]
                value = float(achieved[feasible[0]])
                label = "roomy" if value >= cfg.roomy_clearance else "tight"
                return (
                    dict(
                        center_xy=[float(center[0]), float(center[1])],
                        radius=float(radius[pick]),
                        height=float(cfg.obstacle_height),
                        label=label,
                        min_footprint_clearance=value,
                    ),
                    None,
                )
    return None, "no_feasible_cylinder"


# --------------------------------------------------------------------------- #
# Feedback waypoint tracker (public seam)
# --------------------------------------------------------------------------- #


@dataclass
class WaypointTracker:
    """Clipped-delta feedback tracker for one Cartesian waypoint.

    Each step the tracker reads the live TCP position and commands
    ``clip((target - tcp) / POS_ACTION_SCALE, -1, 1)`` on the position axes,
    zero rotation (fixed downward orientation) and the given gripper value.
    ``step(tcp)`` records the error and reports convergence; tracking errors
    are recorded, never gated beyond waypoint convergence.
    """

    target: np.ndarray
    gripper: float
    tolerance: float = 0.004
    max_steps: int = 60
    step_index: int = 0
    max_error: float = 0.0
    final_error: float = float("inf")
    z_values: list = field(default_factory=list)

    def action_for(self, tcp_xyz: np.ndarray) -> np.ndarray:
        delta = np.asarray(self.target, dtype=np.float64) - np.asarray(tcp_xyz, dtype=np.float64)
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(delta / POS_ACTION_SCALE, -1.0, 1.0)
        action[GRIPPER_AXIS] = self.gripper
        return action

    def step(self, tcp_xyz: np.ndarray) -> bool:
        """Record one tracking step; returns True when converged."""
        tcp_xyz = np.asarray(tcp_xyz, dtype=np.float64)
        error = float(np.linalg.norm(np.asarray(self.target) - tcp_xyz))
        self.z_values.append(float(tcp_xyz[2]))
        self.max_error = max(self.max_error, error)
        self.final_error = error
        self.step_index += 1
        return error < self.tolerance

    @property
    def exhausted(self) -> bool:
        return self.step_index >= self.max_steps


class PolicyFailure(RuntimeError):
    """Finite-budget policy failure with a reason and optional phase."""

    def __init__(self, reason: str, phase: Optional[str] = None):
        super().__init__(reason if phase is None else f"{reason} (phase={phase})")
        self.reason = reason
        self.phase = phase


# --------------------------------------------------------------------------- #
# Waypoint policy
# --------------------------------------------------------------------------- #


def build_episode_plan(
    cube_a_xyz: np.ndarray,
    cube_b_xyz: np.ndarray,
    waypoint_xy: np.ndarray,
    lift_height: float,
) -> dict:
    """Vertical grasp/lift/place with one constant-height A->W->B carry."""
    grasp_z = float(cube_a_xyz[2])
    carry_z = grasp_z + lift_height
    place_z = float(cube_b_xyz[2]) + 2.0 * CUBE_HALF_SIZE
    a = np.asarray(cube_a_xyz, dtype=np.float64)
    b = np.asarray(cube_b_xyz, dtype=np.float64)
    w_xy = np.asarray(waypoint_xy, dtype=np.float64)
    waypoints = [
        ("hover", np.array([a[0], a[1], carry_z]), GRIPPER_OPEN),
        ("descend", np.array([a[0], a[1], grasp_z]), GRIPPER_OPEN),
        ("lift", np.array([a[0], a[1], carry_z]), GRIPPER_CLOSE),
        ("carry_to_waypoint", np.array([w_xy[0], w_xy[1], carry_z]), GRIPPER_CLOSE),
        ("carry_to_place", np.array([b[0], b[1], carry_z]), GRIPPER_CLOSE),
        ("place", np.array([b[0], b[1], place_z]), GRIPPER_CLOSE),
        ("retreat", np.array([b[0], b[1], carry_z]), GRIPPER_OPEN),
    ]
    return dict(
        waypoints=waypoints,
        waypoint_xy=w_xy,
        carry_z=carry_z,
        grasp_z=grasp_z,
        place_z=place_z,
    )


# Only the two horizontal carry phases count as "actual carry" for height
# statistics; lift/place are vertical and must not fold into them.
CARRY_PHASES = ("carry_to_waypoint", "carry_to_place")


def _robot_shape_cache(base) -> list:
    """Per-link collision geometry, cached once for footprint projection."""
    cache = []
    for link in base.agent.robot.links:
        shapes = []
        for shape in link._bodies[0].collision_shapes:
            try:
                vertices = common.to_numpy(shape.get_vertices())
            except AttributeError:
                get_half_size = getattr(shape, "get_half_size", None)
                if get_half_size is None:
                    continue
                half_size = common.to_numpy(get_half_size()).reshape(3)
                signs = np.array([
                    [x, y, z]
                    for x in (-1.0, 1.0)
                    for y in (-1.0, 1.0)
                    for z in (-1.0, 1.0)
                ])
                vertices = signs * half_size
            except Exception:
                continue
            local = common.to_numpy(shape.get_local_pose().to_transformation_matrix())
            shapes.append((local, vertices))
        if shapes:
            cache.append((link, shapes))
    return cache


def _cube_corners(base) -> np.ndarray:
    half = CUBE_HALF_SIZE
    local = np.array([
        [sx * half, sy * half, sz * half, 1.0]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)
    ]).T
    matrix = common.to_numpy(base.cubeA.pose.to_transformation_matrix()[0])
    return (matrix @ local)[:2].T


def collect_footprint(base, cache, height: float) -> np.ndarray:
    """2D projection of all robot + carried-cube collision vertices below the
    cylinder top at the current step (the actual occupied region)."""
    limit = height + FOOTPRINT_Z_SLACK
    points = []
    for link, shapes in cache:
        matrix = common.to_numpy(link.pose.to_transformation_matrix()[0])
        for local, vertices in shapes:
            world = (matrix @ local @ np.concatenate([vertices.T, np.ones((1, vertices.shape[0]))]))[:3].T
            below = world[world[:, 2] < limit]
            if len(below):
                points.append(below[:, :2])
    points.append(_cube_corners(base))
    return np.concatenate(points, axis=0)


def run_episode(env, plan: dict, cfg: Config, record_trace: bool = False,
                record_footprint: bool = False) -> dict:
    """Execute the waypoint policy; returns the episode trace.

    Raises :class:`PolicyFailure` when the grasp fails or the episode is
    truncated by the step cap. Waypoint phases other than the grasp descent
    are convergence-gated within their budget; the descent instead runs to its
    looser ``descend_tolerance`` or budget and is gated by the real grasp
    check (contact-rich descents legitimately stall a few millimetres short).
    The env's ``terminated`` flag is ``info["success"]`` for this task, so an
    early termination by success simply ends the episode (the cube is stacked,
    static and released). The trace records approximate tracking (per-phase
    max/final errors), the observed carry height over the two horizontal carry
    phases only, and the maximum orientation drift over all steps —
    observations, not acceptance gates.
    """
    base = env.unwrapped
    quat0 = common.to_numpy(base.agent.tcp.pose.q[0]).reshape(4)
    height = float(cfg.obstacle_height)
    cache = _robot_shape_cache(base) if record_footprint else None
    footprint_points = [] if record_footprint else None
    tcp_trace = [] if record_trace else None
    cube_trace = [] if record_trace else None
    drift_max = 0.0
    phases = []
    early_success = False

    def tcp_xyz() -> np.ndarray:
        return common.to_numpy(base.agent.tcp.pose.p[0]).reshape(3)

    def check_done(terminated, truncated, phase, step_info=None):
        nonlocal early_success
        if step_info is not None and "stack_success" in step_info:
            stack_success = bool(
                common.to_numpy(step_info["stack_success"]).reshape(-1)[0]
            )
        elif terminated:
            stack_success = bool(
                common.to_numpy(base.evaluate()["stack_success"]).reshape(-1)[0]
            )
        else:
            stack_success = False
        if stack_success:
            early_success = True
            return
        if truncated:
            raise PolicyFailure("episode_truncated", phase)
        if terminated:
            raise PolicyFailure("terminated_unsuccessful", phase)

    def run_waypoint(phase, target, gripper, tolerance=None, require_convergence=True):
        nonlocal drift_max
        tracker = WaypointTracker(
            target, gripper,
            cfg.waypoint_tolerance if tolerance is None else tolerance,
            cfg.max_steps_per_waypoint,
        )
        cap = cfg.waypoint_action_cap

        def stepped_action():
            action = tracker.action_for(tcp_xyz())
            action[:3] = np.clip(action[:3], -cap, cap)
            return action

        while True:
            _, _, terminated, truncated, step_info = env.step(stepped_action()[None])
            converged = tracker.step(tcp_xyz())
            quat_now = common.to_numpy(base.agent.tcp.pose.q[0]).reshape(4)
            drift_max = max(drift_max, _quat_angle(quat0, quat_now))
            if record_footprint:
                footprint_points.append(collect_footprint(base, cache, height))
            if record_trace and phase in CARRY_PHASES:
                tcp_trace.append([float(v) for v in tcp_xyz()[:2]])
                cube_now = common.to_numpy(base.cubeA.pose.p[0]).reshape(3)
                cube_trace.append([float(v) for v in cube_now[:2]])
            check_done(terminated, truncated, phase, step_info)
            if early_success:
                break
            if converged:
                break
            if tracker.exhausted:
                if require_convergence:
                    raise PolicyFailure("convergence_timeout", phase)
                break
        if phase == "hover" and cfg.hover_hold_steps > 0 and not early_success:
            for _ in range(cfg.hover_hold_steps):
                _, _, terminated, truncated, step_info = env.step(stepped_action()[None])
                check_done(terminated, truncated, phase, step_info)
                if early_success:
                    break
        entry = dict(
            phase=phase, steps=tracker.step_index,
            max_error=tracker.max_error, final_error=tracker.final_error,
        )
        if phase in CARRY_PHASES and tracker.z_values:
            entry["observed_z_min"] = min(tracker.z_values)
            entry["observed_z_max"] = max(tracker.z_values)
        phases.append(entry)

    for phase, target, gripper in plan["waypoints"]:
        if phase == "descend":
            run_waypoint(phase, target, gripper, tolerance=cfg.descend_tolerance,
                         require_convergence=False)
            close_tracker = WaypointTracker(
                target, GRIPPER_CLOSE, cfg.waypoint_tolerance, cfg.max_steps_per_waypoint
            )
            for _ in range(cfg.grasp_steps):
                _, _, terminated, truncated, step_info = env.step(
                    close_tracker.action_for(tcp_xyz())[None])
                check_done(terminated, truncated, phase, step_info)
                if early_success:
                    break
            if not early_success:
                info = base.evaluate()
                if not bool(common.to_numpy(info["is_cubeA_grasped"]).reshape(-1)[0]):
                    raise PolicyFailure("grasp_failed", phase)
            phases.append(dict(phase="grasp", steps=cfg.grasp_steps,
                               max_error=None, final_error=None))
        else:
            run_waypoint(phase, target, gripper)
        if early_success:
            break

    if not early_success:
        for _ in range(cfg.settle_steps):
            action = np.zeros(7, dtype=np.float32)
            action[GRIPPER_AXIS] = GRIPPER_OPEN
            _, _, terminated, truncated, step_info = env.step(action[None])
            check_done(terminated, truncated, "settle", step_info)
            if early_success:
                break
        phases.append(dict(phase="settle", steps=cfg.settle_steps,
                           max_error=None, final_error=None))

    info = base.evaluate()
    carry_z_values = [
        z for entry in phases if entry["phase"] in CARRY_PHASES and "observed_z_min" in entry
        for z in (entry["observed_z_min"], entry["observed_z_max"])
    ]
    trace = dict(
        phases=phases,
        success=bool(common.to_numpy(info["success"]).reshape(-1)[0]),
        stack_success=bool(common.to_numpy(info["stack_success"]).reshape(-1)[0]),
        obstacle_collision=bool(common.to_numpy(info["obstacle_collision"]).reshape(-1)[0]),
        obstacle_crossing=bool(common.to_numpy(info["obstacle_crossing"]).reshape(-1)[0]),
        steps=int(common.to_numpy(base._elapsed_steps).reshape(-1)[0]),
        early_success=bool(early_success),
        initial_tcp_quat=[float(v) for v in quat0],
        orientation_drift_max_rad=float(drift_max),
        carry_z_commanded=float(plan["carry_z"]),
        carry_z_observed_min=min(carry_z_values) if carry_z_values else None,
        carry_z_observed_max=max(carry_z_values) if carry_z_values else None,
        final_cubeA_xyz=[float(v) for v in common.to_numpy(base.cubeA.pose.p[0]).reshape(3)],
    )
    if record_trace:
        trace["tcp_xy"] = tcp_trace
        trace["cube_xy"] = cube_trace
    if record_footprint:
        points = np.concatenate(footprint_points, axis=0)
        trace["footprint"] = np.unique(np.round(points, 4), axis=0).tolist()
    return trace


def _quat_angle(q0, q1) -> float:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    dot = min(1.0, max(-1.0, float(q0 @ q1)))
    return 2.0 * float(np.arccos(dot))


# --------------------------------------------------------------------------- #
# Initial-state signature (recorded aggregate H5 rows, robot + cubes)
# --------------------------------------------------------------------------- #


def read_initial_robot_cube_state(h5_path: Path, traj_key: str) -> dict:
    """Row-0 robot/cube state leaves of one recorded trajectory."""
    import h5py

    state = {}
    with h5py.File(h5_path, "r") as handle:
        trajectory = handle[traj_key]
        for leaf in STATE_LEAVES:
            state[leaf] = np.array(trajectory["env_states"][leaf][0])
        articulations = trajectory["env_states/articulations"]
        if len(articulations) != 1:
            raise ValueError("expected exactly one robot articulation in initial state")
        robot_uid = next(iter(articulations))
        state[f"articulations/{robot_uid}"] = np.array(articulations[robot_uid][0])
    return state


def initial_state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for leaf in sorted(state):
        digest.update(leaf.encode("utf-8"))
        digest.update(np.ascontiguousarray(state[leaf], dtype=np.float32).tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Real environment episode (recording + acceptance)
# --------------------------------------------------------------------------- #

# Recording seam: (scratch_dir, trajectory_name, request, cfg) -> (valid,
#   attempt_dict). ``request`` carries ``kind`` ("sanity" | "template" |
#   "obstacle") plus everything that kind needs. ``collect`` publishes the
#   scratch ``{trajectory_name}.h5``/``.json`` pair of accepted obstacle
#   episodes.
EpisodeRecorder = Callable[..., tuple]


def record_episode_once(
    scratch_dir: Path,
    trajectory_name: str,
    request: dict,
    cfg: Config,
) -> tuple:
    """Record one episode attempt in ``scratch_dir``; returns ``(valid, attempt)``.

    ``kind="sanity"`` runs the far-away-cylinder scene with the default cube
    layout; ``kind="template"`` runs the no-obstacle action template for a
    sampled cube pair + waypoint and records the actual swept footprint;
    ``kind="obstacle"`` re-executes a template's exact plan from the template's
    reset seed with a sampled blocking cylinder. Failed attempts leave no
    HDF5/JSON behind: ``flush_trajectory(save=False)`` drops the buffers and
    the caller's scratch directory is removed.
    """
    import gymnasium as gym

    from examples.baselines.constrain.obstacle_record import ObstacleRecordEpisode

    kind = request["kind"]
    seed = int(request["seed"])
    env = gym.make(ENV_ID, **cfg.env_kwargs())
    recorder = ObstacleRecordEpisode(
        env,
        output_dir=str(scratch_dir),
        trajectory_name=trajectory_name,
        save_video=False,
        save_on_reset=False,
        source_type="motionplanning",
        source_desc=(
            "constructive pd_ee_delta_pose feedback waypoint expert "
            "(action-first templates + sampled blocking cylinders, no planner)"
        ),
    )
    metadata = recorder._json_data["env_info"]
    metadata["constructive_config"] = cfg.as_dict()
    metadata["constructive_config"]["fingerprint"] = cfg.fingerprint()
    if kind == "sanity":
        metadata.update(dict(sanity_run=True))
    else:
        metadata.update({
            key: _jsonable(request[key]) for key in
            ("template_index", "reuse_index", "cubes", "cylinder",
             "waypoint_xy", "side")
            if key in request
        })
    attempt = dict(
        kind=str(kind),
        seed=seed,
        control_mode=TARGET_CONTROL_MODE,
        failure=None,
    )
    if kind == "obstacle":
        attempt["template_index"] = int(request["template_index"])
        attempt["reuse_index"] = int(request["reuse_index"])
    elif kind == "template":
        attempt["template_index"] = int(request["template_index"])
    started = time.perf_counter()
    valid = False
    try:
        options = dict(reconfigure=True)
        if kind == "sanity":
            options["obstacle_center"] = [float(v) for v in cfg.sanity_obstacle_center]
            options["obstacle_radius"] = float(cfg.obstacle_radius_default)
        elif kind == "template":
            options["obstacle_center"] = [float(v) for v in cfg.sanity_obstacle_center]
            options["obstacle_radius"] = float(cfg.obstacle_radius_default)
            options["cube_layout"] = {
                "cubeA_xy": [float(v) for v in request["cubes"]["cubeA_xy"]],
                "cubeB_xy": [float(v) for v in request["cubes"]["cubeB_xy"]],
                "cubeA_yaw": float(request["cubes"]["cubeA_yaw"]),
                "cubeB_yaw": float(request["cubes"]["cubeB_yaw"]),
            }
        else:
            cylinder = request["cylinder"]
            options["obstacle_center"] = [float(v) for v in cylinder["center_xy"]]
            options["obstacle_radius"] = float(cylinder["radius"])
            options["cube_layout"] = {
                "cubeA_xy": [float(v) for v in request["cubes"]["cubeA_xy"]],
                "cubeB_xy": [float(v) for v in request["cubes"]["cubeB_xy"]],
                "cubeA_yaw": float(request["cubes"]["cubeA_yaw"]),
                "cubeB_yaw": float(request["cubes"]["cubeB_yaw"]),
            }
        recorder.reset(seed=seed, options=options, save=False)
        base = recorder.base_env
        cube_a = common.to_numpy(base.cubeA.pose.p[0]).reshape(3)
        cube_b = common.to_numpy(base.cubeB.pose.p[0]).reshape(3)
        if kind == "obstacle":
            cylinder = request["cylinder"]
            actual_height = float(base.obstacle_height)
            actual_radius = float(base.obstacle_collision_radius)
            if abs(actual_height - float(cylinder["height"])) > 1e-6 or (
                abs(actual_radius - float(cylinder["radius"])) > 1e-6
            ):
                raise PolicyFailure("obstacle_config_mismatch")
            plan = request["plan"]
            trace = run_episode(recorder, plan, cfg)
        else:
            if kind == "template":
                waypoint = np.asarray(request["waypoint_xy"], dtype=np.float64)
                plan = build_episode_plan(cube_a, cube_b, waypoint, cfg.lift_height)
                trace = run_episode(recorder, plan, cfg,
                                    record_trace=True, record_footprint=True)
            else:
                waypoint = 0.5 * (cube_a[:2] + cube_b[:2])
                plan = build_episode_plan(cube_a, cube_b, waypoint, cfg.lift_height)
                trace = run_episode(recorder, plan, cfg)
        valid = bool(trace["success"])
        trace["place_height_error_m"] = float(
            abs(trace["final_cubeA_xyz"][2] - (float(cube_b[2]) + 2.0 * CUBE_HALF_SIZE))
        )
        attempt["trace"] = trace
        if not valid:
            attempt["failure"] = _failure_reason(trace)
        attempt["success"] = valid
        if kind == "template":
            attempt["plan"] = plan
    except PolicyFailure as failure:
        attempt["failure"] = failure.reason
        attempt["failure_phase"] = failure.phase
    except Exception as exc:  # unexpected: fail closed, drop the buffers
        attempt["failure"] = f"exception:{type(exc).__name__}"
        attempt["failure_detail"] = str(exc)
    finally:
        attempt["elapsed_seconds"] = time.perf_counter() - started
        recorder.flush_trajectory(save=valid)
        if valid:
            state = read_initial_robot_cube_state(
                Path(scratch_dir) / f"{trajectory_name}.h5", "traj_0"
            )
            attempt["initial_state_hash"] = initial_state_hash(state)
        recorder.close()

    return valid, attempt


def _failure_reason(trace) -> str:
    if trace["obstacle_collision"]:
        return "obstacle_collision"
    if trace["obstacle_crossing"]:
        return "obstacle_crossing"
    if not trace["stack_success"]:
        return "stack_failed"
    return "invalid"


# --------------------------------------------------------------------------- #
# Aggregate output helpers
# --------------------------------------------------------------------------- #


def append_episode(source_h5: Path, source_json: Path, output_h5: Path, output_json: Path,
                   episode_index: int) -> None:
    """Append one accepted ``traj_0`` recording as ``traj_{episode_index}``."""
    import h5py

    metadata = json.loads(Path(source_json).read_text())
    key = f"traj_{episode_index}"
    with h5py.File(source_h5, "r") as source, h5py.File(output_h5, "a") as output:
        if key in output:
            raise ValueError(f"trajectory already exists: {key}")
        source.copy("traj_0", output, name=key)
    metadata["episodes"][0]["episode_id"] = episode_index
    aggregate = metadata if episode_index == 0 else json.loads(Path(output_json).read_text())
    if episode_index:
        aggregate["episodes"].append(metadata["episodes"][0])
    temp_json = Path(output_json).with_name(Path(output_json).stem + ".tmp.json")
    try:
        dump_json(temp_json, aggregate, indent=2)
        os.replace(temp_json, output_json)
    except BaseException:
        with h5py.File(output_h5, "a") as output:
            if key in output:
                del output[key]
        temp_json.unlink(missing_ok=True)
        raise


def validate_aggregate(h5_path: Path, json_path: Path, expected: int) -> None:
    """Structural check: T actions, T+1 states, T+1 obs, contiguous IDs."""
    import h5py

    metadata = json.loads(Path(json_path).read_text())
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected:
        raise ValueError("aggregate episode count mismatch")
    if [episode["episode_id"] for episode in episodes] != list(range(expected)):
        raise ValueError("aggregate episode IDs must be contiguous from zero")
    with h5py.File(h5_path, "r") as handle:

        def state_lengths(node):
            if isinstance(node, h5py.Dataset):
                return [len(node)]
            return [l for child in node.values() for l in state_lengths(child)]

        for episode in episodes:
            trajectory = handle[f"traj_{episode['episode_id']}"]
            steps = len(trajectory["actions"])
            if steps == 0:
                raise ValueError("episode has no actions")
            if len(trajectory["obs"]) != steps + 1:
                raise ValueError("episode must have T+1 observations")
            if any(l != steps + 1 for l in state_lengths(trajectory["env_states"])):
                raise ValueError("episode must have T+1 states")


# --------------------------------------------------------------------------- #
# Collection entry point (public seams: recorder + samplers injectable)
# --------------------------------------------------------------------------- #


def collect(
    output_dir,
    config: Config,
    episode_recorder: Optional[EpisodeRecorder] = None,
    cube_sampler: Optional[Callable[[np.random.Generator, Config], Optional[dict]]] = None,
    waypoint_chooser: Optional[Callable[[dict, np.random.Generator], Optional[dict]]] = None,
    cylinder_sampler: Optional[Callable[..., tuple]] = None,
    progress=None,
) -> dict:
    """Collect accepted obstacle episodes until a budget or the target is reached.

    Seams: tests inject fakes; production uses the real recorder and the pure
    geometry samplers. For every template the no-obstacle action run happens
    FIRST (real physics, actual trace); only then are blocking cylinders
    sampled around that fixed action and re-executed from the identical
    initial state with the same plan. Each accepted obstacle episode is
    appended to the aggregate ``trajectory.h5``/``.json`` under its final
    ``traj_N`` id; failed attempts leave no artifacts. Returns the
    ``report.json`` payload.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path, json_path = output_dir / "trajectory.h5", output_dir / "trajectory.json"
    report_path = output_dir / "report.json"
    if h5_path.exists() or json_path.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing outputs in {output_dir}")

    if episode_recorder is None:
        episode_recorder = record_episode_once
    if cube_sampler is None:
        cube_sampler = sample_cube_pair
    if waypoint_chooser is None:
        waypoint_chooser = choose_waypoint
    if cylinder_sampler is None:
        cylinder_sampler = sample_cylinder

    rng = np.random.default_rng(config.seed)
    started = time.perf_counter()
    stage_seconds = dict(sanity=0.0, templates=0.0, sampling=0.0, episodes=0.0)
    report = dict(
        config=config.as_dict(),
        config_fingerprint=config.fingerprint(),
        output_dir=str(output_dir),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sanity=None,
        stage_seconds=stage_seconds,
        episodes=[],
        templates={},
        failures={},
        candidate_templates=0,
        no_obstacle_successful_templates=0,
        cylinder_sampling_calls=0,
        accepted=0,
        physics_attempts=0,
        target_successes=int(config.target_successes),
        complete=False,
    )

    def record_failure(reason: str):
        report["failures"][reason] = report["failures"].get(reason, 0) + 1

    def show_stage(phase: str, template=None):
        if progress is not None:
            progress.set_postfix(
                phase=phase,
                template="-" if template is None else template + 1,
                attempts=report["physics_attempts"],
                refresh=False,
            )
            progress.update(0)

    def write_report() -> dict:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["valid_per_minute"] = (
            report["accepted"] / (report["elapsed_seconds"] / 60.0)
            if report["elapsed_seconds"] > 0 else 0.0
        )
        report["distinct_templates"] = sum(
            1 for entry in report["templates"].values() if entry["reuse_accepted"] > 0
        )
        temp = report_path.with_name("report.tmp.json")
        dump_json(temp, report, indent=2)
        os.replace(temp, report_path)
        return report

    def budget_left() -> bool:
        return (report["accepted"] < config.target_successes
                and len(report["episodes"]) < config.max_episodes)

    def publish_episode(attempt: dict, request: dict, scratch_dir: Path) -> None:
        episode_index = report["accepted"]
        append_episode(
            Path(scratch_dir) / "traj.h5",
            Path(scratch_dir) / "traj.json",
            h5_path, json_path, episode_index,
        )
        aggregate = json.loads(json_path.read_text())
        episode = aggregate["episodes"][-1]
        episode.update(dict(
            template_index=int(request["template_index"]),
            reuse_index=int(request["reuse_index"]),
            seed=int(request["seed"]),
            cylinder=request["cylinder"],
            plan=request["plan"],
            initial_state_hash=attempt.get("initial_state_hash"),
            elapsed_seconds=attempt.get("elapsed_seconds"),
            control_mode=TARGET_CONTROL_MODE,
        ))
        temp_json = json_path.with_name("trajectory.tmp.json")
        dump_json(temp_json, aggregate, indent=2)
        os.replace(temp_json, json_path)

    # 1. Sanity episode with a far-away cylinder: the policy must stack.
    show_stage("sanity")
    with tempfile.TemporaryDirectory(prefix="constructive-sanity-") as scratch:
        stage_started = time.perf_counter()
        sanity_valid, sanity_attempt = episode_recorder(
            Path(scratch), "sanity", dict(kind="sanity", seed=int(rng.integers(0, 2**31 - 1))),
            config,
        )
        stage_seconds["sanity"] = time.perf_counter() - stage_started
    sanity_attempt["valid"] = sanity_valid
    report["sanity"] = sanity_attempt
    if not sanity_valid:
        record_failure("sanity_failed")
        return write_report()

    # 2. Action-first templates: no-obstacle run first, then obstacle reuses.
    for template_index in range(config.max_templates):
        if not budget_left():
            break
        report["candidate_templates"] += 1
        show_stage("template", template_index)
        cubes = cube_sampler(rng, config)
        if cubes is None:
            record_failure("cube_sampling_exhausted")
            continue
        choice = waypoint_chooser(cubes, rng)
        if choice is None:
            record_failure("waypoint_infeasible")
            continue
        seed = int(rng.integers(0, 2**31 - 1))
        with tempfile.TemporaryDirectory(prefix="constructive-template-") as scratch:
            stage_started = time.perf_counter()
            template_valid, template_attempt = episode_recorder(
                Path(scratch), "template",
                dict(
                    kind="template",
                    template_index=int(template_index),
                    seed=seed,
                    cubes=cubes,
                    waypoint_xy=choice["waypoint_xy"],
                    side=choice["side"],
                ),
                config,
            )
            stage_seconds["templates"] += time.perf_counter() - stage_started
        if not template_valid:
            record_failure(str(template_attempt.get("failure") or "template_failed"))
            continue
        report["no_obstacle_successful_templates"] += 1
        template_record = dict(
            index=int(template_index),
            seed=seed,
            cubes={key: _jsonable(value) for key, value in cubes.items()},
            waypoint=dict(
                side=choice["side"], offset=choice["offset"], lead=choice["lead"],
                waypoint_xy=_jsonable(choice["waypoint_xy"]),
            ),
            plan=_jsonable(template_attempt["plan"]),
            trace=template_attempt["trace"],
            initial_state_hash=template_attempt["initial_state_hash"],
            elapsed_seconds=template_attempt["elapsed_seconds"],
            reuse=[],
            reuse_accepted=0,
        )
        report["templates"][str(template_index)] = template_record
        template_state = None
        template_view = dict(
            cubes=cubes,
            waypoint_xy=choice["waypoint_xy"],
            carry_z=float(template_record["plan"]["carry_z"]),
            footprint=np.asarray(template_record["trace"]["footprint"], dtype=np.float64),
        )
        for reuse_index in range(config.reuse_count):
            preferred = CLEARANCE_TARGETS[reuse_index % len(CLEARANCE_TARGETS)]
            fallback = CLEARANCE_TARGETS[(reuse_index + 1) % len(CLEARANCE_TARGETS)]
            per_label = max(1, config.max_cylinder_attempts // 2)
            accepted_slot = False
            for label in (preferred, fallback):
                if accepted_slot or not budget_left():
                    break
                for _ in range(per_label):
                    if not budget_left():
                        break
                    show_stage("sampling", template_index)
                    stage_started = time.perf_counter()
                    report["cylinder_sampling_calls"] += 1
                    cylinder, reason = cylinder_sampler(
                        rng, template_view, config, getattr(config, f"{label}_clearance")
                    )
                    stage_seconds["sampling"] += time.perf_counter() - stage_started
                    if cylinder is None:
                        record_failure(str(reason or "cylinder_sampling_exhausted"))
                        continue
                    request = dict(
                        kind="obstacle",
                        template_index=int(template_index),
                        reuse_index=int(reuse_index),
                        seed=seed,
                        cubes=cubes,
                        cylinder=cylinder,
                        plan=template_record["plan"],
                    )
                    with tempfile.TemporaryDirectory(prefix="constructive-episode-") as scratch:
                        show_stage("executing", template_index)
                        stage_started = time.perf_counter()
                        valid, attempt = episode_recorder(
                            Path(scratch), "traj", request, config,
                        )
                        stage_seconds["episodes"] += time.perf_counter() - stage_started
                        if valid:
                            publish_episode(attempt, request, Path(scratch))
                    report["physics_attempts"] += 1
                    report["episodes"].append(attempt)
                    if not valid:
                        record_failure(str(attempt.get("failure") or "unknown"))
                        continue
                    state = read_initial_robot_cube_state(h5_path, f"traj_{report['accepted']}")
                    if initial_state_hash(state) != template_record["initial_state_hash"]:
                        raise RuntimeError(
                            f"template {template_index}: episode initial state differs "
                            "from the template run's recorded initial state"
                        )
                    if template_state is not None and not all(
                        np.array_equal(state[leaf], template_state[leaf])
                        for leaf in state
                    ):
                        raise RuntimeError(
                            f"template {template_index}: reuse episodes do not share the "
                            "identical recorded initial robot/cube state"
                        )
                    template_state = state
                    report["accepted"] += 1
                    template_record["reuse_accepted"] += 1
                    template_record["reuse"].append(dict(
                        reuse_index=int(reuse_index),
                        seed=seed,
                        cylinder=cylinder,
                        initial_state_hash=attempt.get("initial_state_hash"),
                        elapsed_seconds=attempt.get("elapsed_seconds"),
                    ))
                    accepted_slot = True
                    if progress is not None:
                        progress.set_postfix(
                            phase="saved", template=template_index + 1,
                            attempts=report["physics_attempts"], refresh=False,
                        )
                        progress.update(1)
                    break

    report["complete"] = report["accepted"] >= config.target_successes
    if report["complete"]:
        show_stage("validating")
        validate_aggregate(h5_path, json_path, report["accepted"])
    show_stage("done" if report["complete"] else "budget_exhausted")
    return write_report()


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="outputs/stackcube_obstacle_constructive_action_first"
    )
    parser.add_argument("--target-successes", type=int, default=10)
    parser.add_argument("--reuse-count", type=int, default=2)
    parser.add_argument("--max-templates", type=int, default=200)
    parser.add_argument("--max-episodes", type=int, default=400)
    parser.add_argument("--max-cylinder-attempts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = Config(
        target_successes=args.target_successes,
        reuse_count=args.reuse_count,
        seed=args.seed,
        max_templates=args.max_templates,
        max_episodes=args.max_episodes,
        max_cylinder_attempts=args.max_cylinder_attempts,
    )
    with tqdm(
        total=config.target_successes, desc="Collecting", unit="traj",
        dynamic_ncols=True, miniters=0,
    ) as progress:
        report = collect(args.output_dir, config, progress=progress)
    if not report["complete"]:
        raise RuntimeError(
            f"collection budget exhausted: {report['accepted']}/{config.target_successes}"
        )
    print(
        f"accepted {report['accepted']} episodes from "
        f"{report['distinct_templates']} templates in {report['elapsed_seconds']:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
