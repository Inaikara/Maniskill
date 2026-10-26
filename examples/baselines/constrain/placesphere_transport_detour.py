"""PlaceSphere mid-transport side-detour scenes.

Radius stays 0.03 m. Height is the measured sphere and gripper occupancy of the
direct transport plus a geometric top cover, accepted only when the same-side
detour and both endpoint sweeps clear that column. It is not a prescribed
height and not a surface clearance.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import os
import signal
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from examples.baselines.constrain.pickcube_transport_detour import (
    measure_executed_join,
    midsegment_candidates,
    run_bounded_process,
)


RADIUS_M = 0.03
SPHERE_RADIUS_M = 0.02
SAFETY_MARGIN_M = 0.005
GEOMETRIC_TOP_COVER_M = 0.01
MAX_JOIN_HEADING_STEP_DEG = 45.0
MIN_JOIN_TRANSPORT_STEPS = 4
BIN_DISTANCE_WINDOW_M = None
RETREAT_DZ_M = 0.08
MEASURE_ENV_HEIGHT_M = 0.05
TABLE_BOUNDS = (-0.25, -0.25, 0.25, 0.25)
WAYPOINT_OFFSETS_M = (0.24, 0.22, 0.20, 0.18, 0.16)
LIFT_HEIGHTS_M = (0.10, 0.12)
LAYOUT_TRIES = 6
EXECUTION_CANDIDATES = 2
FORMAT = "placesphere-transport-detour-scenes-v1"
ENDPOINT_START_PHASES = frozenset({"grasp_approach", "grasp_pose", "gripper_close", "lift"})
ENDPOINT_TARGET_PHASES = frozenset({"container_approach", "release", "retreat", "settle"})
POLICY_OBS_DIM = 39
RAW_OBS_DIM = 42
OBS_ATOL = 1e-7


@dataclass(frozen=True)
class HeightDecision:
    accepted: bool
    height_m: float | None
    reason: str | None
    sphere_inside_max_z_m: float | None = None
    gripper_inside_max_z_m: float | None = None
    gripper_top_cover_feasible: bool = True


@dataclass(frozen=True)
class ClearanceReport:
    geometric_vertex_clearance_m: float
    geometric_is_surface_clearance: bool
    object_xy_margin_m: float
    robot_surface_clearance_m: float | None
    safety_proof: str


@dataclass(frozen=True)
class JoinMotionConfig:
    transport_fillet_radius_m: float = 0.04
    waypoint_tolerance: float = 0.01
    max_steps_per_waypoint: int = 60
    transport_lookahead_m: float = 0.04
    action_cap: float = 0.35


def select_cylinder_height(sphere_points, gripper_points, side_points, end_points, *, center_xy, radius):
    """Height from sphere and gripper samples inside the direct column. No height argument."""
    _require_fixed_radius(radius)
    sphere = _xyz(sphere_points)
    gripper = _xyz(gripper_points)
    side = _xyz(side_points)
    end = _xyz(end_points)
    if len(sphere) == 0 or len(gripper) == 0 or len(side) == 0 or len(end) == 0:
        return HeightDecision(False, None, "sweep_not_recorded")
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    sphere_inside = sphere[_xy_distance(sphere, center) <= radius]
    gripper_inside = gripper[_xy_distance(gripper, center) <= radius]
    if len(sphere_inside) == 0 and len(gripper_inside) == 0:
        return HeightDecision(False, None, "direct_transport_misses_cylinder")
    sphere_z = None if len(sphere_inside) == 0 else float(np.max(sphere_inside[:, 2]))
    gripper_z = None if len(gripper_inside) == 0 else float(np.max(gripper_inside[:, 2]))
    full = max(value for value in (sphere_z, gripper_z) if value is not None)
    full_height = float(full + GEOMETRIC_TOP_COVER_M)
    if not _column_occupies(side, center, radius, full_height) and not _column_occupies(end, center, radius, full_height):
        return HeightDecision(True, full_height, None, sphere_z, gripper_z, True)
    if sphere_z is None:
        reason = "side_detour_occupies_cylinder_column" if _column_occupies(side, center, radius, full_height) else "end_region_sweep_occupies_cylinder_column"
        return HeightDecision(False, None, reason, sphere_z, gripper_z, False)
    sphere_height = float(sphere_z + GEOMETRIC_TOP_COVER_M)
    if _column_occupies(side, center, radius, sphere_height):
        return HeightDecision(False, None, "side_detour_occupies_cylinder_column", sphere_z, gripper_z, False)
    if _column_occupies(end, center, radius, sphere_height):
        return HeightDecision(False, None, "end_region_sweep_occupies_cylinder_column", sphere_z, gripper_z, False)
    return HeightDecision(True, sphere_height, None, sphere_z, gripper_z, False)


def classify_clearances(*, vertex_xy_distance_m, radius_m, sphere_center_distance_m,
                        forbidden_radius_m, contact_separation_m, contact_free):
    """Keep the vertex approximation, sphere XY margin, and surface gap distinct."""
    _require_fixed_radius(radius_m)
    if contact_free and contact_separation_m is None:
        proof = "contact_free_execution"
    elif contact_separation_m is not None and contact_free:
        proof = "measured_surface_separation"
    else:
        proof = "contact"
    return ClearanceReport(
        geometric_vertex_clearance_m=float(vertex_xy_distance_m) - float(radius_m),
        geometric_is_surface_clearance=False,
        object_xy_margin_m=float(sphere_center_distance_m) - float(forbidden_radius_m),
        robot_surface_clearance_m=None if contact_separation_m is None else float(contact_separation_m),
        safety_proof=proof,
    )


def same_side_waypoint(start, goal, base_xy, offset, table_bounds):
    """Fixed-base side only. An out-of-bounds point is None, never the other side."""
    start = np.asarray(start, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    base = np.asarray(base_xy, dtype=np.float64).reshape(2)
    direction = goal - start
    distance = float(np.linalg.norm(direction))
    if distance == 0.0:
        return None
    along = direction / distance
    base_side = float(direction[0] * (base[1] - start[1]) - direction[1] * (base[0] - start[0]))
    if base_side == 0.0:
        return None
    perpendicular = np.array([-along[1], along[0]]) * math.copysign(1.0, base_side)
    point = 0.5 * (start + goal) + perpendicular * float(offset) + along * (0.25 * distance)
    xmin, ymin, xmax, ymax = table_bounds
    if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
        return None
    return point


def shared_transport_fork(direct, detour):
    """Share grasp/lift and container release/retreat/settle. Only the detour inserts a waypoint."""
    if _has_waypoint(direct):
        return False, "direct_has_waypoint"
    if detour["targets"].get("waypoint_xyz") is None:
        return False, "detour_missing_waypoint"
    if not _phase_status(direct, "lift", {"reached"}) or not _phase_status(detour, "lift", {"reached"}):
        return False, "lift_not_shared"
    if not _phase_status(direct, "transport", {"reached"}) or not _phase_status(detour, "transport", {"reached"}):
        return False, "transport_not_shared"
    for name in ("container_approach", "release", "retreat", "settle"):
        allowed = {"reached", "episode_ended"}
        if not _phase_status(direct, name, allowed) or not _phase_status(detour, name, allowed):
            return False, f"{name}_not_shared"
    direct_targets = direct["targets"]
    detour_targets = detour["targets"]
    if direct_targets.get("retreat_required") is not True or detour_targets.get("retreat_required") is not True:
        return False, "retreat_not_required"
    if direct_targets.get("release_required") is not True or detour_targets.get("release_required") is not True:
        return False, "release_not_required"
    for key in ("lift_xyz", "transport_exit_xyz", "container_approach_xyz", "retreat_xyz"):
        if not _close(direct_targets.get(key), detour_targets.get(key)):
            return False, f"{key}_mismatch"
    waypoint = np.asarray(detour_targets["waypoint_xyz"], dtype=np.float64)
    lift = np.asarray(detour_targets["lift_xyz"], dtype=np.float64)
    if abs(float(waypoint[2]) - float(lift[2])) > 1e-3:
        return False, "waypoint_z_discontinuous"
    for record in (direct, detour):
        terminal = record.get("terminal") or {}
        if not terminal.get("on_bin") or not terminal.get("static"):
            return False, "native_success_not_place"
        if not terminal.get("ungrasped") or terminal.get("grasped"):
            return False, "native_success_not_released"
        accepted, reason = _join_accepted(record)
        if not accepted:
            return False, reason
    return True, None


def endpoint_sweep_points(body_sweep):
    """Full-robot vertices of shared start and container phases. Not a center-distance window."""
    rows = []
    for step in body_sweep:
        if step["phase"] not in ENDPOINT_START_PHASES and step["phase"] not in ENDPOINT_TARGET_PHASES:
            continue
        for points in step["bodies"].values():
            rows.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return _cat(rows)


def endpoint_column_clear(direct_end, side_end, *, center_xy, radius, height):
    """Geometric vertex occupancy of both paths. Not surface clearance and not a distance window."""
    _require_fixed_radius(radius)
    direct = _xyz_or_empty(direct_end)
    side = _xyz_or_empty(side_end)
    if len(direct) == 0 or len(side) == 0:
        return False, "endpoint_sweep_incomplete"
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    if _column_occupies(direct, center, radius, height):
        return False, "direct_endpoint_occupies_cylinder_column"
    if _column_occupies(side, center, radius, height):
        return False, "side_endpoint_occupies_cylinder_column"
    return True, None


def classify_direct_block(contact):
    """A transport contact stays a transport block even near the bin."""
    if not isinstance(contact, dict) or "phase" not in contact:
        return False, "direct_transport_not_blocked"
    phase = contact["phase"]
    if phase in ENDPOINT_START_PHASES or phase in ENDPOINT_TARGET_PHASES:
        return False, "direct_block_contact_in_endpoint"
    if phase != "transport":
        return False, "direct_block_contact_outside_transport"
    return True, None


def generate(output_dir, *, seed=1, target=1, max_attempts=1, max_wall_seconds=1.0,
             per_attempt_deadline=1.0, worker_argv_fn=None, demo_dir=None,
             reject_metadata=None, note_accepted=None):
    """Finite attempt loop. Every invocation counts, including deadlines."""
    if (target < 1 or max_attempts < 1 or per_attempt_deadline <= 0
            or not np.isfinite(per_attempt_deadline)
            or max_wall_seconds <= 0 or not np.isfinite(max_wall_seconds)):
        raise ValueError("target, attempt budget, deadline, and wall budget must be positive and finite")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    identity = freeze_source_identity()
    (output / "source_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    _write_run_config(output, seed, target, max_attempts, max_wall_seconds, per_attempt_deadline)
    if worker_argv_fn is None:
        contract = _runtime_observation_contract()
        (output / "runtime_observation_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n")
    journal = output / "attempts.jsonl"
    started = time.monotonic()
    accepted_rows = []
    attempts = 0
    failures = {}
    phase_elapsed = {}
    while len(accepted_rows) < target and attempts < max_attempts and (time.monotonic() - started) < max_wall_seconds:
        attempts += 1
        result_dir = output / "workers" / f"{attempts:04d}"
        result_dir.mkdir(parents=True)
        attempt_seed = int(seed) + attempts
        argv = (worker_argv_fn(attempt_seed, result_dir) if worker_argv_fn is not None
                else _default_worker_argv(attempt_seed, result_dir, per_attempt_deadline))
        at = time.monotonic()
        outcome = run_bounded_process(argv, per_attempt_deadline, result_dir / "worker.stderr")
        elapsed = time.monotonic() - at
        result_path = result_dir / "result.json"
        payload = {}
        if outcome["timed_out"]:
            reason = "attempt_deadline"
            accepted_row = False
        elif result_path.exists():
            payload = json.loads(result_path.read_text())
            accepted_row = bool(payload.get("accepted"))
            reason = payload.get("failure")
        else:
            accepted_row = False
            reason = f"worker_exit_{outcome['returncode']}"
        if accepted_row and reason:
            accepted_row = False
        if accepted_row and reject_metadata is not None:
            rejection = reject_metadata(payload.get("metadata") or {})
            if rejection:
                accepted_row = False
                reason = rejection
        if accepted_row and not (result_dir / "scene.h5").exists():
            accepted_row = False
            reason = "missing_scene"
        if accepted_row:
            try:
                ingested = _ingest_scene(output, result_dir, len(accepted_rows), payload, demo_dir)
                accepted_rows.append(ingested)
                if note_accepted is not None:
                    note_accepted(ingested)
            except Exception as exc:
                accepted_row = False
                reason = f"ingest_failed:{type(exc).__name__}"
                payload = dict(payload)
                payload["ingest_error"] = f"{type(exc).__name__}: {exc}"
        if not accepted_row:
            reason = reason or "unspecified"
            failures[reason] = failures.get(reason, 0) + 1
        for phase, value in payload.get("phase_timings", {}).items():
            phase_elapsed[phase] = phase_elapsed.get(phase, 0.0) + float(value)
        record = {"attempt": attempts, "seed": attempt_seed, "accepted": accepted_row,
                  "failure": None if accepted_row else reason, "elapsed_seconds": elapsed,
                  "timed_out": outcome["timed_out"], "returncode": outcome["returncode"],
                  "counters": payload.get("counters", {}),
                  "phase_timings": payload.get("phase_timings", {}),
                  "diagnostics": payload.get("diagnostics", {})}
        stderr_path = result_dir / "worker.stderr"
        if stderr_path.exists() and not accepted_row:
            tail = stderr_path.read_text(errors="replace")[-800:]
            if tail:
                record["stderr_tail"] = tail
        with journal.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    _write_manifest(output, accepted_rows, seed)
    audit_ok = True
    if accepted_rows and worker_argv_fn is None:
        audit_ok = audit(output)
    report = {"target": target, "accepted": len(accepted_rows), "attempts": attempts,
              "complete": len(accepted_rows) == target and audit_ok, "failures": failures,
              "elapsed_seconds": time.monotonic() - started,
              "acceptance_rate": len(accepted_rows) / attempts if attempts else 0.0,
              "wall_budget_seconds": max_wall_seconds,
              "per_attempt_deadline_seconds": per_attempt_deadline,
              "generation_seed": seed, "restore_audit_ok": audit_ok,
              "phase_elapsed_seconds": phase_elapsed}
    identity_bytes = (output / "source_identity.json").read_bytes()
    report["source_files_sha256"] = json.loads(identity_bytes)["files_sha256"]
    report["source_identity_sha256"] = hashlib.sha256(identity_bytes).hexdigest()
    report["git_head_is_code_identity"] = False
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write_artifact_hashes(output)
    return report


class AttemptDeadline(Exception):
    pass


def _alarm(_signum, _frame):
    raise AttemptDeadline()


def run_attempt(seed, result_dir):
    """One layout search. Closes every env it opens."""
    envs = []
    outcome = {"accepted": False, "failure": "attempt_incomplete", "counters": {}, "phase_timings": {}}
    try:
        outcome = _attempt_body(int(seed), Path(result_dir), envs)
    except AttemptDeadline:
        raise
    except Exception as exc:
        outcome = {"accepted": False, "failure": f"worker_error:{type(exc).__name__}",
                   "counters": {}, "phase_timings": {},
                   "diagnostics": {"error": str(exc), "traceback": traceback.format_exc()}}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        close_errors = []
        for env in envs:
            try:
                env.close()
            except Exception as exc:
                close_errors.append(f"{type(exc).__name__}: {exc}")
        if close_errors:
            outcome = {"accepted": False, "failure": "env_close_failed", "counters": {},
                       "phase_timings": {}, "diagnostics": {"errors": close_errors}}
    return outcome


def _attempt_body(seed, result_dir, envs):
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSphereLayoutConfig, PlaceSpherePilotConfig, make_env, validate_placesphere_layout,
    )

    counters = {"geometry_tries": 0, "direct_executions": 0, "side_executions": 0,
                "block_executions": 0, "safe_executions": 0, "centers_tested": 0,
                "waypoints_tested": 0, "lifts_tested": 0, "endpoint_replays": 0}
    timings = {"measure": 0.0, "block": 0.0, "safe": 0.0, "endpoint": 0.0}
    notes = []

    def fail(reason, **extra):
        if len(notes) < 12:
            notes.append({"reason": reason, **extra})
        return {"accepted": False, "failure": reason, "counters": counters,
                "phase_timings": timings, "diagnostics": {"rejections": notes}}

    rng = np.random.default_rng(seed)
    measure = make_env(config=PlaceSpherePilotConfig(
        max_steps=320, obstacle_radius=RADIUS_M, obstacle_height=MEASURE_ENV_HEIGHT_M,
        safety_margin=SAFETY_MARGIN_M))
    envs.append(measure)
    base_xy = measure.unwrapped.agent.robot.pose.p[0, :2].detach().cpu().numpy()
    layout_config = PlaceSphereLayoutConfig()
    start = goal = None
    for _ in range(LAYOUT_TRIES):
        counters["geometry_tries"] += 1
        candidate_start = np.array([rng.uniform(-0.19, -0.08), rng.uniform(-0.13, 0.13)])
        candidate_goal = np.array([rng.uniform(0.04, 0.15), rng.uniform(-0.13, 0.13)])
        valid, _, _, _ = validate_placesphere_layout(
            candidate_start, candidate_goal, base_xy, layout_config, container_half_extent=0.025)
        if valid:
            start, goal = candidate_start, candidate_goal
            break
    if start is None:
        return fail("geometry_retry_exhausted")

    last_reason = "no_feasible_height_at_radius_0.03"
    saw_geometry = False
    for lift in LIFT_HEIGHTS_M:
        counters["lifts_tested"] += 1
        started = time.monotonic()
        direct = run_transport_expert(
            measure, start, goal, detour=False, seed=seed, max_steps=320,
            lift_height=float(lift), record_body_sweep=True)
        counters["direct_executions"] += 1
        timings["measure"] += time.monotonic() - started
        if direct is None or not _placed(direct):
            last_reason = "grasp_pose_unsolved" if direct is None else "direct_template_unsuccessful"
            notes.append({"reason": last_reason, "lift_height_m": float(lift),
                          "phases": None if direct is None else direct["trace"]["phase_status"]})
            continue
        snapshot = direct["initial_snapshot"]
        targets = direct["trace"]["shared_targets"]
        direct_transport_sphere = _named_points(direct["trace"]["body_sweep"], "transport", {"sphere"})
        direct_transport_gripper = _named_points(
            direct["trace"]["body_sweep"], "transport", None, gripper=True)
        direct_end = endpoint_sweep_points(direct["trace"]["body_sweep"])
        if len(direct_transport_sphere) == 0 or len(direct_transport_gripper) == 0 or len(direct_end) == 0:
            last_reason = "sweep_not_recorded"
            continue
        for offset in WAYPOINT_OFFSETS_M:
            counters["waypoints_tested"] += 1
            waypoint = same_side_waypoint(start, goal, base_xy, offset, TABLE_BOUNDS)
            if waypoint is None:
                last_reason = "waypoint_out_of_bounds"
                continue
            started = time.monotonic()
            side = run_transport_expert(
                measure, start, goal, detour=True, waypoint=waypoint, seed=seed, max_steps=320,
                lift_height=float(lift), initial_snapshot=snapshot, shared_targets=targets,
                record_body_sweep=True)
            counters["side_executions"] += 1
            timings["measure"] += time.monotonic() - started
            if side is None or not _placed(side):
                last_reason = "side_template_unsuccessful"
                continue
            fork_ok, fork_reason = shared_transport_fork(_fork_record(direct), _fork_record(side))
            if not fork_ok:
                last_reason = fork_reason
                continue
            side_all = _named_points(side["trace"]["body_sweep"], None, None)
            side_end = endpoint_sweep_points(side["trace"]["body_sweep"])
            if len(side_all) == 0 or len(side_end) == 0:
                last_reason = "sweep_not_recorded"
                continue
            ranked = []
            for center in midsegment_candidates(start, goal, base_xy, radius=RADIUS_M):
                counters["centers_tested"] += 1
                decision = select_cylinder_height(
                    direct_transport_sphere, direct_transport_gripper, side_all,
                    _cat([direct_end, side_end]), center_xy=center, radius=RADIUS_M)
                if not decision.accepted:
                    last_reason = decision.reason
                    continue
                clear, clear_reason = endpoint_column_clear(
                    direct_end, side_end, center_xy=center, radius=RADIUS_M, height=decision.height_m)
                if not clear:
                    last_reason = clear_reason
                    continue
                low = side_all[side_all[:, 2] <= decision.height_m]
                gap = float(np.linalg.norm(low[:, :2] - center, axis=1).min()) if len(low) else 1.0
                ranked.append((gap, decision, center))
            ranked.sort(key=lambda item: item[0], reverse=True)
            saw_geometry = saw_geometry or bool(ranked)
            for gap, decision, center in ranked[:EXECUTION_CANDIDATES]:
                blocked, block_reason, block = _execute_block(
                    envs, snapshot, start, goal, center, decision.height_m, seed, targets, counters, timings)
                if not blocked:
                    last_reason = block_reason
                    if len(notes) < 8:
                        notes.append({"reason": block_reason, "height_m": decision.height_m})
                    continue
                saved, safe_reason, safe = _execute_safe(
                    envs, snapshot, start, goal, waypoint, center, decision.height_m, seed,
                    targets, counters, timings)
                if not saved:
                    last_reason = safe_reason
                    if len(notes) < 8:
                        notes.append({"reason": safe_reason, "height_m": decision.height_m})
                    continue
                entered, entry_reason, entry = _replay_direct_container_phase(
                    envs, direct, center, decision.height_m, seed, start, goal, counters, timings)
                if not entered:
                    last_reason = entry_reason
                    if len(notes) < 8:
                        notes.append({"reason": entry_reason, "height_m": decision.height_m})
                    continue
                clearance = _clearance_from_run(safe, center, decision.height_m)
                metadata = _scene_metadata(
                    seed, start, goal, base_xy, center, decision, clearance, direct, side, safe,
                    block, lift, offset, gap, direct_end, side_end, entry)
                _write_attempt_scene(result_dir, safe, block, metadata, direct=direct)
                return {"accepted": True, "failure": None, "counters": counters,
                        "phase_timings": timings, "metadata": metadata,
                        "diagnostics": {"rejections": notes, "selected_gap_m": gap}}
        if saw_geometry:
            break
    return fail(last_reason)


def run_transport_expert(env, start_xy, container_xy, *, detour, seed, max_steps, waypoint=None,
                         obstacle_enabled=False, obstacle_center=None, initial_snapshot=None,
                         lift_height=0.10, record_body_sweep=True, stop_on_contact=False,
                         shared_targets=None, terminal_mode="legacy", release_margin_m=None):
    """Execute grasp, lift, optional side join, then release, retreat, and settle."""
    import sapien
    import torch
    from examples.baselines.constrain.pickcube_expert import (
        _clone_tree, _follow_transport_polyline, _reset_elapsed_steps, _set_state_dict_exact,
        snapshot_execution_state,
    )
    from examples.baselines.constrain.placesphere_obstacle import OBSTACLE_KEY
    from examples.baselines.constrain.placesphere_pilot import (
        _CompatiblePandaArmMotionPlanningSolver, _obs_vector, _official_grasp_angles,
        _policy_observation, scale_native_euler_action,
    )
    from mani_skill.examples.motionplanning.base_motionplanner.utils import (
        compute_grasp_info_by_obb, get_actor_obb,
    )
    from scipy.spatial.transform import Rotation

    base = env.unwrapped
    start_xy = np.asarray(start_xy, dtype=np.float64)
    container_xy = np.asarray(container_xy, dtype=np.float64)
    if initial_snapshot is None:
        obs, info = env.reset(seed=seed, options={
            "sphere_xy": start_xy, "container_xy": container_xy,
            "obstacle_enabled": obstacle_enabled,
            "obstacle_center": obstacle_center or (0.0, 0.2)})
    else:
        state, controller = initial_snapshot
        state = _clone_tree(state)
        if obstacle_enabled:
            extra = state[OBSTACLE_KEY]
            extra[0, :2] = torch.as_tensor(obstacle_center, dtype=extra.dtype, device=extra.device)
            extra[0, 3] = 1
            extra[0, 4:7] = 0
        _set_state_dict_exact(base, state)
        base.agent.controller.set_state(_clone_tree(controller))
        _reset_elapsed_steps(env, base)
        base._prev_sphere_xy[:] = base.obj.pose.p[:, :2]
        base._first_obstacle_contact = None
        base._obstacle_collision[:] = False
        base._forbidden_crossing[:] = False
        obs = base.get_obs()
        info = base.evaluate()
    policy_obs = _policy_observation(base, info)
    raw_obs = _obs_vector(base.get_obs(info=info, unflattened=True))
    approaching = np.array([0.0, 0.0, -1.0])
    closing = base.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(
        get_actor_obb(base.obj), approaching=approaching, target_closing=closing, depth=0.025)
    grasp_pose = base.agent.build_grasp_pose(approaching, grasp_info["closing"], grasp_info["center"])
    if shared_targets is None:
        planner = _CompatiblePandaArmMotionPlanningSolver(
            env, debug=False, vis=False, base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False)
        chosen = None
        for angle in _official_grasp_angles():
            candidate = grasp_pose * sapien.Pose(q=__import__(
                "transforms3d.euler", fromlist=["euler2quat"]).euler2quat(0, 0, float(angle)))
            if planner.move_to_pose_with_screw(candidate, dry_run=True) == -1:
                continue
            chosen = candidate
            break
        if chosen is None:
            return None
        grasp_approach = np.asarray((chosen * sapien.Pose([0, 0, -0.05])).p, dtype=np.float64)
        grasp_xyz = np.asarray(chosen.p, dtype=np.float64)
        grasp_quat = np.asarray(chosen.q, dtype=np.float64)
        lift_xyz = grasp_xyz.copy()
        lift_xyz[2] += float(lift_height)
    else:
        grasp_approach = np.asarray(shared_targets["grasp_approach_xyz"], dtype=np.float64)
        grasp_xyz = np.asarray(shared_targets["grasp_pose_xyz"], dtype=np.float64)
        grasp_quat = np.asarray(shared_targets["grasp_quat"], dtype=np.float64)
        lift_xyz = np.asarray(shared_targets["lift_xyz"], dtype=np.float64)
    actions, states, policy_rows, raw_rows = [], [_clone_tree(base.get_state_dict())], [policy_obs], [raw_obs]
    infos = [_clone_tree(info)]
    controller_states = [_clone_tree(base.agent.controller.get_state())]
    terminated, truncated = [False], [False]
    current_phase = "grasp_approach"
    trace = {"phase_status": [], "tcp_xyz": [base.agent.tcp.pose.p[0].detach().cpu().numpy().copy()],
             "step_phase": [], "body_sweep": [], "sphere_xy": [],
             "waypoint_xy": None if waypoint is None else np.asarray(waypoint, dtype=np.float64).reshape(2).tolist()}
    done = False
    release_begin = None
    join_config = JoinMotionConfig()

    def step(action):
        nonlocal obs, info, done, policy_obs, raw_obs
        if done or len(actions) >= max_steps:
            done = True
            return True
        action = np.asarray(action, dtype=np.float32).reshape(7)
        _, _, term, trunc, info = env.step(torch.as_tensor(action, dtype=torch.float32, device=base.device))
        policy_obs = _policy_observation(base, info)
        raw_obs = _obs_vector(base.get_obs(info=info, unflattened=True))
        actions.append(action.copy())
        states.append(_clone_tree(base.get_state_dict()))
        policy_rows.append(policy_obs)
        raw_rows.append(raw_obs)
        infos.append(_clone_tree(info))
        controller_states.append(_clone_tree(base.agent.controller.get_state()))
        terminated.append(bool(term))
        truncated.append(bool(trunc))
        trace["tcp_xyz"].append(base.agent.tcp.pose.p[0].detach().cpu().numpy().copy())
        trace["step_phase"].append(current_phase)
        trace["sphere_xy"].append(base.obj.pose.p[0, :2].detach().cpu().numpy().copy())
        if record_body_sweep:
            trace["body_sweep"].append({"phase": current_phase, "bodies": _collision_bodies(base)})
        if base._first_obstacle_contact is not None and "first_obstacle_contact" not in trace:
            trace["first_obstacle_contact"] = {
                **base._first_obstacle_contact, "control_step": len(actions), "phase": current_phase,
                "sphere_xyz": base.obj.pose.p[0].detach().cpu().numpy().copy().tolist(),
                "tcp_xyz": base.agent.tcp.pose.p[0].detach().cpu().numpy().copy().tolist(),
            }
        native = bool(base.evaluate()["native_success"][0].item())
        done = bool(term or trunc or native)
        if stop_on_contact and bool(base._obstacle_collision[0].item()):
            done = True
        return done

    def move(target, target_q=None, grip=1.0, tolerance=0.006, budget=80):
        target = torch.as_tensor(target, dtype=torch.float32, device=base.device)
        for _ in range(budget):
            if done or len(actions) >= max_steps:
                return False
            delta = target - base.agent.tcp.pose.p[0]
            rot = torch.zeros(3, device=base.device)
            rot_done = True
            if target_q is not None:
                current = base.agent.tcp.pose.q[0].detach().cpu().numpy()
                dq = Rotation.from_quat(np.asarray(target_q)[[1, 2, 3, 0]]) * Rotation.from_quat(current[[1, 2, 3, 0]]).inv()
                euler = dq.as_euler("XYZ")
                arm = base.agent.controller.controllers["arm"]
                rot = torch.as_tensor(scale_native_euler_action(euler, arm.config.rot_lower, 0.35),
                                      device=base.device, dtype=torch.float32)
                rot_done = np.linalg.norm(dq.as_rotvec()) < 0.03
            if torch.linalg.norm(delta).item() < tolerance and rot_done:
                return True
            pos = (delta / 0.1).clamp(-0.35, 0.35)
            if step(torch.cat((pos, rot, torch.tensor([grip], device=base.device)))):
                return bool(base.evaluate()["native_success"][0].item())
        return False

    def mark(name, reached):
        trace["phase_status"].append((name, "reached" if reached else "timeout"))
        return reached

    ok = True
    for phase, pos, grip in (
        ("grasp_approach", grasp_approach, 1.0),
        ("grasp_pose", grasp_xyz, 1.0),
    ):
        current_phase = phase
        ok = mark(phase, move(pos, grasp_quat, grip=grip))
        if not ok:
            break
    if ok:
        current_phase = "gripper_close"
        for _ in range(8):
            if step([0, 0, 0, 0, 0, 0, -1]):
                ok = False
                break
        trace["phase_status"].append(("gripper_close", "reached" if ok else "episode_ended"))
    if ok:
        current_phase = "lift"
        ok = mark("lift", move(lift_xyz, grasp_quat, grip=-1.0))
    if ok and shared_targets is None:
        goal_pose = base.bin.pose.sp * sapien.Pose([0, 0, 2 * float(base.block_half_size[2])])
        offset = np.asarray(goal_pose.p, dtype=np.float64) - base.obj.pose.p[0].detach().cpu().numpy()
        container_approach = np.asarray(lift_xyz, dtype=np.float64) + offset
        retreat = container_approach.copy()
        retreat[2] += RETREAT_DZ_M
        transport_exit = np.array([container_approach[0], container_approach[1], lift_xyz[2]], dtype=np.float64)
        waypoint_xyz = None
        if waypoint is not None:
            waypoint_xyz = np.array([float(waypoint[0]), float(waypoint[1]), float(lift_xyz[2])])
        shared_targets = {
            "grasp_approach_xyz": grasp_approach.tolist(),
            "grasp_pose_xyz": grasp_xyz.tolist(),
            "grasp_quat": grasp_quat.tolist(),
            "lift_xyz": np.asarray(lift_xyz, dtype=np.float64).tolist(),
            "transport_exit_xyz": transport_exit.tolist(),
            "container_approach_xyz": container_approach.tolist(),
            "retreat_xyz": retreat.tolist(),
            "waypoint_xyz": None if waypoint_xyz is None else waypoint_xyz.tolist(),
            "retreat_required": True,
            "release_required": True,
            "retreat_dz_m": RETREAT_DZ_M,
            "retreat_dz_is_clearance_window": False,
            "lift_height_m": float(lift_height),
            "official_align": "bin_pose * [0, 0, 2 * block_half_size[2]] applied to the post-lift TCP",
        }
    if ok and detour:
        waypoint_xyz = np.asarray(waypoint, dtype=np.float64).reshape(2)
        waypoint_xyz = np.array([float(waypoint_xyz[0]), float(waypoint_xyz[1]), float(lift_xyz[2])])
        shared_targets = dict(shared_targets)
        shared_targets["waypoint_xyz"] = waypoint_xyz.tolist()
    trace["shared_targets"] = shared_targets
    if ok:
        path = [np.asarray(shared_targets["lift_xyz"], dtype=np.float64)]
        if detour:
            path.append(np.asarray(shared_targets["waypoint_xyz"], dtype=np.float64))
        path.append(np.asarray(shared_targets["transport_exit_xyz"], dtype=np.float64))
        current_phase = "transport"
        status = _follow_transport_polyline(
            base, path, step, join_config, done_fn=lambda: done or len(actions) >= max_steps)
        reached = status is None
        trace["phase_status"].append(("transport", "reached" if reached else (status or "timeout")))
        ok = reached
    if terminal_mode not in ("legacy", "above_release"):
        raise ValueError(f"unknown terminal_mode {terminal_mode}")
    if terminal_mode == "legacy":
        if ok:
            trace["container_approach_action_index"] = len(actions)
            trace["container_approach_snapshot"] = snapshot_execution_state(env)
            current_phase = "container_approach"
            ok = mark("container_approach", move(shared_targets["container_approach_xyz"], grasp_quat, grip=-1.0))
        if ok and not done:
            current_phase = "retreat"
            retreat_reached = move(shared_targets["retreat_xyz"], grasp_quat, grip=-1.0, budget=30)
            trace["phase_status"].append(("retreat", "episode_ended" if done else ("reached" if retreat_reached else "timeout")))
            ok = retreat_reached and not done
        if ok and not done:
            current_phase = "release"
            release_begin = len(actions)
            if step([0, 0, 0, 0, 0, 0, 1]):
                pass
            trace["phase_status"].append(("release", "episode_ended" if done else "reached"))
            ok = not done
        if ok and not done:
            current_phase = "settle"
            for _ in range(36):
                if step([0, 0, 0, 0, 0, 0, 1]):
                    break
            trace["phase_status"].append(("settle", "episode_ended" if done else "reached"))
    elif ok:
        from examples.baselines.constrain.placesphere_matched_terminal import execute_above_release

        if release_margin_m is None:
            raise ValueError("above_release requires a predeclared margin")
        trace["container_approach_snapshot"] = snapshot_execution_state(env)
        status = execute_above_release(
            step=step, move=move, base=base, grasp_quat=grasp_quat, shared_targets=shared_targets,
            trace=trace, cylinder_top_z=float(base.obstacle_collision_height), margin_m=release_margin_m,
            is_done=lambda: done, action_count=lambda: len(actions), bodies_fn=lambda: _collision_bodies(base))
        release_begin = status.get("release_begin")
        ok = bool(status.get("ok"))
        trace["matched_terminal"] = status
    if release_begin is not None:
        trace["release_control_step"] = next(
            (index for index in range(release_begin, len(infos)) if not _flag(infos[index], "is_obj_grasped")),
            None)
    evaluated = base.evaluate()
    terminal = _terminal(evaluated)
    release_index = trace.get("release_control_step")
    release_rows = infos[release_index:] if release_index is not None else []
    trace["post_release_safety"] = {
        "checked_through_native_success": bool(terminal["native_success"] and release_rows and _flag(infos[-1], "native_success")),
        "all_ungrasped": bool(release_rows) and all(not _flag(row, "is_obj_grasped") for row in release_rows),
        "no_obstacle_contact": bool(release_rows) and all(not _flag(row, "obstacle_collision") for row in release_rows),
        "no_forbidden_crossing": bool(release_rows) and all(not _flag(row, "obstacle_crossing") for row in release_rows),
        "terminal_on_bin": terminal["on_bin"],
        "terminal_static": terminal["static"],
        "terminal_ungrasped": terminal["ungrasped"],
    }
    return {
        "actions": np.asarray(actions, dtype=np.float32),
        "states": states,
        "policy_observations": np.asarray(policy_rows, dtype=np.float32),
        "raw_observations": np.asarray(raw_rows, dtype=np.float32),
        "infos": infos,
        "controller_states": controller_states,
        "terminated": np.asarray(terminated, dtype=np.uint8),
        "truncated": np.asarray(truncated, dtype=np.uint8),
        "initial_snapshot": (states[0], controller_states[0]),
        "initial_state": states[0],
        "initial_controller": controller_states[0],
        "trace": trace,
        "terminal": terminal,
        "safety": {
            "native_success": terminal["native_success"],
            "on_bin": terminal["on_bin"],
            "static": terminal["static"],
            "ungrasped": terminal["ungrasped"],
            "obstacle_collision": bool(base._obstacle_collision[0].item()),
            "obstacle_crossing": bool(base._forbidden_crossing[0].item()),
            "safe_success": bool(evaluated["success"][0].item()) and not bool(base._obstacle_collision[0].item())
            and not bool(base._forbidden_crossing[0].item()) and terminal["native_success"],
        },
    }


def _execute_block(envs, snapshot, start, goal, center, height, seed, targets, counters, timings):
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=320, obstacle_radius=RADIUS_M, obstacle_height=float(height), safety_margin=SAFETY_MARGIN_M))
    envs.append(env)
    started = time.monotonic()
    try:
        env.reset(seed=seed, options={
            "sphere_xy": np.asarray(start), "container_xy": np.asarray(goal),
            "obstacle_enabled": True, "obstacle_center": np.asarray(center)})
        result = run_transport_expert(
            env, start, goal, detour=False, seed=seed, max_steps=320, obstacle_enabled=True,
            obstacle_center=center, initial_snapshot=snapshot, shared_targets=targets,
            stop_on_contact=True, record_body_sweep=False)
    finally:
        env.close()
        envs.remove(env)
    counters["block_executions"] += 1
    timings["block"] += time.monotonic() - started
    if result is None:
        return False, "grasp_pose_unsolved", None
    contact = result["trace"].get("first_obstacle_contact")
    if not result["safety"]["obstacle_collision"] or contact is None:
        return False, "direct_transport_not_blocked", result
    blocked, reason = classify_direct_block(contact)
    if not blocked:
        return False, reason, result
    return True, None, result


def _execute_safe(envs, snapshot, start, goal, waypoint, center, height, seed, targets, counters, timings):
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSpherePilotConfig, _actions_are_legal, make_env,
    )

    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=320, obstacle_radius=RADIUS_M, obstacle_height=float(height), safety_margin=SAFETY_MARGIN_M))
    envs.append(env)
    started = time.monotonic()
    try:
        env.reset(seed=seed, options={
            "sphere_xy": np.asarray(start), "container_xy": np.asarray(goal),
            "obstacle_enabled": True, "obstacle_center": np.asarray(center)})
        result = run_transport_expert(
            env, start, goal, detour=True, waypoint=waypoint, seed=seed, max_steps=320,
            obstacle_enabled=True, obstacle_center=center, initial_snapshot=snapshot,
            shared_targets=targets, record_body_sweep=True)
        result["forbidden_radius_m"] = float(env.unwrapped._forbidden_radius)
        result["observation_contract_ok"] = _observation_contract(env, result, center)
    finally:
        env.close()
        envs.remove(env)
    counters["safe_executions"] += 1
    timings["safe"] += time.monotonic() - started
    if result is None or not _placed(result):
        return False, "obstacle_native_unsuccessful", result
    safety = result["safety"]
    if safety["obstacle_collision"] or safety["obstacle_crossing"] or not safety["safe_success"]:
        return False, "obstacle_contact_or_crossing", result
    proof = result["trace"].get("post_release_safety") or {}
    if not all(proof.get(key) for key in (
            "checked_through_native_success", "all_ungrasped", "no_obstacle_contact",
            "no_forbidden_crossing", "terminal_on_bin", "terminal_static", "terminal_ungrasped")):
        return False, "post_release_safety_failed", result
    if not _actions_are_legal(result["actions"]) or not result.get("observation_contract_ok"):
        return False, "observation_action_contract", result
    join = measure_executed_join(
        result["trace"]["tcp_xyz"], result["actions"], result["trace"]["step_phase"],
        max_heading_step_deg=MAX_JOIN_HEADING_STEP_DEG)
    if not join.continuous:
        return False, join.reason or "transport_join_discontinuous", result
    result["executed_join"] = join
    return True, None, result


def _replay_direct_container_phase(envs, direct, center, height, seed, start, goal, counters, timings):
    """Replay the shared container phase with the obstacle present. No distance mask."""
    import torch
    from examples.baselines.constrain.pickcube_expert import (
        _clone_tree, _reset_elapsed_steps, _set_state_dict_exact,
    )
    from examples.baselines.constrain.placesphere_obstacle import OBSTACLE_KEY
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    snapshot = direct["trace"].get("container_approach_snapshot")
    index = direct["trace"].get("container_approach_action_index")
    if snapshot is None or index is None:
        return False, "direct_container_phase_not_recorded", None
    actions = np.asarray(direct["actions"][index:])
    if len(actions) == 0:
        return False, "direct_container_phase_not_recorded", None
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=320, obstacle_radius=RADIUS_M, obstacle_height=float(height), safety_margin=SAFETY_MARGIN_M))
    envs.append(env)
    started = time.monotonic()
    try:
        env.reset(seed=seed, options={
            "sphere_xy": np.asarray(start), "container_xy": np.asarray(goal),
            "obstacle_enabled": True, "obstacle_center": np.asarray(center)})
        state, controller = snapshot
        state = _clone_tree(state)
        extra = state[OBSTACLE_KEY]
        extra[0, :2] = torch.as_tensor(center, dtype=extra.dtype, device=extra.device)
        extra[0, 3] = 1
        extra[0, 4:7] = 0
        base = env.unwrapped
        _set_state_dict_exact(base, state)
        base.agent.controller.set_state(_clone_tree(controller))
        base._first_obstacle_contact = None
        base._obstacle_collision[:] = False
        base._forbidden_crossing[:] = False
        base._prev_sphere_xy[:] = base.obj.pose.p[:, :2]
        _reset_elapsed_steps(env, base)
        for action in actions:
            env.step(torch.as_tensor(action, dtype=torch.float32, device=base.device))
            if bool(base._obstacle_collision[0].item()) or bool(base._forbidden_crossing[0].item()):
                return False, "direct_container_phase_contact", {
                    "contact": _jsonable(base._first_obstacle_contact),
                    "crossing": bool(base._forbidden_crossing[0].item()),
                }
        terminal = _terminal(base.evaluate())
        if not (terminal["native_success"] and terminal["on_bin"] and terminal["static"] and terminal["ungrasped"]):
            return False, "direct_container_phase_unsuccessful", terminal
        return True, None, {"steps": int(len(actions)), "contact": False, "crossing": False, **terminal}
    finally:
        timings["endpoint"] += time.monotonic() - started
        counters["endpoint_replays"] += 1
        env.close()
        envs.remove(env)


def _observation_contract(env, result, center):
    from examples.baselines.constrain.placesphere_obstacle import OBSTACLE_KEY
    from examples.baselines.constrain.placesphere_pilot import _policy_observation_schema

    policy = np.asarray(result["policy_observations"])
    raw = np.asarray(result["raw_observations"])
    actions = np.asarray(result["actions"])
    if policy.ndim != 2 or raw.ndim != 2 or policy.shape[1] != POLICY_OBS_DIM or raw.shape[1] != RAW_OBS_DIM:
        return False
    if policy.shape[0] != len(actions) + 1 or raw.shape[0] != len(actions) + 1:
        return False
    if not np.isfinite(policy).all() or not np.isfinite(raw).all():
        return False
    schema = _policy_observation_schema(env.unwrapped, result["infos"][0], include_obstacle_hint=True)
    offset = 0
    obstacle_slice = None
    kept = []
    for field in schema:
        width = int(field["flattened_dimension"])
        if OBSTACLE_KEY in field["path"]:
            obstacle_slice = slice(offset, offset + width)
        else:
            kept.extend(range(offset, offset + width))
        offset += width
    if obstacle_slice is None or offset != RAW_OBS_DIM or len(kept) != POLICY_OBS_DIM:
        return False
    if not np.array_equal(raw[:, kept], policy):
        return False
    hint = raw[0, obstacle_slice]
    if abs(float(hint[-1]) - RADIUS_M) > 1e-5:
        return False
    return float(np.linalg.norm(hint[:2] - np.asarray(center, dtype=np.float64))) <= 1e-4


def _collision_bodies(env):
    from examples.baselines.constrain.pickcube_expert import _shape_world_vertices

    bodies = {}
    for link in env.agent.robot.links:
        points = [_shape_world_vertices(link, shape) for shape in link._bodies[0].collision_shapes]
        points = [item for item in points if len(item)]
        if points:
            bodies[link.name] = np.concatenate(points, axis=0)
    center = env.obj.pose.p[0].detach().cpu().numpy()
    bodies["sphere"] = _sphere_surface_samples(center, float(env.radius))
    return bodies


def _sphere_surface_samples(center, radius):
    """Axis samples of the native sphere. A geometric approximation, not a mesh gap."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    axes = np.eye(3) * float(radius)
    return np.vstack([center, center + axes, center - axes])


def _named_points(body_sweep, phase, names, gripper=False):
    rows = []
    for step in body_sweep:
        if phase is not None and step["phase"] != phase:
            continue
        for name, points in step["bodies"].items():
            if gripper and ("hand" in name or "finger" in name):
                rows.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
            elif names is not None and name in names:
                rows.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
            elif names is None and not gripper:
                rows.append(np.asarray(points, dtype=np.float64).reshape(-1, 3))
    return _cat(rows)


def _placed(result):
    if result is None:
        return False
    terminal = result.get("terminal") or {}
    phases = {name for name, status in result["trace"]["phase_status"] if status in ("reached", "episode_ended")}
    needed = {"lift", "transport", "container_approach", "release", "retreat", "settle"}
    return bool(terminal.get("native_success") and terminal.get("on_bin") and terminal.get("static")
                and terminal.get("ungrasped") and not terminal.get("grasped") and needed <= phases)


def _fork_record(result):
    join = measure_executed_join(
        result["trace"]["tcp_xyz"], result["actions"], result["trace"]["step_phase"],
        max_heading_step_deg=MAX_JOIN_HEADING_STEP_DEG)
    return {
        "phases": list(result["trace"]["phase_status"]),
        "targets": result["trace"]["shared_targets"],
        "terminal": result["terminal"],
        "join": {
            "measured": join.reason != "join_not_measured",
            "continuous": join.continuous,
            "max_commanded_heading_step_deg": join.max_commanded_heading_step_deg,
            "max_tcp_heading_step_deg": join.max_tcp_heading_step_deg,
            "transport_steps": join.transport_steps,
        },
    }


def _terminal(info):
    return {
        "on_bin": _flag(info, "is_obj_on_bin"),
        "static": _flag(info, "is_obj_static"),
        "ungrasped": not _flag(info, "is_obj_grasped"),
        "grasped": _flag(info, "is_obj_grasped"),
        "native_success": _flag(info, "native_success"),
    }


def _flag(info, name):
    value = info.get(name, False)
    if hasattr(value, "detach"):
        return bool(value.reshape(-1)[0].item())
    return bool(np.asarray(value).reshape(-1)[0])


def _clearance_from_run(result, center, height):
    points = _named_points(result["trace"].get("body_sweep", []), None, None)
    occupied = points[points[:, 2] <= float(height)] if len(points) else points
    vertex_distance = None if len(occupied) == 0 else float(np.linalg.norm(occupied[:, :2] - np.asarray(center), axis=1).min())
    forbidden = float(result.get("forbidden_radius_m") or (RADIUS_M + SPHERE_RADIUS_M + SAFETY_MARGIN_M))
    sphere_distance = _grasped_sphere_distance(result, center)
    report = classify_clearances(
        vertex_xy_distance_m=0.0 if vertex_distance is None else vertex_distance,
        radius_m=RADIUS_M,
        sphere_center_distance_m=0.0 if sphere_distance is None else sphere_distance,
        forbidden_radius_m=forbidden,
        contact_separation_m=None,
        contact_free=True,
    )
    payload = report.__dict__.copy()
    payload["vertex_xy_distance_m"] = vertex_distance
    payload["geometric_vertex_clearance_m"] = None if vertex_distance is None else vertex_distance - RADIUS_M
    payload["sphere_center_distance_m"] = sphere_distance
    payload["object_xy_margin_m"] = None if sphere_distance is None else sphere_distance - forbidden
    payload["forbidden_radius_m"] = forbidden
    payload["surface_note"] = "no PhysX separation was measured; contact-free execution through native success is the safety proof"
    payload["sphere_samples_are_surface_clearance"] = False
    return payload


def _grasped_sphere_distance(result, center):
    distances = []
    for xy, info in zip(result["trace"]["sphere_xy"], result["infos"][1:]):
        if _flag(info, "is_obj_grasped"):
            distances.append(float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - np.asarray(center))))
    return None if not distances else min(distances)


def _scene_metadata(seed, start, goal, base_xy, center, decision, clearance, direct, side, safe, block,
                    lift, offset, gap, direct_end, side_end, entry):
    contact = _jsonable(block["trace"]["first_obstacle_contact"])
    join = safe["executed_join"]
    targets = side["trace"]["shared_targets"]
    corner = _turn_deg(targets["lift_xyz"][:2], targets["waypoint_xyz"][:2], targets["transport_exit_xyz"][:2])
    return {
        "seed": int(seed),
        "start_xy": np.asarray(start, dtype=np.float64).tolist(),
        "container_xy": np.asarray(goal, dtype=np.float64).tolist(),
        "base_xy": np.asarray(base_xy, dtype=np.float64).tolist(),
        "cylinder": {
            "center_xy": np.asarray(center, dtype=np.float64).tolist(),
            "radius": RADIUS_M,
            "height": float(decision.height_m),
            "height_source": (
                "measured_direct_sphere_and_gripper_occupancy_plus_geometric_top_cover"
                if decision.gripper_top_cover_feasible
                else "measured_direct_sphere_occupancy_plus_geometric_top_cover_gripper_top_infeasible"
            ),
            "gripper_top_cover_feasible": decision.gripper_top_cover_feasible,
            "geometric_top_cover_m": GEOMETRIC_TOP_COVER_M,
            "geometric_top_cover_is_surface_clearance": False,
            "sphere_occupancy_max_z_m": decision.sphere_inside_max_z_m,
            "gripper_occupancy_max_z_m": decision.gripper_inside_max_z_m,
            "direct_occupancy_max_z_m": float(decision.height_m) - GEOMETRIC_TOP_COVER_M,
            "measure_env_height_m": MEASURE_ENV_HEIGHT_M,
            "measure_env_height_is_scene_height": False,
            "sphere_surface_samples": "analytic axis samples at the native 0.02 m radius; not a PhysX mesh",
        },
        "expert": {
            "native_success": True,
            "on_bin": True,
            "static": True,
            "ungrasped": True,
            "grasped_at_success": False,
            "obstacle_collision": False,
            "obstacle_crossing": False,
            "safe_success": True,
            "release_phase_reached": True,
            "retreat_phase_reached": True,
            "settle_phase_reached": True,
            "direct_release_phase_reached": _phase_reached(direct["trace"], "release"),
            "direct_retreat_phase_reached": _phase_reached(direct["trace"], "retreat"),
            "direct_settle_phase_reached": _phase_reached(direct["trace"], "settle"),
            "direct_native_success": bool(direct["terminal"]["native_success"]),
            "direct_on_bin": bool(direct["terminal"]["on_bin"]),
            "direct_static": bool(direct["terminal"]["static"]),
            "direct_ungrasped": bool(direct["terminal"]["ungrasped"]),
            "steps": int(len(safe["actions"])),
            "waypoint_xy": side["trace"]["waypoint_xy"],
            "lift_height_m": float(lift),
            "waypoint_offset_m": float(offset),
            "alternate_side_fallback": False,
            "direct_transport_blocked": True,
            "direct_block_contact": contact,
            "clearance": clearance,
            "side_column_gap_m": float(gap),
            "fork_targets": targets,
            "retreat_required": True,
            "terminal_phase_order": [name for name, _status in safe["trace"]["phase_status"]],
            "retreat_while_grasped": True,
            "retreat_before_release_reason": "opening before the upward retreat made native success terminate during retreat and skipped settle; retreat is executed grasped, then release, then settle through the success step",
            "post_release_safety": safe["trace"]["post_release_safety"],
            "endpoint_clearance": {
                "method": "full_robot_phase_vertices",
                "phases": ["grasp_approach", "grasp_pose", "gripper_close", "lift",
                           "container_approach", "release", "retreat", "settle"],
                "paths": ["direct", "side"],
                "bin_distance_window_m": BIN_DISTANCE_WINDOW_M,
                "geometric_is_surface_clearance": False,
                "direct_min_vertex_gap_m": _vertex_gap(direct_end, center, decision.height_m),
                "side_min_vertex_gap_m": _vertex_gap(side_end, center, decision.height_m),
                "physical_direct_start_contact": False,
                "physical_direct_container_phase_contact": False,
                "physical_direct_block_phase": contact["phase"],
                "physical_side_contact": False,
                "direct_container_phase_replay": entry,
                "note": "vertex gap is column occupancy, not surface clearance; physical proof is contact-free endpoint execution",
            },
            "join": {
                "continuous": True,
                "max_commanded_heading_step_deg": join.max_commanded_heading_step_deg,
                "max_tcp_heading_step_deg": join.max_tcp_heading_step_deg,
                "transport_steps": join.transport_steps,
                "polyline_corner_deg": corner,
                "polyline_corner_is_acceptance_criterion": False,
                "criterion": "executed command and tcp XY heading step <= 45 deg",
                "slew_helper": "pickcube_expert._follow_transport_polyline heading limit 25 deg/step",
            },
        },
    }


def _phase_reached(trace, name):
    return any(phase == name and status in ("reached", "episode_ended")
               for phase, status in (trace or {}).get("phase_status", []))


def _write_attempt_scene(result_dir, safe, block, metadata, direct=None):
    import h5py

    path = Path(result_dir) / "scene.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("scene")
        _write_tree(group.create_group("state"), safe["initial_state"])
        _write_tree(group.create_group("controller"), safe["initial_controller"])
        _write_tree(group.create_group("env_states"), _stack_rows(safe["states"]))
        _write_tree(group.create_group("controller_states"), _stack_rows(safe["controller_states"]))
        group.create_dataset("expert_actions", data=np.asarray(safe["actions"], dtype=np.float32))
        group.create_dataset("policy_observations", data=np.asarray(safe["policy_observations"], dtype=np.float32))
        group.create_dataset("raw_observations", data=np.asarray(safe["raw_observations"], dtype=np.float32))
        group.create_dataset("terminated", data=np.asarray(safe["terminated"], dtype=np.uint8))
        group.create_dataset("truncated", data=np.asarray(safe["truncated"], dtype=np.uint8))
        group.create_dataset("native_success", data=np.asarray(
            [_flag(info, "native_success") for info in safe["infos"]], dtype=np.uint8))
        group.create_dataset("safe_success", data=np.asarray(
            [_flag(info, "success") for info in safe["infos"]], dtype=np.uint8))
        group.create_dataset("direct_block_actions", data=np.asarray(block["actions"], dtype=np.float32))
        if direct is not None:
            group.create_dataset("direct_actions", data=np.asarray(direct["actions"], dtype=np.float32))
            group.create_dataset("direct_policy_observations", data=np.asarray(
                direct["policy_observations"], dtype=np.float32))
            group.create_dataset("direct_raw_observations", data=np.asarray(
                direct["raw_observations"], dtype=np.float32))
            group.create_dataset("direct_terminated", data=np.asarray(direct["terminated"], dtype=np.uint8))
            group.create_dataset("direct_truncated", data=np.asarray(direct["truncated"], dtype=np.uint8))
            terminal = direct.get("terminal") or {}
            trace = direct.get("trace") or {}
            group.attrs["direct_native_success"] = int(bool(terminal.get("native_success")))
            group.attrs["direct_on_bin"] = int(bool(terminal.get("on_bin")))
            group.attrs["direct_static"] = int(bool(terminal.get("static")))
            group.attrs["direct_ungrasped"] = int(bool(terminal.get("ungrasped")))
            group.attrs["direct_release_reached"] = int(_phase_reached(trace, "release"))
            group.attrs["direct_retreat_reached"] = int(_phase_reached(trace, "retreat"))
            group.attrs["direct_settle_reached"] = int(_phase_reached(trace, "settle"))
        group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)


def _ingest_scene(output, result_dir, index, payload, demo_dir):
    import h5py

    scene_id = f"placesphere-{index:03d}"
    with h5py.File(result_dir / "scene.h5", "r") as source:
        state = _read_tree(source["scene"]["state"])
        controller = _read_tree(source["scene"]["controller"])
        actions = np.asarray(source["scene"]["expert_actions"])
        policy = np.asarray(source["scene"]["policy_observations"])
        raw = np.asarray(source["scene"]["raw_observations"])
        metadata = json.loads(source["scene"].attrs["metadata_json"])
        demo_payload = {
            "actions": actions,
            "policy": policy,
            "raw": raw,
            "terminated": np.asarray(source["scene"]["terminated"]),
            "truncated": np.asarray(source["scene"]["truncated"]),
            "native_success": np.asarray(source["scene"]["native_success"]),
            "safe_success": np.asarray(source["scene"]["safe_success"]),
            "env_states": _read_tree(source["scene"]["env_states"]),
            "controller_states": _read_tree(source["scene"]["controller_states"]),
        }
        direct_names = (
            "direct_actions", "direct_policy_observations", "direct_raw_observations",
            "direct_terminated", "direct_truncated",
        )
        direct_arrays = {name: np.asarray(source["scene"][name]) for name in direct_names
                         if name in source["scene"]}
        direct_attr_names = (
            "direct_native_success", "direct_on_bin", "direct_static", "direct_ungrasped",
            "direct_release_reached", "direct_retreat_reached", "direct_settle_reached",
        )
        direct_attrs = {key: int(source["scene"].attrs[key]) for key in direct_attr_names
                        if key in source["scene"].attrs}
        obstacle_arrays = {
            "terminated": np.asarray(source["scene"]["terminated"]),
            "truncated": np.asarray(source["scene"]["truncated"]),
            "native_success": np.asarray(source["scene"]["native_success"]),
            "safe_success": np.asarray(source["scene"]["safe_success"]),
        }
    metadata = dict(metadata)
    metadata["scene_id"] = scene_id
    metadata["state_sha256"] = _tree_hash(state)
    metadata["controller_sha256"] = _tree_hash(controller)
    metadata["policy_observation_sha256"] = hashlib.sha256(np.asarray(policy[0]).tobytes()).hexdigest()
    metadata["raw_observation_sha256"] = hashlib.sha256(np.asarray(raw[0]).tobytes()).hexdigest()
    expert = metadata["expert"]
    if not expert.get("direct_transport_blocked") or not expert.get("ungrasped") or not expert.get("static"):
        raise ValueError("scene is missing block or released-static evidence")
    metadata["source_files_sha256"] = json.loads((Path(output) / "source_identity.json").read_text())["files_sha256"]
    with h5py.File(Path(output) / "scenes.h5", "a") as handle:
        group = handle.create_group(scene_id)
        _write_tree(group.create_group("state"), state)
        _write_tree(group.create_group("controller"), controller)
        group.create_dataset("expert_actions", data=actions)
        group.create_dataset("policy_observations", data=policy)
        group.create_dataset("raw_observations", data=raw)
        group.create_dataset("initial_policy_observation", data=policy[0])
        group.create_dataset("initial_raw_observation", data=raw[0])
        for name, value in obstacle_arrays.items():
            group.create_dataset(name, data=value)
        for name, value in direct_arrays.items():
            group.create_dataset(name, data=value)
        for key, value in direct_attrs.items():
            group.attrs[key] = value
        group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
    if demo_dir is not None:
        _append_demo(Path(demo_dir), scene_id, metadata, demo_payload, state)
    return metadata


def _append_demo(demo_dir, scene_id, metadata, payload, state):
    import h5py

    demo_dir.mkdir(parents=True, exist_ok=True)
    key = f"traj_{int(scene_id.rsplit('-', 1)[-1])}"
    with h5py.File(demo_dir / "trajectory.h5", "a") as handle:
        group = handle.create_group(key)
        group.attrs["kind"] = "obstacle"
        group.attrs["scene_id"] = scene_id
        group.create_dataset("actions", data=payload["actions"])
        group.create_dataset("obs", data=payload["policy"])
        group.create_dataset("raw_obs", data=payload["raw"])
        group.create_dataset("success", data=payload["safe_success"])
        group.create_dataset("native_success", data=payload["native_success"])
        group.create_dataset("terminated", data=payload["terminated"])
        group.create_dataset("truncated", data=payload["truncated"])
        _write_tree(group.create_group("env_states"), payload["env_states"])
        _write_tree(group.create_group("controller_states"), payload["controller_states"])
    with (demo_dir / "episodes.jsonl").open("a") as stream:
        stream.write(json.dumps({"scene_id": scene_id, "traj": key, "metadata": metadata}, sort_keys=True) + "\n")


def _write_manifest(output, rows, seed):
    output = Path(output)
    scenes = output / "scenes.h5"
    payload = {
        "format": FORMAT,
        "scene_count": len(rows),
        "scenes_file": "scenes.h5",
        "scenes": rows,
        "generation_seed": seed,
        "radius_m": RADIUS_M,
        "height_rule": (
            "per scene, max z of direct-transport sphere surface samples and gripper "
            "collision vertices inside the 0.03 m column, plus 0.01 m geometric top cover"
        ),
        "native_success": "on_bin and static and not grasped",
        "source_files_sha256": json.loads((output / "source_identity.json").read_text())["files_sha256"],
        "git_head_is_code_identity": False,
        "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest() if (output / "config.json").exists() else None,
        "scenes_file_sha256": hashlib.sha256(scenes.read_bytes()).hexdigest() if scenes.exists() else None,
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


SOURCE_IDENTITY_FILES = (
    "examples/baselines/constrain/placesphere_transport_detour.py",
    "examples/baselines/constrain/placesphere_pilot.py",
    "examples/baselines/constrain/placesphere_obstacle.py",
    "examples/baselines/constrain/layout_geometry.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/pickcube_transport_detour.py",
)


def freeze_source_identity(repo_root=None):
    """Hash generator bytes before artifacts exist. Git HEAD is context, not identity."""
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in SOURCE_IDENTITY_FILES}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    return {
        "basis": "working-tree bytes frozen before this run writes artifacts",
        "git_head": head,
        "git_head_is_code_identity": False,
        "files": files,
        "files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }


def _write_run_config(output, seed, target, max_attempts, max_wall_seconds, per_attempt_deadline):
    config = {
        "radius_m": RADIUS_M,
        "height_is_predetermined": False,
        "historical_0_08_m_height_used": False,
        "stackcube_height_used": False,
        "height_rule": "measured direct-transport sphere and gripper occupancy plus 0.01 m geometric top cover",
        "geometric_top_cover_m": GEOMETRIC_TOP_COVER_M,
        "geometric_top_cover_is_surface_clearance": False,
        "object_xy_safety_margin_m": SAFETY_MARGIN_M,
        "sphere_radius_m": SPHERE_RADIUS_M,
        "robot_surface_clearance": "not measured from vertices; safety proof is contact-free execution through native success",
        "over_top_remains_xy_forbidden": True,
        "alternate_side_fallback": False,
        "native_success": "on_bin and static and not grasped",
        "retreat_required": True,
        "retreat_while_grasped": True,
        "terminal_phase_order": ["container_approach", "retreat", "release", "settle"],
        "retreat_dz_m": RETREAT_DZ_M,
        "retreat_dz_is_clearance_window": False,
        "endpoint_sweep": "full-robot collision vertices of grasp/lift and container approach/release/retreat/settle on direct and side paths",
        "bin_distance_window_m": BIN_DISTANCE_WINDOW_M,
        "join_criterion": "executed command and tcp XY heading step <= 45 deg; polyline corner is not the criterion",
        "join_max_heading_step_deg": MAX_JOIN_HEADING_STEP_DEG,
        "seed": seed,
        "target": target,
        "max_attempts": max_attempts,
        "max_wall_seconds": max_wall_seconds,
        "per_attempt_deadline_seconds": per_attempt_deadline,
        "waypoint_offsets_m": list(WAYPOINT_OFFSETS_M),
        "lift_heights_tried_m": list(LIFT_HEIGHTS_M),
        "measure_env_height_m": MEASURE_ENV_HEIGHT_M,
        "measure_env_height_is_scene_height": False,
        "control_mode": "pd_ee_delta_pose",
        "policy_observation_dimension": POLICY_OBS_DIM,
        "raw_observation_dimension": RAW_OBS_DIM,
    }
    (Path(output) / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _runtime_observation_contract():
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env, _policy_observation, _obs_vector

    env = make_env(config=PlaceSpherePilotConfig(max_steps=20, obstacle_radius=RADIUS_M, obstacle_height=0.08))
    try:
        _obs, info = env.reset(seed=0, options={"obstacle_enabled": True, "obstacle_center": [0.0, 0.1]})
        base = env.unwrapped
        policy = _policy_observation(base, info)
        raw = _obs_vector(base.get_obs(info=info, unflattened=True))
        if len(policy) != POLICY_OBS_DIM or len(raw) != RAW_OBS_DIM:
            raise RuntimeError(f"PlaceSphere observation contract is {len(raw)}/{len(policy)}, expected 42/39")
        return {"policy_observation_dimension": int(len(policy)), "raw_observation_dimension": int(len(raw)),
                "action_dim": 7, "control_mode": "pd_ee_delta_pose",
                "action_position_rotation_abs_max": 0.35, "action_gripper_abs_max": 1.0,
                "native_success": "on_bin and static and not grasped"}
    finally:
        env.close()


def _write_artifact_hashes(output):
    output = Path(output)
    identity = json.loads((output / "source_identity.json").read_text())
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            files[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {"basis": "artifact bytes after the run; not an input to source identity",
               "source_files_sha256": identity["files_sha256"], "files": files}
    (output / "artifact_hashes.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class TransportDetourSceneSet:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError("unsupported PlaceSphere transport-detour scene format")
        payload = {key: value for key, value in self.manifest.items() if key != "manifest_sha256"}
        encoded = json.dumps(payload, sort_keys=True, indent=2).encode()
        if hashlib.sha256(encoded).hexdigest() != self.manifest["manifest_sha256"]:
            raise ValueError("scene manifest checksum mismatch")

    def __len__(self):
        return len(self.manifest["scenes"])

    def load(self, index):
        import h5py

        row = self.manifest["scenes"][index]
        expert = row["expert"]
        if not all(expert.get(key) is True for key in (
                "native_success", "on_bin", "static", "ungrasped", "safe_success",
                "release_phase_reached", "retreat_phase_reached", "settle_phase_reached",
                "direct_transport_blocked")):
            raise ValueError("scene expert acceptance evidence is incomplete")
        if expert.get("obstacle_collision") is not False or expert.get("obstacle_crossing") is not False:
            raise ValueError("scene expert acceptance evidence is unsafe")
        clearance = expert.get("endpoint_clearance") or {}
        if clearance.get("method") != "full_robot_phase_vertices" or clearance.get("bin_distance_window_m") is not None:
            raise ValueError("scene endpoint sweep is not a phase-based full-robot sweep")
        if clearance.get("paths") != ["direct", "side"]:
            raise ValueError("scene endpoint sweep did not cover both paths")
        if (expert.get("direct_block_contact") or {}).get("phase") != "transport":
            raise ValueError("scene direct block is not a transport contact")
        if row["cylinder"]["height_source"] not in {
                "measured_direct_sphere_and_gripper_occupancy_plus_geometric_top_cover",
                "measured_direct_sphere_occupancy_plus_geometric_top_cover_gripper_top_infeasible"}:
            raise ValueError("scene height was not measured from sphere and gripper occupancy")
        if abs(float(row["cylinder"]["radius"]) - RADIUS_M) > 1e-12:
            raise ValueError("scene radius is not the fixed 0.03 m")
        with h5py.File(self.directory / self.manifest["scenes_file"], "r") as handle:
            group = handle[row["scene_id"]]
            state = _read_tree(group["state"])
            controller = _read_tree(group["controller"])
            policy = np.asarray(group["initial_policy_observation"])
            raw = np.asarray(group["initial_raw_observation"])
        if _tree_hash(state) != row["state_sha256"] or _tree_hash(controller) != row["controller_sha256"]:
            raise ValueError("scene state checksum mismatch")
        if policy.shape != (POLICY_OBS_DIM,) or raw.shape != (RAW_OBS_DIM,):
            raise ValueError("scene observation dimensions differ from the PlaceSphere contract")
        return {"metadata": row, "state": state, "controller": controller,
                "policy_observation": policy, "raw_observation": raw}

    def restore(self, index, env=None):
        import torch
        from examples.baselines.constrain.pickcube_expert import _reset_elapsed_steps, _set_state_dict_exact
        from examples.baselines.constrain.placesphere_pilot import (
            PlaceSpherePilotConfig, _obs_vector, _policy_observation, make_env,
        )
        from examples.baselines.constrain.placesphere_scenes import _controller_to_runtime, _tree_max_abs

        scene = self.load(index)
        owned = env is None
        geometry = scene["metadata"]["cylinder"]
        if owned:
            env = make_env(config=PlaceSpherePilotConfig(
                max_steps=320, obstacle_radius=geometry["radius"], obstacle_height=geometry["height"]))
        try:
            base = env.unwrapped
            if (abs(float(base.obstacle_collision_radius) - geometry["radius"]) > 1e-7
                    or abs(float(base.obstacle_collision_height) - geometry["height"]) > 1e-7):
                raise ValueError("constructed cylinder geometry differs from scene")
            env.reset(seed=scene["metadata"]["seed"], options={
                "obstacle_enabled": True, "obstacle_center": geometry["center_xy"],
                "sphere_xy": scene["metadata"]["start_xy"],
                "container_xy": scene["metadata"]["container_xy"]})
            template = base.get_state_dict()

            def convert(saved, like):
                if isinstance(like, dict):
                    return {key: convert(saved[key], like[key]) for key in like}
                value = torch.as_tensor(saved, dtype=like.dtype, device=like.device)
                if value.shape != like.shape:
                    raise ValueError("restored state shape differs from runtime")
                return value

            _set_state_dict_exact(base, convert(scene["state"], template))
            base.agent.controller.set_state(_controller_to_runtime(scene["controller"], base.agent.controller.get_state()))
            _reset_elapsed_steps(env, base)
            info = base.evaluate()
            raw = _obs_vector(base.get_obs(info=info, unflattened=True))
            policy = _policy_observation(base, info)
            norms = {
                "state": _tree_max_abs(scene["state"], base.get_state_dict()),
                "controller": _tree_max_abs(scene["controller"], base.agent.controller.get_state()),
                "raw": float(np.max(np.abs(scene["raw_observation"] - raw), initial=0.0)),
                "policy": float(np.max(np.abs(scene["policy_observation"] - policy), initial=0.0)),
            }
            if any(not np.isfinite(value) or value > OBS_ATOL for value in norms.values()):
                raise ValueError(f"restored observation/state mismatch: {norms}")
            return env, {"norms": norms, "bit_exact": all(value == 0.0 for value in norms.values())}
        except BaseException:
            if owned:
                env.close()
            raise


def audit(directory):
    catalog = TransportDetourSceneSet(directory)
    evidence = []
    for index in range(len(catalog)):
        env, restored = catalog.restore(index)
        try:
            evidence.append({"scene_id": catalog.manifest["scenes"][index]["scene_id"],
                             "roundtrip": True, "norms": restored["norms"],
                             "bit_exact": restored["bit_exact"]})
        finally:
            env.close()
    path = Path(directory) / "restore_audit.json"
    path.write_text(json.dumps({"checked": len(evidence), "all_restored": len(evidence) == len(catalog),
                                "scenes": evidence}, indent=2) + "\n")
    return len(evidence) == len(catalog) and all(row["roundtrip"] for row in evidence)


def _default_worker_argv(seed, result_dir, deadline):
    import sys
    return [sys.executable, "-m", "examples.baselines.constrain.placesphere_transport_detour",
            "--worker", "--seed", str(seed), "--result-dir", str(result_dir),
            "--deadline", str(max(1.0, float(deadline) - 8.0))]


def _worker_main(seed, result_dir, deadline):
    previous = signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, float(deadline)))
    try:
        payload = run_attempt(int(seed), result_dir)
    except AttemptDeadline:
        payload = {"accepted": False, "failure": "attempt_deadline", "counters": {}, "phase_timings": {}}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    Path(result_dir, "result.json").write_text(json.dumps(_jsonable(payload), sort_keys=True) + "\n")
    if payload.get("diagnostics", {}).get("traceback"):
        print(payload["diagnostics"]["traceback"], file=__import__("sys").stderr)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--demo-dir")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--target", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--max-wall-seconds", type=float, default=1000)
    parser.add_argument("--per-attempt-deadline", type=float, default=180)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir")
    parser.add_argument("--deadline", type=float, default=170)
    args = parser.parse_args()
    if args.worker:
        _worker_main(args.seed, args.result_dir, args.deadline)
        return
    report = generate(
        args.output_dir, seed=args.seed, target=args.target, max_attempts=args.max_attempts,
        max_wall_seconds=args.max_wall_seconds, per_attempt_deadline=args.per_attempt_deadline,
        demo_dir=args.demo_dir)
    if not report["complete"]:
        raise SystemExit(f"placesphere transport-detour batch incomplete: {report}")


def _require_fixed_radius(radius):
    if abs(float(radius) - RADIUS_M) > 1e-12:
        raise ValueError("radius stays 0.03 m until recorded infeasibility")


def _xyz(points):
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(array).all():
        raise ValueError("sweep points must be finite")
    return array


def _xyz_or_empty(points):
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return _xyz(array)


def _xy_distance(points, center):
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(points[:, :2] - center, axis=1)


def _column_occupies(points, center, radius, height):
    inside = (_xy_distance(points, center) <= float(radius)) & (points[:, 2] <= float(height))
    return bool(np.any(inside))


def _has_waypoint(record):
    phases = [name for name, _status in record["phases"]]
    return "waypoint" in phases or record["targets"].get("waypoint_xyz") is not None


def _phase_status(record, name, allowed):
    return any(phase == name and status in allowed for phase, status in record["phases"])


def _join_accepted(record):
    join = record.get("join")
    if not isinstance(join, dict) or not join.get("measured"):
        return False, "join_not_measured"
    if int(join.get("transport_steps", 0)) < MIN_JOIN_TRANSPORT_STEPS:
        return False, "join_not_measured"
    commanded = join.get("max_commanded_heading_step_deg")
    tcp = join.get("max_tcp_heading_step_deg")
    if commanded is None or tcp is None:
        return False, "join_not_measured"
    if float(commanded) > MAX_JOIN_HEADING_STEP_DEG or float(tcp) > MAX_JOIN_HEADING_STEP_DEG:
        return False, "transport_join_discontinuous"
    return True, None


def _close(left, right):
    if left is None or right is None:
        return False
    return np.allclose(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64), atol=1e-3)


def _cat(rows):
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(rows, axis=0)


def _turn_deg(start, corner, end):
    incoming = np.asarray(corner, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    outgoing = np.asarray(end, dtype=np.float64) - np.asarray(corner, dtype=np.float64)
    denom = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denom == 0.0:
        return 180.0
    cosine = float(np.dot(incoming, outgoing) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _vertex_gap(points, center, height):
    points = _xyz_or_empty(points)
    if len(points) == 0:
        return None
    occupied = points[points[:, 2] <= float(height)]
    if len(occupied) == 0:
        return None
    return float(np.linalg.norm(occupied[:, :2] - np.asarray(center), axis=1).min() - RADIUS_M)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_tree(group, tree):
    import torch
    for key, value in tree.items():
        if isinstance(value, dict):
            _write_tree(group.create_group(str(key)), value)
        else:
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            group.create_dataset(str(key), data=array)


def _read_tree(group):
    import h5py
    return {key: (_read_tree(value) if isinstance(value, h5py.Group) else np.asarray(value[()]))
            for key, value in group.items()}


def _stack_rows(rows):
    import torch
    if isinstance(rows[0], dict):
        keys = rows[0].keys()
        if any(row.keys() != keys for row in rows):
            raise ValueError("state schema changed during the episode")
        return {key: _stack_rows([row[key] for row in rows]) for key in keys}
    arrays = []
    for value in rows:
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if array.ndim == 0:
            array = array.reshape(1)
        arrays.append(array)
    return np.concatenate(arrays, axis=0)


def _tree_hash(tree):
    import torch
    digest = hashlib.sha256()

    def visit(value, path=""):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
            return
        array = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
        digest.update(path.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    visit(tree)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
