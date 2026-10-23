"""PickCube mid-transport side-detour scenes.

Cylinder radius stays 0.03 m. Height is selected from measured direct-transport
occupancy and same-side clearance. It is not a StackCube height and is not an
argument of the selector.
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


RADIUS_M = 0.03
GEOMETRIC_TOP_COVER_M = 0.01
MAX_DETOUR_TURN_DEG = 150.0


@dataclass(frozen=True)
class HeightDecision:
    accepted: bool
    height_m: float | None
    reason: str | None


@dataclass(frozen=True)
class ClearanceReport:
    geometric_vertex_clearance_m: float
    geometric_is_surface_clearance: bool
    object_xy_margin_m: float
    robot_surface_clearance_m: float | None
    safety_proof: str


def select_cylinder_height(direct_points, side_points, end_points, *, center_xy, radius):
    """Choose height from measured envelopes. Callers cannot pass a scene height."""
    _require_fixed_radius(radius)
    direct = _xyz(direct_points)
    side = _xyz(side_points)
    end = _xyz(end_points)
    if len(side) == 0 or len(end) == 0:
        return HeightDecision(False, None, "sweep_not_recorded")
    center = np.asarray(center_xy, dtype=np.float64).reshape(2)
    inside = _xy_distance(direct, center) <= radius
    if not np.any(inside):
        return HeightDecision(False, None, "direct_transport_misses_cylinder")
    height = float(np.max(direct[inside, 2]) + GEOMETRIC_TOP_COVER_M)
    if np.any((_xy_distance(side, center) <= radius) & (side[:, 2] <= height)):
        return HeightDecision(False, None, "side_detour_occupies_cylinder_column")
    if np.any((_xy_distance(end, center) <= radius) & (end[:, 2] <= height)):
        return HeightDecision(False, None, "end_region_sweep_occupies_cylinder_column")
    return HeightDecision(True, height, None)


def classify_clearances(
    *,
    vertex_xy_distance_m,
    radius_m,
    cube_center_distance_m,
    forbidden_radius_m,
    contact_separation_m,
    contact_free,
):
    """Keep the vertex approximation, object XY margin, and surface gap distinct."""
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
        object_xy_margin_m=float(cube_center_distance_m) - float(forbidden_radius_m),
        robot_surface_clearance_m=None if contact_separation_m is None else float(contact_separation_m),
        safety_proof=proof,
    )


def midsegment_candidates(start, goal, base_xy, *, radius):
    """Finite 40%–60% grid, offset away from the fixed base side."""
    _require_fixed_radius(radius)
    start = np.asarray(start, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    base = np.asarray(base_xy, dtype=np.float64).reshape(2)
    direction = goal - start
    distance = float(np.linalg.norm(direction))
    if distance == 0.0:
        raise ValueError("degenerate transport segment")
    along = direction / distance
    base_side = float(direction[0] * (base[1] - start[1]) - direction[1] * (base[0] - start[0]))
    if base_side == 0.0:
        raise ValueError("base_on_transport_line")
    base_perpendicular = np.array([-along[1], along[0]]) * math.copysign(1.0, base_side)
    anti = -base_perpendicular
    rows = []
    for t in np.linspace(0.40, 0.60, 5):
        for offset in np.linspace(0.0, 0.04, 5):
            rows.append(start + t * (goal - start) + anti * float(offset))
    return np.asarray(rows, dtype=np.float64)


def generate(output_dir, *, seed=1, target=1, max_attempts=1, max_wall_seconds=1.0,
             per_attempt_deadline=1.0, worker_argv_fn=None, demo_dir=None):
    """Run a finite attempt loop. Every invocation counts, including deadlines."""
    if (target < 1 or max_attempts < 1 or per_attempt_deadline <= 0
            or not np.isfinite(per_attempt_deadline)
            or max_wall_seconds <= 0 or not np.isfinite(max_wall_seconds)):
        raise ValueError("target, attempt budget, deadline, and wall budget must be positive and finite")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_run_config(output, seed, target, max_attempts, max_wall_seconds, per_attempt_deadline)
    if worker_argv_fn is None:
        from examples.baselines.constrain.pickcube_base_dp import verify_runtime_observation_contract
        contract = verify_runtime_observation_contract()
        contract["action_dim"] = 7
        contract["action_bounds"] = [-1.0, 1.0]
        contract["control_mode"] = "pd_ee_delta_pose"
        (output / "runtime_observation_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
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
        if accepted_row and not (result_dir / "scene.h5").exists():
            accepted_row = False
            reason = "missing_scene"
        if accepted_row:
            try:
                accepted_rows.append(_ingest_scene(output, result_dir, len(accepted_rows), payload, demo_dir))
            except Exception as exc:
                accepted_row = False
                reason = f"ingest_failed:{type(exc).__name__}"
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
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_bounded_process(argv, deadline_s, stderr_path=None):
    """Run argv and kill its process group if it exceeds the deadline."""
    stderr = open(stderr_path, "w") if stderr_path is not None else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(argv, start_new_session=True, stdout=subprocess.DEVNULL, stderr=stderr)
        try:
            returncode = proc.wait(timeout=deadline_s)
            return {"timed_out": False, "returncode": returncode}
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            return {"timed_out": True, "returncode": proc.returncode}
    finally:
        if stderr is not subprocess.DEVNULL:
            stderr.close()


def _kill_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _default_worker_argv(seed, result_dir, deadline):
    import sys
    return [sys.executable, "-m", "examples.baselines.constrain.pickcube_transport_detour",
            "--worker", "--seed", str(seed), "--result-dir", str(result_dir),
            "--deadline", str(max(1.0, float(deadline) - 8.0))]


def shared_transport_fork(direct, detour):
    """Direct and detour share lift/goal; only the detour inserts a side waypoint."""
    if _has_waypoint(direct):
        return False, "direct_has_waypoint"
    if not _phase_reached(detour, "waypoint") or detour["targets"]["waypoint_xyz"] is None:
        return False, "detour_missing_waypoint"
    if not _phase_reached(direct, "lift") or not _phase_reached(detour, "lift"):
        return False, "lift_not_shared"
    direct_targets = direct["targets"]
    detour_targets = detour["targets"]
    if not _close(direct_targets["lift_xyz"], detour_targets["lift_xyz"]):
        return False, "lift_target_mismatch"
    if not _close(direct_targets["goal_xyz"], detour_targets["goal_xyz"]):
        return False, "goal_target_mismatch"
    waypoint = np.asarray(detour_targets["waypoint_xyz"], dtype=np.float64)
    lift = np.asarray(detour_targets["lift_xyz"], dtype=np.float64)
    if abs(float(waypoint[2]) - float(lift[2])) > 1e-3:
        return False, "waypoint_z_discontinuous"
    goal = np.asarray(detour_targets["goal_xyz"], dtype=np.float64)
    if _turn_deg(lift[:2], waypoint[:2], goal[:2]) > MAX_DETOUR_TURN_DEG:
        return False, "detour_not_side_continuous"
    return True, None


def _require_fixed_radius(radius):
    if abs(float(radius) - RADIUS_M) > 1e-12:
        raise ValueError("radius stays 0.03 m until recorded infeasibility")


def _xyz(points):
    array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(array).all():
        raise ValueError("sweep points must be finite")
    return array


def _xy_distance(points, center):
    if len(points) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(points[:, :2] - center, axis=1)


def _has_waypoint(record):
    phases = [name for name, _status in record["phases"]]
    return "waypoint" in phases or record["targets"]["waypoint_xyz"] is not None


def _phase_reached(record, name):
    return any(phase == name and status == "reached" for phase, status in record["phases"])


def _close(left, right):
    return np.allclose(np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64), atol=1e-3)


def _turn_deg(start, corner, end):
    incoming = np.asarray(corner, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    outgoing = np.asarray(end, dtype=np.float64) - np.asarray(corner, dtype=np.float64)
    denom = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denom == 0.0:
        return 180.0
    cosine = float(np.dot(incoming, outgoing) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


FORMAT = "pickcube-transport-detour-scenes-v1"
TRANSPORT_BODIES = frozenset({
    "cube", "panda_hand", "panda_link6", "panda_link7",
    "panda_leftfinger", "panda_rightfinger",
    "panda_leftfinger_pad", "panda_rightfinger_pad",
})
START_PHASES = frozenset({"grasp_approach", "grasp_pose", "gripper_close", "lift"})
GOAL_ENTRY_M = 0.10
TABLE_BOUNDS = (-0.25, -0.25, 0.25, 0.35)
WAYPOINT_OFFSETS_M = (0.20, 0.24, 0.28, 0.32, 0.16)
LIFT_HEIGHTS_M = (0.12, 0.10, 0.14)
MEASURE_ENV_HEIGHT_M = 0.05
LAYOUT_TRIES = 8
EXECUTION_CANDIDATES = 3


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
    from examples.baselines.constrain.layout_geometry import LayoutGeometryConfig, validate_layout
    from examples.baselines.constrain.pickcube_expert import (
        PickCubeExpertConfig, make_pickcube_env, run_pickcube_expert,
    )

    counters = {"geometry_tries": 0, "direct_executions": 0, "side_executions": 0,
                "block_executions": 0, "safe_executions": 0, "centers_tested": 0,
                "waypoints_tested": 0, "lifts_tested": 0}
    timings = {"measure": 0.0, "block": 0.0, "safe": 0.0}
    notes = []

    def fail(reason, **extra):
        if len(notes) < 16:
            notes.append({"reason": reason, **extra})
        return {"accepted": False, "failure": reason, "counters": counters,
                "phase_timings": timings, "diagnostics": {"rejections": notes}}

    rng = np.random.default_rng(seed)
    measure = make_pickcube_env(obstacle_radius=RADIUS_M, obstacle_height=MEASURE_ENV_HEIGHT_M)
    envs.append(measure)
    base_xy = measure.unwrapped.agent.robot.pose.p[0, :2].detach().cpu().numpy()
    layout_config = LayoutGeometryConfig(
        table_bounds=TABLE_BOUNDS, min_transport_distance=0.14, min_base_line_distance=0.06,
        waypoint_offset_range=(0.16, 0.32), waypoint_lead_fraction=0.25, max_attempts=LAYOUT_TRIES,
    )
    start = goal = None
    for _ in range(LAYOUT_TRIES):
        counters["geometry_tries"] += 1
        candidate_start = np.array([rng.uniform(-0.18, -0.02), rng.uniform(-0.06, 0.12)])
        candidate_goal = np.array([rng.uniform(0.02, 0.18), rng.uniform(-0.06, 0.12)])
        if validate_layout(candidate_start, candidate_goal, base_xy, layout_config).valid:
            start, goal = candidate_start, candidate_goal
            break
    if start is None:
        return fail("geometry_retry_exhausted")

    last_reason = "no_feasible_height_at_radius_0.03"
    for lift in LIFT_HEIGHTS_M:
        counters["lifts_tested"] += 1
        config = PickCubeExpertConfig(seed=seed, max_steps=360, lift_height=float(lift))
        started = time.monotonic()
        direct = run_pickcube_expert(
            measure, cube_xy=start, goal_xy=goal, detour=False, obstacle_enabled=False,
            config=config, record_body_sweep=True, layout_config=layout_config,
        )
        counters["direct_executions"] += 1
        timings["measure"] += time.monotonic() - started
        if not direct["safety"]["native_success"] or not direct["safety"]["held_at_success"]:
            last_reason = "direct_template_unsuccessful"
            notes.append({"reason": last_reason, "lift_height_m": float(lift)})
            continue
        snapshot = direct["initial_snapshot"]
        saw_geometry = False
        direct_transport, _, direct_end = _sweep_regions(
            direct, start, goal, transport=True, transport_bodies={"cube"}
        )
        if len(direct_transport) == 0 or len(direct_end) == 0:
            last_reason = "sweep_not_recorded"
            continue
        for offset in WAYPOINT_OFFSETS_M:
            waypoint = _explicit_waypoint(start, goal, base_xy, offset)
            counters["waypoints_tested"] += 1
            if waypoint is None:
                last_reason = "waypoint_out_of_bounds"
                continue
            started = time.monotonic()
            side = run_pickcube_expert(
                measure, cube_xy=start, goal_xy=goal, detour=True, waypoint_xy=waypoint,
                obstacle_enabled=False, initial_snapshot=snapshot, config=config,
                record_body_sweep=True, layout_config=layout_config,
            )
            counters["side_executions"] += 1
            timings["measure"] += time.monotonic() - started
            if not side["safety"]["native_success"] or not side["safety"]["held_at_success"]:
                last_reason = "side_template_unsuccessful"
                continue
            fork_ok, fork_reason = shared_transport_fork(
                _fork_record(direct), _fork_record(side)
            )
            if not fork_ok:
                last_reason = fork_reason
                continue
            _, side_all, side_end = _sweep_regions(side, start, goal, transport=False)
            end_points = side_end
            ranked = []
            for center in midsegment_candidates(start, goal, base_xy, radius=RADIUS_M):
                counters["centers_tested"] += 1
                decision = select_cylinder_height(
                    direct_transport, side_all, end_points, center_xy=center, radius=RADIUS_M
                )
                if not decision.accepted:
                    last_reason = decision.reason
                    continue
                gap = float(np.linalg.norm(side_all[:, :2] - center, axis=1).min())
                ranked.append((gap, decision, center))
            ranked.sort(key=lambda item: item[0], reverse=True)
            saw_geometry = saw_geometry or bool(ranked)
            for gap, decision, center in ranked[:EXECUTION_CANDIDATES]:
                blocked, block_reason, block = _execute_block(
                    envs, snapshot, start, goal, center, decision.height_m, config, layout_config, counters, timings
                )
                if not blocked:
                    last_reason = block_reason
                    continue
                saved, safe_reason, safe = _execute_safe(
                    envs, snapshot, start, goal, waypoint, center, decision.height_m, config,
                    layout_config, counters, timings,
                )
                if not saved:
                    last_reason = safe_reason
                    continue
                clearance = _clearance_from_run(safe, center, decision.height_m)
                metadata = _scene_metadata(
                    seed, start, goal, base_xy, center, decision, clearance, direct, side, safe, block, lift, offset, gap
                )
                _write_attempt_scene(result_dir, safe, block, metadata)
                return {"accepted": True, "failure": None, "counters": counters, "phase_timings": timings,
                        "metadata": metadata, "diagnostics": {"rejections": notes, "selected_gap_m": gap}}
        if saw_geometry:
            break
    return fail(last_reason)


def _execute_block(envs, snapshot, start, goal, center, height, config, layout_config, counters, timings):
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env, run_pickcube_expert

    env = make_pickcube_env(obstacle_radius=RADIUS_M, obstacle_height=float(height))
    envs.append(env)
    started = time.monotonic()
    try:
        env.reset(seed=config.seed, options={
            "cube_xy": np.asarray(start), "goal_pos": [*np.asarray(goal), 0.12],
            "obstacle_enabled": True, "obstacle_center": np.asarray(center),
        })
        result = run_pickcube_expert(
            env, cube_xy=start, goal_xy=goal, detour=False, obstacle_enabled=True,
            obstacle_center=center, initial_snapshot=snapshot, config=config,
            stop_on_contact=True, layout_config=layout_config,
        )
    finally:
        env.close()
        envs.remove(env)
    counters["block_executions"] += 1
    timings["block"] += time.monotonic() - started
    contact = result["trace"].get("first_obstacle_contact")
    if not result["safety"]["obstacle_collision"] or contact is None:
        return False, "direct_transport_not_blocked", result
    if contact["phase"] in START_PHASES or contact["phase"] == "settle":
        return False, "direct_block_contact_in_end_region", result
    cube = np.asarray(contact["cube_xyz"][:2], dtype=np.float64)
    if float(np.linalg.norm(cube - np.asarray(goal))) < GOAL_ENTRY_M:
        return False, "direct_block_contact_in_end_region", result
    if contact["phase"] != "goal":
        return False, "direct_block_contact_outside_transport", result
    return True, None, result


def _execute_safe(envs, snapshot, start, goal, waypoint, center, height, config, layout_config, counters, timings):
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env, run_pickcube_expert

    env = make_pickcube_env(obstacle_radius=RADIUS_M, obstacle_height=float(height))
    envs.append(env)
    started = time.monotonic()
    try:
        env.reset(seed=config.seed, options={
            "cube_xy": np.asarray(start), "goal_pos": [*np.asarray(goal), 0.12],
            "obstacle_enabled": True, "obstacle_center": np.asarray(center),
        })
        result = run_pickcube_expert(
            env, cube_xy=start, goal_xy=goal, detour=True, waypoint_xy=waypoint,
            obstacle_enabled=True, obstacle_center=center, initial_snapshot=snapshot,
            config=config, record_body_sweep=True, layout_config=layout_config,
        )
        result["forbidden_radius_m"] = float(env.unwrapped._forbidden_radius)
    finally:
        env.close()
        envs.remove(env)
    counters["safe_executions"] += 1
    timings["safe"] += time.monotonic() - started
    safety = result["safety"]
    if not safety["native_success"] or not safety["held_at_success"]:
        return False, "obstacle_native_unsuccessful", result
    if safety["obstacle_collision"] or safety["obstacle_crossing"] or not safety["safe_success"]:
        return False, "obstacle_contact_or_crossing", result
    if not _action_observation_contract(result, center):
        return False, "observation_action_contract", result
    return True, None, result


def _action_observation_contract(result, center):
    actions = np.asarray(result["actions"])
    obs = np.stack([np.asarray(item).reshape(-1) for item in result["observations"]])
    if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
        return False
    if np.any(np.abs(actions) > 1.0 + 1e-5):
        return False
    if obs.ndim != 2 or obs.shape[1] != 45 or obs.shape[0] != len(actions) + 1:
        return False
    if abs(float(obs[0, -1]) - RADIUS_M) > 1e-5:
        return False
    return float(np.linalg.norm(obs[0, -3:-1] - np.asarray(center, dtype=np.float64))) <= 1e-4


def _sweep_regions(result, start, goal, *, transport, transport_bodies=TRANSPORT_BODIES):
    transport_rows, all_rows, end_rows = [], [], []
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    for step in result["trace"].get("body_sweep", []):
        phase = step["phase"]
        bodies = step["bodies"]
        cube = bodies.get("cube")
        cube_xy = None if cube is None else np.asarray(cube, dtype=np.float64).mean(axis=0)[:2]
        near_goal = cube_xy is not None and float(np.linalg.norm(cube_xy - goal)) <= GOAL_ENTRY_M
        for name, points in bodies.items():
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            all_rows.append(points)
            if phase in START_PHASES or phase == "settle" or (phase == "goal" and near_goal):
                end_rows.append(points)
            if transport and phase == "goal" and name in transport_bodies:
                window = (_along(points[:, :2], start, goal) >= 0.40) & (_along(points[:, :2], start, goal) <= 0.60)
                if np.any(window):
                    transport_rows.append(points[window])
    return _cat(transport_rows), _cat(all_rows), _cat(end_rows)


def _along(points, start, goal):
    direction = goal - start
    length_sq = float(direction @ direction)
    if length_sq == 0.0:
        return np.zeros(len(points))
    return ((points - start) @ direction) / length_sq


def _cat(rows):
    if not rows:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(rows, axis=0)


def _explicit_waypoint(start, goal, base, offset):
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
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
    xmin, ymin, xmax, ymax = TABLE_BOUNDS
    if not (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax):
        return None
    return point


def _fork_record(result):
    return {"phases": list(result["trace"]["phase_status"]), "targets": result["trace"]["shared_targets"]}


def _clearance_from_run(result, center, height):
    points = _sweep_regions(result, result["layout"]["start_xy"], result["layout"]["goal_xy"], transport=False)[1]
    occupied = points[points[:, 2] <= float(height)] if len(points) else points
    if len(occupied) == 0:
        vertex_distance = None
    else:
        vertex_distance = float(np.linalg.norm(occupied[:, :2] - np.asarray(center), axis=1).min())
    forbidden = _forbidden_radius(result)
    cube_distance = _grasped_cube_distance(result, center)
    report = classify_clearances(
        vertex_xy_distance_m=0.0 if vertex_distance is None else vertex_distance,
        radius_m=RADIUS_M,
        cube_center_distance_m=0.0 if cube_distance is None else cube_distance,
        forbidden_radius_m=forbidden,
        contact_separation_m=None,
        contact_free=True,
    )
    payload = report.__dict__.copy()
    payload["vertex_xy_distance_m"] = vertex_distance
    payload["geometric_vertex_clearance_m"] = None if vertex_distance is None else vertex_distance - RADIUS_M
    payload["cube_center_distance_m"] = cube_distance
    payload["object_xy_margin_m"] = None if cube_distance is None else cube_distance - forbidden
    payload["forbidden_radius_m"] = forbidden
    payload["surface_note"] = "no PhysX separation was measured; contact-free execution is the safety proof, not a surface gap"
    return payload


def _forbidden_radius(result):
    if result.get("forbidden_radius_m") is not None:
        return float(result["forbidden_radius_m"])
    state = result["states"][0]["pickcube_obstacle"]
    radius = float(np.asarray(state).reshape(-1)[2])
    return radius + float(np.sqrt(2.0) * 0.02) + 0.005


def _grasped_cube_distance(result, center):
    distances = []
    infos = result["infos"][1:]
    for xy, info in zip(result["trace"]["cube_xy"], infos):
        grasped = info.get("is_grasped") if isinstance(info, dict) else None
        if grasped is None:
            continue
        if hasattr(grasped, "detach"):
            grasped = bool(grasped.reshape(-1)[0].item())
        if grasped:
            distances.append(float(np.linalg.norm(np.asarray(xy, dtype=np.float64) - np.asarray(center))))
    return None if not distances else min(distances)


def _scene_metadata(seed, start, goal, base_xy, center, decision, clearance, direct, side, safe, block, lift, offset, gap):
    contact = block["trace"]["first_obstacle_contact"]
    return {
        "seed": int(seed),
        "start_xy": np.asarray(start, dtype=np.float64).tolist(),
        "goal_xy": np.asarray(goal, dtype=np.float64).tolist(),
        "base_xy": np.asarray(base_xy, dtype=np.float64).tolist(),
        "cylinder": {
            "center_xy": np.asarray(center, dtype=np.float64).tolist(),
            "radius": RADIUS_M,
            "height": float(decision.height_m),
            "height_source": "measured_direct_cube_occupancy_plus_geometric_top_cover",
            "geometric_top_cover_m": GEOMETRIC_TOP_COVER_M,
            "direct_occupancy_max_z_m": float(decision.height_m) - GEOMETRIC_TOP_COVER_M,
            "measure_env_height_m": MEASURE_ENV_HEIGHT_M,
            "measure_env_height_is_scene_height": False,
        },
        "expert": {
            "native_success": True,
            "held_at_success": True,
            "obstacle_collision": False,
            "obstacle_crossing": False,
            "safe_success": True,
            "steps": int(len(safe["actions"])),
            "waypoint_xy": side["trace"]["waypoint_xy"],
            "lift_height_m": float(lift),
            "waypoint_offset_m": float(offset),
            "direct_transport_blocked": True,
            "direct_block_contact": contact,
            "clearance": clearance,
            "side_column_gap_m": float(gap),
            "fork_targets": side["trace"]["shared_targets"],
        },
    }


def _write_attempt_scene(result_dir, safe, block, metadata):
    import h5py

    from examples.baselines.constrain.pickcube_scenes import _write_tree

    path = Path(result_dir) / "scene.h5"
    state = safe["states"][0]
    obs = np.asarray(safe["observations"][0]).reshape(-1)
    actions = np.asarray(safe["actions"], dtype=np.float32)
    controller = _jsonable(safe["initial_snapshot"][1])
    with h5py.File(path, "w") as handle:
        group = handle.create_group("scene")
        _write_tree(group.create_group("state"), state)
        group.create_dataset("initial_observation", data=obs)
        group.create_dataset("expert_actions", data=actions)
        group.create_dataset("observations", data=np.stack([
            np.asarray(item).reshape(-1) for item in safe["observations"]
        ]).astype(np.float32))
        group.create_dataset("direct_block_actions", data=np.asarray(block["actions"], dtype=np.float32))
        group.attrs["controller_json"] = json.dumps(controller, sort_keys=True)
        group.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)


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
    return value


def _ingest_scene(output, result_dir, index, payload, demo_dir):
    import h5py

    from examples.baselines.constrain.pickcube_scenes import _read_tree, _tree_hash, _write_tree

    scene_id = f"pickcube-{index:03d}"
    with h5py.File(result_dir / "scene.h5", "r") as source:
        state = _read_tree(source["scene"]["state"])
        obs = np.asarray(source["scene"]["initial_observation"])
        actions = np.asarray(source["scene"]["expert_actions"])
        observations = np.asarray(source["scene"]["observations"])
        block_actions = np.asarray(source["scene"]["direct_block_actions"])
        controller = json.loads(source["scene"].attrs["controller_json"])
        metadata = json.loads(source["scene"].attrs["metadata_json"])
    metadata = dict(metadata)
    metadata["scene_id"] = scene_id
    metadata["state_sha256"] = _tree_hash(state)
    metadata["observation_sha256"] = hashlib.sha256(np.asarray(obs).tobytes()).hexdigest()
    metadata["controller_sha256"] = _tree_hash(controller)
    metadata["observation_shape"] = list(np.asarray(obs).shape)
    metadata["observation_dtype"] = str(np.asarray(obs).dtype)
    if not metadata["expert"].get("direct_transport_blocked") or not metadata["expert"].get("held_at_success"):
        raise ValueError("scene is missing block or held-success evidence")
    aggregate = Path(output) / "scenes.h5"
    with h5py.File(aggregate, "a") as handle:
        if scene_id in handle:
            del handle[scene_id]
        group = handle.create_group(scene_id)
        _write_tree(group.create_group("state"), state)
        group.create_dataset("initial_observation", data=np.asarray(obs))
        group.create_dataset("expert_actions", data=actions)
        group.create_dataset("observations", data=observations)
        group.create_dataset("direct_block_actions", data=block_actions)
        group.attrs["controller_json"] = json.dumps(controller, sort_keys=True)
    if demo_dir is not None:
        _append_demo(Path(demo_dir), scene_id, metadata, actions, observations, state)
    return metadata


def _append_demo(demo_dir, scene_id, metadata, actions, observations, state):
    import h5py

    from examples.baselines.constrain.pickcube_scenes import _write_tree

    demo_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(demo_dir / "trajectory.h5", "a") as handle:
        group = handle.create_group(scene_id)
        group.attrs["kind"] = "obstacle"
        group.create_dataset("actions", data=np.asarray(actions))
        group.create_dataset("obs", data=np.asarray(observations))
        _write_tree(group.create_group("env_states"), {"initial": state})
    manifest = demo_dir / "episodes.jsonl"
    with manifest.open("a") as stream:
        stream.write(json.dumps({"scene_id": scene_id, "metadata": metadata}, sort_keys=True) + "\n")


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
            "per scene, max z of direct-transport cube samples inside the 0.03 m cylinder "
            "column, plus 0.01 m geometric top cover; not a StackCube height"
        ),
        "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest() if (output / "config.json").exists() else None,
        "scenes_file_sha256": hashlib.sha256(scenes.read_bytes()).hexdigest() if scenes.exists() else None,
    }
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode()
    payload["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_run_config(output, seed, target, max_attempts, max_wall_seconds, per_attempt_deadline):
    config = {
        "radius_m": RADIUS_M,
        "height_is_predetermined": False,
        "stackcube_height_used": False,
        "height_rule": "measured direct-transport cube occupancy plus 0.01 m geometric top cover",
        "geometric_top_cover_m": GEOMETRIC_TOP_COVER_M,
        "geometric_top_cover_is_surface_clearance": False,
        "object_xy_safety_margin_m": 0.005,
        "robot_surface_clearance": "not measured from vertices; safety proof is contact-free execution",
        "over_top_remains_xy_forbidden": True,
        "native_success": "held cube at the goal; no release added",
        "seed": seed,
        "target": target,
        "max_attempts": max_attempts,
        "max_wall_seconds": max_wall_seconds,
        "per_attempt_deadline_seconds": per_attempt_deadline,
        "waypoint_offsets_m": list(WAYPOINT_OFFSETS_M),
        "lift_heights_tried_m": list(LIFT_HEIGHTS_M),
        "measure_env_height_m": MEASURE_ENV_HEIGHT_M,
        "measure_env_height_is_scene_height": False,
    }
    (Path(output) / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


class TransportDetourSceneSet:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError("unsupported transport-detour scene format")
        payload = {key: value for key, value in self.manifest.items() if key != "manifest_sha256"}
        encoded = json.dumps(payload, sort_keys=True, indent=2).encode()
        if hashlib.sha256(encoded).hexdigest() != self.manifest["manifest_sha256"]:
            raise ValueError("scene manifest checksum mismatch")

    def __len__(self):
        return len(self.manifest["scenes"])

    def load(self, index):
        from examples.baselines.constrain.pickcube_scenes import _read_tree, _tree_hash

        row = self.manifest["scenes"][index]
        expert = row["expert"]
        if not expert.get("direct_transport_blocked") or not expert.get("held_at_success"):
            raise ValueError("scene expert acceptance evidence is unsafe or incomplete")
        if expert.get("obstacle_collision") is not False or expert.get("obstacle_crossing") is not False:
            raise ValueError("scene expert acceptance evidence is unsafe or incomplete")
        if row["cylinder"]["height_source"] != "measured_direct_cube_occupancy_plus_geometric_top_cover":
            raise ValueError("scene height was not measured from direct transport")
        if abs(float(row["cylinder"]["radius"]) - RADIUS_M) > 1e-12:
            raise ValueError("scene radius is not the fixed 0.03 m")
        with h5py_open(self.directory / self.manifest["scenes_file"]) as handle:
            group = handle[row["scene_id"]]
            state = _read_tree(group["state"])
            obs = np.asarray(group["initial_observation"])
            controller = json.loads(group.attrs["controller_json"])
        if _tree_hash(state) != row["state_sha256"]:
            raise ValueError("scene state checksum mismatch")
        if hashlib.sha256(obs.tobytes()).hexdigest() != row["observation_sha256"]:
            raise ValueError("scene observation checksum mismatch")
        flags = np.asarray(state["pickcube_obstacle"]).reshape(-1)
        if flags[3] != 1 or np.any(flags[4:7] != 0):
            raise ValueError("unsafe obstacle state flags in scene")
        return {"metadata": row, "state": state, "observation": obs, "controller": controller}

    def restore(self, index, env=None):
        from examples.baselines.constrain.pickcube_expert import _set_state_dict_exact, make_pickcube_env
        import torch

        scene = self.load(index)
        owned = env is None
        geometry = scene["metadata"]["cylinder"]
        if owned:
            env = make_pickcube_env(obstacle_radius=geometry["radius"], obstacle_height=geometry["height"])
        try:
            base = env.unwrapped
            if (abs(float(base.obstacle_collision_radius) - geometry["radius"]) > 1e-7
                    or abs(float(base.obstacle_height) - geometry["height"]) > 1e-7):
                raise ValueError("constructed cylinder geometry differs from scene")
            env.reset(seed=scene["metadata"]["seed"], options={
                "obstacle_enabled": True, "obstacle_center": geometry["center_xy"],
                "cube_xy": scene["metadata"]["start_xy"],
                "goal_pos": [*scene["metadata"]["goal_xy"], 0.12],
            })

            def convert(value, like):
                if isinstance(like, dict):
                    return {key: convert(value[key], like[key]) for key in like}
                return torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(like.shape)

            _set_state_dict_exact(base, convert(scene["state"], base.get_state_dict()))
            base.agent.controller.set_state(_restore_controller(scene["controller"], base.agent.controller.get_state()))
            from examples.baselines.constrain.pickcube_expert import _reset_elapsed_steps
            _reset_elapsed_steps(env, base)
            actual = np.asarray(base.get_obs().detach().cpu()).reshape(-1)
            if not np.array_equal(actual, scene["observation"]):
                raise ValueError("restored initial observation mismatch")
            return env, base.get_obs()
        except BaseException:
            if owned:
                env.close()
            raise


def _restore_controller(value, template):
    import torch
    if isinstance(template, dict):
        return {key: _restore_controller(value[key], item) for key, item in template.items()}
    if torch.is_tensor(template):
        return torch.as_tensor(value, dtype=template.dtype, device=template.device).reshape(template.shape)
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype).reshape(template.shape)
    return type(template)(value)


def h5py_open(path):
    import h5py
    return h5py.File(path, "r")


def audit(directory):
    catalog = TransportDetourSceneSet(directory)
    evidence = []
    for index in range(len(catalog)):
        env, _obs = catalog.restore(index)
        try:
            evidence.append({"scene_id": catalog.manifest["scenes"][index]["scene_id"], "roundtrip": True})
        finally:
            env.close()
    path = Path(directory) / "restore_audit.json"
    path.write_text(json.dumps({"checked": len(evidence), "all_restored": len(evidence) == len(catalog),
                                "scenes": evidence}, indent=2) + "\n")
    return len(evidence) == len(catalog)


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
    Path(result_dir, "result.json").write_text(json.dumps(payload, sort_keys=True) + "\n")
    if payload.get("diagnostics", {}).get("traceback"):
        print(payload["diagnostics"]["traceback"], file=__import__("sys").stderr)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--demo-dir")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--target", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=24)
    parser.add_argument("--max-wall-seconds", type=float, default=1800)
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
        demo_dir=args.demo_dir,
    )
    if not report["complete"]:
        raise SystemExit(f"transport-detour batch incomplete: {report}")


if __name__ == "__main__":
    main()
