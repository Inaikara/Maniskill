"""Tests for action-first constructive StackCubeObstacle collection.

For each template a
successful no-obstacle episode is executed FIRST and its actual robot/cube
swept trace recorded; only then are blocking cylinders sampled around that
fixed action and re-executed from the identical initial state with the same
plan. ``reuse_count`` means that many DIFFERENT cylinder placements for the
same path, seed and initial robot/cube state.

Seams under test (no mirrored implementations, no sim required unless noted):
- :class:`~examples.baselines.constrain.collect_constructive.WaypointTracker`
  (feedback waypoint control contract),
- :func:`choose_waypoint` / :func:`sample_cube_pair` / :func:`sample_cylinder`
  (pure geometry: waypoint independent of any obstacle, cylinder in the direct
  AB blocking strip but outside the actual carried footprint),
- :func:`run_episode` carry-z / orientation-drift reporting via a stub env,
- :func:`collect` with an injected episode recorder (ordering, reuse caps,
  budgets, initial-state H5 equality, artifact/metadata structure), plus CLI
  argument plumbing.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import gymnasium as gym
from transforms3d.quaternions import qmult, quat2axangle, axangle2quat

from examples.baselines.constrain import collect_constructive as cc
from examples.baselines.constrain.stackcube_obstacle import ENV_ID


def make_cubes(cube_a, cube_b, yaw_a=0.0, yaw_b=0.0):
    return dict(
        cubeA_xy=np.asarray(cube_a, dtype=np.float64),
        cubeB_xy=np.asarray(cube_b, dtype=np.float64),
        cubeA_yaw=float(yaw_a),
        cubeB_yaw=float(yaw_b),
    )


# Feasible template fixture: separation 0.30 along y so the cylinder window
# between the measured palm-horn reach (0.10 m, see README) is non-empty.
FEASIBLE_CUBES = make_cubes((0.0, 0.02), (0.0, 0.32))
CUBE_HALF_DIAG = float(np.hypot(0.02, 0.02))


def synthetic_footprint(cube_pair, waypoint_xy, step=0.01):
    """Footprint fixture mimicking the measured panda sweep (README): the
    carried-cube corners along the TCP polyline plus palm-horn bars at both
    endpoints (bars carry no cube)."""
    a = np.asarray(cube_pair["cubeA_xy"], dtype=np.float64)
    b = np.asarray(cube_pair["cubeB_xy"], dtype=np.float64)
    w = np.asarray(waypoint_xy, dtype=np.float64)
    path = []

    def add_segment(p, q):
        n = max(2, int(np.linalg.norm(q - p) / step))
        for t in np.linspace(0.0, 1.0, n):
            path.append(p + t * (q - p))

    add_segment(a, w)
    add_segment(w, b)
    points = []
    for p in path:
        for dx in (-CUBE_HALF_DIAG, CUBE_HALF_DIAG):
            for dy in (-CUBE_HALF_DIAG, CUBE_HALF_DIAG):
                points.append(p + np.array([dx, dy]))
    for endpoint in (a, b):
        for dy in np.linspace(-0.10, 0.10, 21):
            for dx in (-0.02, 0.02):
                points.append(endpoint + np.array([dx, dy]))
    return np.unique(np.round(np.array(points), 5), axis=0)


# --------------------------------------------------------------------------- #
# WaypointTracker (feedback control seam)
# --------------------------------------------------------------------------- #


def test_tracker_commands_clipped_deltas_with_zero_rotation():
    tracker = cc.WaypointTracker(np.array([0.1, -0.05, 0.2]), cc.GRIPPER_CLOSE)
    action = tracker.action_for(np.zeros(3))
    np.testing.assert_allclose(action[:3], [1.0, -0.5, 1.0], atol=1e-7)
    np.testing.assert_allclose(action[3:6], 0.0, atol=1e-7)
    assert action[6] == pytest.approx(cc.GRIPPER_CLOSE)
    # Independent spec formula: position action = clip(delta / 0.1 m, -1, 1).
    near = cc.WaypointTracker(np.array([0.1, 0.1, 0.1]), cc.GRIPPER_OPEN)
    action = near.action_for(np.array([0.091, 0.1, 0.104]))
    np.testing.assert_allclose(action[:3], [0.09, 0.0, -0.04], atol=1e-7)
    assert action[6] == pytest.approx(cc.GRIPPER_OPEN)


def test_collection_config_preserves_native_stackcube_wristcam_and_reset_noise():
    config = cc.Config()

    assert config.env_kwargs()["robot_uids"] == "panda_wristcam"
    assert config.env_kwargs()["control_mode"] == "pd_ee_delta_pose"
    assert config.env_kwargs()["robot_init_qpos_noise"] == pytest.approx(0.02)
    assert config.env_kwargs()["num_envs"] == 1
    assert config.env_kwargs()["sim_backend"] == "physx_cpu"
    assert config.env_kwargs()["max_episode_steps"] == 500


def test_wristcam_collision_geometry_is_retained_for_footprint_sampling():
    env = gym.make(ENV_ID, **cc.Config().env_kwargs())
    try:
        env.reset(seed=0)
        cache = cc._robot_shape_cache(env.unwrapped)
        camera_shapes = [
            shape
            for link, shapes in cache if str(link.name) == "camera_link"
            for shape in shapes
        ]
        assert camera_shapes
        assert sum(vertices.shape[0] for _, vertices in camera_shapes) > 0
    finally:
        env.close()


def test_tracker_converges_records_errors_and_exhausts_budget():
    target = np.array([0.3, 0.0, 0.0])
    tracker = cc.WaypointTracker(target, 0.0, tolerance=0.004, max_steps=100)
    tcp = np.array([0.0, 0.0, 0.0])
    steps = 0
    while not tracker.step(tcp) and steps < 100:
        tcp = tcp + 0.01 * (target - tcp) / np.linalg.norm(target - tcp)
        steps += 1
    assert tracker.final_error < 0.004
    assert tracker.max_error == pytest.approx(0.3, abs=1e-9)
    assert tracker.step_index == steps + 1
    assert not tracker.exhausted

    stuck = cc.WaypointTracker(target, 0.0, tolerance=0.004, max_steps=7)
    for _ in range(10):
        stuck.step(np.zeros(3))
    assert stuck.exhausted
    assert stuck.max_error == pytest.approx(0.3, abs=1e-9)


# --------------------------------------------------------------------------- #
# Pure geometry: waypoint / cube pair / cylinder sampling
# --------------------------------------------------------------------------- #


def test_choose_waypoint_offsets_from_midpoint_with_forward_lead():
    rng = np.random.default_rng(3)
    for _ in range(20):
        choice = cc.choose_waypoint(FEASIBLE_CUBES, rng)
        assert choice is not None
        a = FEASIBLE_CUBES["cubeA_xy"]
        b = FEASIBLE_CUBES["cubeB_xy"]
        mid = 0.5 * (a + b)
        direction = (b - a) / np.linalg.norm(b - a)
        waypoint = np.asarray(choice["waypoint_xy"])
        # Waypoint offset sideways from the midpoint and led forward toward B.
        lead = float((waypoint - mid) @ direction)
        lateral = float((waypoint - mid) @ np.array([-direction[1], direction[0]]))
        assert lead > 0.0
        assert abs(lateral) >= cc.WAYPOINT_OFFSET_RANGE[0] - 1e-9
        assert abs(lateral) <= cc.WAYPOINT_OFFSET_RANGE[1] + 1e-9
        base = np.array(cc.PANDA_BASE_XY)
        assert np.linalg.norm(waypoint - base) <= cc.WAYPOINT_MAX_BASE_DIST


def test_sample_cube_pair_respects_separation_and_alignment():
    rng = np.random.default_rng(11)
    cfg = cc.Config()
    for _ in range(20):
        cubes = cc.sample_cube_pair(rng, cfg)
        assert cubes is not None
        a = np.asarray(cubes["cubeA_xy"])
        b = np.asarray(cubes["cubeB_xy"])
        separation = float(np.linalg.norm(b - a))
        assert cfg.min_cube_separation <= separation <= cfg.layout.max_separation
        mid = 0.5 * (a + b)
        reach = mid - np.asarray(cfg.layout.base_xy, dtype=np.float64)
        direction = (b - a) / separation
        alignment = abs(float(direction @ reach) / np.linalg.norm(reach))
        assert alignment <= cfg.layout.max_reach_alignment


def test_sample_cylinder_blocks_direct_segment_and_clears_footprint():
    cfg = cc.Config()
    template = dict(
        cubes=FEASIBLE_CUBES,
        waypoint_xy=np.array([-0.13, 0.245]),
        carry_z=0.14,
        footprint=synthetic_footprint(FEASIBLE_CUBES, np.array([-0.13, 0.245])),
    )
    rng = np.random.default_rng(5)
    seen = 0
    for _ in range(10):
        cylinder, reason = cc.sample_cylinder(rng, template, cfg, cfg.roomy_clearance)
        if cylinder is None:
            continue
        seen += 1
        center = np.asarray(cylinder["center_xy"], dtype=np.float64)
        radius = float(cylinder["radius"])
        assert cfg.layout.radius_range[0] <= radius <= cfg.layout.radius_range[1]
        # Blocks the direct A->B XY segment (center within the forbidden strip).
        forbidden = radius + CUBE_HALF_DIAG + cfg.safety_margin
        seg_distance = cc.segment_point_distance(
            center, FEASIBLE_CUBES["cubeA_xy"], FEASIBLE_CUBES["cubeB_xy"]
        )
        assert seg_distance <= forbidden - cfg.layout.min_segment_clearance
        # Endpoints outside the disk.
        assert np.linalg.norm(center - FEASIBLE_CUBES["cubeA_xy"]) > forbidden
        assert np.linalg.norm(center - FEASIBLE_CUBES["cubeB_xy"]) > forbidden
        # Outside every recorded footprint point by radius + margin.
        distances = np.linalg.norm(template["footprint"] - center, axis=1)
        achieved = float(distances.min()) - radius
        assert achieved >= cc.FOOTPRINT_MARGIN - 1e-9
        assert achieved >= cfg.roomy_clearance - 1e-9
        assert cylinder["label"] == "roomy"
        assert cylinder["min_footprint_clearance"] == pytest.approx(achieved, abs=1e-9)
        # Exact configured height, recorded, and it blocks passage over the top.
        assert cylinder["height"] == pytest.approx(cfg.obstacle_height)
        assert cfg.obstacle_height > template["carry_z"] + cc.CUBE_HALF_SIZE
    assert seen >= 5


def test_sample_cylinder_rejects_when_footprint_leaves_no_room():
    cfg = cc.Config()
    template = dict(
        cubes=FEASIBLE_CUBES,
        waypoint_xy=np.array([-0.13, 0.245]),
        carry_z=0.14,
        # Direct path footprint: no cylinder can block AB and clear the trace.
        footprint=synthetic_footprint(
            FEASIBLE_CUBES, 0.5 * (FEASIBLE_CUBES["cubeA_xy"] + FEASIBLE_CUBES["cubeB_xy"])
        ),
    )
    rng = np.random.default_rng(5)
    cylinder, reason = cc.sample_cylinder(rng, template, cfg, cfg.tight_clearance)
    assert cylinder is None
    assert reason == "no_feasible_cylinder"


def test_sample_cylinder_prior_favors_side_away_from_swept_footprint():
    cfg = cc.Config()
    cubes = FEASIBLE_CUBES
    # The recorded detour occupies the negative-x side of the direct segment.
    footprint = synthetic_footprint(cubes, np.array([-0.14, 0.17]))
    template = dict(cubes=cubes, waypoint_xy=np.array([-0.14, 0.17]),
                    carry_z=0.14, footprint=footprint)
    rng = np.random.default_rng(0)
    cylinders = [cc.sample_cylinder(rng, template, cfg, cfg.tight_clearance)[0]
                 for _ in range(30)]
    centers = np.asarray([cylinder["center_xy"] for cylinder in cylinders
                          if cylinder is not None])
    assert len(centers) >= 20
    # The footprint projects to negative x relative to AB (which runs along y).
    assert np.mean(centers[:, 0] > 0.0) >= 0.8


def test_sample_cylinder_requires_cylinder_taller_than_carried_cube_top():
    cfg = cc.Config(obstacle_height=0.10)
    template = dict(
        cubes=FEASIBLE_CUBES,
        waypoint_xy=np.array([-0.13, 0.245]),
        carry_z=0.14,
        footprint=synthetic_footprint(FEASIBLE_CUBES, np.array([-0.13, 0.245])),
    )
    rng = np.random.default_rng(5)
    cylinder, reason = cc.sample_cylinder(rng, template, cfg, cfg.tight_clearance)
    assert cylinder is None
    assert reason == "cylinder_not_tall_enough_for_side_detour"


# --------------------------------------------------------------------------- #
# run_episode stats seam (stub env, no sim)
# --------------------------------------------------------------------------- #


class _StubPose:
    def __init__(self, p, q):
        self._p = p
        self._q = q

    @property
    def p(self):
        return np.asarray(self._p)[None]

    @property
    def q(self):
        return np.asarray(self._q)[None]


class _StubNode:
    pass


class StubBase:
    def __init__(self, drift_schedule):
        self.tcp = np.array([0.0, 0.0, 0.10])
        self.quat = np.array([0.0, 1.0, 0.0, 0.0])
        self.drift_schedule = drift_schedule
        self.elapsed = 0
        self.grasped = False
        self.cubeA_xyz = np.array([0.0, 0.05, 0.02])
        self.cubeB_xyz = np.array([0.0, 0.30, 0.02])

    def evaluate(self):
        return dict(
            success=np.array([True]),
            stack_success=np.array([True]),
            obstacle_collision=np.array([False]),
            obstacle_crossing=np.array([False]),
            is_cubeA_grasped=np.array([self.grasped]),
        )

    @property
    def agent(self):
        link = _StubNode()
        link.pose = _StubPose(self.tcp, self.quat)
        agent = _StubNode()
        agent.tcp = link
        return agent

    @property
    def _elapsed_steps(self):
        return np.array([self.elapsed])

    @property
    def p(self):
        return np.asarray(self.cubeA_xyz)[None]

    @property
    def cubeA(self):
        cube = _StubNode()
        cube.pose = _StubPose(self.cubeA_xyz, None)
        return cube

    def step_sim(self, action):
        self.elapsed += 1
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        target = self.tcp + action[:3] * 0.1
        moved = self.tcp + 0.5 * (target - self.tcp)
        # Descend stall: with the gripper open and commanded low, the TCP stops
        # 20 mm short of the target (contact-rich grasp descent) — the strict
        # 4 mm convergence gate would time out here.
        if action[6] == cc.GRIPPER_OPEN and target[2] < 0.05:
            moved[2] = max(moved[2], 0.04)
        self.tcp = moved
        if action[6] == cc.GRIPPER_CLOSE and self.tcp[2] < 0.05:
            self.grasped = True
        if self.grasped:
            self.cubeA_xyz = np.array([self.tcp[0], self.tcp[1], self.tcp[2]])
        # Non-monotonic drift: rises for the first 40 steps, then returns.
        angle = self.drift_schedule(self.elapsed)
        axis, _ = quat2axangle(self.quat)
        delta = axangle2quat(np.asarray(axis, dtype=np.float64), angle)
        self.quat = qmult(self.quat, delta)


class StubEnv:
    def __init__(self, drift_schedule):
        self.base = StubBase(drift_schedule)
        self.done = False

    @property
    def unwrapped(self):
        return self.base

    def step(self, action):
        self.base.step_sim(action[0])
        return None, 0.0, False, False, {}

    def evaluate(self):
        return dict(
            success=True,
            stack_success=True,
            obstacle_collision=False,
            obstacle_crossing=False,
            is_cubeA_grasped=self.base.grasped,
        )


def build_stub_plan(cfg):
    return cc.build_episode_plan(
        np.array([0.0, 0.05, 0.02]),
        np.array([0.0, 0.30, 0.02]),
        np.array([-0.13, 0.175]),
        cfg.lift_height,
    )


def test_run_episode_reports_carry_z_for_horizontal_phases_only():
    cfg = cc.Config()
    env = StubEnv(lambda step: 0.0)
    trace = cc.run_episode(env, build_stub_plan(cfg), cfg)
    assert trace["success"] and trace["stack_success"]
    # The horizontal carry phases hold commanded carry z (0.14); lift/place
    # dip to grasp/place heights and must NOT be folded into the carry stats.
    assert trace["carry_z_observed_min"] > 0.10
    assert trace["carry_z_observed_max"] < 0.20
    phase_names = [entry["phase"] for entry in trace["phases"]]
    assert "carry_to_waypoint" in phase_names and "carry_to_place" in phase_names


def test_run_episode_records_maximum_orientation_drift_not_final_only():
    cfg = cc.Config()
    peak = 0.02
    # Per-step drift delta: rises for the first 40 steps, unwinds by step 80.
    env = StubEnv(lambda step: peak / 40.0 if step <= 40 else (
        -peak / 40.0 if step <= 80 else 0.0))
    trace = cc.run_episode(env, build_stub_plan(cfg), cfg)
    # Drift peaked mid-episode and returned to ~0 by the end; the trace must
    # report the maximum over steps, not the final value.
    assert trace["orientation_drift_max_rad"] == pytest.approx(peak, rel=0.25)


def test_run_episode_descend_does_not_gate_on_convergence_and_records_trace():
    cfg = cc.Config()
    env = StubEnv(lambda step: 0.0)
    plan = build_stub_plan(cfg)
    trace = cc.run_episode(env, plan, cfg, record_trace=True)
    # The stalled descend (20 mm short) must not fail the episode: the grasp
    # check gates the descent instead of a strict convergence tolerance.
    assert trace["success"] is True
    descend = [entry for entry in trace["phases"] if entry["phase"] == "descend"][0]
    assert descend["final_error"] > cfg.waypoint_tolerance
    assert descend["final_error"] <= cfg.descend_tolerance + 0.02
    # Actual TCP and cube traces recorded during the horizontal carry phases:
    # the TCP must visit the waypoint and both cube columns, and the carried
    # cube must travel from A's column to B's column.
    tcp = np.asarray(trace["tcp_xy"], dtype=np.float64)
    cube = np.asarray(trace["cube_xy"], dtype=np.float64)
    assert tcp.shape[1] == 2 and cube.shape[1] == 2
    waypoint = np.asarray(plan["waypoint_xy"], dtype=np.float64)
    assert tcp[:, 0].min() <= waypoint[0] + 0.01
    assert tcp[:, 1].max() >= 0.30 - 0.05 and tcp[:, 1].min() <= 0.05 + 0.05
    assert cube[:, 1].max() >= 0.30 - 0.05


def test_run_episode_stops_at_first_stack_success_even_with_same_step_violation():
    cfg = cc.Config()
    env = StubEnv(lambda step: 0.0)
    step_calls = []
    env.base.evaluate = lambda: dict(
        success=np.array([False]),
        stack_success=np.array([True]),
        obstacle_collision=np.array([True]),
        obstacle_crossing=np.array([False]),
        is_cubeA_grasped=np.array([False]),
    )
    def step(action):
        step_calls.append(action)
        return (
            None, 0.0, True, False,
            dict(success=np.array([False]), stack_success=np.array([True]),
                 obstacle_collision=np.array([True]), obstacle_crossing=np.array([False])),
        )

    env.step = step

    trace = cc.run_episode(env, build_stub_plan(cfg), cfg)

    assert trace["stack_success"] is True
    assert trace["success"] is False
    assert trace["obstacle_collision"] is True
    assert len(step_calls) == 1


# --------------------------------------------------------------------------- #
# collect() seam: ordering, reuse, budgets, initial-state equality
# --------------------------------------------------------------------------- #


class FakeRecorder:
    """Stands in for ``record_episode_once`` at the collect() seam."""

    def __init__(self, obstacle_outcomes=(True,), vary_initial_state=False):
        self.obstacle_outcomes = list(obstacle_outcomes)
        self.vary_initial_state = vary_initial_state
        self.call_index = 0
        self.calls = []

    def _write_h5_json(self, scratch, name, template_index, cylinder, steps=3):
        rows = steps + 1
        with h5py.File(scratch / f"{name}.h5", "w") as handle:
            group = handle.create_group("traj_0")
            group.create_dataset("actions", data=np.zeros((steps, 7), dtype=np.float32))
            group.create_dataset("obs", data=np.zeros((rows, 4), dtype=np.float32))
            states = group.create_group("env_states")
            actors = states.create_group("actors")
            robot_state = np.zeros((rows, 31), dtype=np.float32)
            if self.vary_initial_state:
                self.call_index += 1
                robot_state[0, 0] = float(self.call_index)
            actors.create_dataset("cubeA", data=np.zeros((rows, 13), dtype=np.float32))
            actors.create_dataset("cubeB", data=np.zeros((rows, 13), dtype=np.float32))
            states.create_group("articulations").create_dataset(
                "panda", data=robot_state
            )
            obstacle_row = np.zeros((rows, 3), dtype=np.float32)
            if cylinder is not None:
                obstacle_row[:, 0] = cylinder["center_xy"][0]
                obstacle_row[:, 1] = cylinder["center_xy"][1]
                obstacle_row[:, 2] = cylinder["radius"]
            states.create_dataset("obstacle", data=obstacle_row)
        episode = dict(
            success=True, stack_success=True, obstacle_collision=False,
            obstacle_crossing=False,
        )
        (scratch / f"{name}.json").write_text(json.dumps(dict(episodes=[episode])))
        state = cc.read_initial_robot_cube_state(scratch / f"{name}.h5", "traj_0")
        return cc.initial_state_hash(state)

    def __call__(self, scratch_dir, trajectory_name, request, cfg):
        kind = request["kind"]
        self.calls.append(dict(request=request))
        scratch = Path(scratch_dir)
        seed = int(request["seed"])
        template_index = int(request.get("template_index", -1))
        if kind == "sanity":
            state_hash = self._write_h5_json(scratch, trajectory_name, -1, None)
            return True, dict(kind=kind, seed=seed, failure=None,
                              initial_state_hash=state_hash,
                              elapsed_seconds=0.001, trace=dict(steps=30))
        if kind == "template":
            cubes = request["cubes"]
            waypoint = np.asarray(request["waypoint_xy"], dtype=np.float64)
            plan = cc.build_episode_plan(
                np.array([cubes["cubeA_xy"][0], cubes["cubeA_xy"][1], 0.02]),
                np.array([cubes["cubeB_xy"][0], cubes["cubeB_xy"][1], 0.02]),
                waypoint, cfg.lift_height,
            )
            footprint = synthetic_footprint(cubes, waypoint)
            trace = dict(steps=50, success=True, stack_success=True,
                         obstacle_collision=False, obstacle_crossing=False,
                         orientation_drift_max_rad=0.01,
                         carry_z_observed_min=0.139, carry_z_observed_max=0.141,
                         footprint=[[float(p[0]), float(p[1])] for p in footprint])
            state_hash = self._write_h5_json(scratch, trajectory_name, template_index, None)
            return True, dict(
                kind=kind, template_index=template_index, seed=seed,
                failure=None, elapsed_seconds=0.002, plan=plan, trace=trace,
                initial_state_hash=state_hash,
            )
        # obstacle episode
        index = sum(1 for call in self.calls if call["request"]["kind"] == "obstacle") - 1
        valid = self.obstacle_outcomes[min(index, len(self.obstacle_outcomes) - 1)]
        attempt = dict(
            kind=kind, template_index=template_index,
            reuse_index=int(request["reuse_index"]), seed=seed,
            failure=None if valid else "obstacle_collision",
            elapsed_seconds=0.003,
            cylinder=request["cylinder"], trace=dict(steps=42),
        )
        if valid:
            attempt["initial_state_hash"] = self._write_h5_json(
                scratch, trajectory_name, template_index, request["cylinder"])
        return valid, attempt


def run_collect(tmp_path, recorder, config=None):
    config = config or cc.Config(target_successes=4, reuse_count=2, seed=7,
                                 max_templates=20, max_episodes=40,
                                 max_cylinder_attempts=30)
    report = cc.collect(tmp_path, config, episode_recorder=recorder)
    return report


def test_collect_runs_no_obstacle_template_before_its_obstacle_episodes(tmp_path):
    recorder = FakeRecorder()
    report = run_collect(tmp_path / "ordering", recorder)
    assert report["complete"] and report["accepted"] == 4
    kinds = [call["request"]["kind"] for call in recorder.calls]
    assert kinds[0] == "sanity"
    # Each template's no-obstacle run precedes every obstacle episode that
    # reuses it instead of adapting the path to each obstacle.
    template_positions = {}
    for position, call in enumerate(recorder.calls):
        request = call["request"]
        if request["kind"] == "template":
            template_positions[request["template_index"]] = position
    for position, call in enumerate(recorder.calls):
        request = call["request"]
        if request["kind"] == "obstacle":
            assert position > template_positions[request["template_index"]]


def test_collect_reuses_same_seed_plan_state_with_different_cylinders(tmp_path):
    recorder = FakeRecorder()
    report = run_collect(tmp_path / "reuse", recorder,
                         cc.Config(target_successes=2, reuse_count=2, seed=7,
                                   max_templates=20, max_episodes=40,
                                   max_cylinder_attempts=30))
    assert report["complete"] and report["accepted"] == 2
    obstacle_calls = [call["request"] for call in recorder.calls
                      if call["request"]["kind"] == "obstacle"]
    assert len(obstacle_calls) == 2
    first, second = obstacle_calls
    # SAME plan (waypoint), SAME seed, SAME cube layout — only the cylinder
    # placement differs between the two reuses of one template.
    assert first["seed"] == second["seed"]
    assert first["plan"] == second["plan"]
    assert first["cubes"] == second["cubes"]
    assert first["cylinder"]["center_xy"] != second["cylinder"]["center_xy"]
    # Clearance preference: reuse slot 0 starts roomy, slot 1 tight.
    assert first["cylinder"]["label"] == "roomy"
    assert second["cylinder"]["label"] == "tight"
    # Report links both episodes to one template with distinct templates>=1.
    assert report["distinct_templates"] == 1
    template = report["templates"]["0"]
    assert template["reuse_accepted"] == 2
    assert template["seed"] == first["seed"]


def test_collect_template_failure_skips_obstacle_episodes_and_counts_failure(tmp_path):
    recorder = FakeRecorder()

    def failing_template(scratch_dir, trajectory_name, request, cfg):
        if request["kind"] == "template" and request["template_index"] == 0:
            recorder.calls.append(dict(request=request))
            return False, dict(kind="template", seed=int(request["seed"]),
                               template_index=int(request["template_index"]),
                               failure="grasp_failed", elapsed_seconds=0.002)
        return recorder(scratch_dir, trajectory_name, request, cfg)

    config = cc.Config(target_successes=1, reuse_count=2, seed=7,
                       max_templates=3, max_episodes=40, max_cylinder_attempts=30)
    report = cc.collect(tmp_path / "tfail", config, episode_recorder=failing_template)
    assert report["complete"] and report["accepted"] == 1
    assert report["failures"].get("grasp_failed") == 1
    kinds = [call["request"]["kind"] for call in recorder.calls]
    # The failed template contributed zero obstacle episodes; the next
    # template's first accepted reuse already reaches the target of 1.
    assert "obstacle" not in kinds[:2]
    assert kinds.count("obstacle") == 1


def test_collect_bounds_cylinder_attempts_per_reuse_slot(tmp_path):
    recorder = FakeRecorder(obstacle_outcomes=(False,))
    config = cc.Config(target_successes=1, reuse_count=2, seed=7,
                       max_templates=1, max_episodes=100,
                       max_cylinder_attempts=6)
    report = cc.collect(tmp_path / "budget", config, episode_recorder=recorder)
    assert not report["complete"] and report["accepted"] == 0
    obstacle_calls = [call["request"] for call in recorder.calls
                      if call["request"]["kind"] == "obstacle"]
    # Finite per-slot budget: 6 attempts per reuse slot (3 preferred label,
    # then 3 fallback), then the template is abandoned.
    assert len(obstacle_calls) == 12
    labels = [request["cylinder"]["label"] for request in obstacle_calls]
    achieved = [request["cylinder"]["min_footprint_clearance"] for request in obstacle_calls]
    # Labels are accurate for the achieved clearance, and the preferred
    # (roomy) phases always accept roomy candidates.
    assert labels[:3] == ["roomy"] * 3
    assert labels[9:] == ["roomy"] * 3
    for label, value in zip(labels, achieved):
        expect = "roomy" if value >= config.roomy_clearance else "tight"
        assert label == expect
    assert all(value >= config.tight_clearance for value in achieved)
    assert report["failures"]["obstacle_collision"] == 12


def test_collect_verifies_identical_initial_state_from_h5(tmp_path):
    recorder = FakeRecorder(vary_initial_state=True)
    with pytest.raises(RuntimeError, match="initial state"):
        run_collect(tmp_path / "mismatch", recorder)


def test_initial_state_signature_accepts_native_wristcam_articulation(tmp_path):
    path = tmp_path / "wristcam.h5"
    with h5py.File(path, "w") as handle:
        trajectory = handle.create_group("traj_0")
        states = trajectory.create_group("env_states")
        actors = states.create_group("actors")
        actors.create_dataset("cubeA", data=np.zeros((2, 13), dtype=np.float32))
        actors.create_dataset("cubeB", data=np.ones((2, 13), dtype=np.float32))
        articulations = states.create_group("articulations")
        articulations.create_dataset(
            "panda_wristcam", data=np.full((2, 9), 0.25, dtype=np.float32)
        )

    state = cc.read_initial_robot_cube_state(path, "traj_0")

    assert set(state) == {
        "actors/cubeA", "actors/cubeB", "articulations/panda_wristcam"
    }
    assert len(cc.initial_state_hash(state)) == 64


def test_collect_writes_report_stage_timings_and_denominators(tmp_path):
    recorder = FakeRecorder()
    report = run_collect(tmp_path / "timings", recorder)
    assert report["accepted"] == 4
    assert set(report["stage_seconds"]) == {"sanity", "templates", "sampling", "episodes"}
    assert report["stage_seconds"]["templates"] > 0
    assert report["stage_seconds"]["episodes"] > 0
    assert report["elapsed_seconds"] >= sum(report["stage_seconds"].values())
    # Physics episode denominators: every obstacle attempt is reported.
    obstacle_attempts = [call for call in recorder.calls
                         if call["request"]["kind"] == "obstacle"]
    assert len(report["episodes"]) == len(obstacle_attempts)
    assert report["physics_attempts"] == len(report["episodes"])
    assert report["candidate_templates"] == 2
    assert report["no_obstacle_successful_templates"] == 2
    assert report["cylinder_sampling_calls"] == 4
    assert report["valid_per_minute"] > 0
    metadata = json.loads((tmp_path / "timings" / "trajectory.json").read_text())
    first = metadata["episodes"][0]
    assert first["cylinder"]["height"] == pytest.approx(report["config"]["obstacle_height"])
    assert len(first["initial_state_hash"]) == 64
    assert first["template_index"] == 0


def test_collect_retains_template_sweep_footprint_once_in_trace(tmp_path):
    report = run_collect(tmp_path / "footprint", FakeRecorder(), cc.Config(
        target_successes=1, reuse_count=1, seed=7, max_templates=2,
        max_episodes=2, max_cylinder_attempts=2,
    ))

    template = report["templates"]["0"]
    footprint = template["trace"]["footprint"]
    assert len(footprint) > 100
    assert "footprint" not in template
    assert np.asarray(footprint).shape[1] == 2


def test_collect_refuses_to_overwrite_existing_outputs(tmp_path):
    (tmp_path / "trajectory.h5").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        cc.collect(tmp_path, cc.Config(seed=0), episode_recorder=FakeRecorder())


def test_cli_builds_config_and_reports_incomplete(monkeypatch, tmp_path):
    captured = {}

    def fake_collect(output_dir, config, **kwargs):
        captured["output_dir"] = output_dir
        captured["config"] = config
        return dict(complete=False, accepted=0, elapsed_seconds=0.0,
                    distinct_templates=0)

    monkeypatch.setattr(cc, "collect", fake_collect)
    with pytest.raises(RuntimeError):
        cc.main([
            "--output-dir", str(tmp_path / "out"),
            "--target-successes", "4", "--reuse-count", "1",
            "--max-templates", "7", "--max-episodes", "9", "--seed", "5",
            "--max-cylinder-attempts", "11",
        ])
    assert captured["output_dir"] == str(tmp_path / "out")
    config = captured["config"]
    assert config.target_successes == 4
    assert config.reuse_count == 1
    assert config.max_templates == 7
    assert config.max_episodes == 9
    assert config.seed == 5
    assert config.max_cylinder_attempts == 11
    assert config.env_kwargs()["max_episode_steps"] == config.max_episode_steps
