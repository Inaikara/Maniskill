from pathlib import Path

import numpy as np
import pytest
import torch

from examples.baselines.constrain.pickcube_transport_detour import (
    classify_clearances,
    generate,
    midsegment_candidates,
    select_cylinder_height,
    shared_transport_fork,
)


def test_height_covers_measured_direct_body_and_is_not_the_stackcube_cylinder():
    direct = np.array([[0.0, 0.0, 0.16]])
    side = np.array([[0.20, 0.0, 0.30]])
    end = np.array([[0.25, 0.0, 0.05]])

    decision = select_cylinder_height(
        direct, side, end, center_xy=(0.0, 0.0), radius=0.03
    )

    assert decision.accepted
    assert decision.height_m == pytest.approx(0.17)
    assert decision.height_m != 0.25
    assert decision.reason is None


def test_side_column_occupancy_is_not_a_feasible_height():
    decision = select_cylinder_height(
        np.array([[0.0, 0.0, 0.16]]),
        np.array([[0.0, 0.0, 0.10]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert not decision.accepted
    assert decision.height_m is None
    assert decision.reason == "side_detour_occupies_cylinder_column"


def test_side_point_above_measured_height_is_not_column_occupancy():
    decision = select_cylinder_height(
        np.array([[0.0, 0.0, 0.16]]),
        np.array([[0.0, 0.0, 0.40]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert decision.accepted
    assert decision.height_m == pytest.approx(0.17)


def test_end_region_sweep_rejects_a_transport_blocking_column():
    decision = select_cylinder_height(
        np.array([[0.0, 0.0, 0.16]]),
        np.array([[0.20, 0.0, 0.30]]),
        np.array([[0.02, 0.0, 0.08]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert not decision.accepted
    assert decision.reason == "end_region_sweep_occupies_cylinder_column"


def test_direct_transport_that_misses_the_column_is_not_blocked():
    decision = select_cylinder_height(
        np.array([[0.20, 0.0, 0.16]]),
        np.array([[0.20, 0.0, 0.30]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert not decision.accepted
    assert decision.reason == "direct_transport_misses_cylinder"


def test_clearance_report_keeps_vertex_gap_object_margin_and_surface_apart():
    report = classify_clearances(
        vertex_xy_distance_m=0.08,
        radius_m=0.03,
        cube_center_distance_m=0.11,
        forbidden_radius_m=0.0632842712474619,
        contact_separation_m=None,
        contact_free=True,
    )

    assert report.geometric_vertex_clearance_m == pytest.approx(0.05)
    assert report.geometric_is_surface_clearance is False
    assert report.object_xy_margin_m == pytest.approx(0.0467157287525381)
    assert report.robot_surface_clearance_m is None
    assert report.safety_proof == "contact_free_execution"


def test_measured_contact_separation_is_not_copied_from_the_vertex_gap():
    report = classify_clearances(
        vertex_xy_distance_m=0.08,
        radius_m=0.03,
        cube_center_distance_m=0.11,
        forbidden_radius_m=0.0632842712474619,
        contact_separation_m=0.0,
        contact_free=False,
    )

    assert report.geometric_vertex_clearance_m == pytest.approx(0.05)
    assert report.robot_surface_clearance_m == pytest.approx(0.0)
    assert report.safety_proof == "contact"


def test_midsegment_candidates_stay_on_the_anti_detour_side_of_the_segment():
    candidates = midsegment_candidates(
        start=(0.0, 0.0), goal=(1.0, 0.0), base_xy=(0.5, -1.0), radius=0.03
    )

    assert len(candidates) == 25
    assert candidates.shape[1] == 2
    assert np.all(candidates[:, 0] >= 0.40)
    assert np.all(candidates[:, 0] <= 0.60)
    assert np.all(candidates[:, 1] >= 0.0)
    assert np.all(candidates[:, 1] <= 0.04)


def test_fork_shares_lift_and_goal_and_inserts_only_a_side_waypoint():
    join = {"measured": True, "max_commanded_heading_step_deg": 8.0,
            "max_tcp_heading_step_deg": 9.0, "transport_steps": 8}
    direct = {
        "phases": [("lift", "reached"), ("transport", "reached"), ("goal_entry", "reached")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14],
            "goal_xyz": [1.0, 0.0, 0.12],
            "waypoint_xyz": None,
            "transport_exit_xyz": [1.0, 0.0, 0.14],
            "goal_entry_xyz": [1.0, 0.0, 0.12],
            "retreat_required": False,
        },
        "join": join,
    }
    detour = {
        "phases": [("lift", "reached"), ("waypoint", "reached"), ("transport", "reached"),
                   ("goal_entry", "reached")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14],
            "goal_xyz": [1.0, 0.0, 0.12],
            "waypoint_xyz": [0.5, 0.2, 0.14],
            "transport_exit_xyz": [1.0, 0.0, 0.14],
            "goal_entry_xyz": [1.0, 0.0, 0.12],
            "retreat_required": False,
        },
        "join": join,
    }

    ok, reason = shared_transport_fork(direct, detour)

    assert ok
    assert reason is None


def test_fork_rejects_a_direct_waypoint_and_a_hooked_detour():
    direct = {
        "phases": [("lift", "reached"), ("waypoint", "reached"), ("goal", "reached")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14],
            "goal_xyz": [0.2, 0.0, 0.12],
            "waypoint_xyz": [0.1, 0.5, 0.14],
        },
    }
    detour = {
        "phases": [("lift", "reached"), ("waypoint", "reached"), ("goal", "reached")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14],
            "goal_xyz": [0.2, 0.0, 0.12],
            "waypoint_xyz": [0.1, 0.5, 0.14],
        },
    }

    ok, reason = shared_transport_fork(direct, detour)

    assert not ok
    assert reason == "direct_has_waypoint"

    direct["phases"] = [("lift", "reached"), ("goal", "reached")]
    direct["targets"]["waypoint_xyz"] = None
    ok, reason = shared_transport_fork(direct, detour)
    assert not ok
    assert reason == "shared_endpoint_missing"


def test_deadline_kills_the_worker_and_counts_the_attempt(tmp_path):
    report = generate(
        tmp_path / "batch", seed=1, target=1, max_attempts=1, max_wall_seconds=10,
        per_attempt_deadline=0.3,
        worker_argv_fn=lambda seed, result_dir: ["sleep", "30"],
    )

    assert report["attempts"] == 1
    assert report["accepted"] == 0
    assert report["failures"] == {"attempt_deadline": 1}
    assert report["acceptance_rate"] == 0.0
    assert not list(Path("/proc").glob("*/cmdline")) or _no_lingering_sleep()


def test_rejected_worker_stays_in_the_attempt_denominator(tmp_path):
    def argv(seed, result_dir):
        script = (
            "import json, pathlib, sys; "
            f"pathlib.Path({str(Path(result_dir) / 'result.json')!r}).write_text("
            "json.dumps({'accepted': False, 'failure': 'direct_transport_not_blocked'})); "
            "sys.exit(0)"
        )
        return ["python", "-c", script]

    report = generate(
        tmp_path / "out", seed=1, target=1, max_attempts=1, max_wall_seconds=10,
        per_attempt_deadline=5, worker_argv_fn=argv,
    )

    assert report["attempts"] == 1
    assert report["accepted"] == 0
    assert report["failures"] == {"direct_transport_not_blocked": 1}


def test_attempt_budget_counts_invocations_not_the_unrun_target(tmp_path):
    calls = []

    def argv(seed, result_dir):
        calls.append(seed)
        return ["python", "-c", "import sys; sys.exit(3)"]

    report = generate(
        tmp_path / "batch", seed=1, target=4, max_attempts=2, max_wall_seconds=30,
        per_attempt_deadline=5, worker_argv_fn=argv,
    )

    assert report["attempts"] == 2
    assert len(calls) == 2
    assert report["accepted"] == 0
    assert report["complete"] is False


def test_nonpositive_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate(tmp_path, target=1, max_attempts=1, max_wall_seconds=0, per_attempt_deadline=1)


def test_cube_center_above_the_cylinder_still_crosses_the_xy_disk():
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env

    env = make_pickcube_env(obstacle_radius=0.03, obstacle_height=0.15)
    try:
        env.reset(seed=3, options={
            "obstacle_enabled": True, "obstacle_center": [0.0, 0.1],
            "cube_xy": [-0.08, 0.1], "goal_pos": [0.08, 0.1, 0.12],
        })
        base = env.unwrapped
        base._grasp_started[:] = True
        base._prev_cube_xy[:] = torch.tensor([[-0.08, 0.1]])
        state = base.get_state_dict()
        state["actors"]["cube"][0, 0] = 0.08
        state["actors"]["cube"][0, 2] = 0.30
        base.set_state_dict(state)
        _, _, terminated, truncated, info = env.step(torch.zeros(7))
        assert not bool(terminated)
        assert not bool(truncated)
        assert float(base.cube.pose.p[0, 2]) > float(base.obstacle_height)
        assert bool(info["obstacle_crossing"].item())
    finally:
        env.close()


def test_goal_entry_episode_end_still_shares_the_target_phase():
    join = {"measured": True, "max_commanded_heading_step_deg": 8.0,
            "max_tcp_heading_step_deg": 9.0, "transport_steps": 8}
    direct = {
        "phases": [("lift", "reached"), ("transport", "reached"), ("goal_entry", "episode_ended")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14], "goal_xyz": [1.0, 0.0, 0.12], "waypoint_xyz": None,
            "transport_exit_xyz": [1.0, 0.0, 0.14], "goal_entry_xyz": [1.0, 0.0, 0.12],
            "retreat_required": False,
        },
        "join": join,
    }
    detour = {
        "phases": [("lift", "reached"), ("transport", "reached"), ("goal_entry", "episode_ended")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14], "goal_xyz": [1.0, 0.0, 0.12],
            "waypoint_xyz": [0.5, 0.2, 0.14],
            "transport_exit_xyz": [1.0, 0.0, 0.14], "goal_entry_xyz": [1.0, 0.0, 0.12],
            "retreat_required": False,
        },
        "join": join,
    }

    ok, reason = shared_transport_fork(direct, detour)

    assert ok
    assert reason is None


def test_angle_under_150_without_executed_motion_is_not_a_smooth_join():
    lift = [0.0, 0.0, 0.14]
    waypoint = [0.30, 0.0, 0.14]
    goal = [0.15, 0.2598076211353316, 0.12]
    transport_exit = [goal[0], goal[1], lift[2]]
    direct = {
        "phases": [("lift", "reached"), ("transport", "reached"), ("goal_entry", "reached")],
        "targets": {
            "lift_xyz": lift, "goal_xyz": goal, "waypoint_xyz": None,
            "transport_exit_xyz": transport_exit, "goal_entry_xyz": goal,
            "retreat_required": False,
        },
        "join": {"measured": True, "max_commanded_heading_step_deg": 4.0,
                 "max_tcp_heading_step_deg": 5.0, "transport_steps": 8},
    }
    detour = {
        "phases": [("lift", "reached"), ("waypoint", "reached"), ("transport", "reached"),
                   ("goal_entry", "reached")],
        "targets": {
            "lift_xyz": lift, "goal_xyz": goal, "waypoint_xyz": waypoint,
            "transport_exit_xyz": transport_exit, "goal_entry_xyz": goal,
            "retreat_required": False,
        },
    }
    corner = _turn_deg(lift[:2], waypoint[:2], goal[:2])

    ok, reason = shared_transport_fork(direct, detour)

    assert 90.0 < corner < 150.0
    assert not ok
    assert reason == "join_not_measured"


def test_executed_heading_step_cap_is_45_degrees_not_a_polyline_corner():
    from examples.baselines.constrain.pickcube_transport_detour import measure_executed_join

    sharp_actions = np.array([
        [0.30, 0.00, 0, 0, 0, 0, -1],
        [0.30, 0.00, 0, 0, 0, 0, -1],
        [0.00, 0.30, 0, 0, 0, 0, -1],
        [0.00, 0.30, 0, 0, 0, 0, -1],
    ], dtype=np.float64)
    sharp_tcp = np.array([
        [0.00, 0.00, 0.14],
        [0.02, 0.00, 0.14],
        [0.04, 0.00, 0.14],
        [0.04, 0.02, 0.14],
        [0.04, 0.04, 0.14],
    ])
    sharp = measure_executed_join(sharp_tcp, sharp_actions, ["transport"] * 4)

    assert not sharp.continuous
    assert sharp.reason == "transport_join_discontinuous"
    assert sharp.max_commanded_heading_step_deg == pytest.approx(90.0)
    assert sharp.max_tcp_heading_step_deg == pytest.approx(90.0)

    angles = np.deg2rad(np.arange(8) * 10.0)
    smooth_actions = np.column_stack([
        0.30 * np.cos(angles), 0.30 * np.sin(angles), np.zeros((8, 5)),
    ])
    smooth_actions[:, 6] = -1.0
    smooth_tcp = np.zeros((9, 3))
    smooth_tcp[0] = [0.0, 0.0, 0.14]
    for index, angle in enumerate(angles):
        smooth_tcp[index + 1, :2] = smooth_tcp[index, :2] + 0.01 * np.array([np.cos(angle), np.sin(angle)])
        smooth_tcp[index + 1, 2] = 0.14
    smooth = measure_executed_join(smooth_tcp, smooth_actions, ["transport"] * 8)

    assert smooth.continuous
    assert smooth.reason is None
    assert smooth.max_commanded_heading_step_deg == pytest.approx(10.0, abs=1e-6)
    assert smooth.max_commanded_heading_step_deg <= 45.0


def test_endpoint_sweep_is_phase_based_and_keeps_both_paths():
    from examples.baselines.constrain.pickcube_transport_detour import (
        classify_direct_block,
        endpoint_column_clear,
        endpoint_sweep_points,
    )

    sweep = [
        {"phase": "goal_entry", "bodies": {
            "cube": np.array([[0.20, 0.0, 0.12]]),
            "panda_link6": np.array([[0.01, 0.0, 0.08]]),
        }},
        {"phase": "transport", "bodies": {
            "panda_hand": np.array([[0.0, 0.0, 0.08]]),
            "cube": np.array([[0.05, 0.0, 0.12]]),
        }},
        {"phase": "settle", "bodies": {"panda_leftfinger": np.array([[0.02, 0.01, 0.10]])}},
    ]
    points = endpoint_sweep_points(sweep)

    assert any(np.allclose(point, [0.01, 0.0, 0.08]) for point in points)
    assert any(np.allclose(point, [0.02, 0.01, 0.10]) for point in points)
    assert not any(np.allclose(point, [0.0, 0.0, 0.08]) for point in points)
    ok, reason = endpoint_column_clear(
        points, np.array([[0.20, 0.0, 0.05]]), center_xy=(0.0, 0.0), radius=0.03, height=0.15,
    )
    assert not ok
    assert reason == "direct_endpoint_occupies_cylinder_column"
    ok, reason = endpoint_column_clear(
        np.zeros((0, 3)), np.array([[0.20, 0.0, 0.05]]), center_xy=(0.0, 0.0), radius=0.03, height=0.15,
    )
    assert not ok
    assert reason == "endpoint_sweep_incomplete"
    ok, reason = classify_direct_block({"phase": "transport", "cube_xyz": [0.02, 0.0, 0.12]})
    assert ok
    assert reason is None
    ok, reason = classify_direct_block({"phase": "goal_entry", "cube_xyz": [5.0, 5.0, 0.12]})
    assert not ok
    assert reason == "direct_block_contact_in_endpoint"


def test_generate_freezes_working_tree_source_identity_before_artifacts(tmp_path):
    import hashlib
    import json

    out = tmp_path / "batch"
    generate(
        out, seed=1, target=1, max_attempts=1, max_wall_seconds=10, per_attempt_deadline=5,
        worker_argv_fn=lambda seed, result_dir: ["python", "-c", "import sys; sys.exit(3)"],
    )
    identity = json.loads((out / "source_identity.json").read_text())
    report = json.loads((out / "report.json").read_text())

    assert identity["git_head_is_code_identity"] is False
    assert "source_identity_sha256" not in identity
    rel = "examples/baselines/constrain/pickcube_transport_detour.py"
    assert identity["files"][rel] == hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    assert "examples/baselines/constrain/pickcube_expert.py" in identity["files"]
    assert "source_identity.json" not in identity["files"]
    assert report["source_files_sha256"] == identity["files_sha256"]
    assert report["source_identity_sha256"] == hashlib.sha256((out / "source_identity.json").read_bytes()).hexdigest()


def test_continuous_join_execution_is_smoother_than_its_polyline_corner():
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env, run_pickcube_expert
    from examples.baselines.constrain.pickcube_transport_detour import measure_executed_join

    env = make_pickcube_env(obstacle_radius=0.03, obstacle_height=0.05)
    try:
        result = run_pickcube_expert(
            env, cube_xy=(-0.12, 0.07), goal_xy=(0.12, 0.07), detour=True,
            waypoint_xy=(0.0, 0.24), obstacle_enabled=False, continuous_join=True,
            record_body_sweep=True,
        )
        targets = result["trace"]["shared_targets"]
        corner = _turn_deg(targets["lift_xyz"][:2], targets["waypoint_xyz"][:2], targets["transport_exit_xyz"][:2])
        join = measure_executed_join(
            result["trace"]["tcp_xyz"], result["actions"], result["trace"]["step_phase"],
        )
        phases = [name for name, _status in result["trace"]["phase_status"]]
        assert result["safety"]["native_success"] and result["safety"]["held_at_success"]
        assert targets["retreat_required"] is False
        assert "transport" in phases and "goal_entry" in phases and "settle" in phases
        assert corner > 90.0
        assert join.continuous
        assert join.max_commanded_heading_step_deg <= 45.0
        assert join.max_tcp_heading_step_deg <= 45.0
        assert join.max_commanded_heading_step_deg + 20.0 < corner
    finally:
        env.close()


def _turn_deg(start, corner, end):
    start = np.asarray(start, dtype=np.float64)
    corner = np.asarray(corner, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    incoming = corner - start
    outgoing = end - corner
    cosine = float(np.dot(incoming, outgoing) / (np.linalg.norm(incoming) * np.linalg.norm(outgoing)))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _no_lingering_sleep():
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if text.strip() == "sleep 30":
            return False
    return True
