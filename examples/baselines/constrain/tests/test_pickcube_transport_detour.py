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
    direct = {
        "phases": [("lift", "reached"), ("goal", "episode_ended")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14],
            "goal_xyz": [1.0, 0.0, 0.12],
            "waypoint_xyz": None,
        },
    }
    detour = {
        "phases": [("lift", "reached"), ("waypoint", "reached"), ("goal", "episode_ended")],
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.14],
            "goal_xyz": [1.0, 0.0, 0.12],
            "waypoint_xyz": [0.5, 0.2, 0.14],
        },
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
    assert reason == "detour_not_side_continuous"


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


def _no_lingering_sleep():
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            text = cmdline.read_bytes().replace(b"\x00", b" ").decode()
        except OSError:
            continue
        if text.strip() == "sleep 30":
            return False
    return True
