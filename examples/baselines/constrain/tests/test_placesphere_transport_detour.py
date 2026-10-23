from pathlib import Path

import numpy as np
import pytest
import torch

from examples.baselines.constrain.placesphere_transport_detour import (
    BIN_DISTANCE_WINDOW_M,
    classify_clearances,
    classify_direct_block,
    endpoint_column_clear,
    endpoint_sweep_points,
    generate,
    same_side_waypoint,
    select_cylinder_height,
    shared_transport_fork,
)


def test_height_covers_measured_sphere_and_gripper_not_a_prescribed_cylinder():
    decision = select_cylinder_height(
        np.array([[0.0, 0.0, 0.12]]),
        np.array([[0.0, 0.0, 0.19]]),
        np.array([[0.20, 0.0, 0.30]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert decision.accepted
    assert decision.height_m == pytest.approx(0.20)
    assert decision.height_m not in (0.08, 0.25)
    assert decision.sphere_inside_max_z_m == pytest.approx(0.12)
    assert decision.gripper_inside_max_z_m == pytest.approx(0.19)
    assert decision.reason is None


def test_side_or_endpoint_column_occupancy_is_not_a_feasible_height():
    side = select_cylinder_height(
        np.array([[0.0, 0.0, 0.12]]),
        np.array([[0.0, 0.0, 0.19]]),
        np.array([[0.0, 0.0, 0.10]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )
    end = select_cylinder_height(
        np.array([[0.0, 0.0, 0.12]]),
        np.array([[0.0, 0.0, 0.19]]),
        np.array([[0.20, 0.0, 0.30]]),
        np.array([[0.01, 0.0, 0.10]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert not side.accepted and side.reason == "side_detour_occupies_cylinder_column"
    assert not end.accepted and end.reason == "end_region_sweep_occupies_cylinder_column"


def test_infeasible_gripper_top_falls_back_to_the_measured_sphere_cover():
    decision = select_cylinder_height(
        np.array([[0.0, 0.0, 0.12]]),
        np.array([[0.0, 0.0, 0.25]]),
        np.array([[0.0, 0.0, 0.20]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert decision.accepted
    assert decision.height_m == pytest.approx(0.13)
    assert decision.gripper_inside_max_z_m == pytest.approx(0.25)
    assert decision.gripper_top_cover_feasible is False


def test_direct_transport_that_misses_the_column_is_not_blocked():
    decision = select_cylinder_height(
        np.array([[0.20, 0.0, 0.16]]),
        np.array([[0.20, 0.0, 0.22]]),
        np.array([[0.20, 0.0, 0.30]]),
        np.array([[0.25, 0.0, 0.05]]),
        center_xy=(0.0, 0.0),
        radius=0.03,
    )

    assert not decision.accepted
    assert decision.reason == "direct_transport_misses_cylinder"


def test_clearance_keeps_vertex_gap_sphere_margin_and_surface_apart():
    report = classify_clearances(
        vertex_xy_distance_m=0.08,
        radius_m=0.03,
        sphere_center_distance_m=0.12,
        forbidden_radius_m=0.055,
        contact_separation_m=None,
        contact_free=True,
    )

    assert report.geometric_vertex_clearance_m == pytest.approx(0.05)
    assert report.geometric_is_surface_clearance is False
    assert report.object_xy_margin_m == pytest.approx(0.065)
    assert report.robot_surface_clearance_m is None
    assert report.safety_proof == "contact_free_execution"


def test_same_side_waypoint_does_not_fall_back_when_that_side_is_out_of_bounds():
    inside = same_side_waypoint((0.0, 0.0), (1.0, 0.0), (0.5, -1.0), 0.20, (-2.0, -2.0, 2.0, 2.0))
    blocked = same_side_waypoint((0.0, 0.0), (1.0, 0.0), (0.5, -1.0), 0.20, (-0.1, -0.05, 1.1, 0.05))

    assert inside[1] < 0.0
    assert blocked is None


def test_fork_shares_container_release_retreat_settle_and_only_the_detour_has_a_waypoint():
    join = {"measured": True, "max_commanded_heading_step_deg": 8.0,
            "max_tcp_heading_step_deg": 9.0, "transport_steps": 8}
    direct = _fork(join, waypoint=None)
    detour = _fork(join, waypoint=[0.5, -0.2, 0.16])

    ok, reason = shared_transport_fork(direct, detour)

    assert ok and reason is None


def test_fork_rejects_a_held_release_and_an_unmeasured_join():
    join = {"measured": True, "max_commanded_heading_step_deg": 8.0,
            "max_tcp_heading_step_deg": 9.0, "transport_steps": 8}
    direct = _fork(join, waypoint=None)
    detour = _fork(join, waypoint=[0.5, -0.2, 0.16])
    detour["terminal"]["ungrasped"] = False
    detour["terminal"]["grasped"] = True

    ok, reason = shared_transport_fork(direct, detour)

    assert not ok and reason == "native_success_not_released"

    detour["terminal"]["ungrasped"] = True
    detour["terminal"]["grasped"] = False
    detour["join"] = {"measured": False}
    ok, reason = shared_transport_fork(direct, detour)
    assert not ok and reason == "join_not_measured"


def test_endpoint_sweep_keeps_near_bin_settle_vertices_and_both_paths():
    assert BIN_DISTANCE_WINDOW_M is None
    sweep = [
        {"phase": "settle", "bodies": {"panda_hand": np.array([[0.0, 0.0, 0.05]])}},
        {"phase": "transport", "bodies": {"sphere": np.array([[0.0, 0.0, 0.16]])}},
        {"phase": "release", "bodies": {"panda_leftfinger": np.array([[0.02, 0.0, 0.08]])}},
    ]
    points = endpoint_sweep_points(sweep)

    assert any(np.allclose(point, [0.0, 0.0, 0.05]) for point in points)
    assert any(np.allclose(point, [0.02, 0.0, 0.08]) for point in points)
    assert not any(np.allclose(point, [0.0, 0.0, 0.16]) for point in points)
    ok, reason = endpoint_column_clear(
        points, np.array([[0.20, 0.0, 0.05]]), center_xy=(0.0, 0.0), radius=0.03, height=0.15,
    )
    assert not ok and reason == "direct_endpoint_occupies_cylinder_column"
    ok, reason = endpoint_column_clear(
        np.zeros((0, 3)), points, center_xy=(0.0, 0.0), radius=0.03, height=0.15,
    )
    assert not ok and reason == "endpoint_sweep_incomplete"


def test_transport_contact_blocks_and_release_contact_does_not():
    ok, reason = classify_direct_block({"phase": "transport"})
    assert ok and reason is None
    ok, reason = classify_direct_block({"phase": "release"})
    assert not ok and reason == "direct_block_contact_in_endpoint"
    ok, reason = classify_direct_block({"phase": "retreat"})
    assert not ok and reason == "direct_block_contact_in_endpoint"


def test_overtop_sphere_still_crosses_the_xy_disk():
    from examples.baselines.constrain.placesphere_obstacle import OBSTACLE_KEY
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    env = make_env(config=PlaceSpherePilotConfig(max_steps=20, obstacle_height=0.15))
    try:
        env.reset(seed=3, options={
            "obstacle_enabled": True, "obstacle_center": [0.0, 0.1],
            "sphere_xy": [-0.08, 0.1], "container_xy": [0.10, 0.1],
        })
        base = env.unwrapped
        state = base.get_state_dict()
        state["actors"]["sphere"][0, 0] = 0.08
        state["actors"]["sphere"][0, 1] = 0.1
        state["actors"]["sphere"][0, 2] = 0.30
        state[OBSTACLE_KEY][0, 6] = 1
        state[OBSTACLE_KEY][0, 7:9] = torch.tensor([-0.08, 0.1])
        base.set_state_dict(state)
        _, _, terminated, truncated, info = env.step(torch.zeros(7))
        assert not bool(terminated)
        assert not bool(truncated)
        assert float(base.obj.pose.p[0, 2]) > float(base.obstacle_height)
        assert bool(info["obstacle_crossing"].reshape(-1)[0])
    finally:
        env.close()


def test_deadline_kills_the_worker_and_counts_the_attempt(tmp_path):
    report = generate(
        tmp_path / "batch", seed=1, target=1, max_attempts=1, max_wall_seconds=10,
        per_attempt_deadline=0.3, worker_argv_fn=lambda seed, result_dir: ["sleep", "30"],
    )

    assert report["attempts"] == 1
    assert report["accepted"] == 0
    assert report["failures"] == {"attempt_deadline": 1}


def test_nonpositive_budget_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate(tmp_path, target=1, max_attempts=1, max_wall_seconds=0, per_attempt_deadline=1)


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
    rel = "examples/baselines/constrain/placesphere_transport_detour.py"

    assert identity["git_head_is_code_identity"] is False
    assert "source_identity_sha256" not in identity
    assert identity["files"][rel] == hashlib.sha256(Path(rel).read_bytes()).hexdigest()
    assert "source_identity.json" not in identity["files"]
    assert report["source_files_sha256"] == identity["files_sha256"]
    assert report["source_identity_sha256"] == hashlib.sha256((out / "source_identity.json").read_bytes()).hexdigest()


def _fork(join, waypoint):
    phases = [("lift", "reached"), ("transport", "reached"), ("container_approach", "reached"),
              ("release", "reached"), ("retreat", "reached"), ("settle", "reached")]
    if waypoint is not None:
        phases.insert(1, ("waypoint", "reached"))
    return {
        "phases": phases,
        "targets": {
            "lift_xyz": [0.0, 0.0, 0.16],
            "transport_exit_xyz": [0.4, 0.0, 0.16],
            "container_approach_xyz": [0.4, 0.0, 0.11],
            "retreat_xyz": [0.4, 0.0, 0.19],
            "waypoint_xyz": waypoint,
            "retreat_required": True,
            "release_required": True,
        },
        "join": join,
        "terminal": {"on_bin": True, "static": True, "ungrasped": True, "grasped": False},
    }
