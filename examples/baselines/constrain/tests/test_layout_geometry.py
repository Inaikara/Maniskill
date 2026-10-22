import numpy as np
import pytest

from examples.baselines.constrain.layout_geometry import (
    LayoutGeometryConfig,
    sample_same_side_waypoint,
    validate_layout,
)


def test_layout_accepts_inclusive_distance_and_table_boundaries():
    config = LayoutGeometryConfig(
        table_bounds=(-1.0, -1.0, 1.0, 1.0),
        min_transport_distance=1.0,
        min_base_line_distance=1.0,
    )

    result = validate_layout((0.0, 0.0), (1.0, 0.0), (0.0, -1.0), config)

    assert result.valid
    assert result.reason is None
    assert result.transport_distance == pytest.approx(1.0)
    assert result.base_line_distance == pytest.approx(1.0)


@pytest.mark.parametrize(
    "start,goal,expected_reason",
    [
        ((0.0, 0.0), (0.0, 0.0), "transport_distance"),
        ((0.0, 0.0), (0.5, 0.0), "transport_distance"),
        ((-1.01, 0.0), (0.0, 0.0), "table_bounds"),
    ],
)
def test_layout_rejects_degenerate_short_and_out_of_bounds_segments(
    start, goal, expected_reason
):
    config = LayoutGeometryConfig(
        table_bounds=(-1.0, -1.0, 1.0, 1.0),
        min_transport_distance=1.0,
        min_base_line_distance=0.1,
    )

    result = validate_layout(start, goal, (0.0, -1.0), config)

    assert not result.valid
    assert result.reason == expected_reason


def test_rotated_layout_uses_oriented_line_side_and_workspace():
    config = LayoutGeometryConfig(
        table_bounds=(-2.0, -2.0, 2.0, 2.0),
        min_transport_distance=1.0,
        min_base_line_distance=0.5,
    )
    start = np.array([0.0, 0.0])
    goal = np.array([1.0, 1.0])
    base = np.array([0.0, 1.0])

    result = validate_layout(start, goal, base, config)
    sample = sample_same_side_waypoint(
        start, goal, base, np.random.default_rng(2), config
    )

    assert result.valid
    assert sample.waypoint_xy is not None
    waypoint = sample.waypoint_xy
    assert (goal - start)[0] * (waypoint[1] - start[1]) - (
        goal - start
    )[1] * (waypoint[0] - start[0]) > 0


def test_waypoint_sampler_never_falls_back_to_opposite_side_and_is_bounded():
    config = LayoutGeometryConfig(
        table_bounds=(-2.0, -2.0, 2.0, 0.0),
        min_transport_distance=1.0,
        min_base_line_distance=0.1,
        waypoint_offset_range=(0.2, 0.2),
        waypoint_lead_fraction=0.0,
        max_attempts=3,
    )

    result = sample_same_side_waypoint(
        (0.0, 0.0), (1.0, 0.0), (0.5, 1.0), np.random.default_rng(0), config
    )

    assert result.waypoint_xy is None
    assert result.attempts == 3
    assert result.reason == "same_side_waypoint_space_exhausted"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"table_bounds": (1.0, 0.0, -1.0, 1.0)},
        {"min_transport_distance": -1.0},
        {"min_base_line_distance": 0.0},
        {"waypoint_offset_range": (0.2, 0.1)},
        {"max_attempts": 0},
    ],
)
def test_layout_config_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        LayoutGeometryConfig(**kwargs)
