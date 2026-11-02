import numpy as np
import pytest

from examples.baselines.constrain.initial_clear_xy_scenes import initial_tcp_admission, sample_layout


def test_admission_requires_finite_measured_tcp_strictly_outside_profile():
    result = initial_tcp_admission([0.035001, 0.0], [0.0, 0.0], 0.03, 0.005)
    assert result["accepted"] is True
    assert result["signed_initial_gap_m"] == pytest.approx(1e-6)
    assert initial_tcp_admission([0.035, 0], [0, 0], .03, .005)["reason"] == "boundary"
    assert initial_tcp_admission([0.02, 0], [0, 0], .03, .005)["reason"] == "inside"
    assert initial_tcp_admission([np.nan, 0], [0, 0], .03, .005)["reason"] == "non_finite"
    assert initial_tcp_admission(None, [0, 0], .03, .005)["reason"] == "missing_measured_tcp"


def test_admission_rejects_grazing_geometry_and_invalid_profile():
    assert initial_tcp_admission([.1, 0], [0, 0], .03, .005,
                                  chord_start=[-.1, 0])["reason"] == "grazing_chord"
    with pytest.raises(ValueError, match="radius"):
        initial_tcp_admission([.1, 0], [0, 0], -1, .005)


def test_native_reset_goal_cannot_leak_into_pick_layout_and_training_bounds_are_restored():
    from examples.baselines.constrain.pickcube_transport_detour import TABLE_BOUNDS

    # The native goal reset can be anywhere; sampler draws explicitly from the
    # original layout distribution and fixes the original goal height.
    accepted = 0
    for seed in range(1000):
        try:
            start, goal, goal_z, checked = sample_layout(
                "pickcube", [-.6, 0], np.random.default_rng(seed))
        except ValueError:
            continue
        assert -.18 <= start[0] <= -.02 and -.06 <= start[1] <= .12
        assert .02 <= goal[0] <= .18 and -.06 <= goal[1] <= .12
        assert goal_z == .12
        assert checked.transport_distance >= .14
        assert checked.base_line_distance >= .06
        assert TABLE_BOUNDS == (-.25, -.25, .25, .35)
        accepted += 1
        if accepted == 100:
            break
    assert accepted == 100


def test_placesphere_layout_keeps_original_container_footprint_and_transport_contract():
    accepted = 0
    for seed in range(1000):
        try:
            start, goal, goal_z, checked = sample_layout(
                "placesphere", [-.6, 0], np.random.default_rng(seed),
                container_half_extent=.04, native_goal_z=.08)
        except ValueError:
            continue
        assert -.19 <= start[0] <= -.08 and -.13 <= start[1] <= .13
        assert .04 <= goal[0] <= .15 and -.13 <= goal[1] <= .13
        assert -.21 <= goal[0] <= .21 and -.21 <= goal[1] <= .21
        assert goal_z == .08
        assert checked.transport_distance >= .14
        assert checked.base_line_distance >= .06
        accepted += 1
        if accepted == 100:
            break
    assert accepted == 100
