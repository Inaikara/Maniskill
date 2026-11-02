import numpy as np
import pytest

from examples.baselines.constrain.initial_clear_xy_scenes import initial_tcp_admission


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
