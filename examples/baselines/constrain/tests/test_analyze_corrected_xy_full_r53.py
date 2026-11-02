import pytest

from examples.baselines.constrain.analyze_corrected_xy_full_r53 import validate_trace_geometry


def test_trace_geometry_must_match_explicit_catalog_geometry():
    trace = [{"obstacle_center_xy": [0.12, -0.03], "forbidden_radius_m": 0.035}]
    validate_trace_geometry(trace, [0.12, -0.03], 0.035)
    with pytest.raises(ValueError, match="catalog geometry"):
        validate_trace_geometry(trace, [0.13, -0.03], 0.035)
