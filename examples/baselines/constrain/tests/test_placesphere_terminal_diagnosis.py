from examples.baselines.constrain.placesphere_terminal_diagnosis import classify_motion


def test_sphere_static_failure_distinguishes_linear_motion_from_spin():
    assert classify_motion(0.012, 0.10)["cause"] == "linear"
    assert classify_motion(0.004, 0.51)["cause"] == "angular"
    assert classify_motion(0.012, 0.51)["cause"] == "linear_and_angular"
    assert classify_motion(0.004, 0.49)["cause"] == "within_native_thresholds"


def test_native_speed_threshold_boundaries_are_inclusive():
    result = classify_motion(0.01, 0.5)
    assert result == {
        "linear_over_threshold": False,
        "angular_over_threshold": False,
        "cause": "within_native_thresholds",
    }
