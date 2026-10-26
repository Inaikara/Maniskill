from examples.baselines.constrain.audit_placesphere_matched_windows import audit


DATA = "demos/PlaceSphere-transport-detour/20261026_133000_matched_terminal_r21b/trajectory.h5"


def test_matched_settle_windows_separate_linear_static_from_angular_failure():
    report = audit(DATA)

    direct = report["roles"]["direct"]
    obstacle = report["roles"]["obstacle"]
    assert direct["fixed_stride_8_training_trajectory_windows"] == 16189
    assert obstacle["fixed_stride_8_training_trajectory_windows"] == 6175
    assert direct["action_only_open_near_zero_windows"] == 3311
    assert obstacle["action_only_open_near_zero_windows"] == 1513
    assert direct["linear_static_but_angular_only_failure_windows"] == 3301
    assert obstacle["linear_static_but_angular_only_failure_windows"] == 1506
    assert direct["both_native_static_within_action_only_windows"] == 0
    assert obstacle["both_native_static_within_action_only_windows"] == 0
    assert report["roles"]["direct"]["per_phase"]["settle"]["action_only_open_near_zero_windows"] == 3311
    assert report["roles"]["obstacle"]["per_phase"]["settle"]["action_only_open_near_zero_windows"] == 1513
