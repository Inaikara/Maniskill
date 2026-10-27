from examples.baselines.constrain.placesphere_ensemble_r29_analysis import transition


def test_transition_counts_are_ordered_and_use_requested_adverse_orientation():
    old = [{"result": {"native_success": value, "obstacle_contact": value}}
           for value in (False, True, True, False)]
    new = [{"result": {"native_success": value, "obstacle_contact": value}}
           for value in (True, False, True, False)]
    assert transition(old, new, "native_success", adverse=False) == {
        "improved": 1, "regressed": 1, "both": 1, "neither": 1}
    assert transition(old, new, "obstacle_contact", adverse=True) == {
        "cleared": 1, "introduced": 1, "both": 1, "neither": 1}
