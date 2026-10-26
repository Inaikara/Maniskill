import pytest

from examples.baselines.constrain.placesphere_matched_hold_probe import (
    HOLDS,
    TRAIN_OBSTACLE_IDS,
    choose_physics_hold,
)


def _summaries(safe, speed):
    return {str(hold): {"count": 8, "safe_native_success_count": safe[hold],
                        "terminal_speed_pass_count": speed[hold]}
            for hold in HOLDS}


def test_hold_selection_uses_only_frozen_train_physics_counts_then_shortest_hold():
    assert TRAIN_OBSTACLE_IDS == (1, 3, 5, 7, 9, 11, 13, 15)
    assert choose_physics_hold(_summaries({0: 5, 8: 7, 16: 7}, {0: 5, 8: 7, 16: 8})) == 16
    assert choose_physics_hold(_summaries({0: 8, 8: 8, 16: 8}, {0: 8, 8: 8, 16: 8})) == 0


def test_hold_selection_refuses_incomplete_candidate_denominators():
    rows = _summaries({0: 8, 8: 8, 16: 8}, {0: 8, 8: 8, 16: 8})
    rows["8"]["count"] = 7
    with pytest.raises(ValueError, match="complete"):
        choose_physics_hold(rows)
