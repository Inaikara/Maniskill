"""Seam for whether the existing expert descent is a recovery demo."""

import numpy as np

from examples.baselines.constrain.placesphere_recovery_probe import probe_decision


def _descent():
    action = np.zeros(7)
    action[2] = -0.35
    action[6] = -1.0
    return [action.copy() for _ in range(8)]


def test_same_closed_descent_does_not_authorize_collection():
    decision = probe_decision(_descent(), _descent(), perturbed_contact=False, endpoint_open=False)
    assert decision["same_closed_descent"] is True
    assert decision["collect_this_expert"] is False
    assert decision["reward_claim"] is False
    assert decision["reason"] == "same_closed_descent"


def test_contact_during_perturbation_does_not_authorize_collection():
    other = _descent()
    other[0] = other[0].copy()
    other[0][6] = 1.0
    decision = probe_decision(other, other, perturbed_contact=True, endpoint_open=True)
    assert decision["collect_this_expert"] is False
    assert decision["reason"] == "perturbed_contact"
