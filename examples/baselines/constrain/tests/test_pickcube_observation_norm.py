import numpy as np
import pytest
import torch

from examples.baselines.constrain.pickcube_observation_norm import (
    FLOOR, fit_direct_observations, normalize_observation, validate_stats,
)


def test_fit_uses_only_exact_direct_t_plus_one_frames_and_applies_floor_without_clipping():
    episodes = {}
    for i in range(1000):
        pair = np.zeros((2, 45), np.float32)
        pair[:, 0] = (i, i + 2)
        episodes[i] = pair
    stats = fit_direct_observations(episodes)
    assert stats["fit_frame_count"] == 2000
    assert stats["mean"][0] == pytest.approx(500.5)
    assert stats["population_std"][0] == pytest.approx(np.std(np.concatenate([
        np.array([i, i + 2], dtype=np.float64) for i in range(1000)
    ])))
    assert len(stats["floored_fields"]) == 41
    assert "agent/qpos[0]" not in stats["floored_fields"]
    assert "extra/is_grasped[0]" in stats["floored_fields"]
    normalized = normalize_observation(torch.full((42,), 10000.0), stats)
    assert torch.isfinite(normalized).all() and normalized.max() > 1
    with pytest.raises(ValueError, match="exactly direct"):
        fit_direct_observations({**episodes, 1000: np.zeros((2, 45))})
    with pytest.raises(ValueError, match="exactly direct"):
        fit_direct_observations(episodes, list(range(1, 1001)))


def test_profile_rejects_tampered_or_nonfinite_statistics():
    stats = fit_direct_observations({i: np.zeros((1, 45), np.float32) for i in range(1000)})
    validate_stats(stats)
    bad = dict(stats, mean=[float("nan")] * 42)
    with pytest.raises(ValueError, match="finite"):
        validate_stats(bad)
    bad = dict(stats, mean=[1.0] * 42)
    with pytest.raises(ValueError, match="hash"):
        validate_stats(bad)
    assert FLOOR == 0.01
