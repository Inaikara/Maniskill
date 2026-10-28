import numpy as np
import pytest
import torch

from examples.baselines.constrain.pickcube_base_dp import NAMED_FIELDS
from examples.baselines.constrain.pickcube_position_units import (
    OBS_DIM, PROFILE, PROFILE_SHA256, SCALE_VECTOR, encode_observation,
    position_indices, profile_metadata, validate_profile,
)
from examples.baselines.constrain.pickcube_observation_norm import fit_direct_observations
from examples.baselines.constrain.pickcube_constrain import (
    PickCubeConditionNorm, pickcube_relative_condition,
)


def test_fixed_encoding_scales_only_named_meter_positions_by_ten():
    raw = torch.arange(90, dtype=torch.float32).reshape(2, 45)[:, :OBS_DIM]
    encoded = encode_observation(raw)
    expected = raw.clone()
    indices = position_indices()
    expected[:, list(indices)] *= 10.0
    torch.testing.assert_close(encoded, expected, rtol=0, atol=0)
    unchanged = sorted(set(range(OBS_DIM)) - set(indices))
    assert torch.equal(encoded[:, unchanged], raw[:, unchanged])
    assert len(indices) == 15
    assert [i for i, scale in enumerate(SCALE_VECTOR) if scale == 10.0] == list(indices)


def test_profile_is_fixed_schema_versioned_and_not_data_fitted():
    metadata = profile_metadata()
    assert metadata["profile"] == PROFILE == "pickcube_position_controller_units_v1"
    assert metadata["position_reference_m"] == 0.1
    assert metadata["sha256"] == PROFILE_SHA256
    assert metadata["offset"] == [0.0] * 42
    assert not {"mean", "std", "population_std", "minimum", "rms"}.intersection(metadata)
    assert validate_profile(metadata) is metadata
    wrong = dict(metadata, scale=[1.0] * 42)
    with pytest.raises(ValueError, match="profile/schema/hash"):
        validate_profile(wrong)


def test_fixed_profile_rejects_legacy_fitted_zscore_metadata():
    legacy = fit_direct_observations(
        {i: np.zeros((1, 45), dtype=np.float32) for i in range(1000)}
    )
    with pytest.raises(ValueError, match="profile/schema/hash"):
        validate_profile(legacy)


def test_encoding_rejects_wrong_dimension_or_named_schema():
    with pytest.raises(ValueError, match="42 features"):
        encode_observation(torch.zeros(45))
    with pytest.raises(ValueError, match="frozen named schema"):
        encode_observation(torch.zeros(42), fields=NAMED_FIELDS[:-1])


def test_relative_obstacle_hint_remains_in_raw_world_meters():
    raw = torch.zeros((2, 45), dtype=torch.float32)
    raw[:, 19:21] = torch.tensor([0.1, 0.2])
    raw[:, 42:45] = torch.tensor([0.4, 0.3, 0.03])
    hint = pickcube_relative_condition(raw, PickCubeConditionNorm())
    np.testing.assert_allclose(hint.numpy(), [0.6, 0.2, 0.375], atol=1e-6)
