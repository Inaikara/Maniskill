import pytest
import torch

from examples.baselines.constrain.placesphere_action_codec import (
    CONTROLLER_RANGE, IDENTITY, decode_to_native, encode_native, profile_sha256,
    validate_profile,
)
from examples.baselines.constrain.placesphere_base_dp import contract
from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter


def test_controller_range_roundtrip_preserves_native_pose_gripper_and_padding():
    native = torch.tensor([
        [.35, -.35, .2, .1, -.3, 0., -1.],
        [0., 0., 0., 0., 0., 0., 1.],
    ])
    encoded = encode_native(native, CONTROLLER_RANGE)
    torch.testing.assert_close(encoded[0], torch.tensor([1., -1., .2/.35, .1/.35, -3/3.5, 0., -1.]))
    torch.testing.assert_close(decode_to_native(encoded, CONTROLLER_RANGE), native)
    assert encoded[1, :6].count_nonzero() == 0
    assert encoded[1, 6] == 1


@pytest.mark.parametrize("actions", [
    torch.zeros(2, 6),
    torch.tensor([[.35001, 0, 0, 0, 0, 0, 0.]]),
    torch.tensor([[0., 0, 0, 0, 0, 0, 1.00001]]),
    torch.tensor([[float("nan"), 0, 0, 0, 0, 0, 0.]]),
])
def test_codec_rejects_malformed_or_out_of_registered_bounds(actions):
    with pytest.raises(ValueError):
        encode_native(actions, CONTROLLER_RANGE)


def test_profiles_are_versioned_fixed_and_legacy_contract_stays_identity():
    assert validate_profile(IDENTITY) is IDENTITY
    assert contract() == contract(IDENTITY)
    assert profile_sha256(CONTROLLER_RANGE) != profile_sha256(IDENTITY)
    changed = dict(CONTROLLER_RANGE, scale=[.34] * 6 + [1.])
    with pytest.raises(ValueError, match="parameters"):
        validate_profile(changed)
    with pytest.raises(ValueError, match="unknown or missing"):
        validate_profile(None)


def test_identity_default_codec_is_bit_exact_and_encoded_dimensions_stay_seven():
    actions = torch.randn(2, 16, 7)
    actions[0, 0, 0] = -0.0
    assert torch.equal(encode_native(actions, IDENTITY), actions)
    assert torch.equal(decode_to_native(actions, IDENTITY), actions)
    assert torch.equal(torch.signbit(encode_native(actions, IDENTITY)), torch.signbit(actions))
    assert torch.equal(torch.signbit(decode_to_native(actions, IDENTITY)), torch.signbit(actions))


def test_base_adapter_decodes_encoded_prediction_before_native_controller_clamp():
    class Agent:
        def get_action(self, _obs):
            return torch.ones(1, 8, 7)

    adapter = object.__new__(PlaceSphereBaseDPAdapter)
    adapter.agent = Agent()
    adapter.device = torch.device("cpu")
    adapter.action_encoding = CONTROLLER_RANGE
    adapter.sampler = "ddpm"
    native = adapter.get_action(torch.zeros(1, 2, 39))
    torch.testing.assert_close(native, torch.tensor([[[.35] * 6 + [1.]] * 8]))
    mapped = adapter.map_action_ee_delta_pose(native)
    torch.testing.assert_close(mapped, native)
