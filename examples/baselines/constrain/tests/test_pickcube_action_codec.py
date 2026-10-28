import pytest
import torch

from examples.baselines.constrain.pickcube_action_codec import (
    CONTROLLER_RANGE, decode_prediction_to_native, decode_to_native, encode_native, profile_sha256,
)


def test_pickcube_codec_round_trips_native_controller_coordinates():
    native = torch.tensor([[0.35, -0.175, 0.0, -0.35, 0.07, 0.21, -1.0]])
    encoded = encode_native(native)
    torch.testing.assert_close(encoded, torch.tensor([[1.0, -0.5, 0.0, -1.0, 0.2, 0.6, -1.0]]))
    torch.testing.assert_close(decode_to_native(encoded), native)


def test_pickcube_codec_rejects_native_bounds_and_profile_mismatch():
    with pytest.raises(ValueError, match="bounds"):
        encode_native(torch.tensor([[0.3501, 0, 0, 0, 0, 0, 0.0]]))
    with pytest.raises(ValueError, match="mismatched"):
        encode_native(torch.zeros(1, 7), dict(CONTROLLER_RANGE, name="identity_native_v1"))
    assert len(profile_sha256()) == 64


def test_pickcube_codec_decodes_without_clipping_model_predictions():
    decoded = decode_to_native(torch.tensor([[1.2, 0, 0, 0, 0, 0, -1.0]]))
    assert decoded[0, 0].item() == pytest.approx(0.42)


def test_pickcube_actor_rejects_decoded_predictions_outside_native_bounds_without_clipping():
    with pytest.raises(ValueError, match="native controller bounds"):
        decode_prediction_to_native(torch.tensor([[3.0, 0, 0, 0, 0, 0, 0.0]]))
