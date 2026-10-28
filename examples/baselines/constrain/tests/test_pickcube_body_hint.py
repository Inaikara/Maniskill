import torch

from examples.baselines.constrain.pickcube_body_hint import (
    BodyHintNorm,
    body_hint_from_raw,
)
from examples.baselines.constrain.pickcube_constrain import pickcube_relative_condition


def sample_observation():
    raw = torch.zeros(2, 45)
    raw[:, 19:22] = torch.tensor([0.2, -0.1, 0.4])
    raw[:, 22] = 1.0
    raw[:, 42:45] = torch.tensor([0.3, 0.2, 0.03])
    return raw


def test_hint9_preserves_hint3_and_computes_world_body_points():
    raw = sample_observation()
    result = body_hint_from_raw(raw)
    old = pickcube_relative_condition(raw[None])
    torch.testing.assert_close(result[:3], old[0])
    # Identity TCP rotation: hand and link7 are fixed offsets along TCP -Z.
    torch.testing.assert_close(result[3:6], torch.tensor([0.2, 0.6, 0.5932]))
    torch.testing.assert_close(result[6:9], torch.tensor([0.2, 0.6, 0.3792]))
    assert result.shape == (9,)


def test_body_points_are_quaternion_sign_and_global_xy_gauge_invariant():
    raw = sample_observation()
    base = body_hint_from_raw(raw)
    negated = raw.clone(); negated[:, 22:26] *= -1
    torch.testing.assert_close(body_hint_from_raw(negated), base)
    translated = raw.clone(); translated[:, [19, 20, 42, 43]] += torch.tensor([0.017, -0.012, 0.017, -0.012])
    torch.testing.assert_close(body_hint_from_raw(translated), base)


def test_wrist_fixed_frame_rotation_changes_xy_point_but_not_transform_or_z_semantics():
    raw = sample_observation()
    raw[:, 22:26] = torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0])
    result = body_hint_from_raw(raw)
    # Rotation about world X maps the TCP-local negative-Z offsets into +Y.
    torch.testing.assert_close(result[3:6], torch.tensor([0.2, 0.3932, 0.8]))
    torch.testing.assert_close(result[6:9], torch.tensor([0.2, 0.1792, 0.8]))


def test_legacy_hint3_contract_remains_default_and_zero_quaternion_is_rejected():
    from examples.baselines.constrain.pickcube_constrain import (
        MODEL_CONFIG, PickCubeConditionNorm,
    )

    assert MODEL_CONFIG["cond_dim"] == 3
    raw = sample_observation()
    legacy = pickcube_relative_condition(raw[None])
    torch.testing.assert_close(body_hint_from_raw(raw)[None, :3], legacy)
    raw[:, 22:26] = 0
    try:
        body_hint_from_raw(raw)
    except ValueError as exc:
        assert "zero norm" in str(exc)
    else:
        raise AssertionError("zero TCP quaternion must be rejected")
    assert PickCubeConditionNorm().to_dict()["definition"] == (
        "[obstacle_x-tcp_x, obstacle_y-tcp_y, cylinder_radius]")


def test_hint_normalizer_pins_feature_order_and_frames():
    value = BodyHintNorm().to_dict()
    assert len(value["feature_order"]) == 9
    assert BodyHintNorm.from_dict(value).to_dict() == value
    value["T_tcp_hand"][2][3] += 0.001
    try:
        BodyHintNorm.from_dict(value)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched fixed transforms must be rejected")


def test_r41_checkpoint_profile_is_typed_and_rejects_legacy_three_channel(tmp_path):
    from examples.baselines.constrain.pickcube_body_hint_round41 import (
        CONTROL_CONFIG, CONTROL_FORMAT, load_body_control,
    )
    from examples.baselines.constrain.pickcube_constrain import MODEL_CONFIG

    assert CONTROL_CONFIG["cond_dim"] == 9
    assert CONTROL_FORMAT == "pickcube_body_hint9_constrain_v1"
    assert MODEL_CONFIG["cond_dim"] == 3
    legacy = tmp_path / "legacy3.pt"
    torch.save({"format": "pickcube_direct_zscore_constrain_v1", "config": {"model": dict(CONTROL_CONFIG, cond_dim=3)}}, legacy)
    try:
        load_body_control(legacy, stats={})
    except ValueError as exc:
        assert "contract mismatch" in str(exc)
    else:
        raise AssertionError("legacy three-channel checkpoints must not load as R41")
