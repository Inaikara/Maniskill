import pytest
import torch
from types import SimpleNamespace

from examples.baselines.constrain.pickcube_residual_guidance import (
    ResidualGuidanceAdapter,
    combine_epsilon,
    validate_guidance_scale,
)


def test_residual_guidance_uses_exact_registered_epsilon_formula():
    base = torch.tensor([[1.0, -2.0, 0.25]])
    control = torch.tensor([[3.0, 2.0, -0.75]])
    torch.testing.assert_close(
        combine_epsilon(base, control, 1.5),
        torch.tensor([[4.0, 4.0, -1.25]]), rtol=0, atol=0,
    )


def test_scale_zero_and_one_return_the_original_tensors_exactly():
    base = torch.tensor([0.1234567])
    control = torch.tensor([0.7654321])
    assert combine_epsilon(base, control, 0.0) is base
    assert combine_epsilon(base, control, 1.0) is control


@pytest.mark.parametrize("scale", [float("nan"), float("inf"), -0.1, 2.01])
def test_guidance_scale_rejects_unregistered_invalid_values(scale):
    with pytest.raises(ValueError):
        validate_guidance_scale(scale)


def test_residual_adapter_uses_identical_inputs_and_does_not_draw_extra_rng(monkeypatch):
    from examples.baselines.constrain import pickcube_zscore_round37 as engine
    from examples.baselines.constrain.pickcube_position_units_round38 import _install_profile

    for name in ("PROFILE", "BASE_FORMAT", "CONSTRAIN_FORMAT", "normalize_observation",
                 "validate_stats", "EPISODE_TIMEOUT_SECONDS", "load_stats", "source_identity"):
        monkeypatch.setattr(engine, name, getattr(engine, name))
    _install_profile()

    class Denoiser:
        def __init__(self):
            self.inputs = []

        def __call__(self, sample, timestep, global_cond, hint, enabled):
            self.inputs.append((sample.clone(), timestep, global_cond.clone(), hint.clone(), enabled,
                                torch.random.get_rng_state().clone()))
            return torch.full_like(sample, 3.0 if enabled else 1.0)

    class Scheduler:
        timesteps = (7,)

        @staticmethod
        def step(epsilon, timestep, sample):
            return SimpleNamespace(prev_sample=epsilon)

    denoiser = Denoiser()
    base = SimpleNamespace(
        device=torch.device("cpu"), scheduler=Scheduler(), stats={},
        denoiser=denoiser, metadata={"checkpoint_sha256": "fixed"},
    )
    adapter = ResidualGuidanceAdapter(base, 1.5)
    queued = torch.zeros(2, 45)
    torch.manual_seed(915)
    output = adapter.get_action(queued)
    control, unconditional_base = denoiser.inputs
    assert torch.equal(torch.random.get_rng_state(), control[5])
    assert control[4] is True and unconditional_base[4] is False
    assert torch.equal(control[0], unconditional_base[0])
    assert control[1] == unconditional_base[1]
    assert torch.equal(control[2], unconditional_base[2])
    assert torch.equal(control[3], unconditional_base[3])
    assert torch.equal(control[5], unconditional_base[5])
    torch.testing.assert_close(output, torch.full((1, 8, 7), 4.0), rtol=0, atol=0)


def test_r38_position_profile_metadata_validates_as_frozen_profile():
    from examples.baselines.constrain.pickcube_position_units import (
        PROFILE_SHA256, profile_metadata, validate_profile,
    )

    metadata = profile_metadata()
    assert metadata["sha256"] == PROFILE_SHA256
    assert validate_profile(metadata) == metadata
