import pytest
import torch

from examples.baselines.constrain.placesphere_inference_sampler import (
    build_inference_scheduler,
    sample_actions,
)


def test_ddim_uses_training_scheduler_config_and_is_deterministic():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    training = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler = build_inference_scheduler(training, sampler="ddim", eta=0.0)
    assert scheduler.config.num_train_timesteps == 100
    assert scheduler.config.beta_schedule == "squaredcos_cap_v2"
    assert scheduler.config.clip_sample is True
    assert torch.equal(scheduler.betas, training.betas)
    assert torch.equal(scheduler.timesteps, torch.arange(99, -1, -1))

    def predict(sample, timestep):
        return sample * 0.125 + timestep.to(sample).reshape(()) * 0.0001

    initial = torch.linspace(-1, 1, 16 * 7).reshape(1, 16, 7)
    first = sample_actions(scheduler, predict, initial.clone(), eta=0.0)
    second = sample_actions(scheduler, predict, initial.clone(), eta=0.0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (1, 16, 7)


@pytest.mark.parametrize("sampler,eta", [("unknown", 0.0), ("ddim", 0.1), ("ddpm", 0.0)])
def test_sampler_rejects_unregistered_runtime_configurations(sampler, eta):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    training = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    with pytest.raises(ValueError):
        build_inference_scheduler(training, sampler=sampler, eta=eta)


def test_inference_grid_cannot_be_changed_after_registration():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    training = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    scheduler = build_inference_scheduler(training, sampler="ddim", eta=0.0)
    scheduler.set_timesteps(50)
    with pytest.raises(ValueError, match="100 timesteps"):
        sample_actions(scheduler, lambda sample, timestep: torch.zeros_like(sample),
                       torch.zeros(1, 16, 7))
