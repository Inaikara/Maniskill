import pytest
import torch

from examples.baselines.constrain.placesphere_inference_sampler import (
    build_inference_scheduler,
    sample_actions,
    sample_action_ensemble,
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


def test_four_plan_ensemble_averages_post_denoised_native_action_slice():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    training = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    scheduler = build_inference_scheduler(training, sampler="ddim", eta=0.0)
    initial = torch.stack([torch.full((16, 7), float(i)) for i in range(1, 5)])[None]
    condition = torch.tensor([[3.0, 9.0]])
    seen = []

    def predict(sample, timestep, repeated):
        seen.append((sample.shape, repeated.clone()))
        return torch.zeros_like(sample)

    result = sample_action_ensemble(scheduler, predict, initial, condition)
    # With a zero denoiser DDIM scales the initial plans linearly. The ensemble
    # is the arithmetic mean of fully denoised plans, sliced only afterwards.
    plans = []
    for plan in initial[0]:
        one = sample_actions(scheduler, lambda x, t: torch.zeros_like(x), plan[None])[0]
        plans.append(one[1:9])
    torch.testing.assert_close(result, torch.stack(plans).mean(0, keepdim=True))
    assert result.shape == (1, 8, 7)
    assert len(seen) == 100 and all(shape == (4, 16, 7) for shape, _ in seen)
    assert all(torch.equal(cond, condition.expand(4, -1)) for _, cond in seen)


def test_single_plan_ddim_path_matches_the_preensemble_reference_bitwise():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    training = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
    scheduler = build_inference_scheduler(training, sampler="ddim", eta=0.0)
    initial = torch.linspace(-0.75, 0.75, 16 * 7).reshape(1, 16, 7)

    def predict(sample, timestep):
        return sample * 0.03125 + timestep.to(sample).reshape(()) * 0.0002

    reference = initial.clone()
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            noise = predict(reference, timestep)
            reference = scheduler.step(noise, timestep, reference, eta=0.0).prev_sample
    actual = sample_actions(scheduler, predict, initial.clone(), eta=0.0)
    assert torch.equal(actual, reference)


@pytest.mark.parametrize("size,sampler", [(0, "ddim"), (2, "ddim"), (4, "ddpm")])
def test_adapter_rejects_unregistered_ensemble_sizes(size, sampler):
    from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter

    with pytest.raises(ValueError, match="ensemble_size"):
        PlaceSphereBaseDPAdapter.__init__(object.__new__(PlaceSphereBaseDPAdapter),
                                         "unused", sampler=sampler,
                                         eta=0.0 if sampler == "ddim" else None,
                                         ensemble_size=size)
