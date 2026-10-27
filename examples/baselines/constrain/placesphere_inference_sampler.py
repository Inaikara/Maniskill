"""Inference-only scheduler seam for the frozen PlaceSphere policies."""

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


def build_inference_scheduler(training_scheduler, sampler="ddpm", eta=0.0):
    if sampler not in ("ddpm", "ddim"):
        raise ValueError("sampler must be ddpm or ddim")
    if sampler == "ddpm":
        if eta is not None:
            raise ValueError("eta is only defined for DDIM; use eta=None for DDPM")
        return training_scheduler
    if eta != 0.0:
        raise ValueError("the registered PlaceSphere DDIM sampler requires eta=0")
    if int(training_scheduler.config.num_train_timesteps) != 100:
        raise ValueError("the registered inference sampler requires 100 training timesteps")
    scheduler = DDIMScheduler.from_config(training_scheduler.config)
    scheduler.set_timesteps(100)
    return scheduler


def sample_actions(scheduler, predict_noise, initial_sample, eta=0.0):
    """Run a preconfigured deterministic DDIM reverse process."""
    if not isinstance(scheduler, DDIMScheduler) or eta != 0.0:
        raise ValueError("sample_actions requires the registered DDIM eta=0 scheduler")
    if len(scheduler.timesteps) != 100:
        raise ValueError("the registered DDIM inference grid must contain 100 timesteps")
    sample = initial_sample
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            noise = predict_noise(sample, timestep)
            sample = scheduler.step(noise, timestep, sample, eta=0.0).prev_sample
    return sample
