"""Public Peg Constrain dataset and policy seams.

Confirmed seams: PegConstrainDataset windows and epsilon-loss/adapter sampling.
"""

import numpy as np
import pytest
import torch

from examples.baselines.constrain.peg_orientation_constrain import (
    PegConstrainDataset,
    FORMAT,
    MODEL_CONFIG,
    PegConstrainAdapter,
    build_peg_denoiser,
    compute_peg_epsilon_loss,
    _load_base,
    BASE,
)
from examples.baselines.constrain.constrain_ckpt import frozen_base_sha16


def test_dataset_uses_current_hint_and_preserves_native_seven_d_actions(tmp_path):
    h5py = pytest.importorskip("h5py")
    data = tmp_path / "tiny.h5"
    manifest = tmp_path / "coverage.json"
    import json

    obs = np.zeros((17, 39), dtype=np.float32)
    obs[:, 25:32] = [0, 0, 0, 1, 0, 0, 0]
    obs[1, 28:32] = [np.cos(0.125), 0, 0, np.sin(0.125)]
    actions = np.arange(112, dtype=np.float32).reshape(16, 7) / 112
    with h5py.File(data, "w") as f:
        group = f.create_group("traj_0")
        group.create_dataset("obs", data=obs)
        group.create_dataset("actions", data=actions)
    manifest.write_text(json.dumps({"episodes": [{"episode_id": 0, "target_yaw_rad": 0.5}]}))
    ds = PegConstrainDataset(data, manifest, expected_episodes=1)
    row = ds[0]
    assert row["observations"].shape == (2, 39)
    assert row["actions"].shape == (16, 7)
    assert row["hint"].shape == (2,)
    assert torch.equal(row["actions"][0], torch.from_numpy(actions[0]))
    assert torch.allclose(row["hint"], torch.tensor([-np.sin(0.5), np.cos(0.5)], dtype=torch.float32), atol=1e-6)
    latest = ds[1]
    assert torch.allclose(latest["hint"], torch.tensor([np.sin(-0.25), np.cos(-0.25)], dtype=torch.float32), atol=1e-6)
    assert torch.equal(latest["actions"][0], torch.from_numpy(actions[0]))
    assert torch.equal(latest["actions"][3, 3:6], torch.from_numpy(actions[3, 3:6]))


def test_balanced_replay_batch_contains_equal_random_and_fixed_windows():
    from examples.baselines.constrain.peg_orientation_constrain import balanced_replay_batch

    random_batch = {"source": torch.zeros(2, dtype=torch.int64)}
    fixed_batch = {"source": torch.ones(2, dtype=torch.int64)}
    batch = balanced_replay_batch(random_batch, fixed_batch)
    assert batch["source"].tolist() == [0, 0, 1, 1]


def test_resume_restores_raw_weights_optimizer_ema_and_rng_but_not_lr_schedule():
    import random
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from examples.baselines.constrain.peg_orientation_constrain import restore_training_state

    def components():
        model = torch.nn.Linear(2, 1, bias=False)
        params = list(model.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        ema = EMAModel(parameters=params, power=.75)
        schedule = get_scheduler("cosine", optimizer, num_warmup_steps=500,
                                 num_training_steps=20000)
        return model, optimizer, ema, schedule

    source, source_optimizer, source_ema, old_schedule = components()
    with torch.no_grad(): source.weight.fill_(2)
    source_optimizer.zero_grad()
    source.weight.grad = torch.ones_like(source.weight)
    source_optimizer.step(); source_ema.step(source.parameters())
    old_schedule.step()
    source_ema.shadow_params[0].fill_(3)
    state = {"model": source.state_dict(), "optimizer": source_optimizer.state_dict(),
             "ema": source_ema.state_dict(), "lr_scheduler": old_schedule.state_dict(),
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state()}}
    target, optimizer, ema, fresh_schedule = components()
    restore_training_state(state, target, optimizer, ema, "cpu")
    assert torch.equal(target.weight, torch.full_like(target.weight, 2))
    assert torch.equal(ema.shadow_params[0], torch.full_like(ema.shadow_params[0], 3))
    assert ema.optimization_step == source_ema.optimization_step
    assert optimizer.state[target.weight]["step"].item() == source_optimizer.state[source.weight]["step"].item()
    assert torch.equal(optimizer.state[target.weight]["exp_avg"], source_optimizer.state[source.weight]["exp_avg"])
    assert fresh_schedule.last_epoch == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-7)
    optimizer.zero_grad()
    target.weight.grad = torch.zeros_like(target.weight)
    optimizer.step()
    fresh_schedule.step()
    assert optimizer.param_groups[0]["lr"] > 0


def test_fresh_train_seed_reproduces_random_model_initialization(tmp_path, monkeypatch):
    import argparse
    import examples.baselines.constrain.peg_orientation_constrain as training

    initial_states = []

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return {"observations": torch.zeros(2, 39), "actions": torch.zeros(16, 7),
                    "hint": torch.zeros(2)}

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_net = torch.nn.Linear(1, 1)
            self.hint_encoder = torch.nn.Linear(1, 1)
            initial_states.append({key: value.detach().clone()
                                   for key, value in self.state_dict().items()})

        def trainable_parameters(self):
            return self.parameters()

    class EMA:
        def __init__(self, parameters, power):
            pass

        def step(self, parameters):
            pass

        def state_dict(self):
            return {}

    class NoiseScheduler:
        timesteps = ()

    monkeypatch.setattr(training, "PegConstrainDataset", Dataset)
    monkeypatch.setattr(training, "_load_base", lambda *args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(training, "build_peg_denoiser", lambda base: Denoiser())
    monkeypatch.setattr(training, "sha256_file", lambda path: "test-hash")
    monkeypatch.setattr(training.PegConstrainAdapter, "_scheduler", lambda _: NoiseScheduler())
    monkeypatch.setattr(training, "compute_peg_epsilon_loss",
                        lambda model, *args: sum(p.sum() for p in model.parameters()) * 0)

    import diffusers.training_utils
    import diffusers.optimization
    monkeypatch.setattr(diffusers.training_utils, "EMAModel", EMA)
    monkeypatch.setattr(diffusers.optimization, "get_scheduler",
                        lambda *args, **kwargs: type("Schedule", (), {"step": lambda self: None,
                                                                         "state_dict": lambda self: {}})())

    for run in range(2):
        args = argparse.Namespace(seed=1, device="cpu", data="random", manifest="manifest",
            random_episodes=1, batch_size=2, fixed_data=None, fixed_manifest=None,
            fixed_episodes=0, base="base", resume_from=None, updates=1, run=tmp_path / f"run{run}",
            ckpt=tmp_path / f"ckpt{run}", save_every=10)
        training.train(args)

    assert len(initial_states) == 2
    assert all(torch.equal(initial_states[0][key], initial_states[1][key])
               for key in initial_states[0])


def test_resume_validation_requires_expected_parent_and_exact_frozen_inputs(tmp_path):
    from examples.baselines.constrain.peg_orientation_constrain import validate_resume

    args = type("Args", (), {"random_episodes": 500, "fixed_episodes": 1000,
                              "updates": 50000})()
    checkpoint = {"step": 30000,
                  "config": {"training": {"random_episodes": 500,
                      "data_sha256": "random", "manifest_sha256": "random-manifest",
                      "fixed_episodes": 1000, "fixed_data_sha256": "fixed",
                      "fixed_manifest_sha256": "fixed-manifest"},
                      "base_weights": {"sha256": "base"}}}
    parent_hash = "6e26647b5ee1ad356d57f06464f77a1884fcd76b1fad52c8ed181e9e0d1de1de"
    validate_resume(checkpoint, args, "base", "random", "random-manifest", "fixed", "fixed-manifest", parent_hash)
    with pytest.raises(ValueError, match="data_sha256"):
        validate_resume(checkpoint, args, "base", "wrong", "random-manifest", "fixed", "fixed-manifest", parent_hash)
    with pytest.raises(ValueError, match="checkpoint identity"):
        validate_resume(checkpoint, args, "base", "random", "random-manifest", "fixed", "fixed-manifest", "bad-parent")
    checkpoint["step"] = 29999
    with pytest.raises(ValueError, match="update-30000"):
        validate_resume(checkpoint, args, "base", "random", "random-manifest", "fixed", "fixed-manifest", parent_hash)


def test_training_cli_forwards_random_episode_count(tmp_path, monkeypatch):
    import examples.baselines.constrain.peg_orientation_constrain as training

    captured = {}
    monkeypatch.setattr(training, "train", lambda args: captured.update(vars(args)))
    training.main([
        "--random-episodes", "200",
        "--run", str(tmp_path / "run"), "--ckpt", str(tmp_path / "ckpt"),
        "--fixed-data", str(tmp_path / "fixed.h5"),
        "--fixed-manifest", str(tmp_path / "fixed.json"),
    ])

    assert captured["random_episodes"] == 200


def test_training_cli_accepts_only_cumulative_50000_resume_target(tmp_path, monkeypatch):
    import examples.baselines.constrain.peg_orientation_constrain as training

    captured = {}
    monkeypatch.setattr(training, "train", lambda args: captured.update(vars(args)))
    training.main(["--updates", "50000", "--resume-from", str(tmp_path / "parent.pt"),
                   "--run", str(tmp_path / "run"), "--ckpt", str(tmp_path / "ckpt"),
                   "--fixed-data", str(tmp_path / "fixed.h5"),
                   "--fixed-manifest", str(tmp_path / "fixed.json")])
    assert captured["updates"] == 50000
    assert captured["resume_from"] == tmp_path / "parent.pt"


def test_zero_residual_control_matches_base_and_disabled_path(tmp_path):
    torch.manual_seed(4)
    base = torch.nn.Module()
    from examples.baselines.constrain.constrain_denoiser import _official_conditional_unet1d

    net_cls = _official_conditional_unet1d()
    base = net_cls(input_dim=7, global_cond_dim=78, diffusion_step_embed_dim=64,
                   down_dims=[64, 128, 256], n_groups=8)
    denoiser = build_peg_denoiser(base)
    sample = torch.randn(1, 16, 7)
    cond = torch.randn(1, 78)
    hint = torch.randn(1, 16, 2)
    timestep = torch.tensor([3])
    with torch.no_grad():
        expected = base(sample, timestep, cond)
        enabled = denoiser(sample, timestep, cond, hint=hint)
        disabled = denoiser(sample, timestep, cond, hint=hint, enabled=False)
    assert torch.equal(enabled, expected)
    assert torch.equal(disabled, expected)
    checkpoint = tmp_path / "peg-control-zero.pt"
    torch.save({"format": FORMAT, "model": denoiser.state_dict(), "ema": None,
                "step": 0, "config": {"model": MODEL_CONFIG,
                "base_weights": {"source": "test", "state_sha16": frozen_base_sha16(denoiser)}}}, checkpoint)

    class LiveTask:
        @staticmethod
        def get_orientation_hint():
            return np.asarray([0.2, 0.98], dtype=np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    on = PegConstrainAdapter(checkpoint, device=device, task=LiveTask())
    off = PegConstrainAdapter(checkpoint, device=device, task=LiveTask(), control_enabled=False)
    history = torch.randn(2, 39)
    torch.manual_seed(17)
    if device == "cuda": torch.cuda.manual_seed_all(17)
    action_on = on.get_action(history)
    torch.manual_seed(17)
    if device == "cuda": torch.cuda.manual_seed_all(17)
    action_off = off.get_action(history)
    assert torch.equal(action_on, action_off)


def test_real_base_loss_updates_hint_branch_and_freezes_base_exactly():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    base = _load_base(BASE, "cpu")
    denoiser = build_peg_denoiser(base)
    frozen = {name: value.detach().clone() for name, value in denoiser.base_net.state_dict().items()}
    optimizer = torch.optim.AdamW(denoiser.trainable_parameters(), lr=1e-4)
    scheduler = PegConstrainAdapter._scheduler(100)
    obs = torch.randn(2, 2, 39)
    actions = torch.randn(2, 16, 7)
    hints = torch.tensor([[0.6, 0.8], [-0.8, 0.6]])
    noise = torch.randn_like(actions)
    timesteps = torch.tensor([17, 63])

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = compute_peg_epsilon_loss(denoiser, obs, actions, hints, noise, timesteps, scheduler)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    for module in (denoiser.hint_encoder, denoiser.control_down_modules, denoiser.control_mid_modules):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() for g in grads)
    assert all(torch.equal(frozen[name], value) for name, value in denoiser.base_net.state_dict().items())


def test_zero_control_full_sampler_matches_real_base_adapter(tmp_path):
    base = _load_base(BASE, "cpu")
    denoiser = build_peg_denoiser(base)
    checkpoint = tmp_path / "zero-real-base.pt"
    torch.save({"format": FORMAT, "model": denoiser.state_dict(), "ema": None,
                "step": 0, "config": {"model": MODEL_CONFIG,
                "base_weights": {"source": str(BASE), "state_sha16": frozen_base_sha16(denoiser)}}}, checkpoint)
    from examples.baselines.constrain.peg_orientation_base_dp import PegBaseAdapter
    constrain = PegConstrainAdapter(checkpoint, device="cpu", task=object())
    base_adapter = PegBaseAdapter(BASE, device="cpu")
    history = torch.randn(2, 39)
    hint = torch.tensor([0.6, 0.8])
    torch.manual_seed(73)
    expected = base_adapter.get_action(history[None])
    torch.manual_seed(73)
    actual = constrain.get_action(history, hint=hint)
    assert torch.equal(actual, expected)


def test_hint_interventions_are_live_and_correct_mode_keeps_native_task_hint():
    import math

    from examples.baselines.constrain.peg_orientation_data import build_hint_for_mode

    class LiveTask:
        def __init__(self):
            self.hint = np.asarray([0.25, 0.96824586], dtype=np.float32)
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.set_peg_yaw(1.2)

        def set_peg_yaw(self, yaw):
            self.peg.pose.q = torch.tensor([[
                math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2),
            ]])

        def get_orientation_hint(self):
            return self.hint.copy()

    task = LiveTask()
    correct = PegConstrainAdapter.__new__(PegConstrainAdapter)
    correct.task = task
    correct.hint_mode = "correct"
    correct.actual_target_yaw = np.pi / 2
    correct.shuffled_target_yaw = None
    assert np.array_equal(correct._live_hint(), task.get_orientation_hint())

    fixed = PegConstrainAdapter.__new__(PegConstrainAdapter)
    fixed.task = task
    fixed.hint_mode = "fixed_standard"
    fixed.set_episode_context(7, {"target_yaw_rad": 0.8})
    fixed_first = fixed._live_hint()
    task.set_peg_yaw(1.4)
    fixed_after_motion = fixed._live_hint()
    shuffled = PegConstrainAdapter.__new__(PegConstrainAdapter)
    shuffled.task = task
    shuffled.hint_mode = "shuffled"
    shuffled.set_episode_context(7, {"target_yaw_rad": 0.8}, 0.4)
    shuffled_first = shuffled._live_hint()
    task.set_peg_yaw(1.6)
    shuffled_after_motion = shuffled._live_hint()
    assert np.allclose(fixed_first, build_hint_for_mode("fixed_standard", 1.2, 0.8))
    assert not np.array_equal(fixed_first, np.asarray([0.0, 1.0]))
    assert not np.array_equal(fixed_first, fixed_after_motion)
    assert np.allclose(shuffled_first, build_hint_for_mode("shuffled", 1.4, 0.8, 0.4))
    assert not np.array_equal(shuffled_first, shuffled_after_motion)
    assert shuffled.shuffled_target_yaw == 0.4


def test_adapter_can_retain_every_sampler_hint_without_changing_actions_or_default_retention():
    class Denoiser:
        def __call__(self, sample, *_args, **_kwargs):
            return torch.zeros_like(sample)

    class Scheduler:
        timesteps = []

    def adapter(record_all_hints):
        result = PegConstrainAdapter.__new__(PegConstrainAdapter)
        result.device = torch.device("cpu")
        result.task = None
        result.actual_target_yaw = None
        result.shuffled_target_yaw = None
        result.hint_mode = "correct"
        result.hint_samples = []
        result.sampler_calls = 0
        result.record_all_hints = record_all_hints
        result.scheduler = Scheduler()
        result.denoiser = Denoiser()
        result.control_enabled = True
        return result

    default = adapter(False)
    complete = adapter(True)
    default_actions = []
    complete_actions = []
    for call in range(5):
        hint = torch.tensor([float(call), 1.0])
        torch.manual_seed(991 + call)
        default_actions.append(default.get_action(torch.zeros(2, 39), hint=hint))
        default_rng = torch.get_rng_state().clone()
        torch.manual_seed(991 + call)
        complete_actions.append(complete.get_action(torch.zeros(2, 39), hint=hint))
        assert torch.equal(torch.get_rng_state(), default_rng)

    assert len(default.hint_samples) == 3
    assert len(complete.hint_samples) == 5
    assert [sample["sampler_call"] for sample in complete.hint_samples] == list(range(1, 6))
    assert all(torch.equal(left, right) for left, right in zip(default_actions, complete_actions))
