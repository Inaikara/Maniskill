"""Tests for durable training-run logging and checkpoint saves (Ticket 04).

Seams under test:
- ``RunLogger`` — durable per-step JSONL loss log (append, never clobber),
  atomic ``progress.json`` updates and an exact ``run_config.json`` snapshot;
- ``train_constrain.main`` — writes the run config (demo path, data count,
  base checkpoint + sha16, hyperparameters), streams per-step losses, keeps
  ``progress.json`` current and saves a periodic checkpoint that can be
  resumed from;
- ``save_constrain_checkpoint`` — atomic writes: an interrupted prior save
  (leftover ``.tmp`` file) never leaves a corrupt or partially written
  checkpoint at the target path.

Tests run the real dataset and the real base checkpoint on CPU with tiny
step budgets.
"""

import json
from pathlib import Path

import pytest
import torch

from examples.baselines.constrain.base_dp import DEFAULT_CKPT
from examples.baselines.constrain.constrain_ckpt import (
    load_constrain_denoiser,
    save_constrain_checkpoint,
)
from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser
from examples.baselines.constrain.train_logging import RunLogger

REAL_DEMO = "outputs/stackcube_obstacle/trajectory.h5"


def test_log_step_appends_one_json_line_per_call(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log_step(step=1, loss=0.5, lr=1e-4)
    logger.log_step(step=2, loss=0.25, lr=1e-4)
    lines = (tmp_path / "logs" / "loss.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["step"] == 1 and first["loss"] == pytest.approx(0.5)
    assert first["lr"] == pytest.approx(1e-4)
    assert json.loads(lines[1])["step"] == 2


def test_log_step_appends_across_reopened_loggers(tmp_path):
    """A resumed run reopening the run dir appends; prior lines survive."""
    first = RunLogger(tmp_path)
    first.log_step(step=1, loss=0.5, lr=1e-4)
    second = RunLogger(tmp_path)
    second.log_step(step=2, loss=0.25, lr=1e-4)
    lines = (tmp_path / "logs" / "loss.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]


def test_log_progress_writes_atomic_progress_json(tmp_path):
    logger = RunLogger(tmp_path)
    logger.log_progress(step=7, total=100, extra=dict(mean_loss=0.4))
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["step"] == 7 and progress["total"] == 100
    assert progress["mean_loss"] == pytest.approx(0.4)
    assert "updated_at" in progress
    assert not list(tmp_path.glob("progress.json.tmp"))


def test_truncate_after_drops_steps_beyond_checkpoint_and_torn_lines(tmp_path):
    """Crash-recovery seam: resuming from a periodic checkpoint re-truncates
    the per-step log to the checkpoint step; a torn (partially written) final
    line from a crashed process is dropped instead of poisoning the log."""
    logger = RunLogger(tmp_path)
    for step in (1, 2, 3, 4, 5):
        logger.log_step(step=step, loss=0.5, lr=1e-4)
    with open(logger.loss_path, "a") as f:
        f.write('{"step": 6, "loss": 0.2')  # torn line from a crashed write
    removed = logger.truncate_after(step=3)
    lines = logger.loss_path.read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2, 3]
    assert removed == 3
    # appending continues after the truncation
    logger.log_step(step=4, loss=0.1, lr=1e-4)
    lines = logger.loss_path.read_text().splitlines()
    assert json.loads(lines[-1])["step"] == 4
    assert len(lines) == 4


def test_write_config_records_exact_config(tmp_path):
    logger = RunLogger(tmp_path)
    logger.write_config(dict(seed=1, total_iters=30000, batch_size=1024))
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config == dict(seed=1, total_iters=30000, batch_size=1024)


def test_save_constrain_checkpoint_is_atomic_over_leftover_tmp(tmp_path):
    """A leftover .tmp from an interrupted prior save never corrupts the next
    checkpoint: the target file is always a complete, loadable artifact and
    no .tmp remains afterwards."""
    target = tmp_path / "periodic.pt"
    target.with_name("periodic.pt.tmp").write_bytes(b"garbage-from-crash")
    denoiser = build_constrain_denoiser()
    config = dict(
        model=dict(input_dim=4, global_cond_dim=96, diffusion_step_embed_dim=64,
                   down_dims=[64, 128, 256], n_groups=8, cond_dim=3,
                   num_diffusion_iters=100),
        policy=dict(obs_horizon=2, act_horizon=8, pred_horizon=16,
                    base_obs_dim=48, act_dim=4),
        normalization=dict(),
        base_weights=dict(source="x", subkey="ema_agent", state_sha16="0" * 16),
    )
    save_constrain_checkpoint(
        target, denoiser=denoiser, ema=None, optimizer=None, lr_scheduler=None,
        step=3, rng={}, config=config,
    )
    assert not target.with_name("periodic.pt.tmp").exists()
    denoiser2, ck = load_constrain_denoiser(target, device="cpu", weights="model")
    assert ck["step"] == 3
    # overwriting an existing checkpoint also replaces it cleanly
    save_constrain_checkpoint(
        target, denoiser=denoiser, ema=None, optimizer=None, lr_scheduler=None,
        step=4, rng={}, config=config,
    )
    _, ck2 = load_constrain_denoiser(target, device="cpu", weights="model")
    assert ck2["step"] == 4


@pytest.fixture(scope="module")
def tiny_main_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("tiny_main")
    from examples.baselines.constrain.train_constrain import main

    summary = main([
        "--demo-path", REAL_DEMO,
        "--base-checkpoint", str(DEFAULT_CKPT),
        "--total-iters", "2", "--batch-size", "4", "--device", "cpu",
        "--seed", "1", "--save-every", "1",
        "--run-dir", str(run_dir),
    ])
    return run_dir, summary


def test_main_writes_run_config_with_data_count_and_base_sha(tiny_main_run):
    run_dir, _ = tiny_main_run
    config = json.loads((run_dir / "run_config.json").read_text())
    assert config["training"]["demo_path"].endswith("trajectory.h5")
    assert config["data"]["num_windows"] == 10356
    assert config["data"]["num_demos"] == 100
    assert config["base_weights"]["source"].endswith("best_eval_success_at_end.pt")
    assert len(config["base_weights"]["state_sha16"]) == 16
    assert config["training"]["total_iters"] == 2
    assert config["training"]["seed"] == 1


def test_main_streams_per_step_losses_and_progress(tiny_main_run):
    run_dir, _ = tiny_main_run
    lines = (run_dir / "logs" / "loss.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2]
    progress = json.loads((run_dir / "progress.json").read_text())
    assert progress["step"] == 2 and progress["total"] == 2


def test_main_periodic_checkpoint_is_loadable(tiny_main_run):
    run_dir, _ = tiny_main_run
    _, ck = load_constrain_denoiser(
        run_dir / "checkpoints" / "periodic.pt", device="cpu", weights="ema"
    )
    assert ck["step"] == 2
