"""Official PegInsertionSide DP launcher routing.

Seam: the unmodified official ``save_ckpt`` writes
``runs/<exp_name>/checkpoints/<tag>.pt`` relative to the process cwd. The
launcher may only prepare that path so the file lands in the physical
checkpoint directory.
"""

import sys
from pathlib import Path

import pytest

OFFICIAL_DP_DIR = Path(__file__).resolve().parents[2] / "diffusion_policy"
if str(OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DP_DIR))

from examples.baselines.constrain.peg_official_dp import (  # noqa: E402
    official_train_argv,
    prepare_official_output,
)


EXP_NAME = "diffusion_policy-PegInsertionSide-v1-state-100_motionplanning_demos-1"


class _State:
    def parameters(self):
        return []

    def state_dict(self):
        return {"marker": 1}

    def copy_to(self, params):
        return None


def test_official_save_ckpt_writes_through_prepared_symlink(tmp_path, monkeypatch):
    exp = tmp_path / "outputs" / "exp" / "20261112_000000_peg_official_dp"
    ckpt = tmp_path / "outputs" / "ckpt" / "20261112_000000_peg_official_dp"
    link = prepare_official_output(exp, ckpt, EXP_NAME)
    assert link.is_symlink()
    assert link.resolve() == ckpt.resolve()

    import train as official_train

    monkeypatch.setattr(official_train, "ema", _State(), raising=False)
    monkeypatch.setattr(official_train, "ema_agent", _State(), raising=False)
    monkeypatch.setattr(official_train, "agent", _State(), raising=False)
    monkeypatch.chdir(exp)
    official_train.save_ckpt(EXP_NAME, "best_eval_success_once")

    saved = ckpt / "best_eval_success_once.pt"
    routed = exp / "runs" / EXP_NAME / "checkpoints" / "best_eval_success_once.pt"
    assert saved.is_file()
    assert routed.resolve() == saved.resolve()
    assert link.is_symlink()


def test_prepare_refuses_existing_checkpoint_directory(tmp_path):
    ckpt = tmp_path / "outputs" / "ckpt" / "already-there"
    ckpt.mkdir(parents=True)
    (ckpt / "user_file").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_official_output(tmp_path / "outputs" / "exp" / "run", ckpt, EXP_NAME)
    assert (ckpt / "user_file").read_text() == "keep"


def test_official_argv_keeps_unpassed_defaults_and_omits_tracking():
    argv = official_train_argv(
        "demos/PegInsertionSide-v1/motionplanning/trajectory.state.pd_ee_delta_pose.physx_cpu.h5",
        EXP_NAME,
        track=False,
    )
    text = " ".join(argv)
    assert "--env-id PegInsertionSide-v1" in text
    assert "--control-mode pd_ee_delta_pose" in text
    assert "--sim-backend physx_cpu" in text
    assert "--num-demos 100" in text
    assert "--max_episode_steps 300" in text
    assert "--total_iters 100000" in text
    assert f"--exp-name {EXP_NAME}" in text
    assert "--demo_type motionplanning" in text
    for omitted in ("--seed", "--track", "--batch-size", "--batch_size", "--lr", "--eval-freq", "--save-freq"):
        assert omitted not in argv
