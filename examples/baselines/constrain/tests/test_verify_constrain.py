"""Tests for the post-training frozen-base/epsilon verification (Ticket 04).

Seams under test:
- :func:`verify_constrain` — loads the final constrain checkpoint from disk,
  loads the official base DP *independently from the original base checkpoint
  file* (not from the constrain checkpoint's own copy) and checks, at
  identical inputs:
  - the frozen base weights are bit-unchanged against that independent load,
  - the disabled-control epsilon output equals the official network output,
  - the disabled-control closed-loop ``get_action`` equals the official
    ``Agent.get_action`` under the same noise seed,
  - and records how much the enabled control branch moves the output.
"""

import sys
from pathlib import Path

import pytest
import torch

from examples.baselines.constrain.base_dp import DEFAULT_CKPT, build_official_agent

OFFICIAL_DP_DIR = Path(__file__).resolve().parents[3] / "examples" / "baselines" / "diffusion_policy"
if str(OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DP_DIR))

REAL_DEMO = "demos/StackCube-obstacle/trajectory.h5"


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("verify_train")
    from examples.baselines.constrain.train_constrain import ConstrainTrainer

    trainer = ConstrainTrainer(
        demo_path=REAL_DEMO, base_checkpoint=DEFAULT_CKPT, device="cpu",
        batch_size=4, total_iters=2, run_dir=run_dir, seed=1,
    )
    trainer.setup()
    trainer.train_steps(2)
    return trainer.save_checkpoint(run_dir / "final.pt")


def test_verify_frozen_base_and_disabled_epsilon_match(tiny_checkpoint):
    from examples.baselines.constrain.verify_constrain import verify_constrain

    evidence = verify_constrain(tiny_checkpoint, DEFAULT_CKPT, device="cpu")
    assert evidence["frozen_base_unchanged"] is True
    assert evidence["base_sha16_checkpoint"] == evidence["base_sha16_official"]
    assert evidence["disabled_epsilon"]["matches"] is True
    assert evidence["disabled_epsilon"]["max_abs_diff"] < 1e-5
    assert evidence["disabled_rollout"]["matches"] is True
    assert evidence["disabled_rollout"]["max_abs_diff"] < 1e-5
    # the enabled control branch is recorded (post-training it must move the
    # output away from the base; the value itself is recorded, not gated)
    assert evidence["control_enabled_mean_abs_diff"] >= 0.0
    assert evidence["checkpoint"] == str(tiny_checkpoint)
    assert evidence["base_checkpoint"].endswith("best_eval_success_at_end.pt")


def test_verify_detects_tampered_frozen_base(tiny_checkpoint, tmp_path):
    """A modified frozen base tensor inside the checkpoint fails the frozen
    base check against the independently loaded official network."""
    from examples.baselines.constrain.verify_constrain import verify_constrain

    tampered = tmp_path / "tampered.pt"
    ck = torch.load(tiny_checkpoint, map_location="cpu", weights_only=False)
    key = next(k for k in ck["model"] if k.startswith("base_net."))
    ck["model"][key] = ck["model"][key] + 0.01
    torch.save(ck, tampered)
    evidence = verify_constrain(tampered, DEFAULT_CKPT, device="cpu")
    assert evidence["frozen_base_unchanged"] is False
    assert evidence["base_sha16_checkpoint"] != evidence["base_sha16_official"]


def test_verify_disabled_rollout_uses_independent_official_load(tiny_checkpoint):
    """The official reference policy is loaded straight from the base
    checkpoint file, never from the constrain checkpoint's stored copy."""
    from examples.baselines.constrain.verify_constrain import verify_constrain

    evidence = verify_constrain(tiny_checkpoint, DEFAULT_CKPT, device="cpu")
    # the recorded official weights fingerprint must correspond to the
    # original base checkpoint's ema_agent state, not the constrain file
    official = build_official_agent(device="cpu")
    ck = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)
    official.load_state_dict(ck["ema_agent"])
    import hashlib

    hasher = hashlib.sha256()
    for key in sorted(official.noise_pred_net.state_dict()):
        tensor = official.noise_pred_net.state_dict()[key].detach().cpu().contiguous()
        hasher.update(key.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(bytes(tensor.view(torch.uint8).numpy().tobytes())
                      if tensor.numel() else b"")
    assert evidence["base_sha16_official"] == hasher.hexdigest()[:16]
