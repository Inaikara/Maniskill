"""Standalone constrain checkpoint save/load (Ticket 03).

The checkpoint is a complete, self-contained training artifact: it stores the
full denoiser state dict (frozen base + control branch), the EMA copy of the
trainable parameters, the optimizer and lr-scheduler state, the training
step, the RNG streams needed to resume, and a config block recording the
model channels, the exact normalization contract and the base weight source
(path, subkey and a sha16 checksum of the frozen base tensors).

Reload builds the model strictly from the saved config and tensors
(``load_state_dict(..., strict=True)``) — inference never depends on the
original base checkpoint file because the frozen base weights are stored
inside ``model``. Resuming re-verifies the frozen-base checksum so drifted
weights fail loudly instead of silently continuing training.
"""

import hashlib
import os
from pathlib import Path

import torch

from examples.baselines.constrain.constrain_denoiser import (
    MODEL_CONFIG,
    build_constrain_denoiser,
)

CHECKPOINT_FORMAT = "constrain_dp_v1"

_BASE_NET_PREFIX = "base_net."


def frozen_base_sha16(denoiser):
    """Deterministic sha16 over the frozen base state dict (sorted by key)."""
    state = denoiser.base_net.state_dict()
    hasher = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        hasher.update(key.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(bytes(tensor.view(torch.uint8).numpy().tobytes())
                      if tensor.numel() else b"")
    return hasher.hexdigest()[:16]


def save_constrain_checkpoint(
    path,
    denoiser,
    ema=None,
    optimizer=None,
    lr_scheduler=None,
    step=0,
    rng=None,
    config=None,
):
    """Write the standalone checkpoint atomically enough for repeated saves.

    The payload is written to a ``.tmp`` sibling and moved into place with
    :func:`os.replace`, so an interrupted save (process crash mid-write) can
    never leave a partial file at the target path; a leftover ``.tmp`` from a
    previous crash is simply replaced by the next save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        format=CHECKPOINT_FORMAT,
        model=denoiser.state_dict(),
        ema=ema.state_dict() if ema is not None else None,
        optimizer=optimizer.state_dict() if optimizer is not None else None,
        lr_scheduler=lr_scheduler.state_dict() if lr_scheduler is not None else None,
        step=int(step),
        rng=rng if rng is not None else {},
        config=config,
    )
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def _strict_load(denoiser, state_dict):
    result = denoiser.load_state_dict(state_dict, strict=True)
    return result


def load_constrain_denoiser(checkpoint, device="cpu", weights="ema"):
    """Build and load the denoiser strictly from the checkpoint contents.

    Returns ``(denoiser, checkpoint_dict)``. ``weights`` selects the raw
    trained weights (``'model'``) or the EMA copy of the trainable parameters
    (``'ema'``, the evaluation weights) overlaid on the raw state. The frozen
    base weights come from the checkpoint itself; nothing is read from the
    original base checkpoint file.
    """
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    if ck.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint {checkpoint} is not a {CHECKPOINT_FORMAT!r} constrain checkpoint"
        )
    model_cfg = ck["config"]["model"]
    denoiser = build_constrain_denoiser(model_cfg, device=device)
    _strict_load(denoiser, ck["model"])
    if weights == "model":
        return denoiser, ck
    if weights != "ema":
        raise ValueError(f"Unknown weights selector {weights!r} (use 'model' or 'ema')")
    from diffusers.training_utils import EMAModel

    params = denoiser.trainable_parameters()
    ema = EMAModel(parameters=params, power=MODEL_CONFIG.get("ema_power", 0.75))
    if ck.get("ema") is not None:
        ema.load_state_dict(ck["ema"])
        ema.copy_to(params)
    return denoiser, ck


def verify_base_weights(denoiser, config):
    """Fail resume if the current frozen base drifted from the recorded one."""
    recorded = (config or {}).get("base_weights", {}).get("state_sha16")
    if recorded is None:
        return
    current = frozen_base_sha16(denoiser)
    if current != recorded:
        raise ValueError(
            "Frozen base weights drifted from the checkpoint's recorded base "
            f"(recorded sha16 {recorded}, current {current}); refusing to resume"
        )
