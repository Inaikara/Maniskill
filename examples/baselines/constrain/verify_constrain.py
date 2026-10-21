"""Post-training verification against the independently loaded official base.

Loads the final constrain checkpoint from disk, builds the official ManiSkill
DP agent *independently from the original base checkpoint file* (never from
the constrain checkpoint's own stored copy) and, at identical inputs, checks:

- the frozen base weights inside the checkpoint are bit-unchanged against
  that independent official load (sha16 of the base state dicts),
- the disabled-control epsilon output equals the official network output at
  the same (sample, timestep, global condition) inputs,
- the disabled-control closed-loop ``get_action`` equals the official
  ``Agent.get_action`` rollout under the same noise seed,
- and records how much the enabled control branch moves the epsilon output
  (recorded evidence, not a gate).

The evidence dict is written to JSON next to the checkpoint by ``main`` so
the result is auditable without re-running anything.
"""

import argparse
import json
import sys
from pathlib import Path
from examples.baselines.constrain.output_paths import experiment_dir

import torch

from examples.baselines.constrain.base_dp import DEFAULT_CKPT, build_official_agent
from examples.baselines.constrain.constrain_ckpt import frozen_base_sha16
from examples.baselines.constrain.constrain_adapter import ConstrainDPAdapter

_OFFICIAL_DP_DIR = Path(__file__).resolve().parents[3] / "examples" / "baselines" / "diffusion_policy"
if str(_OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_DP_DIR))

DISABLED_EPSILON_MAX_DIFF = 1e-5


def _sha16_of_state_dict(state_dict):
    import hashlib

    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        hasher.update(key.encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(str(tensor.dtype).encode())
        hasher.update(bytes(tensor.view(torch.uint8).numpy().tobytes())
                      if tensor.numel() else b"")
    return hasher.hexdigest()[:16]


def _epsilon_inputs(seed, num_batches, batch_size, device):
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(num_batches):
        batches.append(
            dict(
                sample=torch.randn(batch_size, 16, 4, generator=generator).to(device),
                timestep=torch.randint(0, 100, (batch_size,), generator=generator).to(device).long(),
                global_cond=torch.randn(batch_size, 96, generator=generator).to(device),
                hint=torch.randn(batch_size, 16, 3, generator=generator).to(device),
            )
        )
    return batches


def verify_constrain(
    checkpoint,
    base_checkpoint=DEFAULT_CKPT,
    device="cpu",
    num_batches=8,
    batch_size=8,
    seed=0,
) -> dict:
    """Run all frozen-base / disabled-epsilon checks; returns evidence."""
    adapter = ConstrainDPAdapter(
        checkpoint, device=device, weights="ema", control_enabled=False
    )
    constrain_net = adapter.denoiser

    official = build_official_agent(device=device)
    ck = torch.load(base_checkpoint, map_location=device, weights_only=False)
    official.load_state_dict(ck["ema_agent"])
    official_net = official.noise_pred_net

    base_state_checkpoint = {
        k[len("base_net."):]: v for k, v in constrain_net.state_dict().items()
        if k.startswith("base_net.")
    }
    official_state = official_net.state_dict()
    frozen_base_unchanged = all(
        torch.equal(base_state_checkpoint[k], official_state[k].cpu())
        for k in official_state
    ) and set(official_state) == set(base_state_checkpoint)

    evidence = dict(
        checkpoint=str(checkpoint),
        base_checkpoint=str(base_checkpoint),
        base_sha16_checkpoint=frozen_base_sha16(constrain_net),
        base_sha16_official=_sha16_of_state_dict(official_state),
        frozen_base_unchanged=bool(frozen_base_unchanged),
        weights="ema",
        step=int(torch.load(checkpoint, map_location="cpu", weights_only=False)["step"]),
        epsilon_inputs=dict(num_batches=num_batches, batch_size=batch_size, seed=seed),
        device=str(device),
    )

    max_diff = 0.0
    control_diff_sum = 0.0
    control_count = 0
    for inputs in _epsilon_inputs(seed, num_batches, batch_size, device):
        with torch.no_grad():
            disabled = constrain_net(
                inputs["sample"], inputs["timestep"],
                global_cond=inputs["global_cond"], hint=inputs["hint"], enabled=False,
            )
            official_out = official_net(
                sample=inputs["sample"], timestep=inputs["timestep"],
                global_cond=inputs["global_cond"],
            )
            enabled = constrain_net(
                inputs["sample"], inputs["timestep"],
                global_cond=inputs["global_cond"], hint=inputs["hint"], enabled=True,
            )
        max_diff = max(max_diff, float((disabled - official_out).abs().max()))
        control_diff_sum += float((enabled - official_out).abs().mean())
        control_count += 1
    evidence["disabled_epsilon"] = dict(
        max_abs_diff=max_diff,
        matches=bool(max_diff < DISABLED_EPSILON_MAX_DIFF),
        threshold=DISABLED_EPSILON_MAX_DIFF,
    )
    evidence["control_enabled_mean_abs_diff"] = control_diff_sum / max(control_count, 1)

    stacked = torch.zeros(1, adapter.obs_horizon, adapter.base_obs_dim + 3, device=device)
    cond = adapter.base_cond(stacked)
    torch.manual_seed(seed)
    official_action = official.get_action(cond)
    torch.manual_seed(seed)
    adapter_action = adapter.get_action(cond)
    rollout_diff = float((official_action - adapter_action).abs().max())
    evidence["disabled_rollout"] = dict(
        max_abs_diff=rollout_diff,
        matches=bool(rollout_diff < DISABLED_EPSILON_MAX_DIFF),
        threshold=DISABLED_EPSILON_MAX_DIFF,
        executed_shape=list(adapter_action.shape),
    )
    return evidence


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="final constrain checkpoint to verify from disk")
    p.add_argument("--base-checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=Path, default=None,
                   help="evidence JSON path (default: timestamped outputs/exp directory)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    evidence = verify_constrain(
        args.checkpoint, args.base_checkpoint, device=args.device
    )
    output = args.output if args.output is not None else (
        experiment_dir("constrainnet_verification") / "verify_evidence.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n")
    ok = (
        evidence["frozen_base_unchanged"]
        and evidence["disabled_epsilon"]["matches"]
        and evidence["disabled_rollout"]["matches"]
    )
    print(json.dumps(evidence, indent=2))
    print(f"saved: {output}")
    if not ok:
        raise SystemExit(1)
    return evidence


if __name__ == "__main__":
    main(sys.argv[1:])
