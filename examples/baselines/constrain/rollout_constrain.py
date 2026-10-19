"""Reproducible single-episode closed-loop rollout of the constrain policy.

Loads a standalone constrain checkpoint (saved by ``train_constrain``) in this
process only — inference needs no training-time state and no dependency on
the original base checkpoint file — and executes it closed-loop in one
``StackCubeObstacle-v1`` episode on the CPU physics backend (network device
configurable):

- the queued 2x51 stacked obs is passed through the shared condition seam
  (``ConstrainDPAdapter.base_cond`` captures it; the relative-condition hint
  comes from its most recent frame),
- the official epsilon/DDPM rollout runs with the control branch and the
  8-step executed horizon exactly as at the base rollout seam,
- each executed action is mapped 4d -> 7d ``pd_ee_delta_pose`` and stepped,
- termination reason, stack success, obstacle contact, forbidden crossing
  and the safe-success flag are recorded in a JSON file under ``outputs/``.

With ``--disable-control`` the control branch is bypassed (base DP behavior
at the same seeds) — useful as a paired comparison, not an acceptance gate.

The obstacle geometry defaults to the demo configuration: height 0.18 m and
a radius sampled in 0.02-0.04 m. This is a behavior record, not an avoidance
acceptance gate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.rollout_base_dp import (
    build_env,
    unique_path,
    unwrap_final_info,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", type=str, default="cpu", help="network device (cpu or cuda)")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--obstacle-height", type=float, default=0.18)
    p.add_argument(
        "--obstacle-radius",
        type=float,
        default=None,
        help="fixed radius in [0.02, 0.04]; sampled within the range when omitted",
    )
    p.add_argument("--disable-control", action="store_true",
                   help="bypass the control branch (base DP behavior)")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO_ROOT / "outputs" / "constrain_rollout")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.obstacle_radius is None:
        rng = np.random.default_rng(args.seed)
        args.obstacle_radius = float(rng.uniform(0.02, 0.04))
    if not 0.02 <= args.obstacle_radius <= 0.04:
        raise ValueError(f"obstacle radius {args.obstacle_radius} outside [0.02, 0.04]")

    from examples.baselines.constrain.constrain_adapter import ConstrainDPAdapter

    # reload from the standalone checkpoint only (strict load inside)
    adapter = ConstrainDPAdapter(
        args.checkpoint, device=args.device, control_enabled=not args.disable_control
    )
    torch.manual_seed(args.seed)

    env = build_env(args.seed, args.max_steps, args.obstacle_height, args.obstacle_radius)
    obs, _ = env.reset(seed=args.seed)

    result = {
        "checkpoint": str(adapter.metadata["checkpoint"]),
        "trained_step": adapter.metadata["step"],
        "weights": adapter.metadata["weights"],
        "base_weights_source": adapter.metadata["base_weights"]["source"],
        "network_device": args.device,
        "sim_backend": "physx_cpu",
        "control_enabled": adapter.control_enabled,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "obstacle": {"radius": args.obstacle_radius, "height": args.obstacle_height},
        "pred_horizon": adapter.metadata["pred_horizon"],
        "act_horizon": adapter.metadata["act_horizon"],
        "obs_horizon": adapter.metadata["obs_horizon"],
        "num_diffusion_iters": adapter.metadata["num_diffusion_iters"],
    }

    first_collision_step = None
    first_crossing_step = None
    elapsed = 0
    terminated_reason = "max_steps"
    truncated = False

    while elapsed < args.max_steps:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(args.device)
        cond = adapter.base_cond(obs_t)  # captures the full 2x51 history
        executed = adapter.get_action(cond)  # (1, act_horizon, 4)
        env_actions = adapter.map_action_ee_delta_pose(executed)  # (1, act_horizon, 7)
        plan_steps = min(executed.shape[1], args.max_steps - elapsed)
        for k in range(plan_steps):
            obs, _, terminated, truncated, info = env.step(
                env_actions[0, k].cpu().numpy()[None]
            )
            elapsed += 1
            info = {
                key: np.asarray(value)
                for key, value in unwrap_final_info(info).items()
            }
            if info.get("obstacle_collision", np.array([False]))[0] and first_collision_step is None:
                first_collision_step = elapsed
            if info.get("obstacle_crossing", np.array([False]))[0] and first_crossing_step is None:
                first_crossing_step = elapsed
            if info.get("success", np.array([False]))[0]:
                terminated_reason = "success"
                break
        if terminated_reason == "success" or truncated or elapsed >= args.max_steps:
            break

    flags = {k: bool(np.asarray(info[k])[0]) for k in (
        "success", "stack_success", "obstacle_collision", "obstacle_crossing",
    ) if k in info}
    result.update(
        {
            "elapsed_steps": elapsed,
            "truncated": bool(truncated),
            "termination_reason": terminated_reason,
            "first_collision_step": first_collision_step,
            "first_crossing_step": first_crossing_step,
            **flags,
            "safe_success": bool(
                flags.get("stack_success", False)
                and not flags.get("obstacle_collision", False)
                and not flags.get("obstacle_crossing", False)
                and flags.get("success", False)
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = "disabled" if args.disable_control else "constrain"
    out_path = unique_path(args.output_dir / f"rollout_{tag}_seed{args.seed}_{args.device}.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    env.close()

    print(json.dumps(result, indent=2))
    print(f"saved: {out_path}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
