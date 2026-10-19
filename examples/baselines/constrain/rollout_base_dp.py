"""Reproducible single-episode closed-loop rollout of the base DP policy.

Loads the obstacle-free StackCube DP checkpoint through
``BaseDPAdapter`` and executes it closed-loop in one ``StackCubeObstacle-v1``
episode on the CPU physics backend (network device configurable):

- the queued 51d obstacle obs history is trimmed to the base 48d condition,
- the official epsilon/DDPM rollout executes the 8-step action horizon,
- each executed action is the 4d base output mapped to the 7d
  ``pd_ee_delta_pose`` action with three zero rotations appended,
- termination reason, stack success, obstacle contact, forbidden crossing and
  the safe-success flag are recorded in a JSON file under ``outputs/``.

The obstacle geometry defaults to the demo configuration: height 0.18 m and a
radius sampled in 0.02-0.04 m. This is a behavior record, not an avoidance
acceptance gate.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
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
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "base_dp_rollout")
    return p.parse_args(argv)


def build_env(args, obstacle_radius):
    import gymnasium as gym

    import examples.baselines.constrain.stackcube_obstacle  # noqa: F401 (registers env)
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

    def make_env():
        env = gym.make(
            "StackCubeObstacle-v1",
            reconfiguration_freq=1,
            robot_uids="panda_wristcam",
            control_mode="pd_ee_delta_pose",
            reward_mode="sparse",
            obs_mode="state",
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            num_envs=1,
            max_episode_steps=args.max_steps,
            obstacle_radius=obstacle_radius,
            obstacle_height=args.obstacle_height,
            human_render_camera_configs=dict(shader_pack="default"),
        )
        env = FrameStack(env, num_stack=2)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        return env

    return gym.vector.SyncVectorEnv([make_env])


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while path.with_name(f"{path.stem}-{i}{path.suffix}").exists():
        i += 1
    return path.with_name(f"{path.stem}-{i}{path.suffix}")


def main(argv=None):
    args = parse_args(argv)
    if args.obstacle_radius is None:
        rng = np.random.default_rng(args.seed)
        args.obstacle_radius = float(rng.uniform(0.02, 0.04))
    if not 0.02 <= args.obstacle_radius <= 0.04:
        raise ValueError(f"obstacle radius {args.obstacle_radius} outside [0.02, 0.04]")

    from examples.baselines.constrain.base_dp import DEFAULT_CKPT, BaseDPAdapter

    adapter = BaseDPAdapter(
        checkpoint=args.checkpoint if args.checkpoint is not None else DEFAULT_CKPT,
        device=args.device,
    )
    torch.manual_seed(args.seed)

    env = build_env(args, args.obstacle_radius)
    obs, _ = env.reset(seed=args.seed)

    result = {
        "weights_source": adapter.metadata["weights_source"],
        "weights_subkey": adapter.metadata["weights_subkey"],
        "network_device": args.device,
        "sim_backend": "physx_cpu",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "obstacle": {"radius": args.obstacle_radius, "height": args.obstacle_height},
        "pred_horizon": adapter.metadata["pred_horizon"],
        "act_horizon": adapter.metadata["act_horizon"],
        "obs_horizon": adapter.metadata["obs_horizon"],
    }

    first_collision_step = None
    first_crossing_step = None
    elapsed = 0
    terminated_reason = "max_steps"
    truncated = False

    while elapsed < args.max_steps:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(args.device)
        executed = adapter.get_action(obs_t[..., :48])  # (B, act_horizon, 4)
        env_actions = adapter.map_action_ee_delta_pose(executed)  # (B, act_horizon, 7)
        plan_steps = min(executed.shape[1], args.max_steps - elapsed)
        for k in range(plan_steps):
            obs, _, terminated, truncated, info = env.step(
                env_actions[0, k].cpu().numpy()[None]
            )
            elapsed += 1
            info = {k: np.asarray(v) for k, v in info.items()}
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
    out_path = unique_path(args.output_dir / f"rollout_seed{args.seed}_{args.device}.json")
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    env.close()

    print(json.dumps(result, indent=2))
    print(f"saved: {out_path}")
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
