"""Launch the unmodified official PegInsertionSide-v1 Diffusion Policy entrypoint.

The launcher only resolves the official command, working directory, and checkpoint
symlink. It does not change the training algorithm or official defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_TRAIN = ROOT / "examples/baselines/diffusion_policy/train.py"
EXP_NAME = "diffusion_policy-PegInsertionSide-v1-state-100_motionplanning_demos-1"


def official_train_argv(demo_path: str, exp_name: str, *, track: bool) -> list[str]:
    argv = [
        sys.executable,
        str(OFFICIAL_TRAIN),
        "--env-id",
        "PegInsertionSide-v1",
        "--demo-path",
        demo_path,
        "--control-mode",
        "pd_ee_delta_pose",
        "--sim-backend",
        "physx_cpu",
        "--num-demos",
        "100",
        "--max_episode_steps",
        "300",
        "--total_iters",
        "100000",
        "--exp-name",
        exp_name,
        "--demo_type",
        "motionplanning",
    ]
    if track:
        argv.append("--track")
    return argv


def prepare_official_output(exp_dir: Path, ckpt_dir: Path, exp_name: str) -> Path:
    exp_dir = Path(exp_dir)
    ckpt_dir = Path(ckpt_dir)
    if ckpt_dir.exists() or ckpt_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing checkpoint path: {ckpt_dir}")
    link_parent = exp_dir / "runs" / exp_name
    link = link_parent / "checkpoints"
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"refusing to replace existing checkpoint link: {link}")
    link_parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True)
    link.symlink_to(os.path.relpath(ckpt_dir, link_parent), target_is_directory=True)
    return link


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--demo-path", required=True)
    parser.add_argument("--exp-name", default=EXP_NAME)
    parser.add_argument("--track", action="store_true")
    args = parser.parse_args(argv)
    prepare_official_output(args.exp_dir, args.ckpt_dir, args.exp_name)
    demo_path = str(Path(args.demo_path).resolve())
    command = official_train_argv(demo_path, args.exp_name, track=args.track)
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    (args.exp_dir / "resolved_command.json").write_text(
        json.dumps(
            {
                "cwd": str(args.exp_dir.resolve()),
                "ckpt_dir": str(args.ckpt_dir.resolve()),
                "argv": command,
                "track": args.track,
                "seed_flag_passed": False,
            },
            indent=2,
        )
        + "\n"
    )
    os.chdir(args.exp_dir)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
