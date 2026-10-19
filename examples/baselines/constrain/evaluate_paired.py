"""Paired closed-loop evaluation: base DP vs ConstrainNet on identical scenes.

One shared scene list is sampled exactly once (fixed seed rule: scene ``i``
uses seed ``base_seed + i``, default ``base_seed 10000`` — fresh scenes, never
the training demonstrations) and fed to two serial single-environment
``BaseDPEvaluator`` runs:

- ``base_dp`` — the frozen official base DP policy (its own checkpoint),
- ``constrain_dp`` — the trained ConstrainNet policy loaded strictly from its
  standalone checkpoint file (final EMA weights) in this independent process.

Both policies act through the identical rollout seam
(``base_cond -> get_action -> map_action_ee_delta_pose``) on the same saved
scene configs, env settings and per-episode seed list; per-episode torch
reseeding makes each policy's noise stream comparable. Sampling failures are
recorded separately from policy failures and a policy that fails mid-run is
reported with its error while the other policy's results are kept. Outputs go
to a new unique run directory: ``scenes_manifest.json`` (the shared scene
configs), per-policy subdirectories with the evaluator's standard
``manifest.json`` / ``episodes/`` / ``summary.json`` / ``preview.mp4``
(recording restricted to predesignated scene indexes when provided), and a
merged ``paired_summary.json`` with metrics, scene-equality evidence, actual
completed counts and wall-clock timings.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from examples.baselines.constrain.evaluate_base_dp import (
    BaseDPEvaluator,
    EvalSceneConfig,
    sample_scenes,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

POLICY_NAMES = ("base_dp", "constrain_dp")


def _scene_filter_payload(scene_cfg: EvalSceneConfig) -> dict:
    from dataclasses import asdict

    from examples.baselines.constrain.collect_constructive import (
        CUBE_XY_RADIUS,
        CYLINDER_ENDPOINT_SLACK,
    )
    from examples.baselines.constrain.evaluate_base_dp import FILTER_RULES

    return dict(
        rules=dict(FILTER_RULES),
        params={k: list(v) if isinstance(v, tuple) else v
                for k, v in asdict(scene_cfg).items()},
        cube_xy_radius=CUBE_XY_RADIUS,
        endpoint_slack=CYLINDER_ENDPOINT_SLACK,
    )


def run_paired_evaluation(
    base_adapter,
    constrain_adapter,
    num_episodes=100,
    max_steps=500,
    base_seed=10000,
    device="cpu",
    output_dir=_REPO_ROOT / "outputs" / "constrain_runs" / "paired_eval",
    scene_cfg=None,
    video_scene_indexes=None,
) -> dict:
    """Evaluate both policies on one shared sampled scene list; merge results."""
    cfg = scene_cfg if scene_cfg is not None else EvalSceneConfig()
    scenes = sample_scenes(base_seed, num_episodes, cfg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes_manifest = dict(
        base_seed=base_seed,
        num_episodes=num_episodes,
        max_steps=max_steps,
        scene_seed_rule="fixed seed list: scene i uses seed base_seed + i",
        scene_filter=_scene_filter_payload(cfg),
        scenes=scenes,
    )
    (output_dir / "scenes_manifest.json").write_text(
        json.dumps(scenes_manifest, indent=2) + "\n"
    )

    adapters = {"base_dp": base_adapter, "constrain_dp": constrain_adapter}
    results = {}
    timings = {}
    for name in POLICY_NAMES:
        evaluator = BaseDPEvaluator(
            num_episodes=num_episodes,
            max_steps=max_steps,
            base_seed=base_seed,
            device=device,
            scene_cfg=cfg,
            output_dir=output_dir / name,
            video=video_scene_indexes is None or len(video_scene_indexes) > 0,
            adapter=adapters[name],
            video_scene_indexes=video_scene_indexes,
        )
        started = time.time()
        summary = None
        error = None
        status = "completed"
        try:
            summary = evaluator.evaluate_all(scenes=scenes)
        except Exception as exc:  # record the runtime failure; keep the other policy
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            evaluator.close()
        timings[name] = round(time.time() - started, 1)
        results[name] = dict(
            status=status,
            error=error,
            summary=summary,
            metrics=dict(summary["metrics"]) if summary is not None else None,
            run_dir=str(output_dir / name),
            adapter_metadata=dict(adapters[name].metadata),
        )

    scene_equality = None
    if all(r["status"] == "completed" for r in results.values()):
        manifests = [
            json.loads((Path(r["run_dir"]) / "manifest.json").read_text())
            for r in results.values()
        ]
        scene_equality = all(
            m["scenes"] == manifests[0]["scenes"] for m in manifests[1:]
        )

    paired = dict(
        settings=dict(
            num_episodes_requested=num_episodes,
            max_steps=max_steps,
            base_seed=base_seed,
            network_device=device,
            sim_backend="physx_cpu",
            num_envs=1,
            video_scene_indexes=(
                sorted(video_scene_indexes) if video_scene_indexes is not None else None
            ),
            scene_seed_rule=scenes_manifest["scene_seed_rule"],
        ),
        scenes_manifest=str(output_dir / "scenes_manifest.json"),
        scene_equality_verified=scene_equality,
        policies=results,
        timings_sec=timings,
    )
    (output_dir / "paired_summary.json").write_text(json.dumps(paired, indent=2) + "\n")
    return paired


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--constrain-checkpoint", type=Path, required=True,
                   help="standalone constrain checkpoint (final.pt) to evaluate from disk")
    p.add_argument("--base-checkpoint", type=Path, default=None,
                   help="base DP checkpoint (default: the official default)")
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=10000,
                   help="base seed; scene i uses seed+i (fresh scenes, not training demos)")
    p.add_argument("--device", type=str, default="cpu", help="network device")
    p.add_argument("--output-dir", type=Path,
                   default=_REPO_ROOT / "outputs" / "constrain_runs" / "paired_eval")
    p.add_argument("--video-episodes", type=str, default=None,
                   help="comma-separated scene indexes to record into preview.mp4 "
                        "(e.g. '0,17,42'); default: all episodes")
    p.add_argument("--no-video", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from examples.baselines.constrain.base_dp import DEFAULT_CKPT, BaseDPAdapter
    from examples.baselines.constrain.constrain_adapter import ConstrainDPAdapter

    base_adapter = BaseDPAdapter(
        checkpoint=args.base_checkpoint if args.base_checkpoint is not None else DEFAULT_CKPT,
        device=args.device,
    )
    constrain_adapter = ConstrainDPAdapter(
        args.constrain_checkpoint, device=args.device, weights="ema"
    )
    if args.no_video:
        video_indexes = []
    elif args.video_episodes:
        video_indexes = [int(v) for v in args.video_episodes.split(",")]
    else:
        video_indexes = None
    paired = run_paired_evaluation(
        base_adapter=base_adapter,
        constrain_adapter=constrain_adapter,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        base_seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        video_scene_indexes=video_indexes,
    )
    print(json.dumps({
        name: dict(
            status=pol["status"],
            error=pol["error"],
            metrics=pol["metrics"],
            run_dir=pol["run_dir"],
        )
        for name, pol in paired["policies"].items()
    } | dict(
        scene_equality_verified=paired["scene_equality_verified"],
        timings_sec=paired["timings_sec"],
    ), indent=2))
    print(f"saved: {Path(paired['scenes_manifest']).parent / 'paired_summary.json'}")
    return paired


if __name__ == "__main__":
    main(sys.argv[1:])
