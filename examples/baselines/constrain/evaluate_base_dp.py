"""Serial randomized-scene evaluation of the base DP policy in one obstacle env.

Samples a fixed, configurable seed list of random scenes (cube layout + a
blocking vertical cylinder) with the generator ranges from
``collect_constructive``, filters invalid geometry with explicit finite-budget
rules, applies each scene per episode — including the real cylinder collision
radius — in a single CPU ``StackCubeObstacle-v1`` environment, and runs the
base DP policy closed-loop through ``BaseDPAdapter.base_cond`` /
``get_action`` / ``map_action_ee_delta_pose`` on the full 2x51 stacked
observation history. No expert, demonstration action, demonstration replay or
solvability validation is involved: actions come only from the policy closed
loop, and geometric validity of a scene says nothing about expert solvability.

Outputs (in a new unique run directory, never overwriting existing assets):
``manifest.json`` (settings + exact world-coordinate scenes), per-episode
``episodes/episode_XXX.json`` results, ``summary.json`` aggregate metrics, and
a ``preview.mp4`` rendered from the real rollouts (the frame after the
episode's final step is the fresh episode's auto-reset scene and is never
appended, since the episode's final observation is a state vector that cannot
be rendered). Episode flags are read from the gymnasium 0.29 ``final_info`` on
the auto-reset step, so contact, crossing, stacking or success occurring
exactly on the last step are recorded. Sampling-budget exhaustion
is reported explicitly and recorded separately from policy failures; the
requested episode count is never silently reduced. A recorded episode entry
can be replayed with ``--replay-episode`` to reproduce the exact layout and
result.
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.base_dp import DEFAULT_CKPT, BaseDPAdapter
from examples.baselines.constrain.collect_constructive import (
    CUBE_XY_RADIUS,
    CYLINDER_ENDPOINT_SLACK,
    Config,
    LayoutConfig,
    segment_point_distance,
)
from examples.baselines.constrain.rollout_base_dp import (
    build_env,
    unique_path,
    unwrap_final_info,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


class SceneSamplingError(RuntimeError):
    """The finite scene-sampling retry budget was exhausted."""


class SceneApplicationError(RuntimeError):
    """The environment did not apply the requested scene geometry."""


@dataclass(frozen=True)
class EvalSceneConfig:
    """Scene sampling ranges, reused from ``collect_constructive`` exactly."""

    cube_xy_range: tuple = LayoutConfig().cube_xy_range
    obstacle_xy_range: tuple = LayoutConfig().obstacle_xy_range
    radius_range: tuple = LayoutConfig().radius_range
    min_segment_clearance: float = LayoutConfig().min_segment_clearance
    max_cube_separation: float = LayoutConfig().max_separation
    max_attempts: int = LayoutConfig().max_attempts
    obstacle_height: float = Config().obstacle_height
    safety_margin: float = Config().safety_margin
    min_cube_separation: float = Config().min_cube_separation


FILTER_RULES = {
    "cube_separation": (
        "min_cube_separation <= |cubeB_xy - cubeA_xy| <= max_cube_separation"
    ),
    "initial_overlap": (
        "|cylinder_center_xy - cube_xy| >= radius + cube_xy_radius for both cubes"
    ),
    "endpoint_clearance": (
        "|cylinder_center_xy - cube_xy| >= radius + cube_xy_radius + safety_margin "
        "+ endpoint_slack for both cubes"
    ),
    "blocks_direct_segment": (
        "segment_distance(cylinder_center_xy, cubeA_xy, cubeB_xy) < "
        "radius + cube_xy_radius + safety_margin - min_segment_clearance"
    ),
}


def scene_seeds(base_seed: int, count: int) -> list:
    """The fixed seed list: scene ``i`` uses seed ``base_seed + i``."""
    return [base_seed + i for i in range(count)]


def sample_scene(rng: np.random.Generator, cfg: EvalSceneConfig):
    """Sample one filtered scene; returns ``(scene, None)`` or ``(None, reason)``.

    Cube positions, yaws, cylinder centre and radius are drawn uniformly from
    the ``collect_constructive`` ranges and then filtered: cube separation, no
    initial overlap, endpoint clearance, and the cylinder must block the
    direct A->B segment. Rejections are retried within ``max_attempts``.
    """
    box, obox = cfg.cube_xy_range, cfg.obstacle_xy_range
    for attempts_used in range(1, cfg.max_attempts + 1):
        cube_a = np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])])
        cube_b = np.array([rng.uniform(box[0], box[2]), rng.uniform(box[1], box[3])])
        yaw_a = float(rng.uniform(0.0, 2.0 * np.pi))
        yaw_b = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = float(rng.uniform(*cfg.radius_range))
        center = np.array(
            [rng.uniform(obox[0], obox[2]), rng.uniform(obox[1], obox[3])]
        )
        separation = float(np.linalg.norm(cube_b - cube_a))
        if not cfg.min_cube_separation <= separation <= cfg.max_cube_separation:
            continue
        forbidden = radius + CUBE_XY_RADIUS + cfg.safety_margin
        dist_a = float(np.linalg.norm(center - cube_a))
        dist_b = float(np.linalg.norm(center - cube_b))
        if min(dist_a, dist_b) < radius + CUBE_XY_RADIUS:
            continue  # initial overlap
        if min(dist_a, dist_b) < forbidden + CYLINDER_ENDPOINT_SLACK:
            continue  # endpoint clearance
        blocked = segment_point_distance(center, cube_a, cube_b)
        if blocked >= forbidden - cfg.min_segment_clearance:
            continue  # does not block the direct segment
        return (
            dict(
                cube_layout=dict(
                    cubeA_xy=[float(cube_a[0]), float(cube_a[1])],
                    cubeB_xy=[float(cube_b[0]), float(cube_b[1])],
                    cubeA_yaw=yaw_a,
                    cubeB_yaw=yaw_b,
                ),
                cylinder=dict(
                    center_xy=[float(center[0]), float(center[1])],
                    radius=radius,
                    height=float(cfg.obstacle_height),
                ),
                clearances=dict(
                    cubeA_endpoint_dist=dist_a,
                    cubeB_endpoint_dist=dist_b,
                    direct_segment_dist=blocked,
                    forbidden_radius=forbidden,
                ),
                attempts_used=attempts_used,
            ),
            None,
        )
    return None, "sampling_budget_exhausted"


def sample_scenes(base_seed: int, num_episodes: int, cfg: EvalSceneConfig) -> list:
    """Sample the full fixed seed list; raises :class:`SceneSamplingError`
    when any scene's finite retry budget is exhausted (never reduces the
    requested scene count silently)."""
    scenes = []
    for index, seed in enumerate(scene_seeds(base_seed, num_episodes)):
        scene, reason = sample_scene(np.random.default_rng(seed), cfg)
        if scene is None:
            raise SceneSamplingError(
                f"scene {index} (seed {seed}) sampling failed after "
                f"{cfg.max_attempts} attempts: {reason}"
            )
        scene["scene_index"] = index
        scene["seed"] = seed
        scenes.append(scene)
    return scenes


def summarize_results(records: list) -> dict:
    """Aggregate metrics from per-episode records (evaluated episodes only)."""
    evaluated = [r.get("result", r) for r in records
                 if r.get("status", "evaluated") == "evaluated"]
    total = len(evaluated)
    if total == 0:
        return {"total_episodes": 0}
    reasons: dict = {}
    for record in evaluated:
        reasons[record["termination_reason"]] = (
            reasons.get(record["termination_reason"], 0) + 1
        )
    rate = lambda key: sum(bool(r[key]) for r in evaluated) / total
    return dict(
        total_episodes=total,
        stack_success_rate=rate("stack_success"),
        obstacle_contact_rate=rate("obstacle_collision"),
        obstacle_crossing_rate=rate("obstacle_crossing"),
        safe_success_rate=rate("safe_success"),
        termination_reasons=reasons,
    )


class BaseDPEvaluator:
    """Serial per-scene closed-loop evaluation of the base DP policy.

    One CPU ``StackCubeObstacle-v1`` environment is reused across episodes;
    every episode reconfigures the exact sampled scene (cube layout, cylinder
    centre and real collision radius), reseeds the env/torch streams from the
    scene seed and starts from a fresh reset, so no observation history or
    policy state leaks across episodes. Actions come only from the adapter
    closed loop (``base_cond`` -> ``get_action`` -> ``map_action``); any
    adapter exposing that seam (e.g. a future ConstrainNet checkpoint wrapper)
    can be passed instead.
    """

    def __init__(
        self,
        num_episodes: int = 100,
        max_steps: int = 500,
        base_seed: int = 0,
        device: str = "cpu",
        checkpoint=None,
        scene_cfg: EvalSceneConfig = None,
        output_dir: Path = _REPO_ROOT / "outputs" / "base_dp_eval",
        video: bool = True,
        adapter: BaseDPAdapter = None,
    ):
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.device = device
        self.checkpoint = checkpoint
        self.scene_cfg = scene_cfg if scene_cfg is not None else EvalSceneConfig()
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.video = video
        self.adapter = adapter
        self.scenes = None
        self.run_dir = None
        self._env = None

    # -- setup -------------------------------------------------------------- #

    def setup(self):
        """Build the adapter and the single CPU env (lazy, after sampling)."""
        if self._env is not None:
            return
        if self.adapter is None:
            self.adapter = BaseDPAdapter(
                checkpoint=self.checkpoint if self.checkpoint is not None else DEFAULT_CKPT,
                device=self.device,
            )
        self._env = build_env(
            seed=self.base_seed,
            max_steps=self.max_steps,
            obstacle_height=self.scene_cfg.obstacle_height,
            obstacle_radius=Config().obstacle_radius_default,
        )

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def _base_env(self):
        return self._env.envs[0].unwrapped

    # -- per-episode rollout ------------------------------------------------- #

    def run_episode(self, scene: dict, writer=None) -> dict:
        """Apply one sampled scene and run one real closed-loop episode."""
        self.setup()
        cylinder, layout = scene["cylinder"], scene["cube_layout"]
        options = dict(
            reconfigure=True,
            obstacle_radius=float(cylinder["radius"]),
            obstacle_center=[float(v) for v in cylinder["center_xy"]],
            cube_layout=dict(layout),
        )
        obs, _ = self._env.reset(seed=int(scene["seed"]), options=options)
        applied = self._base_env().obstacle_config()
        self._verify_scene(scene, applied)
        torch.manual_seed(int(scene["seed"]))
        if writer is not None:
            writer.append_data(self._render())

        elapsed = 0
        truncated = False
        termination_reason = "max_steps"
        first_collision_step = None
        first_crossing_step = None
        stack_ever = False
        collision_ever = False
        crossing_ever = False
        success_ever = False
        first_action = None
        info = {}
        while elapsed < self.max_steps:
            stacked = torch.as_tensor(np.asarray(obs), dtype=torch.float32).to(self.device)
            cond = self.adapter.base_cond(stacked)  # full 2x51 history -> base condition
            executed = self.adapter.get_action(cond)
            env_actions = self.adapter.map_action_ee_delta_pose(executed)
            plan_steps = min(executed.shape[1], self.max_steps - elapsed)
            for k in range(plan_steps):
                obs, _, terminated, truncated, info = self._env.step(
                    env_actions[0, k].cpu().numpy()[None]
                )
                elapsed += 1
                if first_action is None:
                    first_action = [float(v) for v in env_actions[0, k].cpu().numpy()]
                # On the episode's final step gymnasium 0.29 auto-resets and
                # returns the reset info at the top level; the real step info
                # (and any flags occurring exactly there) lives in
                # ``final_info``, guarded by the ``_final_info`` mask.
                info = {
                    key: np.asarray(value)
                    for key, value in unwrap_final_info(info).items()
                }
                stack_ever |= bool(info.get("stack_success", np.array([False]))[0])
                collision_ever |= bool(
                    info.get("obstacle_collision", np.array([False]))[0]
                )
                crossing_ever |= bool(
                    info.get("obstacle_crossing", np.array([False]))[0]
                )
                success_ever |= bool(info.get("success", np.array([False]))[0])
                if collision_ever and first_collision_step is None:
                    first_collision_step = elapsed
                if crossing_ever and first_crossing_step is None:
                    first_crossing_step = elapsed
                if success_ever:
                    termination_reason = "success"
                # The returned observation on the auto-reset step is the fresh
                # episode's reset scene, not the episode's final observation
                # (a state vector that cannot be rendered), so it must not be
                # appended to the video.
                if writer is not None and not (
                    bool(np.asarray(terminated)[0]) or bool(np.asarray(truncated)[0])
                ):
                    writer.append_data(self._render())
                if termination_reason == "success":
                    break
            if termination_reason == "success" or truncated or elapsed >= self.max_steps:
                break

        result = dict(
            elapsed_steps=elapsed,
            truncated=bool(truncated),
            termination_reason=termination_reason,
            first_collision_step=first_collision_step,
            first_crossing_step=first_crossing_step,
            success=bool(success_ever),
            stack_success=bool(stack_ever),
            obstacle_collision=bool(collision_ever),
            obstacle_crossing=bool(crossing_ever),
            safe_success=bool(
                stack_ever and not collision_ever and not crossing_ever
            ),
        )
        return dict(
            scene_index=int(scene["scene_index"]),
            seed=int(scene["seed"]),
            status="evaluated",
            scene=dict(
                cube_layout=dict(scene["cube_layout"]),
                cylinder=dict(scene["cylinder"]),
                clearances=dict(scene["clearances"]),
            ),
            applied=dict(
                obstacle_collision_radius=float(applied["radius"]),
                obstacle_center=[float(v) for v in applied["center"]],
                cube_layout=dict(applied["cube_layout"]),
            ),
            result=result,
            first_action=first_action,
            run=dict(
                max_steps=self.max_steps,
                base_seed=self.base_seed,
                device=self.device,
                checkpoint=str(
                    self.checkpoint if self.checkpoint is not None else DEFAULT_CKPT
                ),
            ),
        )

    @staticmethod
    def _verify_scene(scene: dict, applied: dict):
        cylinder, layout = scene["cylinder"], scene["cube_layout"]
        if abs(float(applied["radius"]) - float(cylinder["radius"])) > 1e-6:
            raise SceneApplicationError(
                f"applied obstacle radius {applied['radius']} != requested "
                f"{cylinder['radius']}"
            )
        if not np.allclose(applied["center"], cylinder["center_xy"], atol=1e-6):
            raise SceneApplicationError(
                f"applied obstacle center {applied['center']} != requested "
                f"{cylinder['center_xy']}"
            )
        for cube in ("cubeA", "cubeB"):
            if not np.allclose(
                applied["cube_layout"][f"{cube}_xy"], layout[f"{cube}_xy"], atol=1e-6
            ):
                raise SceneApplicationError(
                    f"applied {cube} position {applied['cube_layout'][f'{cube}_xy']} "
                    f"!= requested {layout[f'{cube}_xy']}"
                )
            if abs(applied["cube_layout"][f"{cube}_yaw"] - layout[f"{cube}_yaw"]) > 1e-6:
                raise SceneApplicationError(
                    f"applied {cube} yaw {applied['cube_layout'][f'{cube}_yaw']} "
                    f"!= requested {layout[f'{cube}_yaw']}"
                )

    def _render(self) -> np.ndarray:
        from mani_skill.utils import common

        frame = common.to_numpy(self._base_env().render())
        if frame.ndim == 4:
            frame = frame[0]
        return frame.astype(np.uint8)

    # -- full run ------------------------------------------------------------ #

    def _settings(self) -> dict:
        return dict(
            num_episodes_requested=self.num_episodes,
            max_steps=self.max_steps,
            base_seed=self.base_seed,
            scene_seed_rule=(
                "fixed seed list: scene i uses seed base_seed + i; each episode "
                "resets the env with its scene seed and reseeds torch with it "
                "before policy inference"
            ),
            network_device=self.device,
            checkpoint=str(
                self.checkpoint if self.checkpoint is not None else DEFAULT_CKPT
            ),
            weights_subkey=self.adapter.metadata["weights_subkey"],
            sim_backend="physx_cpu",
            num_envs=1,
            control_mode="pd_ee_delta_pose",
            obs_mode="state",
            robot_uids="panda_wristcam",
            obstacle_height=self.scene_cfg.obstacle_height,
            scene_filter=dict(
                rules=dict(FILTER_RULES),
                params={k: list(v) if isinstance(v, tuple) else v
                        for k, v in asdict(self.scene_cfg).items()},
                cube_xy_radius=CUBE_XY_RADIUS,
                endpoint_slack=CYLINDER_ENDPOINT_SLACK,
            ),
            video=self.video,
        )

    def evaluate_all(self) -> dict:
        """Sample the fixed seed list, evaluate every scene, write outputs."""
        if self.output_dir is not None:
            self.run_dir = unique_path(self.output_dir)
            self.run_dir.mkdir(parents=True)
        self.scenes = []
        sampling_failures = []
        try:
            for index, seed in enumerate(scene_seeds(self.base_seed, self.num_episodes)):
                scene, reason = sample_scene(np.random.default_rng(seed), self.scene_cfg)
                if scene is None:
                    sampling_failures.append(
                        dict(
                            scene_index=index,
                            seed=seed,
                            status="sampling_exhausted",
                            reason=reason,
                            attempts_used=self.scene_cfg.max_attempts,
                        )
                    )
                    raise SceneSamplingError(
                        f"scene {index} (seed {seed}) sampling failed after "
                        f"{self.scene_cfg.max_attempts} attempts: {reason}"
                    )
                scene["scene_index"] = index
                scene["seed"] = seed
                self.scenes.append(scene)
            self.setup()
            self._write_json(
                "manifest.json",
                dict(settings=self._settings(), scenes=self.scenes,
                     sampling_failures=sampling_failures),
            )
            records = []
            writer, temporary = self._open_writer()
            try:
                for scene in self.scenes:
                    record = self.run_episode(scene, writer=writer)
                    self._write_json(
                        f"episodes/episode_{record['scene_index']:03d}.json", record
                    )
                    records.append(record)
            finally:
                self._close_writer(writer, temporary)
            summary = dict(
                run_dir=str(self.run_dir),
                settings=self._settings(),
                metrics=summarize_results(records),
                sampling_failures=sampling_failures,
                episodes=[
                    dict(
                        scene_index=record["scene_index"],
                        seed=record["seed"],
                        status=record["status"],
                        path=str(
                            self.run_dir
                            / f"episodes/episode_{record['scene_index']:03d}.json"
                        ),
                    )
                    for record in records
                ],
            )
            self._write_json("summary.json", summary)
            return summary
        except SceneSamplingError:
            if self.run_dir is not None:
                self._write_json(
                    "manifest.json",
                    dict(
                        settings=self._settings_if_built(),
                        scenes=self.scenes,
                        sampling_failures=sampling_failures,
                    ),
                )
            raise

    def _settings_if_built(self) -> dict:
        settings = {
            "num_episodes_requested": self.num_episodes,
            "max_steps": self.max_steps,
            "base_seed": self.base_seed,
            "network_device": self.device,
            "video": self.video,
        }
        return settings

    def _open_writer(self):
        if not self.video or self.run_dir is None:
            return None, None
        import imageio.v2 as imageio

        temporary = self.run_dir / ".preview.tmp.mp4"
        writer = imageio.get_writer(
            str(temporary), fps=self._control_freq(), codec="libx264",
            macro_block_size=1,
        )
        return writer, temporary

    def _close_writer(self, writer, temporary):
        if writer is None:
            return
        writer.close()
        if temporary.exists():
            temporary.replace(self.run_dir / "preview.mp4")

    def _control_freq(self) -> float:
        try:
            return float(self._base_env().control_freq)
        except Exception:
            return 20.0

    def _write_json(self, relative: str, payload: dict):
        if self.run_dir is None:
            return
        path = self.run_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")


def replay_episode(record: dict, device: str = None) -> dict:
    """Replay one recorded episode entry: rebuild the exact recorded scene
    layout, cylinder geometry and seeds, rerun the policy closed loop and
    verify the reproduced result matches the recorded one."""
    run = record["run"]
    evaluator = BaseDPEvaluator(
        max_steps=int(run["max_steps"]),
        base_seed=int(run["base_seed"]),
        device=device if device is not None else run["device"],
        checkpoint=Path(run["checkpoint"]),
        output_dir=None,
        video=False,
    )
    try:
        scene = dict(record["scene"])
        scene["scene_index"] = int(record["scene_index"])
        scene["seed"] = int(record["seed"])
        reproduced = evaluator.run_episode(scene)
    finally:
        evaluator.close()
    if reproduced["result"] != record["result"]:
        raise ValueError(
            "replay result mismatch:\n"
            f"recorded:   {json.dumps(record['result'])}\n"
            f"reproduced: {json.dumps(reproduced['result'])}"
        )
    return reproduced


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=0, help="base seed; scene i uses seed+i")
    p.add_argument("--device", type=str, default="cpu", help="network device (cpu or cuda)")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "outputs" / "base_dp_eval")
    p.add_argument("--no-video", action="store_true")
    p.add_argument(
        "--replay-episode", type=Path, default=None,
        help="replay one recorded episodes/episode_XXX.json and compare results",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.replay_episode is not None:
        record = json.loads(args.replay_episode.read_text())
        reproduced = replay_episode(record, device=args.device)
        print(json.dumps(reproduced["result"], indent=2))
        print(
            "replay reproduced the recorded result exactly "
            f"(scene {record['scene_index']}, seed {record['seed']})"
        )
        return reproduced
    evaluator = BaseDPEvaluator(
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        base_seed=args.seed,
        device=args.device,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        video=not args.no_video,
    )
    summary = evaluator.evaluate_all()
    evaluator.close()
    print(json.dumps(summary["metrics"], indent=2))
    print(f"saved: {summary['run_dir']}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
