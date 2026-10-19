"""Tests for randomized-scene evaluation of the base DP policy (Ticket 02).

Seams under test:
- :func:`sample_scene` — deterministic, finite-budget scene sampling whose
  accepted geometry satisfies the recorded filter rules (cube separation, no
  initial overlap, endpoint clearance, blocks the direct A->B segment) inside
  the generator ranges from ``collect_constructive``.
- :class:`BaseDPEvaluator` — per-scene application of the exact layout and
  real cylinder collision radius in one CPU env, closed-loop base DP actions
  through ``BaseDPAdapter.base_cond/get_action/map_action_ee_delta_pose``,
  per-episode history reset without cross-episode leakage, and reproducible
  results from a recorded scene manifest entry.
- :func:`summarize_results` — aggregate stack/contact/crossing/safe-success
  metrics from per-episode records.

Real-episode tests use the actual trained base checkpoint with tiny step
limits; no expert, demonstration action, or solvability check is involved.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.baselines.constrain.collect_constructive import (
    CUBE_XY_RADIUS,
    CYLINDER_ENDPOINT_SLACK,
    Config,
    LayoutConfig,
    segment_point_distance,
)
from examples.baselines.constrain.evaluate_base_dp import (
    BaseDPEvaluator,
    EvalSceneConfig,
    SceneSamplingError,
    replay_episode,
    sample_scene,
    sample_scenes,
    summarize_results,
)


def forbidden_radius(cfg, radius):
    return radius + CUBE_XY_RADIUS + cfg.safety_margin


def test_eval_scene_config_uses_collect_constructive_ranges_exactly():
    cfg = EvalSceneConfig()
    layout, full = LayoutConfig(), Config()
    assert cfg.cube_xy_range == layout.cube_xy_range
    assert cfg.obstacle_xy_range == layout.obstacle_xy_range
    assert cfg.radius_range == layout.radius_range
    assert cfg.min_segment_clearance == layout.min_segment_clearance
    assert cfg.max_cube_separation == layout.max_separation
    assert cfg.max_attempts == layout.max_attempts
    assert cfg.obstacle_height == full.obstacle_height == 0.18
    assert cfg.safety_margin == full.safety_margin
    assert cfg.min_cube_separation == full.min_cube_separation


def test_sample_scene_is_deterministic_for_the_same_seed():
    cfg = EvalSceneConfig()
    scene_a, err_a = sample_scene(np.random.default_rng(7), cfg)
    scene_b, err_b = sample_scene(np.random.default_rng(7), cfg)
    assert err_a is None and err_b is None
    assert scene_a == scene_b


def test_sample_scene_geometry_satisfies_recorded_filter_rules():
    cfg = EvalSceneConfig()
    box, obox = cfg.cube_xy_range, cfg.obstacle_xy_range
    checked = 0
    for seed in range(60):
        scene, reason = sample_scene(np.random.default_rng(seed), cfg)
        if scene is None:
            continue
        checked += 1
        layout, cylinder = scene["cube_layout"], scene["cylinder"]
        cube_a = np.asarray(layout["cubeA_xy"])
        cube_b = np.asarray(layout["cubeB_xy"])
        center = np.asarray(cylinder["center_xy"])
        radius = cylinder["radius"]
        for xy in (cube_a, cube_b):
            assert box[0] <= xy[0] <= box[2] and box[1] <= xy[1] <= box[3]
        assert box[0] <= center[0] <= obox[2] and box[1] <= center[1] <= obox[3]
        assert obox[0] <= center[0] <= obox[2] and obox[1] <= center[1] <= obox[3]
        assert cfg.radius_range[0] <= radius <= cfg.radius_range[1]
        assert cylinder["height"] == cfg.obstacle_height
        for yaw in (layout["cubeA_yaw"], layout["cubeB_yaw"]):
            assert 0.0 <= yaw < 2.0 * np.pi
        separation = float(np.linalg.norm(cube_b - cube_a))
        assert cfg.min_cube_separation <= separation <= cfg.max_cube_separation
        forbidden = forbidden_radius(cfg, radius)
        dist_a = float(np.linalg.norm(center - cube_a))
        dist_b = float(np.linalg.norm(center - cube_b))
        assert min(dist_a, dist_b) >= forbidden + CYLINDER_ENDPOINT_SLACK
        blocked = segment_point_distance(center, cube_a, cube_b)
        assert blocked < forbidden - cfg.min_segment_clearance
        clearances = scene["clearances"]
        assert clearances["forbidden_radius"] == pytest.approx(forbidden)
        assert clearances["direct_segment_dist"] == pytest.approx(blocked)
    assert checked >= 20


def test_sample_scene_exhausts_finite_budget_explicitly():
    cfg = EvalSceneConfig(max_attempts=0)
    scene, reason = sample_scene(np.random.default_rng(0), cfg)
    assert scene is None
    assert reason == "sampling_budget_exhausted"


def test_sample_scenes_raises_explicit_error_on_exhaustion():
    cfg = EvalSceneConfig(max_attempts=0)
    with pytest.raises(SceneSamplingError) as excinfo:
        sample_scenes(base_seed=0, num_episodes=3, cfg=cfg)
    assert "sampling_budget_exhausted" in str(excinfo.value)


def test_sample_scenes_seed_list_is_fixed_and_indexed():
    scenes = sample_scenes(base_seed=100, num_episodes=3, cfg=EvalSceneConfig())
    assert [scene["scene_index"] for scene in scenes] == [0, 1, 2]
    assert [scene["seed"] for scene in scenes] == [100, 101, 102]
    replay = sample_scenes(base_seed=100, num_episodes=3, cfg=EvalSceneConfig())
    assert scenes == replay


def test_summarize_results_rates_and_reasons():
    records = [
        {"status": "evaluated", "stack_success": True, "obstacle_collision": False,
         "obstacle_crossing": False, "safe_success": True, "termination_reason": "success"},
        {"status": "evaluated", "stack_success": True, "obstacle_collision": True,
         "obstacle_crossing": False, "safe_success": False, "termination_reason": "max_steps"},
        {"status": "evaluated", "stack_success": False, "obstacle_collision": False,
         "obstacle_crossing": True, "safe_success": False, "termination_reason": "max_steps"},
        {"status": "evaluated", "stack_success": False, "obstacle_collision": False,
         "obstacle_crossing": False, "safe_success": False, "termination_reason": "max_steps"},
    ]
    metrics = summarize_results(records)
    assert metrics["total_episodes"] == 4
    assert metrics["stack_success_rate"] == pytest.approx(0.5)
    assert metrics["obstacle_contact_rate"] == pytest.approx(0.25)
    assert metrics["obstacle_crossing_rate"] == pytest.approx(0.25)
    assert metrics["safe_success_rate"] == pytest.approx(0.25)
    assert metrics["termination_reasons"] == {"success": 1, "max_steps": 3}


def test_safe_success_requires_stack_without_contact_or_crossing():
    records = [
        {"stack_success": True, "obstacle_collision": True, "obstacle_crossing": False,
         "safe_success": False, "termination_reason": "max_steps"},
        {"stack_success": True, "obstacle_collision": False, "obstacle_crossing": True,
         "safe_success": False, "termination_reason": "max_steps"},
    ]
    metrics = summarize_results(records)
    assert metrics["safe_success_rate"] == pytest.approx(0.0)
    assert metrics["stack_success_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Real closed-loop runner (actual checkpoint, tiny step budgets)
# --------------------------------------------------------------------------- #

TINY_KWARGS = dict(max_steps=2, video=False, device="cpu")


def make_evaluator(tmp_path, **overrides):
    kwargs = dict(TINY_KWARGS)
    kwargs.update(overrides)
    return BaseDPEvaluator(output_dir=tmp_path / "eval_run", **kwargs)


def test_base_cond_seam_composes_with_get_action_and_map_action():
    """The runner's policy seam: full 2x51 history -> base_cond -> get_action,
    then map_action to the 7d env action; identical condition and noise stream
    as trimming the frames directly."""
    from examples.baselines.constrain.base_dp import BaseDPAdapter

    adapter = BaseDPAdapter(device="cpu")
    frames = torch.randn(1, 2, 51)
    torch.manual_seed(3)
    executed_a = adapter.get_action(adapter.base_cond(frames))
    torch.manual_seed(3)
    executed_b = adapter.get_action(frames[..., :48])
    torch.testing.assert_close(executed_a, executed_b)
    env_action = adapter.map_action_ee_delta_pose(executed_a)
    assert env_action.shape == (1, adapter.metadata["act_horizon"], 7)
    torch.testing.assert_close(env_action[..., 3:6], torch.zeros(1, 8, 3))
    torch.testing.assert_close(env_action[..., 6:7], executed_a[..., 3:4])


def test_runner_applies_scene_layout_and_real_cylinder_radius(tmp_path):
    evaluator = make_evaluator(tmp_path)
    try:
        scene = sample_scenes(base_seed=0, num_episodes=1, cfg=EvalSceneConfig())[0]
        record = evaluator.run_episode(scene)
        assert record["status"] == "evaluated"
        applied = record["applied"]
        assert applied["obstacle_collision_radius"] == pytest.approx(
            scene["cylinder"]["radius"], abs=1e-6
        )
        assert np.allclose(
            applied["obstacle_center"], scene["cylinder"]["center_xy"], atol=1e-6
        )
        for cube in ("cubeA", "cubeB"):
            assert np.allclose(
                applied["cube_layout"][f"{cube}_xy"],
                scene["cube_layout"][f"{cube}_xy"],
                atol=1e-6,
            )
            assert applied["cube_layout"][f"{cube}_yaw"] == pytest.approx(
                scene["cube_layout"][f"{cube}_yaw"], abs=1e-6
            )
        result = record["result"]
        assert result["elapsed_steps"] == 2
        assert result["termination_reason"] == "max_steps"
        assert record["first_action"] is not None and len(record["first_action"]) == 7
    finally:
        evaluator.close()


def test_runner_resets_policy_state_no_cross_episode_leakage(tmp_path):
    """Episode 2 run after episode 1 in the same env must be identical to
    episode 2 run alone in a fresh evaluator: obs histories and the policy
    noise stream are reseeded per episode."""
    scenes = sample_scenes(base_seed=0, num_episodes=2, cfg=EvalSceneConfig())
    continued = make_evaluator(tmp_path)
    try:
        continued.run_episode(scenes[0])
        record_second = continued.run_episode(scenes[1])
    finally:
        continued.close()
    fresh = make_evaluator(tmp_path)
    try:
        record_fresh = fresh.run_episode(scenes[1])
    finally:
        fresh.close()
    assert record_second["result"] == record_fresh["result"]
    assert record_second["first_action"] == record_fresh["first_action"]
    assert record_second["scene"] == record_fresh["scene"]


def test_replay_episode_reproduces_recorded_result(tmp_path):
    scenes = sample_scenes(base_seed=0, num_episodes=1, cfg=EvalSceneConfig())
    original = make_evaluator(tmp_path)
    try:
        record = original.run_episode(scenes[0])
    finally:
        original.close()
    reproduced = replay_episode(record, device="cpu")
    assert reproduced["result"] == record["result"]
    assert reproduced["first_action"] == record["first_action"]
    assert reproduced["applied"]["obstacle_collision_radius"] == pytest.approx(
        record["applied"]["obstacle_collision_radius"], abs=1e-6
    )


def test_evaluate_all_writes_manifest_episodes_summary(tmp_path):
    evaluator = make_evaluator(tmp_path, num_episodes=2)
    summary = evaluator.evaluate_all()
    run_dir = Path(summary["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["scenes"]) == 2
    assert manifest["scenes"][0]["seed"] == 0 and manifest["scenes"][1]["seed"] == 1
    assert manifest["sampling_failures"] == []
    settings = manifest["settings"]
    assert settings["num_episodes_requested"] == 2
    assert settings["max_steps"] == 2
    assert settings["sim_backend"] == "physx_cpu"
    assert settings["scene_seed_rule"]
    assert settings["scene_filter"]["rules"]["blocks_direct_segment"]
    assert settings["scene_filter"]["params"]["radius_range"] == [0.02, 0.04]
    assert settings["scene_filter"]["params"]["obstacle_height"] == 0.18
    episode_paths = [Path(entry["path"]) for entry in summary["episodes"]]
    assert all(path.exists() for path in episode_paths)
    record = json.loads(episode_paths[1].read_text())
    assert record["scene_index"] == 1 and record["seed"] == 1
    assert record["run"]["max_steps"] == 2
    assert summary["metrics"]["total_episodes"] == 2


def test_evaluate_all_records_sampling_failure_separately(tmp_path):
    evaluator = make_evaluator(
        tmp_path, num_episodes=2, scene_cfg=EvalSceneConfig(max_attempts=0)
    )
    with pytest.raises(SceneSamplingError):
        evaluator.evaluate_all()
    manifest = json.loads((tmp_path / "eval_run" / "manifest.json").read_text())
    failures = manifest["sampling_failures"]
    assert len(failures) == 1
    assert failures[0]["status"] == "sampling_exhausted"
    assert failures[0]["scene_index"] == 0
    assert failures[0]["reason"] == "sampling_budget_exhausted"
    assert manifest["settings"]["num_episodes_requested"] == 2


# --------------------------------------------------------------------------- #
# Terminal-step info regression: the gymnasium 0.29 SyncVectorEnv auto-reset  #
# moves the real final step info into ``final_info`` and returns the reset    #
# info at the top level, so flags occurring exactly on the last step must be  #
# read from there.                                                            #
# --------------------------------------------------------------------------- #

EPISODE_FLAGS = ("success", "stack_success", "obstacle_collision", "obstacle_crossing")


class _StubAdapter:
    """Minimal stand-in at the documented BaseDPAdapter seam (no checkpoint)."""

    def __init__(self, act_horizon=8):
        self.metadata = dict(weights_subkey="stub", act_horizon=act_horizon)
        self.act_horizon = act_horizon

    def base_cond(self, stacked):
        return stacked

    def get_action(self, cond):
        return torch.zeros(1, self.act_horizon, 4)

    def map_action_ee_delta_pose(self, base_seq):
        return torch.zeros(1, self.act_horizon, 7)


class _FakeObstacleVectorEnv:
    """Fake single-env ``SyncVectorEnv`` reproducing the gymnasium 0.29
    auto-reset contract of the real evaluator env (built by ``build_env``):
    on the step where the episode ends, the returned info is the *reset* info
    of the fresh episode (sticky flags all False) plus ``final_info[0]`` /
    ``_final_info`` holding the real final step info, and the observation is
    the reset observation. ``step_flags[i]`` are the flags the real env would
    report per step; the last step additionally truncates."""

    def __init__(self, step_flags, max_steps):
        self.envs = [SimpleNamespace(unwrapped=self)]
        self._step_flags = step_flags
        self._max_steps = max_steps
        self._elapsed = 0
        self._in_reset = False
        self._applied = None

    def obstacle_config(self):
        return dict(self._applied)

    def reset(self, seed=None, options=None):
        self._elapsed = 0
        self._in_reset = False
        self._applied = dict(
            radius=float(options["obstacle_radius"]),
            center=np.asarray(options["obstacle_center"], dtype=float),
            cube_layout=dict(options["cube_layout"]),
        )
        return self._obs(), self._reset_info()

    def step(self, action):
        self._elapsed += 1
        real_flags = self._step_flags[self._elapsed - 1]
        real_info = {
            flag: np.asarray([value]) for flag, value in real_flags.items()
        }
        if self._elapsed < self._max_steps:
            return (
                self._obs(),
                np.array([0.0]),
                np.array([False]),
                np.array([False]),
                real_info,
            )
        # gymnasium 0.29 SyncVectorEnv: auto-reset and move the real step
        # info into final_info, guarded by the _final_info mask; the CPUGymWrapper
        # final step info holds raw unbatched (0-dimensional) values.
        self._in_reset = True
        info = self._reset_info()
        info["final_observation"] = np.array([self._obs()], dtype=object)
        info["_final_observation"] = np.array([True])
        info["final_info"] = np.array(
            [{flag: np.asarray(value) for flag, value in real_flags.items()}],
            dtype=object,
        )
        info["_final_info"] = np.array([True])
        return (
            self._obs(),
            np.array([0.0]),
            np.array([False]),
            np.array([True]),
            info,
        )

    def _obs(self):
        return np.zeros((1, 2, 51), dtype=np.float32)

    def _reset_info(self):
        return {flag: np.array([False]) for flag in EPISODE_FLAGS}

    def render(self):
        marker = 255.0 if self._in_reset else 0.0
        return np.full((1, 4, 4, 3), marker, dtype=np.float32)

    def close(self):
        pass


class _FakeWriter:
    def __init__(self):
        self.frames = []

    def append_data(self, frame):
        self.frames.append(frame)


def make_fake_env_evaluator(tmp_path, fake_env, **overrides):
    kwargs = dict(TINY_KWARGS)
    kwargs.update(overrides)
    evaluator = BaseDPEvaluator(
        output_dir=tmp_path / "eval_run", adapter=_StubAdapter(), **kwargs
    )
    evaluator._env = fake_env
    return evaluator


def _all_false_flags():
    return {flag: False for flag in EPISODE_FLAGS}


def test_run_episode_records_flags_occurring_only_on_final_step(tmp_path):
    """Contact, crossing and stack success landing exactly on the episode's
    final (auto-reset) step must still be recorded, with the success
    termination flag and the safe-success composition."""
    max_steps = 3
    step_flags = [
        _all_false_flags(),
        _all_false_flags(),
        dict(
            success=True,
            stack_success=True,
            obstacle_collision=True,
            obstacle_crossing=True,
        ),
    ]
    evaluator = make_fake_env_evaluator(
        tmp_path, _FakeObstacleVectorEnv(step_flags, max_steps=max_steps),
        max_steps=max_steps,
    )
    scene = sample_scenes(base_seed=0, num_episodes=1, cfg=EvalSceneConfig())[0]
    try:
        record = evaluator.run_episode(scene)
    finally:
        evaluator.close()
    result = record["result"]
    assert result["elapsed_steps"] == max_steps
    assert result["truncated"] is True
    assert result["termination_reason"] == "success"
    assert result["success"] is True
    assert result["stack_success"] is True
    assert result["obstacle_collision"] is True
    assert result["obstacle_crossing"] is True
    assert result["first_collision_step"] == max_steps
    assert result["first_crossing_step"] == max_steps
    assert result["safe_success"] is False


def test_run_episode_safe_success_when_final_step_is_clean(tmp_path):
    """A clean stack achieved exactly on the final step counts as safe
    success (stack success without contact or crossing)."""
    max_steps = 3
    step_flags = [
        _all_false_flags(),
        _all_false_flags(),
        dict(
            success=True,
            stack_success=True,
            obstacle_collision=False,
            obstacle_crossing=False,
        ),
    ]
    evaluator = make_fake_env_evaluator(
        tmp_path, _FakeObstacleVectorEnv(step_flags, max_steps=max_steps),
        max_steps=max_steps,
    )
    scene = sample_scenes(base_seed=0, num_episodes=1, cfg=EvalSceneConfig())[0]
    try:
        record = evaluator.run_episode(scene)
    finally:
        evaluator.close()
    result = record["result"]
    assert result["termination_reason"] == "success"
    assert result["truncated"] is True
    assert result["success"] is True
    assert result["safe_success"] is True


def test_run_episode_video_never_appends_the_auto_reset_frame(tmp_path):
    """The frame after the final step is the auto-reset scene of the fresh
    episode, not the episode's final observation (a state vector that cannot
    be rendered); the video must not end with that reset-frame lie."""
    max_steps = 3
    step_flags = [
        _all_false_flags(),
        _all_false_flags(),
        dict(
            success=True,
            stack_success=True,
            obstacle_collision=False,
            obstacle_crossing=False,
        ),
    ]
    evaluator = make_fake_env_evaluator(
        tmp_path, _FakeObstacleVectorEnv(step_flags, max_steps=max_steps),
        max_steps=max_steps, video=True,
    )
    scene = sample_scenes(base_seed=0, num_episodes=1, cfg=EvalSceneConfig())[0]
    writer = _FakeWriter()
    try:
        record = evaluator.run_episode(scene, writer=writer)
    finally:
        evaluator.close()
    assert record["result"]["success"] is True
    # reset frame + every real step except the auto-reset step
    assert len(writer.frames) == max_steps
    assert all(frame.max() == 0 for frame in writer.frames)
