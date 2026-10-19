"""Tests for generic adapter metadata and paired-scene evaluation (Ticket 04).

Seams under test:
- ``BaseDPEvaluator._settings`` — works for any adapter at the documented
  rollout seam: records the adapter's own checkpoint and weights key and
  captures the adapter metadata (checkpoint + normalization) instead of
  hardcoding base-DP fields,
- ``BaseDPEvaluator.__init__(video_scene_indexes=...)`` — representative
  video recording restricted to predesignated scene indexes,
- ``BaseDPEvaluator.evaluate_all(scenes=...)`` — evaluates an externally
  sampled, shared scene list (the paired-eval seam: both policies receive the
  exact same scene configs) and records that the scenes were provided rather
  than sampled internally.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.baselines.constrain.evaluate_base_dp import (
    BaseDPEvaluator,
    EvalSceneConfig,
    sample_scenes,
)
from examples.baselines.constrain.tests.test_evaluate_base_dp import (
    _FakeObstacleVectorEnv,
    _FakeWriter,
    _all_false_flags,
)


class _StubConstrainAdapter:
    """Stub at the adapter seam whose metadata mirrors ``ConstrainDPAdapter``:
    no ``weights_subkey`` key; checkpoint + normalization recorded instead."""

    def __init__(self):
        self.metadata = dict(
            checkpoint="/tmp/fake_constrain_final.pt",
            format="constrain_dp_v1",
            weights="ema",
            step=30000,
            control_enabled=True,
            normalization=dict(rel_xy_range=[-1.0, 1.0], radius_range=[0.0, 0.08]),
        )

    def base_cond(self, stacked):
        return stacked

    def get_action(self, cond):
        return torch.zeros(1, 8, 4)

    def map_action_ee_delta_pose(self, base_seq):
        return torch.zeros(1, 8, 7)


def make_stub_evaluator(tmp_path, adapter, **overrides):
    kwargs = dict(max_steps=2, video=False, device="cpu", adapter=adapter)
    kwargs.update(overrides)
    return BaseDPEvaluator(output_dir=tmp_path / "eval_run", **kwargs)


def test_settings_record_adapter_checkpoint_and_metadata(tmp_path):
    adapter = _StubConstrainAdapter()
    evaluator = make_stub_evaluator(tmp_path, adapter)
    settings = evaluator._settings()
    assert settings["checkpoint"] == "/tmp/fake_constrain_final.pt"
    assert settings["weights_subkey"] == "ema"
    assert settings["adapter_metadata"]["checkpoint"] == "/tmp/fake_constrain_final.pt"
    assert settings["adapter_metadata"]["step"] == 30000
    assert settings["adapter_metadata"]["normalization"]["radius_range"] == [0.0, 0.08]


def test_settings_still_record_base_dp_defaults(tmp_path):
    """The base DP path keeps its recorded checkpoint/weights_subkey behavior
    through the same generic settings code."""
    adapter = _BaseStubAdapter()
    evaluator = make_stub_evaluator(tmp_path, adapter)
    settings = evaluator._settings()
    assert settings["checkpoint"] == str(adapter.metadata["checkpoint"])
    assert settings["weights_subkey"] == "ema_agent"
    assert settings["adapter_metadata"]["weights_subkey"] == "ema_agent"


class _BaseStubAdapter(_StubConstrainAdapter):
    def __init__(self):
        super().__init__()
        self.metadata = dict(
            checkpoint="/tmp/fake_base.pt",
            weights_subkey="ema_agent",
        )


def test_run_episode_record_reports_adapter_checkpoint(tmp_path):
    adapter = _StubConstrainAdapter()
    fake_env = _FakeObstacleVectorEnv([_all_false_flags(), _all_false_flags()], max_steps=2)
    evaluator = make_stub_evaluator(tmp_path, adapter)
    evaluator._env = fake_env
    scene = sample_scenes(base_seed=0, num_episodes=1, cfg=EvalSceneConfig())[0]
    record = evaluator.run_episode(scene)
    assert record["run"]["checkpoint"] == "/tmp/fake_constrain_final.pt"


def _fake_env(max_steps=2):
    return _FakeObstacleVectorEnv(
        [_all_false_flags() for _ in range(max_steps)], max_steps=max_steps
    )


def test_evaluate_all_accepts_provided_shared_scenes(tmp_path):
    scenes = sample_scenes(base_seed=10000, num_episodes=2, cfg=EvalSceneConfig())
    evaluator = make_stub_evaluator(tmp_path, _StubConstrainAdapter(), num_episodes=2)
    evaluator._env = _fake_env()
    summary = evaluator.evaluate_all(scenes=scenes)
    manifest = json.loads((Path(summary["run_dir"]) / "manifest.json").read_text())
    assert manifest["settings"]["scenes_provided"] is True
    assert [s["seed"] for s in manifest["scenes"]] == [10000, 10001]
    # the exact same scene configs that were passed in (identical geometry)
    for recorded, provided in zip(manifest["scenes"], scenes):
        assert recorded["cube_layout"] == provided["cube_layout"]
        assert recorded["cylinder"] == provided["cylinder"]
    assert summary["metrics"]["total_episodes"] == 2


def test_evaluate_all_sampled_run_records_scenes_provided_false(tmp_path):
    evaluator = make_stub_evaluator(tmp_path, _StubConstrainAdapter(), num_episodes=1)
    evaluator._env = _fake_env()
    summary = evaluator.evaluate_all()
    manifest = json.loads((Path(summary["run_dir"]) / "manifest.json").read_text())
    assert manifest["settings"]["scenes_provided"] is False
    assert manifest["scenes"][0]["seed"] == 0


def test_video_scene_indexes_limits_recorded_frames(tmp_path):
    evaluator = make_stub_evaluator(
        tmp_path, _StubConstrainAdapter(), num_episodes=2, video=True,
        video_scene_indexes=[1],
    )
    evaluator._env = _fake_env()
    writer = _FakeWriter()
    evaluator._open_writer = lambda: (writer, None)
    evaluator._close_writer = lambda w, temporary: None
    evaluator.evaluate_all()
    # only the predesignated scene index 1 contributes frames
    # (reset frame + each non-terminal step: max_steps frames for the fake env)
    assert len(writer.frames) == 2


# --------------------------------------------------------------------------- #
# Paired evaluation driver: one sampled scene list fed to both policies.      #
# --------------------------------------------------------------------------- #

def test_run_paired_evaluation_feeds_same_scenes_and_merges(tmp_path, monkeypatch):
    """Both policies receive the exact same scene list object, the shared
    manifest is written once, per-policy summaries are merged with a scene
    equality check and actual completed counts."""
    from examples.baselines.constrain import evaluate_paired

    seen_scenes = []

    class _StubEvaluator:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            _StubEvaluator.instances.append(self)

        def evaluate_all(self, scenes=None):
            seen_scenes.append(scenes)
            name = Path(self.kwargs["output_dir"]).name
            run_dir = tmp_path / "paired" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = dict(settings=self.kwargs and {}, scenes=scenes)
            (run_dir / "manifest.json").write_text(json.dumps(manifest))
            return dict(
                run_dir=str(run_dir),
                settings=dict(adapter_metadata=self.kwargs["adapter"].metadata),
                metrics=dict(total_episodes=len(scenes), stack_success_rate=0.5),
                episodes=[
                    dict(scene_index=s["scene_index"], seed=s["seed"],
                         status="evaluated", path=str(run_dir / "ep.json"))
                    for s in scenes
                ],
            )

        def close(self):
            pass

    monkeypatch.setattr(evaluate_paired, "BaseDPEvaluator", _StubEvaluator)
    result = evaluate_paired.run_paired_evaluation(
        base_adapter=_BaseStubAdapter(),
        constrain_adapter=_StubConstrainAdapter(),
        num_episodes=3, max_steps=7, base_seed=10000,
        output_dir=tmp_path / "paired", device="cpu",
    )
    assert len(seen_scenes) == 2
    assert seen_scenes[0] is seen_scenes[1]  # the exact same scene objects
    assert [s["seed"] for s in seen_scenes[0]] == [10000, 10001, 10002]
    shared = json.loads((tmp_path / "paired" / "scenes_manifest.json").read_text())
    assert [s["seed"] for s in shared["scenes"]] == [10000, 10001, 10002]
    assert shared["base_seed"] == 10000 and shared["max_steps"] == 7
    assert result["scene_equality_verified"] is True
    for name in ("base_dp", "constrain_dp"):
        policy = result["policies"][name]
        assert policy["status"] == "completed"
        assert policy["metrics"]["total_episodes"] == 3
        assert policy["metrics"]["stack_success_rate"] == 0.5
        assert policy["adapter_metadata"]
    assert result["settings"]["num_episodes_requested"] == 3
    assert result["settings"]["base_seed"] == 10000
    assert "timings_sec" in result and set(result["timings_sec"]) == {
        "base_dp", "constrain_dp"
    }
    assert (tmp_path / "paired" / "paired_summary.json").exists()


def test_run_paired_evaluation_records_policy_failure_separately(tmp_path, monkeypatch):
    """One policy failing mid-run does not lose the other policy's results:
    the failure is recorded with its error and the other policy completes."""
    from examples.baselines.constrain import evaluate_paired

    class _FailingEvaluator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def evaluate_all(self, scenes=None):
            if "constrain" in str(self.kwargs["output_dir"]):
                raise RuntimeError("boom during episode 1")
            run_dir = Path(self.kwargs["output_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(json.dumps(dict(scenes=scenes)))
            return dict(
                run_dir=str(run_dir), settings=dict(), metrics=dict(total_episodes=2),
                episodes=[dict(scene_index=s["scene_index"], seed=s["seed"],
                               status="evaluated", path="x") for s in scenes],
            )

        def close(self):
            pass

    monkeypatch.setattr(evaluate_paired, "BaseDPEvaluator", _FailingEvaluator)
    result = evaluate_paired.run_paired_evaluation(
        base_adapter=_BaseStubAdapter(), constrain_adapter=_StubConstrainAdapter(),
        num_episodes=2, max_steps=5, base_seed=0, output_dir=tmp_path / "paired",
        device="cpu",
    )
    assert result["policies"]["constrain_dp"]["status"] == "failed"
    assert "boom during episode 1" in result["policies"]["constrain_dp"]["error"]
    assert result["policies"]["base_dp"]["status"] == "completed"
    assert result["policies"]["base_dp"]["metrics"]["total_episodes"] == 2
    # equality evidence is only produced when both policies completed
    assert result["scene_equality_verified"] is not True


def test_run_paired_evaluation_real_env_end_to_end(tmp_path):
    """Real single-env paired run with stub policies: both policies complete
    all episodes against the same manifest seeds; summary files exist."""
    from examples.baselines.constrain.evaluate_paired import run_paired_evaluation

    result = run_paired_evaluation(
        base_adapter=_BaseStubAdapter(), constrain_adapter=_StubConstrainAdapter(),
        num_episodes=2, max_steps=2, base_seed=10000,
        output_dir=tmp_path / "paired_real", device="cpu",
    )
    assert result["scene_equality_verified"] is True
    for name in ("base_dp", "constrain_dp"):
        policy = result["policies"][name]
        assert policy["status"] == "completed", policy.get("error")
        assert policy["metrics"]["total_episodes"] == 2
        run_dir = Path(policy["run_dir"])
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert [s["seed"] for s in manifest["scenes"]] == [10000, 10001]
        assert (run_dir / "summary.json").exists()
    base_manifest = json.loads(
        (Path(result["policies"]["base_dp"]["run_dir"]) / "manifest.json").read_text()
    )
    constrain_manifest = json.loads(
        (Path(result["policies"]["constrain_dp"]["run_dir"]) / "manifest.json").read_text()
    )
    assert base_manifest["scenes"] == constrain_manifest["scenes"]
