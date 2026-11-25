import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.baselines.constrain import peg_orientation_near_hole_eval as study


def test_failed_first_arm_is_journaled_with_numpy_values(tmp_path):
    row = {"catalog_index": 0, "replicate": 0, "seed": np.int64(1),
           "history": [np.asarray([1.0, 2.0])], "native_success": np.bool_(False)}
    path = tmp_path / "pair.jsonl"
    study.append_pair_journal(path, row)
    saved = json.loads(path.read_text())
    assert saved == {"catalog_index": 0, "replicate": 0, "seed": 1,
                     "history": [[1.0, 2.0]], "native_success": False}


def test_public_controller_mismatch_rejects_before_any_policy_call():
    class Controller:
        def __init__(self, value): self.value = value
        def get_state(self): return {"target": np.asarray([self.value])}
    class Agent:
        def __init__(self, value): self.controller = Controller(value)
    class Env:
        def __init__(self, value): self.unwrapped = self; self.agent = Agent(value); self.control_mode = "pd_ee_delta_pose"
    class Policy:
        calls = 0
        def get_action(self, _): self.calls += 1
    base, constrain = Policy(), Policy()
    with pytest.raises(ValueError, match="controller state mismatch"):
        study.assert_controller_match(Env(0), Env(1), 1e-5)
    assert base.calls == constrain.calls == 0


def test_paired_analysis_counts_outcomes_per_seed():
    rows = [
        {"group": "random", "catalog_index": 4, "replicate": 0, "seed": 5,
         "base": {"native_success": True}, "constrain": {"native_success": False}},
        {"group": "random", "catalog_index": 4, "replicate": 1, "seed": 1005,
         "base": {"native_success": True}, "constrain": {"native_success": True}},
    ]
    result = study.summarize_pairs(rows, "random", 4)
    assert result["per_seed"]["1"]["base_successes"] == 1
    assert result["per_seed"]["1001"]["paired"]["base_only"] == 0
    assert result["replicate_draws_per_scene"] == 3


def test_jsonable_handles_nested_zero_dimensional_numpy_and_torch_values():
    import torch

    result = study.jsonable({"scalar_array": np.asarray(3), "nested": [np.bool_(True), torch.tensor(4)]})
    assert result == {"scalar_array": 3, "nested": [True, 4]}
    assert type(result["scalar_array"]) is int
    assert type(result["nested"][0]) is bool


def test_runtime_source_closure_does_not_pin_prior_outer_wrapper():
    assert not any("130407_peg_near_hole_policy_comparison/attempt_001.sh" in path
                   for path in study.SOURCE_PATHS)


def test_prepare_arm_uses_real_task_predicate_after_replaying_native_prefix(monkeypatch):
    class Pose:
        def __init__(self, p): self.p = torch.as_tensor([p], dtype=torch.float32)
        def inv(self): return self
        def __mul__(self, other): return other

    class Task:
        control_mode = "pd_ee_delta_pose"
        peg = type("Peg", (), {"pose": Pose([0, 0, 0])})()
        box_hole_pose = Pose([0, 0, 0])
        peg_head_pose = Pose([-.03, 0, 0])
        def __init__(self):
            self.agent = type("Agent", (), {"controller": type("C", (), {"get_state": lambda s: {}})(),
                                              "is_grasping": lambda _self, _peg: torch.asarray([True])})()
        def has_peg_inserted(self): return (torch.asarray([False]), {})

    class Env:
        def __init__(self): self.unwrapped = Task(); self.steps = 0
        def step(self, action):
            self.steps += 1
            return torch.tensor([1., 2.]), 0., False, False, {"success": False}

    env = Env()
    monkeypatch.setattr(study, "restore_peg_scene", lambda _env, _scene: torch.tensor([0., 0.]))
    monkeypatch.setattr(study, "capture_physical_state", lambda _task: {"q": [0.]})
    monkeypatch.setattr(study, "_relative_yaw", lambda _task: 0.)
    action = np.asarray([.25], dtype=np.float32)
    digest = hashlib.sha256(action.tobytes()).hexdigest()
    bank = {"prefix_actions_sha256": digest, "controller_mode": "pd_ee_delta_pose",
            "physical_state": {"q": [0.]}, "observation_history": [[0., 0.], [1., 2.]],
            "head_in_hole_xyz_m": [-.03, 0., 0.]}
    arm = study._prepare_arm(env, object(), action[None], 1, bank, [0, time.monotonic()])
    assert env.steps == 1
    assert arm["grasped"] is True and arm["inserted"] is False


def _install_fake_comparison(tmp_path, monkeypatch, *, fail_constrain=False, fail_prepare=False):
    class Pose:
        def __init__(self, p): self.p = torch.as_tensor([p], dtype=torch.float32)
        def inv(self): return self
        def __mul__(self, other): return other

    class Task:
        control_mode = "pd_ee_delta_pose"
        def __init__(self):
            self.peg = type("Peg", (), {"pose": Pose([0., 0., 0.])})()
            self.box_hole_pose = Pose([0., 0., 0.])
            self.peg_head_pose = Pose([-.06 if fail_prepare else -.03, 0., 0.])
            self.agent = type("Agent", (), {
                "controller": type("Controller", (), {"get_state": lambda _self: {}})(),
                "is_grasping": lambda _self, _peg: torch.asarray([True]),
            })()
            self.controls = 0
        def has_peg_inserted(self): return (torch.asarray([False]), {})

    class Env:
        def __init__(self): self.unwrapped = Task(); self.closed = False
        def step(self, action):
            self.unwrapped.controls += 1
            return torch.ones(39), 0., False, False, {"success": self.unwrapped.controls > 1}
        def close(self): self.closed = True

    class Catalog:
        manifest = {"scenes": [{"scene_id": "scene-0"}]}
        def __init__(self, directory): self.directory = Path(directory)
        def load(self, index): return (None, None, None)

    class Adapter:
        metadata = {"identity": "fake"}
        hint_samples = []
        def __init__(self, *args, **kwargs): self.task = None
        def set_episode_context(self, *args): pass
        def get_action(self, history):
            history = np.asarray(history)
            assert history.shape == (1, 2, 39)
            np.testing.assert_array_equal(history[0, 0], np.zeros(39, dtype=np.float32))
            np.testing.assert_array_equal(history[0, 1], np.ones(39, dtype=np.float32))
            if fail_constrain and isinstance(self, ConstrainAdapter):
                raise RuntimeError("injected constrain inference failure")
            return np.zeros((1, 8, 7), dtype=np.float32)

    class ConstrainAdapter(Adapter): pass

    output = tmp_path / "result"
    output.mkdir()
    (output / "attempt_001.sh").write_text("current fake wrapper\n")
    catalogs_root, bank_root = tmp_path / "catalogs", tmp_path / "bank"
    for split in ("random", "fixed"):
        catalog_dir = catalogs_root / ("validation_random_scenes" if split == "random" else "validation_fixed_scenes")
        catalog_dir.mkdir(parents=True)
        (catalog_dir / "manifest.json").write_text("{}")
        bank_dir = bank_root / f"{split}_bank"
        bank_dir.mkdir(parents=True)
        (bank_dir / ("branches_reconciled.jsonl" if split == "random" else "branches.jsonl")).write_text("{}\n")
        data_dir = tmp_path / f"data-{split}"
        data_dir.mkdir()
        (data_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text('{"episodes": []}')
        (data_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5").write_bytes(b"fake")

    registered_sources = tuple(study.SOURCE_PATHS)
    for rel in registered_sources:
        source = tmp_path / rel
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"fixture source: {rel}\n")
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "BANK", bank_root)
    monkeypatch.setattr(study, "CATALOG_ROOT", catalogs_root)
    monkeypatch.setattr(study, "DATA_DIRS", {"random": tmp_path / "data-random", "fixed": tmp_path / "data-fixed"})
    monkeypatch.setattr(study, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(study, "_bank_rows", lambda split: [{"eligible": True, "catalog_index": 0,
        "prefix_controls": 1, "source_episode_id": 0, "prefix_actions_sha256": hashlib.sha256(np.asarray([.25], dtype=np.float32).tobytes()).hexdigest(),
        "controller_mode": "pd_ee_delta_pose", "physical_state": {"q": [0.]},
         "observation_history": [np.zeros(39).tolist(), np.ones(39).tolist()], "head_in_hole_xyz_m": [-.03, 0., 0.]}])
    monkeypatch.setattr(study, "_source_actions", lambda *_args: np.asarray([[.25]], dtype=np.float32))
    monkeypatch.setattr(study, "capture_physical_state", lambda _task: {"q": [0.]})
    monkeypatch.setattr(study, "restore_peg_scene", lambda _env, _scene: torch.zeros(39))
    monkeypatch.setattr(study, "_relative_yaw", lambda _task: 0.)
    monkeypatch.setattr(study, "_make_env", lambda _split: Env())
    monkeypatch.setattr(study, "sha256", lambda path: hashlib.sha256(str(path).encode()).hexdigest())
    monkeypatch.setattr(study, "BASE_SHA", hashlib.sha256(str(study.BASE_CKPT).encode()).hexdigest())
    monkeypatch.setattr(study, "CN_SHA", hashlib.sha256(str(study.CN_CKPT).encode()).hexdigest())
    monkeypatch.setattr(study, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(study, "PegConstrainAdapter", ConstrainAdapter)
    monkeypatch.setattr(study, "runtime_metadata", lambda: {"test": True})
    monkeypatch.setattr(study, "_seed", lambda _seed: None)
    monkeypatch.setattr(study, "_video_frame", lambda _env: np.zeros((1, 1, 3), dtype=np.uint8))
    class Writer:
        def append_data(self, _frame): pass
        def close(self): pass
    monkeypatch.setattr(study.imageio, "get_writer", lambda *_args, **_kwargs: Writer())
    monkeypatch.setattr(study, "VIDEO_CASES", {})
    return output


def test_full_comparison_keeps_prefix_metadata_finalizes_geometry_and_archives_current_wrapper(tmp_path, monkeypatch):
    output = _install_fake_comparison(tmp_path, monkeypatch)
    result = study.run_comparison(output)
    assert result["pairs_completed"] == 6
    assert result["native_controls_including_prefix_replay"] == 24
    pairs = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    complete = [pair for pair in pairs if pair["status"] == "complete"]
    assert len(complete) == 6
    assert sum(len(pair["arms"]) for pair in complete) == 12
    assert [pair["seed"] for pair in complete if pair["group"] == "random"] == [1, 1001, 2001]
    assert [pair["seed"] for pair in complete if pair["group"] == "fixed"] == [1, 1001, 2001]
    for pair in complete:
        assert pair["prefix"]["history_sha256"]
        assert pair["prefix"]["controller"]["public_state"] == {}
        assert pair["prefix"]["controller"]["base"]["state"] == pair["prefix"]["controller"]["constrain"]["state"]
    assert result["groups"]["random"]["pooled_denominator_per_policy"] == 3
    assert result["measured_initial_geometry"]["fixed"]["replicate_rows"] == 3
    assert (output / "results.json").is_file()
    registration = json.loads((output / "preregistration.json").read_text())
    wrapper = output / "attempt_001.sh"
    assert registration["outer_wrapper"]["sha256"] == hashlib.sha256(wrapper.read_bytes()).hexdigest()
    assert (output / "source_archive/outer_attempt_001.sh").read_bytes() == wrapper.read_bytes()
    assert not any("130407_peg_near_hole_policy_comparison/attempt_001.sh" in path
                   for path in study.SOURCE_PATHS)


def test_failure_journal_preserves_completed_base_arm_and_physical_budget(tmp_path, monkeypatch):
    output = _install_fake_comparison(tmp_path, monkeypatch, fail_constrain=True)
    with pytest.raises(RuntimeError, match="injected constrain inference failure"):
        study.run_comparison(output)
    rows = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["physical_controls_at_failure"] >= 2
    assert rows[-1]["arms"]["base"]["native_success"] is True
    assert json.loads((output / "failure.json").read_text())["physical_controls_including_prefix_replay"] >= 2


def test_failed_prefix_prepare_persists_loaded_physical_control_budget(tmp_path, monkeypatch):
    output = _install_fake_comparison(tmp_path, monkeypatch, fail_prepare=True)
    with pytest.raises(ValueError, match="near-hole eligibility"):
        study.run_comparison(output)
    pair = json.loads((output / "pairs.jsonl").read_text().splitlines()[-1])
    failure = json.loads((output / "failure.json").read_text())
    assert pair["physical_controls_at_failure"] == 1
    assert failure["physical_controls_including_prefix_replay"] == 1
