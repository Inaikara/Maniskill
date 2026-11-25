import hashlib
import json

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain import plug_orientation_base_dp as base
from examples.baselines.constrain import plug_orientation_scenes as scenes


def test_dataset_window_padding_and_episode_boundaries(tmp_path):
    h5path, jsonpath = tmp_path / "d.h5", tmp_path / "d.json"
    with h5py.File(h5path, "w") as h5:
        for i in range(1000):
            group = h5.create_group(f"traj_{i}")
            group.create_dataset("obs", data=np.full((17, 46), i, np.float32))
            group.create_dataset("actions", data=np.tile(np.array([[0.2] * 6 + [(-1) ** i]], np.float32), (16, 1)))
    jsonpath.write_text(json.dumps({"episodes": [{"episode_id": i} for i in range(1000)]}))
    ds = base.PlugBaseDataset(h5path, jsonpath, _test_mode=True)
    assert len(ds) == 15000
    first = ds[0]
    assert first["observations"].shape == (2, 46)
    assert first["actions"].shape == (16, 7)
    assert torch.equal(first["observations"][0], first["observations"][1])
    trailing = ds[14]["actions"]
    assert torch.all(trailing[-1, :6] == 0)
    assert trailing[-1, -1] == 1
    assert int(ds[-1]["observations"][0, 0]) == 999


class _TinyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
    def compute_loss(self, obs, actions): return (obs.mean() * self.weight - actions.mean()) ** 2
    def get_action(self, obs): return torch.zeros((1, 8, 7), device=obs.device) + self.weight


def test_tiny_training_seeds_initialization_and_writes_resolved_provenance(tmp_path):
    h5path, jsonpath = tmp_path / "d.h5", tmp_path / "trajectory.json"
    with h5py.File(h5path, "w") as h5:
        for i in range(1000):
            group = h5.create_group(f"traj_{i}")
            group.create_dataset("obs", data=np.ones((17, 46), np.float32))
            group.create_dataset("actions", data=np.full((16, 7), 0.5, np.float32))
    jsonpath.write_text(json.dumps({"episodes": [{"episode_id": i} for i in range(1000)]}))
    values = []
    class Factory(_TinyAgent):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.rand(()))
            values.append(float(self.weight.detach()))
    result = base.train(h5path, jsonpath, tmp_path / "exp", tmp_path / "ckpt",
                        _updates=2, _batch_size=2, device="cpu", agent_factory=lambda device: Factory(),
                        _test_mode=True)
    assert result["updates"] == 2
    torch.manual_seed(1)
    assert values == [float(torch.rand(()))]
    config = json.loads((tmp_path / "exp/resolved_config.json").read_text())
    assert config["format"] == base.FORMAT and config["observation_dim"] == 46


def test_public_training_repeats_nonempty_loader_across_epochs(tmp_path, monkeypatch):
    h5path, jsonpath = tmp_path / "d.h5", tmp_path / "trajectory.json"
    with h5py.File(h5path, "w") as h5:
        for i in range(1000):
            group = h5.create_group(f"traj_{i}")
            group.create_dataset("obs", data=np.ones((17, 46), np.float32))
            group.create_dataset("actions", data=np.full((16, 7), 0.5, np.float32))
    jsonpath.write_text(json.dumps({"episodes": [{"episode_id": i} for i in range(1000)]}))
    agent = _TinyAgent()
    before = agent.weight.detach().clone()
    optimizer_steps, scheduler_steps, ema_steps = [], [], []
    real_adamw = torch.optim.AdamW
    class CountingAdamW(real_adamw):
        def step(self, *args, **kwargs):
            optimizer_steps.append(1)
            return super().step(*args, **kwargs)
    monkeypatch.setattr(torch.optim, "AdamW", CountingAdamW)
    import diffusers.optimization as optimization
    import diffusers.training_utils as training_utils
    real_scheduler = optimization.get_scheduler
    def counting_scheduler(*args, **kwargs):
        scheduler = real_scheduler(*args, **kwargs)
        original = scheduler.step
        def step(*step_args, **step_kwargs):
            scheduler_steps.append(1)
            return original(*step_args, **step_kwargs)
        scheduler.step = step
        return scheduler
    monkeypatch.setattr(optimization, "get_scheduler", counting_scheduler)
    real_ema = training_utils.EMAModel
    class CountingEMA(real_ema):
        def step(self, *args, **kwargs):
            ema_steps.append(1)
            return super().step(*args, **kwargs)
    monkeypatch.setattr(training_utils, "EMAModel", CountingEMA)
    result = base.train(h5path, jsonpath, tmp_path / "exp", tmp_path / "ckpt",
        _updates=6, _batch_size=15000, device="cpu", agent_factory=lambda _: agent,
        _test_mode=True)
    assert result["updates"] == 6
    assert not torch.equal(agent.weight.detach(), before)
    assert len(optimizer_steps) == len(scheduler_steps) == len(ema_steps) == 6


def test_checkpoint_keeps_raw_and_ema_namespaces_distinct(tmp_path):
    from diffusers.training_utils import EMAModel
    agent = _TinyAgent(); optimizer = torch.optim.AdamW(agent.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ema = EMAModel(agent.parameters(), power=0.75)
    with torch.no_grad(): ema.shadow_params[0].fill_(0.75)
    raw = agent.weight.detach().clone(); path = tmp_path / "checkpoint.pt"
    base._save_checkpoint(path, agent, ema, optimizer, scheduler, 20000,
                          base.resolved_config(), {"fixture": True})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(payload["agent"]["weight"], raw)
    assert torch.equal(payload["ema_agent"]["weight"], torch.tensor(0.75))
    assert torch.equal(agent.weight.detach(), raw)


def test_validation_selection_is_frozen_and_gate_is_exact():
    contract = json.loads(base.Path(__file__).parents[4].joinpath(
        "examples/baselines/constrain/plug_orientation_contract.json").read_text())
    assert base._validate_contract_config(contract)["observation_dim"] == 46
    contract["base_policy"]["updates"] = 30000
    with pytest.raises(ValueError): base._validate_contract_config(contract)
    rows = [{"update": step, "successes": 20} for step in base.CHECKPOINT_STEPS]
    assert base.select_checkpoint(rows)["update"] == 100000
    with pytest.raises(ValueError): base.select_checkpoint(rows[:-1])
    assert base.validation_authorizes_test(40)
    assert not base.validation_authorizes_test(39)
    with pytest.raises(ValueError): base.validation_authorizes_test(40, total=100)


def test_selection_artifact_binds_latest_tie_checkpoint_sha(tmp_path):
    rows = []
    for step in base.CHECKPOINT_STEPS:
        path = tmp_path / f"checkpoint_{step:06d}.pt"; path.write_bytes(str(step).encode())
        summary_path = tmp_path / f"summary_{step}.json"
        summary_path.write_text(json.dumps({"episodes": 50, "successes": 42,
            "checkpoint_sha256": base.sha256_file(path),
            "catalog_manifest_sha256": "frozen-validation-fixture"}))
        rows.append({"update": step, "successes": 42, "episodes": 50, "checkpoint": str(path),
                     "catalog_manifest_sha256": "frozen-validation-fixture",
                     "catalog_split": "fixed_validation",
                     "evaluation_summary_path": str(summary_path),
                     "evaluation_summary_sha256": base.sha256_file(summary_path)})
    artifact = base.lock_validation_selection(rows, tmp_path / "selection.json")
    assert artifact["selected_update"] == 100000
    assert artifact["selected_checkpoint_sha256"] == base.sha256_file(rows[-1]["checkpoint"])


def test_selection_rejects_duplicate_checkpoint_content(tmp_path):
    rows = []
    for step in base.CHECKPOINT_STEPS:
        path = tmp_path / f"checkpoint_{step:06d}.pt"; path.write_bytes(b"same-payload")
        summary_path = tmp_path / f"summary_{step}.json"
        summary_path.write_text(json.dumps({"episodes": 50, "successes": 42,
            "checkpoint_sha256": base.sha256_file(path), "catalog_manifest_sha256": "validation"}))
        rows.append({"update": step, "successes": 42, "episodes": 50, "checkpoint": str(path),
                     "catalog_manifest_sha256": "validation", "catalog_split": "fixed_validation",
                     "evaluation_summary_path": str(summary_path),
                     "evaluation_summary_sha256": base.sha256_file(summary_path)})
    with pytest.raises(ValueError, match="distinct content"):
        base.lock_validation_selection(rows, tmp_path / "selection.json")


def test_selection_rejects_nondefault_stride_but_accepts_legacy_summary(tmp_path):
    rows = []
    for step in base.CHECKPOINT_STEPS:
        checkpoint = tmp_path / f"checkpoint_{step:06d}.pt"
        checkpoint.write_bytes(str(step).encode())
        summary = {"episodes": 50, "successes": 42,
                   "checkpoint_sha256": base.sha256_file(checkpoint),
                   "catalog_manifest_sha256": "validation"}
        summary_path = tmp_path / f"summary_{step}.json"
        summary_path.write_text(json.dumps(summary))
        rows.append({"update": step, "successes": 42, "episodes": 50,
                     "checkpoint": str(checkpoint), "catalog_manifest_sha256": "validation",
                     "catalog_split": "fixed_validation", "evaluation_summary_path": str(summary_path),
                     "evaluation_summary_sha256": base.sha256_file(summary_path)})
    # Historical complete summaries predate the stride field and remain readable.
    assert base.lock_validation_selection(rows, tmp_path / "legacy_selection.json")["selected_update"] == 100000
    summary_path = tmp_path / "summary_20000.json"
    summary = json.loads(summary_path.read_text()); summary["execution_stride"] = 1
    summary_path.write_text(json.dumps(summary))
    rows[0]["evaluation_summary_sha256"] = base.sha256_file(summary_path)
    with pytest.raises(ValueError, match="complete evaluation summary"):
        base.lock_validation_selection(rows, tmp_path / "rejected_selection.json")
    summary["execution_stride"] = 8; summary["episodes"] = 6
    summary_path.write_text(json.dumps(summary))
    rows[0]["episodes"] = 6
    rows[0]["evaluation_summary_sha256"] = base.sha256_file(summary_path)
    with pytest.raises(ValueError, match="50 scenes"):
        base.lock_validation_selection(rows, tmp_path / "subset_selection.json")


def test_authorized_test_rejects_nondefault_stride_before_policy_or_environment(tmp_path, monkeypatch):
    class Catalog:
        manifest = {"split": "fixed_base_test"}
        def __len__(self): return 100
    monkeypatch.setattr(base, "PlugSceneCatalog", lambda _: Catalog())
    called = []
    with pytest.raises(ValueError, match="original execution stride 8"):
        base.evaluate_checkpoint(tmp_path / "checkpoint.pt", tmp_path / "catalog", tmp_path / "out",
            _authorized_test=True, _selection_artifact=tmp_path / "selection.json", _test_mode=True,
            execution_stride=1, policy_factory=lambda _: called.append("policy"),
            env_factory=lambda: called.append("environment"))
    assert called == []


def test_authorized_test_rejects_partial_catalog_before_policy_or_environment(tmp_path, monkeypatch):
    class Catalog:
        manifest = {"split": "fixed_base_test"}
        def __len__(self): return 100
    monkeypatch.setattr(base, "PlugSceneCatalog", lambda _: Catalog())
    called = []
    with pytest.raises(ValueError, match="complete frozen 100-scene"):
        base.evaluate_checkpoint(tmp_path / "checkpoint.pt", tmp_path / "catalog", tmp_path / "out",
            _authorized_test=True, _selection_artifact=tmp_path / "selection.json", _test_mode=True,
            scene_indices=list(range(80)), policy_factory=lambda _: called.append("policy"),
            env_factory=lambda: called.append("environment"))
    assert called == []


def test_evaluator_rejects_bad_checkpoint_identity_before_environment(tmp_path, monkeypatch):
    class Catalog:
        manifest = {"split": "fixed_validation"}
        def __len__(self): return 50
    monkeypatch.setattr(base, "PlugSceneCatalog", lambda _: Catalog())
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {
        "format": base.FORMAT, "config": base.resolved_config(),
        "checkpoint_role": "validation_candidate", "update": 20000,
        "dataset_provenance": {"contract_sha256": "wrong-contract"}})
    called = []
    with pytest.raises(ValueError, match="contract identity mismatch"):
        base.evaluate_checkpoint(tmp_path / "checkpoint.pt", tmp_path / "catalog",
            tmp_path / "eval", env_factory=lambda: called.append(True))
    assert called == []


def test_adapter_rejects_changed_training_data_before_agent_construction(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    data_path = tmp_path / "data.h5"; data_path.write_bytes(b"changed")
    provenance = {"contract_sha256": base.sha256_file(base.Path(base.__file__).with_name(
        "plug_orientation_contract.json")), "dataset_h5_path": str(data_path),
        "dataset_h5_sha256": "not-the-current-hash"}
    payload = {"format": base.FORMAT, "observation_dim": 46, "action_dim": 7,
        "config": base.resolved_config(), "checkpoint_role": "validation_candidate",
        "update": 20000, "dataset_provenance": provenance}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    monkeypatch.setattr(base, "build_agent", lambda *_: pytest.fail("model must not be constructed"))
    with pytest.raises(ValueError, match="dataset_h5_path provenance mismatch"):
        base.PlugBaseAdapter(checkpoint)


def test_validation_evaluator_uses_two_real_history_frames_and_incremental_records(tmp_path):
    catalog_dir = tmp_path / "fixed_validation"; catalog_dir.mkdir()
    manifest = {"format": "plug-orientation-scenes-v1", "split": "fixed_validation",
                "scene_count": 50, "scenes_file": "scenes.h5", "scenes": []}
    state = {"fixture": np.zeros(1, np.float32)}
    with h5py.File(catalog_dir / "scenes.h5", "w") as h5:
        for index in range(50):
            sid = f"scene-{index}"; group = h5.create_group(sid)
            scenes._write_state(group.create_group("initial_state"), state)
            group.create_dataset("initial_observation", data=np.zeros(46, np.float32))
            manifest["scenes"].append({"scene_id": sid, "seed": 2026124100 + index,
                "target_yaw_rad": 0.0, "initial_state_sha256": scenes._state_hash(state),
                "initial_observation_sha256": hashlib.sha256(np.zeros(46, np.float32).tobytes()).hexdigest()})
    manifest["scenes_file_sha256"] = scenes._sha(catalog_dir / "scenes.h5")
    manifest["manifest_sha256"] = scenes._canonical(manifest)
    (catalog_dir / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "ckpt.pt"; checkpoint.write_bytes(b"offline-fixture-checkpoint")
    class Env:
        device = "cpu"
        def __init__(self): self.unwrapped = self
        def reset(self, seed, options): return torch.zeros((1, 46)), {}
        def set_state_dict(self, state): self.state = state
        def get_obs(self): return torch.zeros((1, 46))
        def get_state_dict(self): return {"fixture": torch.zeros((1, 1))}
        def step(self, action):
            assert action.shape == (7,)
            return torch.zeros((1, 46)), 0.0, True, False, {"success": True}
        def close(self): pass
    class Policy:
        device = torch.device("cpu")
        def get_action(self, observations):
            assert observations.shape == (1, 2, 46)
            return torch.zeros((1, 8, 7))
    result = base.evaluate_checkpoint(checkpoint, catalog_dir, tmp_path / "eval",
        policy_factory=lambda _: Policy(), env_factory=Env, _test_mode=True)
    assert result["episodes"] == 50 and result["successes"] == 50
    assert len((tmp_path / "eval/episodes.jsonl").read_text().splitlines()) == 50
    assert len(list((tmp_path / "eval").glob("episode_*.events.jsonl"))) == 50


def test_validation_evaluator_supports_selected_scenes_and_execution_stride(tmp_path):
    catalog_dir = tmp_path / "fixed_validation"; catalog_dir.mkdir()
    manifest = {"format": "plug-orientation-scenes-v1", "split": "fixed_validation",
                "scene_count": 50, "scenes_file": "scenes.h5", "scenes": []}
    state = {"fixture": np.zeros(1, np.float32)}
    with h5py.File(catalog_dir / "scenes.h5", "w") as h5:
        for index in range(50):
            group = h5.create_group(f"scene-{index}")
            scenes._write_state(group.create_group("initial_state"), state)
            group.create_dataset("initial_observation", data=np.zeros(46, np.float32))
            manifest["scenes"].append({"scene_id": f"scene-{index}", "seed": 2026124100 + index,
                "target_yaw_rad": 0.0, "initial_state_sha256": scenes._state_hash(state),
                "initial_observation_sha256": hashlib.sha256(np.zeros(46, np.float32).tobytes()).hexdigest()})
    manifest["scenes_file_sha256"] = scenes._sha(catalog_dir / "scenes.h5")
    manifest["manifest_sha256"] = scenes._canonical(manifest)
    (catalog_dir / "manifest.json").write_text(json.dumps(manifest))
    class Env:
        device = "cpu"
        def __init__(self): self.unwrapped = self; self.actions = 0
        def reset(self, seed, options): self.actions = 0; return torch.zeros((1, 46)), {}
        def set_state_dict(self, state): pass
        def get_obs(self): return torch.zeros((1, 46))
        def get_state_dict(self): return {"fixture": torch.zeros((1, 1))}
        def step(self, action):
            assert action.shape == (7,); self.actions += 1
            return torch.zeros((1, 46)), 0.0, self.actions == 3, False, {"success": False}
        def close(self): pass
    class Policy:
        device = torch.device("cpu")
        def __init__(self): self.calls = []
        def get_action(self, observations):
            assert observations.shape == (1, 2, 46); self.calls.append(observations.clone())
            return torch.zeros((1, 8, 7))
    policy = Policy()
    checkpoint = tmp_path / "ckpt.pt"; checkpoint.write_bytes(b"fixture")
    result = base.evaluate_checkpoint(checkpoint, catalog_dir, tmp_path / "eval",
        policy_factory=lambda _: policy, env_factory=Env, _test_mode=True,
        scene_indices=[2, 5], execution_stride=1)
    assert [r["index"] for r in result["records"]] == [2, 5]
    assert [r["executed_controls"] for r in result["records"]] == [3, 3]
    assert len(policy.calls) == 6
    assert result["execution_stride"] == 1
    with pytest.raises(ValueError, match="execution stride"):
        base.evaluate_checkpoint(checkpoint, catalog_dir, tmp_path / "eval2",
            policy_factory=lambda _: Policy(), env_factory=Env, _test_mode=True,
            scene_indices=[0], execution_stride=0)


def test_public_evaluator_default_env_factory_runs_actual_evaluation(tmp_path, monkeypatch):
    catalog_dir = tmp_path / "fixed_validation"
    catalog_dir.mkdir()
    manifest = {"format": "plug-orientation-scenes-v1", "split": "fixed_validation",
                "scene_count": 50, "scenes_file": "scenes.h5", "scenes": []}
    state = {"fixture": np.zeros(1, np.float32)}
    with h5py.File(catalog_dir / "scenes.h5", "w") as h5:
        for index in range(50):
            sid = f"scene-{index}"
            group = h5.create_group(sid)
            scenes._write_state(group.create_group("initial_state"), state)
            group.create_dataset("initial_observation", data=np.zeros(46, np.float32))
            manifest["scenes"].append({"scene_id": sid, "seed": 2026124100 + index,
                "target_yaw_rad": 0.0, "initial_state_sha256": scenes._state_hash(state),
                "initial_observation_sha256": hashlib.sha256(np.zeros(46, np.float32).tobytes()).hexdigest()})
    manifest["scenes_file_sha256"] = scenes._sha(catalog_dir / "scenes.h5")
    manifest["manifest_sha256"] = scenes._canonical(manifest)
    (catalog_dir / "manifest.json").write_text(json.dumps(manifest))

    class Env:
        device = "cpu"
        def __init__(self): self.unwrapped = self
        def reset(self, seed, options): return torch.zeros((1, 46)), {}
        def set_state_dict(self, state): self.state = state
        def get_obs(self): return torch.zeros((1, 46))
        def get_state_dict(self): return {"fixture": torch.zeros((1, 1))}
        def step(self, action):
            assert action.shape == (7,)
            return torch.zeros((1, 46)), 0.0, True, False, {"success": True}
        def close(self): pass

    class Policy:
        device = torch.device("cpu")
        def get_action(self, observations):
            assert observations.shape == (1, 2, 46)
            return torch.zeros((1, 8, 7))

    import gymnasium as gym
    made = []
    def make(env_id, **kwargs):
        made.append((env_id, kwargs))
        return Env()
    monkeypatch.setattr(gym, "make", make)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"offline-fixture-checkpoint")
    result = base.evaluate_checkpoint(checkpoint, catalog_dir, tmp_path / "eval",
        policy_factory=lambda _: Policy(), _test_mode=True)
    assert made == [("PlugChargerOrientation-v1", {"obs_mode": "state",
        "control_mode": "pd_ee_delta_pose", "sim_backend": "physx_cpu", "max_episode_steps": 1000})]
    assert result["episodes"] == 50 and result["successes"] == 50


def test_official_agent_sampler_contract_smoke():
    agent = base.build_agent("cpu"); agent.eval()
    with torch.no_grad(): actions = agent.get_action(torch.zeros((1, 2, 46)))
    assert actions.shape == (1, 8, 7) and torch.isfinite(actions).all()


def test_adapter_enforces_exact_single_scene_input_shape():
    adapter = object.__new__(base.PlugBaseAdapter)
    adapter.device, adapter.agent = torch.device("cpu"), _TinyAgent()
    assert adapter.get_action(torch.zeros((1, 2, 46))).shape == (1, 8, 7)
    with pytest.raises(ValueError): adapter.get_action(torch.zeros((2, 2, 46)))


def test_production_training_budget_cannot_be_shortened(tmp_path):
    with pytest.raises(ValueError, match="frozen at 100000"):
        base.train(tmp_path / "data.h5", tmp_path / "data.json", tmp_path / "exp",
                   tmp_path / "ckpt", _updates=2, _batch_size=2)
