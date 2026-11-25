import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from examples.baselines.constrain import plug_orientation_base_scaled as scaled
from examples.baselines.constrain import plug_orientation_base_dp as base


class ToyAgent(torch.nn.Module):
    obs_horizon, act_horizon, pred_horizon, act_dim = 2, 8, 16, 7
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.seen = None
    def compute_loss(self, obs, actions):
        self.seen = actions.detach().clone()
        return actions.mean() * self.weight
    def get_action(self, obs):
        return torch.ones((1, 8, 7), device=obs.device) * self.weight


def test_scaled_agent_normalizes_targets_and_decodes_sampler_in_native_units():
    toy = ToyAgent()
    wrapped = scaled.ScaledAgent(toy, [0.05, 0.05, 0.05, 0.2835, 0.2955, 0.3, 1.0])
    actions = torch.tensor([[[0.025, -0.05, 0.0, 0.0, 0.0, 0.15, 0.5]]])
    wrapped.compute_loss(torch.zeros((1, 2, 46)), actions)
    assert torch.allclose(toy.seen, torch.tensor([[[0.5, -1., 0., 0., 0., .5, .5]]]))
    native = wrapped.get_action(torch.zeros((1, 2, 46)))
    assert torch.allclose(native[0, 0], torch.tensor([.025, .025, .025, .14175, .14775, .15, .5]))
    assert "action_scale" in wrapped.state_dict()
    assert all(parameter is not wrapped.action_scale for parameter in wrapped.parameters())
    original = actions.clone()
    assert torch.equal(actions, original)


def test_fit_scale_uses_every_admitted_train_row_and_binds_all_input_hashes(tmp_path, monkeypatch):
    h5path, manifest_path = tmp_path / "train.h5", tmp_path / "train.json"
    dataset_manifest = tmp_path / "dataset_manifest.json"
    row = np.asarray([.05, -.04, .03, .2, -.25, .3, -1.], np.float32)
    original = np.tile(row, (1000, 16, 1))
    with h5py.File(h5path, "w") as h5:
        for i in range(1000):
            h5.create_group(f"traj_{i}").create_dataset("actions", data=original[i])
    manifest_path.write_text(json.dumps({"episodes": [{"episode_id": i} for i in range(1000)]}))
    dataset_manifest.write_text(json.dumps({"accepted_count": 1000}))
    expected = {"hdf5_sha256": scaled.sha256_file(h5path), "json_sha256": scaled.sha256_file(manifest_path),
        "dataset_manifest_sha256": scaled.sha256_file(dataset_manifest), "action_rows": 16000}
    monkeypatch.setattr(scaled, "_contract", lambda: {"training_data": expected})
    stats_path = tmp_path / "scale.json"
    stats = scaled.fit_action_scale(h5path, manifest_path, stats_path)
    assert np.array_equal(np.asarray(stats["scale"], np.float32), np.asarray([.05,.04,.03,.2,.25,.3,1.],np.float32))
    assert stats["action_row_count"] == 16000
    assert stats["full_action_rows_sha256"] == hashlib.sha256(original.reshape(-1).tobytes()).hexdigest()
    with h5py.File(h5path) as h5:
        assert np.array_equal(h5["traj_0/actions"][:], original[0])
    _, loaded = scaled._read_scale(stats_path, h5path, manifest_path)
    assert np.array_equal(loaded, np.asarray(stats["scale"], np.float32))
    with pytest.raises(FileExistsError): scaled.fit_action_scale(h5path, manifest_path, stats_path)


def test_scale_fit_rejects_data_changed_after_admission(tmp_path, monkeypatch):
    h5path, manifest_path = tmp_path / "train.h5", tmp_path / "train.json"
    (tmp_path / "dataset_manifest.json").write_text("{}")
    with h5py.File(h5path, "w") as h5:
        for i in range(1000): h5.create_group(f"traj_{i}").create_dataset("actions", data=np.ones((16, 7), np.float32))
    manifest_path.write_text(json.dumps({"episodes": [{"episode_id": i} for i in range(1000)]}))
    monkeypatch.setattr(scaled, "_contract", lambda: {"training_data": {
        "hdf5_sha256": "wrong", "json_sha256": scaled.sha256_file(manifest_path),
        "dataset_manifest_sha256": scaled.sha256_file(tmp_path / "dataset_manifest.json"), "action_rows": 16000}})
    with pytest.raises(ValueError, match="admitted TRAIN hdf5 identity"):
        scaled.fit_action_scale(h5path, manifest_path, tmp_path / "scale.json")


def test_checkpoint_format_is_explicit_and_distinct(tmp_path):
    from diffusers.training_utils import EMAModel
    agent = scaled.ScaledAgent(ToyAgent(), torch.ones(7))
    optimizer = torch.optim.AdamW(agent.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    ema = EMAModel(agent.parameters(), power=.75)
    path = tmp_path / "v2.pt"
    base._save_checkpoint(path, agent, ema, optimizer, scheduler, 20000,
                          scaled.resolved_config(), {"model_family": scaled.FORMAT}, format_id=scaled.FORMAT)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["format"] == scaled.FORMAT
    assert payload["config"]["normalization"] == "symmetric_train_maxabs"
    assert payload["format"] != base.FORMAT


def test_v2_adapter_rejects_v1_checkpoint_before_agent_build(tmp_path, monkeypatch):
    path = tmp_path / "v1.pt"; path.write_bytes(b"old model")
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {
        "format": base.FORMAT, "config": base.resolved_config(), "checkpoint_role": "validation_candidate",
        "update": 80000})
    monkeypatch.setattr(scaled, "_build_scaled_agent", lambda *a: pytest.fail("must reject before construction"))
    with pytest.raises(ValueError, match="v2 checkpoint family"):
        scaled.ScaledAdapter(path)


def test_v2_fixed_test_cannot_open_with_v1_selection_or_partial_scenes(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"format": "plug-orientation-validation-selection-v1",
        "validation_successes": 50, "validation_episodes": 50}))
    with pytest.raises(ValueError):
        scaled.evaluate_fixed_test(tmp_path / "ckpt.pt", tmp_path / "catalog", selection, tmp_path / "out")


def test_v2_fixed_test_checks_validation_gate_before_catalog_access(tmp_path, monkeypatch):
    candidates = [{"update": step, "checkpoint_sha256": str(step)} for step in base.CHECKPOINT_STEPS]
    selection = {"format": "plug-orientation-scaled-validation-selection-v2", "model_family": scaled.FORMAT,
        "normalization_method": "per_channel_full_train_maxabs", "validation_successes": 39,
        "validation_episodes": 50, "selected_checkpoint_sha256": "not-used", "candidate_results": candidates}
    selection["selection_sha256"] = base._canonical_sha256(selection)
    selection_path = tmp_path / "v2_selection.json"
    selection_path.write_text(json.dumps(selection))
    monkeypatch.setattr(scaled.base, "PlugSceneCatalog", lambda *_: pytest.fail("test catalog opened before gate"))
    with pytest.raises(ValueError, match="at least 40/50"):
        scaled.evaluate_fixed_test(tmp_path / "ckpt.pt", tmp_path / "test_catalog", selection_path, tmp_path / "out")


@pytest.mark.parametrize("state_key", ["agent", "ema_agent"])
def test_checkpoint_rejects_wrong_persistent_scale_before_agent_or_environment(tmp_path, monkeypatch, state_key):
    """Provenance cannot mask a raw/EMA persistent-buffer mismatch."""
    # This regression targets the new v2 validator's payload boundary.
    wrong = torch.tensor([2., 2., 2., 2., 2., 2., 2.])
    source = tmp_path / "source.json"
    source.write_text("fixture")
    provenance = {"action_scale": [1.] * 7, "contract_sha256": "x", "addendum_sha256": "x",
        "base_contract_sha256": "x", "dataset_h5_sha256": "x", "dataset_json_sha256": "x",
        "dataset_manifest_sha256": "x", "catalog_set_sha256": "x",
        "dataset_h5_path": str(source), "dataset_json_path": str(source),
        "dataset_manifest_path": str(source), "catalog_set_path": str(source),
        "scale_statistics_path": str(source), "scale_statistics_sha256": "x",
        "full_action_rows_sha256": "x"}
    payload = {
        "format": scaled.FORMAT, "model_family": scaled.FORMAT,
        "config": scaled.resolved_config(), "checkpoint_role": "validation_candidate",
        "update": 20000, "dataset_provenance": provenance,
        "agent": {"action_scale": torch.ones(7)}, "ema_agent": {"action_scale": torch.ones(7)},
    }
    payload[state_key]["action_scale"] = wrong
    monkeypatch.setattr(torch, "load", lambda *a, **kw: payload)
    monkeypatch.setattr(scaled, "_contract", lambda: {"training_data": {
        "hdf5_sha256": "x", "json_sha256": "x", "dataset_manifest_sha256": "x",
        "catalog_set_sha256": "x"}})
    monkeypatch.setattr(scaled, "sha256_file", lambda *_: "x")
    monkeypatch.setattr(scaled, "_read_scale", lambda *_: ({"full_action_rows_sha256": "x"}, np.ones(7, np.float32)))
    monkeypatch.setattr(scaled, "_build_scaled_agent", lambda *a: pytest.fail("built agent before rejecting corrupt buffer"))
    with pytest.raises(ValueError, match="action_scale buffer"):
        scaled._validate_checkpoint(tmp_path / "checkpoint.pt")


def test_strict_ema_restore_keeps_raw_checkpoint_state_and_registered_scale(tmp_path, monkeypatch):
    scale = np.asarray([.1, .2, .3, .4, .5, .6, .7], dtype=np.float32)
    source = tmp_path / "source.json"; source.write_text("fixture")
    stats = tmp_path / "scale.json"; stats.write_text("fixture")
    provenance = {"contract_sha256": "id", "addendum_sha256": "id", "base_contract_sha256": "id",
        "dataset_h5_sha256": "id", "dataset_json_sha256": "id", "dataset_manifest_sha256": "id",
        "catalog_set_sha256": "id", "dataset_h5_path": str(source), "dataset_json_path": str(source),
        "dataset_manifest_path": str(source), "catalog_set_path": str(source),
        "scale_statistics_path": str(stats), "scale_statistics_sha256": "id",
        "action_scale": scale.tolist(), "full_action_rows_sha256": "rows",
        "frozen_scene_catalog_sha256": {}}
    agent = {"agent.weight": torch.tensor(.2), "action_scale": torch.from_numpy(scale.copy())}
    ema = {"agent.weight": torch.tensor(.9), "action_scale": torch.from_numpy(scale.copy())}
    payload = {"format": scaled.FORMAT, "model_family": scaled.FORMAT, "config": scaled.resolved_config(),
        "checkpoint_role": "validation_candidate", "update": 20000, "dataset_provenance": provenance,
        "agent": agent, "ema_agent": ema}
    path = tmp_path / "checkpoint_020000.pt"; torch.save(payload, path)
    monkeypatch.setattr(scaled, "_contract", lambda: {"training_data": {"hdf5_sha256": "id",
        "json_sha256": "id", "dataset_manifest_sha256": "id", "catalog_set_sha256": "id"}})
    monkeypatch.setattr(scaled, "sha256_file", lambda *_: "id")
    monkeypatch.setattr(scaled, "_read_scale", lambda *_: ({"full_action_rows_sha256": "rows"}, scale))
    monkeypatch.setattr(scaled, "_build_scaled_agent", lambda s, d: scaled.ScaledAgent(ToyAgent(), s).to(d))
    raw_before = torch.load(path, map_location="cpu", weights_only=False)["agent"]
    adapter = scaled.ScaledAdapter(path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    assert adapter.agent.agent.weight.item() == pytest.approx(.9)
    assert torch.equal(adapter.agent.action_scale, torch.from_numpy(scale))
    assert torch.equal(restored["agent"]["agent.weight"], raw_before["agent.weight"])
    assert torch.equal(restored["agent"]["action_scale"], torch.from_numpy(scale))


@pytest.mark.parametrize("tamper", [None, "wrong_family", "partial", "checkpoint", "config", "summary", "test_catalog"])
def test_v2_test_selection_admits_or_rejects_before_rollout(tmp_path, monkeypatch, tamper):
    scale = np.ones(7, dtype=np.float32)
    source = tmp_path / "source.json"; source.write_text("fixture")
    scale_path = tmp_path / "scale.json"; scale_path.write_text("fixture")
    catalog_root = tmp_path / "catalogs"
    validation_dir = catalog_root / "fixed_validation"; test_dir = catalog_root / "fixed_base_test"
    validation_dir.mkdir(parents=True); test_dir.mkdir(parents=True)
    (validation_dir / "manifest.json").write_text('{"split":"fixed_validation"}')
    (test_dir / "manifest.json").write_text('{"split":"fixed_base_test"}')
    catalog_set = catalog_root / "catalog_set.json"
    catalog_set.write_text(json.dumps({"catalogs": {"fixed_validation": {"manifest_sha256": scaled.sha256_file(validation_dir / "manifest.json")},
        "fixed_base_test": {"manifest_sha256": scaled.sha256_file(test_dir / "manifest.json")}}}))
    val_hash = scaled.sha256_file(validation_dir / "manifest.json")
    test_hash = scaled.sha256_file(test_dir / "manifest.json")
    data_hash = scaled.sha256_file(source)
    monkeypatch.setattr(scaled, "_contract", lambda: {"training_data": {"hdf5_sha256": data_hash,
        "json_sha256": data_hash, "dataset_manifest_sha256": data_hash, "catalog_set_sha256": scaled.sha256_file(catalog_set)}})
    monkeypatch.setattr(scaled, "_read_scale", lambda *_: ({"full_action_rows_sha256": "rows"}, scale))
    def provenance():
        return {"contract_sha256": scaled.sha256_file(scaled.ADDENDUM), "addendum_sha256": scaled.sha256_file(scaled.ADDENDUM),
            "base_contract_sha256": scaled.sha256_file(scaled.BASE_CONTRACT), "dataset_h5_sha256": data_hash,
            "dataset_json_sha256": data_hash, "dataset_manifest_sha256": data_hash, "catalog_set_sha256": scaled.sha256_file(catalog_set),
            "dataset_h5_path": str(source), "dataset_json_path": str(source), "dataset_manifest_path": str(source),
            "catalog_set_path": str(catalog_set), "scale_statistics_path": str(scale_path),
            "scale_statistics_sha256": scaled.sha256_file(scale_path), "action_scale": scale.tolist(),
            "full_action_rows_sha256": "rows", "frozen_scene_catalog_sha256": {
                "fixed_validation": val_hash, "fixed_base_test": test_hash}}
    ckpt_dir = tmp_path / "checkpoints"; ckpt_dir.mkdir()
    rows = []
    for step, successes in zip(base.CHECKPOINT_STEPS, [20, 30, 40, 49, 49]):
        path = ckpt_dir / f"checkpoint_{step:06d}.pt"
        torch.save({"format": scaled.FORMAT, "model_family": scaled.FORMAT,
            "config": scaled.resolved_config(), "checkpoint_role": "validation_candidate", "update": step,
            "dataset_provenance": provenance(),
            "agent": {"agent.weight": torch.tensor(.1), "action_scale": torch.from_numpy(scale.copy())},
            "ema_agent": {"agent.weight": torch.tensor(.2), "action_scale": torch.from_numpy(scale.copy())}}, path)
        digest = scaled.sha256_file(path)
        summary_path = tmp_path / f"summary_{step}.json"
        summary_path.write_text(json.dumps({"model_family": scaled.FORMAT,
            "normalization_method": "per_channel_full_train_maxabs", "episodes": 50,
            "successes": successes, "execution_stride": 8, "checkpoint_sha256": digest,
            "catalog_manifest_sha256": val_hash}))
        rows.append({"update": step, "checkpoint": str(path), "checkpoint_sha256": digest,
            "successes": successes, "episodes": 50, "catalog_split": "fixed_validation",
            "catalog_manifest_sha256": val_hash, "evaluation_summary_path": str(summary_path),
            "evaluation_summary_sha256": scaled.sha256_file(summary_path), "model_family": scaled.FORMAT})
    chosen = max(rows, key=lambda row: (row["successes"], row["update"]))
    selection = {"format": "plug-orientation-scaled-validation-selection-v2", "model_family": scaled.FORMAT,
        "normalization_method": "per_channel_full_train_maxabs", "validation_successes": 49,
        "validation_episodes": 50, "selected_update": chosen["update"], "selected_checkpoint": chosen["checkpoint"],
        "selected_checkpoint_sha256": chosen["checkpoint_sha256"], "validation_catalog_manifest_sha256": val_hash,
        "candidate_results": rows}
    selection["selection_sha256"] = base._canonical_sha256(selection)
    selection_path = tmp_path / "selection.json"; selection_path.write_text(json.dumps(selection))
    if tamper == "wrong_family":
        selection["model_family"] = base.FORMAT
        selection["selection_sha256"] = base._canonical_sha256({k: v for k, v in selection.items() if k != "selection_sha256"})
        selection_path.write_text(json.dumps(selection))
    elif tamper == "partial":
        selection["candidate_results"] = selection["candidate_results"][:-1]
        selection["selection_sha256"] = base._canonical_sha256({k: v for k, v in selection.items() if k != "selection_sha256"})
        selection_path.write_text(json.dumps(selection))
    elif tamper == "checkpoint":
        with Path(chosen["checkpoint"]).open("ab") as checkpoint_file: checkpoint_file.write(b"changed")
    elif tamper == "config":
        checkpoint_payload = torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)
        checkpoint_payload["config"]["normalization"] = "tampered"
        torch.save(checkpoint_payload, chosen["checkpoint"])
        digest = scaled.sha256_file(chosen["checkpoint"])
        chosen["checkpoint_sha256"] = digest
        selection["selected_checkpoint_sha256"] = digest
        summary_path = Path(chosen["evaluation_summary_path"])
        summary = json.loads(summary_path.read_text()); summary["checkpoint_sha256"] = digest
        summary_path.write_text(json.dumps(summary))
        chosen["evaluation_summary_sha256"] = scaled.sha256_file(summary_path)
        selection["selection_sha256"] = base._canonical_sha256({k: v for k, v in selection.items() if k != "selection_sha256"})
        selection_path.write_text(json.dumps(selection))
    elif tamper == "summary":
        with Path(chosen["evaluation_summary_path"]).open("a") as summary_file: summary_file.write(" ")
    elif tamper == "test_catalog":
        (test_dir / "manifest.json").write_text('{"split":"fixed_base_test","changed":true}')
    class Catalog:
        def __init__(self, directory):
            self.manifest = json.loads((Path(directory) / "manifest.json").read_text())
            self.count = 50 if self.manifest["split"] == "fixed_validation" else 100
        def __len__(self): return self.count
    monkeypatch.setattr(base, "PlugSceneCatalog", Catalog)
    calls = []
    def backend(checkpoint, catalog_dir, output_dir, **kwargs):
        calls.append((checkpoint, catalog_dir, kwargs))
        Path(output_dir).mkdir(parents=True)
        return {"successes": 80, "episodes": 100, "checkpoint_sha256": scaled.sha256_file(checkpoint),
            "catalog_manifest_sha256": test_hash, "execution_stride": 8}
    monkeypatch.setattr(base, "evaluate_checkpoint", backend)
    if tamper is not None:
        with pytest.raises(ValueError):
            scaled.evaluate_fixed_test(chosen["checkpoint"], test_dir, selection_path, tmp_path / "out")
        assert calls == []
    else:
        result = scaled.evaluate_fixed_test(chosen["checkpoint"], test_dir, selection_path, tmp_path / "out")
        assert result["successes"] == 80
        assert len(calls) == 1
        assert calls[0][2]["_authorized_test"] is True and calls[0][2]["execution_stride"] == 8
