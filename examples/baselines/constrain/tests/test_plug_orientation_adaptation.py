import numpy as np
import pytest
import h5py
import json
import torch
from pathlib import Path
import subprocess
import sys

from examples.baselines.constrain import plug_orientation_adaptation as adaptation


def test_heldout_cli_is_a_public_command_without_opening_catalogs():
    result = subprocess.run(
        [sys.executable, "-m", "examples.baselines.constrain.plug_orientation_adaptation",
         "evaluate-heldout", "--help"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_heldout_production_preflight_checks_real_selection_checkpoint_and_catalog_hashes():
    root = adaptation.ROOT
    contract = adaptation._heldout_preflight(
        adaptation.HELDOUT_CHECKPOINT, adaptation.HELDOUT_SELECTION,
        adaptation.HELDOUT_CATALOG_ROOT / "random_final_test",
        adaptation.HELDOUT_CATALOG_ROOT / "fixed_base_test")
    assert contract["evaluation"]["episodes_total"] == 600
    assert contract["validation_selection"]["recorded_random_gain_pp"] == 10.0
    assert contract["evaluation"]["execution_stride"] == 8


def test_heldout_rejects_wrong_checkpoint_before_constructing_env_or_policy(tmp_path):
    called = []
    with pytest.raises(ValueError, match="selection/checkpoint identity"):
        adaptation.evaluate_heldout(tmp_path / "not-selected.pt", "unused-random",
            "unused-fixed", tmp_path / "out", env_factory=lambda: called.append("env"),
            policy_factory=lambda _: called.append("policy"), base_policy_factory=lambda: called.append("base"))
    assert called == []


def test_heldout_rejects_rewritten_selection_gate_before_constructing_env_or_policy(tmp_path):
    selection = json.loads(adaptation.HELDOUT_SELECTION.read_text())
    selection["random_gain_pp"] = 20.0
    selection_path = tmp_path / "rewritten_selection.json"
    selection_path.write_text(json.dumps(selection))
    called = []
    with pytest.raises(ValueError, match="selection/checkpoint identity"):
        adaptation.evaluate_heldout(adaptation.HELDOUT_CHECKPOINT, "unused-random",
            "unused-fixed", tmp_path / "out", selection_path=selection_path,
            env_factory=lambda: called.append("env"),
            policy_factory=lambda _: called.append("policy"), base_policy_factory=lambda: called.append("base"))
    assert called == []


def test_heldout_plan_freezes_six_cohorts_and_seeded_shuffled_assignment():
    class Catalog:
        def __init__(self, split, yaws):
            self.manifest = {"split": split, "scenes": [{"target_yaw_rad": yaw} for yaw in yaws]}
        def __len__(self): return 100

    random_cat = Catalog("random_final_test", np.linspace(-0.3, 0.3, 100))
    fixed_cat = Catalog("fixed_base_test", np.zeros(100))
    plan_a = adaptation._heldout_cohort_plan(random_cat, fixed_cat)
    plan_b = adaptation._heldout_cohort_plan(random_cat, fixed_cat)
    assert [(split, role) for split, role, *_ in plan_a] == [
        ("random_final_test", "base"), ("random_final_test", "constrain_correct"),
        ("random_final_test", "constrain_fixed_standard"), ("random_final_test", "constrain_shuffled"),
        ("fixed_base_test", "base"), ("fixed_base_test", "constrain_correct")]
    assert [len(catalog) for _, _, catalog, _ in plan_a] == [100] * 6
    shuffled_a, shuffled_b = plan_a[3][3], plan_b[3][3]
    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    np.testing.assert_array_equal(np.sort(shuffled_a), np.sort(np.linspace(-0.3, 0.3, 100)))
    assert np.count_nonzero(shuffled_a == np.linspace(-0.3, 0.3, 100)) > 0  # self-matches retained
    assert len(np.unique(shuffled_a)) > 1


def test_heldout_fixed_standard_uses_zero_yaw_while_correct_uses_scene_yaw():
    assert adaptation._heldout_hint_target("constrain_correct", 0.37) == pytest.approx(0.37)
    assert adaptation._heldout_hint_target("constrain_fixed_standard", 0.37) == 0.0
    assert adaptation._heldout_hint_target("constrain_shuffled", 0.37, -0.21) == pytest.approx(-0.21)


class _HeldoutFakeCatalog:
    def __init__(self, split, yaw):
        self.manifest = {"split": split, "scenes": [{"target_yaw_rad": yaw,
            "scene_id": f"{split}-{i}"} for i in range(100)]}
    def __len__(self): return 100
    def load(self, index):
        row = self.manifest["scenes"][index]
        return row, {}, np.zeros(46, np.float32)


class _HeldoutFakePolicy:
    hint_mode = "correct"
    def __init__(self, fail_on_call=None): self.contexts, self.calls, self.fail_on_call = [], 0, fail_on_call
    def set_episode_context(self, target): self.contexts.append(target)
    def get_action(self, observations):
        self.calls += 1
        if self.calls == self.fail_on_call: raise RuntimeError("deliberate fake policy failure")
        return torch.zeros((1, 8, 7))


class _HeldoutFakeEnv:
    def __init__(self): self.closed = False
    def step(self, action):
        obs = torch.zeros((1, 46)); obs[0, 28] = 1.0
        return obs, 0.0, True, False, {"success": True,
            "obj_to_goal_dist": torch.tensor([0.001]), "obj_to_goal_angle": torch.tensor([0.05])}
    def close(self): self.closed = True


def test_heldout_fake_loop_runs_one_scene_for_each_of_six_arms(tmp_path):
    random_cat = _HeldoutFakeCatalog("random_final_test", 0.2)
    fixed_cat = _HeldoutFakeCatalog("fixed_base_test", 0.0)
    base_policy, constrain_policy, env = _HeldoutFakePolicy(), _HeldoutFakePolicy(), _HeldoutFakeEnv()
    def identity_obs(*_args):
        obs = torch.zeros((1, 46)); obs[0, 28] = 1.0
        return obs
    result = adaptation._run_heldout_rollouts(random_cat, fixed_cat, tmp_path / "run",
        base_policy, constrain_policy, lambda: env, restore_fn=identity_obs,
        _episode_limit=1)
    assert len(result["completed_episodes"]) == 6
    assert [row["cohort"] for row in result["completed_episodes"]] == [
        "base", "constrain_correct", "constrain_fixed_standard", "constrain_shuffled",
        "base", "constrain_correct"]
    assert all(row["success"] and row["controls"] == 1 for row in result["completed_episodes"])
    assert all(row["official_position_error_m"] == pytest.approx(0.001)
               and row["official_rotation_error_rad"] == pytest.approx(0.05)
               for row in result["completed_episodes"])
    assert 0.0 in constrain_policy.contexts
    assert env.closed
    assert (tmp_path / "run" / "progress.json").is_file()


def test_public_heldout_runs_in_directory_after_writing_its_contract(tmp_path, monkeypatch):
    class Catalog(_HeldoutFakeCatalog):
        def __init__(self, split, yaw):
            self.manifest = {"split": split, "scenes": [
                {"target_yaw_rad": yaw, "scene_id": f"{split}-{i}"} for i in range(100)]}

    class Env(_HeldoutFakeEnv):
        def __init__(self):
            super().__init__()
            self.steps = 0
        def step(self, action):
            self.steps += 1
            return super().step(action)

    catalogs = {"random": Catalog("random_final_test", 0.2),
                "fixed": Catalog("fixed_base_test", 0.0)}
    monkeypatch.setattr(adaptation, "_heldout_preflight", lambda *args: {"registered": True})
    monkeypatch.setattr(adaptation, "sha256_file", lambda path: "fake-sha256")
    monkeypatch.setattr(adaptation, "BASE_SHA256", "fake-base-sha256")
    monkeypatch.setattr(adaptation, "PlugConstrainAdapter", lambda *args, **kwargs: _HeldoutFakePolicy())
    monkeypatch.setattr(adaptation.base, "PlugBaseAdapter", lambda *args, **kwargs: _HeldoutFakePolicy())
    scenes_module = sys.modules["examples.baselines.constrain.plug_orientation_scenes"]
    monkeypatch.setattr(scenes_module, "PlugSceneCatalog", lambda path: catalogs["random"] if path == "random" else catalogs["fixed"])
    monkeypatch.setattr(scenes_module, "restore_plug_scene", lambda env, scene: _identity_heldout_obs())
    env = Env()
    output = tmp_path / "public-run"
    result = adaptation.evaluate_heldout("checkpoint", "random", "fixed", output,
        selection_path="selection", contract_path="contract",
        policy_factory=lambda _: _HeldoutFakePolicy(),
        base_policy_factory=lambda: _HeldoutFakePolicy(), env_factory=lambda: env)
    assert len(result["completed_episodes"]) == 600
    assert env.steps == 600 and env.closed
    assert (output / "heldout_contract.json").is_file()
    assert len(list(output.glob("*_base.jsonl"))) == 2


def _identity_heldout_obs():
    obs = torch.zeros((1, 46)); obs[0, 28] = 1.0
    return obs


def test_heldout_fake_loop_preserves_completed_rows_on_failure(tmp_path):
    env = _HeldoutFakeEnv()
    with pytest.raises(RuntimeError, match="deliberate fake policy failure"):
        def identity_obs(*_args):
            obs = torch.zeros((1, 46)); obs[0, 28] = 1.0
            return obs
        adaptation._run_heldout_rollouts(_HeldoutFakeCatalog("random_final_test", 0.2),
            _HeldoutFakeCatalog("fixed_base_test", 0.0), tmp_path / "partial",
            _HeldoutFakePolicy(), _HeldoutFakePolicy(fail_on_call=2), lambda: env,
            restore_fn=identity_obs, _episode_limit=2)
    partial = json.loads((tmp_path / "partial" / "partial_summary.json").read_text())
    assert len(partial["completed_episodes"]) == 3
    assert env.closed


def test_live_hint_uses_latest_observed_charger_yaw_and_wraps_goal_error():
    obs = np.zeros(46, dtype=np.float32)
    # WXYZ identity with a +0.3 rad world-Z rotation in observation field [28:32].
    obs[28:32] = [np.cos(0.15), 0, 0, np.sin(0.15)]
    hint = adaptation.orientation_hint(obs, target_yaw=-0.2)
    np.testing.assert_allclose(hint, [np.sin(0.5), np.cos(0.5)], atol=1e-6)
    with pytest.raises(ValueError):
        adaptation.orientation_hint(np.zeros(39), 0.0)


def test_forward_contract_pins_frozen_v1_identity_without_qualification_gate():
    contract = adaptation.build_adaptation_contract()
    assert contract["frozen_base"]["format"] == "plug_orientation_fixed_base_dp_v1"
    assert contract["frozen_base"]["update"] == 80000
    assert contract["frozen_base"]["validation_successes"] == 30
    assert contract["base_absolute_gate_waived"] is True
    assert contract["method_gates"]["fixed_retention_loss_pp_max"] == 5


def test_dataset_uses_native_actions_and_hint_from_latest_history_frame(tmp_path):
    h5path, manifest_path = tmp_path / "native.h5", tmp_path / "native.json"
    observations = np.zeros((17, 46), np.float32)
    observations[:, 28] = 1.0
    observations[1, 28:32] = [np.cos(0.1), 0, 0, np.sin(0.1)]
    actions = np.arange(16 * 7, dtype=np.float32).reshape(16, 7) / 200
    with h5py.File(h5path, "w") as handle:
        traj = handle.create_group("traj_0")
        traj.create_dataset("obs", data=observations)
        traj.create_dataset("actions", data=actions)
        traj.create_dataset("env_states/plug_orientation_target", data=np.tile([0.0, 0, 0, 0], (17, 1)))
    manifest_path.write_text(json.dumps({"episodes": [{"target_yaw_rad": 0.0}]}))
    dataset = adaptation.PlugAdaptationDataset(h5path, manifest_path, 1)
    assert len(dataset) == 15  # official PlugBaseDataset includes terminal padding windows
    sample = dataset[1]
    np.testing.assert_array_equal(sample["actions"].numpy(), actions)
    np.testing.assert_allclose(sample["hint"].numpy(), [np.sin(0.2), np.cos(0.2)], atol=1e-6)
    terminal = dataset[-1]
    np.testing.assert_array_equal(terminal["actions"].numpy()[-1, :6], np.zeros(6))
    assert terminal["actions"].numpy()[-1, -1] == actions[-1, -1]


def test_adaptation_selection_obeys_retention_without_absolute_base_gate():
    rows = [{"update": step, "base_fixed_successes": 30, "base_random_successes": 0,
             "fixed_successes": 28,
             "random_successes": step // 3000} for step in adaptation.CHECKPOINT_STEPS]
    result = adaptation.select_adaptation_checkpoint(rows)
    assert result["minimum_fixed_successes"] == 28
    assert result["selected"] == 30000
    assert result["method_gates_met"] is True
    rows[-1]["fixed_successes"] = 27
    assert adaptation.select_adaptation_checkpoint(rows)["selected"] == 25000
    for row in rows:
        row["base_fixed_successes"] = 31
        row["fixed_successes"] = 28
    result = adaptation.select_adaptation_checkpoint(rows)
    assert result["minimum_fixed_successes"] == 29
    assert result["selected"] is None


def test_random_collector_preregisters_all_yaws_and_keeps_failed_attempts(tmp_path):
    calls = []
    def write_trajectory(path, yaw):
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        obs = np.zeros((17, 46), np.float32); obs[:, 28:32] = q
        actions = np.zeros((16, 7), np.float32)
        target = np.tile([yaw, 0, 0, 0], (17, 1))
        charger = np.tile([0, 0, 0, *q], (17, 1))
        with h5py.File(path, "w") as h5:
            traj = h5.create_group("traj_0")
            traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
            traj.create_dataset("env_states/plug_orientation_target", data=target)
            traj.create_dataset("env_states/actors/charger", data=charger)
            traj.create_dataset("env_states/actors/receptacle", data=np.tile([0, 0, 0, 1, 0, 0, 0], (17, 1)))
    def fake_case(seed, output, target_yaw, deadline):
        calls.append((seed, target_yaw))
        output.mkdir(parents=True)
        (output / "replay").mkdir()
        write_trajectory(output / "expert.h5", target_yaw)
        write_trajectory(output / "replay/replay.h5", target_yaw)
        success = seed == 2026123001
        return {"success": success, "target_yaw": target_yaw,
            "expert": {"success": success, "official_success": success},
            "replay": {"success": success}, "expert_controls": 16, "replay_controls": 16}
    result = adaptation.collect_random100(tmp_path / "data", tmp_path / "exp",
        run_case_fn=fake_case, _candidate_count=2, _accepted_count=1)
    prereg = json.loads((tmp_path / "exp/preregistration.json").read_text())
    assert prereg["candidates"] == [{"seed": s, "target_yaw_rad": y} for s, y in calls]
    assert [r["status"] for r in result["attempts"]] == ["failed", "accepted"]
    assert result["attempts"][0]["source_h5_sha256"]
    assert (tmp_path / "data/trajectory.state.pd_ee_delta_pose.physx_cpu.h5").is_file()


def test_random200_expansion_seam_is_registered_and_starts_after_original_candidates():
    assert hasattr(adaptation, "collect_random200")
    plan = adaptation.random200_candidate_plan()
    assert len(plan) == 150
    assert [row["seed"] for row in plan] == list(range(2026123150, 2026123300))
    rng = np.random.default_rng(1704)
    expected = rng.uniform(-np.pi / 8, np.pi / 8, size=300)[150:]
    np.testing.assert_array_equal([row["target_yaw_rad"] for row in plan], expected)


def test_random300_expansion_plan_starts_after_the_complete_random200_stream():
    assert hasattr(adaptation, "collect_random300")
    assert adaptation.REGISTERED_RANDOM_EPISODE_COUNTS == {100, 200, 300, 500}
    plan = adaptation.random300_candidate_plan()
    assert len(plan) == 150
    assert [row["seed"] for row in plan] == list(range(2026123300, 2026123450))
    expected = np.random.default_rng(1704).uniform(-np.pi / 8, np.pi / 8, 450)[300:]
    np.testing.assert_array_equal([row["target_yaw_rad"] for row in plan], expected)


def test_random500_expansion_plan_uses_only_new_draws_and_reserved_seeds():
    assert hasattr(adaptation, "collect_random500")
    assert adaptation.REGISTERED_RANDOM_EPISODE_COUNTS == {100, 200, 300, 500}
    plan = adaptation.random500_candidate_plan()
    assert len(plan) == 300
    assert [row["seed"] for row in plan] == list(range(2026123450, 2026123750))
    expected = np.random.default_rng(1704).uniform(-np.pi / 8, np.pi / 8, 750)[450:]
    np.testing.assert_array_equal([row["target_yaw_rad"] for row in plan], expected)


def test_random500_cli_routes_to_immutable_parent_and_explicit_training_count(monkeypatch):
    calls = []
    monkeypatch.setattr(adaptation, "collect_random500",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    adaptation.main(["collect-random500", "--parent-dataset", str(adaptation.RANDOM300_DATASET),
        "--dataset-dir", "data500", "--exp-dir", "exp500"])
    assert calls == [(("data500", "exp500"), {"parent_dataset": adaptation.RANDOM300_DATASET})]
    training = []
    monkeypatch.setattr(adaptation, "train", lambda *args, **kwargs: training.append((args, kwargs)))
    adaptation.main(["train-expanded500", "--random-h5", "random.h5", "--random-manifest", "manifest.json",
        "--fixed-h5", "fixed.h5", "--fixed-manifest", "fixed.json", "--exp-dir", "train-exp",
        "--ckpt-dir", "train-ckpt"])
    assert training[0][1]["_random_episodes"] == 500


def test_random500_audits_real_immutable_300_parent_before_first_callback(tmp_path):
    observed = []
    def stop_before_native(seed, output, *, target_yaw, deadline):
        observed.append((seed, target_yaw))
        raise RuntimeError("offline preflight stop")
    with pytest.raises(RuntimeError, match="offline preflight stop"):
        adaptation.collect_random500(tmp_path / "data", tmp_path / "exp",
            run_case_fn=stop_before_native)
    assert len(observed) == 1
    assert observed[0][0] == 2026123450
    assert observed[0][1] == adaptation.random500_candidate_plan()[0]["target_yaw_rad"]


def test_random500_collector_and_real_lineage_reader_admit_parent_first_pool(tmp_path):
    from examples.baselines.constrain import plug_orientation_data as native_data
    calls = []
    def synthetic_native_pair(seed, output, *, target_yaw, deadline):
        calls.append(seed)
        output.mkdir(parents=True)
        (output / "replay").mkdir()
        q = [np.cos(target_yaw / 2), 0, 0, np.sin(target_yaw / 2)]
        obs = np.zeros((17, 46), np.float32); obs[:, 28:32] = q
        actions = np.zeros((16, 7), np.float32)
        target = np.tile([target_yaw, 0, 0, 0], (17, 1))
        charger = np.tile([0, 0, 0, *q], (17, 1))
        receptacle = np.tile([0, 0, 0, 1, 0, 0, 0], (17, 1))
        for path in (output / "expert.h5", output / "replay/replay.h5"):
            with h5py.File(path, "w") as handle:
                traj = handle.create_group("traj_0")
                traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
                traj.create_dataset("env_states/plug_orientation_target", data=target)
                traj.create_dataset("env_states/actors/charger", data=charger)
                traj.create_dataset("env_states/actors/receptacle", data=receptacle)
        return {"target_yaw": target_yaw,
            "expert": {"success": True, "official_success": True}, "replay": {"success": True},
            "expert_controls": 16, "replay_controls": 16}
    dataset = tmp_path / "random500"
    exp = tmp_path / "random500_exp"
    manifest = adaptation.collect_random500(dataset, exp, run_case_fn=synthetic_native_pair)
    plan = adaptation.random500_candidate_plan()
    assert calls == [row["seed"] for row in plan[:200]]
    assert manifest["accepted_count"] == 500 and manifest["new_accepted_count"] == 200
    assert manifest["new_source_controls"] == manifest["new_replay_controls"] == 3200
    assert all(manifest["new_yaw_bin_counts"])
    assert all(manifest["effective_yaw_bin_counts"])
    reader = adaptation.PlugAdaptationDataset(dataset / native_data.DATA_H5,
        dataset / "dataset_manifest.json", 500)
    assert len(reader.obs) > 0
    assert manifest["parent_dataset_manifest_sha256"] == adaptation.PARENT_RANDOM300_MANIFEST_SHA256
    fixed_h5, fixed_json = tmp_path / "fixed.h5", tmp_path / "fixed.json"
    with h5py.File(fixed_h5, "w") as handle:
        traj = handle.create_group("traj_0")
        obs = np.zeros((17, 46), np.float32); obs[:, 28] = 1
        traj.create_dataset("obs", data=obs)
        traj.create_dataset("actions", data=np.zeros((16, 7), np.float32))
        traj.create_dataset("env_states/plug_orientation_target", data=np.zeros((17, 4), np.float32))
    fixed_json.write_text(json.dumps({"episodes": [{"target_yaw_rad": 0.0}]}))
    trained = adaptation.train(dataset / native_data.DATA_H5, dataset / "dataset_manifest.json",
        fixed_h5, fixed_json, tmp_path / "toy_train", tmp_path / "toy_ckpt", device="cpu",
        _updates=2, _batch_size=2, _test_mode=True, _random_episodes=500, _fixed_episodes=1)
    assert trained["updates"] == 2
    toy = torch.load(tmp_path / "toy_ckpt/checkpoint_000002.pt", map_location="cpu", weights_only=False)
    assert toy["config"]["training"]["random_episode_count"] == 500
    assert toy["random_manifest_sha256"] == native_data.sha256_file(dataset / "dataset_manifest.json")
    registration = json.loads((dataset / "dataset_manifest.json").read_text())
    lineage_path = Path(registration["parent_lineage_path"])
    original_lineage = lineage_path.read_bytes()
    lineage_path.write_bytes(original_lineage + b"\n")
    with pytest.raises(ValueError, match="parent-lineage record hash"):
        adaptation.PlugAdaptationDataset(dataset / native_data.DATA_H5,
            dataset / "dataset_manifest.json", 500)


def test_random300_cli_routes_only_to_registered_parent_and_training_count(monkeypatch):
    calls = []
    monkeypatch.setattr(adaptation, "collect_random300",
        lambda *args, **kwargs: calls.append((args, kwargs)))
    adaptation.main(["collect-random300", "--parent-dataset", str(adaptation.RANDOM200_DATASET),
        "--dataset-dir", "data300", "--exp-dir", "exp300"])
    assert calls == [(("data300", "exp300"), {"parent_dataset": adaptation.RANDOM200_DATASET})]
    training = []
    monkeypatch.setattr(adaptation, "train", lambda *args, **kwargs: training.append((args, kwargs)))
    adaptation.main(["train-expanded300", "--random-h5", "random.h5", "--random-manifest", "manifest.json",
        "--fixed-h5", "fixed.h5", "--fixed-manifest", "fixed.json", "--exp-dir", "train-exp",
        "--ckpt-dir", "train-ckpt"])
    assert training[0][1]["_random_episodes"] == 300


def test_random200_training_count_is_explicit_not_arbitrary():
    assert 200 in adaptation.REGISTERED_RANDOM_EPISODE_COUNTS
    assert adaptation.REGISTERED_RANDOM_EPISODE_COUNTS == {100, 200, 300, 500}
    legacy = {"config": {"training": {"seed": 1}}}
    assert adaptation._checkpoint_random_episode_count(legacy) == 100
    expanded = {"random_episode_count": 200,
        "config": {"training": {"random_episode_count": 200}},
        "expansion_contract_sha256": adaptation.sha256_file(
            adaptation.Path(adaptation.__file__).with_name("plug_orientation_adaptation_expansion200_contract.json")),
        "parent_dataset_manifest_sha256": adaptation.PARENT_RANDOM100_MANIFEST_SHA256,
        "random_h5_sha256": "a" * 64, "random_manifest_sha256": "b" * 64}
    assert adaptation._checkpoint_random_episode_count(expanded) == 200
    for bad_count in (199, 301):
        with pytest.raises(ValueError):
            adaptation._checkpoint_random_episode_count({"random_episode_count": bad_count})


def test_random300_checkpoint_count_requires_exact_expansion_contract_and_parent():
    payload = {"random_episode_count": 300, "random_h5_sha256": "a" * 64,
        "random_manifest_sha256": "b" * 64,
        "parent_dataset_manifest_sha256": adaptation.PARENT_RANDOM200_MANIFEST_SHA256,
        "expansion_contract_sha256": adaptation.sha256_file(
            adaptation.Path(adaptation.__file__).with_name("plug_orientation_adaptation_expansion300_contract.json")),
        "config": {"training": {"random_episode_count": 300}}}
    assert adaptation._checkpoint_random_episode_count(payload) == 300
    payload["parent_dataset_manifest_sha256"] = "wrong"
    with pytest.raises(ValueError, match="300-demo checkpoint"):
        adaptation._checkpoint_random_episode_count(payload)


def test_random500_checkpoint_count_requires_exact_contract_parent_and_payload_hashes():
    payload = {"random_episode_count": 500, "random_h5_sha256": "a" * 64,
        "random_manifest_sha256": "b" * 64,
        "expansion_contract_sha256": adaptation.sha256_file(adaptation.Path(adaptation.__file__).with_name(
            "plug_orientation_adaptation_expansion500_contract.json")),
        "parent_dataset_manifest_sha256": adaptation.PARENT_RANDOM300_MANIFEST_SHA256,
        "config": {"training": {"random_episode_count": 500}}}
    assert adaptation._checkpoint_random_episode_count(payload) == 500
    for field, value in (("parent_dataset_manifest_sha256", "f" * 64),
                         ("expansion_contract_sha256", "f" * 64), ("random_h5_sha256", "bad")):
        with pytest.raises(ValueError, match="500-demo checkpoint"):
            adaptation._checkpoint_random_episode_count({**payload, field: value})


def test_random300_real_parent_lineage_is_preflighted_before_callback(tmp_path):
    calls = []
    def stop_before_native(seed, output, target_yaw, deadline):
        calls.append((seed, target_yaw))
        raise RuntimeError("intentional offline callback stop")

    with pytest.raises(RuntimeError, match="intentional offline callback stop"):
        adaptation.collect_random300(tmp_path / "data", tmp_path / "exp",
            parent_dataset=adaptation.RANDOM200_DATASET, run_case_fn=stop_before_native)
    assert len(calls) == 1 and calls[0][0] == 2026123300
    attempts = [json.loads(line) for line in (tmp_path / "data/attempts.jsonl").read_text().splitlines()]
    assert attempts[0]["failure_class"] == "unexpected_exception"
    assert attempts[0]["expert_controls"] is None


def test_random300_collector_admits_only_fresh_first_eligible_rows_offline(tmp_path):
    from examples.baselines.constrain import plug_orientation_data as native_data
    from examples.baselines.constrain import plug_orientation_adaptation as adaptation_module

    calls = []
    def fake_native_case(seed, output, target_yaw, deadline):
        calls.append(seed)
        output.mkdir(parents=True)
        (output / "replay").mkdir()
        q = [np.cos(target_yaw / 2), 0, 0, np.sin(target_yaw / 2)]
        obs = np.zeros((17, 46), np.float32); obs[:, 28:32] = q
        actions = np.zeros((16, 7), np.float32)
        target = np.tile([target_yaw, 0, 0, 0], (17, 1))
        charger = np.tile([0, 0, 0, *q], (17, 1))
        receptacle = np.tile([0, 0, 0, 1, 0, 0, 0], (17, 1))
        for path in (output / "expert.h5", output / "replay/replay.h5"):
            with h5py.File(path, "w") as handle:
                traj = handle.create_group("traj_0")
                traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
                traj.create_dataset("env_states/plug_orientation_target", data=target)
                traj.create_dataset("env_states/actors/charger", data=charger)
                traj.create_dataset("env_states/actors/receptacle", data=receptacle)
        return {"target_yaw": target_yaw,
            "expert": {"success": True, "official_success": True}, "replay": {"success": True},
            "expert_controls": 16, "replay_controls": 16}

    dataset_dir, exp_dir = tmp_path / "random300", tmp_path / "random300_exp"
    result = adaptation.collect_random300(dataset_dir, exp_dir, run_case_fn=fake_native_case)
    plan = adaptation.random300_candidate_plan()
    assert calls == [row["seed"] for row in plan[:100]]
    assert result["accepted_count"] == 300 and result["new_accepted_count"] == 100
    assert result["new_candidate_attempt_count"] == 100
    assert result["new_source_controls"] == result["new_replay_controls"] == 1600
    prereg = json.loads((exp_dir / "preregistration.json").read_text())
    assert prereg["candidate_plan"] == plan
    assert prereg["yaw_rng_draw_start"] == 300
    assert result["new_yaw_bin_counts"] == np.histogram(
        [row["target_yaw_rad"] for row in result["attempts"]], bins=np.linspace(-np.pi / 8, np.pi / 8, 6))[0].tolist()
    manifest_path = dataset_dir / "dataset_manifest.json"
    reader = adaptation_module.PlugAdaptationDataset(
        dataset_dir / native_data.DATA_H5, manifest_path, 300)
    assert len(reader.obs) == 300
    assert result["parent_dataset_manifest_sha256"] == adaptation.PARENT_RANDOM200_MANIFEST_SHA256


def test_random300_loader_checks_real_parent_and_ordered_new_source_values(tmp_path, monkeypatch):
    from examples.baselines.constrain import plug_orientation_data as native_data

    parent = tmp_path / "parent200"; parent.mkdir()
    parent_h5 = parent / native_data.DATA_H5
    parent_json = parent / native_data.DATA_JSON
    parent_rows = []
    def write_native(path, yaw, action_value):
        obs = np.zeros((17, 46), np.float32); obs[:, 28] = 1
        actions = np.full((16, 7), action_value, np.float32)
        with h5py.File(path, "w") as handle:
            traj = handle.create_group("traj_0")
            traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
            traj.create_dataset("env_states/plug_orientation_target", data=np.tile([yaw, 0, 0, 0], (17, 1)))
        return hashlib.sha256(actions.tobytes()).hexdigest()
    import hashlib
    def raw_row(seed, yaw, source, replay, case, action_hash, lineage=None):
        row = {"episode_seed": seed, "source_h5": str(source),
            "source_h5_sha256": native_data.sha256_file(source), "replay_h5": str(replay),
            "replay_h5_sha256": native_data.sha256_file(replay), "case_json": str(case),
            "case_json_sha256": native_data.sha256_file(case), "source_actions_sha256": action_hash,
            "target_yaw_rad": float(yaw)}
        if lineage is not None: row["lineage"] = lineage
        return row
    with h5py.File(parent_h5, "w") as output:
        for i in range(200):
            yaw = (-0.314, -0.157, 0.0, 0.157, 0.314)[i % 5]
            source, replay, case = (tmp_path / f"parent_source_{i}.h5", tmp_path / f"parent_replay_{i}.h5",
                                    tmp_path / f"parent_case_{i}.json")
            action_hash = write_native(source, yaw, 0.1)
            replay.write_bytes(source.read_bytes())
            case.write_text(json.dumps({"target_yaw": yaw}))
            parent_rows.append(raw_row(2026123000 + i, yaw, source, replay, case, action_hash))
            with h5py.File(source, "r") as source_h5:
                source_h5.copy("traj_0", output, name=f"traj_{i}")
    parent_json.write_text(json.dumps({"episodes": parent_rows}))
    parent_manifest = parent / "dataset_manifest.json"
    parent_manifest.write_text(json.dumps({"protocol": "plug_orientation_random200_expansion_v1",
        "status": "complete", "accepted_count": 200, "effective_episode_count": 200}))
    parent_hashes = {"manifest": native_data.sha256_file(parent_manifest),
        "h5": native_data.sha256_file(parent_h5), "json": native_data.sha256_file(parent_json)}
    monkeypatch.setattr(adaptation, "RANDOM200_DATASET", parent)
    monkeypatch.setattr(adaptation, "PARENT_RANDOM200_MANIFEST_SHA256", parent_hashes["manifest"])
    monkeypatch.setattr(adaptation, "PARENT_RANDOM200_H5_SHA256", parent_hashes["h5"])
    monkeypatch.setattr(adaptation, "PARENT_RANDOM200_JSON_SHA256", parent_hashes["json"])
    monkeypatch.setattr(adaptation, "_validate_random200_lineage", lambda *args: None)
    monkeypatch.setattr(native_data, "_native_episode_ok", lambda *args, **kwargs: (True, {}))
    monkeypatch.setattr(native_data, "_pair_native_match", lambda *args, **kwargs: True)

    dataset = tmp_path / "combined"; dataset.mkdir()
    exp = tmp_path / "collection"; (exp / "source_archive").mkdir(parents=True)
    plan = adaptation.random300_candidate_plan()
    prereg_path = exp / "preregistration.json"
    prereg_path.write_text(json.dumps({"candidate_plan": plan, "source_sha256": {}}))
    rows, attempts = [], []
    combined_h5 = dataset / native_data.DATA_H5
    with h5py.File(combined_h5, "w") as output, h5py.File(parent_h5, "r") as frozen:
        for i, row in enumerate(parent_rows):
            frozen.copy(f"traj_{i}", output, name=f"traj_{i}")
            rows.append({**row, "lineage": {"kind": "immutable_parent_episode", "parent_episode_id": i,
                "parent_manifest_sha256": parent_hashes["manifest"],
                "parent_h5_sha256": parent_hashes["h5"], "parent_json_sha256": parent_hashes["json"],
                "parent_episode_metadata": row}})
        for i, candidate in enumerate(plan[:100]):
            source, replay, case = (dataset / f"source_{i}.h5", dataset / f"replay_{i}.h5",
                                    dataset / f"case_{i}.json")
            action_hash = write_native(source, candidate["target_yaw_rad"], 0.2)
            replay.write_bytes(source.read_bytes())
            case.write_text(json.dumps({"target_yaw": candidate["target_yaw_rad"]}))
            row = raw_row(candidate["seed"], float(np.float32(candidate["target_yaw_rad"])),
                          source, replay, case, action_hash, {"kind": "new_candidate"})
            with h5py.File(source, "r") as raw: raw.copy("traj_0", output, name=f"traj_{200+i}")
            rows.append(row)
            attempts.append({**candidate, "candidate_index": i + 1, "accepted": True,
                "expert_controls": 16, "replay_controls": 16,
                "source_h5": str(source), "source_h5_sha256": row["source_h5_sha256"],
                "replay_h5": str(replay), "replay_h5_sha256": row["replay_h5_sha256"],
                "case_json": str(case), "case_json_sha256": row["case_json_sha256"],
                "source_actions_sha256": action_hash})
    training_json = dataset / native_data.DATA_JSON
    training_json.write_text(json.dumps({"episodes": rows}))
    journal = dataset / "attempts.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in attempts))
    registration = {"protocol": "plug_orientation_random300_expansion_v1", "status": "complete",
        "accepted_count": 300, "effective_episode_count": 300, "new_accepted_count": 100,
        "parent_dataset_manifest_sha256": parent_hashes["manifest"],
        "parent_h5_sha256": parent_hashes["h5"], "parent_json_sha256": parent_hashes["json"],
        "expansion_contract_sha256": adaptation.sha256_file(adaptation.Path(adaptation.__file__).with_name(
            "plug_orientation_adaptation_expansion300_contract.json")),
        "collection_experiment_dir": str(exp), "preregistration_sha256": native_data.sha256_file(prereg_path),
        "attempts": attempts, "attempts_sha256": native_data.sha256_file(journal),
        "new_source_controls": 1600, "new_replay_controls": 1600,
        "new_yaw_bin_counts": np.histogram([row["target_yaw_rad"] for row in rows[200:]],
            bins=np.linspace(-np.pi / 8, np.pi / 8, 6))[0].tolist(),
        "effective_yaw_bin_counts": np.histogram([row["target_yaw_rad"] for row in rows],
            bins=np.linspace(-np.pi / 8, np.pi / 8, 6))[0].tolist(),
        "training_h5_sha256": native_data.sha256_file(combined_h5), "training_h5_path": str(combined_h5),
        "training_json_sha256": native_data.sha256_file(training_json), "training_json_path": str(training_json)}
    registration_path = dataset / "dataset_manifest.json"
    registration_path.write_text(json.dumps(registration))
    loaded = adaptation.PlugAdaptationDataset(combined_h5, registration_path, 300)
    assert len(loaded) == 300 * 15


def test_random200_fake_native_collection_preserves_parent_and_consolidates_first(tmp_path):
    from examples.baselines.constrain import plug_orientation_data as native_data

    parent = tmp_path / "parent"
    parent.mkdir()
    parent_rows = []

    def write_case(seed, out, yaw):
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        obs = np.zeros((17, 46), np.float32); obs[:, 28:32] = q
        actions = np.full((16, 7), seed % 3 / 10, np.float32)
        states = np.tile([0, 0, 0, *q], (17, 1))
        target = np.tile([yaw, 0, 0, 0], (17, 1))
        for path in (out / "expert.h5", out / "replay/replay.h5"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(path, "w") as h5:
                traj = h5.create_group("traj_0")
                traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
                traj.create_dataset("env_states/plug_orientation_target", data=target)
                traj.create_dataset("env_states/actors/charger", data=states)
                traj.create_dataset("env_states/actors/receptacle", data=np.tile(
                    [0, 0, 0, 1, 0, 0, 0], (17, 1)))
        case = {"target_yaw": yaw, "expert": {"success": True, "official_success": True},
                "replay": {"success": True}, "expert_controls": 16, "replay_controls": 16}
        (out / "case.json").write_text(json.dumps(case))
        return case

    for i, yaw in enumerate((0.0, 0.1)):
        out = parent / f"parent_{i}"; case = write_case(i, out, yaw)
        parent_rows.append({"accepted": True, "seed": i, "target_yaw": yaw,
            "source_h5": str(out / "expert.h5"), "replay_h5": str(out / "replay/replay.h5"),
            "case_json": str(out / "case.json")})
    parent_h5 = parent / native_data.DATA_H5
    with h5py.File(parent_h5, "w") as target:
        for i in range(2):
            with h5py.File(parent / f"parent_{i}/expert.h5", "r") as src:
                src.copy("traj_0", target, name=f"traj_{i}")
    parent_json = parent / native_data.DATA_JSON
    parent_json.write_text(json.dumps({"episodes": [{"episode_id": i} for i in range(2)]}))
    manifest_path = parent / "dataset_manifest.json"
    parent_exp = tmp_path / "parent_exp"; parent_exp.mkdir()
    parent_prereg = parent_exp / "preregistration.json"; parent_prereg.write_text("{}\n")
    manifest_path.write_text(json.dumps({"protocol": "plug_orientation_random100_v1",
        "accepted_count": 2, "attempts": parent_rows,
        "collection_experiment_dir": str(parent_exp),
        "preregistration_sha256": native_data.sha256_file(parent_prereg),
        "training_h5_sha256": native_data.sha256_file(parent_h5),
        "training_json_sha256": native_data.sha256_file(parent_json)}))
    before = {p.name: native_data.sha256_file(p) for p in (parent_h5, parent_json, manifest_path)}

    def fake_case(seed, output, target_yaw, deadline):
        return write_case(seed, output, target_yaw)

    manifest_bytes = manifest_path.read_bytes()
    malformed = json.loads(manifest_bytes); malformed["training_h5_sha256"] = "bad-parent-hash"
    manifest_path.write_text(json.dumps(malformed))
    callback_calls = []
    with pytest.raises(ValueError, match="parent consolidated"):
        adaptation.collect_random200(tmp_path / "bad", tmp_path / "bad_exp",
            parent_dataset=parent, run_case_fn=lambda *a, **k: callback_calls.append(a),
            _candidate_count=2, _accepted_count=2, _parent_count=2)
    assert callback_calls == []
    manifest_path.write_bytes(manifest_bytes)

    result = adaptation.collect_random200(tmp_path / "combined", tmp_path / "exp",
        parent_dataset=parent, run_case_fn=fake_case, _candidate_count=2,
        _accepted_count=2, _parent_count=2)
    assert result["accepted_count"] == 4
    prereg = json.loads((tmp_path / "exp/preregistration.json").read_text())
    assert [x["seed"] for x in prereg["candidate_plan"]] == [2026123150, 2026123151]
    np.testing.assert_array_equal([x["target_yaw_rad"] for x in prereg["candidate_plan"]],
                                  np.random.default_rng(1704).uniform(-np.pi / 8, np.pi / 8, 300)[150:152])
    output_h5 = tmp_path / "combined" / native_data.DATA_H5
    with h5py.File(parent_h5, "r") as old, h5py.File(output_h5, "r") as new:
        for i in range(2):
            old_arrays, new_arrays = {}, {}
            def collect(target):
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset): target[name] = obj[:]
                return visit
            old[f"traj_{i}"].visititems(collect(old_arrays))
            new[f"traj_{i}"].visititems(collect(new_arrays))
            assert old_arrays.keys() == new_arrays.keys()
            for name in old_arrays:
                np.testing.assert_array_equal(old_arrays[name], new_arrays[name])
    assert before == {p.name: native_data.sha256_file(p) for p in (parent_h5, parent_json, manifest_path)}

    callback_count = 0
    def seal_mutating_final_callback(seed, output, target_yaw, deadline):
        nonlocal callback_count
        callback_count += 1
        result = write_case(seed, output, target_yaw)
        if callback_count == 2:
            prereg_path = tmp_path / "tamper_exp/preregistration.json"
            prereg = json.loads(prereg_path.read_text())
            prereg["candidate_plan"][0]["target_yaw_rad"] += 0.01
            prereg_path.write_text(json.dumps(prereg))
        return result
    with pytest.raises(RuntimeError, match="before expansion consolidation"):
        adaptation.collect_random200(tmp_path / "tampered", tmp_path / "tamper_exp",
            parent_dataset=parent, run_case_fn=seal_mutating_final_callback,
            _candidate_count=2, _accepted_count=2, _parent_count=2)
    assert callback_count == 2
    assert not (tmp_path / "tampered" / native_data.DATA_H5).exists()


def test_random200_real_parent_preflight_finishes_before_fake_teacher_callback(tmp_path):
    from examples.baselines.constrain import plug_orientation_data as native_data
    parent = adaptation.RANDOM100_DATASET
    before = native_data.sha256_file(parent / "dataset_manifest.json")
    callbacks = []

    def stop_before_native(seed, output, target_yaw, deadline):
        callbacks.append((seed, target_yaw))
        raise RuntimeError("intentional offline callback stop")

    with pytest.raises(RuntimeError, match="intentional offline callback stop"):
        adaptation.collect_random200(tmp_path / "data", tmp_path / "exp",
            parent_dataset=parent, run_case_fn=stop_before_native)
    assert len(callbacks) == 1 and callbacks[0][0] == 2026123150
    assert native_data.sha256_file(parent / "dataset_manifest.json") == before
    attempts = [json.loads(line) for line in (tmp_path / "data/attempts.jsonl").read_text().splitlines()]
    assert attempts[0]["failure_class"] == "unexpected_exception"
    assert attempts[0]["expert_controls"] is None


def test_random200_tampered_parent_manifest_identity_fails_before_callback(tmp_path):
    import os
    from examples.baselines.constrain import plug_orientation_data as native_data
    real_parent = adaptation.RANDOM100_DATASET
    parent = tmp_path / "tampered_parent"
    parent.mkdir()
    for name in (native_data.DATA_H5, native_data.DATA_JSON):
        os.symlink((real_parent / name).resolve(), parent / name)
    manifest_path = real_parent / "dataset_manifest.json"
    (parent / "dataset_manifest.json").write_bytes(manifest_path.read_bytes() + b" ")
    callbacks = []
    with pytest.raises(ValueError, match="original random-100 registration"):
        adaptation.collect_random200(tmp_path / "data", tmp_path / "exp",
            parent_dataset=parent, run_case_fn=lambda *args, **kwargs: callbacks.append(args))
    assert callbacks == []


def test_random200_known_planning_failure_with_unknown_controls_is_journaled(tmp_path):
    parent = adaptation.RANDOM100_DATASET
    def fail_without_counter(*args, **kwargs):
        raise RuntimeError("MPLib screw planning failed")
    with pytest.raises(RuntimeError, match="lacks auditable control count"):
        adaptation.collect_random200(tmp_path / "data", tmp_path / "exp",
            parent_dataset=parent, run_case_fn=fail_without_counter)
    row = json.loads((tmp_path / "data/attempts.jsonl").read_text().splitlines()[0])
    summary = json.loads((tmp_path / "data/partial_summary.json").read_text())
    assert row["expert_controls"] is None and row["replay_controls"] is None
    assert summary["unknown_source_control_attempts"] == 1
    assert summary["unknown_replay_control_attempts"] == 1


def test_random200_candidate_finishing_after_deadline_is_not_admitted(tmp_path, monkeypatch):
    parent = adaptation.RANDOM100_DATASET
    clock = [0.0]
    monkeypatch.setattr(adaptation.time, "monotonic", lambda: clock[0])
    def late_candidate(seed, output, target_yaw, deadline):
        clock[0] = 2.0
        output.mkdir(parents=True)
        return {"expert_controls": 10, "replay_controls": 10}
    with pytest.raises(TimeoutError, match="deadline expired before manifest publication"):
        adaptation.collect_random200(tmp_path / "data", tmp_path / "exp",
            parent_dataset=parent, run_case_fn=late_candidate, _wall_seconds=1)
    row = json.loads((tmp_path / "data/attempts.jsonl").read_text().splitlines()[0])
    assert row["accepted"] is False
    assert row["expert_controls"] == 10 and row["replay_controls"] == 10
    assert (tmp_path / "data/partial_summary.json").is_file()
    assert not (tmp_path / "data/trajectory.state.pd_ee_delta_pose.physx_cpu.h5").exists()


def test_real_frozen_base_zero_control_adapter_matches_official_sampler(tmp_path):
    agent, _ = adaptation._load_frozen_base()
    denoiser = adaptation.ConstrainDenoiser(agent.noise_pred_net, cond_dim=2)
    checkpoint = tmp_path / "zero-control.pt"
    torch.save({"format": adaptation.FORMAT, "config": {"model": adaptation.MODEL_CONFIG},
        "model": denoiser.state_dict(), "ema_control": [p.detach().clone()
            for p in denoiser.trainable_parameters()], "frozen_base_sha256": adaptation.BASE_SHA256,
        "frozen_base_ema_sha256": adaptation.BASE_EMA_SHA256}, checkpoint)
    adapter = adaptation.PlugConstrainAdapter(checkpoint, control_enabled=True)
    obs = torch.zeros((1, 2, 46), dtype=torch.float32)
    obs[:, :, 28] = 1.0
    torch.manual_seed(77)
    expected = agent.get_action(obs)
    torch.manual_seed(77)
    actual = adapter.get_action(obs)
    assert expected.shape == actual.shape == (1, 8, 7)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
    disabled = adaptation.PlugConstrainAdapter(checkpoint, control_enabled=False)
    torch.manual_seed(77)
    torch.testing.assert_close(disabled.get_action(obs), expected, atol=1e-6, rtol=0)
    expanded_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expanded_payload.update(random_episode_count=200,
        expansion_contract_sha256=adaptation.sha256_file(
            adaptation.Path(adaptation.__file__).with_name("plug_orientation_adaptation_expansion200_contract.json")),
        parent_dataset_manifest_sha256=adaptation.PARENT_RANDOM100_MANIFEST_SHA256,
        random_h5_sha256="a" * 64, random_manifest_sha256="b" * 64)
    expanded_payload["config"]["training"] = {"random_episode_count": 200}
    expanded = tmp_path / "zero-control-random200.pt"; torch.save(expanded_payload, expanded)
    expanded_adapter = adaptation.PlugConstrainAdapter(expanded, control_enabled=True)
    assert expanded_adapter.metadata["checkpoint_sha256"] == adaptation.sha256_file(expanded)
    assert expanded_adapter.metadata["random_episode_count"] == 200
    payload300 = torch.load(expanded, map_location="cpu", weights_only=False)
    payload300.update(random_episode_count=300,
        expansion_contract_sha256=adaptation.sha256_file(
            adaptation.Path(adaptation.__file__).with_name("plug_orientation_adaptation_expansion300_contract.json")),
        parent_dataset_manifest_sha256=adaptation.PARENT_RANDOM200_MANIFEST_SHA256)
    payload300["config"]["training"]["random_episode_count"] = 300
    random300_checkpoint = tmp_path / "zero-control-random300.pt"
    torch.save(payload300, random300_checkpoint)
    random300_adapter = adaptation.PlugConstrainAdapter(random300_checkpoint, control_enabled=True)
    assert random300_adapter.metadata["random_episode_count"] == 300
    payload500 = torch.load(expanded, map_location="cpu", weights_only=False)
    payload500.update(random_episode_count=500,
        expansion_contract_sha256=adaptation.sha256_file(
            adaptation.Path(adaptation.__file__).with_name("plug_orientation_adaptation_expansion500_contract.json")),
        parent_dataset_manifest_sha256=adaptation.PARENT_RANDOM300_MANIFEST_SHA256)
    payload500["config"]["training"]["random_episode_count"] = 500
    random500_checkpoint = tmp_path / "zero-control-random500.pt"
    torch.save(payload500, random500_checkpoint)
    random500_adapter = adaptation.PlugConstrainAdapter(random500_checkpoint, control_enabled=True)
    assert random500_adapter.metadata["random_episode_count"] == 500


def test_adapter_rejects_checkpoint_that_changes_frozen_base_with_valid_metadata(tmp_path):
    agent, _ = adaptation._load_frozen_base()
    denoiser = adaptation.ConstrainDenoiser(agent.noise_pred_net, cond_dim=2)
    model = {key: value.detach().clone() for key, value in denoiser.state_dict().items()}
    base_key = next(key for key in model if key.startswith("base_net.") and model[key].is_floating_point())
    model[base_key].view(-1)[0] += 1
    checkpoint = tmp_path / "changed-base.pt"
    torch.save({"format": adaptation.FORMAT, "config": {"model": adaptation.MODEL_CONFIG},
        "model": model, "ema_control": [p.detach().clone() for p in denoiser.trainable_parameters()],
        "frozen_base_sha256": adaptation.BASE_SHA256,
        "frozen_base_ema_sha256": adaptation.BASE_EMA_SHA256}, checkpoint)

    with pytest.raises(ValueError, match="pinned frozen Base"):
        adaptation.PlugConstrainAdapter(checkpoint)


@pytest.mark.parametrize("random_count", [100, 300, 500])
def test_validation_reuses_fresh_base_once_for_six_candidate_cohorts(tmp_path, monkeypatch, random_count):
    from examples.baselines.constrain import plug_orientation_scenes
    agent, _ = adaptation._load_frozen_base()
    denoiser = adaptation.ConstrainDenoiser(agent.noise_pred_net, cond_dim=2)
    ema_control = [p.detach().clone() for p in denoiser.trainable_parameters()]
    ckpt_dir = tmp_path / "ckpt"; ckpt_dir.mkdir()
    provenance = {key: "frozen-" + key for key in (
        "random_h5_sha256", "random_manifest_sha256", "fixed_h5_sha256",
        "fixed_manifest_sha256", "contract_sha256", "source_closure_sha256")}
    provenance.update(frozen_base_sha256=adaptation.BASE_SHA256,
                      frozen_base_ema_sha256=adaptation.BASE_EMA_SHA256)
    if random_count in {300, 500}:
        expansion_count = random_count
        provenance.update(expansion_contract_sha256=adaptation.sha256_file(
            Path(adaptation.__file__).with_name(f"plug_orientation_adaptation_expansion{expansion_count}_contract.json")),
            parent_dataset_manifest_sha256=(adaptation.PARENT_RANDOM200_MANIFEST_SHA256 if random_count == 300
                                            else adaptation.PARENT_RANDOM300_MANIFEST_SHA256),
            random_h5_sha256="a" * 64, random_manifest_sha256="b" * 64)
    for update in adaptation.CHECKPOINT_STEPS:
        torch.save({"format": adaptation.FORMAT, "update": update,
            "checkpoint_role": "validation_candidate", "model_config": adaptation.MODEL_CONFIG,
            "config": {"model": adaptation.MODEL_CONFIG, "training": {
                "seed": 1, "updates": 30000, "batch_size": 1024,
                "random_episode_count": random_count}},
            "model": denoiser.state_dict(), "ema_control": ema_control, **provenance},
            ckpt_dir / f"checkpoint_{update:06d}.pt")
        payload_path = ckpt_dir / f"checkpoint_{update:06d}.pt"
        if random_count in {300, 500}:
            payload = torch.load(payload_path, map_location="cpu", weights_only=False)
            payload.update(random_episode_count=random_count, expansion_contract_sha256=provenance["expansion_contract_sha256"],
                parent_dataset_manifest_sha256=provenance["parent_dataset_manifest_sha256"])
            torch.save(payload, payload_path)

    class Catalog:
        def __init__(self, path):
            self.manifest = {"split": "random_validation" if "random_validation" in str(path) else "fixed_validation",
                             "scenes": [{"target_yaw_rad": 0.0} for _ in range(50)]}
        def __len__(self): return 50
        def load(self, index): return {"index": index}
    monkeypatch.setattr(plug_orientation_scenes, "PlugSceneCatalog", Catalog)
    monkeypatch.setattr(plug_orientation_scenes, "restore_plug_scene", lambda env, scene: torch.as_tensor(env.obs()))
    base_calls, constrain_calls = [], []
    class Actor:
        def __init__(self, counter): self.counter, self.context_calls = counter, 0
        def set_episode_context(self, target): self.context_calls += 1
        def get_action(self, obs):
            self.counter.append(tuple(obs.shape))
            return torch.zeros((1, 8, 7))
    base_actor = Actor(base_calls)
    class BasePolicy:
        def __init__(self): self.agent = agent
        def get_action(self, obs): return base_actor.get_action(obs)
    monkeypatch.setattr(adaptation.base, "PlugBaseAdapter", lambda *args, **kwargs: BasePolicy())
    monkeypatch.setattr(adaptation, "sha256_file", adaptation.sha256_file)
    class Env:
        def obs(self):
            value = np.zeros(46, np.float32); value[28] = 1
            return value
        def step(self, action): return torch.as_tensor(self.obs()), 0, True, False, {"success": True}
        def close(self): pass
    def factory(_):
        actor = Actor(constrain_calls); actors.append(actor); return actor
    actors = []
    output = tmp_path / "validation"
    result = adaptation.evaluate_validation_candidates(ckpt_dir,
        "outputs/exp/20261119_184307_plug_orientation_catalogs/random_validation",
        "outputs/exp/20261119_184307_plug_orientation_catalogs/fixed_validation",
        output, policy_factory=factory, env_factory=Env)
    assert result["selected"] == 30000
    assert len(base_calls) == 100
    assert len(constrain_calls) == 600
    assert all(actor.context_calls == 100 for actor in actors)
    assert len(list(output.glob("*_base.json"))) == 2


def test_training_toy_run_updates_control_only_and_saves_ema_and_recovery_state(tmp_path):
    def fixture(name):
        h5path, manifest = tmp_path / f"{name}.h5", tmp_path / f"{name}.json"
        obs = np.zeros((17, 46), np.float32); obs[:, 28] = 1
        actions = np.zeros((16, 7), np.float32)
        with h5py.File(h5path, "w") as h5:
            traj = h5.create_group("traj_0")
            traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
            traj.create_dataset("env_states/plug_orientation_target", data=np.zeros((17, 4), np.float32))
        manifest.write_text(json.dumps({"episodes": [{"target_yaw_rad": 0.0}]}))
        return h5path, manifest
    random_h5, random_json = fixture("random")
    fixed_h5, fixed_json = fixture("fixed")
    result = adaptation.train(random_h5, random_json, fixed_h5, fixed_json,
        tmp_path / "exp", tmp_path / "ckpt", device="cpu", _updates=2,
        _batch_size=2, _test_mode=True, _random_episodes=1, _fixed_episodes=1)
    assert result["updates"] == 2
    payload = torch.load(tmp_path / "ckpt/checkpoint_000002.pt", map_location="cpu", weights_only=False)
    assert len(payload["ema_control"]) > 0
    assert all(torch.isfinite(p).all() for p in payload["ema_control"])
    assert len(payload["optimizer"]["param_groups"][0]["params"]) == len(payload["ema_control"])
    agent, _ = adaptation._load_frozen_base()
    for key, value in agent.noise_pred_net.state_dict().items():
        torch.testing.assert_close(payload["model"][f"base_net.{key}"], value, rtol=0, atol=0)


@pytest.mark.parametrize("episode_count", [200, 300])
def test_training_toy_expanded_fixture_uses_fresh_registered_dataset_lineage(tmp_path, monkeypatch, episode_count):
    from examples.baselines.constrain import plug_orientation_data as native_data
    from pathlib import Path

    obs = np.zeros((17, 46), np.float32); obs[:, 28] = 1
    actions = np.zeros((16, 7), np.float32)
    target = np.zeros((17, 4), np.float32)
    charger = np.tile([0, 0, 0, 1, 0, 0, 0], (17, 1)).astype(np.float32)
    source = tmp_path / "source.h5"; replay = tmp_path / "replay.h5"
    case_path = tmp_path / "case.json"
    case = {"target_yaw": 0.0, "expert": {"success": True, "official_success": True},
            "replay": {"success": True}}
    case_path.write_text(json.dumps(case))
    for path in (source, replay):
        with h5py.File(path, "w") as h5:
            traj = h5.create_group("traj_0")
            traj.create_dataset("obs", data=obs); traj.create_dataset("actions", data=actions)
            traj.create_dataset("env_states/plug_orientation_target", data=target)
            traj.create_dataset("env_states/actors/charger", data=charger)
            traj.create_dataset("env_states/actors/receptacle", data=charger)
    combined_h5 = tmp_path / "random200.h5"
    rows = []
    with h5py.File(combined_h5, "w") as output, h5py.File(source, "r") as source_file:
        for i in range(episode_count):
            source_file.copy("traj_0", output, name=f"traj_{i}")
            rows.append({"episode_id": i, "episode_seed": 2026123000 + i,
                "source_h5": str(source), "source_h5_sha256": native_data.sha256_file(source),
                "replay_h5": str(replay), "replay_h5_sha256": native_data.sha256_file(replay),
                "case_json": str(case_path), "case_json_sha256": native_data.sha256_file(case_path),
                "source_actions_sha256": adaptation.hashlib.sha256(actions.tobytes()).hexdigest(),
                "target_yaw_rad": 0.0,
                "lineage": {"kind": "immutable_parent_episode" if i < episode_count - 100 else "new_candidate",
                    "parent_episode_id": i if i < episode_count - 100 else None,
                    "parent_manifest_sha256": (adaptation.PARENT_RANDOM200_MANIFEST_SHA256 if episode_count == 300
                        else adaptation.PARENT_RANDOM100_MANIFEST_SHA256) if i < episode_count - 100 else None}})
    training_json = tmp_path / f"random{episode_count}.json"
    training_json.write_text(json.dumps({"episodes": rows}))
    registration_path = tmp_path / f"dataset_manifest{episode_count}.json"
    contract_number = 300 if episode_count == 300 else 200
    parent_hash = (adaptation.PARENT_RANDOM200_MANIFEST_SHA256 if episode_count == 300
                   else adaptation.PARENT_RANDOM100_MANIFEST_SHA256)
    registration_path.write_text(json.dumps({"protocol": f"plug_orientation_random{episode_count}_expansion_v1",
        "status": "complete", "accepted_count": episode_count, "effective_episode_count": episode_count,
        "parent_dataset_manifest_sha256": parent_hash,
        "expansion_contract_sha256": adaptation.sha256_file(Path(adaptation.__file__).with_name(
            f"plug_orientation_adaptation_expansion{contract_number}_contract.json")),
        "training_h5_sha256": native_data.sha256_file(combined_h5),
        "training_h5_path": str(combined_h5),
        "training_json_sha256": native_data.sha256_file(training_json),
        "training_json_path": str(training_json)}))

    fixed_h5 = tmp_path / "fixed.h5"; fixed_json = tmp_path / "fixed.json"
    with h5py.File(fixed_h5, "w") as handle:
        handle.create_group("traj_0").create_dataset("obs", data=obs)
        handle["traj_0"].create_dataset("actions", data=actions)
        handle["traj_0"].create_dataset("env_states/plug_orientation_target", data=target)
    fixed_json.write_text(json.dumps({"episodes": [{"target_yaw_rad": 0.0}]}))
    validator_calls = []
    def validate_fixture(*args):
        validator_calls.append(args)
        assert args[0] == combined_h5 and args[1] == training_json
    monkeypatch.setattr(adaptation, "_validate_random300_lineage" if episode_count == 300
                        else "_validate_random200_lineage", validate_fixture)
    result = adaptation.train(combined_h5, registration_path, fixed_h5, fixed_json,
        tmp_path / f"exp{episode_count}", tmp_path / f"ckpt{episode_count}", device="cpu", _updates=2,
        _batch_size=2, _test_mode=True, _random_episodes=episode_count, _fixed_episodes=1)
    payload = torch.load(tmp_path / f"ckpt{episode_count}/checkpoint_000002.pt", map_location="cpu", weights_only=False)
    assert result["updates"] == 2
    assert payload["random_episode_count"] == episode_count
    assert payload["config"]["training"]["random_episode_count"] == episode_count
    assert payload["parent_dataset_manifest_sha256"] == parent_hash
    run_contract = json.loads((tmp_path / f"exp{episode_count}/plug_orientation_adaptation_contract.json").read_text())
    assert run_contract["expansion_run_lineage"]["random_episode_count"] == episode_count
    if episode_count == 200:
        assert run_contract["expansion_run_lineage"]["parent_random100_forward_contract_sha256"] == adaptation.PARENT_RANDOM100_FORWARD_CONTRACT_SHA256
        assert run_contract["expansion_run_lineage"]["random200_dataset_manifest_sha256"] == native_data.sha256_file(registration_path)
    else:
        assert run_contract["expansion_run_lineage"]["parent_random200_dataset_manifest_sha256"] == adaptation.PARENT_RANDOM200_MANIFEST_SHA256
        assert run_contract["expansion_run_lineage"]["random300_dataset_manifest_sha256"] == native_data.sha256_file(registration_path)
    assert len(validator_calls) == 1


def test_random200_lineage_rejects_claimed_parent_hash_when_actual_parent_values_differ(tmp_path, monkeypatch):
    from examples.baselines.constrain import plug_orientation_data as native_data
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_manifest = parent / "dataset_manifest.json"
    parent_json = parent / native_data.DATA_JSON
    parent_h5 = parent / native_data.DATA_H5
    collection = tmp_path / "parent_collection"
    collection.mkdir()
    (collection / "preregistration.json").write_text("{}\n")
    parent_manifest.write_text(json.dumps({"collection_experiment_dir": str(collection),
        "collection_source_archive": str(collection), "preregistration_sha256": "pending"}))
    parent_json.write_text(json.dumps({"episodes": []}))
    combined_h5 = tmp_path / "combined.h5"
    with h5py.File(parent_h5, "w") as old, h5py.File(combined_h5, "w") as combined:
        for i in range(100):
            old.create_group(f"traj_{i}").create_dataset("values", data=np.asarray([i]))
            combined.create_group(f"traj_{i}").create_dataset("values", data=np.asarray([999 if i == 0 else i]))
    parent_hashes = (native_data.sha256_file(parent_manifest), native_data.sha256_file(parent_h5),
                     native_data.sha256_file(parent_json), native_data.sha256_file(collection / "preregistration.json"))
    parent_manifest.write_text(json.dumps({"collection_experiment_dir": str(collection),
        "collection_source_archive": str(collection), "preregistration_sha256": parent_hashes[3]}))
    parent_hashes = (native_data.sha256_file(parent_manifest), parent_hashes[1], parent_hashes[2], parent_hashes[3])
    monkeypatch.setattr(adaptation, "RANDOM100_DATASET", parent)
    for name, value in zip(("PARENT_RANDOM100_MANIFEST_SHA256", "PARENT_RANDOM100_H5_SHA256",
                            "PARENT_RANDOM100_JSON_SHA256"), parent_hashes[:3]):
        monkeypatch.setattr(adaptation, name, value)
    monkeypatch.setattr(adaptation, "PARENT_RANDOM100_PREREGISTRATION_SHA256", parent_hashes[3])
    registration = {"parent_dataset_manifest_sha256": parent_hashes[0],
        "parent_h5_sha256": parent_hashes[1], "parent_json_sha256": parent_hashes[2],
        "parent_preregistration_sha256": parent_hashes[3]}
    combined_json = tmp_path / "empty.json"
    combined_json.write_text(json.dumps({"episodes": []}))
    with pytest.raises(ValueError, match="differs from immutable parent"):
        adaptation._validate_random200_lineage(combined_h5, combined_json, registration, {})


def _sealed_lineage_fixture(tmp_path, monkeypatch, *, historical_parent_source=True):
    from examples.baselines.constrain import plug_orientation_data as native_data

    parent = tmp_path / "parent"
    collection = tmp_path / "parent_collection"
    archive = collection / "source_archive"
    archive.mkdir(parents=True)
    (collection / "preregistration.json").write_text(json.dumps({"source_sha256": {
        "examples/baselines/constrain/plug_orientation_adaptation.py": "historical-source-hash"}}))
    historical_source = archive / "examples/baselines/constrain/plug_orientation_adaptation.py"
    historical_source.parent.mkdir(parents=True)
    historical_source.write_bytes(b"historical collector source\n" if historical_parent_source else
                                  (adaptation.ROOT / "examples/baselines/constrain/plug_orientation_adaptation.py").read_bytes())
    (collection / "preregistration.json").write_text(json.dumps({"source_sha256": {
        "examples/baselines/constrain/plug_orientation_adaptation.py": native_data.sha256_file(historical_source)}}))

    parent.mkdir()
    parent_h5 = parent / native_data.DATA_H5
    parent_json = parent / native_data.DATA_JSON
    raw = tmp_path / "parent_raw.h5"
    case = tmp_path / "parent_case.json"
    with h5py.File(raw, "w") as handle:
        handle.create_group("traj_0").create_dataset("values", data=np.asarray([7]))
    case.write_text(json.dumps({"target_yaw": 0.0}))
    source_hash, case_hash = native_data.sha256_file(raw), native_data.sha256_file(case)
    parent_rows = [{"episode_seed": 1000 + i, "target_yaw_rad": 0.0,
        "source_h5": str(raw), "source_h5_sha256": source_hash,
        "replay_h5": str(raw), "replay_h5_sha256": source_hash,
        "case_json": str(case), "case_json_sha256": case_hash} for i in range(100)]
    with h5py.File(parent_h5, "w") as handle:
        for i in range(100):
            handle.create_group(f"traj_{i}").create_dataset("values", data=np.asarray([7]))
    parent_json.write_text(json.dumps({"episodes": parent_rows}))
    parent_prereg = collection / "preregistration.json"
    parent_manifest_path = parent / "dataset_manifest.json"
    parent_manifest_path.write_text(json.dumps({"collection_experiment_dir": str(collection),
        "collection_source_archive": str(archive), "preregistration_sha256": native_data.sha256_file(parent_prereg)}))
    hashes = [native_data.sha256_file(path) for path in
              (parent_manifest_path, parent_h5, parent_json, parent_prereg)]
    for name, digest in zip(("PARENT_RANDOM100_MANIFEST_SHA256", "PARENT_RANDOM100_H5_SHA256",
                             "PARENT_RANDOM100_JSON_SHA256", "PARENT_RANDOM100_PREREGISTRATION_SHA256"), hashes):
        monkeypatch.setattr(adaptation, name, digest)
    monkeypatch.setattr(adaptation, "RANDOM100_DATASET", parent)

    dataset = tmp_path / "expanded"
    dataset.mkdir()
    exp = tmp_path / "exp"
    exp.mkdir()
    source_archive = exp / "source_archive"
    rel = "examples/baselines/constrain/plug_orientation_adaptation.py"
    archived_current = source_archive / rel
    archived_current.parent.mkdir(parents=True)
    archived_current.write_bytes((adaptation.ROOT / rel).read_bytes())
    source_hashes = {rel: native_data.sha256_file(archived_current)}
    expected_yaws = np.random.default_rng(1704).uniform(-np.pi / 8, np.pi / 8, 300)[150:]
    plan = [{"seed": 2026123150 + i, "target_yaw_rad": float(yaw)}
            for i, yaw in enumerate(expected_yaws)]
    combined_h5 = dataset / native_data.DATA_H5
    training_json = dataset / native_data.DATA_JSON
    rows = []
    attempts = []
    with h5py.File(combined_h5, "w") as output, h5py.File(parent_h5, "r") as old:
        for i in range(100):
            old.copy(f"traj_{i}", output, name=f"traj_{i}")
            rows.append({**parent_rows[i], "lineage": {"kind": "immutable_parent_episode", "parent_episode_id": i}})
        for j, yaw in enumerate(expected_yaws[:100]):
            source = dataset / f"source_{j}.h5"
            replay = dataset / f"replay_{j}.h5"
            case_path = dataset / f"case_{j}.json"
            with h5py.File(source, "w") as handle:
                handle.create_group("traj_0").create_dataset("values", data=np.asarray([j]))
            replay.write_bytes(source.read_bytes())
            case_path.write_text(json.dumps({"target_yaw": float(yaw)}))
            sh, rh, ch = (native_data.sha256_file(p) for p in (source, replay, case_path))
            output.create_group(f"traj_{100+j}").create_dataset("values", data=np.asarray([j]))
            candidate = plan[j]
            attempt = {**candidate, "candidate_index": j + 1, "accepted": True,
                "source_h5": str(source), "source_h5_sha256": sh,
                "replay_h5": str(replay), "replay_h5_sha256": rh,
                "case_json": str(case_path), "case_json_sha256": ch}
            attempts.append(attempt)
            # Metadata is read back from the native float32 target state.
            rows.append({"episode_seed": candidate["seed"], "target_yaw_rad": float(np.float32(yaw)),
                "source_h5": str(source), "source_h5_sha256": sh,
                "replay_h5": str(replay), "replay_h5_sha256": rh,
                "case_json": str(case_path), "case_json_sha256": ch,
                "lineage": {"kind": "new_candidate"}})
    training_json.write_text(json.dumps({"episodes": rows}))
    prereg = exp / "preregistration.json"
    prereg.write_text(json.dumps({"candidate_plan": plan, "source_sha256": source_hashes}))
    journal = dataset / "attempts.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in attempts))
    registration_path = dataset / "dataset_manifest.json"
    registration = {"protocol": "plug_orientation_random200_expansion_v1", "status": "complete",
        "parent_dataset_manifest_sha256": hashes[0], "parent_h5_sha256": hashes[1],
        "parent_json_sha256": hashes[2], "parent_preregistration_sha256": hashes[3],
        "collection_experiment_dir": str(exp), "preregistration_sha256": native_data.sha256_file(prereg),
        "attempts": attempts, "attempts_sha256": native_data.sha256_file(journal)}
    registration_path.write_text(json.dumps(registration))
    monkeypatch.setattr(native_data, "_native_episode_ok", lambda *args, **kwargs: (True, {}))
    monkeypatch.setattr(native_data, "_pair_native_match", lambda *args, **kwargs: True)
    return combined_h5, training_json, registration, registration_path


def test_actual_random200_lineage_accepts_archived_parent_source_different_from_checkout(tmp_path, monkeypatch):
    combined_h5, training_json, registration, registration_path = _sealed_lineage_fixture(tmp_path, monkeypatch)
    adaptation._validate_random200_lineage(combined_h5, training_json, registration,
                                            json.loads(training_json.read_text()))


def test_actual_random200_lineage_rejects_material_float32_yaw_tampering(tmp_path, monkeypatch):
    combined_h5, training_json, registration, registration_path = _sealed_lineage_fixture(tmp_path, monkeypatch)
    payload = json.loads(training_json.read_text())
    payload["episodes"][100]["target_yaw_rad"] = float(np.float32(
        payload["episodes"][100]["target_yaw_rad"] + 1e-4))
    training_json.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ordered first-eligible"):
        adaptation._validate_random200_lineage(combined_h5, training_json, registration, payload)


def test_actual_random200_lineage_rejects_journal_only_tampering(tmp_path, monkeypatch):
    combined_h5, training_json, registration, registration_path = _sealed_lineage_fixture(
        tmp_path, monkeypatch, historical_parent_source=False)
    journal = registration_path.parent / "attempts.jsonl"
    journal.write_text(journal.read_text().replace('"accepted": true', '"accepted": false', 1))
    with pytest.raises(ValueError, match="journal hash mismatch"):
        adaptation._validate_random200_lineage(combined_h5, training_json, registration,
                                                json.loads(training_json.read_text()))


def test_actual_random200_lineage_rejects_unselected_candidate_plan_seed_tampering(tmp_path, monkeypatch):
    combined_h5, training_json, registration, registration_path = _sealed_lineage_fixture(
        tmp_path, monkeypatch, historical_parent_source=False)
    prereg = Path(registration["collection_experiment_dir"]) / "preregistration.json"
    payload = json.loads(prereg.read_text())
    payload["candidate_plan"][149]["seed"] += 1
    prereg.write_text(json.dumps(payload))
    registration["preregistration_sha256"] = adaptation.sha256_file(prereg)
    registration_path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="candidate plan differs"):
        adaptation._validate_random200_lineage(combined_h5, training_json, registration,
                                                json.loads(training_json.read_text()))
