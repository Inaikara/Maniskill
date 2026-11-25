import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.baselines.constrain import peg_orientation_entry_perturb_eval as scorer
from examples.baselines.constrain import peg_orientation_near_hole_eval as near


def _bank_case(profile="random", condition="aligned", index=0):
    prefix = np.zeros((2, 7), dtype=np.float32)
    init = np.ones((1, 7), dtype=np.float32)
    full = np.concatenate((prefix, init))
    return {"profile": profile, "condition": condition, "catalog_index": index,
        "result": {"admitted": True, "admission_mode": "pose_window", "native_grasped": True,
            "official_inserted": False, "actual_entry_clearance_m": .020,
            "executed_prefix_actions": prefix.tolist(),
            "executed_initializer_actions": init.tolist(), "controls": 3,
            "full_action_sha256": hashlib.sha256(full.tobytes()).hexdigest()}}


def test_bank_parser_accepts_all_admitted_pose_window_rows_and_rejects_bad_full_action_hash(tmp_path):
    cases = [_bank_case(profile=p, condition=c, index=i) for p in ("random", "fixed")
             for c in scorer.CONDITIONS for i in range(6)]
    path = tmp_path / "journal.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in cases))
    assert len(scorer.bank_rows(path)) == 60
    cases[0]["result"]["full_action_sha256"] = "bad"
    path.write_text("".join(json.dumps(row) + "\n" for row in cases))
    with pytest.raises(ValueError, match="full replay action hash"):
        scorer.bank_rows(path)


def test_native_caps_cover_the_frozen_360_arm_maximum_and_env_horizon():
    assert scorer.MAX_NATIVE_CONTROLS == 360 * (420 + 200)
    assert scorer.ENV_TIME_LIMIT > 420 + 200
    assert scorer.STAGE_SECONDS == 10800


class _FakePose:
    def __init__(self, point):
        self.p = torch.tensor([point], dtype=torch.float32)

    def inv(self):
        return self

    def __mul__(self, other):
        return other


class _FakeController:
    def get_state(self):
        return {}


class _FakeEnv:
    def __init__(self, profile, envs):
        self.profile, self.envs = profile, envs
        self.unwrapped = self
        self.agent = SimpleNamespace(controller=_FakeController())
        self.control_mode = "pd_ee_delta_pose"
        self.peg_half_sizes = torch.tensor([[.085, .01, .01]])
        self.box_hole_pose = _FakePose([0., 0., 0.])
        self.peg_head_pose = _FakePose([-.105, 0., 0.])
        self.peg = SimpleNamespace(pose=object())
        self.physical_state = {"qpos": [0.]}
        self.grasped = True
        self.count = 0
        self.history = []
        self.handoff_history = None
        envs.append(self)

    def reset_to_fixture(self, scene):
        self.scene = scene
        self.count = 0
        self.physical_state = scene["physical_state"]
        self.history = [np.zeros(39, dtype=np.float32)]
        return self.history[0].copy()

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        self.count += 1
        obs = np.full(39, self.count, dtype=np.float32)
        self.history.append(obs)
        if self.count == 3:
            self.handoff_history = np.asarray(self.history[-2:]).copy()
        return torch.tensor(obs), 0., False, False, {"success": self.count == 4}

    def has_peg_inserted(self):
        return (torch.tensor([False]),)

    def close(self):
        pass


class _StrictAdapter:
    def __init__(self, name, envs, *, fail=False):
        self.name, self.envs, self.fail = name, envs, fail
        self.calls = 0
        self.contexts = []
        self.task = None
        self.metadata = {"name": name}

    def set_episode_context(self, scene_index, metadata):
        self.contexts.append((scene_index, metadata))

    def get_action(self, observations):
        assert np.asarray(observations).shape == (1, 2, 39)
        pair = self.envs[-2:]
        assert len(pair) == 2
        assert pair[1].count == 3
        assert pair[0].count == (3 if self.name == "base" else 4)
        np.testing.assert_array_equal(pair[0].handoff_history, pair[1].handoff_history)
        assert all(env.control_mode == "pd_ee_delta_pose" for env in pair)
        if self.name == "constrain":
            assert self.task is pair[1]
        assert all(env.agent.controller.get_state() == {} for env in pair)
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected Constrain sampler failure")
        actions = np.zeros((1, 8, 7), dtype=np.float32)
        assert actions.shape == (1, 8, 7)
        return actions


class _FakeCatalog:
    def __init__(self, profile, fixtures):
        self.profile, self.fixtures = profile, fixtures
        self.manifest = {"scenes": [{"target_yaw_rad": .2, "seed": 7000 + i} for i in range(6)]}

    def load(self, index):
        return self.fixtures[(self.profile, index)]


def _fixture_bank():
    rows, source_actions, scenes = [], {}, {}
    for profile in ("random", "fixed"):
        for condition in scorer.CONDITIONS:
            for index in range(6):
                prefix = np.full((2, 7), .01 if profile == "random" else .02, dtype=np.float32)
                initializer = np.full((1, 7), .001 * (scorer.CONDITIONS.index(condition) + 1), dtype=np.float32)
                actions = np.concatenate((prefix, initializer))
                physical = {"qpos": [float(index)]}
                # Real public admission checker is exercised on valid outside-entry samples.
                window = [{"control_index": i, "monotonic_seconds": i * .025,
                    "simulation_elapsed_seconds": i * .025, "head_error_xyz_m": [0., 0., 0.],
                    "entry_clearance_m": .020, "yaw_error_rad": 0., "grasped": True,
                    "official_inserted": False, "head_hole_xyz_m": [-.105, 0., 0.],
                    "peg_center_world_m": [0., 0., 0.], "peg_quaternion_wxyz": [1., 0., 0., 0.],
                    "arm_qpos_rad": [0.] * 7} for i in range(11)]
                last_two = np.stack((np.full(39, 2., dtype=np.float32), np.full(39, 3., dtype=np.float32)))
                result = {"admitted": True, "admission_mode": "pose_window", "native_grasped": True,
                    "official_inserted": False, "actual_entry_clearance_m": .020, "controls": 3,
                    "source_prefix_controls": 2, "executed_prefix_actions": prefix.tolist(),
                    "executed_initializer_actions": initializer.tolist(), "controller_mode": "pd_ee_delta_pose",
                    "physical_state": physical, "last_two_observations": last_two.tolist(),
                    "last_pose_window_candidate": window, "control_timestep_s": .025,
                    "full_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest()}
                rows.append({"profile": profile, "condition": condition, "catalog_index": index,
                    "source_scene_id": f"peg-validation_{profile}-{index:03}", "result": result})
                source_actions[(profile, index)] = prefix
                scenes[(profile, index)] = {"physical_state": physical}
    return rows, source_actions, scenes


def _install_fake_physics(monkeypatch, rows, source_actions, scenes, tmp_path, *, fail_constrain=False, mismatch=False):
    envs = []
    monkeypatch.setattr(near, "restore_peg_scene", lambda env, scene: env.reset_to_fixture(scene))
    monkeypatch.setattr(near, "capture_physical_state", lambda task: task.physical_state)
    monkeypatch.setattr(near, "native_grasped", lambda task: task.grasped)
    monkeypatch.setattr(near, "_relative_yaw", lambda task: 0.)
    adapters = {name: _StrictAdapter(name, envs, fail=(fail_constrain and name == "constrain"))
                for name in ("base", "constrain")}
    def env_factory(profile):
        return _FakeEnv(profile, envs)
    def catalog_factory(profile):
        return _FakeCatalog(profile, scenes)
    def action_loader(profile, episode):
        return source_actions[(profile, episode)]
    for profile in {row["profile"] for row in rows}:
        source_dir = tmp_path / "source" / profile
        source_dir.mkdir(parents=True)
        (source_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text(json.dumps({
            "episodes": [{"episode_id": i, "episode_seed": 7000 + i} for i in range(6)]}))
        monkeypatch.setitem(near.DATA_DIRS, profile, source_dir)
    if mismatch:
        rows[0]["result"]["physical_state"] = {"qpos": [999.]}
    return envs, adapters, env_factory, catalog_factory, action_loader


def _precreate_output(path, *, extra=()):
    path.mkdir()
    (path / "attempt_001.sh").write_text("#!/bin/bash\n# wrapper fixture\n")
    (path / "attempt_001.stdout.log").write_text("")
    (path / "attempt_001.stderr.log").write_text("")
    for name in extra:
        (path / name).write_text("{}");


def test_run_entrypoint_replays_and_scores_all_fixture_pairs_through_real_suffix(tmp_path, monkeypatch):
    rows, sources, scenes = _fixture_bank()
    parsed_path = tmp_path / "bank.jsonl"
    parsed_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    parsed = scorer.bank_rows(parsed_path)
    selected = [r for r in parsed if r["catalog_index"] == 0]
    envs, adapters, env_factory, catalog_factory, action_loader = _install_fake_physics(
        monkeypatch, selected, sources, scenes, tmp_path)
    monkeypatch.setattr(scorer, "_seed", lambda seed: None)
    output = tmp_path / "attempt"
    _precreate_output(output)
    summary = scorer.run_comparison(output, adapters=adapters, env_factory=env_factory,
        catalog_factory=catalog_factory, source_action_loader=action_loader, bank=selected)
    assert summary["pairs_completed"] == 30
    assert summary["native_controls_including_replay"] == 240
    assert sum(adapter.calls for adapter in adapters.values()) == 60
    assert all(adapter.calls == 30 for adapter in adapters.values())
    assert len(envs) == 60
    assert all(adapter.contexts for adapter in adapters.values())
    assert all(summary["per_condition"][p][c]["draws_completed"] == 3
               for p in ("random", "fixed") for c in scorer.CONDITIONS)
    assert all(summary["per_condition"][p][c]["unique_origins"] == 1
               for p in ("random", "fixed") for c in scorer.CONDITIONS)
    assert all(summary["handoff_geometry"][f"{p}/{c}"]["rows"] == 3
               for p in ("random", "fixed") for c in scorer.CONDITIONS)
    assert summary["setup"]["wrapper_sha256"] == hashlib.sha256((output / "attempt_001.sh").read_bytes()).hexdigest()
    assert summary["setup"]["runner_sha256"] == summary["setup"]["archived_runner_sha256"]
    journal = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    complete = [row for row in journal if row["status"] == "complete"]
    assert len(complete) == 30
    assert {(row["profile"], row["condition"]) for row in complete} == {
        (profile, condition) for profile in ("random", "fixed") for condition in scorer.CONDITIONS}
    assert all(sorted(row["seed"] for row in complete if row["profile"] == profile and
                      row["condition"] == condition) == [1, 1001, 2001]
               for profile in ("random", "fixed") for condition in scorer.CONDITIONS)
    assert all("handoff" in row and "bank_actual_entry_clearance_m" in row["handoff"]
               and row["handoff"]["replayed_prefix_controls"] == 3
               for row in complete)
    assert all(len(adapter.contexts) == 30 for adapter in adapters.values())


@pytest.mark.parametrize("profile,catalog_index", [("random", 2), ("fixed", 5)])
def test_runner_resolves_source_episode_from_catalog_seed_not_scene_id_suffix(
        tmp_path, monkeypatch, profile, catalog_index):
    rows, sources, scenes = _fixture_bank()
    selected = [r for r in rows if r["profile"] == profile and
                r["condition"] == scorer.CONDITIONS[0] and r["catalog_index"] == catalog_index]
    envs, adapters, env_factory, catalog_factory, _ = _install_fake_physics(
        monkeypatch, selected, sources, scenes, tmp_path)
    seed = 7101
    catalog = catalog_factory(profile)
    catalog.manifest["scenes"][catalog_index]["seed"] = seed
    source_dir = tmp_path / profile
    source_dir.mkdir()
    metadata_path = source_dir / "trajectory.state.pd_ee_delta_pose.physx_cpu.json"
    metadata_path.write_text(json.dumps({"episodes": [
        {"episode_id": 0, "episode_seed": 7000},
        {"episode_id": 1, "episode_seed": seed},
        {"episode_id": 2, "episode_seed": 7002},
    ]}))
    monkeypatch.setitem(near.DATA_DIRS, profile, source_dir)
    catalog_factory = lambda requested: catalog if requested == profile else _FakeCatalog(requested, scenes)
    requested = []

    def strict_loader(requested_profile, episode_id):
        requested.append((requested_profile, episode_id))
        assert episode_id == 1
        return sources[(profile, catalog_index)]

    monkeypatch.setattr(scorer, "_seed", lambda seed: None)
    output = tmp_path / "attempt"
    _precreate_output(output)
    summary = scorer.run_comparison(output, adapters=adapters, env_factory=env_factory,
        catalog_factory=catalog_factory, source_action_loader=strict_loader, bank=selected)
    assert summary["pairs_completed"] == 3
    assert requested == [(profile, 1)] * 3


def test_run_entrypoint_journals_completed_base_and_physical_count_on_cn_error(tmp_path, monkeypatch):
    rows, sources, scenes = _fixture_bank()
    selected = [r for r in rows if r["profile"] == "random" and r["condition"] == scorer.CONDITIONS[0] and r["catalog_index"] == 0]
    envs, adapters, env_factory, catalog_factory, action_loader = _install_fake_physics(
        monkeypatch, selected, sources, scenes, tmp_path, fail_constrain=True)
    monkeypatch.setattr(scorer, "_seed", lambda seed: None)
    output = tmp_path / "attempt"
    _precreate_output(output)
    with pytest.raises(RuntimeError, match="injected Constrain sampler failure"):
        scorer.run_comparison(output, adapters=adapters, env_factory=env_factory,
            catalog_factory=catalog_factory, source_action_loader=action_loader, bank=selected)
    rows_journal = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    failed = rows_journal[-1]
    assert failed["status"] == "failed"
    assert failed["failure"]["message"] == "injected Constrain sampler failure"
    assert failed["arms"]["base"]["native_success"] is True
    assert failed["physical_controls_at_failure"] == 7
    assert sum(env.count for env in envs) == 7
    assert adapters["base"].calls == adapters["constrain"].calls == 1
    assert json.loads((output / "failure.json").read_text())["physical_controls"] == 7


def test_run_entrypoint_replay_mismatch_stops_before_any_policy_query(tmp_path, monkeypatch):
    rows, sources, scenes = _fixture_bank()
    selected = [r for r in rows if r["profile"] == "random" and r["condition"] == scorer.CONDITIONS[0] and r["catalog_index"] == 0]
    envs, adapters, env_factory, catalog_factory, action_loader = _install_fake_physics(
        monkeypatch, selected, sources, scenes, tmp_path, mismatch=True)
    output = tmp_path / "attempt"
    _precreate_output(output)
    with pytest.raises(ValueError, match="replayed physical state differs"):
        scorer.run_comparison(output, adapters=adapters, env_factory=env_factory,
            catalog_factory=catalog_factory, source_action_loader=action_loader, bank=selected)
    assert sum(adapter.calls for adapter in adapters.values()) == 0
    assert sum(env.count for env in envs) == 3


def test_output_setup_accepts_own_wrapper_and_rejects_preregistration_collision(tmp_path):
    valid = tmp_path / "valid"
    _precreate_output(valid)
    setup = scorer._prepare_output(valid)
    assert setup["wrapper_sha256"] == hashlib.sha256((valid / "attempt_001.sh").read_bytes()).hexdigest()
    assert setup["runner_sha256"] == setup["archived_runner_sha256"]
    invalid = tmp_path / "invalid"
    _precreate_output(invalid, extra=("preregistration.json",))
    with pytest.raises(FileExistsError, match="preregistration/results"):
        scorer._prepare_output(invalid)
