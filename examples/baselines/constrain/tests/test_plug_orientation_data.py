import hashlib
import json
import time

import h5py
import numpy as np
import torch

from examples.baselines.constrain import plug_orientation_data as data
from examples.baselines.constrain import plug_orientation_base_dp as base
from examples.baselines.constrain import plug_orientation_scenes as scenes


def _episode(path, seed, *, valid=True):
    path.mkdir(parents=True)
    for fname in ("expert.h5", "replay/replay.h5"):
        p = path / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(p, "w") as h5:
            traj = h5.create_group("traj_0")
            traj.create_dataset("actions", data=np.zeros((1, 7), np.float32))
            traj.create_dataset("obs", data=np.zeros((2, 46), np.float32))
            states = traj.create_group("env_states")
            actors = states.create_group("actors")
            charger = np.zeros((2, 13), np.float32); charger[:, 3] = 1
            if not valid: charger[-1, 0] = 0.01
            actors.create_dataset("charger", data=charger)
            receptacle = np.zeros((2, 13), np.float32); receptacle[:, 3] = 1
            actors.create_dataset("receptacle", data=receptacle)
            states.create_dataset("plug_orientation_target", data=np.zeros((2, 4), np.float32))
            states["plug_orientation_target"][:, 0] = 0
    case = {"target_yaw": 0.0, "expert": {"success": valid, "official_success": valid},
            "replay": {"success": valid}, "expert_controls": 1, "replay_controls": 1}
    (path / "case.json").write_text(json.dumps(case))
    return case


def test_ordered_collection_journals_failures_and_consolidates_first_valid(tmp_path):
    calls = []

    def fake(seed, path, target_yaw, deadline=None):
        calls.append(seed)
        if seed == 2026121000:
            path.mkdir(parents=True)
            (path / "partial_failure.json").write_text(json.dumps({"expert_controls": 5}))
            raise RuntimeError("MPLib screw planning failed")
        return _episode(path, seed)

    ds, exp = tmp_path / "data", tmp_path / "exp"
    manifest = data.collect_fixed_dataset(ds, exp, run_case_fn=fake,
                                           expected_count=2, max_attempts=3, _test_mode=True)
    combined = json.loads((ds / data.DATA_JSON).read_text())
    assert [row["episode_seed"] for row in combined["episodes"]] == [2026121001, 2026121002]
    assert hashlib.sha256((ds / data.DATA_H5).read_bytes()).hexdigest()
    assert calls == [2026121000, 2026121001, 2026121002]
    assert manifest["candidate_attempt_count"] == 3
    assert manifest["expert_controls"] == 7
    assert manifest["failure_classes"]["known_planning_failure"] == 1
    assert len((ds / "attempts.jsonl").read_text().splitlines()) == 3


def test_real_600_cap_failure_retains_partial_controls_and_continues(tmp_path):
    calls = []
    def fake(seed, path, target_yaw, deadline=None):
        calls.append(seed)
        if len(calls) == 1:
            _episode(path, seed)
            with h5py.File(path / "expert.h5", "a") as h5:
                del h5["traj_0/actions"]
                del h5["traj_0/obs"]
                h5["traj_0"].create_dataset("actions", data=np.zeros((600, 7), np.float32))
                h5["traj_0"].create_dataset("obs", data=np.zeros((601, 46), np.float32))
            (path / "partial_failure.json").write_text(json.dumps({"expert_controls": 600}))
            raise RuntimeError("native teacher exceeded 600-control episode cap")
        return _episode(path, seed)
    ds, exp = tmp_path / "data", tmp_path / "exp"
    manifest = data.collect_fixed_dataset(ds, exp, run_case_fn=fake,
        expected_count=1, max_attempts=2, _test_mode=True)
    attempts = [json.loads(line) for line in (ds / "attempts.jsonl").read_text().splitlines()]
    assert calls == [2026121000, 2026121001]
    assert attempts[0]["failure_class"] == "known_planning_failure"
    assert attempts[0]["expert_controls"] == 600 and attempts[0]["replay_controls"] == 0
    with h5py.File(attempts[0]["source_h5"], "r") as h5:
        assert h5["traj_0/actions"].shape == (600, 7)
        assert h5["traj_0/obs"].shape == (601, 46)
    assert manifest["status"] == "complete"


def test_collection_deadline_interrupts_slow_callback_without_completion(tmp_path):
    def slow(seed, path, target_yaw, deadline=None):
        time.sleep(0.6)
        return _episode(path, seed)
    contract = json.loads(data.CONTRACT_PATH.read_text())
    contract["data_collection"]["wall_clock_cap_seconds"] = 0.5
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    ds, exp = tmp_path / "data", tmp_path / "exp"
    import pytest
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        data.collect_fixed_dataset(ds, exp, contract_path=contract_path,
            run_case_fn=slow, expected_count=1, max_attempts=1, _test_mode=True)
    assert time.monotonic() - started < 0.6
    assert not (ds / data.DATA_H5).exists()
    assert json.loads((ds / "attempts.jsonl").read_text().splitlines()[0])["status"] == "failed"


def test_fixed_catalog_manifest_and_split_ranges_are_disjoint():
    assert scenes.SPLITS["fixed_validation"] == tuple(range(2026124100, 2026124150))
    assert scenes.SPLITS["random_final_test"] == tuple(range(2026125000, 2026125100))
    assert scenes.SPLITS["fixed_base_test"] == tuple(range(2026125100, 2026125200))
    all_eval = [s for values in scenes.SPLITS.values() for s in values]
    assert len(all_eval) == len(set(all_eval))
    assert not set(all_eval) & set(scenes.RESERVED_ADAPTATION)


def test_native_admission_rejects_zero_or_nonfinite_episodes(tmp_path):
    zero = tmp_path / "zero"
    case = _episode(zero, 2026121000)
    with h5py.File(zero / "expert.h5", "a") as h5:
        del h5["traj_0/actions"]
        h5["traj_0"].create_dataset("actions", data=np.zeros((0, 7), np.float32))
    assert data._native_episode_ok(zero / "expert.h5", case) == (False, "invalid_actions")
    nonfinite = tmp_path / "nonfinite"
    case = _episode(nonfinite, 2026121001)
    with h5py.File(nonfinite / "expert.h5", "a") as h5:
        h5["traj_0/obs"][1, 3] = np.nan
    assert data._native_episode_ok(nonfinite / "expert.h5", case) == (False, "invalid_t_plus_one_or_nonfinite")
    assert base._fixed_candidate_is_allowed(2026121000, 0.0)
    assert not base._fixed_candidate_is_allowed(2026113001, 0.0)
    assert not base._fixed_candidate_is_allowed(2026121000, 0.1)


def test_public_catalog_builder_freezes_and_restores_initial_scene_only(tmp_path):
    class Pose:
        def __init__(self):
            self.raw_pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])
            self.p = self.raw_pose[:, :3]
    class Task:
        device = "cpu"
        def __init__(self): self.charger = type("Actor", (), {"pose": Pose()})(); self.goal_pose = Pose()
        def get_state_dict(self): return {"state": torch.zeros((1, 4))}
        def set_state_dict(self, state): assert state["state"].shape == (1, 4)
        def get_obs(self): return torch.zeros((1, 46))
    class Env:
        def __init__(self): self.unwrapped = Task(); self.closed = False
        def reset(self, seed, options): return torch.zeros((1, 46)), {}
        def close(self): self.closed = True
    manifest = scenes.build_scene_catalog(tmp_path / "catalog", "fixed_validation", env_factory=Env)
    catalog = scenes.PlugSceneCatalog(tmp_path / "catalog")
    assert manifest["scene_count"] == len(catalog) == 50
    obs = scenes.restore_plug_scene(Env(), catalog.load(0))
    assert obs.shape == (1, 46)
    random_manifest = scenes.build_scene_catalog(tmp_path / "random_catalog", "random_validation", env_factory=Env)
    first = random_manifest["scenes"][0]
    expected = np.random.default_rng(first["seed"] + 1703).uniform(-np.pi / 8, np.pi / 8)
    assert first["target_yaw_rad"] == expected
