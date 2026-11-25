import numpy as np
import pytest
import hashlib
import json
import time
from types import SimpleNamespace
import torch

from examples.baselines.constrain import peg_orientation_entry_scaling_eval as scaling
from examples.baselines.constrain import peg_orientation_entry_continuation_eval as continuation
from examples.baselines.constrain import peg_orientation_near_hole_eval as near


def test_scorer_requires_six_distinct_finite_native_action_contracts():
    assert len(scaling.ARMS) == len(set(scaling.ARMS)) == 6
    assert scaling.PRIMARY == ("base", "cn100_30k", "cn200_30k", "cn500_30k")
    assert scaling.SECONDARY == ("base", "cn100_15k", "cn200_30k", "cn500_25k")
    actions = np.zeros((1, 8, 7), dtype=np.float32)
    assert scaling.validate_arm_result("fake", actions).shape == (1, 8, 7)
    with pytest.raises(ValueError, match=r"finite \(1, 8, 7\)"):
        scaling.validate_arm_result("fake", np.zeros((1, 8, 7, 1)))
    actions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match=r"finite \(1, 8, 7\)"):
        scaling.validate_arm_result("fake", actions)


def test_primary_secondary_curves_share_one_base_and_cn200_arm_per_draw():
    assert scaling.PRIMARY.count("base") == scaling.SECONDARY.count("base") == 1
    assert scaling.PRIMARY.count("cn200_30k") == scaling.SECONDARY.count("cn200_30k") == 1
    assert scaling.MAX_NATIVE_CONTROLS == 1080 * (420 + 200)
    assert scaling.ENV_TIME_LIMIT >= 420 + 200


@pytest.mark.parametrize(
    ("budget", "elapsed_before_call", "control_limit", "stage_limit"),
    [([223200, 0], 0., 669600, 32400), ([0, 0], 10801., 669600, 32400)],
)
def test_scaling_prepare_uses_its_registered_replay_limits(
        tmp_path, monkeypatch, budget, elapsed_before_call, control_limit, stage_limit):
    rows, _, _, _, env_factory, _, _, _ = _integration_fixtures(tmp_path, monkeypatch)
    row = rows[0]
    env = env_factory(row["profile"])
    initial_controls = budget[0]
    started = time.monotonic() - elapsed_before_call
    scaling.paired._prepare(env, {"physical_state": {"qpos": [0.]}}, row, budget, started,
        max_native_controls=control_limit, stage_seconds=stage_limit)
    assert budget[0] == initial_controls + 3
    assert env.count == 3


@pytest.mark.parametrize(("budget", "elapsed_before_call"), [([223200, 0], 0.), ([0, 0], 10801.)])
def test_paired_prepare_defaults_still_enforce_two_arm_limits(tmp_path, monkeypatch, budget, elapsed_before_call):
    rows, _, _, _, env_factory, _, _, _ = _integration_fixtures(tmp_path, monkeypatch)
    row = rows[0]
    env = env_factory(row["profile"])
    started = time.monotonic() - elapsed_before_call
    with pytest.raises(TimeoutError, match="entry evaluation native budget"):
        scaling.paired._prepare(env, {"physical_state": {"qpos": [0.]}}, row, budget, started)
    assert env.count == 0


@pytest.mark.parametrize(("budget", "elapsed_before_call"), [([669600, 0], 0.), ([0, 0], 32401.)])
def test_scaling_prepare_enforces_its_registered_limits(tmp_path, monkeypatch, budget, elapsed_before_call):
    rows, _, _, _, env_factory, _, _, _ = _integration_fixtures(tmp_path, monkeypatch)
    row = rows[0]
    env = env_factory(row["profile"])
    started = time.monotonic() - elapsed_before_call
    with pytest.raises(TimeoutError, match="entry evaluation native budget"):
        scaling.paired._prepare(env, {"physical_state": {"qpos": [0.]}}, row, budget, started,
            max_native_controls=scaling.MAX_NATIVE_CONTROLS, stage_seconds=scaling.STAGE_SECONDS)
    assert env.count == 0


def test_aggregate_keeps_repeated_draw_denominator_and_paired_transitions():
    row = {"profile": "random", "condition": scaling.CONDITIONS[0], "catalog_index": 2, "replicate": 0,
           "arms": {name: {"native_success": name in ("base", "cn100_30k", "cn200_30k", "cn500_30k")}
                    for name in scaling.ARMS}}
    result = scaling._aggregate([row])
    cell = result["random"][scaling.CONDITIONS[0]]
    assert cell["draws"] == 1 and cell["unique_origins"] == 1
    assert cell["successes"]["cn100_30k"] == 1
    assert cell["paired_transitions"]["base->cn100_30k"] == {
        "improved": 0, "regressed": 0, "both": 1, "neither": 0}
    assert cell["paired_transitions"]["cn200_30k->cn500_30k"]["both"] == 1
    assert cell["paired_transitions"]["base->cn500_30k"]["both"] == 1
    assert cell["monotonicity"]["primary"]["patterns"] == {"111": 1}


class _Pose:
    def __init__(self, point): self.p = torch.tensor([point], dtype=torch.float32)
    def inv(self): return self
    def __mul__(self, other): return other


class _Env:
    def __init__(self, created):
        self.unwrapped = self
        self.agent = SimpleNamespace(controller=SimpleNamespace(get_state=lambda: {}))
        self.control_mode = "pd_ee_delta_pose"
        self.peg_half_sizes = torch.tensor([[.085, .01, .01]])
        self.box_hole_pose, self.peg_head_pose = _Pose([0., 0., 0.]), _Pose([-.105, 0., 0.])
        self.peg = SimpleNamespace(pose=object())
        self.grasped, self.count = True, 0
        self.physical_state = {"qpos": [0.]}
        created.append(self)
    def reset_to_fixture(self, scene):
        self.count, self.physical_state = 0, scene["physical_state"]
        self.history = [np.zeros(39, dtype=np.float32)]
        return self.history[0]
    def step(self, action):
        self.count += 1
        obs = np.full(39, self.count, dtype=np.float32)
        self.history.append(obs)
        return torch.tensor(obs), 0., False, False, {"success": self.count >= 4}
    def has_peg_inserted(self): return (torch.tensor([False]),)
    def close(self): pass


class _Catalog:
    def __init__(self, profile, fixtures):
        self.profile, self.fixtures = profile, fixtures
        self.manifest = {"scenes": [{"target_yaw_rad": .2, "seed": 7000 + i} for i in range(6)]}
    def load(self, index): return self.fixtures[(self.profile, index)]


class _Adapter:
    def __init__(self, name, created, query_log, fail=False, roster_size=6):
        self.name, self.created, self.query_log, self.fail = name, created, query_log, fail
        self.roster_size = roster_size
        self.task, self.calls, self.contexts = None, 0, []
    def set_episode_context(self, index, metadata): self.contexts.append((index, metadata))
    def get_action(self, obs):
        assert np.asarray(obs).shape == (1, 2, 39)
        assert self.task is not None if self.name != "base" else True
        assert self.task.count == 3 if self.name != "base" else True
        group_start = ((len(self.created) - 1) // self.roster_size) * self.roster_size
        if not any(created_count == group_start + self.roster_size for _, created_count in self.query_log):
            assert len(self.created) == group_start + self.roster_size
            assert all(env.count == 3 for env in self.created[group_start:group_start + self.roster_size])
        self.query_log.append((self.name, len(self.created)))
        self.calls += 1
        if self.fail: raise RuntimeError("injected adapter failure")
        return np.zeros((1, 8, 7), dtype=np.float32)


def _integration_fixtures(tmp_path, monkeypatch, *, fail_arm=None, mismatch=False, arm_names=None):
    rows, fixtures, source_actions = [], {}, {}
    source_ids = {"random": [0, 3, 1, 4, 5, 6], "fixed": [0, 3, 4, 5, 6, 1]}
    for profile in ("random", "fixed"):
        near_conditions = scaling.CONDITIONS
        for condition in near_conditions:
            for index in range(6):
                prefix = np.full((2, 7), .01 if profile == "random" else .02, dtype=np.float32)
                initializer = np.full((1, 7), .001 * (near_conditions.index(condition) + 1), dtype=np.float32)
                actions = np.concatenate((prefix, initializer))
                physical = {"qpos": [float(index)]}
                window = [{"control_index": i, "monotonic_seconds": i * .025,
                    "simulation_elapsed_seconds": i * .025, "head_error_xyz_m": [0., 0., 0.],
                    "entry_clearance_m": .020, "yaw_error_rad": 0., "grasped": True,
                    "official_inserted": False, "head_hole_xyz_m": [-.105, 0., 0.],
                    "peg_center_world_m": [0., 0., 0.], "peg_quaternion_wxyz": [1., 0., 0., 0.],
                    "arm_qpos_rad": [0.] * 7} for i in range(11)]
                result = {"admitted": True, "admission_mode": "pose_window", "native_grasped": True,
                    "official_inserted": False, "actual_entry_clearance_m": .020, "controls": 3,
                    "source_prefix_controls": 2, "executed_prefix_actions": prefix.tolist(),
                    "executed_initializer_actions": initializer.tolist(), "controller_mode": "pd_ee_delta_pose",
                    "physical_state": {"qpos": [999.]} if mismatch and not rows else physical,
                    "last_two_observations": [np.full(39, 2.).tolist(), np.full(39, 3.).tolist()],
                    "last_pose_window_candidate": window, "control_timestep_s": .025,
                    "full_action_sha256": hashlib.sha256(actions.tobytes()).hexdigest()}
                rows.append({"profile": profile, "condition": condition, "catalog_index": index,
                    "source_scene_id": f"peg-validation_{profile}-{index:03}", "result": result})
                fixtures[(profile, index)] = {"physical_state": physical}
                source_actions[(profile, source_ids[profile][index])] = prefix
    created, query_log = [], []
    monkeypatch.setattr(near, "restore_peg_scene", lambda env, scene: env.reset_to_fixture(scene))
    monkeypatch.setattr(near, "capture_physical_state", lambda task: task.physical_state)
    monkeypatch.setattr(near, "native_grasped", lambda task: task.grasped)
    monkeypatch.setattr(near, "_relative_yaw", lambda task: 0.)
    arm_names = tuple(scaling.ARMS) if arm_names is None else tuple(arm_names)
    adapters = {name: _Adapter(name, created, query_log, fail=(fail_arm == name), roster_size=len(arm_names)) for name in arm_names}
    def env_factory(_profile): return _Env(created)
    def catalog_factory(profile): return _Catalog(profile, fixtures)
    loaded_sources = []
    def action_loader(profile, episode):
        loaded_sources.append((profile, episode))
        return source_actions[(profile, episode)]
    for profile in ("random", "fixed"):
        source = tmp_path / profile
        source.mkdir()
        (source / "trajectory.state.pd_ee_delta_pose.physx_cpu.json").write_text(json.dumps({
            "episodes": [{"episode_id": source_ids[profile][i], "episode_seed": 7000 + i} for i in range(6)]}))
        monkeypatch.setitem(near.DATA_DIRS, profile, source)
    return rows, created, query_log, adapters, env_factory, catalog_factory, action_loader, loaded_sources


def _output(path):
    path.mkdir()
    (path / "attempt_001.sh").write_text("#!/bin/bash\n# test wrapper\n")
    (path / "attempt_001.stdout.log").write_text("")
    (path / "attempt_001.stderr.log").write_text("")


def test_full_fake_orchestration_prepares_all_six_arms_and_preserves_sparse_source_mapping(tmp_path, monkeypatch):
    rows, created, queries, adapters, env_factory, catalog_factory, loader, loaded = _integration_fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(scaling.paired, "_seed", lambda seed: None)
    output = tmp_path / "output"
    _output(output)
    result = scaling.run_comparison(output, adapters=adapters, env_factory=env_factory,
        catalog_factory=catalog_factory, source_action_loader=loader, bank=rows)
    assert result["pairs_completed"] == 180 and result["arm_rows"] == 1080
    assert result["native_controls_including_replay"] == 180 * 6 * 4
    assert len(queries) == 1080 and len(created) == 1080
    assert all(result["per_condition"][p][c]["draws"] == 18
               for p in ("random", "fixed") for c in scaling.CONDITIONS)
    assert all(result["per_condition"][p][c]["unique_origins"] == 6
               for p in ("random", "fixed") for c in scaling.CONDITIONS)
    for profile in ("random", "fixed"):
        aggregate = result["per_condition"][profile]["_all_conditions"]
        assert aggregate["draws"] == 90 and aggregate["unique_origins"] == 6
        assert all(aggregate["per_seed"][str(seed)]["draws"] == 30 for seed in scaling.SEEDS)
        assert sum(aggregate["monotonicity"]["primary"]["patterns"].values()) == 90
    completed = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()
                 if json.loads(line)["status"] == "complete"]
    assert len(completed) == 180
    assert all(set(row["arms"]) == set(scaling.ARMS) for row in completed)
    assert all(row["handoff"]["max_state_error_vs_base"][name]["qpos"] == 0
               for row in completed for name in scaling.ARMS)
    assert ("random", 1) in loaded and ("fixed", 1) in loaded


def test_handoff_mismatch_fails_before_any_adapter_query(tmp_path, monkeypatch):
    rows, created, queries, adapters, env_factory, catalog_factory, loader, _ = _integration_fixtures(tmp_path, monkeypatch, mismatch=True)
    output = tmp_path / "output"
    _output(output)
    with pytest.raises(ValueError, match="replayed physical state differs"):
        scaling.run_comparison(output, adapters=adapters, env_factory=env_factory,
            catalog_factory=catalog_factory, source_action_loader=loader, bank=rows[:1])
    assert queries == []
    journal = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    assert journal[-1]["status"] == "failed" and journal[-1]["physical_controls_at_failure"] == 3


def test_completed_arm_is_durably_retained_when_later_distinct_model_fails(tmp_path, monkeypatch):
    rows, created, queries, adapters, env_factory, catalog_factory, loader, _ = _integration_fixtures(
        tmp_path, monkeypatch, fail_arm="cn100_15k")
    output = tmp_path / "output"
    _output(output)
    with pytest.raises(RuntimeError, match="injected adapter failure"):
        scaling.run_comparison(output, adapters=adapters, env_factory=env_factory,
            catalog_factory=catalog_factory, source_action_loader=loader, bank=rows[:1])
    journal = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    failed = journal[-1]
    assert failed["status"] == "failed" and "base" in failed["arms"]
    assert failed["arms"]["base"]["total_controls"] == 4
    assert failed["physical_controls_at_failure"] == 19


def test_three_arm_continuation_runs_fresh_pairs_with_only_30k_to_50k_curve(tmp_path, monkeypatch):
    rows, created, queries, adapters, env_factory, catalog_factory, loader, _ = _integration_fixtures(
        tmp_path, monkeypatch, arm_names=continuation.ARMS)
    monkeypatch.setattr(scaling.paired, "_seed", lambda seed: None)
    factory_calls = []
    monkeypatch.setattr(continuation, "create_adapters", lambda: factory_calls.append("three") or adapters)
    monkeypatch.setattr(scaling, "create_adapters", lambda: pytest.fail("six-arm factory must not be used"))
    suffix_limits = []
    run_suffix = near._run_suffix
    def checked_suffix(*args, **kwargs):
        suffix_limits.append((kwargs["max_native_controls"], kwargs["stage_seconds"]))
        return run_suffix(*args, **kwargs)
    monkeypatch.setattr(near, "_run_suffix", checked_suffix)
    output = tmp_path / "continuation"
    _output(output)
    result = continuation.run_comparison(output, env_factory=env_factory,
        catalog_factory=catalog_factory, source_action_loader=loader, bank=rows[:3])
    assert result["pairs_completed"] == 9 and result["arm_rows"] == 27
    assert factory_calls == ["three"]
    assert len(queries) == 27 and len(created) == 27
    assert set(suffix_limits) == {(continuation.MAX_NATIVE_CONTROLS, continuation.STAGE_SECONDS)}
    assert result["primary_curve"] == ["base", "cn500_30k", "cn500_50k"]
    assert result["model_roster"] == list(continuation.ARMS)
    assert "shared_base_and_cn20030_evaluated_once_per_draw" not in result
    assert result["per_condition"]["random"][scaling.CONDITIONS[0]]["paired_transitions"]["cn500_30k->cn500_50k"]["both"] == 9
    assert result["per_condition"]["random"][scaling.CONDITIONS[0]]["paired_transitions"]["base->cn500_50k"]["both"] == 9
    archived = output / "source_archive" / "peg_orientation_entry_continuation_eval.py"
    assert archived.read_bytes() == continuation.Path(continuation.__file__).read_bytes()
    assert all(set(row["arms"]) == set(continuation.ARMS) for row in
               (json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()
                if json.loads(line)["status"] == "complete"))


def test_three_arm_failure_at_50k_retains_base_and_30k_outcomes_and_controls(tmp_path, monkeypatch):
    rows, _, _, adapters, env_factory, catalog_factory, loader, _ = _integration_fixtures(
        tmp_path, monkeypatch, fail_arm=continuation.CN500_50K, arm_names=continuation.ARMS)
    monkeypatch.setattr(scaling.paired, "_seed", lambda seed: None)
    output = tmp_path / "continuation_failure"
    _output(output)
    with pytest.raises(RuntimeError, match="injected adapter failure"):
        continuation.run_comparison(output, adapters=adapters, env_factory=env_factory,
            catalog_factory=catalog_factory, source_action_loader=loader, bank=rows[:1])
    journal = [json.loads(line) for line in (output / "pairs.jsonl").read_text().splitlines()]
    failed = journal[-1]
    assert failed["status"] == "failed"
    assert set(failed["arms"]) == {continuation.BASE, continuation.CN500_30K}
    assert all(arm["native_success"] for arm in failed["arms"].values())
    assert failed["physical_controls_at_failure"] == 11
