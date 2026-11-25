import numpy as np
import json
import hashlib
import pytest
import torch

from examples.baselines.constrain import peg_orientation_takeover_study as study


class _Env:
    def __init__(self):
        self.n = 0

    def step(self, action):
        self.n += 1
        return np.array([[self.n, 0.0]]), 0.0, False, False, {"success": False}


class _Adapter:
    def __init__(self):
        self.calls = []

    def get_action(self, observations):
        self.calls.append(np.asarray(observations).copy())
        return np.arange(16, dtype=np.float32).reshape(1, 8, 2)


def test_policy_prefix_records_actual_chunk_actions_and_advances_live_history(monkeypatch):
    monkeypatch.setattr(study, "TOTAL_CONTROL_CAP", 100)
    monkeypatch.setattr(study, "_trace", lambda *args: {})
    env, adapter, budget = _Env(), _Adapter(), [0, study.time.monotonic()]
    prefix = study._record_prefix(env, None, 8, budget, policy=True,
                                  history=[np.array([0.0, 0.0])] * 2, adapter=adapter)
    assert prefix["controls"] == 8
    assert len(prefix["actions"]) == 8
    assert len(adapter.calls) == 1
    assert adapter.calls[0].shape == (2, 2)
    assert np.array_equal(prefix["history"][-1], [8.0, 0.0])
    assert np.array_equal(prefix["actions"][0], [0.0, 1.0])
    assert np.array_equal(prefix["actions"][-1], [14.0, 15.0])
    assert budget[0] == 8


def test_match_requires_physical_state_history_and_prefix_action_identity():
    base = {"physical_state": {"q": np.zeros(2)}, "observation_history": np.zeros((2, 3)),
            "prefix_action_sha256": "same"}
    assert max(study._match(base, base).values()) == 0
    different = {**base, "prefix_action_sha256": "different"}
    assert np.isinf(study._match(base, different)["prefix_action"])


def test_registration_archives_runner_and_its_regression_tests(tmp_path):
    outdir = tmp_path / "registered"
    study.prepare(outdir)
    registration = json.loads((outdir / "preregistration.json").read_text())
    closure = registration["source_closure_sha256"]
    assert registration["policy"]["training_seed"] == "10000+episode_id"
    assert registration["policy"]["validation_seed"] == "1+catalog_index"
    assert "continuous through control 296" in registration["policy"]["validation_history"]
    for path in ("examples/baselines/constrain/peg_orientation_takeover_study.py",
                 "examples/baselines/constrain/tests/test_peg_orientation_takeover_study.py",
                 "examples/baselines/constrain/peg_orientation_takeover_experiment.py",
                 "examples/baselines/constrain/tests/test_peg_orientation_takeover.py"):
        assert path in closure
        assert (outdir / "registered_sources" / path).is_file()
    assert registration["validation_catalog_path"] == str(study.VALIDATION_CATALOG.relative_to(study.ROOT))
    assert registration["validation_split"] == "validation_random"
    manifest = json.loads((study.VALIDATION_CATALOG / "manifest.json").read_text())
    assert manifest["split"] == registration["validation_split"]
    assert registration["validation_manifest_sha256"] == study._sha(study.VALIDATION_CATALOG / "manifest.json")


def test_planner_proxy_counts_control_before_executing_action(monkeypatch):
    class Raw:
        def __init__(self): self.calls = 0
        @property
        def unwrapped(self): return self
        def step(self, action):
            self.calls += 1
            return (None, 0, False, False, {})

    raw, budget = Raw(), [0, study.time.monotonic()]
    monkeypatch.setattr(study, "_trace", lambda *args: {})
    proxy = study._BudgetEnv(raw, budget)
    proxy.step({"control_mode": "pd_joint_pos", "action": np.zeros((1, 8))})
    assert raw.calls == 1 and budget[0] == 1


def test_validation_entrypoint_wires_seed_two_frame_history_and_exact_prefix(monkeypatch):
    calls = {}

    class Env:
        @property
        def unwrapped(self): return self
        def close(self): pass

    class Adapter:
        def __init__(self, *args, **kwargs): calls["adapter_task"] = kwargs["task"]
        def set_episode_context(self, index, metadata): calls["context"] = (index, metadata)

    scene = ({"target_yaw_rad": 0.2}, {}, np.array([1.0, 2.0]))
    monkeypatch.setattr(study, "_env", Env)
    monkeypatch.setattr(study, "PegConstrainAdapter", Adapter)
    monkeypatch.setattr(study, "restore_peg_scene", lambda env, s: np.array([[1.0, 2.0]]))
    monkeypatch.setattr(study, "_snapshot", lambda env, history, digest, count: {
        "observation_history": history, "prefix_action_sha256": digest, "controls": count})
    monkeypatch.setattr(study, "native_grasped", lambda task: False)

    def record(env, actions, cap, budget, *, policy, history, adapter):
        calls.update(cap=cap, history=[x.copy() for x in history], policy=policy,
                     seeded=True)
        return {"controls": cap, "history": [np.array([2.0, 0.0]), np.array([3.0, 0.0])],
                "success": False, "terminal": False, "actions": [np.zeros(7)] * cap,
                "trace": []}

    monkeypatch.setattr(study, "_record_prefix", record)
    result = study._run_validation_episode(4, type("Catalog", (), {"load": lambda self, i: scene})(), [0])
    assert calls["cap"] == 296 and calls["policy"] is True
    assert len(calls["history"]) == 2
    assert np.array_equal(calls["history"][0], calls["history"][1])
    assert result["prefix_controls"] == 296
    assert result["policy_seed"] == 5
    assert result["status"] == "unheld_at_296_ineligible"


def _registered_outdir(tmp_path, monkeypatch):
    outdir = tmp_path / "registered"
    outdir.mkdir()
    registration = {"source_closure_sha256": {}, "inputs_sha256": {}}
    registration["registration_sha256"] = study._canonical_hash(registration)
    (outdir / "preregistration.json").write_text(json.dumps(registration))
    monkeypatch.setattr(study, "_sources", lambda: {})
    monkeypatch.setattr(study, "_input_hashes", lambda: {})
    return outdir


def test_run_finalization_normalizes_completed_rows_before_hash_and_persistence(tmp_path, monkeypatch):
    outdir = _registered_outdir(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "TRAIN_IDS", (7,))
    monkeypatch.setattr(study, "VALIDATION_IDS", ())

    def run_episode(index, budget, on_arm):
        on_arm({"episode_id": index, "arm": "frozen_policy", "native_controls_total": 3,
                "native_success": np.bool_(True), "trace": np.array([1, 2]),
                "history": torch.tensor([3.0, 4.0]), "planner": (np.int64(5), np.bool_(False))})

    monkeypatch.setattr(study, "_run_train_episode", run_episode)
    study.run(outdir, "test")

    attempt = outdir / "attempt_test"
    result = json.loads((attempt / "results.json").read_text())
    digest = result.pop("results_sha256")
    assert digest == study._canonical_hash(result)
    row = result["rows"][0]
    assert row["native_success"] is True
    assert isinstance(row["native_success"], bool)
    assert row["trace"] == [1, 2]
    assert row["history"] == [3.0, 4.0]
    assert row["planner"] == [5, False]
    assert json.loads((attempt / "execution_ledger.json").read_text())["status"] == "completed"


def test_failed_run_leaves_hashed_full_arm_journal_before_next_arm(tmp_path, monkeypatch):
    outdir = _registered_outdir(tmp_path, monkeypatch)
    monkeypatch.setattr(study, "TRAIN_IDS", (7,))
    monkeypatch.setattr(study, "VALIDATION_IDS", ())
    original = RuntimeError("injected after first arm")

    def run_episode(index, budget, on_arm):
        on_arm({"episode_id": index, "arm": "live_expert_continuation", "native_controls_total": 9,
                "status": "native_success", "branch": {"qpos": np.array([1.0])},
                "native_success": np.bool_(True), "prefix_action_sha256": "action-hash",
                "policy_rng_after_prefix": {"torch_cpu": torch.tensor([7], dtype=torch.uint8), "torch_cuda": []},
                "trace": [{"native_success": np.bool_(True)}], "history": [torch.tensor([2.0])]})
        raise original

    monkeypatch.setattr(study, "_run_train_episode", run_episode)
    with pytest.raises(RuntimeError, match="injected after first arm") as raised:
        study.run(outdir, "failed")
    assert raised.value is original

    attempt = outdir / "attempt_failed"
    failure = attempt / "attempt_failure.json"
    assert failure.stat().st_size > 0
    assert json.loads(failure.read_text())["error"] == "injected after first arm"
    ledger = json.loads((attempt / "execution_ledger.json").read_text())
    arm = ledger["arms"][0]
    reference = arm["journals"][0]
    journal = attempt / reference["path"]
    assert journal.is_file()
    assert hashlib.sha256(journal.read_bytes()).hexdigest() == reference["sha256"]
    saved = json.loads(journal.read_text())
    assert saved["row"]["trace"][0]["native_success"] is True
    assert saved["row"]["status"] == "native_success"
    assert saved["row"]["branch"]["qpos"] == [1.0]
    assert saved["row"]["native_success"] is True
    assert saved["row"]["prefix_action_sha256"] == "action-hash"
    assert saved["row"]["policy_rng_after_prefix"]["torch_cpu"] == [7]


@pytest.mark.parametrize(
    ("policy_success", "expert_status", "expected"),
    [(True, "native_success", {"both": 1, "expert_only": 0, "policy_only": 0, "neither": 0}),
     (False, "native_success", {"both": 0, "expert_only": 1, "policy_only": 0, "neither": 0}),
     (True, "plan_failure", {"both": 0, "expert_only": 0, "policy_only": 1, "neither": 0}),
     (False, "control_cap", {"both": 0, "expert_only": 0, "policy_only": 0, "neither": 1})],
)
def test_analysis_counts_actual_validation_arm_success_schemas(policy_success, expert_status, expected):
    rows = [{"arm": "live_expert_continuation", "episode_id": 0, "triggered": np.bool_(True),
             "expert": {"status": "native_success"}},
            {"arm": "validation_pair", "status": "completed", "episode_id": 1,
             "arms": [{"arm": "unchanged_policy_continuation", "native_success": np.bool_(policy_success)},
                      {"arm": "live_expert_continuation", "status": expert_status}]}]
    analysis = study._analysis(rows)
    outcomes = analysis["validation_policy_296"]["paired_outcomes"]
    assert outcomes == expected
    assert analysis["training_near_branch"]["live_expert_continuation"]["n_success"] == 1


def test_validation_policy_uses_seed_one_after_restore_and_keeps_rng_stream_at_handoff(monkeypatch):
    events = []

    class Env(_Env):
        @property
        def unwrapped(self): return self
        def close(self): pass
        def get_obs(self): return np.array([[0.0, 0.0]])
        def step(self, action):
            self.n += 1
            return np.array([[self.n, 0.0]]), 0.0, False, False, {"success": False}
        def has_peg_inserted(self): return [torch.tensor(False)]

    class Adapter:
        def __init__(self, *args, **kwargs): pass
        def set_episode_context(self, index, metadata): pass
        def get_action(self, observations):
            events.append(("infer", float(torch.rand(()))))
            return np.zeros((1, 8, 2), dtype=np.float32)

    scene = ({"target_yaw_rad": 0.2}, {}, np.array([0.0, 0.0]))
    monkeypatch.setattr(study, "_env", Env)
    monkeypatch.setattr(study, "PegConstrainAdapter", Adapter)
    monkeypatch.setattr(study, "restore_peg_scene", lambda env, value: (events.append(("restore", env.n)) or np.array([[0.0, 0.0]])))
    monkeypatch.setattr(study, "_snapshot", lambda env, history, digest, count: {
        "physical_state": {"q": [0]}, "observation_history": np.asarray(history[-2:]),
        "prefix_action_sha256": digest, "controls": count})
    monkeypatch.setattr(study, "_trace", lambda *args: {})
    monkeypatch.setattr(study, "_head", lambda task: np.array([0.0, 0.0, 0.0]))
    monkeypatch.setattr(study, "native_grasped", lambda task: True)
    monkeypatch.setattr(study, "_expert_live", lambda env, budget: {"controls": 0, "status": "control_cap", "native_success": False, "trace": []})
    monkeypatch.setattr(study.torch, "manual_seed", lambda seed: (events.append(("seed", seed)) or torch.random.default_generator.manual_seed(seed)))
    monkeypatch.setattr(study, "TOTAL_CONTROL_CAP", 10000)
    monkeypatch.setattr(study, "_head", lambda task: np.array([0.0, 0.0, 0.0]))

    result = study._run_validation_episode(2, type("Catalog", (), {"load": lambda self, i: scene})(), [0])

    restore_index = next(i for i, item in enumerate(events) if item[0] == "restore")
    seed_index = next(i for i, item in enumerate(events) if item[0] == "seed")
    assert restore_index < seed_index
    assert events[seed_index] == ("seed", 3)
    draws = [value for kind, value in events if kind == "infer"]
    generator = torch.Generator().manual_seed(3)
    expected = [float(torch.rand((), generator=generator)) for _ in range(len(draws))]
    assert draws == expected
    assert len(draws) == 37 + 25
    assert result["policy_seed"] == 3
    assert study._seed_policy(None, 2)[0] == 10002
