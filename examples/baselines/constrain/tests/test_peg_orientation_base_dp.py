import json

import h5py
import numpy as np
import torch

from examples.baselines.constrain import peg_orientation_base_dp
from examples.baselines.constrain.peg_orientation_base_dp import (
    PegBaseDataset,
    _video_frame,
)


def test_dataset_windows_keep_all_seven_actions_and_only_39d_observations(tmp_path):
    data_path = tmp_path / "demo.h5"
    manifest_path = tmp_path / "demo.json"
    obs = np.arange(20 * 39, dtype=np.float32).reshape(20, 39)
    actions = np.arange(19 * 7, dtype=np.float32).reshape(19, 7) / 200
    with h5py.File(data_path, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("obs", data=obs)
        group.create_dataset("actions", data=actions)
    manifest_path.write_text('{"episodes": [{"episode_id": 0}]}')

    dataset = PegBaseDataset(data_path, manifest_path, expected_episodes=1)
    first = dataset[0]
    last = dataset[len(dataset) - 1]

    assert first["observations"].shape == (2, 39)
    assert first["actions"].shape == (16, 7)
    torch.testing.assert_close(first["observations"], torch.from_numpy(obs[:1].repeat(2, axis=0)))
    torch.testing.assert_close(first["actions"][1:], torch.from_numpy(actions[:15]))
    torch.testing.assert_close(last["actions"][-1], torch.cat((torch.zeros(6), torch.from_numpy(actions[-1, 6:7]))))
    assert torch.count_nonzero(first["actions"][:, 3:6]) > 0


def test_video_frame_converts_batched_tensor_to_uint8_image():
    class Env:
        def render(self):
            return torch.ones((1, 12, 16, 3), dtype=torch.float32) * 255

    frame = _video_frame(Env())
    assert frame.shape == (12, 16, 3)
    assert frame.dtype == np.uint8


def test_evaluator_video_scene_indices_capture_only_selected_episode(tmp_path, monkeypatch):
    import gymnasium as gym
    import imageio.v2 as imageio

    class Catalog:
        def __init__(self, _): pass
        def __len__(self): return 2
        def load(self, index):
            return ({"hole_center_m": np.zeros(3), "target_yaw_rad": 1.57,
                     "seed": 7 + index}, {}, None)

    class Adapter:
        metadata = {}
        def __init__(self, *_args): pass
        def get_action(self, _obs): return torch.zeros((1, 1, 7))

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.render_calls = 0
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.peg.pose.p = torch.zeros((1, 3))
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        def render(self):
            self.render_calls += 1
            return np.zeros((4, 4, 3), dtype=np.uint8)
        def step(self, _action): return torch.zeros((1, 39)), 0, True, False, {"success": True}
        def close(self): pass

    class Writer:
        def __init__(self): self.frames = []; self.closed = False
        def append_data(self, frame): self.frames.append(frame)
        def close(self): self.closed = True

    env, writer = Env(), Writer()
    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(peg_orientation_base_dp, "restore_peg_scene", lambda *_args: torch.zeros((1, 39)))
    monkeypatch.setattr(imageio, "get_writer", lambda *_args, **_kwargs: writer)

    peg_orientation_base_dp.evaluate(
        "unused.pt", "unused-scenes", tmp_path / "result", device="cpu", video=True,
        video_scene_indices=[0], policy_factory=lambda _task: Adapter(),
    )

    assert len(writer.frames) == 1
    assert env.render_calls == 1
    assert writer.closed


def test_evaluator_closes_env_when_policy_factory_raises(tmp_path, monkeypatch):
    import gymnasium as gym
    import pytest

    class Env:
        unwrapped = None

        def __init__(self):
            self.unwrapped = self
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    env = Env()
    failure = RuntimeError("policy construction failed")
    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", lambda _path: [None])

    def raise_from_factory(_task):
        raise failure

    with pytest.raises(RuntimeError) as error:
        peg_orientation_base_dp.evaluate(
            "unused.pt", "unused-scenes", tmp_path / "result", device="cpu",
            policy_factory=raise_from_factory,
        )

    assert error.value is failure
    assert env.close_calls == 1


def test_evaluator_captures_first_preaction_sample_and_terminal_native_failure(tmp_path, monkeypatch):
    class Catalog:
        def __init__(self, _):
            pass

        def __len__(self):
            return 1

        def load(self, _):
            return ({"hole_center_m": np.zeros(3), "target_yaw_rad": 1.57, "seed": 7}, {}, None)

    class Adapter:
        def __init__(self, *_args):
            self.metadata = {}
            self.context = None

        def set_episode_context(self, index, scene_metadata, pseudo_target):
            self.context = (index, scene_metadata["seed"], pseudo_target)

        def get_action(self, _obs):
            return torch.zeros((1, 1, 7))

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.positions = [0.3, 0.2, 0.09, 0.01]
            self.index = 0
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.peg.pose.p = torch.tensor([[self.positions[0], 0.0, 0.0]])
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        def step(self, _action):
            self.index += 1
            self.peg.pose.p = torch.tensor([[self.positions[self.index], 0.0, 0.0]])
            return torch.zeros((1, 39)), 0.0, self.index == 3, False, {"success": False, "fail": "native_failure"}

        def close(self):
            pass

    env = Env()
    import gymnasium as gym

    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(peg_orientation_base_dp, "restore_peg_scene", lambda *_args: torch.zeros((1, 39)))
    adapter = Adapter()
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", lambda *_args: adapter)
    peg_orientation_base_dp.evaluate(
        "unused.pt", "unused-scenes", tmp_path / "result", device="cpu",
        hint_mode="shuffled", hint_assignment={"target_yaws": [0.4]},
    )
    record = json.loads((tmp_path / "result/episode_000.json").read_text())
    diagnostic = record["preinsertion"]
    assert diagnostic["status"] == "sampled"
    assert diagnostic["sample_step"] == 2
    assert np.isclose(diagnostic["distance_m"], 0.09)
    assert diagnostic["terminal"] is True
    assert diagnostic["native_success"] is False
    assert diagnostic["failure_type"] == "native_failure"
    assert adapter.context == (0, 7, 0.4)
    assert record["evaluation_identity"]["hint_mode"] == "shuffled"


def test_evaluator_honors_explicit_native_action_budget(tmp_path, monkeypatch):
    class Catalog:
        def __init__(self, _):
            pass

        def __len__(self):
            return 2

        def load(self, index):
            return ({"hole_center_m": np.zeros(3), "target_yaw_rad": 1.57, "seed": 7 + index}, {}, None)

    class Adapter:
        def __init__(self, *_args):
            self.metadata = {"checkpoint_sha256": "captured-before-loop"}

        def get_action(self, _obs):
            return torch.zeros((1, 8, 7))

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.steps = 0
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.peg.pose.p = torch.tensor([[0.3, 0.0, 0.0]])
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        def step(self, _action):
            self.steps += 1
            return torch.zeros((1, 39)), 0.0, False, False, {"success": False}

        def close(self):
            pass

    env = Env()
    import gymnasium as gym

    monkeypatch.setattr(gym, "make", lambda *_args, **kwargs: (setattr(env, "configured_budget", kwargs["max_episode_steps"]) or env))
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(peg_orientation_base_dp, "restore_peg_scene", lambda *_args: torch.zeros((1, 39)))
    policy_seeds = []
    original_manual_seed = torch.manual_seed
    monkeypatch.setattr(torch, "manual_seed", lambda value: (policy_seeds.append(value), original_manual_seed(value))[1])

    peg_orientation_base_dp.evaluate("unused.pt", "unused-scenes", tmp_path / "result",
                                     device="cpu", max_episode_steps=3, seed=11)
    record = json.loads((tmp_path / "result/episode_000.json").read_text())
    assert env.configured_budget == peg_orientation_base_dp.EVAL_ENV_TIME_LIMIT == 1000
    next_record = json.loads((tmp_path / "result/episode_001.json").read_text())
    assert env.steps == 6
    assert record["executed_controls"] == next_record["executed_controls"] == 3
    assert record["policy_sampling_seed"] == 11
    assert record["evaluation_identity"]["checkpoint_sha256"] == "captured-before-loop"
    assert record["evaluation_identity"]["catalog_manifest_sha256"] is None
    assert record["evaluation_identity"]["evaluator_source_sha256"] == peg_orientation_base_dp.sha256_file(
        peg_orientation_base_dp.__file__
    )
    assert next_record["policy_sampling_seed"] == 12
    assert policy_seeds == [11, 12]
    assert record["first_success_control"] is next_record["first_success_control"] is None
    assert record["failure_type"] == next_record["failure_type"] == "max_episode_steps"
    assert record["preinsertion"]["terminal"] is True
    assert "phase_trace" not in record
    assert "stage_geometry_terminal" not in record


def test_evaluator_stops_at_first_native_success_and_records_control(tmp_path, monkeypatch):
    class Catalog:
        def __init__(self, _): pass
        def __len__(self): return 1
        def load(self, _):
            return ({"hole_center_m": np.zeros(3), "target_yaw_rad": 1.57, "seed": 7}, {}, None)

    class Adapter:
        def __init__(self, *_args): self.metadata = {}
        def get_action(self, _obs): return torch.zeros((1, 8, 7))

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.steps = 0
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.peg.pose.p = torch.tensor([[0.3, 0.0, 0.0]])
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        def step(self, _action):
            self.steps += 1
            success = self.steps == 3
            return torch.zeros((1, 39)), 0.0, success, False, {"success": success}
        def close(self): pass

    env = Env()
    import gymnasium as gym
    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(peg_orientation_base_dp, "restore_peg_scene", lambda *_args: torch.zeros((1, 39)))

    peg_orientation_base_dp.evaluate("unused.pt", "unused-scenes", tmp_path / "result",
                                     device="cpu", max_episode_steps=8, seed=11)
    record = json.loads((tmp_path / "result/episode_000.json").read_text())
    assert env.steps == 3
    assert record["executed_controls"] == record["first_success_control"] == 3
    assert record["executed_chunk_lengths"] == [3]
    assert sum(record["executed_chunk_lengths"]) == record["executed_controls"]
    assert record["success"] is True
    assert record["failure_type"] == "success"


def test_training_rollout_comparison_replans_at_requested_horizon(tmp_path, monkeypatch):
    import h5py
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    data_path = tmp_path / "training.h5"
    manifest_path = tmp_path / "manifest.json"
    observation = np.zeros(39, dtype=np.float32)
    observation[25] = 0.3
    observation[35] = 0.9
    observation[36] = 0.0
    with h5py.File(data_path, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("obs", data=np.repeat(observation[None], 4, axis=0))
        group.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
    manifest_path.write_text(json.dumps({"episodes": [{"episode_seed": 17}]}))

    class Policy:
        def __init__(self):
            self.calls = []
        def load_state_dict(self, *_args, **_kwargs): pass
        def eval(self): pass
        def get_action(self, _observation):
            self.calls.append(1)
            return torch.zeros((1, 16, 7))

    policy = Policy()

    class TaskAgent:
        def is_grasping(self, _peg):
            return torch.tensor([False])

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.agent = TaskAgent()
            self.peg = object()
            self.steps = 0
        def reset(self, seed):
            self.seed = seed
            self.steps = 0
            return torch.as_tensor(observation[None]), {}
        def step(self, _action):
            self.steps += 1
            return torch.as_tensor(observation[None]), 0.0, False, False, {"success": False}
        def close(self): pass

    envs = []
    import gymnasium as gym
    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: (envs.append(Env()) or envs[-1]))
    monkeypatch.setattr(diagnosis.torch, "load", lambda *_args, **_kwargs: {"ema_agent": {}})
    monkeypatch.setattr(diagnosis, "build_agent", lambda *_args: policy)
    monkeypatch.setattr(diagnosis, "sha256_file", lambda _path: "frozen")

    result = diagnosis.run_training_rollout_horizon_comparison(
        "checkpoint.pt", data_path, manifest_path, tmp_path / "result.json",
        episode_ids=[0], max_episode_steps=3, execution_horizons=(8, 2),
    )

    assert [row["execution_horizon"] for row in result["episodes"]] == [8, 2]
    assert [row["executed_controls"] for row in result["episodes"]] == [3, 3]
    assert len(policy.calls) == 3
    assert [env.seed for env in envs] == [17, 17]
    assert result["episodes"][0]["initial_observation_max_abs_error"] == 0
    assert np.isclose(result["episodes"][0]["minimum_peg_hole_xy_distance_m"], 0.3)


def test_evaluator_records_validation_phase_without_changing_action_profile(tmp_path, monkeypatch):
    observation = np.zeros(39, dtype=np.float32)
    observation[25] = 0.4
    observation[35] = 0.9
    observation[36] = 0.0
    observation[38] = 0.2

    class Catalog:
        def __init__(self, _):
            pass

        def __len__(self):
            return 2

        def load(self, index):
            return (
                {"hole_center_m": [0.0, 0.0, 0.0], "target_yaw_rad": 1.57, "seed": 2026112250 + index},
                {},
                observation.copy(),
            )

    class Adapter:
        def __init__(self, *_args):
            self.metadata = {}
            self.observations = []

        def get_action(self, obs):
            self.observations.append(obs.detach().cpu().numpy().copy())
            return torch.zeros((1, 8, 7))

    class TaskAgent:
        def is_grasping(self, _peg):
            return torch.tensor([self.grasp_flags[self.steps - 1]])

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.steps = 0
            self.grasp_flags = [False, True, False]
            self.agent = TaskAgent()
            self.agent.steps = 0
            self.agent.grasp_flags = self.grasp_flags
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.peg.pose.p = torch.tensor([[0.05, 0.0, 0.0]])
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self.peg.pose.raw_pose = torch.tensor([[0.05, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
            self.box_hole_pose = type("Hole", (), {})()
            self.box_hole_pose.raw_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

        def has_peg_inserted(self):
            return torch.tensor([False]), torch.tensor([[0.02, 0.0, 0.0]])

        def step(self, _action):
            self.steps += 1
            self.agent.steps = self.steps
            yaw = -3.10 if self.steps == 1 else 3.10
            self.peg.pose.raw_pose = torch.tensor([[0.05 + 0.01 * self.steps, 0.0, 0.0,
                                                    np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]])
            return torch.as_tensor(observation[None]), 0.0, False, False, {"success": False}

        def close(self):
            pass

    env = Env()
    adapter = Adapter()
    import gymnasium as gym

    monkeypatch.setattr(
        gym, "make",
        lambda *_args, **kwargs: (setattr(env, "configured_budget", kwargs["max_episode_steps"]) or env),
    )
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", lambda *_args: adapter)
    monkeypatch.setattr(
        peg_orientation_base_dp, "restore_peg_scene",
        lambda *_args: torch.as_tensor(observation[None]),
    )
    policy_seeds = []
    original_manual_seed = torch.manual_seed
    monkeypatch.setattr(
        torch, "manual_seed",
        lambda value: (policy_seeds.append(value), original_manual_seed(value))[1],
    )

    summary = peg_orientation_base_dp.evaluate(
        "unused.pt", "unused-scenes", tmp_path / "result", device="cpu",
        max_episode_steps=3, seed=1907, scene_indices=[1], record_stage_geometry=True,
    )
    record = json.loads((tmp_path / "result/episode_001.json").read_text())
    phase = record["phase"]
    expected_source_difference = abs(0.4 - 0.05)

    assert summary["episodes"] == 1
    assert env.configured_budget == 1000
    assert env.steps == 3
    assert policy_seeds == [1908]
    assert record["policy_sampling_seed"] == 1908
    assert record["executed_controls"] == 3
    assert record["success"] is False
    assert len(adapter.observations) == 1
    assert adapter.observations[0].shape == (1, 2, 39)
    assert np.allclose(adapter.observations[0][0], observation)
    assert phase["actions_per_sampler_call"] == [3]
    assert phase["ever_grasped"] is True
    assert phase["first_grasp_control"] == 2
    assert phase["contact_losses"] == 1
    assert phase["first_pose_xy_proximity_control"] == 0
    assert phase["first_observation_xy_proximity_control"] is None
    assert np.isclose(phase["minimum_pose_peg_hole_xy_m"], 0.05)
    assert np.isclose(phase["minimum_observation_peg_hole_xy_m"], 0.4)
    assert np.isclose(phase["max_preaction_xy_source_abs_difference_m"], expected_source_difference)
    assert np.isclose(
        record["phase_trace"][0]["observation_peg_hole_center_distance_m"],
        np.linalg.norm(np.array([0.4, 0.0, 0.0]) - np.array([0.0, 0.0, 0.2])),
    )
    assert phase["initial_observation_max_abs_error"] == 0
    assert phase["xy_le_0.1_meaning"] == "proximity_trigger_not_official_success_or_axis_alignment"
    assert len(record["phase_trace"]) == 3
    assert record["phase_trace"][1]["is_grasping"] is True
    assert record["phase_trace"][2]["contact_lost"] is True
    assert record["stage_geometry_terminal"]["official_insertion"] is False
    assert np.isclose(record["stage_geometry_terminal"]["peg_center_to_hole_center_distance_m"], 0.08)
    assert np.isclose(record["stage_geometry_terminal"]["wrapped_peg_plus_x_minus_box_plus_x_yaw_rad"], 3.10)
    assert len([row for row in record["phase_trace"] if "stage_geometry_preaction" in row]) == 3
    assert np.isclose(record["phase_trace"][0]["stage_geometry_preaction"]["peg_yaw_rad"], 0.0)
    assert np.isclose(record["phase_trace"][0]["stage_geometry_postaction"]["peg_yaw_rad"], -3.10)
    assert np.isclose(record["phase_trace"][1]["stage_geometry_preaction"]["wrapped_peg_plus_x_minus_box_plus_x_yaw_rad"], -3.10)
    assert np.isclose(record["phase_trace"][1]["stage_geometry_postaction"]["wrapped_peg_plus_x_minus_box_plus_x_yaw_rad"], 3.10)
    assert np.isclose(record["phase_trace"][2]["stage_geometry_postaction"]["peg_center_to_hole_center_distance_m"], 0.08)


def test_known_validation_failure_symptom_is_cap_exhaustion_without_success():
    from examples.baselines.constrain.peg_orientation_base_diagnosis import (
        known_validation_failure_reproduced,
    )

    recorded_failure = {
        "scene_index": 0, "policy_sampling_seed": 1907, "executed_controls": 500,
        "first_success_control": None, "success": False, "failure_type": "max_episode_steps",
        "preinsertion": {"status": "not_reached", "distance_m": 0.37840459788731284},
    }
    assert known_validation_failure_reproduced(recorded_failure, 0) is True
    assert known_validation_failure_reproduced({**recorded_failure, "success": True}, 0) is False
    assert known_validation_failure_reproduced({**recorded_failure, "executed_controls": 100}, 0) is False


def test_validation_phase_command_reuses_evaluator_profile(tmp_path, monkeypatch):
    import pytest
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    captured = {}

    def fake_evaluate(*_args, **kwargs):
        Path = __import__("pathlib").Path
        Path(kwargs["output_dir"]).mkdir(parents=True)
        captured["kwargs"] = kwargs
        return {
            "records": [{"scene_index": 0, "phase_trace": [{"control": 1}], "phase": {"ever_grasped": False}}],
            "episodes": 1, "successes": 0,
            "checkpoint": {"checkpoint_sha256": "ckpt"},
        }

    monkeypatch.setattr(diagnosis, "evaluate", fake_evaluate)
    monkeypatch.setattr(diagnosis, "sha256_file", lambda _path: "frozen")
    diagnosis.run_validation_phase_diagnosis(
        "ckpt.pt", tmp_path / "catalog", tmp_path / "out", scene_indices=[0],
        max_episode_steps=500, seed=1907, device="cpu",
    )
    assert captured["kwargs"]["record_phase_trace"] is True
    assert captured["kwargs"]["scene_indices"] == [0]
    assert captured["kwargs"]["max_episode_steps"] == 500
    assert captured["kwargs"]["seed"] == 1907
    assert captured["kwargs"]["video"] is False
    assert captured["kwargs"]["device"] == "cpu"
    identity = json.loads((tmp_path / "out" / "phase_identity.json").read_text())
    assert identity["scene_indices"] == [0]
    assert identity["record_phase_trace"] is True
    assert "phase_trace" not in identity["records"][0]
    with pytest.raises(ValueError):
        diagnosis.run_validation_phase_diagnosis(
            "ckpt.pt", tmp_path / "catalog", tmp_path / "too-wide",
            scene_indices=list(range(7)), max_episode_steps=500,
        )


def test_evaluator_records_official_peg_head_and_action_only_inside_window(tmp_path, monkeypatch):
    observation = np.zeros(39, dtype=np.float32)
    observation[25:28] = [0.5, 0.0, 0.0]
    action = [0.1, -0.2, 0.3, 0.0, 0.0, 0.4, -1.0]
    tcp_pose = [0.11, 0.22, 0.33, 1.0, 0.0, 0.0, 0.0]
    peg_pose = [0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    hole_pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
    root_pose = [-0.615, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    official_pre_head = [0.2, -0.03, 0.04]
    official_post_head = [0.21, -0.02, 0.01]

    class Catalog:
        def __init__(self, _):
            pass

        def __len__(self):
            return 1

        def load(self, index):
            return (
                {"hole_center_m": [0.0, 0.0, 0.0], "target_yaw_rad": 1.57, "seed": 2026112250},
                {},
                observation.copy(),
            )

    class Adapter:
        def __init__(self, *_args):
            self.metadata = {}

        def get_action(self, _obs):
            return torch.tensor([[action] * 8], dtype=torch.float32)

    def pose(values):
        wrapped = type("Pose", (), {})()
        wrapped.raw_pose = torch.tensor([values], dtype=torch.float32)
        return wrapped

    class Config:
        frame = "test_root_frame"
        use_delta = True
        use_target = False
        pos_lower = -0.1
        pos_upper = 0.1
        rot_lower = -0.1

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.steps = 0
            self.agent = type("Agent", (), {})()
            self.agent.tcp = type("Tcp", (), {"pose": pose(tcp_pose)})()
            self.agent.robot = type("Robot", (), {"pose": pose(root_pose)})()
            self.agent.robot.get_qpos = lambda: torch.zeros(9)
            self.agent.robot.get_qvel = lambda: torch.zeros(9)
            self.peg_half_sizes = torch.zeros(3)
            self.agent.controller = type("Controller", (), {})()
            self.agent.controller.controllers = {"arm": type("Arm", (), {"config": Config()})()}
            self.agent.is_grasping = lambda _peg: torch.tensor([False])
            self.peg = type("Peg", (), {"pose": pose(peg_pose)})()
            self.peg.pose.p = torch.tensor([[0.5, 0.0, 0.0]])
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self.box_hole_pose = pose(hole_pose)
            self.box_hole_radii = torch.tensor([0.025])

        def has_peg_inserted(self):
            return torch.tensor([False]), torch.tensor([official_pre_head])

        def step(self, _action):
            self.steps += 1
            return (
                torch.as_tensor(observation[None]), 0.0, False, False,
                {"success": False, "peg_head_pos_at_hole": torch.tensor([official_post_head])},
            )

        def close(self):
            pass

    env = Env()
    import gymnasium as gym

    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", lambda *_args: Adapter())
    monkeypatch.setattr(
        peg_orientation_base_dp, "restore_peg_scene",
        lambda *_args: torch.as_tensor(observation[None]),
    )

    summary = peg_orientation_base_dp.evaluate(
        "unused.pt", "unused-scenes", tmp_path / "result", device="cpu",
        max_episode_steps=3, seed=1907, scene_indices=[0], record_phase_trace=True,
        insertion_trace_window=(2, 2),
    )
    trace = summary["records"][0]["phase_trace"]
    window = trace[1]

    assert env.steps == 3
    assert "executed_action_7d" not in trace[0]
    assert "executed_action_7d" not in trace[2]
    assert np.allclose(window["executed_action_7d"], action)
    assert np.allclose(window["preaction_peg_head_in_hole_m"], official_pre_head)
    assert not np.allclose(window["preaction_peg_head_in_hole_m"], [0.5, 0.0, 0.0])
    assert np.allclose(window["postaction_official_peg_head_in_hole_m"], official_post_head)
    assert window["postaction_peg_head_source"] == "info"
    assert np.allclose(window["peg_head_displacement_in_hole_m"], [0.01, 0.01, -0.03])
    assert np.allclose(window["commanded_translation_in_hole"], [0.1, -0.2, 0.3])
    assert np.allclose(window["preaction_tcp_pose"], tcp_pose)
    assert np.allclose(window["preaction_peg_pose"], peg_pose)
    assert np.allclose(window["postaction_tcp_pose"], tcp_pose)
    assert np.allclose(window["postaction_peg_pose"], peg_pose)
    assert np.allclose(window["postaction_root_pose"], root_pose)
    assert np.allclose(window["tcp_displacement_m"], [0.0, 0.0, 0.0])
    assert np.allclose(window["peg_center_displacement_m"], [0.0, 0.0, 0.0])
    assert np.allclose(window["root_displacement_m"], [0.0, 0.0, 0.0])
    assert np.isclose(window["box_hole_radius_m"], 0.025)
    assert window["preaction_official_insertion"] is False
    assert np.isclose(window["action_translation_l2"], 0.37416573867739417)
    assert np.isclose(window["action_rotation_l2"], 0.4)
    assert summary["records"][0]["phase"]["controller_frame"]["frame"] == "test_root_frame"
    assert summary["records"][0]["phase"]["insertion_trace_window"] == [2, 2]


def test_insertion_window_records_sampler_history_chunk_and_rng(tmp_path, monkeypatch):
    """Sampler calls that cover the window keep the history actually passed in, not a later one."""
    import gymnasium as gym

    def obs_at(step):
        flat = np.zeros(39, dtype=np.float32)
        flat[0:9] = step
        flat[9:18] = step + 0.25
        flat[18:25] = step + 1
        flat[25:32] = step + 2
        flat[32:35] = 0.01 * (step + 1)
        flat[35] = 0.018
        flat[36:39] = step + 3
        return flat

    class Catalog:
        def __init__(self, _path):
            pass

        def __len__(self):
            return 1

        def load(self, _index):
            return (
                {"hole_center_m": [0.0, 0.0, 0.0], "target_yaw_rad": 1.57, "seed": 2026112250},
                {},
                obs_at(0),
            )

    class Adapter:
        def __init__(self, *_args):
            self.metadata = {}
            self.calls = 0
            self.entry_rng = []
            self.received = []
            self.returned = []

        def get_action(self, observations):
            self.calls += 1
            self.entry_rng.append(torch.get_rng_state().clone())
            self.received.append(observations.detach().cpu().clone())
            noise = torch.randn(1, 8, 7)
            self.returned.append(noise.detach().cpu().clone())
            return noise

    class LivePose:
        def __init__(self, env, start):
            self.env = env
            self.start = start

        @property
        def raw_pose(self):
            values = self.env.current()[self.start:self.start + 7]
            return torch.tensor(values[None], dtype=torch.float32)

    class HolePose:
        def __init__(self, env):
            self.env = env

        @property
        def raw_pose(self):
            center = self.env.current()[36:39]
            return torch.tensor([[*center, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    class Config:
        frame = "test_root_frame"
        use_delta = True
        use_target = False
        pos_lower = -0.1
        pos_upper = 0.1
        rot_lower = -0.1

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.k = 0
            self.steps = 0
            self.agent = type("Agent", (), {})()
            self.agent.tcp = type("Tcp", (), {"pose": LivePose(self, 18)})()
            self.agent.robot = type("Robot", (), {"pose": LivePose(self, 0)})()
            self.agent.robot.get_qpos = lambda: torch.tensor(self.current()[0:9])
            self.agent.robot.get_qvel = lambda: torch.tensor(self.current()[9:18])
            self.agent.controller = type("Controller", (), {})()
            self.agent.controller.controllers = {"arm": type("Arm", (), {"config": Config()})()}
            self.agent.is_grasping = lambda _peg: torch.tensor([True])
            self.peg = type("Peg", (), {"pose": LivePose(self, 25)})()
            self.peg.pose.p = torch.zeros((1, 3))
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self.box_hole_pose = HolePose(self)
            self.box_hole_radii = torch.tensor([0.018])
            self.peg_half_sizes = torch.tensor(obs_at(0)[32:35])

        def current(self):
            return obs_at(self.k)

        def has_peg_inserted(self):
            return torch.tensor([False]), torch.tensor([float(self.k), -0.035, 0.001])

        def step(self, _action):
            self.steps += 1
            self.k += 1
            self.peg_half_sizes = torch.tensor(self.current()[32:35])
            self.box_hole_radii = torch.tensor([float(self.current()[35])])
            return torch.as_tensor(self.current()[None]), 0.0, False, False, {"success": False}

        def close(self):
            pass

    env = Env()
    adapter = Adapter()
    monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(
        peg_orientation_base_dp, "restore_peg_scene",
        lambda *_args, **_kwargs: torch.as_tensor(obs_at(0)[None]),
    )

    summary = peg_orientation_base_dp.evaluate(
        "unused.pt", "unused-scenes", tmp_path / "result", device="cpu",
        max_episode_steps=10, seed=1907, scene_indices=[0], record_phase_trace=True,
        insertion_trace_window=(9, 10),
    )
    record = summary["records"][0]
    captures = record["sampler_input_window"]

    assert adapter.calls == 2
    assert record["phase"]["sampler_calls"] == 2
    assert len(captures) == 1
    capture = captures[0]
    assert capture["preaction_step_index"] == 8
    assert capture["first_completed_control"] == 9
    assert capture["last_completed_control"] == 10
    assert capture["executed_chunk_length"] == 2
    assert capture["scene_index"] == 0
    assert capture["policy_sampling_seed"] == 1907
    assert np.allclose(capture["history_39d"][0], obs_at(7))
    assert np.allclose(capture["history_39d"][1], obs_at(8))
    assert np.allclose(capture["history_39d"], adapter.received[1][0].numpy())
    assert not np.allclose(capture["history_39d"][0], capture["history_39d"][1])
    assert np.allclose(capture["executed_chunk_7d"], adapter.returned[1][0, :2].numpy())
    assert capture["rng"]["torch_cpu"] == adapter.entry_rng[1].tolist()
    assert "torch_cuda" not in capture["rng"]
    assert np.allclose(capture["official_peg_head_in_hole_m"], [8.0, -0.035, 0.001])
    assert capture["current_vs_native_max_abs"]["qpos"] == 0.0
    assert capture["current_vs_native_max_abs"]["tcp_pose"] == 0.0
    assert capture["current_vs_native_max_abs"]["peg_pose"] == 0.0
    assert capture["current_vs_native_max_abs"]["hole_center"] == 0.0
    assert capture["prior_vs_current_max_abs"] > 0.0
    assert record["phase"]["sampler_input_boundaries"] == [[9, 10]]
    motion = record["phase_trace"][8]
    assert motion["control"] == 9
    assert np.allclose(motion["preaction_tcp_pose"], obs_at(8)[18:25])
    assert np.allclose(motion["postaction_tcp_pose"], obs_at(9)[18:25])
    assert np.allclose(motion["tcp_displacement_m"], [1.0, 1.0, 1.0])
    assert np.allclose(motion["peg_center_displacement_m"], [1.0, 1.0, 1.0])

    reproduced = peg_orientation_base_dp.reproduce_sampler_chunk(adapter.get_action, capture, "cpu")
    assert reproduced["matched"] is True
    assert reproduced["max_abs_error"] == 0.0
    assert adapter.calls == 3


def test_validation_phase_command_forwards_insertion_window(tmp_path, monkeypatch):
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    captured = {}

    def fake_evaluate(*_args, **kwargs):
        Path = __import__("pathlib").Path
        Path(kwargs["output_dir"]).mkdir(parents=True)
        captured["kwargs"] = kwargs
        return {
            "records": [{"scene_index": 0, "phase_trace": [{"control": 1}], "phase": {}}],
            "episodes": 1, "successes": 0, "checkpoint": {},
        }

    monkeypatch.setattr(diagnosis, "evaluate", fake_evaluate)
    monkeypatch.setattr(diagnosis, "sha256_file", lambda _path: "frozen")
    diagnosis.run_validation_phase_diagnosis(
        "ckpt.pt", tmp_path / "catalog", tmp_path / "out", scene_indices=[0],
        max_episode_steps=200, seed=1907, device="cpu", insertion_trace_window=(183, 199),
    )
    assert captured["kwargs"]["insertion_trace_window"] == (183, 199)
    assert captured["kwargs"]["record_phase_trace"] is True
    assert captured["kwargs"]["max_episode_steps"] == 200
    identity = json.loads((tmp_path / "out" / "phase_identity.json").read_text())
    assert identity["insertion_trace_window"] == [183, 199]


def test_hole_x_intervention_replaces_eight_translations_and_stops(tmp_path, monkeypatch):
    import pytest
    import gymnasium as gym
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    policy_action = [0.2, -0.3, 0.4, 0.01, 0.02, 0.03, 0.5]
    hole_quat = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
    # +90 deg about Z maps hole +X to world +Y. Identity root and ±0.1 m scale map 5 mm to [0, 0.05, 0].
    expected_xyz = [0.0, 0.05, 0.0]
    expected_root_delta_m = [0.0, 0.005, 0.0]
    held = [
        [0.11, -0.21, 0.31, -0.91],
        [0.12, -0.22, 0.32, -0.92],
        [0.13, -0.23, 0.33, -0.93],
        [0.14, -0.24, 0.34, -0.94],
        [0.15, -0.25, 0.35, -0.95],
        [0.16, -0.26, 0.36, -0.96],
        [0.17, -0.27, 0.37, -0.97],
        [0.18, -0.28, 0.38, -0.98],
    ]
    prefix, count = 10, 8
    peg_pose = [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]
    tcp_pose = [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]
    hole_pose = [1.0, 2.0, 3.0, *hole_quat]
    root_pose = [-0.615, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    head = [-0.035316, -0.003051, 0.001135]
    pose_xy = 0.1419355650506847
    observation = np.zeros(39, dtype=np.float32)

    def write_baseline(path):
        trace = []
        for control in range(1, prefix + count + 1):
            row = {"control": control, "is_grasping": control == prefix}
            if control == prefix:
                row["executed_action_7d"] = list(policy_action)
            if control == prefix + 1:
                row.update({
                    "preaction_peg_pose": peg_pose,
                    "preaction_tcp_pose": tcp_pose,
                    "preaction_box_hole_pose": hole_pose,
                    "preaction_root_pose": root_pose,
                    "preaction_peg_head_in_hole_m": head,
                    "preaction_pose_peg_hole_xy_m": pose_xy,
                })
            if prefix < control <= prefix + count:
                row["executed_action_7d"] = [0.9, 0.9, 0.9, *held[control - prefix - 1]]
            trace.append(row)
        path.write_text(json.dumps({"phase_trace": trace}))

    class Catalog:
        def __init__(self, _path):
            pass

        def __len__(self):
            return 1

        def load(self, _index):
            return (
                {"hole_center_m": [pose_xy, 0.0, 0.1], "target_yaw_rad": 1.57, "seed": 1},
                {},
                observation.copy(),
            )

    class Adapter:
        calls = 0

        def __init__(self, *_args):
            self.metadata = {}

        def get_action(self, _obs):
            Adapter.calls += 1
            return torch.tensor([[policy_action] * 8], dtype=torch.float32)

    def pose(values):
        wrapped = type("Pose", (), {})()
        wrapped.raw_pose = torch.tensor([values], dtype=torch.float64)
        wrapped.p = torch.tensor([values[:3]], dtype=torch.float64)
        wrapped.q = torch.tensor([values[3:7]], dtype=torch.float64)
        return wrapped

    class Config:
        frame = "root_translation:root_aligned_body_rotation"
        use_delta = True
        use_target = False
        pos_lower = -0.1
        pos_upper = 0.1
        rot_lower = -0.1

    def make_env(success_on=None, peg_override=None):
        class Env:
            def __init__(self):
                self.unwrapped = self
                self.steps = 0
                self.stepped = []
                self.agent = type("Agent", (), {})()
                self.agent.tcp = type("Tcp", (), {"pose": pose(tcp_pose)})()
                self.agent.robot = type("Robot", (), {"pose": pose(root_pose)})()
                self.agent.controller = type("Controller", (), {})()
                self.agent.controller.controllers = {"arm": type("Arm", (), {"config": Config()})()}
                self.agent.is_grasping = lambda _peg: torch.tensor([True])
                live_peg = peg_override or peg_pose
                self.peg = type("Peg", (), {"pose": pose(live_peg)})()
                self.peg.pose.p = torch.tensor([[live_peg[0], live_peg[1], live_peg[2]]], dtype=torch.float64)
                self.box_hole_pose = pose(hole_pose)
                self.box_hole_radii = torch.tensor([0.01805])

            def has_peg_inserted(self):
                return torch.tensor([False]), torch.tensor([head], dtype=torch.float64)

            def step(self, action):
                self.steps += 1
                action = np.asarray(action, dtype=np.float64).reshape(-1)
                self.stepped.append(action.copy())
                replaced = sum(np.allclose(row[:3], expected_xyz) for row in self.stepped)
                success = success_on is not None and replaced >= success_on
                return (
                    torch.as_tensor(observation[None]), 0.0, bool(success), False,
                    {"success": bool(success), "peg_head_pos_at_hole": torch.tensor([head])},
                )

            def close(self):
                pass

        return Env()

    def run(env, output):
        monkeypatch.setattr(gym, "make", lambda *_args, **_kwargs: env)
        return diagnosis.run_hole_x_intervention(
            "ckpt.pt", tmp_path / "catalog", output, baseline_path,
            prefix_controls=prefix, intervention_controls=count,
            physical_hole_plus_x_m=0.005, scene_indices=[0], seed=1907,
            device="cpu", max_episode_steps=40,
        )

    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(
        peg_orientation_base_dp, "restore_peg_scene",
        lambda *_args: torch.as_tensor(observation[None]),
    )
    baseline_path = tmp_path / "baseline.json"
    write_baseline(baseline_path)

    Adapter.calls = 0
    env = make_env()
    run(env, tmp_path / "out")
    assert Adapter.calls == 2
    assert env.steps == prefix + count
    assert all(np.allclose(row, policy_action) for row in env.stepped[:prefix])
    for index, row in enumerate(env.stepped[prefix:]):
        assert np.allclose(row[:3], expected_xyz)
        assert np.allclose(row[3:], held[index])
    record = json.loads((tmp_path / "out" / "episode_000.json").read_text())
    replaced = [row for row in record["phase_trace"] if row.get("translation_replaced")]
    assert len(replaced) == count
    assert record["executed_controls"] == prefix + count
    assert record["success"] is False
    assert record["translation_intervention"]["prefix_matched"] is True
    assert record["translation_intervention"]["intervention_controls_executed"] == count
    for index, row in enumerate(replaced):
        assert np.allclose(row["executed_action_7d"][:3], expected_xyz)
        assert np.allclose(row["executed_action_7d"][3:], held[index])
        assert np.allclose(row["baseline_action_7d"], [0.9, 0.9, 0.9, *held[index]])
        assert np.allclose(row["requested_physical_root_delta_m"], expected_root_delta_m)
        assert row["clipped"] is False
        assert len(row["tcp_displacement_m"]) == 3
        assert len(row["peg_center_displacement_m"]) == 3

    Adapter.calls = 0
    success_env = make_env(success_on=3)
    run(success_env, tmp_path / "success")
    assert success_env.steps == prefix + 3
    assert np.allclose(success_env.stepped[-1][:3], expected_xyz)
    success_record = json.loads((tmp_path / "success" / "episode_000.json").read_text())
    assert success_record["success"] is True
    assert success_record["translation_intervention"]["stopped_on_official_success"] is True
    assert success_record["translation_intervention"]["success_label"] == (
        "scripted_diagnostic_not_model_performance"
    )

    Adapter.calls = 0
    mismatch_env = make_env(peg_override=[9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0])
    with pytest.raises(SystemExit) as exc:
        run(mismatch_env, tmp_path / "mismatch")
    assert exc.value.code == 2
    assert mismatch_env.steps == prefix
    assert all(np.allclose(row, policy_action) for row in mismatch_env.stepped)

    opened = {"count": 0}

    def refuse_make(*_args, **_kwargs):
        opened["count"] += 1
        raise AssertionError("environment opened without baseline channels")

    monkeypatch.setattr(gym, "make", refuse_make)
    (tmp_path / "broken.json").write_text(json.dumps({"phase_trace": [{"control": 1}]}))
    with pytest.raises(ValueError, match="baseline"):
        diagnosis.run_hole_x_intervention(
            "ckpt.pt", tmp_path / "catalog", tmp_path / "broken-out", tmp_path / "broken.json",
            prefix_controls=prefix, intervention_controls=count,
            physical_hole_plus_x_m=0.005, scene_indices=[0], seed=1907,
            device="cpu", max_episode_steps=40,
        )
    assert opened["count"] == 0


def test_expert_translation_realization_uses_next_tcp_and_root_frame_scale():
    """Matched expert steps use action t against TCP t to t+1, not the same row."""
    import pytest
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    yaw90 = 0.7071067811865476
    observations = np.zeros((5, 39), dtype=np.float64)
    observations[1, 0:3] = (9.0, 0.0, 0.0)
    observations[:, 18:21] = [
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.004),
        (1.0, 2.001, 3.004),
        (1.0, 2.003, 3.004),
        (2.0, 2.003, 3.004),
    ]
    observations[1, 25:32] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    observations[1, 32] = 0.1
    observations[1, 35] = 0.02
    observations[1, 36:39] = (0.12, 0.0, 0.0)
    observations[[0, 2, 3, 4], 25:28] = (1.0, 1.0, 1.0)
    observations[[0, 2, 3, 4], 28] = 1.0
    observations[[0, 2, 3, 4], 32] = 0.1
    observations[[0, 2, 3, 4], 35] = 0.02
    actions = np.array([
        [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.05, 0.0, 0.0, 0.0, 0.0, -1.0],
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -0.8],
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ], dtype=np.float64)
    root_poses = np.zeros((5, 7), dtype=np.float64)
    root_poses[:, 3] = 1.0
    root_poses[2, 3:7] = (yaw90, 0.0, 0.0, yaw90)
    box_quats = np.zeros((5, 4), dtype=np.float64)
    box_quats[:, 0] = 1.0
    episodes = [
        {"episode_id": 7, "actions": actions, "observations": observations,
         "root_poses": root_poses, "box_quats": box_quats},
        {"episode_id": 8, "actions": np.array([[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]),
         "observations": np.zeros((2, 39)),
         "root_poses": np.array([[0, 0, 0, 1, 0, 0, 0.0], [0, 0, 0, 1, 0, 0, 0.0]]),
         "box_quats": np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])},
    ]
    result = diagnosis.summarize_expert_translation_realization(
        episodes, intervention_tcp_max_m=0.000843,
    )
    negative = [row for row in result["rows"] if row["gripper_request_negative"]]
    rotated = next(row for row in negative if row["action_index"] == 2)
    assert np.allclose(rotated["physical_world_delta_m"], [0.0, 0.005, 0.0], atol=1e-12)
    assert np.allclose(rotated["tcp_displacement_m"], [0.0, 0.002, 0.0], atol=1e-12)
    assert rotated["projection_m"] == pytest.approx(0.002, abs=1e-12)
    assert rotated["operational_phase"] == "at_or_after_close_request"
    identity_row = next(row for row in negative if row["action_index"] == 1)
    assert np.allclose(identity_row["physical_root_delta_m"], [0.0, 0.005, 0.0], atol=1e-12)
    assert identity_row["tcp_norm_m"] == pytest.approx(0.001, abs=1e-12)
    assert identity_row["insertion_proximal"] is True
    positive = [row for row in result["rows"] if not row["gripper_request_negative"]]
    assert len(positive) == 1 and positive[0]["tcp_norm_m"] == pytest.approx(0.004, abs=1e-12)
    assert positive[0]["operational_phase"] == "pre_close_request"
    stratum = result["strata"]["gripper_request_negative"]
    assert stratum["count"] == 2 and stratum["episodes"] == 1
    assert stratum["tcp_norm_m"] == pytest.approx({
        "q00": 0.001, "q10": 0.0011, "q25": 0.00125, "q50": 0.0015,
        "q75": 0.00175, "q90": 0.0019, "q100": 0.002,
    }, abs=1e-12)
    assert result["decision"]["classification"] == "unusual_low"
    assert result["decision"]["assumption_5mm_implies_5mm_tcp"] == "refuted"
    assert result["native_grasp"] == "unavailable"
    assert all(row["tcp_norm_m"] < 0.1 for row in result["rows"])
    empty = diagnosis.summarize_expert_translation_realization(
        episodes[1:], intervention_tcp_max_m=0.000843,
    )
    assert empty["strata"]["gripper_request_negative"]["count"] == 0
    assert empty["strata"]["gripper_request_negative"]["tcp_norm_m"] is None
    assert empty["decision"]["classification"] == "unavailable_empty_stratum"


def test_proximal_expert_requests_select_preaction_and_project_hole_advance():
    """Proximal requests use action t, not the terminal state and not the post-state label."""
    import pytest
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    yaw90 = 0.7071067811865476
    observations = np.zeros((4, 39), dtype=np.float64)
    observations[:, 28] = 1.0
    observations[:, 32] = 0.1
    observations[:, 35] = 0.02
    observations[0, 25:28] = (1.0, 1.0, 1.0)
    observations[1, 25:28] = (-0.1, -0.025, 0.0)
    observations[2, 25:28] = (-0.1, -0.010, 0.0)
    observations[3, 25:28] = (-0.1, -0.030, 0.0)
    observations[1, 18:21] = (1.0, 2.0, 3.0)
    observations[2, 18:21] = (1.0, 2.002, 3.001)
    actions = np.array([
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.05, 0.0, 0.1, 0.2, 0.0, -0.5],
        [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
    ], dtype=np.float64)
    poses = np.zeros((4, 7), dtype=np.float64)
    poses[:, 3] = 1.0
    box_quats = np.zeros((4, 4), dtype=np.float64)
    box_quats[:, 3] = yaw90
    box_quats[:, 0] = yaw90
    selected = diagnosis.summarize_proximal_expert_requests([
        {"episode_id": 3, "actions": actions, "observations": observations,
         "root_poses": poses, "box_quats": box_quats},
    ])
    assert [row["action_index"] for row in selected["rows"]] == [1]
    assert selected["counts"]["all_states_proximal"] == 2
    assert selected["counts"]["terminal_only_proximal"] == 1
    assert selected["counts"]["actionable_proximal"] == 1
    assert selected["counts"]["all_states_inserted"] == 1
    row = selected["rows"][0]
    assert row["requested_hole_plus_x_m"] == pytest.approx(0.005, abs=1e-12)
    assert row["realized_head_x_advance_m"] == pytest.approx(0.015, abs=1e-12)
    assert row["tcp_projection_m"] == pytest.approx(0.002, abs=1e-12)
    assert selected["decision"]["classification"] == "comparable"
    far = np.vstack([observations[0], observations[0]])
    empty = diagnosis.summarize_proximal_expert_requests([
        {"episode_id": 4, "actions": actions[:1], "observations": far,
         "root_poses": poses[:2], "box_quats": box_quats[:2]},
    ])
    assert empty["counts"]["actionable_proximal"] == 0
    assert empty["counts"]["terminal_only_proximal"] == 0
    assert empty["request_norm"] is None
    assert empty["decision"]["classification"] == "unavailable_empty"


def test_proximal_policy_first_action_uses_history_and_executed_index():
    """Teacher-forced comparison uses observation t-1,t and the first executed action."""
    import pytest
    from examples.baselines.constrain import peg_orientation_base_diagnosis as diagnosis

    yaw90 = 0.7071067811865476
    identity = np.zeros((2, 7), dtype=np.float64)
    identity[:, 3] = 1.0
    identity_box = np.zeros((2, 4), dtype=np.float64)
    identity_box[:, 0] = 1.0
    early = np.zeros((2, 39), dtype=np.float64)
    early[:, 0] = 4.0
    early[:, 28] = 1.0
    early[:, 32] = 0.1
    early[:, 35] = 0.02
    early[:, 25] = -0.125
    later = np.zeros((3, 39), dtype=np.float64)
    later[:, 28] = 1.0
    later[:, 32] = 0.1
    later[:, 35] = 0.02
    later[0, 0] = 90.0
    later[0, 25:28] = (1.0, 1.0, 1.0)
    later[1, 0] = 91.0
    later[1, 25:28] = (-0.1, -0.025, 0.0)
    later[2, 0] = 92.0
    later[2, 25:28] = (-0.1, -0.025, 0.0)
    rotated_box = np.zeros((3, 4), dtype=np.float64)
    rotated_box[:, 0] = yaw90
    rotated_box[:, 3] = yaw90
    episodes = [
        {
            "episode_id": 9,
            "actions": np.array([
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.40, 0.0, 0.0, 0.0, 0.0, -1.0],
            ]),
            "observations": later,
            "root_poses": np.vstack((identity, identity[:1])),
            "box_quats": rotated_box,
        },
        {
            "episode_id": 4,
            "actions": np.array([[0.30, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]),
            "observations": early,
            "root_poses": identity,
            "box_quats": identity_box,
        },
    ]
    seen = []

    def get_action(batch):
        array = batch.detach().cpu().numpy()
        seen.append(array.copy())
        out = torch.zeros(array.shape[0], 8, 7)
        out[:, 1, 0] = 0.77
        out[:, 0, 0] = 5.0
        if array.shape[0] >= 1 and array[0, 1, 0] == 4.0:
            out[0, 0, 0] = 0.03125
        if array.shape[0] >= 2 and array[1, 1, 0] == 91.0:
            out[1, 0, 0] = 0.0625
        return out

    result = diagnosis.compare_proximal_policy_first_actions(
        episodes, get_action, seed=1907, batch_size=4,
    )
    assert len(seen) == 1 and seen[0].shape == (4, 2, 39)
    assert np.allclose(seen[0][0, 0], seen[0][0, 1]) and seen[0][0, 1, 0] == 4.0
    assert seen[0][1, 0, 0] == 90.0 and seen[0][1, 1, 0] == 91.0
    assert np.allclose(seen[0][2], seen[0][1]) and np.allclose(seen[0][3], seen[0][1])
    assert result["counts"]["actionable_proximal"] == 2
    assert result["counts"]["policy_norm_lt_window_max"] == 1
    assert result["counts"]["policy_norm_le_intervention"] == 1
    assert result["counts"]["policy_norm_lt_expert"] == 2
    assert result["counts"]["policy_norm_le_expert"] == 2
    assert result["counts"]["policy_norm_gt_expert"] == 0
    assert result["counts"]["padded_rows_discarded"] == 2
    assert result["policy_translation_norm"]["q50"] == pytest.approx(0.046875, abs=1e-12)
    assert result["paired_translation_error_norm"]["q00"] == pytest.approx(0.26875, abs=1e-12)
    assert result["paired_translation_error_norm"]["q100"] == pytest.approx(0.4048533685175412, abs=1e-12)
    assert result["norm_ratio_policy_over_expert"]["q50"] == pytest.approx(0.13020833333333334, abs=1e-12)
    assert result["policy_hole_plus_x_m"]["q00"] == pytest.approx(0.0, abs=1e-12)
    assert result["policy_hole_plus_x_m"]["q100"] == pytest.approx(0.003125, abs=1e-12)
    assert result["expert_hole_plus_x_m"]["q50"] == pytest.approx(0.035, abs=1e-12)
    assert result["hole_plus_x_error_m"]["q50"] == pytest.approx(-0.0334375, abs=1e-12)
    assert result["decision"]["classification"] == "inconclusive"
    assert result["decision"]["equal_to_either_threshold_is"] == "inconclusive"
    assert result["rotation_analyzed"] is False
    assert result["gripper_sign_is_not_native_grasp"] is True


def test_resolved_official_peg_training_matches_args_and_baselines_line():
    import sys

    cfg = peg_orientation_base_dp.resolved_official_peg_training()
    sys.path.insert(0, str(peg_orientation_base_dp.OFFICIAL_DP))
    from train import Args

    official = Args()
    assert cfg["seed"] == official.seed == 1
    assert cfg["batch_size"] == official.batch_size == 1024
    assert official.total_iters == 1_000_000
    assert cfg["total_iters"] == 100_000
    assert cfg["lr"] == official.lr == 1e-4
    assert cfg["betas"] == (0.95, 0.999)
    assert cfg["weight_decay"] == 1e-6
    assert cfg["warmup_steps"] == 500
    assert cfg["scheduler"] == "cosine"
    assert cfg["ema_power"] == 0.75
    assert cfg["obs_horizon"] == official.obs_horizon == 2
    assert cfg["act_horizon"] == official.act_horizon == 8
    assert cfg["pred_horizon"] == official.pred_horizon == 16
    assert cfg["unet_dims"] == list(official.unet_dims) == [64, 128, 256]
    assert cfg["diffusion_step_embed_dim"] == official.diffusion_step_embed_dim == 64
    assert cfg["n_groups"] == official.n_groups == 8
    assert cfg["num_diffusion_iters"] == 100
    assert cfg["beta_schedule"] == "squaredcos_cap_v2"
    assert cfg["prediction_type"] == "epsilon"
    assert cfg["clip_sample"] is True
    assert cfg["normalization"] == "identity"
    assert cfg["num_dataload_workers"] == official.num_dataload_workers == 0
    assert cfg["log_freq"] == official.log_freq == 1000
    assert cfg["max_episode_steps"] == 300
    assert cfg["env_time_limit"] == 300
    assert cfg["checkpoint_every"] == 5000
    assert cfg["method_exceptions"]["num_demos"] == 1000
    assert cfg["method_exceptions"]["official_num_demos"] == 100
    assert cfg["method_exceptions"]["observation_dim"] == 39
    assert cfg["method_exceptions"]["official_observation_dim"] == 43
    assert cfg["method_exceptions"]["evaluation"] == "frozen_validation_50"
    assert cfg["method_exceptions"]["checkpoint_policy"] == "save_every_5000_not_official_best_only"
    assert cfg["method_exceptions"]["selection"] == "max_native_success_tie_latest"
    assert cfg["method_exceptions"]["inference_seed_rule"] == "1 + scene_index"


def test_training_batches_match_official_iteration_sampler():
    import sys

    from torch.utils.data import BatchSampler, RandomSampler

    sys.path.insert(0, str(peg_orientation_base_dp.OFFICIAL_DP))
    from diffusion_policy.utils import IterationBasedBatchSampler

    class _Len:
        def __len__(self):
            return 23

    def oracle():
        sampler = RandomSampler(_Len(), replacement=False)
        batches = BatchSampler(sampler, batch_size=8, drop_last=True)
        return list(IterationBasedBatchSampler(batches, 5))

    torch.manual_seed(1)
    expected = oracle()
    torch.manual_seed(1)
    actual = list(peg_orientation_base_dp.official_iteration_batches(23, 8, 5))
    assert actual == expected
    assert len(actual) == 5
    assert all(len(batch) == 8 for batch in actual)


def test_train_rejects_non_official_budget(tmp_path):
    import argparse

    args = argparse.Namespace(
        updates=30000, seed=2026111102, batch_size=256, checkpoint_every=5000,
        data=tmp_path, manifest=tmp_path, ckpt_dir=tmp_path, log_dir=tmp_path,
        smoke_only=True,
    )
    try:
        peg_orientation_base_dp.train(args)
    except ValueError as err:
        assert "official Peg" in str(err)
    else:
        raise AssertionError("non-official budget was accepted")


def test_official_eval_caps_are_300_and_final_chunk_is_partial(tmp_path, monkeypatch):
    class Catalog:
        def __init__(self, _):
            pass

        def __len__(self):
            return 1

        def load(self, _):
            return ({"hole_center_m": np.zeros(3), "target_yaw_rad": 1.57, "seed": 7}, {}, None)

    class Adapter:
        def __init__(self, *_args):
            self.metadata = {}

        def get_action(self, _obs):
            return torch.zeros((1, 8, 7))

    class Env:
        def __init__(self):
            self.unwrapped = self
            self.steps = 0
            self.peg = type("Peg", (), {})()
            self.peg.pose = type("Pose", (), {})()
            self.peg.pose.p = torch.tensor([[0.3, 0.0, 0.0]])
            self.peg.pose.q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        def step(self, _action):
            self.steps += 1
            return torch.zeros((1, 39)), 0.0, False, False, {"success": False}

        def close(self):
            pass

    env = Env()
    import gymnasium as gym

    monkeypatch.setattr(
        gym, "make",
        lambda *_args, **kwargs: (setattr(env, "configured_budget", kwargs["max_episode_steps"]) or env),
    )
    monkeypatch.setattr(peg_orientation_base_dp, "PegSceneCatalog", Catalog)
    monkeypatch.setattr(peg_orientation_base_dp, "PegBaseAdapter", Adapter)
    monkeypatch.setattr(peg_orientation_base_dp, "restore_peg_scene", lambda *_args: torch.zeros((1, 39)))
    caps = peg_orientation_base_dp.resolved_official_peg_training()

    peg_orientation_base_dp.evaluate(
        "unused.pt", "unused-scenes", tmp_path / "result", device="cpu",
        max_episode_steps=caps["max_episode_steps"], seed=caps["seed"],
        env_time_limit=caps["env_time_limit"],
    )
    record = json.loads((tmp_path / "result/episode_000.json").read_text())
    summary = json.loads((tmp_path / "result/summary.json").read_text())
    assert env.configured_budget == 300
    assert record["executed_controls"] == 300
    assert record["executed_chunk_lengths"][0] == 8
    assert record["executed_chunk_lengths"][-1] == 4
    assert sum(record["executed_chunk_lengths"]) == 300
    assert summary["evaluation_settings"]["env_time_limit"] == 300
    assert record["policy_sampling_seed"] == 1


def test_selection_keeps_latest_tie_and_retention_needs_forty():
    chosen = peg_orientation_base_dp.select_checkpoint([
        {"update": 5000, "successes": 10},
        {"update": 10000, "successes": 12},
        {"update": 15000, "successes": 12},
    ])
    assert chosen["update"] == 15000
    assert peg_orientation_base_dp.retention_authorized(39) is False
    assert peg_orientation_base_dp.retention_authorized(40) is True
