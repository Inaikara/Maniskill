import json

import h5py
import numpy as np

from examples.baselines.constrain import render_dataset_video as video


def test_selects_converted_dataset_without_duplicate_experts(tmp_path):
    source = tmp_path / "trajectory.h5"
    converted = tmp_path / "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    source.touch()
    assert video.dataset_trajectory(tmp_path) == source
    converted.touch()
    assert video.dataset_trajectory(tmp_path) == converted


def test_streams_all_episodes_into_one_video_in_numeric_order(tmp_path, monkeypatch):
    source = tmp_path / "trajectory.h5"
    with h5py.File(source, "w") as handle:
        for episode_id, values in ((2, [20, 21, 22]), (10, [100, 101])):
            group = handle.create_group(f"traj_{episode_id}")
            group.create_dataset("actions", data=np.zeros((len(values) - 1, 4)))
            group.create_dataset("env_states/value", data=values)
    source.with_suffix(".json").write_text(json.dumps({
        "env_info": {"env_id": "TestEnv", "env_kwargs": {}},
        "episodes": [
            {"episode_id": 10, "reset_kwargs": {"seed": 10}},
            {"episode_id": 2, "reset_kwargs": {"seed": 2}},
        ],
    }))
    frames, resets, writer_calls = [], [], []

    class Env:
        control_freq = 20

        @property
        def unwrapped(self):
            return self

        def reset(self, **kwargs):
            resets.append(kwargs)

        def set_state_dict(self, state):
            self.value = state["value"]

        def render(self):
            return np.full((1, 2, 2, 3), self.value, dtype=np.uint8)

        def close(self):
            pass

    class Writer:
        def __init__(self, path, **kwargs):
            writer_calls.append((path, kwargs))
            open(path, "wb").close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, frame):
            frames.append(int(frame[0, 0, 0]))

    monkeypatch.setattr(video.gym, "make", lambda *args, **kwargs: Env())
    monkeypatch.setattr(video.imageio, "get_writer", Writer)
    output = video.render_dataset(tmp_path)
    assert output == tmp_path / "dataset.mp4"
    assert len(writer_calls) == 1
    assert frames == [20, 21, 22, 100, 101]
    assert [reset["seed"] for reset in resets] == [2, 10]
    assert all(reset["options"]["reconfigure"] for reset in resets)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "dataset.mp4", "trajectory.h5", "trajectory.json",
    ]


def test_restores_recorded_obstacle_geometry_when_metadata_differs(tmp_path, monkeypatch):
    source = tmp_path / "trajectory.h5"
    with h5py.File(source, "w") as handle:
        group = handle.create_group("traj_0")
        group.create_dataset("actions", data=np.zeros((2, 4)))
        group.create_dataset("env_states/obstacle", data=[
            [0.0, 0.2, 0.025], [0.0, 0.2, 0.03], [0.0, 0.2, 0.03],
        ])
    source.with_suffix(".json").write_text(json.dumps({
        "env_info": {"env_id": "TestEnv", "env_kwargs": {}},
        "episodes": [{"episode_id": 0, "reset_kwargs": {
            "seed": 1, "options": {"obstacle_radius": 0.02},
        }}],
    }))
    restored_radii = []

    class Env:
        control_freq = 20
        obstacle_collision_radius = 0.02

        @property
        def unwrapped(self):
            return self

        def reset(self, **kwargs):
            self.obstacle_collision_radius = kwargs["options"]["obstacle_radius"]

        def set_state_dict(self, state):
            radius = float(state["obstacle"][2])
            assert np.isclose(self.obstacle_collision_radius, radius, atol=1e-6)
            restored_radii.append(radius)

        def render(self):
            return np.zeros((1, 2, 2, 3), dtype=np.uint8)

        def close(self):
            pass

    class Writer:
        def __init__(self, path, **kwargs):
            open(path, "wb").close()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, frame):
            pass

    monkeypatch.setattr(video.gym, "make", lambda *args, **kwargs: Env())
    monkeypatch.setattr(video.imageio, "get_writer", Writer)
    video.render_dataset(tmp_path)
    assert restored_radii == [0.025, 0.03, 0.03]
