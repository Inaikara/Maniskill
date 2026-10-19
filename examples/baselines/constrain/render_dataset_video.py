"""Render every recorded episode into one dataset.mp4, without re-execution."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm.auto import tqdm

from examples.baselines.constrain import stackcube_obstacle  # noqa: F401
from mani_skill.trajectory.utils import index_dict
from mani_skill.utils import common


def dataset_trajectory(dataset_dir):
    directory = Path(dataset_dir).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    preferred = directory / "trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    if preferred.exists():
        return preferred
    converted = sorted(directory.glob("trajectory.state.*.physx_cpu.h5"))
    if len(converted) == 1:
        return converted[0]
    if len(converted) > 1:
        raise ValueError("multiple CPU conversions found; cannot select one dataset")
    source = directory / "trajectory.h5"
    if not source.exists():
        raise FileNotFoundError(f"no trajectory dataset in {directory}")
    return source


def render_dataset(dataset_dir):
    trajectory_path = dataset_trajectory(dataset_dir)
    metadata = json.loads(trajectory_path.with_suffix(".json").read_text())
    episodes = sorted(metadata["episodes"], key=lambda episode: episode["episode_id"])
    if not episodes:
        raise ValueError("dataset has no episodes")
    env_info = metadata["env_info"]
    kwargs = dict(env_info["env_kwargs"])
    kwargs.update(sim_backend="physx_cpu", num_envs=1, obs_mode="none", render_mode="rgb_array")
    output = trajectory_path.parent / "dataset.mp4"
    temporary = output.with_name(".dataset.tmp.mp4")
    env = gym.make(env_info["env_id"], **kwargs)
    try:
        with h5py.File(trajectory_path, "r") as trajectories:
            total_frames = sum(
                len(trajectories[f"traj_{episode['episode_id']}/actions"]) + 1
                for episode in episodes
            )
            with imageio.get_writer(
                str(temporary), fps=env.unwrapped.control_freq,
                codec="libx264", macro_block_size=1,
            ) as writer, tqdm(total=total_frames, desc="Rendering", unit="frame") as progress:
                for episode in episodes:
                    reset_kwargs = dict(episode["reset_kwargs"])
                    options = dict(reset_kwargs.get("options") or {})
                    options["reconfigure"] = True
                    reset_kwargs["options"] = options
                    env.reset(**reset_kwargs)
                    trajectory = trajectories[f"traj_{episode['episode_id']}"]
                    states = trajectory["env_states"]
                    frame_count = len(trajectory["actions"]) + 1
                    for frame_index in range(frame_count):
                        state = index_dict(states, frame_index)
                        current_radius = getattr(env.unwrapped, "obstacle_collision_radius", None)
                        if current_radius is not None and "obstacle" in state:
                            recorded_radius = float(np.asarray(state["obstacle"]).reshape(-1, 3)[0, 2])
                            if not np.isclose(current_radius, recorded_radius, atol=1e-6, rtol=0.0):
                                options["obstacle_radius"] = recorded_radius
                                env.reset(**{**reset_kwargs, "options": options})
                        env.unwrapped.set_state_dict(state)
                        frame = common.to_numpy(env.render())
                        if frame.ndim == 4:
                            frame = frame[0]
                        writer.append_data(np.asarray(frame, dtype=np.uint8))
                        progress.update(1)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        env.close()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args(argv)
    render_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
