"""Tests for the DP base policy adapter for the obstacle environment.

Tests are written at the adapter seam:
- the queued stacked 51d obstacle flat obs is trimmed to the base 48d (the
  first 48 dims match the base StackCube flat obs; the appended obstacle
  features map nowhere),
- 4d (xyz + gripper) base actions map to the 7d pd_ee_delta_pose action by
  appending three zero rotations,
- epsilon and get_action at the same base flat obs, torch noise stream and
  diffusion timesteps are numerically equivalent to a fresh official Agent
  build at the exact trained channels (nothing passed silently),
- the base weight source (checkpoint + ema_agent state) is recorded, and
  mismatched weights are strictly rejected.

Uses deterministic inputs and the actual trained base checkpoint.
"""

import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from examples.baselines.constrain.base_dp import (
    DEFAULT_CKPT,
    BaseDPAdapter,
    build_official_agent,
)
from examples.baselines.constrain.stackcube_obstacle import OBSTACLE_STATE_KEY

OFFICIAL_DP_DIR = Path(__file__).resolve().parents[3] / "examples" / "baselines" / "diffusion_policy"
if str(OFFICIAL_DP_DIR) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_DP_DIR))


def make_vec_env(env_id="StackCubeObstacle-v1", control_mode="pd_ee_delta_pose", max_steps=500, **ek):
    import gymnasium as gym

    import examples.baselines.constrain.stackcube_obstacle  # noqa: F401 (registers env)
    from mani_skill.utils.wrappers import CPUGymWrapper, FrameStack

    def env_fn():
        def thunk():
            env = gym.make(
                env_id,
                reconfiguration_freq=1,
                control_mode=control_mode,
                reward_mode="sparse",
                obs_mode="state",
                render_mode="rgb_array",
                human_render_camera_configs=dict(shader_pack="default"),
                max_episode_steps=max_steps,
                **ek,
            )
            env = FrameStack(env, num_stack=2)
            env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
            env.action_space.seed(0)
            env.observation_space.seed(0)
            return env

        return thunk

    return gym.vector.SyncVectorEnv([env_fn()])


def reset_stacked_obs(env, seed=0):
    obs, _ = env.reset(seed=seed)
    return torch.as_tensor(np.asarray(obs)[0], dtype=torch.float32)


def test_queued_51d_trim_to_48d_matches_base_flat_obs():
    """For an identical copied physical state, the adapter's condition input is
    the identical base flat obs sequence: the obstacle env state dict (without
    the obstacle key) copied into a plain StackCube-v1 env produces the same
    robotic/cube flat obs (the naming region). The queued 2x51 frame block is
    the FrameStack(num_stack=2) reset layout used at the rollout seam."""
    obstacle_env = gym.make("StackCubeObstacle-v1", reconfiguration_freq=1, robot_uids="panda_wristcam")
    obstacle_env.reset(seed=11)
    base_env = gym.make("StackCube-v1", reconfiguration_freq=1)
    base_env.reset(seed=11)
    state = obstacle_env.unwrapped.get_state_dict()
    state.pop(OBSTACLE_STATE_KEY)
    base_env.unwrapped.set_state_dict(state)
    frame = torch.as_tensor(
        np.asarray(obstacle_env.unwrapped.get_obs()).flatten(), dtype=torch.float32
    )
    frame_base = torch.as_tensor(
        np.asarray(base_env.unwrapped.get_obs()).flatten(), dtype=torch.float32
    )
    assert frame.shape == (51,)
    assert frame_base.shape == (48,)
    torch.testing.assert_close(frame[:48], frame_base)
    queued = torch.stack([frame, frame])  # FrameStack reset repeats the first frame
    adapter = BaseDPAdapter(device="cpu")
    cond = adapter.base_cond(queued)
    assert cond.shape == (1, 96)
    torch.testing.assert_close(cond, torch.stack([frame_base, frame_base]).flatten()[None])


def test_appended_obstacle_features_map_nowhere():
    """The appended (x, y, radius) obstacle features never enter the condition
    for a queued 2-frame history, even when each frame carries different
    appended values."""
    adapter = BaseDPAdapter(device="cpu")
    frame_a = torch.tensor(
        [1.0 + i for i in range(48)] + [0.9, 0.25, 0.03], dtype=torch.float32
    )
    frame_b = torch.tensor(
        [2.0 + i for i in range(48)] + [-0.7, 0.02, 0.04], dtype=torch.float32
    )
    cond = adapter.base_cond(torch.stack([frame_a, frame_b]))
    for frame in (frame_a, frame_b):
        for v in frame[48:]:
            assert not torch.any(torch.isclose(cond, v))


def test_base_cond_rejects_one_frame_history():
    """One-frame history without the queued 2-frame layout is rejected."""
    adapter = BaseDPAdapter(device="cpu")
    with pytest.raises(ValueError):
        adapter.base_cond(torch.zeros(1, 51))


def test_base_cond_rejects_wrong_layout():
    """Frames without the 51d obstacle flat layout are rejected up front."""
    adapter = BaseDPAdapter(device="cpu")
    with pytest.raises(ValueError):
        adapter.base_cond(torch.zeros(2, 50))
    with pytest.raises(ValueError):
        adapter.base_cond(torch.zeros(2, 52))


def test_map_action_4d_to_7d_appends_zero_rotations():
    """A base 4d (xyz, gripper) sequence maps to the 7d ee_delta_pose action with
    three zero rotations appended."""
    adapter = BaseDPAdapter(device="cpu")
    base_seq = torch.tensor(
        [0.1 * (i % 40) for i in range(8 * 4)], dtype=torch.float32
    ).reshape(1, 8, 4)
    executed = adapter.map_action_ee_delta_pose(base_seq)
    assert executed.shape == (1, 8, 7)
    torch.testing.assert_close(executed[..., :3], base_seq[..., :3])
    torch.testing.assert_close(executed[..., 3:6], torch.zeros(1, 8, 3))
    torch.testing.assert_close(executed[..., 6], base_seq[..., 3])


def test_executed_slice_semantics():
    """The adapter's executed slicing keeps obs_horizon-1 : obs_horizon-1+act_horizon."""
    adapter = BaseDPAdapter(device="cpu")
    pred_seq = torch.arange(16 * 4, dtype=torch.float32).reshape(1, 16, 4)
    expected = pred_seq[:, 1:9]
    adapter_executed = adapter.select_executed(pred_seq)
    assert adapter_executed.shape == (1, 8, 4)
    torch.testing.assert_close(adapter_executed, expected)


def test_default_config_loads_actual_base_checkpoint():
    """The default base weight source (best_eval_success_at_end ema_agent) loads
    and the source is recorded in the adapter metadata."""
    adapter = BaseDPAdapter(device="cpu")
    metadata = adapter.metadata
    assert Path(metadata["weights_source"]) == Path(DEFAULT_CKPT)
    assert metadata["weights_subkey"] == "ema_agent"
    assert metadata["act_dim"] == 4 and metadata["base_obs_dim"] == 48


def load_official_agent_from_disk(weights_subkey="ema_agent"):
    """Build an independent official Agent and load it straight from the
    checkpoint on disk (never via the adapter's own state)."""
    official = build_official_agent()
    ck = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)
    official.load_state_dict(ck[weights_subkey])
    official.eval()
    return official


def test_default_ckpt_is_ema_weights():
    """The adapter's default weights subkey is the EMA copy in the checkpoint."""
    ck = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)
    adapter = BaseDPAdapter(device="cpu")
    adapter_sd = adapter.net_state_dict()
    for k, v in adapter_sd.items():
        torch.testing.assert_close(v, ck["ema_agent"][k])


def test_get_action_matches_official_agent():
    """The adapter's get_action executed rollout matches an independent
    official Agent loaded from the checkpoint on disk under the same torch
    noise stream, flat obs and diffusion steps."""
    adapter = BaseDPAdapter(device="cpu")
    official = load_official_agent_from_disk()
    obs_seq = torch.zeros(1, 2, 48)
    torch.manual_seed(3)
    official_executed = official.get_action(obs_seq)
    torch.manual_seed(3)
    executed = adapter.get_action(obs_seq)
    assert executed.shape == (1, 8, 4)
    torch.testing.assert_close(executed, official_executed)


def test_epsilon_matches_official_agent():
    """At the same base condition, noisy action and diffusion timestep the
    epsilon output is numerically identical to the official DP Agent loaded
    independently from the checkpoint on disk."""
    adapter = BaseDPAdapter(device="cpu")
    official = load_official_agent_from_disk()
    torch.manual_seed(7)
    noisy_seq = torch.randn(1, 16, 4)
    torch.manual_seed(11)
    cond = torch.randn(1, 96)
    timestep = 77
    official_epsilon = official.noise_pred_net(noisy_seq, timestep, global_cond=cond)
    adapter_epsilon = adapter.noise_pred_net(noisy_seq, timestep, global_cond=cond)
    torch.testing.assert_close(adapter_epsilon, official_epsilon)


def test_mismatched_weights_strictly_rejected():
    """A corrupted base weight tensor fails the adapter build instead of silent
    ignore."""
    ck = torch.load(DEFAULT_CKPT, map_location="cpu", weights_only=False)
    sd = dict(ck["ema_agent"])
    key = next(iter(sd))
    sd[key] = torch.zeros((3,) + tuple(sd[key].shape[1:]))
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "bad.pt"
        torch.save({"agent": sd, "ema_agent": sd}, bad)
        with pytest.raises(RuntimeError):
            BaseDPAdapter(checkpoint=bad, device="cpu")
