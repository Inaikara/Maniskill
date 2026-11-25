"""Yaw-conditioned PlugCharger experiment environment."""

import numpy as np
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill.envs.tasks.tabletop.plug_charger import PlugChargerEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

ENV_ID = "PlugChargerOrientation-v1"
RANDOM_ENV_ID = "PlugChargerRandomOrientation-v1"
TARGET_YAW = 0.0
YAW_HALF_RANGE = np.pi / 8
STATE_KEY = "plug_orientation_target"


def goal_pose_components(position, yaw):
    return np.asarray(position), np.asarray(euler2quat(0, 0, yaw))


def world_x_yaw(quaternion):
    axis = quat2mat(np.asarray(quaternion))[:, 0]
    return float(np.arctan2(axis[1], axis[0]))


def build_orientation_hint(charger_yaw, target_yaw):
    error = np.arctan2(np.sin(np.asarray(charger_yaw) - target_yaw),
                       np.cos(np.asarray(charger_yaw) - target_yaw))
    return np.stack((np.sin(error), np.cos(error)), axis=-1)


def choose_target_yaw(options, random_target, rng):
    if "target_yaw" in options:
        return float(options["target_yaw"])
    if random_target:
        return float(rng.uniform(-YAW_HALF_RANGE, YAW_HALF_RANGE))
    return TARGET_YAW


def filter_orientation_observation(obs):
    obs.pop("receptacle_pose", None)
    obs.pop("goal_pose", None)
    return obs


def local_geometry_observation(device):
    return {
        "charger_base_half_size": torch.tensor([[0.02, 0.015, 0.012]], device=device),
        "charger_peg_half_size": torch.tensor([[0.008, 0.00075, 0.0032]], device=device),
        "charger_peg_gap": torch.tensor([[0.007]], device=device),
        "receptacle_half_size": torch.tensor([[0.01, 0.05, 0.05]], device=device),
        "receptacle_clearance": torch.tensor([[0.0005]], device=device),
    }


class _PlugOrientationEnv(PlugChargerEnv):
    random_target_yaw = False

    def __init__(self, *args, **kwargs):
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("orientation-conditioned plug tasks support one environment")
        kwargs.setdefault("robot_uids", "panda_wristcam")
        super().__init__(*args, **kwargs)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        options = options or {}
        self.target_yaw = choose_target_yaw(
            options, self.random_target_yaw, self._episode_rng
        )
        # Rotate the actual receptacle while preserving its sampled world center.
        receptacle_position = self.receptacle.pose.p.clone()
        receptacle_yaw = self.target_yaw - np.pi
        receptacle_quat = torch.as_tensor(
            euler2quat(0, 0, receptacle_yaw), dtype=receptacle_position.dtype,
            device=self.device,
        ).expand(self.num_envs, -1)
        self.receptacle.set_pose(Pose.create_from_pq(receptacle_position, receptacle_quat))
        goal_quat = torch.as_tensor(
            euler2quat(0, 0, self.target_yaw), dtype=receptacle_position.dtype,
            device=self.device,
        ).expand(self.num_envs, -1)
        self.goal_pose = Pose.create_from_pq(receptacle_position, goal_quat)

    def get_orientation_hint(self):
        charger_q = self.charger.pose.q[0].detach().cpu().numpy()
        charger_yaw = world_x_yaw(charger_q)
        return build_orientation_hint(charger_yaw, self.target_yaw)

    def _get_obs_extra(self, info):
        obs = filter_orientation_observation(super()._get_obs_extra(info))
        if self.obs_mode_struct.use_state:
            obs["goal_position"] = self.goal_pose.p
            obs.update(local_geometry_observation(self.device))
        return obs

    def get_state_dict(self):
        state = super().get_state_dict()
        target = torch.full((self.num_envs, 1), self.target_yaw, device=self.device)
        state[STATE_KEY] = torch.cat((target, self.goal_pose.p), dim=1)
        return state

    def set_state_dict(self, state, env_idx=None):
        orientation_state = state.get(STATE_KEY)
        native = {key: value for key, value in state.items() if key != STATE_KEY}
        super().set_state_dict(native, env_idx)
        if orientation_state is not None:
            values = torch.as_tensor(orientation_state, device=self.device)
            if values.ndim == 1:
                values = values.unsqueeze(0)
            if values.ndim != 2 or values.shape[1] != 4:
                raise ValueError(f"{STATE_KEY} must have shape (4,) or (N, 4), got {tuple(values.shape)}")
            self.target_yaw = float(values[0, 0].item())
            quat = euler2quat(0, 0, self.target_yaw)
            self.goal_pose = Pose.create_from_pq(values[:, 1:], quat)


@register_env(ENV_ID, max_episode_steps=200)
class PlugChargerOrientationEnv(_PlugOrientationEnv):
    pass


@register_env(RANDOM_ENV_ID, max_episode_steps=200)
class PlugChargerRandomOrientationEnv(_PlugOrientationEnv):
    random_target_yaw = True
