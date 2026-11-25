"""PegInsertionSide yaw-conditioned environment and shared observation helpers."""

import numpy as np
import torch
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

ENV_ID = "PegInsertionSideOrientation-v1"
RANDOM_ENV_ID = "PegInsertionSideRandomOrientation-v1"
OBSERVATION_FIELDS = (
    "tcp_pose", "peg_pose", "peg_half_size", "hole_center", "box_hole_radius"
)
TARGET_YAW = np.pi / 2
YAW_HALF_RANGE = np.pi / 8
STATE_KEY = "peg_orientation_target"


def build_orientation_hint(peg_yaw, target_yaw):
    """Encode wrapped world-Z yaw error: peg yaw minus target yaw, in radians."""
    error = np.arctan2(np.sin(np.asarray(peg_yaw) - target_yaw),
                       np.cos(np.asarray(peg_yaw) - target_yaw))
    return np.stack((np.sin(error), np.cos(error)), axis=-1)


class _PegOrientationEnv(PegInsertionSideEnv):
    random_target_yaw = False

    def __init__(self, *args, **kwargs):
        if kwargs.get("num_envs", 1) != 1:
            raise ValueError("orientation-conditioned peg tasks support a single environment")
        kwargs.setdefault("control_mode", "pd_ee_delta_pose")
        super().__init__(*args, **kwargs)

    def get_orientation_hint(self):
        """Return [sin, cos] of current peg world +X yaw minus box world +X yaw."""
        peg_q = self.peg.pose.q[0].detach().cpu().numpy()
        box_q = self.box.pose.q[0].detach().cpu().numpy()
        peg_axis = quat2mat(peg_q)[:, 0]
        box_axis = quat2mat(box_q)[:, 0]
        peg_yaw = np.arctan2(peg_axis[1], peg_axis[0])
        target_yaw = np.arctan2(box_axis[1], box_axis[0])
        return build_orientation_hint(peg_yaw, target_yaw)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        options = options or {}
        hole_center = self.box_hole_pose.p.clone()
        if "target_yaw" in options:
            target_yaw = float(options["target_yaw"])
        elif self.random_target_yaw:
            target_yaw = float(self._episode_rng.uniform(
                TARGET_YAW - YAW_HALF_RANGE, TARGET_YAW + YAW_HALF_RANGE
            ))
        else:
            target_yaw = TARGET_YAW
        self.target_yaw = target_yaw
        quat = torch.as_tensor(euler2quat(0, 0, target_yaw), dtype=torch.float32,
                               device=self.device).expand(self.num_envs, -1)
        # Preserve the sampled world hole center while changing box orientation.
        rotated_offset = Pose.create_from_pq(q=quat) * self.box_hole_offsets
        box_position = hole_center - rotated_offset.p
        self.box.set_pose(Pose.create_from_pq(box_position, quat))

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        if self.obs_mode_struct.use_state:
            obs["hole_center"] = self.box_hole_pose.p
            obs.pop("box_hole_pose", None)
        return obs

    def get_state_dict(self):
        state = super().get_state_dict()
        yaw = torch.full((self.num_envs, 1), self.target_yaw, device=self.device)
        state[STATE_KEY] = torch.cat((yaw, self.box_hole_pose.p), dim=1)
        return state

    def set_state_dict(self, state, env_idx=None):
        orientation_state = state.get(STATE_KEY)
        native = {key: value for key, value in state.items() if key != STATE_KEY}
        super().set_state_dict(native, env_idx)
        if orientation_state is not None:
            if env_idx is None:
                env_idx = torch.arange(self.num_envs, device=self.device)
            orientation_state = torch.as_tensor(orientation_state, device=self.device)
            self.target_yaw = float(orientation_state[0, 0])
            quat = torch.as_tensor(euler2quat(0, 0, self.target_yaw), device=self.device,
                                   dtype=torch.float32).expand(self.num_envs, -1)
            center = orientation_state[:, 1:]
            offset = (Pose.create_from_pq(q=quat) * self.box_hole_offsets).p
            box_position = center - offset
            self.box.set_pose(Pose.create_from_pq(box_position, quat))


@register_env(ENV_ID, max_episode_steps=100)
class PegInsertionSideOrientationEnv(_PegOrientationEnv):
    pass


@register_env(RANDOM_ENV_ID, max_episode_steps=100)
class PegInsertionSideRandomOrientationEnv(_PegOrientationEnv):
    random_target_yaw = True
