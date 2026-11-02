"""PickCube obstacle experiment environment.

The native PickCube cube, target, action mapping, and placement/static success
condition are retained. ``extra.obstacle`` and the ``pickcube_obstacle`` state
observation field is ``[center_x, center_y, collision_radius]`` in world meters;
the state field additionally stores enable/violation/grasp/path-tracker values
needed for continuation. There are no task-specific flattened state indexes.
``cube_xy`` and ``goal_pos`` reset options select the cube start and native 3-D
target. Obstacle checks are only enabled on single-environment PhysX CPU runs;
a disabled cylinder is parked at (-10, -10) and safety checks do not run.
"""

from typing import Optional

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS, sim_backend_name_mapping
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

ENV_ID = "PickCubeObstacle-v1"
OBSTACLE_STATE_KEY = "pickcube_obstacle"
GOAL_STATE_KEY = "pickcube_goal"
PARKED_CENTER = (-10.0, -10.0)


@register_env(ENV_ID, max_episode_steps=500)
class PickCubeObstacleEnv(PickCubeEnv):
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        obstacle_radius=0.03,
        obstacle_height=0.25,
        safety_margin=0.005,
        contact_impulse_threshold=1e-6,
        obstacle_enabled=True,
        obstacle_mode="physical",
        **kwargs,
    ):
        self._reject_unsupported_backend(kwargs)
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_height = float(obstacle_height)
        self.safety_margin = float(safety_margin)
        self.contact_impulse_threshold = float(contact_impulse_threshold)
        self.default_obstacle_enabled = bool(obstacle_enabled)
        if obstacle_mode not in ("physical", "render_only"):
            raise ValueError("obstacle_mode must be 'physical' or 'render_only'")
        self.obstacle_mode = obstacle_mode
        self._obstacle_rotation = euler2quat(0.0, np.pi / 2.0, 0.0)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        if self.gpu_sim_enabled or self.num_envs != 1:
            raise NotImplementedError("PickCubeObstacle-v1 requires single-env PhysX CPU")

    @staticmethod
    def _reject_unsupported_backend(kwargs):
        backend = kwargs.get("sim_backend", "auto")
        if kwargs.get("num_envs", 1) != 1:
            raise NotImplementedError("PickCubeObstacle-v1 requires num_envs=1")
        if backend == "auto":
            kwargs["sim_backend"] = "physx_cpu"
        elif sim_backend_name_mapping.get(backend, backend) not in CPU_SIM_BACKENDS:
            raise NotImplementedError("PickCubeObstacle-v1 requires a PhysX CPU backend")

    def _load_scene(self, options):
        super()._load_scene(options)
        self.obstacle = actors.build_cylinder(
            self.scene,
            radius=self.obstacle_radius,
            half_length=self.obstacle_height / 2,
            color=[0.1, 0.2, 0.9, 1],
            name="pickcube_obstacle",
            body_type="static",
            add_collision=self.obstacle_mode == "physical",
            initial_pose=sapien.Pose(
                p=[*PARKED_CENTER, self.obstacle_height / 2], q=self._obstacle_rotation
            ),
        )
        self.obstacle_collision_radius = (float(self.obstacle._bodies[0].collision_shapes[0].radius)
                                          if self.obstacle_mode == "physical" else self.obstacle_radius)
        self._obstacle_center = torch.tensor(PARKED_CENTER, device=self.device)[None]
        self._obstacle_enabled = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._obstacle_collision = torch.zeros_like(self._obstacle_enabled)
        self._forbidden_crossing = torch.zeros_like(self._obstacle_enabled)
        self._grasp_started = torch.zeros_like(self._obstacle_enabled)
        self._first_obstacle_contact = None
        self._prev_cube_xy = self.cube.pose.p[:, :2].clone()
        self._cube_xy_radius = float(np.sqrt(2) * self.cube_half_size)
        self._forbidden_radius = (
            self.obstacle_collision_radius + self._cube_xy_radius + self.safety_margin
        )

    def _set_obstacle_pose(self, center):
        center = torch.as_tensor(center, dtype=torch.float32, device=self.device).reshape(2)
        self._obstacle_center[0] = center
        self.obstacle._objs[0].pose = sapien.Pose(
            p=[float(center[0]), float(center[1]), self.obstacle_height / 2],
            q=self._obstacle_rotation,
        )

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        if options.get("cube_xy") is not None:
            xy = torch.as_tensor(options["cube_xy"], dtype=torch.float32, device=self.device).reshape(1, 2)
            pose = self.cube.pose.raw_pose.clone()
            pose[:, :2] = xy
            self.cube.set_pose(Pose.create(pose))
        if options.get("goal_pos") is not None:
            goal = torch.as_tensor(options["goal_pos"], dtype=torch.float32, device=self.device).reshape(1, 3)
            self.goal_site.set_pose(Pose.create_from_pq(goal))
        enabled = bool(options.get("obstacle_enabled", self.default_obstacle_enabled))
        center = options.get("obstacle_center", PARKED_CENTER if not enabled else (0.0, 0.2))
        if not enabled:
            center = PARKED_CENTER
        self._obstacle_enabled[:] = enabled
        self._set_obstacle_pose(center)
        self._prev_cube_xy[:] = self.cube.pose.p[:, :2]

    def _clear_sim_state(self):
        super()._clear_sim_state()
        self._obstacle_collision[:] = False
        self._forbidden_crossing[:] = False
        self._grasp_started[:] = False
        self._first_obstacle_contact = None

    def _obstacle_state(self):
        radius = torch.full((1, 1), self.obstacle_collision_radius, device=self.device)
        return torch.cat([self._obstacle_center, radius], dim=1)

    def get_state_dict(self):
        state = super().get_state_dict()
        state[OBSTACLE_STATE_KEY] = torch.cat(
            [
                self._obstacle_state(),
                self._obstacle_enabled.float()[:, None],
                self._obstacle_collision.float()[:, None],
                self._forbidden_crossing.float()[:, None],
                self._grasp_started.float()[:, None],
                self._prev_cube_xy,
            ],
            dim=1,
        )
        state[GOAL_STATE_KEY] = self.goal_site.pose.p.clone()
        return state

    def set_state_dict(self, state: dict, env_idx: Optional[torch.Tensor] = None):
        extra = state.get(OBSTACLE_STATE_KEY)
        goal = state.get(GOAL_STATE_KEY)
        if extra is not None:
            extra = common.to_tensor(extra, device=self.device).reshape(-1, 9)
            if not torch.allclose(extra[:, 2], torch.full_like(extra[:, 2], self.obstacle_collision_radius), atol=1e-6, rtol=0):
                raise ValueError("Restored obstacle radius does not match constructed geometry")
        native_state = {k: v for k, v in state.items() if k not in (OBSTACLE_STATE_KEY, GOAL_STATE_KEY)}
        super().set_state_dict(native_state, env_idx)
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        if goal is not None:
            goal = common.to_tensor(goal, device=self.device)
            self.goal_site.set_pose(Pose.create_from_pq(goal, self.goal_site.pose.q))
        if extra is not None:
            self._obstacle_enabled[env_idx] = extra[:, 3].bool()
            self._set_obstacle_pose(extra[0, :2] if bool(extra[0, 3]) else PARKED_CENTER)
            self._obstacle_collision[env_idx] = extra[:, 4].bool()
            self._forbidden_crossing[env_idx] = extra[:, 5].bool()
            self._grasp_started[env_idx] = extra[:, 6].bool()
            self._prev_cube_xy[env_idx] = extra[:, 7:9]
        else:
            self._prev_cube_xy[:] = self.cube.pose.p[:, :2]

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._prev_cube_xy[:] = self.cube.pose.p[:, :2]
        return result

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs[OBSTACLE_STATE_KEY] = self._obstacle_state()
        return obs

    def _check_obstacle_contacts(self):
        if not bool(self._obstacle_enabled[0]):
            return
        obstacle_entity = self.obstacle._objs[0]
        targets = [link._bodies[0].entity for link in self.agent.robot.links]
        targets.append(self.cube._bodies[0].entity)
        for contact in self.scene.get_contacts():
            bodies = [body.entity for body in contact.bodies]
            if obstacle_entity not in bodies:
                continue
            other = bodies[1] if bodies[0] is obstacle_entity else bodies[0]
            impulses = [float(np.linalg.norm(np.asarray(point.impulse))) for point in contact.points]
            if any(other is target for target in targets) and any(
                impulse > self.contact_impulse_threshold for impulse in impulses
            ):
                if self._first_obstacle_contact is None:
                    if other is self.cube._bodies[0].entity:
                        other_name = "cube"
                    else:
                        other_name = next(link.name for link in self.agent.robot.links
                                          if other is link._bodies[0].entity)
                    self._first_obstacle_contact = {
                        "other": other_name, "max_impulse": max(impulses),
                    }
                self._obstacle_collision[:] = True
                return

    def _check_forbidden_crossing(self):
        cube_xy = self.cube.pose.p[:, :2]
        if bool(self._obstacle_enabled[0]):
            self._grasp_started |= self.agent.is_grasping(self.cube)
            direction = cube_xy - self._prev_cube_xy
            length_sq = (direction * direction).sum(dim=-1).clamp_min(1e-12)
            t = (((self._obstacle_center - self._prev_cube_xy) * direction).sum(dim=-1) / length_sq).clamp(0, 1)
            nearest = self._prev_cube_xy + t[:, None] * direction
            distance = torch.linalg.norm(nearest - self._obstacle_center, dim=-1)
            self._forbidden_crossing |= self._grasp_started & (distance < self._forbidden_radius)
        self._prev_cube_xy = cube_xy.clone()

    def _after_simulation_step(self):
        self._check_obstacle_contacts()
        self._check_forbidden_crossing()

    def evaluate(self):
        info = super().evaluate()
        native_success = info["success"]
        info["native_success"] = native_success
        info["obstacle_collision"] = self._obstacle_collision.clone()
        info["obstacle_crossing"] = self._forbidden_crossing.clone()
        info["obstacle_enabled"] = self._obstacle_enabled.clone()
        if self.obstacle_mode == "physical":
            info["success"] = native_success & ~self._obstacle_collision & ~self._forbidden_crossing
        else:
            info["success"] = native_success
        return info
