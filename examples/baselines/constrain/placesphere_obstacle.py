"""PlaceSphere obstacle pilot environment; native task geometry and evaluate are retained."""

from typing import Optional

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.place_sphere import PlaceSphereEnv
from mani_skill.envs.utils.system.backend import CPU_SIM_BACKENDS, sim_backend_name_mapping
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

ENV_ID = "PlaceSphereObstacle-v1"
OBSTACLE_KEY = "placesphere_obstacle"
GOAL_KEY = "placesphere_goal"
PARKED_CENTER = (-10.0, -10.0)


def segment_enters_forbidden_disk(start, end, center, radius):
    start, end, center = (np.asarray(value, dtype=np.float64).reshape(2)
                          for value in (start, end, center))
    direction = end - start
    t = np.clip(np.dot(center - start, direction) / max(np.dot(direction, direction), 1e-24), 0, 1)
    return bool(np.linalg.norm(start + t * direction - center) < radius)


@register_env(ENV_ID, max_episode_steps=500)
class PlaceSphereObstacleEnv(PlaceSphereEnv):
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(self, *args, robot_uids="panda", obstacle_radius=0.03,
                 obstacle_height=0.25, safety_margin=0.005,
                 contact_impulse_threshold=1e-6, obstacle_enabled=False,
                 obstacle_mode="physical", **kwargs):
        backend = kwargs.get("sim_backend", "auto")
        if kwargs.get("num_envs", 1) != 1:
            raise NotImplementedError("PlaceSphere obstacle checks require one CPU environment")
        if backend == "auto":
            kwargs["sim_backend"] = "physx_cpu"
        elif sim_backend_name_mapping.get(backend, backend) not in CPU_SIM_BACKENDS:
            raise NotImplementedError("PlaceSphere obstacle checks require a CPU backend")
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_height = float(obstacle_height)
        self.safety_margin = float(safety_margin)
        self.contact_impulse_threshold = float(contact_impulse_threshold)
        self.default_obstacle_enabled = bool(obstacle_enabled)
        if obstacle_mode not in ("physical", "render_only"):
            raise ValueError("obstacle_mode must be 'physical' or 'render_only'")
        self.obstacle_mode = obstacle_mode
        self._obstacle_rotation = euler2quat(0, np.pi / 2, 0)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        if self.gpu_sim_enabled or self.num_envs != 1:
            raise NotImplementedError("PlaceSphere obstacle checks require single-env PhysX CPU")

    def _load_scene(self, options):
        super()._load_scene(options)
        self.obstacle = actors.build_cylinder(
            self.scene, radius=self.obstacle_radius, half_length=self.obstacle_height / 2,
            color=[0.1, 0.2, 0.9, 1], name="placesphere_obstacle", body_type="static",
            add_collision=self.obstacle_mode == "physical",
            initial_pose=sapien.Pose(p=[*PARKED_CENTER, self.obstacle_height / 2],
                                     q=self._obstacle_rotation))
        self.obstacle_collision_radius = (float(self.obstacle._bodies[0].collision_shapes[0].radius)
                                          if self.obstacle_mode == "physical" else self.obstacle_radius)
        self.obstacle_collision_height = (float(self.obstacle._bodies[0].collision_shapes[0].half_length * 2)
                                          if self.obstacle_mode == "physical" else self.obstacle_height)
        self._obstacle_center = torch.tensor(PARKED_CENTER, device=self.device)[None]
        self._obstacle_enabled = torch.zeros(1, dtype=torch.bool, device=self.device)
        self._obstacle_collision = torch.zeros_like(self._obstacle_enabled)
        self._forbidden_crossing = torch.zeros_like(self._obstacle_enabled)
        self._grasp_started = torch.zeros_like(self._obstacle_enabled)
        self._first_obstacle_contact = None
        self._prev_sphere_xy = self.obj.pose.p[:, :2].clone()
        self._forbidden_radius = self.obstacle_collision_radius + float(self.radius) + self.safety_margin
        self._obstacle_target_entities = None

    def _set_obstacle_pose(self, center):
        center = torch.as_tensor(center, dtype=torch.float32, device=self.device).reshape(2)
        self._obstacle_center[0] = center
        self.obstacle._objs[0].pose = sapien.Pose(
            p=[float(center[0]), float(center[1]), self.obstacle_collision_height / 2],
            q=self._obstacle_rotation)

    def _initialize_episode(self, env_idx, options):
        super()._initialize_episode(env_idx, options)
        for key, actor in (("sphere_xy", self.obj), ("container_xy", self.bin)):
            if options.get(key) is not None:
                xy = torch.as_tensor(options[key], dtype=torch.float32, device=self.device).reshape(1, 2)
                pose = actor.pose.raw_pose.clone()
                pose[:, :2] = xy
                actor.set_pose(Pose.create(pose))
        enabled = bool(options.get("obstacle_enabled", self.default_obstacle_enabled))
        center = options.get("obstacle_center", (0.0, 0.20) if enabled else PARKED_CENTER)
        self._obstacle_enabled[:] = enabled
        self._set_obstacle_pose(center if enabled else PARKED_CENTER)
        self._prev_sphere_xy[:] = self.obj.pose.p[:, :2]

    def _clear_sim_state(self):
        super()._clear_sim_state()
        self._obstacle_collision[:] = False
        self._forbidden_crossing[:] = False
        self._grasp_started[:] = False
        self._first_obstacle_contact = None

    def get_state_dict(self):
        state = super().get_state_dict()
        state[OBSTACLE_KEY] = torch.cat((self._obstacle_center,
            torch.tensor([[self.obstacle_collision_radius]], device=self.device),
            self._obstacle_enabled.float()[:, None], self._obstacle_collision.float()[:, None],
            self._forbidden_crossing.float()[:, None], self._grasp_started.float()[:, None],
            self._prev_sphere_xy), dim=1)
        state[GOAL_KEY] = self.bin.pose.p.clone()
        return state

    def get_full_state_dict(self):
        """Snapshot environment state and the controller runtime state explicitly."""
        return {"environment": self.get_state_dict(),
                "controller": self.agent.controller.get_state()}

    def set_full_state_dict(self, state):
        """Restore the complete simulation/controller snapshot used by pilots."""
        if set(state) != {"environment", "controller"}:
            raise ValueError("full PlaceSphere state requires environment and controller fields")
        self.set_state_dict(state["environment"])
        self.agent.controller.set_state(state["controller"])

    def set_state_dict(self, state: dict, env_idx: Optional[torch.Tensor] = None):
        extra, goal = state.get(OBSTACLE_KEY), state.get(GOAL_KEY)
        native = {key: value for key, value in state.items() if key not in (OBSTACLE_KEY, GOAL_KEY)}
        super().set_state_dict(native, env_idx)
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        if goal is not None:
            self.bin.set_pose(Pose.create_from_pq(common.to_tensor(goal, device=self.device), self.bin.pose.q))
        if extra is not None:
            extra = common.to_tensor(extra, device=self.device).reshape(-1, 9)
            if not torch.allclose(extra[:, 2], torch.full_like(extra[:, 2], self.obstacle_collision_radius), atol=1e-6, rtol=0):
                raise ValueError("restored obstacle radius differs from constructed geometry")
            self._obstacle_enabled[env_idx] = extra[:, 3].bool()
            self._set_obstacle_pose(extra[0, :2] if bool(extra[0, 3]) else PARKED_CENTER)
            self._obstacle_collision[env_idx] = extra[:, 4].bool()
            self._forbidden_crossing[env_idx] = extra[:, 5].bool()
            self._grasp_started[env_idx] = extra[:, 6].bool()
            self._prev_sphere_xy[env_idx] = extra[:, 7:9]

    def reset(self, *args, **kwargs):
        result = super().reset(*args, **kwargs)
        self._prev_sphere_xy[:] = self.obj.pose.p[:, :2]
        return result

    def _get_obs_extra(self, info):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs[OBSTACLE_KEY] = torch.cat((self._obstacle_center,
                torch.full((1, 1), self.obstacle_collision_radius, device=self.device)), dim=1)
        return obs

    def _check_contacts(self):
        if not bool(self._obstacle_enabled[0]):
            return
        obstacle = self.obstacle._objs[0]
        if self._obstacle_target_entities is None:
            self._obstacle_target_entities = [link._bodies[0].entity for link in self.agent.robot.links]
            self._obstacle_target_entities.append(self.obj._bodies[0].entity)
        for contact in self.scene.get_contacts():
            bodies = [body.entity for body in contact.bodies]
            if obstacle not in bodies:
                continue
            other = bodies[1] if bodies[0] is obstacle else bodies[0]
            if other in self._obstacle_target_entities and any(
                np.linalg.norm(np.asarray(point.impulse)) > self.contact_impulse_threshold
                for point in contact.points):
                self._obstacle_collision[:] = True
                self._first_obstacle_contact = {
                    "other": next((link.name for link in self.agent.robot.links
                                   if link._bodies[0].entity is other), "sphere"),
                    "max_impulse": max(float(np.linalg.norm(np.asarray(point.impulse)))
                                       for point in contact.points),
                }
                return

    def _check_sphere_segment(self):
        if bool(self._obstacle_enabled[0]):
            self._grasp_started |= self.agent.is_grasping(self.obj)
            current = self.obj.pose.p[:, :2]
            direction = current - self._prev_sphere_xy
            length_sq = (direction * direction).sum(-1).clamp_min(1e-12)
            t = (((self._obstacle_center - self._prev_sphere_xy) * direction).sum(-1) / length_sq).clamp(0, 1)
            nearest = self._prev_sphere_xy + t[:, None] * direction
            hit = torch.linalg.norm(nearest - self._obstacle_center, dim=-1) < self._forbidden_radius
            self._forbidden_crossing |= self._grasp_started & hit
        self._prev_sphere_xy = self.obj.pose.p[:, :2].clone()

    def _after_simulation_step(self):
        self._check_contacts()
        self._check_sphere_segment()

    def evaluate(self):
        info = super().evaluate()
        info["native_success"] = info["success"].clone()
        info["obstacle_collision"] = self._obstacle_collision.clone()
        info["obstacle_crossing"] = self._forbidden_crossing.clone()
        info["obstacle_enabled"] = self._obstacle_enabled.clone()
        if self.obstacle_mode == "physical":
            info["success"] = info["native_success"] & ~self._obstacle_collision & ~self._forbidden_crossing
        else:
            info["success"] = info["native_success"]
        return info
