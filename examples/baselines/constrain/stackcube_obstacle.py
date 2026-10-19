from typing import Optional

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.envs.utils.system.backend import (
    CPU_SIM_BACKENDS,
    sim_backend_name_mapping,
)
from mani_skill.utils import common
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

OBSTACLE_STATE_KEY = "obstacle"
ENV_ID = "StackCubeObstacle-v1"


@register_env(ENV_ID, max_episode_steps=50)
class StackCubeObstacleEnv(StackCubeEnv):
    """StackCube with a single vertical cylindrical obstacle at the tabletop.

    The obstacle is a real static cylinder (fixed height, configurable radius). Its
    center XY and real radius are appended to the environment state dictionary and to
    the state observation, after the original StackCube fields. Contact between the
    robot links / gripper / carried cube and the cylinder is detected per physics
    substep, and the XY path of the carried cube is checked against a 2D forbidden
    disk from the moment the cube is grasped. Both checks are sticky until reset.

    The per-substep contact/segment checks only run on the PhysX CPU backend for a
    single environment. GPU simulation and vectorization are rejected up front so a
    silently skipped check can never produce an apparently-valid trajectory.
    """

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam"]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        obstacle_radius: float = 0.03,
        obstacle_height: float = 0.25,
        obstacle_center=(0.0, 0.25),
        safety_margin: float = 0.005,
        contact_impulse_threshold: float = 1e-6,
        **kwargs,
    ):
        self._reject_unsupported_backend(kwargs)
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_height = float(obstacle_height)
        self._obstacle_center_init = np.asarray(obstacle_center, dtype=np.float64).reshape(2)
        self.safety_margin = float(safety_margin)
        self.contact_impulse_threshold = float(contact_impulse_threshold)
        # SAPIEN builds cylinders along their local X axis, so a 90 degree
        # rotation about Y is required for a *vertical* (world-z) obstacle.
        # Without it the "cylinder" lies horizontally and the vertical-axis
        # forbidden-disk model is meaningless.
        self._obstacle_rotation = euler2quat(0.0, np.pi / 2.0, 0.0)
        self._last_cube_layout = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        if self.gpu_sim_enabled or self.num_envs != 1:
            raise NotImplementedError(
                "StackCubeObstacle-v1 requires CPU simulation with a single "
                "environment; GPU/multi-env results would skip the per-substep checks."
            )

    @staticmethod
    def _reject_unsupported_backend(kwargs: dict):
        sim_backend = kwargs.get("sim_backend", "auto")
        num_envs = kwargs.get("num_envs", 1)
        if num_envs != 1:
            raise NotImplementedError(
                "StackCubeObstacle-v1 only supports num_envs=1; vectorized "
                "simulation would skip the per-substep validity checks."
            )
        if sim_backend == "auto":
            kwargs["sim_backend"] = "physx_cpu"
        else:
            resolved = sim_backend_name_mapping.get(sim_backend, sim_backend)
            if resolved not in CPU_SIM_BACKENDS:
                raise NotImplementedError(
                    "StackCubeObstacle-v1 only supports the CPU simulation backend; "
                    f"got sim_backend={sim_backend!r}."
                )

    def obstacle_config(self) -> dict:
        """Deterministic, unit-explicit description of the *current* obstacle.

        This is the geometry-restoration contract written into the original JSON
        metadata: rebuilding with these values reproduces the exact collision and
        visual geometry referenced by the ``obstacle`` state fields. The centre is
        read from the live per-episode state (``_obstacle_center``), not from the
        construction-time default, so a reset that moves the cylinder is recorded
        exactly.
        """
        center = common.to_numpy(self._obstacle_center[0]).reshape(2)
        center_xy = [float(center[0]), float(center[1])]
        center_z = float(self.obstacle_collision_half_height)
        world_pose = [
            float(center[0]),
            float(center[1]),
            center_z,
            *[float(v) for v in self._obstacle_rotation],
        ]
        return dict(
            radius=float(self.obstacle_collision_radius),
            visual_radius=float(self.obstacle_visual_radius),
            height=float(self.obstacle_height),
            center_z=center_z,
            center=center_xy,
            world_pose=world_pose,
            safety_margin=float(self.safety_margin),
            cube_xy_radius=float(self._cube_xy_radius),
            forbidden_radius=float(self._forbidden_radius),
            state_field_order=["center_x", "center_y", "radius"],
            units="meters",
            control_mode=str(self.control_mode),
            cube_layout=self._last_cube_layout,
            geometry_restore=dict(
                type="cylinder",
                radius=float(self.obstacle_collision_radius),
                half_length=float(self.obstacle_collision_half_height),
                initial_pose=[
                    float(center[0]),
                    float(center[1]),
                    center_z,
                ],
                initial_rotation=[float(v) for v in self._obstacle_rotation],
                body_type="static",
            ),
        )

    def _load_scene(self, options: dict):
        # Explicit radius on reconfigure lets a collector sample the real radius
        # per episode while reusing the same task/env.
        if options and options.get("obstacle_radius") is not None:
            self.obstacle_radius = float(options["obstacle_radius"])
        super()._load_scene(options)
        self.obstacle_half_height = self.obstacle_height / 2
        self.obstacle = actors.build_cylinder(
            self.scene,
            radius=self.obstacle_radius,
            half_length=self.obstacle_half_height,
            color=[0.1, 0.2, 0.9, 1],
            name="obstacle",
            body_type="static",
            initial_pose=sapien.Pose(
                p=[
                    self._obstacle_center_init[0],
                    self._obstacle_center_init[1],
                    self.obstacle_half_height,
                ],
                q=self._obstacle_rotation,
            ),
        )
        # Real built geometry (read back from the physics/render shapes).
        self.obstacle_collision_radius = float(
            self.obstacle._bodies[0].collision_shapes[0].radius
        )
        render_body = self.obstacle._objs[0].find_component_by_type(
            sapien.render.RenderBodyComponent
        )
        self.obstacle_visual_radius = float(render_body.render_shapes[0].radius)
        self.obstacle_collision_half_height = float(
            self.obstacle._bodies[0].collision_shapes[0].half_length
        )

        self._obstacle_center = common.to_tensor(
            self._obstacle_center_init, device=self.device
        )[None].repeat(self.num_envs, 1)
        self._obstacle_collision = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._forbidden_crossing = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._grasp_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_cube_xy = self.cubeA.pose.p[:, :2].clone()
        self._cube_xy_radius = float(
            torch.linalg.norm(self.cube_half_size[:2]).item()
        )
        self._forbidden_radius = (
            self.obstacle_collision_radius + self._cube_xy_radius + self.safety_margin
        )
        self._obstacle_target_entities = None

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        if "cube_layout" in options and options["cube_layout"] is not None:
            self._apply_cube_layout(env_idx, options["cube_layout"])
        if "obstacle_center" in options and options["obstacle_center"] is not None:
            center = common.to_tensor(options["obstacle_center"], device=self.device)
            center = center.reshape(-1, 2)
            if center.shape[0] == 1 and len(env_idx) > 1:
                center = center.repeat(len(env_idx), 1)
            self._obstacle_center[env_idx] = center
        if not self.gpu_sim_enabled:
            for i in common.to_numpy(env_idx):
                c = self._obstacle_center[int(i)]
                self.obstacle._objs[int(i)].pose = sapien.Pose(
                    p=[float(c[0]), float(c[1]), self.obstacle_half_height],
                    q=self._obstacle_rotation,
                )
        self._prev_cube_xy[env_idx] = self.cubeA.pose.p[env_idx][:, :2]

    def _apply_cube_layout(self, env_idx: torch.Tensor, layout: dict):
        """Place both cubes explicitly from a recorded blocking layout.

        The layout is stored on the environment so it can be written to the
        original JSON metadata and used to replay the exact initial state.
        """
        for cube, xy_key, yaw_key in (
            (self.cubeA, "cubeA_xy", "cubeA_yaw"),
            (self.cubeB, "cubeB_xy", "cubeB_yaw"),
        ):
            xy = np.asarray(layout[xy_key], dtype=np.float64).reshape(2)
            yaw = float(layout[yaw_key])
            position = torch.tensor(
                [xy[0], xy[1], float(self.cube_half_size[2])],
                dtype=torch.float32,
                device=self.device,
            )
            quat = torch.tensor(
                euler2quat(0.0, 0.0, yaw), dtype=torch.float32, device=self.device
            )
            cube.set_pose(Pose.create_from_pq(p=position[None], q=quat[None]))
        self._last_cube_layout = {
            "cubeA_xy": [float(v) for v in np.asarray(layout["cubeA_xy"]).reshape(2)],
            "cubeB_xy": [float(v) for v in np.asarray(layout["cubeB_xy"]).reshape(2)],
            "cubeA_yaw": float(layout["cubeA_yaw"]),
            "cubeB_yaw": float(layout["cubeB_yaw"]),
        }

    def _clear_sim_state(self):
        super()._clear_sim_state()
        self._obstacle_collision[:] = False
        self._forbidden_crossing[:] = False
        self._grasp_started[:] = False

    def _obstacle_state(self) -> torch.Tensor:
        radius = torch.full(
            (self.num_envs, 1),
            self.obstacle_collision_radius,
            device=self.device,
        )
        return torch.cat([self._obstacle_center, radius], dim=1)

    def get_state_dict(self):
        state = super().get_state_dict()
        state[OBSTACLE_STATE_KEY] = self._obstacle_state()
        return state

    def set_state_dict(self, state: dict, env_idx: Optional[torch.Tensor] = None):
        obstacle_state = None
        if OBSTACLE_STATE_KEY in state:
            obstacle_state = common.to_tensor(
                state[OBSTACLE_STATE_KEY], device=self.device
            ).reshape(-1, 3)
            self._validate_obstacle_radius(obstacle_state[:, 2])
        super().set_state_dict(state, env_idx)
        if obstacle_state is None:
            return
        if env_idx is None:
            env_idx = torch.arange(self.num_envs, device=self.device)
        self._obstacle_center[env_idx] = obstacle_state[:, :2]
        if not self.gpu_sim_enabled:
            for i in common.to_numpy(env_idx):
                c = self._obstacle_center[int(i)]
                self.obstacle._objs[int(i)].pose = sapien.Pose(
                    p=[float(c[0]), float(c[1]), self.obstacle_half_height],
                    q=self._obstacle_rotation,
                )

    def _validate_obstacle_radius(self, radius: torch.Tensor):
        expected = self.obstacle_collision_radius
        radius = radius.to(torch.float64)
        if not torch.allclose(
            radius, torch.full_like(radius, expected), atol=1e-6, rtol=0.0
        ):
            raise ValueError(
                "State obstacle radius does not match the constructed geometry "
                f"(expected {expected}, got {radius.tolist()}); rebuild the "
                "environment with the recorded obstacle radius instead."
            )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        # ``reset_to_env_states`` skips ``_initialize_episode``, so the segment
        # tracker must be rebased on every reset to avoid bridging the previous
        # episode's cube position into the restored one.
        self._prev_cube_xy[:] = self.cubeA.pose.p[:, :2]
        return obs, info

    def _get_obs_extra(self, info: dict):
        obs = super()._get_obs_extra(info)
        if "state" in self.obs_mode:
            obs[OBSTACLE_STATE_KEY] = self._obstacle_state()
        return obs

    def _obstacle_targets(self):
        if self._obstacle_target_entities is None:
            self._obstacle_target_entities = [
                link._bodies[0].entity for link in self.agent.robot.links
            ]
            self._obstacle_target_entities.append(self.cubeA._bodies[0].entity)
        return self._obstacle_target_entities

    def _check_obstacle_contacts(self):
        obstacle_entity = self.obstacle._objs[0]
        targets = self._obstacle_targets()
        hit = False
        for contact in self.scene.get_contacts():
            body0 = contact.bodies[0].entity
            body1 = contact.bodies[1].entity
            if obstacle_entity is body0:
                other = body1
            elif obstacle_entity is body1:
                other = body0
            else:
                continue
            if not any(other is target for target in targets):
                continue
            for point in contact.points:
                if float(np.linalg.norm(np.asarray(point.impulse))) > self.contact_impulse_threshold:
                    hit = True
                    break
            if hit:
                break
        if hit:
            self._obstacle_collision[:] = True

    @staticmethod
    def _segment_point_distance(point, start, end):
        direction = end - start
        length_sq = (direction * direction).sum(dim=-1, keepdim=True)
        t = ((point - start) * direction).sum(dim=-1, keepdim=True) / length_sq.clamp_min(1e-12)
        t = t.clamp(0.0, 1.0)
        projection = start + t * direction
        return torch.linalg.norm(point - projection, dim=-1)

    def _check_forbidden_crossing(self):
        grasped = self.agent.is_grasping(self.cubeA)
        self._grasp_started |= grasped
        cube_xy = self.cubeA.pose.p[:, :2]
        point_distance = torch.linalg.norm(cube_xy - self._obstacle_center, dim=-1)
        point_hit = point_distance < self._forbidden_radius
        segment_distance = self._segment_point_distance(
            self._obstacle_center, self._prev_cube_xy, cube_xy
        )
        segment_hit = segment_distance < self._forbidden_radius
        self._forbidden_crossing |= self._grasp_started & (point_hit | segment_hit)
        self._prev_cube_xy = cube_xy.clone()

    def _after_simulation_step(self):
        self._check_obstacle_contacts()
        self._check_forbidden_crossing()

    def evaluate(self):
        info = super().evaluate()
        stack_success = info["success"]
        info["stack_success"] = stack_success
        info[OBSTACLE_STATE_KEY + "_collision"] = self._obstacle_collision
        info[OBSTACLE_STATE_KEY + "_crossing"] = self._forbidden_crossing
        info["success"] = stack_success & ~self._obstacle_collision & ~self._forbidden_crossing
        return info
