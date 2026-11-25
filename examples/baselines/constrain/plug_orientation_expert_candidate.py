"""Experiment-side copy of the official PlugCharger planner with one tilt knob.

Copied from ManiSkill's ``panda/solutions/plug_charger.py`` at ROUND78.
The only motion-plan parameter change is ``grasp_tilt_deg``; all poses,
limits, timings, gripper commands and success behavior are retained.
"""

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat

DEFAULT_GRASP_TILT_DEG = 15.0


def apply_grasp_tilt(grasp_pose, grasp_tilt_deg=DEFAULT_GRASP_TILT_DEG):
    """Apply a local-Y rotation, matching the official solver's composition."""
    return grasp_pose * sapien.Pose(q=euler2quat(0, np.deg2rad(grasp_tilt_deg), 0))


def solve(env, seed=None, debug=False, vis=False,
          grasp_tilt_deg=DEFAULT_GRASP_TILT_DEG):
    """Run the official expert sequence with a preregisterable grasp tilt."""
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )
    from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb
    import trimesh

    env.reset(seed=seed)
    assert env.unwrapped.control_mode in ["pd_joint_pos", "pd_joint_pos_vel"], env.unwrapped.control_mode
    planner = PandaArmMotionPlanningSolver(
        env, debug=debug, vis=vis, base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False, print_env_info=False,
        joint_vel_limits=0.5, joint_acc_limits=0.5,
    )
    try:
        FINGER_LENGTH = 0.025
        env = env.unwrapped
        charger_base_pose = env.charger_base_pose
        charger_base_size = np.array(env.unwrapped._base_size) * 2
        obb = trimesh.primitives.Box(
            extents=charger_base_size,
            transform=charger_base_pose.sp.to_transformation_matrix(),
        )
        approaching = np.array([0, 0, -1])
        target_closing = env.agent.tcp.pose.sp.to_transformation_matrix()[:3, 1]
        grasp_info = compute_grasp_info_by_obb(
            obb, approaching=approaching, target_closing=target_closing,
            depth=FINGER_LENGTH,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, center)
        grasp_pose = apply_grasp_tilt(grasp_pose, grasp_tilt_deg)

        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        planner.move_to_pose_with_screw(reach_pose)
        planner.move_to_pose_with_screw(grasp_pose)
        planner.close_gripper()
        pre_insert_pose = (
            env.goal_pose.sp * sapien.Pose([-0.05, 0.0, 0.0])
            * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
        )
        insert_pose = env.goal_pose.sp * env.charger.pose.sp.inv() * env.agent.tcp.pose.sp
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=0)
        planner.move_to_pose_with_screw(pre_insert_pose, refine_steps=5)
        result = planner.move_to_pose_with_screw(insert_pose)
        return result
    finally:
        planner.close()
