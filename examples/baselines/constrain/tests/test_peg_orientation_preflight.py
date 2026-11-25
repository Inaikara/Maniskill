import json

import gymnasium as gym
import h5py
import numpy as np

from mani_skill.trajectory.utils.actions.conversion import from_pd_joint_pos_to_ee

from examples.baselines.constrain.peg_orientation import ENV_ID
from examples.baselines.constrain.peg_orientation_preflight import (
    install_mplib_base_pose_compat,
    materialize_valid_episodes,
)


class ActionCapture(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.actions = []

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        return self.env.step(action)


def test_official_conversion_produces_full_pose_action_for_orientation_task():
    source = gym.make(
        ENV_ID, obs_mode="state", control_mode="pd_joint_pos",
        sim_backend="physx_cpu", reward_mode="sparse",
    )
    target = ActionCapture(gym.make(
        ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
        sim_backend="physx_cpu", reward_mode="sparse",
    ))
    try:
        source.reset(seed=91)
        target.reset(seed=91)
        current = source.unwrapped.agent.robot.get_qpos()[0, :7].cpu().numpy()
        target_qpos = current.copy()
        target_qpos[6] += 0.08
        joint_action = np.r_[target_qpos, 1.0]
        from_pd_joint_pos_to_ee(
            "pd_ee_delta_pose", [joint_action], source, target
        )
        assert len(target.actions) >= 1
        pose_actions = np.asarray(target.actions)
        assert pose_actions.shape[1] == 7
        assert np.isfinite(pose_actions).all()
        assert np.max(np.abs(pose_actions[:, 3:6])) > 1e-6
    finally:
        source.close()
        target.close()


def test_mplib_base_pose_adapter_converts_maniskill_seven_vector():
    import mplib

    class PlannerBinding:
        def set_base_pose(self, pose):
            assert isinstance(pose, mplib.Pose)
            np.testing.assert_allclose(pose.p, [-0.615, 0, 0], atol=1e-6)
            np.testing.assert_allclose(pose.q, [1, 0, 0, 0], atol=1e-6)

        def plan_screw(self, goal_pose, current_qpos):
            assert isinstance(goal_pose, mplib.Pose)
            return goal_pose, current_qpos

    install_mplib_base_pose_compat(PlannerBinding)
    planner = PlannerBinding()
    planner.set_base_pose(np.asarray([-0.615, 0, 0, 1, 0, 0, 0], dtype=np.float32))
    goal, qpos = planner.plan_screw(
        np.asarray([0, 0, 0, 1, 0, 0, 0], dtype=np.float32), "qpos",
        use_point_cloud=False,
    )
    assert isinstance(goal, mplib.Pose) and qpos == "qpos"


def test_materialized_dataset_contains_only_requested_replay_verified_rows(tmp_path):
    candidate = tmp_path / "joint_expert.state.pd_ee_delta_pose.physx_cpu.h5"
    with h5py.File(candidate, "w") as h5:
        for index in range(2):
            group = h5.create_group(f"traj_{index}")
            group.create_dataset("actions", data=np.full((2, 7), index, dtype=np.float32))
    metadata = {
        "env_info": {"env_id": "PegInsertionSideOrientation-v1", "env_kwargs": {}},
        "episodes": [{"episode_id": i, "episode_seed": 10 + i} for i in range(2)],
    }
    candidate.with_suffix(".json").write_text(json.dumps(metadata))
    (tmp_path / "joint_expert.h5").write_bytes(b"joint")
    (tmp_path / "collection_report.json").write_text(json.dumps({
        "attempt_count": 2, "expert_accepted": 2, "source_sha256": {"test": "identity"},
    }))
    (tmp_path / "replay_audit.json").write_text(json.dumps({
        "attempts": [], "native_successful_replays": 2,
        "episodes": [
            {"seed": 10, "target_yaw": 1.57, "peg_initial_yaw": 0.2,
             "T": 2, "state_count": 3, "observation_count": 3, "hint_count": 3,
             "action_shape": [2, 7], "rotation_nonzero_steps": 1, "restore_checks": [{}, {}]},
            {"seed": 11, "target_yaw": 1.58, "peg_initial_yaw": 0.3,
             "T": 2, "state_count": 3, "observation_count": 3, "hint_count": 3,
             "action_shape": [2, 7], "rotation_nonzero_steps": 1, "restore_checks": [{}, {}]},
        ],
    }))

    manifest = materialize_valid_episodes(tmp_path, 1)

    with h5py.File(tmp_path / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", "r") as h5:
        assert list(h5) == ["traj_0"]
        np.testing.assert_array_equal(h5["traj_0/actions"], np.zeros((2, 7)))
    assert manifest["episode_count"] == 1
    assert manifest["selected_seeds"] == [10]
