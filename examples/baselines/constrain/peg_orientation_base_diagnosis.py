"""Read-only prediction diagnostics for a retained fixed-yaw Peg BaseDP."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import h5py

from examples.baselines.constrain import peg_orientation_base_dp
from examples.baselines.constrain.peg_orientation_base_dp import (
    EVAL_ENV_TIME_LIMIT,
    PegBaseDataset,
    build_agent,
    evaluate,
    observation_peg_and_hole,
    sha256_file,
)
from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog


def known_validation_failure_reproduced(record, catalog_index, base_seed=1907, control_cap=500):
    """Return whether one catalog row repeats the known cap-exhaustion failure."""
    preinsertion = record.get("preinsertion") or {}
    return (
        record.get("scene_index") == catalog_index
        and record.get("policy_sampling_seed") == base_seed + catalog_index
        and record.get("success") is False
        and record.get("failure_type") == "max_episode_steps"
        and record.get("executed_controls") == control_cap
        and record.get("first_success_control") is None
        and preinsertion.get("status") == "not_reached"
    )


def _stats(values):
    values = np.asarray(values)
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist()}


def audit_training_horizon(dataset_path, manifest_path, output):
    """Audit all recorded expert lengths and first native success/truncation indices."""
    manifest = json.loads(Path(manifest_path).read_text())
    episodes = []
    with h5py.File(dataset_path, "r") as h5:
        for episode_id, metadata in enumerate(manifest["episodes"]):
            group = h5[f"traj_{episode_id}"]
            length = len(group["actions"])
            success_indices = np.flatnonzero(np.asarray(group["success"], dtype=bool))
            terminated_indices = np.flatnonzero(np.asarray(group["terminated"], dtype=bool))
            truncated_indices = np.flatnonzero(np.asarray(group["truncated"], dtype=bool))
            episodes.append({
                "episode_id": episode_id,
                "episode_seed": int(metadata["episode_seed"]),
                "action_steps": length,
                "first_native_success_step": int(success_indices[0] + 1) if len(success_indices) else None,
                "terminated_step": int(terminated_indices[0] + 1) if len(terminated_indices) else None,
                "truncated_step": int(truncated_indices[0] + 1) if len(truncated_indices) else None,
            })
    lengths = np.asarray([row["action_steps"] for row in episodes])
    successes = np.asarray([row["first_native_success_step"] for row in episodes
                            if row["first_native_success_step"] is not None])
    def distribution(values):
        return {"min": int(values.min()), "median": float(np.median(values)),
                "p90": int(np.quantile(values, 0.9)), "p95": int(np.quantile(values, 0.95)),
                "p99": int(np.quantile(values, 0.99)), "max": int(values.max()),
                "step_counts": {str(int(step)): int(np.sum(values == step))
                                for step in np.unique(values)}}
    result = {
        "format": "peg-orientation-training-horizon-audit-v1",
        "dataset": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
        "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
        "episodes": len(episodes), "action_length_steps": distribution(lengths),
        "native_first_success_steps": distribution(successes),
        "action_lengths_gt_100": int(np.sum(lengths > 100)),
        "native_successes_after_100": int(np.sum(successes > 100)),
        "truncated_at_100": sum(row["truncated_step"] == 100 for row in episodes),
        "episodes_without_truncation_marker": sum(row["truncated_step"] is None for row in episodes),
        "per_episode": episodes,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def _infer(agent, history, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        return agent.get_action(torch.as_tensor(history[None], dtype=torch.float32,
                                                device=next(agent.parameters()).device))[0]


def diagnose_matched_phases(checkpoint, dataset_path, manifest_path, output,
                            seed=1907, episode_count=32, close_windows=64):
    """Compare expert actions to pinned-noise predictions at matched episode phases."""
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(Path(manifest_path).read_text())
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    starts, closes = [], []
    with h5py.File(dataset_path, "r") as h5:
        for episode_id in range(min(episode_count, len(manifest["episodes"]))):
            group = h5[f"traj_{episode_id}"]
            obs, actions = np.asarray(group["obs"]), np.asarray(group["actions"])
            starts.append((episode_id, obs, actions))
            transition = next((i for i, action in enumerate(actions[:close_windows])
                               if action[6] <= -0.5), None)
            if transition is not None:
                closes.append((episode_id, transition, obs, actions))

    def compare(rows, phase):
        action_errors, per_episode = [], []
        gripper_sign_matches = []
        for episode_id, obs, actions, index in rows:
            history = np.stack((obs[max(0, index - 1)], obs[index]))
            predicted = _infer(agent, history, seed).cpu().numpy()
            target = actions[index:index + len(predicted)]
            length = min(len(predicted), len(target))
            error = predicted[:length] - target[:length]
            action_errors.append(error)
            gripper_sign_matches.append(bool(np.sign(predicted[0, 6]) == np.sign(target[0, 6])))
            per_episode.append({
                "episode_id": episode_id, "episode_seed": int(manifest["episodes"][episode_id]["episode_seed"]),
                "action_index": index, "true_action": target[0].tolist(),
                "predicted_action": predicted[0].tolist(),
                "first_action_mae": float(np.mean(np.abs(error[0]))),
                "horizon_action_mae": float(np.mean(np.abs(error))),
                "true_gripper": target[:length, 6].tolist(),
                "predicted_gripper": predicted[:length, 6].tolist(),
            })
        flattened = np.concatenate(action_errors, axis=0) if action_errors else np.empty((0, 7))
        return {
            "phase": phase, "samples": len(per_episode),
            "first_action_mae_by_channel": np.mean(np.abs(np.stack([
                np.asarray(row["predicted_action"]) - np.asarray(row["true_action"])
                for row in per_episode
            ])), axis=0).tolist() if per_episode else None,
            "horizon_mae_by_channel": np.mean(np.abs(flattened), axis=0).tolist() if len(flattened) else None,
            "first_gripper_sign_match_rate": float(np.mean(gripper_sign_matches)) if per_episode else None,
            "episodes": per_episode,
        }

    start_rows = [(episode_id, obs, actions, 0) for episode_id, obs, actions in starts]
    close_rows = [(episode_id, obs, actions, index) for episode_id, index, obs, actions in closes]
    noise_repeats = []
    episode0_chunks = []
    if starts and starts[0][0] == 0:
        _, obs, actions = starts[0]
        for action_index in range(0, min(len(actions), 128), 8):
            history = np.stack((obs[max(0, action_index - 1)], obs[action_index]))
            prediction = _infer(agent, history, seed + action_index).cpu().numpy()
            length = min(len(prediction), len(actions) - action_index)
            target = actions[action_index:action_index + length]
            episode0_chunks.append({
                "action_index": action_index,
                "expert_phase": "preclose" if action_index < 48 else "close_chunk" if action_index == 48 else "postclose",
                "noise_seed": seed + action_index,
                "first_action_mae_by_channel": np.abs(prediction[0] - target[0]).tolist(),
                "horizon_mae_by_channel": np.mean(np.abs(prediction[:length] - target), axis=0).tolist(),
                "gripper_sign_mismatches_in_executed_horizon": int(np.sum(
                    np.sign(prediction[:length, 6]) != np.sign(target[:, 6]))),
                "executed_horizon_length": length,
                "true_first_action": target[0].tolist(), "predicted_first_action": prediction[0].tolist(),
            })
    phase_noise_rows = {"preclose": [], "close": []}
    for episode_id, close_index, obs, actions in closes:
        for phase, action_index in (("preclose", close_index - 1), ("close", close_index)):
            if action_index < 0:
                continue
            history = np.stack((obs[max(0, action_index - 1)], obs[action_index]))
            for noise_seed in (1907, 1908, 1909, 1910):
                prediction = _infer(agent, history, noise_seed)[0].cpu().numpy()
                true_gripper = float(actions[action_index, 6])
                row = {"episode_id": episode_id, "action_index": action_index,
                       "noise_seed": noise_seed, "true_gripper": true_gripper,
                       "predicted_gripper": float(prediction[6]),
                       "sign_match": bool(np.sign(prediction[6]) == np.sign(true_gripper))}
                phase_noise_rows[phase].append(row)
    if closes:
        episode_id, close_index, obs, actions = closes[0]
        for action_index in range(max(0, close_index - 2), close_index + 1):
            history = np.stack((obs[max(0, action_index - 1)], obs[action_index]))
            for noise_seed in (1907, 1908, 1909, 1910):
                prediction = _infer(agent, history, noise_seed)[0].cpu().numpy()
                noise_repeats.append({
                    "episode_id": episode_id, "action_index": action_index,
                    "steps_before_expert_close": action_index - close_index,
                    "noise_seed": noise_seed,
                    "true_gripper": float(actions[action_index, 6]),
                    "predicted_gripper": float(prediction[6]),
                    "true_action": actions[action_index].tolist(),
                    "predicted_action": prediction.tolist(),
                })
    result = {
        "format": "peg-orientation-matched-phase-diagnosis-v1", "seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint), "dataset_sha256": sha256_file(dataset_path),
        "episode_count": len(starts), "close_window_limit": close_windows,
        "initial_phase": compare(start_rows, "episode initial state, true action 0"),
        "first_expert_close": compare(close_rows, "first expert gripper action <= -0.5 within window"),
        "episode0_replan_chunks": episode0_chunks,
        "gripper_noise_sensitivity": {
            phase: {"samples": len(rows), "sign_mismatches": sum(not row["sign_match"] for row in rows),
                    "sign_match_rate": float(np.mean([row["sign_match"] for row in rows])) if rows else None,
                    "rows": rows}
            for phase, rows in phase_noise_rows.items()
        },
        "episode0_first_close_noise_repeats": noise_repeats,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def replay_expert_episode(checkpoint, dataset_path, manifest_path, episode_id, output, seed=1907):
    """Replay one recorded successful native 7D episode from its saved state."""
    import gymnasium as gym
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common
    from examples.baselines.constrain import peg_orientation

    manifest = json.loads(Path(manifest_path).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    row = manifest["episodes"][episode_id]
    with h5py.File(dataset_path, "r") as h5:
        group = h5[f"traj_{episode_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        expected_obs = np.asarray(group["obs"], dtype=np.float32)
        recorded_success = bool(np.asarray(group["success"])[-1])
        states = group["env_states"]
        initial_state = trajectory_utils.index_dict(states, 0)
    env = gym.make(peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=len(actions) + 1)
    try:
        env.reset(seed=int(row["episode_seed"]))
        env.unwrapped.set_state_dict(common.batch(initial_state))
        initial_obs = env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)
        observation_deltas, tcp_positions, grasped, replayed_obs = [], [], [], [initial_obs]
        native_success = False
        terminal = None
        for index, action in enumerate(actions):
            obs, _, terminated, truncated, info = env.step(action)
            flat = obs.detach().cpu().numpy().reshape(-1)
            observation_deltas.append(float(np.max(np.abs(flat - expected_obs[index + 1]))))
            tcp_positions.append(flat[18:21].tolist())
            value = env.unwrapped.agent.is_grasping(env.unwrapped.peg)
            grasped.append(bool(value.detach().cpu().reshape(-1)[0]))
            replayed_obs.append(flat)
            native_success = native_success or bool(np.asarray(info.get("success", False)).reshape(-1)[0])
            if terminated or truncated:
                terminal = {"step": index + 1, "terminated": bool(terminated), "truncated": bool(truncated),
                            "native_success": native_success, "fail": str(info.get("fail", ""))}
                break
        result = {
            "format": "peg-orientation-expert-native-replay-v1", "episode_id": episode_id,
            "episode_seed": int(row["episode_seed"]), "recorded_success": recorded_success,
            "action_steps_recorded": len(actions), "action_steps_executed": len(observation_deltas),
            "initial_observation_max_abs_error": float(np.max(np.abs(initial_obs - expected_obs[0]))),
            "replayed_observation_max_abs_error": max(observation_deltas, default=None),
            "replayed_observation_mean_abs_error": float(np.mean(observation_deltas)) if observation_deltas else None,
            "native_success": native_success, "terminal": terminal,
            "grasped_true_steps": int(sum(grasped)),
            "first_grasp_step": next((i + 1 for i, value in enumerate(grasped) if value), None),
            "tcp_displacement_m": float(np.linalg.norm(np.asarray(tcp_positions[-1]) - np.asarray(tcp_positions[0])))
            if tcp_positions else 0.0,
            "action_minmax_by_channel": [[float(actions[:, i].min()), float(actions[:, i].max())]
                                         for i in range(actions.shape[1])],
        }
        first_grasp = result["first_grasp_step"]
        if first_grasp is not None:
            around = []
            for action_index in range(max(0, first_grasp - 3), min(len(observation_deltas), first_grasp + 2)):
                history = np.stack((replayed_obs[max(0, action_index - 1)], replayed_obs[action_index]))
                predicted = _infer(agent, history, seed).cpu().numpy()
                true_action = actions[action_index]
                around.append({
                    "action_index": action_index, "steps_from_first_grasp": action_index - first_grasp,
                    "grasped_before_action": grasped[action_index - 1] if action_index else False,
                    "observation_max_abs_error": float(np.max(np.abs(replayed_obs[action_index] - expected_obs[action_index]))),
                    "true_action": true_action.tolist(), "predicted_action": predicted[0].tolist(),
                    "action_mae": float(np.mean(np.abs(predicted[0] - true_action))),
                })
            result["pinned_noise_policy_near_grasp"] = {"seed": seed, "actions": around}
    finally:
        env.close()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_native_prefix(checkpoint, dataset_path, manifest_path, episode_id, output,
                          seed=1907, steps=3):
    """Compare expert and policy controls from the same restored training state."""
    import gymnasium as gym
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common
    from examples.baselines.constrain import peg_orientation

    manifest = json.loads(Path(manifest_path).read_text())
    with h5py.File(dataset_path, "r") as h5:
        group = h5[f"traj_{episode_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        expected_obs = np.asarray(group["obs"], dtype=np.float32)
        initial_state = trajectory_utils.index_dict(group["env_states"], 0)
    if not 1 <= steps <= min(3, len(actions)):
        raise ValueError("prefix probe steps must be between 1 and 3")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    env = gym.make(peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=len(actions) + 1)

    def restore():
        env.reset(seed=int(manifest["episodes"][episode_id]["episode_seed"]))
        env.unwrapped.set_state_dict(common.batch(initial_state))
        return env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)

    def trace(action_sequence):
        rows = []
        for index, action in enumerate(action_sequence[:steps]):
            before = env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)
            _, _, terminated, truncated, info = env.step(action)
            after = env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)
            grasp = env.unwrapped.agent.is_grasping(env.unwrapped.peg)
            rows.append({
                "step": index, "action": np.asarray(action).tolist(),
                "observation_max_abs_error_before": float(np.max(np.abs(before - expected_obs[index]))),
                "tcp_position_before_m": before[18:21].tolist(), "tcp_position_after_m": after[18:21].tolist(),
                "tcp_translation_m": float(np.linalg.norm(after[18:21] - before[18:21])),
                "peg_position_before_m": before[25:28].tolist(), "peg_position_after_m": after[25:28].tolist(),
                "peg_translation_m": float(np.linalg.norm(after[25:28] - before[25:28])),
                "is_grasping_after": bool(grasp.detach().cpu().reshape(-1)[0]),
                "native_success": bool(np.asarray(info.get("success", False)).reshape(-1)[0]),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
        return rows

    try:
        expert_initial = restore()
        expert_trace = trace(actions)
        policy_initial = restore()
        history = np.stack((policy_initial, policy_initial))
        policy_actions = _infer(agent, history, seed).cpu().numpy()
        policy_trace = trace(policy_actions)
        result = {
            "format": "peg-orientation-native-prefix-differential-v1", "episode_id": episode_id,
            "episode_seed": int(manifest["episodes"][episode_id]["episode_seed"]),
            "checkpoint_sha256": sha256_file(checkpoint), "dataset_sha256": sha256_file(dataset_path),
            "pinned_noise_seed": seed, "steps_requested": steps,
            "initial_observation_expert_restore_max_abs": float(np.max(np.abs(expert_initial - expected_obs[0]))),
            "initial_observation_policy_restore_max_abs": float(np.max(np.abs(policy_initial - expected_obs[0]))),
            "expert_trace": expert_trace, "policy_trace": policy_trace,
            "first_action_delta_by_channel": (policy_trace[0]["action"] - np.asarray(expert_trace[0]["action"])).tolist(),
        }
    finally:
        env.close()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def probe_grasp_chunk(checkpoint, dataset_path, manifest_path, episode_id, output,
                      noise_seeds=(1907, 1908, 1909, 1910)):
    """Replay the expert and pinned-noise policy chunk from the true pre-close state."""
    import gymnasium as gym
    from mani_skill.trajectory import utils as trajectory_utils
    from mani_skill.utils import common
    from examples.baselines.constrain import peg_orientation

    manifest = json.loads(Path(manifest_path).read_text())
    with h5py.File(dataset_path, "r") as h5:
        group = h5[f"traj_{episode_id}"]
        actions = np.asarray(group["actions"], dtype=np.float32)
        observations = np.asarray(group["obs"], dtype=np.float32)
        close_index = next(i for i, action in enumerate(actions) if action[6] <= -0.5)
        state_index = max(0, close_index - 1)
        initial_state = trajectory_utils.index_dict(group["env_states"], state_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    env = gym.make(peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=len(actions) + 1)

    def restore():
        env.reset(seed=int(manifest["episodes"][episode_id]["episode_seed"]))
        env.unwrapped.set_state_dict(common.batch(initial_state))
        return env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)

    def execute(action_sequence, label):
        trace = []
        for offset, action in enumerate(action_sequence[:8]):
            before = env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)
            _, _, terminated, truncated, info = env.step(action)
            after = env.unwrapped.get_obs().detach().cpu().numpy().reshape(-1)
            grasp = env.unwrapped.agent.is_grasping(env.unwrapped.peg)
            trace.append({
                "offset": offset, "action": np.asarray(action).tolist(),
                "grasped_after": bool(grasp.detach().cpu().reshape(-1)[0]),
                "tcp_position_m": after[18:21].tolist(), "peg_position_m": after[25:28].tolist(),
                "tcp_step_m": float(np.linalg.norm(after[18:21] - before[18:21])),
                "peg_step_m": float(np.linalg.norm(after[25:28] - before[25:28])),
                "native_success": bool(np.asarray(info.get("success", False)).reshape(-1)[0]),
                "terminated": bool(terminated), "truncated": bool(truncated),
            })
            if terminated or truncated:
                break
        return {"label": label, "steps": trace,
                "first_grasp_offset": next((row["offset"] for row in trace if row["grasped_after"]), None),
                "final_tcp_position_m": trace[-1]["tcp_position_m"] if trace else None,
                "final_peg_position_m": trace[-1]["peg_position_m"] if trace else None,
                "native_success": any(row["native_success"] for row in trace)}

    try:
        expert_initial = restore()
        expected_initial = observations[state_index]
        expert = execute(actions[state_index:state_index + 8], "recorded_expert")
        policy = []
        for noise_seed in noise_seeds:
            actual_initial = restore()
            history = np.stack((observations[max(0, state_index - 1)], observations[state_index]))
            predicted = _infer(agent, history, noise_seed).cpu().numpy()
            outcome = execute(predicted, f"policy_noise_seed_{noise_seed}")
            outcome.update(noise_seed=noise_seed,
                           restored_observation_max_abs=float(np.max(np.abs(actual_initial - expected_initial))),
                           first_predicted_action=predicted[0].tolist())
            policy.append(outcome)
        result = {
            "format": "peg-orientation-grasp-chunk-counterfactual-v1", "episode_id": episode_id,
            "episode_seed": int(manifest["episodes"][episode_id]["episode_seed"]),
            "checkpoint_sha256": sha256_file(checkpoint), "dataset_sha256": sha256_file(dataset_path),
            "expert_close_action_index": close_index, "restored_preclose_state_action_index": state_index,
            "pinned_noise_seeds": list(noise_seeds),
            "expert_restore_observation_max_abs": float(np.max(np.abs(expert_initial - expected_initial))),
            "expert": expert, "policy_counterfactuals": policy,
        }
    finally:
        env.close()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def diagnose_preregistered_training_phases(checkpoint, dataset_path, manifest_path,
                                           episode_ids, output, noise_seeds=(1907, 1908, 1909, 1910)):
    """Measure pinned teacher-forced action errors at grasp and insertion phases."""
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(Path(manifest_path).read_text())
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    episodes = []
    with h5py.File(dataset_path, "r") as h5:
        for episode_id in episode_ids:
            group = h5[f"traj_{episode_id}"]
            observations = np.asarray(group["obs"], dtype=np.float32)
            actions = np.asarray(group["actions"], dtype=np.float32)
            close_index = next((i for i, action in enumerate(actions) if action[6] <= -0.5), None)
            if close_index is None:
                raise ValueError(f"episode {episode_id} has no expert close action")
            peg_xy, hole_xy = observation_peg_and_hole(observations)
            peg_xy, hole_xy = peg_xy[:, :2], hole_xy[:, :2]
            insertion_index = next((i for i, distance in enumerate(
                np.linalg.norm(peg_xy - hole_xy, axis=1)) if distance <= 0.1), None)
            grasp_index = None
            states = group["env_states"]
            # Inspect saved states directly; do not advance the simulator or alter the data.
            import gymnasium as gym
            from mani_skill.trajectory import utils as trajectory_utils
            from mani_skill.utils import common
            from examples.baselines.constrain import peg_orientation

            env = gym.make(peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                           sim_backend="physx_cpu", reconfiguration_freq=1,
                           max_episode_steps=len(actions) + 1)
            try:
                env.reset(seed=int(manifest["episodes"][episode_id]["episode_seed"]))
                for index in range(close_index, len(actions) + 1):
                    env.unwrapped.set_state_dict(common.batch(trajectory_utils.index_dict(states, index)))
                    grasp = env.unwrapped.agent.is_grasping(env.unwrapped.peg)
                    if bool(grasp.detach().cpu().reshape(-1)[0]):
                        grasp_index = index
                        break
            finally:
                env.close()

            phases = {"pregrasp": max(0, close_index - 1),
                      "postgrasp": grasp_index,
                      "preinsertion": insertion_index}
            phase_rows = {}
            for phase, index in phases.items():
                if index is None or index >= len(actions):
                    phase_rows[phase] = {"action_index": index, "available": False}
                    continue
                history = np.stack((observations[max(0, index - 1)], observations[index]))
                predictions = []
                for noise_seed in noise_seeds:
                    prediction = _infer(agent, history, noise_seed).cpu().numpy()
                    length = min(8, len(actions) - index, len(prediction))
                    error = prediction[:length] - actions[index:index + length]
                    predictions.append({
                        "noise_seed": noise_seed,
                        "first_action_mae_by_channel": np.abs(error[0]).tolist(),
                        "first_8_action_mae_by_channel": np.mean(np.abs(error), axis=0).tolist(),
                        "first_8_action_mae": float(np.mean(np.abs(error))),
                    })
                peg_position, hole_center = observation_peg_and_hole(observations[index])
                phase_rows[phase] = {
                    "available": True,
                    "action_index": index,
                    "peg_hole_xy_distance_m": float(np.linalg.norm(peg_position[:2] - hole_center[:2])),
                    "peg_position_m": peg_position.tolist(),
                    "hole_center_m": hole_center.tolist(),
                    "expert_actions": actions[index:index + 8].tolist(),
                    "noise_predictions": predictions,
                }
            episodes.append({
                "episode_id": episode_id,
                "episode_seed": int(manifest["episodes"][episode_id]["episode_seed"]),
                "action_steps": len(actions), "first_close_action_index": close_index,
                "first_saved_grasp_state_index": grasp_index,
                "first_preinsertion_state_index": insertion_index,
                "phases": phase_rows,
            })
    result = {
        "format": "peg-orientation-preregistered-training-phase-diagnosis-v1",
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset_path),
        "manifest_sha256": sha256_file(manifest_path),
        "noise_seeds": list(noise_seeds), "native_simulator_controls": 0,
        "episodes": episodes,
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_policy_from_replayed_grasp(checkpoint, dataset_path, manifest_path, output,
                                       episode_seeds=((61, 1907), (61, 1908),
                                                      (62, 1907), (77, 1907)),
                                       max_native_controls=500):
    """Compare expert and policy chunks after independently replaying to real grasp."""
    import gymnasium as gym
    from examples.baselines.constrain import peg_orientation

    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads(Path(manifest_path).read_text())
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    result = {
        "format": "peg-orientation-replayed-grasp-closed-loop-differential-v1",
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset_path),
        "manifest_sha256": sha256_file(manifest_path),
        "max_native_controls": int(max_native_controls), "native_controls": 0,
        "arms": [], "skipped": [],
    }
    expert_done = set()
    prefix_lengths = {}

    def grasped(env):
        value = env.unwrapped.agent.is_grasping(env.unwrapped.peg)
        return bool(value.detach().cpu().reshape(-1)[0])

    def metric_row(env, observation, action, expert_action, local_step, phase):
        peg, hole = observation_peg_and_hole(observation)
        tcp = observation[18:21]
        info_grasp = grasped(env)
        return {
            "step": local_step, "phase": phase,
            "action": np.asarray(action).tolist(),
            "expert_action_mae_by_channel": np.abs(np.asarray(action) - expert_action).tolist(),
            "is_grasping": info_grasp,
            "tcp_position_m": tcp.tolist(), "peg_position_m": peg.tolist(),
            "hole_center_m": hole.tolist(),
            "peg_hole_xy_distance_m": float(np.linalg.norm(peg[:2] - hole[:2])),
            "peg_hole_center_distance_m": float(np.linalg.norm(peg - hole)),
            "native_success": False,
        }

    with h5py.File(dataset_path, "r") as h5:
        for episode_id, noise_seed in episode_seeds:
            episode_key = (int(episode_id), int(noise_seed))
            group = h5[f"traj_{episode_id}"]
            actions = np.asarray(group["actions"], dtype=np.float32)
            expected_obs = np.asarray(group["obs"], dtype=np.float32)
            row = manifest["episodes"][episode_id]
            seed = int(row["episode_seed"])
            arms = ["expert8"] if episode_id not in expert_done else []
            arms.extend((f"policy8_seed{noise_seed}", f"policy2_seed{noise_seed}"))
            for arm in arms:
                known_prefix = prefix_lengths.get(episode_id)
                estimate = (known_prefix + 8) if known_prefix is not None else 0
                if known_prefix is not None and result["native_controls"] + estimate > max_native_controls:
                    result["skipped"].append({"episode_id": episode_id, "noise_seed": noise_seed,
                                              "arm": arm, "reason": "500-control budget would be exceeded"})
                    continue
                env = gym.make(peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                               sim_backend="physx_cpu", reconfiguration_freq=1,
                               max_episode_steps=len(actions) + 1)
                try:
                    reset_obs, _ = env.reset(seed=seed)
                    replay_obs = [reset_obs.detach().cpu().numpy().reshape(-1)]
                    grasp_action_index = None
                    prefix_controls = 0
                    for index, expert_action in enumerate(actions):
                        if result["native_controls"] >= max_native_controls:
                            break
                        obs, _, terminated, truncated, info = env.step(expert_action)
                        result["native_controls"] += 1
                        prefix_controls += 1
                        flat = obs.detach().cpu().numpy().reshape(-1)
                        replay_obs.append(flat)
                        if grasped(env):
                            grasp_action_index = index
                            break
                        if terminated or truncated:
                            break
                    if grasp_action_index is None:
                        result["arms"].append({
                            "episode_id": episode_id, "episode_seed": seed, "noise_seed": noise_seed,
                            "arm": arm, "prefix_controls": prefix_controls,
                            "actual_grasp_reached": False,
                            "native_terminal_before_grasp": bool(terminated or truncated),
                            "global_control_budget_exhausted": result["native_controls"] >= max_native_controls,
                        })
                        if result["native_controls"] >= max_native_controls:
                            break
                        continue
                    prefix_lengths.setdefault(episode_id, prefix_controls)
                    branch_error = float(np.max(np.abs(replay_obs[-1] - expected_obs[prefix_controls])))
                    grasp_at_branch = grasped(env)
                    history = [replay_obs[-2].copy(), replay_obs[-1].copy()]
                    trace = []
                    if arm == "expert8":
                        planned = actions[prefix_controls:prefix_controls + 8]
                    else:
                        planned = None
                    control_offset = 0
                    terminated = truncated = False
                    while control_offset < 8 and not (terminated or truncated):
                        if arm == "expert8":
                            chunk = planned[control_offset:control_offset + 1]
                        elif arm.startswith("policy8"):
                            if control_offset == 0:
                                planned = _infer(agent, np.asarray(history), noise_seed).cpu().numpy()
                            chunk = planned[control_offset:control_offset + 1]
                        else:
                            if control_offset % 2 == 0:
                                planned = _infer(agent, np.asarray(history), noise_seed + control_offset // 2).cpu().numpy()
                            chunk = planned[control_offset % 2:control_offset % 2 + 1]
                        action = chunk[0]
                        obs, _, terminated, truncated, info = env.step(action)
                        result["native_controls"] += 1
                        flat = obs.detach().cpu().numpy().reshape(-1)
                        history = [history[-1], flat]
                        detail = metric_row(env, flat, action, actions[prefix_controls + control_offset],
                                            control_offset, arm)
                        detail["native_success"] = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                        trace.append(detail)
                        control_offset += 1
                    result["arms"].append({
                        "episode_id": episode_id, "episode_seed": seed, "noise_seed": noise_seed,
                        "arm": arm, "prefix_controls": prefix_controls,
                        "first_grasp_action_index": grasp_action_index,
                        "branch_observation_max_abs_error_vs_demo": branch_error,
                        "actual_grasp_at_branch": grasp_at_branch,
                        "branch_controls_executed": control_offset,
                        "first_contact_loss_step": next((r["step"] for r in trace if not r["is_grasping"]), None),
                        "native_success": any(r["native_success"] for r in trace),
                        "terminal": bool(terminated or truncated), "trace": trace,
                    })
                    if arm == "expert8":
                        expert_done.add(episode_id)
                finally:
                    env.close()
    result["episodes_sampled"] = sorted({row["episode_id"] for row in result["arms"]})
    result["unexecuted_controls_remaining"] = max_native_controls - result["native_controls"]
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_training_rollout_horizon_comparison(
    checkpoint, dataset_path, manifest_path, output, episode_ids,
    max_episode_steps=500, execution_horizons=(8, 2), seed=1907,
):
    """Run paired full policy rollouts from exact training reset states."""
    import gymnasium as gym
    from examples.baselines.constrain import peg_orientation

    if max_episode_steps < 1 or max_episode_steps > EVAL_ENV_TIME_LIMIT:
        raise ValueError(f"max_episode_steps must be between 1 and {EVAL_ENV_TIME_LIMIT}")
    if not execution_horizons or any(horizon < 1 for horizon in execution_horizons):
        raise ValueError("execution horizons must be positive")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    manifest = json.loads(Path(manifest_path).read_text())
    result = {
        "format": "peg-orientation-full-training-rollout-horizon-comparison-v1",
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset_path),
        "manifest_sha256": sha256_file(manifest_path),
        "episodes_requested": list(map(int, episode_ids)),
        "max_episode_steps": int(max_episode_steps),
        "execution_horizons": list(map(int, execution_horizons)),
        "sampling_seed_rule": "base_seed + episode_id; each profile restarts the same CPU/CUDA Torch seed",
        "episodes": [],
    }

    with h5py.File(dataset_path, "r") as h5:
        for episode_id in episode_ids:
            group = h5[f"traj_{episode_id}"]
            expected_obs = np.asarray(group["obs"], dtype=np.float32)
            row = manifest["episodes"][episode_id]
            reset_seed = int(row["episode_seed"])
            sample_seed = int(seed) + int(episode_id)
            for execution_horizon in execution_horizons:
                env = gym.make(
                    peg_orientation.ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                    sim_backend="physx_cpu", reconfiguration_freq=1,
                    max_episode_steps=EVAL_ENV_TIME_LIMIT,
                )
                try:
                    observation, _ = env.reset(seed=reset_seed)
                    flat = observation.detach().cpu().numpy().reshape(-1)
                    initial_error = float(np.max(np.abs(flat - expected_obs[0])))
                    if initial_error > 1e-5:
                        raise ValueError(
                            f"episode {episode_id} reset observation differs from stored initial state "
                            f"by {initial_error:.8g}"
                        )
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)
                    history = [flat.copy(), flat.copy()]
                    initial_peg_z = float(flat[27])
                    trace, step_index = [], 0
                    first_grasp_control = None
                    ever_grasped = False
                    previous_grasped = False
                    contact_losses = 0
                    grasped_controls = 0
                    peak_peg_lift = 0.0
                    first_preinsertion_control = None
                    initial_peg, initial_hole = observation_peg_and_hole(flat)
                    minimum_xy_distance = float(np.linalg.norm(initial_peg[:2] - initial_hole[:2]))
                    success = False
                    terminal_reason = "control_cap"
                    terminated = truncated = False
                    while step_index < max_episode_steps and not (terminated or truncated):
                        planned = agent.get_action(torch.as_tensor(
                            np.asarray(history)[None], dtype=torch.float32, device=device
                        ))[0]
                        for action_tensor in planned[:min(execution_horizon, max_episode_steps - step_index)]:
                            before = history[-1]
                            before_peg, before_hole = observation_peg_and_hole(before)
                            before_xy = float(np.linalg.norm(before_peg[:2] - before_hole[:2]))
                            if first_preinsertion_control is None and before_xy <= 0.1:
                                first_preinsertion_control = step_index
                            observation, _, terminated, truncated, info = env.step(
                                action_tensor.detach().cpu().numpy()
                            )
                            flat = observation.detach().cpu().numpy().reshape(-1)
                            step_index += 1
                            grasp_value = env.unwrapped.agent.is_grasping(env.unwrapped.peg)
                            is_grasping = bool(grasp_value.detach().cpu().reshape(-1)[0])
                            if is_grasping and first_grasp_control is None:
                                first_grasp_control = step_index
                            contact_lost = previous_grasped and not is_grasping
                            if contact_lost:
                                contact_losses += 1
                            ever_grasped = ever_grasped or is_grasping
                            grasped_controls += int(is_grasping)
                            previous_grasped = is_grasping
                            peg, hole = observation_peg_and_hole(flat)
                            tcp = flat[18:21]
                            xy_distance = float(np.linalg.norm(peg[:2] - hole[:2]))
                            minimum_xy_distance = min(minimum_xy_distance, xy_distance)
                            peak_peg_lift = max(peak_peg_lift, float(peg[2] - initial_peg_z))
                            success = success or bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                            trace.append({
                                "control": step_index,
                                "is_grasping": is_grasping,
                                "contact_lost": bool(contact_lost),
                                "tcp_position_m": tcp.tolist(), "peg_position_m": peg.tolist(),
                                "tcp_to_peg_distance_m": float(np.linalg.norm(tcp - peg)),
                                "peg_hole_xy_distance_m": xy_distance,
                                "peg_hole_center_distance_m": float(np.linalg.norm(peg - hole)),
                                "peg_lift_m": float(peg[2] - initial_peg_z),
                                "native_success": success,
                                "terminated": bool(terminated), "truncated": bool(truncated),
                            })
                            history = [history[-1], flat]
                            if terminated or truncated:
                                terminal_reason = ("success" if success else str(info.get("fail", "native_termination")))
                                break
                        if terminated or truncated:
                            break
                    final = trace[-1] if trace else None
                    result["episodes"].append({
                        "episode_id": int(episode_id), "episode_seed": reset_seed,
                        "policy_sampling_seed": sample_seed,
                        "execution_horizon": int(execution_horizon),
                        "initial_observation_max_abs_error": initial_error,
                        "executed_controls": step_index, "success": success,
                        "terminal_reason": terminal_reason,
                        "ever_grasped": ever_grasped,
                        "first_grasp_control": first_grasp_control,
                        "grasped_controls": grasped_controls,
                        "contact_losses": contact_losses,
                        "peak_peg_lift_m": peak_peg_lift,
                        "first_preinsertion_control": first_preinsertion_control,
                        "minimum_peg_hole_xy_distance_m": minimum_xy_distance,
                        "final_peg_hole_xy_distance_m": final["peg_hole_xy_distance_m"] if final else None,
                        "final_peg_hole_center_distance_m": final["peg_hole_center_distance_m"] if final else None,
                        "trace": trace,
                    })
                finally:
                    env.close()
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def diagnose(checkpoint, dataset_path, manifest_path, output, seed=1907, windows=128):
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = PegBaseDataset(dataset_path, manifest_path)
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    agent = build_agent(device)
    agent.load_state_dict(ck["ema_agent"], strict=True)
    agent.eval()
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(data), min(windows, len(data)), replace=False)
    observations = torch.stack([data[int(i)]["observations"] for i in indices]).to(device)
    actions = torch.stack([data[int(i)]["actions"] for i in indices]).to(device)

    torch.manual_seed(seed)
    noise = torch.randn_like(actions)
    timesteps = torch.arange(len(indices), device=device) % agent.noise_scheduler.config.num_train_timesteps
    noisy = agent.noise_scheduler.add_noise(actions, noise, timesteps)
    with torch.no_grad():
        pred = agent.noise_pred_net(noisy, timesteps, global_cond=observations.flatten(start_dim=1))
        shuffled = observations.roll(1, dims=0)
        shuffled_pred = agent.noise_pred_net(noisy, timesteps, global_cond=shuffled.flatten(start_dim=1))
        torch.manual_seed(seed)
        conditioned_actions = agent.get_action(observations)
        torch.manual_seed(seed)
        shuffled_actions = agent.get_action(shuffled)
        online_agent = build_agent(device)
        online_agent.load_state_dict(ck["agent"], strict=True)
        online_agent.eval()
        online_pred = online_agent.noise_pred_net(
            noisy, timesteps, global_cond=observations.flatten(start_dim=1)
        )
        catalog = PegSceneCatalog("outputs/exp/20261111_172604_peg_orientation_adaptation/validation_fixed_scenes")
        validation_observations = torch.as_tensor(
            np.stack([catalog.load(i)[2] for i in (0, 1, 2, 3)]), dtype=torch.float32, device=device
        )
        validation_history = validation_observations[:, None, :].repeat(1, 2, 1)
        torch.manual_seed(seed)
        validation_actions = agent.get_action(validation_history)

    result = {
        "format": "peg-orientation-base-diagnosis-v1", "seed": seed,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "dataset": str(dataset_path), "dataset_sha256": sha256_file(dataset_path),
        "windows": len(indices), "device": str(device),
        "training_observations": _stats(observations.cpu().numpy().reshape(-1, 39)),
        "expert_actions": _stats(actions.cpu().numpy().reshape(-1, 7)),
        "epsilon_mse_conditioned": float(torch.mean((pred - noise) ** 2).cpu()),
        "epsilon_mse_shuffled_observations": float(torch.mean((shuffled_pred - noise) ** 2).cpu()),
        "epsilon_mse_online_agent": float(torch.mean((online_pred - noise) ** 2).cpu()),
        "sampled_action_channels_conditioned": _stats(conditioned_actions.cpu().numpy().reshape(-1, 7)),
        "sampled_action_channels_shuffled": _stats(shuffled_actions.cpu().numpy().reshape(-1, 7)),
        "conditioned_vs_shuffled_action_mae": float(torch.mean(torch.abs(conditioned_actions - shuffled_actions)).cpu()),
        "prediction_variation_per_channel": conditioned_actions.std(dim=(0, 1)).cpu().tolist(),
        "validation_initial_observations": _stats(validation_observations.cpu().numpy()),
        "validation_initial_sampled_actions": _stats(validation_actions.cpu().numpy().reshape(-1, 7)),
    }
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def intervention_spec_from_baseline(baseline_episode, prefix_controls, intervention_controls,
                                   physical_hole_plus_x_m, tolerance=1e-5):
    """Recover stored rotation/gripper and the branch-state anchors. No physics."""
    episode = json.loads(Path(baseline_episode).read_text())
    trace = episode.get("phase_trace")
    if not isinstance(trace, list):
        raise ValueError("baseline episode has no phase_trace")
    by_control = {int(row["control"]): row for row in trace}
    prefix_controls = int(prefix_controls)
    count = int(intervention_controls)
    start, end = prefix_controls + 1, prefix_controls + count
    if prefix_controls not in by_control or any(control not in by_control for control in range(start, end + 1)):
        raise ValueError("baseline is missing intervention or prefix controls")
    prefix_row = by_control[prefix_controls]
    anchor = by_control[start]
    anchor_keys = (
        "preaction_peg_pose", "preaction_tcp_pose", "preaction_box_hole_pose",
        "preaction_root_pose", "preaction_peg_head_in_hole_m", "preaction_pose_peg_hole_xy_m",
    )
    prefix_action = prefix_row.get("executed_action_7d")
    if (prefix_action is None or len(prefix_action) != 7 or "is_grasping" not in prefix_row
            or any(len(by_control[control].get("executed_action_7d") or []) != 7 for control in range(start, end + 1))
            or any(key not in anchor for key in anchor_keys)):
        raise ValueError("baseline channels cannot be recovered")
    hole_pose = anchor["preaction_box_hole_pose"]
    return {
        "prefix_controls": prefix_controls,
        "count": count,
        "physical_hole_plus_x_m": float(physical_hole_plus_x_m),
        "tolerance": float(tolerance),
        "max_initial_observation_error": float(tolerance),
        "prefix_action_7d": [float(item) for item in prefix_action[:7]],
        "baseline_actions_7d": [
            [float(item) for item in by_control[control]["executed_action_7d"][:7]]
            for control in range(start, end + 1)
        ],
        "frozen_hole_quat": [float(item) for item in hole_pose[3:7]],
        "match": {
            "peg_pose": anchor["preaction_peg_pose"],
            "tcp_pose": anchor["preaction_tcp_pose"],
            "hole_pose": hole_pose,
            "root_pose": anchor["preaction_root_pose"],
            "peg_head_in_hole_m": anchor["preaction_peg_head_in_hole_m"],
            "pose_xy_m": anchor["preaction_pose_peg_hole_xy_m"],
            "grasping": bool(prefix_row["is_grasping"]),
        },
    }


def run_hole_x_intervention(
    checkpoint, catalog_dir, output_dir, baseline_episode, prefix_controls, intervention_controls,
    physical_hole_plus_x_m, scene_indices, seed=1907, device="cuda", max_episode_steps=198,
    insertion_trace_window=None,
):
    """Replay one unchanged prefix, then replace only translation for a fixed hole-+X arm."""
    selected = [int(index) for index in scene_indices]
    if len(selected) != 1:
        raise ValueError("hole +X intervention is one scene")
    authorized = int(prefix_controls) + int(intervention_controls)
    if int(max_episode_steps) < authorized:
        raise ValueError("episode cap is shorter than the authorized prefix plus intervention")
    spec = intervention_spec_from_baseline(
        baseline_episode, prefix_controls, intervention_controls, physical_hole_plus_x_m,
    )
    summary = evaluate(
        checkpoint=checkpoint, catalog_dir=catalog_dir, output_dir=output_dir, device=device,
        video=False, max_episode_steps=int(max_episode_steps), seed=seed, scene_indices=selected,
        record_phase_trace=True, insertion_trace_window=insertion_trace_window,
        translation_intervention=spec,
    )
    output_dir = Path(output_dir)
    record = summary["records"][0]
    report = record.get("translation_intervention") or {}
    identity = {
        "format": "peg-orientation-hole-x-intervention-v1",
        "counts_as_policy_success": False,
        "label": "scripted_diagnostic_not_model_performance",
        "scene_indices": selected,
        "prefix_controls": int(prefix_controls),
        "intervention_controls": int(intervention_controls),
        "physical_hole_plus_x_m": float(physical_hole_plus_x_m),
        "max_episode_steps": int(max_episode_steps),
        "seed": int(seed),
        "device": device,
        "checkpoint": str(checkpoint),
        "baseline_episode": str(baseline_episode),
        "baseline_episode_sha256": sha256_file(baseline_episode),
        "evaluator_sha256": sha256_file(peg_orientation_base_dp.__file__),
        "diagnosis_sha256": sha256_file(Path(__file__)),
        "success": record["success"],
        "executed_controls": record["executed_controls"],
        "failure_type": record["failure_type"],
        "translation_intervention": report,
    }
    (output_dir / "phase_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    if report.get("prefix_matched") is not True or report.get("transform_rejected"):
        raise SystemExit(2)
    return identity


def reproduce_saved_sampler_inputs(checkpoint, episode_path, device):
    """Replay captured sampler calls with the frozen loader. Does not step an environment."""
    episode = json.loads(Path(episode_path).read_text())
    captures = episode.get("sampler_input_window")
    if not isinstance(captures, list) or not captures:
        raise ValueError("episode has no sampler input window")
    adapter = peg_orientation_base_dp.PegBaseAdapter(checkpoint, device)
    rows = []
    for capture in captures:
        if str(capture.get("device")) != str(device):
            raise ValueError("sampler capture device does not match the reproduction device")
        row = peg_orientation_base_dp.reproduce_sampler_chunk(adapter.get_action, capture, device)
        row["current_vs_native_max_abs"] = capture["current_vs_native_max_abs"]
        row["prior_vs_current_max_abs"] = capture["prior_vs_current_max_abs"]
        row["official_peg_head_in_hole_m"] = capture["official_peg_head_in_hole_m"]
        row["official_insertion"] = capture["official_insertion"]
        rows.append(row)
    return {
        "episode": str(episode_path),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "sampler_calls": (episode.get("phase") or {}).get("sampler_calls"),
        "calls": rows,
    }


def run_validation_phase_diagnosis(
    checkpoint, catalog_dir, output_dir, scene_indices, max_episode_steps=500,
    seed=1907, device="cuda", max_total_controls=3000, insertion_trace_window=None,
):
    """Record validation phase metrics through the unchanged evaluator profile."""
    selected = [int(index) for index in scene_indices]
    if len(selected) * int(max_episode_steps) > int(max_total_controls):
        raise ValueError(
            f"requested controls {len(selected) * int(max_episode_steps)} exceed {max_total_controls}"
        )
    if insertion_trace_window is not None:
        insertion_trace_window = (int(insertion_trace_window[0]), int(insertion_trace_window[1]))
    summary = evaluate(
        checkpoint=checkpoint, catalog_dir=catalog_dir, output_dir=output_dir,
        device=device, video=False, max_episode_steps=max_episode_steps, seed=seed,
        scene_indices=selected, record_phase_trace=True,
        insertion_trace_window=insertion_trace_window,
    )
    output_dir = Path(output_dir)
    catalog_dir = Path(catalog_dir)
    records = [
        {key: value for key, value in record.items() if key not in ("phase_trace", "sampler_input_window")}
        for record in summary["records"]
    ]
    identity = {
        "format": "peg-orientation-validation-phase-diagnosis-v1",
        "scene_indices": selected,
        "record_phase_trace": True,
        "insertion_trace_window": None if insertion_trace_window is None else list(insertion_trace_window),
        "max_episode_steps": int(max_episode_steps),
        "env_time_limit": EVAL_ENV_TIME_LIMIT,
        "seed": int(seed),
        "device": device,
        "checkpoint_sha256": sha256_file(checkpoint),
        "evaluator_sha256": sha256_file(peg_orientation_base_dp.__file__),
        "diagnosis_sha256": sha256_file(Path(__file__)),
        "catalog_manifest_sha256": sha256_file(catalog_dir / "manifest.json"),
        "catalog_scenes_sha256": sha256_file(catalog_dir / "scenes.h5"),
        "records": records,
    }
    (output_dir / "phase_identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    return identity


def _rotation_matrix(quaternion):
    """Rotation matrix for a stored wxyz quaternion. Does not renormalize."""
    quat = np.asarray(quaternion, dtype=np.float64)
    w, x, y, z = quat
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _linear_quantiles(values):
    if len(values) == 0:
        return None
    names = ("q00", "q10", "q25", "q50", "q75", "q90", "q100")
    points = np.quantile(np.asarray(values, dtype=np.float64),
                         [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], method="linear")
    return {name: float(point) for name, point in zip(names, points)}


def _request_scale(pos_lower, pos_upper):
    low, high = float(pos_lower), float(pos_upper)
    scale = 0.5 * (high - low)
    offset = 0.5 * (high + low)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("controller position bounds cannot scale a translation")
    return scale, offset


def official_peg_head_in_hole(observation, box_quat):
    """Official peg head in the stored hole frame. Quaternion is not renormalized."""
    obs = np.asarray(observation, dtype=np.float64)
    peg = obs[25:32]
    head = peg[:3] + _rotation_matrix(peg[3:7]) @ np.array((float(obs[32]), 0.0, 0.0))
    return _rotation_matrix(box_quat).T @ (head - obs[36:39])


def official_head_insertion_flags(head, radius, official_head_x_m=-0.015, proximal_axial_window_m=0.020):
    """Frozen labels. Officially inserted states are not proximal."""
    lateral = abs(float(head[1])) <= float(radius) and abs(float(head[2])) <= float(radius)
    axial = float(head[0])
    inserted = bool(axial >= official_head_x_m and lateral)
    proximal = bool(
        official_head_x_m - proximal_axial_window_m <= axial < official_head_x_m and lateral
    )
    return proximal, inserted


def summarize_expert_translation_realization(
    episodes, *, pos_lower=-0.1, pos_upper=0.1, norm_low=0.04, norm_high=0.06,
    tcp_position_slice=(18, 21), intervention_tcp_max_m=0.000843,
    assumption_ratio_below=0.5, official_head_x_m=-0.015, proximal_axial_window_m=0.020,
):
    """Compare stored expert translation requests with the next TCP displacement.

    ``action[t]`` is paired with observation ``t`` to ``t+1``. Normalized
    translation is not searched outside ``[norm_low, norm_high]``. Gripper sign
    is a request label, not a native grasp. Insertion proximity uses the official
    peg-head formula only when a stored box quaternion is supplied.
    """
    scale, offset = _request_scale(pos_lower, pos_upper)
    start, stop = tcp_position_slice
    rows = []
    for episode in episodes:
        actions = np.asarray(episode["actions"], dtype=np.float64)
        observations = np.asarray(episode["observations"], dtype=np.float64)
        root_poses = np.asarray(episode["root_poses"], dtype=np.float64)
        if observations.shape[0] != len(actions) + 1 or root_poses.shape[0] != len(actions) + 1:
            raise ValueError("action t requires observation and root pose rows t and t+1")
        box_quats = episode.get("box_quats")
        if box_quats is not None:
            box_quats = np.asarray(box_quats, dtype=np.float64)
        negative_at = next((index for index, action in enumerate(actions) if action[6] < 0.0), None)
        for index, action in enumerate(actions):
            request = action[:3]
            request_norm = float(np.linalg.norm(request))
            if request_norm < norm_low or request_norm > norm_high:
                continue
            physical_root = offset + scale * np.clip(request, -1.0, 1.0)
            world_rotation = _rotation_matrix(root_poses[index, 3:7])
            physical_world = world_rotation @ physical_root
            requested_norm = float(np.linalg.norm(physical_world))
            displacement = observations[index + 1, start:stop] - observations[index, start:stop]
            direction = physical_world / requested_norm
            projection = float(np.dot(displacement, direction))
            tcp_norm = float(np.linalg.norm(displacement))
            proximal = None
            if box_quats is not None:
                hole = official_peg_head_in_hole(observations[index], box_quats[index])
                proximal, _inserted = official_head_insertion_flags(
                    hole, observations[index, 35], official_head_x_m, proximal_axial_window_m,
                )
            rows.append({
                "episode_id": int(episode["episode_id"]),
                "action_index": int(index),
                "request_norm": request_norm,
                "physical_root_delta_m": [float(item) for item in physical_root],
                "physical_world_delta_m": [float(item) for item in physical_world],
                "tcp_displacement_m": [float(item) for item in displacement],
                "tcp_norm_m": tcp_norm,
                "projection_m": projection,
                "requested_physical_norm_m": requested_norm,
                "ratio_tcp_over_requested": tcp_norm / requested_norm,
                "ratio_projection_over_requested": projection / requested_norm,
                "gripper_request_negative": bool(action[6] < 0.0),
                "operational_phase": (
                    "pre_close_request" if negative_at is None or index < negative_at
                    else "at_or_after_close_request"
                ),
                "insertion_proximal": proximal,
            })

    def stratum(selected):
        selected = list(selected)
        return {
            "count": len(selected),
            "episodes": len({row["episode_id"] for row in selected}),
            "tcp_norm_m": _linear_quantiles([row["tcp_norm_m"] for row in selected]),
            "projection_m": _linear_quantiles([row["projection_m"] for row in selected]),
            "ratio_tcp_over_requested": _linear_quantiles(
                [row["ratio_tcp_over_requested"] for row in selected]),
            "ratio_projection_over_requested": _linear_quantiles(
                [row["ratio_projection_over_requested"] for row in selected]),
            "pre_close_request": sum(row["operational_phase"] == "pre_close_request" for row in selected),
            "at_or_after_close_request": sum(
                row["operational_phase"] == "at_or_after_close_request" for row in selected),
            "insertion_proximal": (
                "unavailable" if any(row["insertion_proximal"] is None for row in selected)
                else sum(bool(row["insertion_proximal"]) for row in selected)
            ),
            "tcp_norm_by_phase_m": {
                "pre_close_request": _linear_quantiles([
                    row["tcp_norm_m"] for row in selected
                    if row["operational_phase"] == "pre_close_request"]),
                "at_or_after_close_request": _linear_quantiles([
                    row["tcp_norm_m"] for row in selected
                    if row["operational_phase"] == "at_or_after_close_request"]),
                "insertion_proximal": (
                    "unavailable" if any(row["insertion_proximal"] is None for row in selected)
                    else _linear_quantiles([
                        row["tcp_norm_m"] for row in selected if row["insertion_proximal"]])),
                "not_insertion_proximal": (
                    "unavailable" if any(row["insertion_proximal"] is None for row in selected)
                    else _linear_quantiles([
                        row["tcp_norm_m"] for row in selected if not row["insertion_proximal"]])),
            },
        }

    negative = stratum(row for row in rows if row["gripper_request_negative"])
    nonnegative = stratum(row for row in rows if not row["gripper_request_negative"])
    if negative["count"] == 0:
        classification = "unavailable_empty_stratum"
        assumption = "unavailable_empty_stratum"
    else:
        low_q, high_q = negative["tcp_norm_m"]["q10"], negative["tcp_norm_m"]["q90"]
        if intervention_tcp_max_m < low_q:
            classification = "unusual_low"
        elif intervention_tcp_max_m > high_q:
            classification = "unusual_high"
        else:
            classification = "typical"
        assumption = (
            "refuted" if negative["ratio_tcp_over_requested"]["q50"] < assumption_ratio_below
            else "not_refuted"
        )
    return {
        "format": "peg-expert-translation-realization-v1",
        "native_grasp": "unavailable",
        "band": [norm_low, norm_high],
        "controller_scale_m": scale,
        "controller_offset_m": offset,
        "tcp_position_slice": [start, stop],
        "quantile_method": "numpy.quantile method=linear",
        "rows": rows,
        "strata": {
            "gripper_request_negative": negative,
            "gripper_request_nonnegative": nonnegative,
        },
        "decision": {
            "stratum": "gripper_request_negative",
            "intervention_tcp_max_m": float(intervention_tcp_max_m),
            "typical_rule": "intervention max TCP norm is inside closed expert [q10, q90]; below q10 is unusual_low",
            "assumption_rule": "refuted if expert median realized-TCP-norm / requested-physical-norm < 0.5",
            "classification": classification,
            "assumption_5mm_implies_5mm_tcp": assumption,
        },
    }


def summarize_proximal_expert_requests(
    episodes, *, pos_lower=-0.1, pos_upper=0.1, tcp_position_slice=(18, 21),
    official_head_x_m=-0.015, proximal_axial_window_m=0.020,
    intervention_request_norm=0.05, policy_window_max_request_norm=0.0373341420915306,
    band_low=0.04, band_high=0.06,
):
    """Summarize stored requests at already-proximal pre-action states.

    ``action[t]`` is labeled by observation ``t``. Observation ``T`` has no action
    and is not paired with ``action[T-1]``. Inserted states stay excluded by the
    frozen head predicate. A normalized request is a controller target, not
    realized motion; gripper sign is not native grasp.
    """
    scale, offset = _request_scale(pos_lower, pos_upper)
    start, stop = tcp_position_slice
    rows = []
    all_states = all_proximal = all_inserted = terminal_proximal = 0
    success_post_states = success_post_inserted = 0
    success_checked = False
    hole_center_step = 0.0
    box_quat_step = 0.0
    per_episode = {}
    for episode in episodes:
        actions = np.asarray(episode["actions"], dtype=np.float64)
        observations = np.asarray(episode["observations"], dtype=np.float64)
        root_poses = np.asarray(episode["root_poses"], dtype=np.float64)
        box_quats = np.asarray(episode["box_quats"], dtype=np.float64)
        if (observations.shape[0] != len(actions) + 1 or root_poses.shape[0] != len(actions) + 1
                or box_quats.shape[0] != len(actions) + 1):
            raise ValueError("action t requires observation, root pose, and box quaternion rows t and t+1")
        success = episode.get("success")
        if success is not None:
            success = np.asarray(success)
            if success.shape[0] != len(actions):
                raise ValueError("success[t] aligns with action[t] and observation[t+1], not observation[t]")
            success_checked = True
        episode_rows = 0
        for index in range(len(observations)):
            head = official_peg_head_in_hole(observations[index], box_quats[index])
            proximal, inserted = official_head_insertion_flags(
                head, observations[index, 35], official_head_x_m, proximal_axial_window_m,
            )
            all_states += 1
            all_proximal += int(proximal)
            all_inserted += int(inserted)
            if index == len(actions):
                terminal_proximal += int(proximal)
                continue
            if success is not None and bool(success[index]):
                post = official_peg_head_in_hole(observations[index + 1], box_quats[index + 1])
                _post_proximal, post_inserted = official_head_insertion_flags(
                    post, observations[index + 1, 35], official_head_x_m, proximal_axial_window_m,
                )
                success_post_states += 1
                success_post_inserted += int(post_inserted)
            if not proximal:
                continue
            action = actions[index]
            request = action[:3]
            request_norm = float(np.linalg.norm(request))
            physical_root = offset + scale * np.clip(request, -1.0, 1.0)
            physical_world = _rotation_matrix(root_poses[index, 3:7]) @ physical_root
            requested_norm = float(np.linalg.norm(physical_world))
            displacement = observations[index + 1, start:stop] - observations[index, start:stop]
            projection = None if requested_norm == 0.0 else float(np.dot(displacement, physical_world / requested_norm))
            next_head = official_peg_head_in_hole(observations[index + 1], box_quats[index + 1])
            hole_request = _rotation_matrix(box_quats[index]).T @ physical_world
            hole_center_step = max(
                hole_center_step,
                float(np.max(np.abs(observations[index + 1, 36:39] - observations[index, 36:39]))),
            )
            box_quat_step = max(
                box_quat_step, float(np.max(np.abs(box_quats[index + 1] - box_quats[index]))),
            )
            rows.append({
                "episode_id": int(episode["episode_id"]),
                "action_index": int(index),
                "request_norm": request_norm,
                "tcp_norm_m": float(np.linalg.norm(displacement)),
                "tcp_projection_m": projection,
                "requested_hole_plus_x_m": float(hole_request[0]),
                "realized_head_x_advance_m": float(next_head[0] - head[0]),
                "rotation_request_norm": float(np.linalg.norm(action[3:6])),
                "gripper_request_negative": bool(action[6] < 0.0),
                "in_frozen_translation_band": bool(band_low <= request_norm <= band_high),
            })
            episode_rows += 1
        per_episode[int(episode["episode_id"])] = episode_rows

    def quantiles(key):
        return _linear_quantiles([row[key] for row in rows if row[key] is not None])

    if not rows:
        classification = "unavailable_empty"
    else:
        median = quantiles("request_norm")["q50"]
        if median > intervention_request_norm:
            classification = "larger"
        elif median < policy_window_max_request_norm:
            classification = "smaller"
        else:
            classification = "comparable"
    return {
        "format": "peg-proximal-expert-request-v1",
        "native_grasp": "unavailable",
        "translation_target_is_not_realized_motion": True,
        "gripper_sign_is_not_native_grasp": True,
        "rotation_request_norm_is_not_a_realized_angle": True,
        "predicate": {
            "official_head_x_m": float(official_head_x_m),
            "proximal_axial_window_m": float(proximal_axial_window_m),
            "rule": (
                "proximal iff official_head_x - window <= head_x < official_head_x "
                "and |head_y|, |head_z| <= stored radius; inserted head_x >= official_head_x is excluded"
            ),
        },
        "timing": "observation[t] labels action[t]; observation[t+1] is the following state; index T is terminal-only",
        "controller_scale_m": scale,
        "band_reported_not_applied": [band_low, band_high],
        "counts": {
            "episodes": len(episodes),
            "all_states": all_states,
            "all_states_proximal": all_proximal,
            "all_states_inserted": all_inserted,
            "terminal_only_proximal": terminal_proximal,
            "actionable_proximal": len(rows),
            "episodes_with_actionable_proximal": sum(value > 0 for value in per_episode.values()),
            "max_actionable_rows_in_one_episode": max(per_episode.values(), default=0),
            "in_frozen_translation_band": sum(row["in_frozen_translation_band"] for row in rows),
            "gripper_request_negative": sum(row["gripper_request_negative"] for row in rows),
            "gripper_request_nonnegative": sum(not row["gripper_request_negative"] for row in rows),
            "zero_translation_request": sum(row["tcp_projection_m"] is None for row in rows),
            "success_post_states": None if not success_checked else success_post_states,
            "success_post_states_reconstructed_inserted": None if not success_checked else success_post_inserted,
            "selected_hole_center_step_max_abs_m": hole_center_step,
            "selected_box_quat_step_max_abs": box_quat_step,
        },
        "request_norm": quantiles("request_norm"),
        "tcp_norm_m": quantiles("tcp_norm_m"),
        "tcp_projection_m": quantiles("tcp_projection_m"),
        "requested_hole_plus_x_m": quantiles("requested_hole_plus_x_m"),
        "realized_head_x_advance_m": quantiles("realized_head_x_advance_m"),
        "rotation_request_norm": quantiles("rotation_request_norm"),
        "decision": {
            "quantity": "median stored normalized translation request L2",
            "intervention_request_norm": float(intervention_request_norm),
            "policy_window_max_request_norm": float(policy_window_max_request_norm),
            "rule": (
                "larger if median > intervention request; smaller if median < policy-window max; "
                "comparable if median is in that closed interval; empty if no actionable proximal state"
            ),
            "classification": classification,
            "interpretation": "descriptive only; not matched to the intervention pose; not a root-cause claim",
        },
        "rows": rows,
    }


def load_fixed_training_episodes(dataset_path):
    """Read every stored fixed-training episode. Does not modify the file."""
    dataset_path = Path(dataset_path)
    episodes = []
    root_min = np.full(7, np.inf)
    root_max = np.full(7, -np.inf)
    qpos_error = 0.0
    peg_error = 0.0
    with h5py.File(dataset_path, "r") as h5:
        names = [f"traj_{index}" for index in range(len(h5))]
        if set(names) != set(h5.keys()):
            raise ValueError("dataset episode ids are not traj_0..traj_N")
        for episode_id, name in enumerate(names):
            group = h5[name]
            actions = np.asarray(group["actions"])
            observations = np.asarray(group["obs"])
            root_poses = np.asarray(group["env_states/articulations/panda_wristcam"][:, 0:7])
            box_quats = np.asarray(group["env_states/actors/box_with_hole"][:, 3:7])
            success = np.asarray(group["success"])
            if (actions.ndim != 2 or actions.shape[1] != 7 or observations.shape != (len(actions) + 1, 39)
                    or root_poses.shape[0] != len(actions) + 1 or box_quats.shape[0] != len(actions) + 1
                    or success.shape[0] != len(actions)):
                raise ValueError(f"{name} does not have T actions and T+1 39d observations")
            articulation = np.asarray(group["env_states/articulations/panda_wristcam"])
            peg = np.asarray(group["env_states/actors/peg"])
            qpos_error = max(qpos_error, float(np.max(np.abs(observations[:, 0:9] - articulation[:, 13:22]))))
            peg_error = max(peg_error, float(np.max(np.abs(observations[:, 25:32] - peg[:, 0:7]))))
            root_min = np.minimum(root_min, root_poses.min(axis=0))
            root_max = np.maximum(root_max, root_poses.max(axis=0))
            episodes.append({
                "episode_id": episode_id,
                "actions": actions,
                "observations": observations,
                "root_poses": root_poses,
                "box_quats": box_quats,
                "success": success,
            })
    schema = {
        "timing": "action[t] uses observation[t] and observation[t+1]; success[t] labels observation[t+1]",
        "tcp_position_slice": [18, 21],
        "peg_pose_slice": [25, 32],
        "peg_half_length_index": 32,
        "hole_radius_index": 35,
        "hole_center_slice": [36, 39],
        "root_pose": "env_states/articulations/panda_wristcam[:, 0:7], xyz+wxyz",
        "box_quat": "env_states/actors/box_with_hole[:, 3:7], wxyz",
        "native_grasp_stored": False,
        "qpos_max_abs_error": qpos_error,
        "peg_pose_max_abs_error": peg_error,
        "root_pose_min": root_min.tolist(),
        "root_pose_max": root_max.tolist(),
    }
    return episodes, schema


def _dataset_summary(dataset_path, episodes, schema, result):
    summary = {key: value for key, value in result.items() if key != "rows"}
    summary["dataset"] = str(dataset_path)
    summary["dataset_sha256"] = sha256_file(dataset_path)
    summary["episodes"] = len(episodes)
    summary["schema"] = schema
    summary["rows_retained"] = False
    return {"summary": summary, "rows": result["rows"]}


def realize_fixed_training_translation(dataset_path, pos_lower=-0.1, pos_upper=0.1):
    """Load every stored fixed-1000 episode and summarize the frozen translation band."""
    dataset_path = Path(dataset_path)
    episodes, schema = load_fixed_training_episodes(dataset_path)
    result = summarize_expert_translation_realization(
        episodes, pos_lower=pos_lower, pos_upper=pos_upper,
    )
    return _dataset_summary(dataset_path, episodes, schema, result)


def realize_proximal_expert_requests(dataset_path, pos_lower=-0.1, pos_upper=0.1):
    """Load every stored episode and summarize already-proximal actionable requests."""
    dataset_path = Path(dataset_path)
    episodes, schema = load_fixed_training_episodes(dataset_path)
    result = summarize_proximal_expert_requests(
        episodes, pos_lower=pos_lower, pos_upper=pos_upper,
    )
    return _dataset_summary(dataset_path, episodes, schema, result)


def _classify_policy_translation_median(median, low_command_norm, intervention_request_norm):
    if median < low_command_norm:
        return "replicates_low_commands_on_evidence"
    if median > intervention_request_norm:
        return "low_command_profile_not_reproduced"
    return "inconclusive"


def compare_proximal_policy_first_actions(
    episodes, get_action, *, seed=1907, batch_size=64, pos_lower=-0.1, pos_upper=0.1,
    official_head_x_m=-0.015, proximal_axial_window_m=0.020,
    low_command_norm=0.0373341420915306, intervention_request_norm=0.05,
):
    """Compare official first executed translations with aligned expert actions.

    Proximal rows come from the existing predicate. History is observation
    ``t-1, t``, repeating observation 0 when ``t`` is 0. ``get_action`` must be
    the official executed-chunk seam, shape ``(batch, 8, 7)``; index 0 is the
    first executed action. One seed is applied once, then every call uses
    ``batch_size`` rows. A short final batch is padded by repeating the last
    real history and those outputs are discarded. This is not closed-loop RNG.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    scale, offset = _request_scale(pos_lower, pos_upper)
    ordered = sorted(episodes, key=lambda episode: int(episode["episode_id"]))
    if len({int(episode["episode_id"]) for episode in ordered}) != len(ordered):
        raise ValueError("episode ids must be unique")
    expert = summarize_proximal_expert_requests(
        ordered, pos_lower=pos_lower, pos_upper=pos_upper,
        official_head_x_m=official_head_x_m, proximal_axial_window_m=proximal_axial_window_m,
        intervention_request_norm=intervention_request_norm,
        policy_window_max_request_norm=low_command_norm,
    )
    by_id = {int(episode["episode_id"]): episode for episode in ordered}
    selected = []
    for row in expert["rows"]:
        episode = by_id[int(row["episode_id"])]
        index = int(row["action_index"])
        observations = np.asarray(episode["observations"], dtype=np.float64)
        history = np.stack((observations[max(0, index - 1)], observations[index]))
        selected.append({
            "episode_id": int(row["episode_id"]),
            "action_index": index,
            "history": history,
            "expert": np.asarray(episode["actions"][index, :3], dtype=np.float64),
            "root_quat": np.asarray(episode["root_poses"][index, 3:7], dtype=np.float64),
            "box_quat": np.asarray(episode["box_quats"][index], dtype=np.float64),
        })
    pad = 0
    calls = 0
    predicted = np.zeros((0, 3), dtype=np.float64)
    if selected:
        histories = np.stack([row["history"] for row in selected]).astype(np.float32)
        pad = (-len(selected)) % batch_size
        if pad:
            histories = np.concatenate((histories, np.repeat(histories[-1:], pad, axis=0)), axis=0)
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        chunks = []
        executed = (batch_size, peg_orientation_base_dp.ACT_HORIZON, peg_orientation_base_dp.ACT_DIM)
        for start in range(0, len(histories), batch_size):
            batch = torch.as_tensor(histories[start:start + batch_size])
            chunk = get_action(batch)
            if not torch.is_tensor(chunk):
                chunk = torch.as_tensor(chunk)
            if tuple(chunk.shape) != executed:
                raise ValueError(
                    f"official get_action must return the executed chunk {executed}, got {tuple(chunk.shape)}"
                )
            chunks.append(chunk[:, 0, :3].detach().cpu().numpy())
            calls += 1
        predicted = np.concatenate(chunks, axis=0)[:len(selected)].astype(np.float64)
        if not np.isfinite(predicted).all():
            raise ValueError("policy first-action translation is not finite")

    def target(translation, root_quat, box_quat):
        physical = offset + scale * np.clip(translation, -1.0, 1.0)
        hole = _rotation_matrix(box_quat).T @ (_rotation_matrix(root_quat) @ physical)
        return physical, float(hole[0])

    paired = []
    outside = 0
    for row, policy_translation in zip(selected, predicted):
        expert_translation = row["expert"]
        policy_norm = float(np.linalg.norm(policy_translation))
        expert_norm = float(np.linalg.norm(expert_translation))
        policy_physical, policy_hole = target(policy_translation, row["root_quat"], row["box_quat"])
        expert_physical, expert_hole = target(expert_translation, row["root_quat"], row["box_quat"])
        outside += int(np.sum(np.abs(policy_translation) > 1.0))
        paired.append({
            "episode_id": row["episode_id"],
            "policy_norm": policy_norm,
            "expert_norm": expert_norm,
            "error_norm": float(np.linalg.norm(policy_translation - expert_translation)),
            "signed_norm_error": policy_norm - expert_norm,
            "ratio": None if expert_norm == 0.0 else policy_norm / expert_norm,
            "policy_physical_norm": float(np.linalg.norm(policy_physical)),
            "expert_physical_norm": float(np.linalg.norm(expert_physical)),
            "physical_error_norm": float(np.linalg.norm(policy_physical - expert_physical)),
            "policy_hole": policy_hole,
            "expert_hole": expert_hole,
            "hole_error": policy_hole - expert_hole,
        })

    def quantiles(key):
        return _linear_quantiles([row[key] for row in paired if row[key] is not None])

    episode_norms = {}
    for row in paired:
        episode_norms.setdefault(row["episode_id"], []).append(row["policy_norm"])
    episode_classes = {
        "replicates_low_commands_on_evidence": 0,
        "inconclusive": 0,
        "low_command_profile_not_reproduced": 0,
    }
    for norms in episode_norms.values():
        episode_classes[_classify_policy_translation_median(
            _linear_quantiles(norms)["q50"], low_command_norm, intervention_request_norm,
        )] += 1
    policy_median = None if not paired else quantiles("policy_norm")["q50"]
    classification = (
        "unavailable_empty" if policy_median is None
        else _classify_policy_translation_median(
            policy_median, low_command_norm, intervention_request_norm,
        )
    )
    return {
        "format": "peg-proximal-policy-first-action-v1",
        "native_controls": 0,
        "optimizer_updates": 0,
        "teacher_forcing": True,
        "closed_loop": False,
        "same_rng_as_closed_loop": False,
        "rotation_analyzed": False,
        "gripper_sign_is_not_native_grasp": True,
        "translation_target_is_not_realized_motion": True,
        "native_action_scaling": "unchanged",
        "predicate": expert["predicate"],
        "timing": "history is observation[t-1], observation[t], with observation[0] repeated at t=0; first executed action is get_action chunk index 0",
        "controller_scale_m": scale,
        "noise_profile": {
            "seed_once": int(seed),
            "seeded_immediately_before_get_action": bool(selected),
            "batch_size": batch_size,
            "inference_calls": calls,
            "padded_rows_discarded": pad,
            "sampler": "official Agent.get_action via the supplied executed-chunk seam",
            "not_rowwise_reseed": True,
            "not_closed_loop_scene_rng": True,
        },
        "counts": {
            "episodes": len(ordered),
            "all_states_proximal": expert["counts"]["all_states_proximal"],
            "terminal_only_proximal": expert["counts"]["terminal_only_proximal"],
            "actionable_proximal": len(paired),
            "episodes_with_actionable_proximal": len(episode_norms),
            "max_actionable_rows_in_one_episode": max((len(norms) for norms in episode_norms.values()), default=0),
            "policy_norm_lt_window_max": sum(row["policy_norm"] < low_command_norm for row in paired),
            "policy_norm_le_intervention": sum(row["policy_norm"] <= intervention_request_norm for row in paired),
            "policy_norm_lt_expert": sum(row["policy_norm"] < row["expert_norm"] for row in paired),
            "policy_norm_le_expert": sum(row["policy_norm"] <= row["expert_norm"] for row in paired),
            "policy_norm_gt_expert": sum(row["policy_norm"] > row["expert_norm"] for row in paired),
            "policy_translation_components_abs_gt_1": outside,
            "zero_expert_translation_norm": sum(row["ratio"] is None for row in paired),
            "inference_calls": calls,
            "padded_rows_discarded": pad,
            "episode_median_replicates_low_commands": episode_classes["replicates_low_commands_on_evidence"],
            "episode_median_inconclusive": episode_classes["inconclusive"],
            "episode_median_not_reproduced": episode_classes["low_command_profile_not_reproduced"],
        },
        "policy_translation_norm": quantiles("policy_norm"),
        "expert_translation_norm": quantiles("expert_norm"),
        "paired_translation_error_norm": quantiles("error_norm"),
        "signed_norm_error": quantiles("signed_norm_error"),
        "norm_ratio_policy_over_expert": quantiles("ratio"),
        "policy_physical_target_norm_m": quantiles("policy_physical_norm"),
        "expert_physical_target_norm_m": quantiles("expert_physical_norm"),
        "paired_physical_target_error_norm_m": quantiles("physical_error_norm"),
        "policy_hole_plus_x_m": quantiles("policy_hole"),
        "expert_hole_plus_x_m": quantiles("expert_hole"),
        "hole_plus_x_error_m": quantiles("hole_error"),
        "decision": {
            "quantity": "pooled median policy first-executed translation L2",
            "low_command_norm": float(low_command_norm),
            "intervention_request_norm": float(intervention_request_norm),
            "low_command_comparison": "strict <",
            "not_reproduced_comparison": "strict >",
            "intermediate_interval": "closed, both endpoints inclusive",
            "equal_to_either_threshold_is": "inconclusive",
            "rule": (
                "replicates_low_commands_on_evidence if median < low_command_norm; "
                "low_command_profile_not_reproduced if median > intervention_request_norm; "
                "inconclusive if low_command_norm <= median <= intervention_request_norm; "
                "unavailable_empty if no actionable proximal row"
            ),
            "classification": classification,
            "episode_median_counts_do_not_reclassify": True,
            "interpretation": (
                "teacher-forced on stored expert states; not a cause of the 13/50 plateau; "
                "batched noise is not the closed-loop scene RNG"
            ),
        },
        "rows_retained": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-catalog", type=Path)
    parser.add_argument("--scene-indices", type=int, nargs="+")
    parser.add_argument("--assert-known-failure", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1907)
    parser.add_argument("--windows", type=int, default=128)
    parser.add_argument("--matched-phases", action="store_true")
    parser.add_argument("--episode-count", type=int, default=32)
    parser.add_argument("--close-windows", type=int, default=64)
    parser.add_argument("--replay-episode", type=int)
    parser.add_argument("--compare-prefix-episode", type=int)
    parser.add_argument("--prefix-steps", type=int, default=3)
    parser.add_argument("--grasp-chunk-episode", type=int)
    parser.add_argument("--audit-training-horizon", action="store_true")
    parser.add_argument("--training-phase-episodes", type=int, nargs="+")
    parser.add_argument("--replayed-grasp-differential", action="store_true")
    parser.add_argument("--full-rollout-training-episodes", type=int, nargs="+")
    parser.add_argument("--episode-control-cap", type=int, default=500)
    parser.add_argument("--insertion-trace-window", type=int, nargs=2)
    parser.add_argument("--execution-horizons", type=int, nargs="+", default=[8, 2])
    parser.add_argument("--hole-x-intervention", type=Path)
    parser.add_argument("--prefix-controls", type=int, default=190)
    parser.add_argument("--intervention-controls", type=int, default=8)
    parser.add_argument("--physical-hole-plus-x-m", type=float, default=0.005)
    parser.add_argument("--expert-translation-realization", action="store_true")
    parser.add_argument("--proximal-expert-requests", action="store_true")
    parser.add_argument("--proximal-policy-first-actions", action="store_true")
    parser.add_argument("--reproduce-sampler-input", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.reproduce_sampler_input is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required")
        result = reproduce_saved_sampler_inputs(args.checkpoint, args.reproduce_sampler_input, args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({
            "sampler_calls": result["sampler_calls"],
            "matched": [row["matched"] for row in result["calls"]],
            "max_abs_error": [row["max_abs_error"] for row in result["calls"]],
        }, indent=2))
        return
    if args.proximal_policy_first_actions:
        if args.checkpoint is None or args.data is None:
            parser.error("--checkpoint and --data are required")
        torch.set_num_threads(1)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        adapter = peg_orientation_base_dp.PegBaseAdapter(args.checkpoint, args.device)
        episodes, schema = load_fixed_training_episodes(args.data)
        result = compare_proximal_policy_first_actions(
            episodes, adapter.get_action, seed=args.seed, batch_size=args.batch_size,
        )
        result["checkpoint"] = str(args.checkpoint)
        result["checkpoint_sha256"] = sha256_file(args.checkpoint)
        result["weights"] = adapter.metadata["weights"]
        result["policy_contract"] = {
            "obs_dim": adapter.metadata["obs_dim"],
            "act_dim": adapter.metadata["act_dim"],
            "obs_horizon": adapter.metadata["obs_horizon"],
            "pred_horizon": adapter.metadata["pred_horizon"],
            "act_horizon": adapter.metadata["act_horizon"],
            "first_executed_index": 0,
        }
        result["dataset"] = str(args.data)
        result["dataset_sha256"] = sha256_file(args.data)
        result["schema"] = schema
        result["device"] = str(args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"counts": result["counts"], "decision": result["decision"]}, indent=2))
        return
    if args.proximal_expert_requests:
        if args.data is None:
            parser.error("--data is required")
        result = realize_proximal_expert_requests(args.data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result["summary"], indent=2) + "\n")
        print(json.dumps({
            "counts": result["summary"]["counts"],
            "decision": result["summary"]["decision"],
        }, indent=2))
        return
    if args.expert_translation_realization:
        if args.data is None:
            parser.error("--data is required")
        result = realize_fixed_training_translation(args.data)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result["summary"], indent=2) + "\n")
        print(json.dumps(result["summary"]["decision"], indent=2))
        return
    if args.hole_x_intervention is not None:
        if args.checkpoint is None or args.validation_catalog is None or not args.scene_indices:
            parser.error("--checkpoint, --validation-catalog, and --scene-indices are required")
        result = run_hole_x_intervention(
            args.checkpoint, args.validation_catalog, args.output, args.hole_x_intervention,
            args.prefix_controls, args.intervention_controls, args.physical_hole_plus_x_m,
            args.scene_indices, args.seed, args.device, args.episode_control_cap,
            insertion_trace_window=args.insertion_trace_window,
        )
        print(json.dumps(result, indent=2))
        return
    if args.validation_catalog is not None:
        if args.checkpoint is None or not args.scene_indices:
            parser.error("--checkpoint and --scene-indices are required for validation phase diagnosis")
        result = run_validation_phase_diagnosis(
            args.checkpoint, args.validation_catalog, args.output, args.scene_indices,
            args.episode_control_cap, args.seed, args.device,
            insertion_trace_window=args.insertion_trace_window,
        )
        print(json.dumps(result, indent=2))
        if args.assert_known_failure:
            if len(args.scene_indices) != 1:
                parser.error("--assert-known-failure requires exactly one scene index")
            reproduced = known_validation_failure_reproduced(
                result["records"][0], args.scene_indices[0], args.seed, args.episode_control_cap,
            )
            raise SystemExit(3 if reproduced else 4)
        return
    if args.data is None or args.manifest is None:
        parser.error("--data and --manifest are required for this probe")
    if args.audit_training_horizon:
        result = audit_training_horizon(args.data, args.manifest, args.output)
    elif args.full_rollout_training_episodes is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for full policy rollouts")
        result = run_training_rollout_horizon_comparison(
            args.checkpoint, args.data, args.manifest, args.output,
            args.full_rollout_training_episodes, args.episode_control_cap,
            args.execution_horizons, args.seed,
        )
    elif args.replayed_grasp_differential:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for checkpoint probes")
        result = compare_policy_from_replayed_grasp(
            args.checkpoint, args.data, args.manifest, args.output
        )
    elif args.training_phase_episodes is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for checkpoint probes")
        result = diagnose_preregistered_training_phases(
            args.checkpoint, args.data, args.manifest, args.training_phase_episodes, args.output
        )
    elif args.matched_phases:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for checkpoint probes")
        result = diagnose_matched_phases(args.checkpoint, args.data, args.manifest, args.output,
                                         args.seed, args.episode_count, args.close_windows)
    elif args.replay_episode is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for native replay probes")
        result = replay_expert_episode(args.checkpoint, args.data, args.manifest,
                                       args.replay_episode, args.output, args.seed)
    elif args.compare_prefix_episode is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for prefix probes")
        result = compare_native_prefix(args.checkpoint, args.data, args.manifest,
                                       args.compare_prefix_episode, args.output,
                                       args.seed, args.prefix_steps)
    elif args.grasp_chunk_episode is not None:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for grasp-chunk probes")
        result = probe_grasp_chunk(args.checkpoint, args.data, args.manifest,
                                   args.grasp_chunk_episode, args.output)
    else:
        if args.checkpoint is None:
            parser.error("--checkpoint is required for checkpoint probes")
        result = diagnose(args.checkpoint, args.data, args.manifest, args.output,
                          args.seed, args.windows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
