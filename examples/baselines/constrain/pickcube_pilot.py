"""Bounded PickCube direct and obstacle pilot collection and 7d data loading."""

import argparse
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree

from examples.baselines.constrain.pickcube_expert import (
    PickCubeExpertConfig,
    make_pickcube_env,
    run_pickcube_expert,
)


@dataclass(frozen=True)
class PickCubePilotConfig:
    seed: int = 42
    target_per_class: int = 20
    max_attempts_per_class: int = 200
    max_steps: int = 500
    obs_horizon: int = 2
    pred_horizon: int = 16


def _default_attempt(kind, seed, rng, config):
    """Run a real bounded expert attempt; failed or truncated tasks are rejected."""
    env = make_pickcube_env()
    try:
        base_xy = env.unwrapped.agent.robot.pose.p[0, :2].detach().cpu().numpy()
        # Sample a varied, reachable layout on the robot-facing side of the
        # table; validate_layout remains the geometric authority.
        for _ in range(32):
            start = np.array([rng.uniform(-0.18, -0.02), rng.uniform(-0.06, 0.12)])
            goal = np.array([rng.uniform(0.02, 0.18), rng.uniform(-0.06, 0.12)])
            try:
                result = run_pickcube_expert(
                    env, cube_xy=start, goal_xy=goal, detour=(kind == "obstacle"),
                    obstacle_enabled=False, config=PickCubeExpertConfig(
                        seed=seed, max_steps=config.max_steps if kind == "direct" else max(1, config.max_steps // 2)
                    ),
                )
                break
            except ValueError as exc:
                if "invalid_pickcube_layout" not in str(exc) and "invalid_pickcube_waypoint" not in str(exc):
                    raise
        else:
            return None, "geometry_retry_exhausted"
        if kind == "obstacle":
            if not result["safety"]["native_success"]:
                return None, "template_native_timeout_or_unsuccessful"
            template_steps = len(result["actions"])
            if template_steps >= config.max_steps:
                return None, "attempt_control_step_budget_exhausted"
            template_state, template_controller = result["initial_snapshot"]
            template_observation = result["initial_observation"]
            # Candidate cylinders block the straight S-G object segment but
            # are kept away from the measured successful detour sweep.
            from examples.baselines.constrain.collect_constructive import segment_point_distance
            sweep = np.asarray(result["trace"]["swept_xy"], dtype=np.float64)
            center = None
            side = np.sign(np.cross(goal - start, base_xy - start))
            direction = (goal - start) / np.linalg.norm(goal - start)
            lateral = np.array([-direction[1], direction[0]])
            candidates = np.asarray([
                start + t * (goal - start) - side * lateral * offset
                for t in np.linspace(0.40, 0.60, 21)
                for offset in np.linspace(0.0, 0.06, 25)
            ])
            distances, _ = cKDTree(sweep).query(candidates, k=1, workers=1)
            blocking = np.asarray([
                segment_point_distance(candidate, start, goal) <= 0.063
                and np.linalg.norm(candidate - start) > 0.068
                and np.linalg.norm(candidate - goal) > 0.068
                and -0.25 <= candidate[0] <= 0.25
                and -0.25 <= candidate[1] <= 0.35
                for candidate in candidates
            ])
            feasible = np.flatnonzero(blocking & (distances >= 0.035))
            if len(feasible):
                chosen = feasible[np.argmax(distances[feasible])]
                center = candidates[chosen]
            if center is None:
                return None, f"cylinder_geometry_retry_exhausted:{len(candidates)}"
            result = run_pickcube_expert(
                env, cube_xy=start, goal_xy=goal, detour=True, obstacle_enabled=True,
                obstacle_center=center, initial_snapshot=(template_state, template_controller),
                config=PickCubeExpertConfig(seed=seed, max_steps=config.max_steps - template_steps),
            )
            if not result["safety"]["safe_success"]:
                if not result["safety"]["native_success"]:
                    return None, "obstacle_native_timeout_or_unsuccessful"
                return None, "obstacle_collision_or_crossing"
            if not _same_initial_except_obstacle(template_state, result["initial_state"]):
                return None, "initial_state_restore_mismatch"
            if not _tree_equal(template_controller, result["initial_snapshot"][1]):
                return None, "controller_restore_mismatch"
            if not _observation_matches_except_obstacle(
                template_observation, result["initial_observation"]
            ):
                return None, "initial_observation_restore_mismatch"
            result["trace"]["obstacle_center"] = center.tolist()
            result["trace"]["cylinder_candidates_tested"] = int(len(candidates))
            result["trace"]["cylinder_sweep_clearance_m"] = float(distances[chosen])
            result["trace"]["initial_state_identity"] = {
                "template_native_state_sha256": _value_hash({
                    key: value for key, value in template_state.items()
                    if key != "pickcube_obstacle"
                }),
                "obstacle_native_state_sha256": _value_hash({
                    key: value for key, value in result["initial_state"].items()
                    if key != "pickcube_obstacle"
                }),
                "template_controller_sha256": _value_hash(template_controller),
                "obstacle_controller_sha256": _value_hash(result["initial_snapshot"][1]),
                "native_observation_equal_except_last_three_obstacle_values": True,
            }
        elif not result["safety"]["native_success"]:
            return None, "native_timeout_or_unsuccessful"
        return {
            "actions": result["actions"],
            "obs": np.stack([np.asarray(x).reshape(-1) for x in result["observations"]]),
            "states": result["states"],
            "infos": result["infos"],
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "success": True,
            "info": {"native_success": True, "safe_success": True},
            "seed": seed,
            "layout": result["layout"],
            "trace": result["trace"],
            "info": {"native_success": result["safety"]["native_success"],
                     "safe_success": result["safety"]["safe_success"]},
        }, None
    finally:
        env.close()


def _write_value(group, key, value):
    if isinstance(value, dict):
        child = group.create_group(str(key))
        for subkey, subvalue in value.items():
            _write_value(child, subkey, subvalue)
    else:
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        group.create_dataset(str(key), data=np.asarray(value))


def _same_initial_except_obstacle(expected, actual):
    for key in expected:
        if key == "pickcube_obstacle":
            continue
        left, right = expected[key], actual[key]
        if isinstance(left, dict):
            if not _same_initial_except_obstacle(left, right):
                return False
        elif not torch.equal(left, right):
            return False
    return True


def _tree_equal(expected, actual):
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _tree_equal(expected[key], actual[key]) for key in expected
        )
    if torch.is_tensor(expected):
        return torch.equal(expected, actual)
    return np.array_equal(expected, actual)


def _value_hash(value):
    digest = hashlib.sha256()

    def visit(node, prefix=""):
        if isinstance(node, dict):
            for name in sorted(node):
                visit(node[name], f"{prefix}/{name}")
        else:
            array = node.detach().cpu().numpy() if torch.is_tensor(node) else np.asarray(node)
            digest.update(prefix.encode())
            digest.update(np.ascontiguousarray(array).tobytes())

    visit(value)
    return digest.hexdigest()


def _observation_matches_except_obstacle(template_obs, obstacle_obs):
    left = torch.as_tensor(template_obs).reshape(-1)
    right = torch.as_tensor(obstacle_obs).reshape(-1)
    return left.numel() >= 3 and left.shape == right.shape and torch.allclose(
        left[:-3], right[:-3], atol=1e-6, rtol=0
    )


def _stack_states(states):
    if isinstance(states[0], dict):
        return {key: _stack_states([state[key] for state in states]) for key in states[0]}
    values = [value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value) for value in states]
    return np.concatenate(values, axis=0)


def _clone_state(state):
    if isinstance(state, dict):
        return {key: _clone_state(value) for key, value in state.items()}
    return state.detach().clone() if torch.is_tensor(state) else np.array(state, copy=True)


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"not JSON serialisable: {type(value)}")


def _trace_metadata(trace):
    trace = dict(trace)
    if "swept_xy" in trace:
        trace["swept_xy_point_count"] = len(trace.pop("swept_xy"))
    return _jsonable(trace)


def collect_pilot(output_dir, config=PickCubePilotConfig(), attempt_runner=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    h5_path = output_dir / "trajectory.h5"
    report = {
        "config": asdict(config), "started_unix": time.time(), "classes": {},
        "complete": False, "attempts": [],
    }
    runner = attempt_runner or _default_attempt
    accepted = []
    try:
        for class_index, kind in enumerate(("direct", "obstacle")):
            rng = np.random.default_rng(config.seed + class_index)
            counts = Counter()
            class_accepted = 0
            started = time.monotonic()
            for attempt_index in range(config.max_attempts_per_class):
                seed = int(rng.integers(0, 2**31 - 1))
                attempt_started = time.monotonic()
                trajectory, reason = runner(kind, seed, rng, config)
                counts["complete_attempts"] += 1
                record = {"class": kind, "attempt": attempt_index + 1, "seed": seed,
                          "elapsed_seconds": time.monotonic() - attempt_started}
                if trajectory is None:
                    counts[f"failure:{reason or 'unspecified'}"] += 1
                    record["failure"] = reason or "unspecified"
                    report["attempts"].append(record)
                    continue
                if not trajectory.get("success") or not trajectory.get("info", {}).get("native_success"):
                    counts["failure:native_unsuccessful"] += 1
                    record["failure"] = "native_unsuccessful"
                    report["attempts"].append(record)
                    continue
                if kind == "obstacle" and not trajectory.get("info", {}).get("safe_success"):
                    counts["failure:unsafe"] += 1
                    record["failure"] = "unsafe"
                    report["attempts"].append(record)
                    continue
                actions = np.asarray(trajectory["actions"])
                obs = np.asarray(trajectory["obs"])
                if actions.ndim != 2 or actions.shape[1] != 7 or obs.shape[0] != len(actions) + 1:
                    raise ValueError("accepted trajectory must have 7d actions and T+1 observations")
                if len(trajectory["states"]) != len(actions) + 1:
                    raise ValueError("accepted trajectory must have T+1 full states")
                trajectory.update(kind=kind, seed=seed)
                accepted.append(trajectory)
                class_accepted += 1
                counts["accepted"] += 1
                record.update(accepted=True, steps=len(actions))
                record["cylinder_candidates_tested"] = int(
                    trajectory.get("trace", {}).get("cylinder_candidates_tested", 0)
                )
                report["attempts"].append(record)
                if class_accepted >= config.target_per_class:
                    break
            report["classes"][kind] = {
                "complete_attempts": counts["complete_attempts"],
                "accepted": class_accepted,
                "failures": {key[8:]: value for key, value in counts.items() if key.startswith("failure:")},
                "stage_counts": {
                    "geometry_failures": sum(
                        value for key, value in counts.items()
                        if key.startswith("failure:cylinder_geometry")
                    ),
                    "waypoint_failures": sum(
                        value for key, value in counts.items()
                        if key.startswith("failure:waypoint")
                    ),
                    "state_restore_failures": counts.get("failure:initial_state_restore_mismatch", 0),
                    "template_executions": counts["complete_attempts"] if kind == "obstacle" else 0,
                    "cylinder_candidate_evaluations": sum(
                        int(attempt.get("cylinder_candidates_tested", 0))
                        for attempt in report["attempts"] if attempt["class"] == kind
                    ),
                },
                "elapsed_seconds": time.monotonic() - started,
            }
        report["complete"] = all(
            report["classes"].get(kind, {}).get("accepted") == config.target_per_class
            for kind in ("direct", "obstacle")
        )
        if accepted:
            with h5py.File(h5_path, "w") as handle:
                for i, trajectory in enumerate(accepted):
                    group = handle.create_group(f"traj_{i}")
                    group.attrs["kind"] = trajectory["kind"]
                    group.attrs["seed"] = trajectory["seed"]
                    group.create_dataset("actions", data=trajectory["actions"])
                    group.create_dataset("obs", data=trajectory["obs"])
                    success = np.zeros(len(trajectory["actions"])+1, dtype=np.uint8)
                    success[-1] = 1
                    group.create_dataset("success", data=success)
                    group.create_dataset("terminated", data=trajectory["terminated"])
                    group.create_dataset("truncated", data=trajectory["truncated"])
                    states = group.create_group("env_states")
                    for key, value in _stack_states(trajectory["states"]).items():
                        _write_value(states, key, value)
            report["trajectory_sha256"] = hashlib.sha256(h5_path.read_bytes()).hexdigest()
        report["finished_unix"] = time.time()
        (output_dir / "trajectory.json").write_text(json.dumps({"episodes": [
            {"episode_id": i, "kind": item["kind"], "seed": item["seed"],
             "layout": _jsonable(item.get("layout")),
             "trace": _trace_metadata(item.get("trace", {})),
             "infos": _jsonable(item.get("infos", [])),
             "terminated": item.get("terminated", []).tolist(),
             "truncated": item.get("truncated", []).tolist()}
            for i, item in enumerate(accepted)]}, indent=2) + "\n")
        (output_dir / "collection_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
        (output_dir / "contract.json").write_text(json.dumps({
            "robot": "panda", "robot_init_qpos_noise": 0.02,
            "control_mode": "pd_ee_delta_pose", "obs_mode": "state",
            "state_semantics": "native ManiSkill state dictionary plus pickcube_obstacle and pickcube_goal; no StackCube indexes",
            "obstacle_observation": "world XY center and collision radius in meters (raw state; no normalization)",
            "action": {"dimension": 7, "position": "first 3 normalized by 0.1m",
                       "rotation": "XYZ Euler delta divided componentwise by controller.rot_lower, including its signed negative values",
                       "gripper": "index 6"},
            "horizons": {"obs": config.obs_horizon, "prediction": config.pred_horizon,
                         "execution": 8},
            "rollout_control_step_cap_per_complete_attempt": config.max_steps,
        }, indent=2) + "\n")
        report["artifact_sha256"] = {
            name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
            for name in ("trajectory.h5", "trajectory.json", "collection_config.json", "contract.json")
            if (output_dir / name).exists()
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (output_dir / "artifact_hashes.json").write_text(json.dumps({
            **report["artifact_sha256"],
            "report.json": hashlib.sha256((output_dir / "report.json").read_bytes()).hexdigest(),
        }, indent=2) + "\n")
        return report
    except BaseException:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


class PickCubePilotDataset(Dataset):
    """Task-specific native state-action windows; preserves all 7 action values."""

    def __init__(self, path, obs_horizon=2, pred_horizon=16):
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.observations, self.actions = [], []
        with h5py.File(path, "r") as handle:
            self.keys = sorted(handle.keys(), key=lambda key: int(key.rsplit("_", 1)[1]))
            self.slices = []
            for index, key in enumerate(self.keys):
                group = handle[key]
                obs, actions = group["obs"][:], group["actions"][:]
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"{key}: expected native seven-dimensional actions")
                if obs.shape[0] != actions.shape[0] + 1:
                    raise ValueError(f"{key}: expected T+1 observations")
                if "success" not in group or not np.all(group["success"][:]):
                    if "success" not in group or not bool(group["success"][-1]):
                        raise ValueError(f"{key}: missing successful terminal evidence")
                for flag in ("terminated", "truncated"):
                    if flag not in group or len(group[flag]) != len(actions) + 1:
                        raise ValueError(f"{key}: missing T+1 {flag} evidence")
                if bool(group["truncated"][:].any()) or not bool(group["terminated"][-1]):
                    raise ValueError(f"{key}: successful terminal cannot be a timeout")
                def check_state_lengths(node):
                    if isinstance(node, h5py.Dataset):
                        if len(node) != len(actions) + 1:
                            raise ValueError(f"{key}: full state must have T+1 rows")
                    else:
                        for child in node.values():
                            check_state_lengths(child)
                if "env_states" not in group:
                    raise ValueError(f"{key}: missing full states")
                check_state_lengths(group["env_states"])
                self.observations.append(torch.as_tensor(obs, dtype=torch.float32))
                self.actions.append(torch.as_tensor(actions, dtype=torch.float32))
                L = len(actions)
                self.slices.extend((index, start) for start in range(-(obs_horizon-1), L))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start = self.slices[index]
        obs, act = self.observations[trajectory], self.actions[trajectory]
        obs_indices = torch.arange(start, start + self.obs_horizon).clamp(0, len(obs) - 1)
        act_indices = torch.arange(start, start + self.pred_horizon).clamp(0, len(act) - 1)
        obs_seq = obs[obs_indices]
        act_seq = act[act_indices]
        return {"observations": obs_seq, "actions": act_seq}


def validate_minibatch(dataset_path, validation_dir, *, expected_obs_dim, device="cpu"):
    """Run one saved batch through a dimension-matched official epsilon model."""
    from torch.utils.data import DataLoader
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from examples.baselines.constrain.constrain_denoiser import _official_conditional_unet1d

    validation_dir = Path(validation_dir)
    validation_dir.mkdir(parents=True, exist_ok=False)
    dataset = PickCubePilotDataset(dataset_path)
    batch = next(iter(DataLoader(dataset, batch_size=min(2, len(dataset)))))
    obs, actions = batch["observations"], batch["actions"]
    if obs.shape[-1] != expected_obs_dim or actions.shape[-1] != 7:
        raise ValueError("saved minibatch does not match initialized PickCube env/action dimensions")
    network = _official_conditional_unet1d()(
        input_dim=7, global_cond_dim=obs.shape[1] * obs.shape[2],
        diffusion_step_embed_dim=64, down_dims=[64, 128, 256], n_groups=8,
    ).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        clip_sample=True, prediction_type="epsilon",
    )
    obs, actions = obs.to(device), actions.to(device)
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, 100, (len(actions),), device=device).long()
    noisy = scheduler.add_noise(actions, noise, timesteps)
    prediction = network(noisy, timesteps, global_cond=obs.flatten(start_dim=1))
    loss = torch.nn.functional.mse_loss(prediction, noise)
    evidence = {"batch_size": len(actions), "obs_shape": list(obs.shape),
                "action_shape": list(actions.shape), "loss": float(loss.detach().cpu()),
                "loss_finite": bool(torch.isfinite(loss).item()),
                "model": "official ConditionalUnet1D, epsilon objective",
                "device": str(device)}
    if not evidence["loss_finite"]:
        raise FloatingPointError("saved minibatch epsilon loss is non-finite")
    (validation_dir / "minibatch_validation.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-per-class", type=int, default=20)
    parser.add_argument("--max-attempts-per-class", type=int, default=200)
    parser.add_argument("--video-dir", required=True)
    args = parser.parse_args(argv)
    report = collect_pilot(args.output_dir, PickCubePilotConfig(
        seed=args.seed, target_per_class=args.target_per_class,
        max_attempts_per_class=args.max_attempts_per_class))
    diagnostics = Path(args.diagnostics_dir)
    diagnostics.mkdir(parents=True, exist_ok=False)
    (diagnostics / "collection_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not report["complete"]:
        raise RuntimeError("pilot target not met; see report.json")
    env = make_pickcube_env()
    try:
        obs = env.reset(seed=args.seed)[0]
        expected_obs_dim = int(np.asarray(obs).reshape(-1).shape[0])
    finally:
        env.close()
    validate_minibatch(Path(args.output_dir) / "trajectory.h5", args.validation_dir,
                       expected_obs_dim=expected_obs_dim)
    render_representative_videos(Path(args.output_dir) / "trajectory.h5", args.video_dir)


def render_representative_videos(dataset_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    env = make_pickcube_env(render_mode="rgb_array")
    try:
        with h5py.File(dataset_path, "r") as handle:
            selected = {}
            for key in sorted(handle.keys(), key=lambda name: int(name.rsplit("_", 1)[1])):
                group = handle[key]
                kind = str(group.attrs["kind"])
                if kind not in selected:
                    selected[kind] = key
                if len(selected) == 2:
                    break
            if set(selected) != {"direct", "obstacle"}:
                raise ValueError("representative videos require accepted direct and obstacle trajectories")
            for kind, key in selected.items():
                destination = output_dir / f"{kind}.mp4"
                with imageio.get_writer(str(destination), fps=env.unwrapped.control_freq,
                                        codec="libx264", macro_block_size=1) as writer:
                    for state in range(len(handle[f"{key}/actions"]) + 1):
                        env.unwrapped.set_state_dict({
                            name: _index_state(value, state)
                            for name, value in handle[f"{key}/env_states"].items()
                        })
                        frame = np.asarray(env.render(), dtype=np.uint8)
                        if frame.ndim == 4:
                            frame = frame[0]
                        writer.append_data(frame)
    finally:
        env.close()


def _index_state(node, index):
    if isinstance(node, h5py.Group):
        return {key: _index_state(value, index) for key, value in node.items()}
    return node[index]


if __name__ == "__main__":
    main()
