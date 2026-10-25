"""Closed-loop diagnosis of the frozen 90k PickCube contact episodes."""

import argparse
import hashlib
import json
import signal
import time
from collections import Counter
from pathlib import Path

import numpy as np

EXPOSURE_EXP = "outputs/exp/20261025_065159_pickcube_exposure90000"
EXPOSURE_CKPT = "outputs/ckpt/20261025_065159_pickcube_exposure90000_constrain/final.pt"
EXPOSURE_CKPT_SHA = "8a5de3ac5f8c4caba5617c30605bc17201f6beb4e4ea139fe0b8cfa8bb6ed5db"
EXPOSURE_REGISTRATION = f"{EXPOSURE_EXP}/preregistration.json"
PUBLISHED_EVAL = f"{EXPOSURE_EXP}/paired/constrain_dp"
DEV_SCENES = "outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes"
BASE_CKPT = "outputs/ckpt/20261024_104348_pickcube_expansion_base/final.pt"
BASE_CKPT_SHA = "dd9457e380e6621b1092aee4cf8c894d6f71070bace0b547e2f8c1a4956ac33d"
OBSTACLE300_H5 = "demos/PickCube-transport-detour/20261024_213325_obstacle300/trajectory.h5"
OBSTACLE300_H5_SHA = "3c15f50993b5ab5967b1a40695e53288b0f9464f4b61504a93d9989499ded7e6"
CONTACT_INDICES = (3, 5, 6, 7, 8, 9, 10, 14, 16, 19, 21, 22, 23, 25, 26)
STALE_QUEUE_INDEX_MIN = 4
CONDITION_DRIFT_M = 0.05
CLEARANCE_DROP_M = 0.02
EXPERT_PATH_LEAVE_M = 0.03
FRESH_DRIFT_M = 0.01
ISOLATION_COUNT = 8
PREDECLARED_STRIDE = 2
COMPARATOR_STRIDE = 8
POS_SCALE_M = 0.1
EXECUTED_SLICE = (1, 9)
OBS_TCP_TOLERANCE_M = 1e-4
EPISODE_WALL_SECONDS = 300


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def diagnosis_source_identity(repo_root=None):
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    files = (
        "examples/baselines/constrain/pickcube_closed_loop_diagnosis.py",
        "examples/baselines/constrain/pickcube_learning_pilot.py",
        "examples/baselines/constrain/pickcube_constrain.py",
        "examples/baselines/constrain/pickcube_transport_detour.py",
        "examples/baselines/constrain/diagnose_pickcube_contact.py",
        "examples/baselines/constrain/evaluate_pickcube_paired.py",
        "examples/baselines/constrain/pickcube_expert.py",
    )
    hashed = {rel: hashlib.sha256((repo / rel).read_bytes()).hexdigest() for rel in files}
    return {
        "basis": "working-tree bytes frozen before the closed-loop diagnosis",
        "git_head_is_code_identity": False,
        "files": hashed,
        "files_sha256": hashlib.sha256(
            json.dumps(hashed, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _relative_xy(observation):
    obs = np.asarray(observation, dtype=np.float64).reshape(-1)
    if obs.shape != (45,):
        raise ValueError(f"expected a 45d observation, got {obs.shape}")
    return obs[42:44] - obs[19:21]


def condition_drift_m(replan_observation, current_observation):
    return float(np.linalg.norm(_relative_xy(current_observation) - _relative_xy(replan_observation)))


def kinematic_queue_endpoint(tcp_xy, actions, pos_scale_m=POS_SCALE_M):
    queued = np.asarray(actions, dtype=np.float64)
    if queued.ndim != 2 or queued.shape[1] < 2:
        raise ValueError("queued actions must be (horizon, >=2)")
    xy = np.asarray(tcp_xy, dtype=np.float64).reshape(2) + queued[:, :2].sum(axis=0) * float(pos_scale_m)
    return {
        "xy": xy.tolist(),
        "is_executed_pose": False,
        "is_surface_clearance": False,
        "pos_scale_m": float(pos_scale_m),
        "method": "sum of normalized XY actions times the 0.1 m controller scale",
    }


def queue_heads_toward_obstacle(replan_clearance_m, endpoint_clearance_m):
    return float(endpoint_clearance_m) < float(replan_clearance_m)


def classify_contact_mechanism(row):
    if row.get("is_surface_clearance") is not False:
        raise ValueError("vertex gap is not surface clearance")
    queue_index = int(row["queue_index"])
    drift = float(row["condition_drift_m"])
    drop = float(row["clearance_drop_m"])
    expert_distance = row["expert_tcp_distance_m"]
    expert_distance = None if expert_distance is None else float(expert_distance)
    stale = (
        queue_index >= STALE_QUEUE_INDEX_MIN
        and drift >= CONDITION_DRIFT_M
        and drop >= CLEARANCE_DROP_M
        and expert_distance is not None
        and expert_distance >= EXPERT_PATH_LEAVE_M
        and row.get("queue_heads_toward_obstacle") is True
        and row.get("expert_physx_contact") is False
    )
    if stale:
        mechanism = "queued_overshoot"
    elif expert_distance is not None and expert_distance < EXPERT_PATH_LEAVE_M and row.get("expert_physx_contact") is False:
        mechanism = "narrow_tracking"
    elif queue_index <= 1 and drift < FRESH_DRIFT_M:
        mechanism = "fresh_error"
    else:
        mechanism = "not_isolated"
    return {
        "mechanism": mechanism,
        "is_surface_clearance": False,
        "rule": {
            "stale_queue_index_min": STALE_QUEUE_INDEX_MIN,
            "condition_drift_m": CONDITION_DRIFT_M,
            "clearance_drop_m": CLEARANCE_DROP_M,
            "expert_path_leave_m": EXPERT_PATH_LEAVE_M,
        },
    }


def decide_execution_factor(episodes):
    counts = dict(Counter(
        classify_contact_mechanism(episode["contact"])["mechanism"] for episode in episodes
    ))
    if any(episode.get("evaluator_mismatch") for episode in episodes):
        return {
            "trial": False, "factor": None, "stride": None, "comparator_stride": COMPARATOR_STRIDE,
            "swept": False, "seed_rule_changes": True, "reason": "evaluator_mismatch", "counts": counts,
        }
    if counts.get("queued_overshoot", 0) >= ISOLATION_COUNT and len(episodes) == len(CONTACT_INDICES):
        return {
            "trial": True, "factor": "execution_stride", "stride": PREDECLARED_STRIDE,
            "comparator_stride": COMPARATOR_STRIDE, "swept": False, "seed_rule_changes": True,
            "reason": "queued action staleness isolated", "counts": counts,
        }
    return {
        "trial": False, "factor": None, "stride": None, "comparator_stride": COMPARATOR_STRIDE,
        "swept": False, "seed_rule_changes": True, "reason": "staleness not isolated", "counts": counts,
    }


def _published_identity(record):
    result = record["result"]
    contact = record.get("first_contact") or {}
    return {
        "scene_index": record["scene_index"],
        "scene_id": record["scene_id"],
        "seed": record["seed"],
        "first_contact": {
            "step": contact.get("step"),
            "phase": contact.get("phase"),
            "link": contact.get("link"),
        },
        "result": {
            "native_success": result["native_success"],
            "obstacle_contact": result["obstacle_contact"],
            "xy_forbidden_crossing": result["xy_forbidden_crossing"],
            "safe_success": result["safe_success"],
            "elapsed_steps": result["elapsed_steps"],
            "termination_reason": result["termination_reason"],
        },
    }


def assert_reproduction(traced, published):
    left = _published_identity(traced)
    right = _published_identity(published)
    if left != right:
        raise ValueError(f"reproduction mismatch: traced {left} published {right}")
    return True


def evaluator_contract(payload):
    failures = []
    if list(payload["action_shape"]) != [8, 7]:
        failures.append("action_width")
    if float(payload["action_abs_max"]) > 1.000001 or payload["action_clipped"] is not False:
        failures.append("action_clip")
    if payload["native_success"] and int(payload["success_step"]) != int(payload["elapsed_steps"]):
        failures.append("success_first_step")
    if payload["controller_matches"] is not True:
        failures.append("controller")
    if list(payload["executed_slice"]) != list(EXECUTED_SLICE):
        failures.append("executed_slice")
    if payload["normalization_matches"] is not True:
        failures.append("normalization")
    if float(payload["obs_tcp_error_m"]) > OBS_TCP_TOLERANCE_M:
        failures.append("observation_latency")
    report = dict(payload)
    report["passed"] = not failures
    report["failures"] = failures
    return report


def build_diagnosis_index(episodes, protocol, identity, contract):
    if len(episodes) != len(CONTACT_INDICES):
        raise ValueError("diagnosis requires the 15 contact episodes")
    if [episode["scene_index"] for episode in episodes] != list(CONTACT_INDICES):
        raise ValueError("contact index set changed")
    required = ("step", "queue_index", "entities", "tcp_xyz", "cube_xyz", "grasp_stage", "condition_drift_m")
    for episode in episodes:
        if episode.get("reproduction", {}).get("matched") is not True:
            raise ValueError("unpublished reproduction")
        if episode.get("is_surface_clearance") is not False:
            raise ValueError("vertex gap is not surface clearance")
        if not episode.get("steps"):
            raise ValueError("missing timesteps")
        contact = episode["contact"]
        missing = [key for key in required if key not in contact]
        if missing:
            raise ValueError(f"missing {missing}")
    return {
        "prose_only": False,
        "formal_performance_claim": False,
        "checkpoint_sha256": EXPOSURE_CKPT_SHA,
        "contact_count": len(CONTACT_INDICES),
        "contact_indices": list(CONTACT_INDICES),
        "vertex_gap_is_surface_clearance": False,
        "protocol": protocol,
        "source_identity": identity,
        "evaluator_contract": contract,
        "decision": decide_execution_factor(episodes),
        "episodes": [
            {
                "scene_index": episode["scene_index"],
                "scene_id": episode["scene_id"],
                "contact": episode["contact"],
                "mechanism": classify_contact_mechanism(episode["contact"])["mechanism"],
                "reproduction": episode["reproduction"],
            }
            for episode in episodes
        ],
    }


def diagnosis_protocol():
    return {
        "execution_stride": COMPARATOR_STRIDE,
        "stale_queue_index_min": STALE_QUEUE_INDEX_MIN,
        "condition_drift_m": CONDITION_DRIFT_M,
        "clearance_drop_m": CLEARANCE_DROP_M,
        "expert_path_leave_m": EXPERT_PATH_LEAVE_M,
        "fresh_drift_m": FRESH_DRIFT_M,
        "isolation_count": ISOLATION_COUNT,
        "predeclared_stride": PREDECLARED_STRIDE,
        "comparator_stride": COMPARATOR_STRIDE,
        "pos_scale_m": POS_SCALE_M,
        "executed_slice": list(EXECUTED_SLICE),
        "max_steps": 500,
        "episode_wall_seconds": EPISODE_WALL_SECONDS,
        "noise_seed_rule": "torch.manual_seed(scene seed) once before the episode; sequential randn at each replan",
        "vertex_gap_is_surface_clearance": False,
        "changed_factor": None,
    }


def build_diagnosis_preregistration(paths):
    return {
        "kind": "authorized_closed_loop_diagnosis",
        "round_index": "closed-loop-diagnosis",
        "no_collection": True,
        "no_training": True,
        "formal_performance_claim": False,
        "trial_authorized_by_this_file": False,
        "checkpoint": EXPOSURE_CKPT,
        "checkpoint_sha256": EXPOSURE_CKPT_SHA,
        "base_checkpoint": BASE_CKPT,
        "base_checkpoint_sha256": BASE_CKPT_SHA,
        "dataset_sha256": OBSTACLE300_H5_SHA,
        "dataset": OBSTACLE300_H5,
        "exposure_registration": EXPOSURE_REGISTRATION,
        "published_eval": PUBLISHED_EVAL,
        "dev_scenes": DEV_SCENES,
        "contact_indices": list(CONTACT_INDICES),
        "published_counts": {"task": 27, "safe": 15, "contact": 15, "xy": 3},
        "comparison_counts": {"task": 26, "safe": 13, "contact": 17, "xy": 3, "label": "expansion-100"},
        "protocol": diagnosis_protocol(),
        "paths": dict(paths),
        "gate": "diagnosis only; a stride trial requires a later preregistration",
    }


def _deadline(_signum, _frame):
    raise TimeoutError("episode wall exceeded")


def _controller_report(env, scene):
    from examples.baselines.constrain.pickcube_scenes import _tree_hash

    controller = env.unwrapped.agent.controller
    state = controller.get_state()
    use_target = {}
    children = getattr(controller, "controllers", {})
    for name, child in children.items():
        use_target[name] = bool(getattr(getattr(child, "config", None), "use_target", False))
    return {
        "matches": _tree_hash(state) == scene["metadata"]["controller_sha256"],
        "snapshot_empty": state == {},
        "use_target": use_target,
        "sha256": _tree_hash(state),
        "frozen_sha256": scene["metadata"]["controller_sha256"],
    }


def _stage(grasped, lifted, near_goal, native_success):
    from examples.baselines.constrain.diagnose_pickcube_contact import classify_contact_phase

    return classify_contact_phase(
        grasped=grasped, lifted=lifted, near_goal=near_goal, native_success=native_success,
    )


def _geometry(point_xy, center, radius):
    from examples.baselines.constrain.diagnose_pickcube_contact import geometry_xy_clearance

    clearance = geometry_xy_clearance(point_xy, center, radius)
    if clearance["is_surface_clearance"] is not False:
        raise ValueError("vertex gap is not surface clearance")
    return clearance


def instrument_episode(adapter, scene_set, index, max_steps=500):
    import torch
    from examples.baselines.constrain.diagnose_pickcube_contact import observe_obstacle_contacts
    from examples.baselines.constrain.evaluate_pickcube_paired import classify_termination
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    from examples.baselines.constrain.pickcube_learning_pilot import _bool_info
    from examples.baselines.constrain.pickcube_scenes import _tree_hash

    scene = scene_set.load(index)
    geometry = scene["metadata"]["cylinder"]
    env = make_pickcube_env(obstacle_radius=geometry["radius"], obstacle_height=geometry["height"])
    try:
        _, obs = scene_set.restore(index, env)
        obs = np.asarray(obs.detach().cpu() if torch.is_tensor(obs) else obs, dtype=np.float32).reshape(-1)
        if _tree_hash(env.unwrapped.get_state_dict()) != scene["metadata"]["state_sha256"]:
            raise ValueError("restored state mismatch")
        if not np.array_equal(obs, scene["observation"].reshape(-1)):
            raise ValueError("restored observation mismatch")
        controller = _controller_report(env, scene)
        if not controller["matches"]:
            raise ValueError("restored controller snapshot mismatch")
        history = np.repeat(obs[None, :], 2, axis=0)
        torch.manual_seed(int(scene["metadata"]["seed"]))
        elapsed = 0
        collision = crossing = grasped = lifted = transported = False
        initial_cube_z = float(env.unwrapped.cube.pose.p[0, 2].detach().cpu())
        goal_xy = env.unwrapped.goal_site.pose.p[0, :2].detach().cpu().numpy()
        center = np.asarray(geometry["center_xy"], dtype=np.float64)
        radius = float(geometry["radius"])
        first_contact = None
        native_success = terminated = truncated = False
        termination = None
        success_step = None
        steps = []
        replans = []
        action_abs_max = 0.0
        obs_tcp_error = 0.0
        while elapsed < max_steps:
            prediction = adapter.get_action(history)
            queued = prediction[0].detach().cpu().numpy()
            if queued.shape != (8, 7):
                raise ValueError(f"expected queued (8, 7) actions, got {queued.shape}")
            action_abs_max = max(action_abs_max, float(np.max(np.abs(queued))))
            replan_obs = history[-1].copy()
            replan_tcp = env.unwrapped.agent.tcp.pose.p[0].detach().cpu().numpy()
            replan_clearance = _geometry(replan_tcp[:2], center, radius)
            kinematic = kinematic_queue_endpoint(replan_tcp[:2], queued)
            kinematic_clearance = _geometry(kinematic["xy"], center, radius)
            replans.append({
                "step": elapsed,
                "queue": queued.tolist(),
                "condition_m": _relative_xy(replan_obs).tolist() + [float(replan_obs[44])],
                "tcp_xyz": replan_tcp.astype(float).tolist(),
                "tcp_geometry_clearance_m": replan_clearance["clearance_m"],
                "kinematic_endpoint": kinematic,
                "kinematic_geometry_clearance_m": kinematic_clearance["clearance_m"],
                "queue_heads_toward_obstacle": queue_heads_toward_obstacle(
                    replan_clearance["clearance_m"], kinematic_clearance["clearance_m"]),
                "is_surface_clearance": False,
            })
            for queue_index, action in enumerate(queued[:min(8, max_steps - elapsed)]):
                obs, _, step_terminated, step_truncated, info = env.step(action[None])
                elapsed += 1
                terminated = bool(terminated or step_terminated)
                truncated = bool(truncated or step_truncated)
                obs = np.asarray(obs.detach().cpu() if torch.is_tensor(obs) else obs, dtype=np.float32).reshape(-1)
                history = np.concatenate((history[1:], obs[None]), axis=0)
                step_success = _bool_info(info, "native_success")
                native_success |= step_success
                step_collision = _bool_info(info, "obstacle_collision")
                collision |= step_collision
                crossing |= _bool_info(info, "obstacle_crossing")
                grasped_now = _bool_info(info, "is_grasped")
                grasped |= grasped_now
                tcp = env.unwrapped.agent.tcp.pose.p[0].detach().cpu().numpy()
                cube = env.unwrapped.cube.pose.p[0].detach().cpu().numpy()
                obs_tcp_error = max(obs_tcp_error, float(np.linalg.norm(obs[19:22] - tcp)))
                lifted_now = bool(cube[2] > initial_cube_z + 0.04)
                near_goal = bool(np.linalg.norm(cube[:2] - goal_xy) < 0.10)
                lifted |= lifted_now
                transported |= near_goal
                entities = [hit["entity"] for hit in observe_obstacle_contacts(env)]
                if step_collision and first_contact is None:
                    stored = getattr(env.unwrapped, "_first_obstacle_contact", None) or {}
                    first_contact = {
                        "step": elapsed,
                        "link": stored.get("other"),
                        "phase": _stage(grasped_now, lifted_now, near_goal, step_success),
                    }
                if step_success and success_step is None:
                    success_step = elapsed
                clearance = _geometry(tcp[:2], center, radius)
                steps.append({
                    "step": elapsed,
                    "queue_index": queue_index,
                    "replan_step": replans[-1]["step"],
                    "action": action.astype(float).tolist(),
                    "tcp_xyz": tcp.astype(float).tolist(),
                    "cube_xyz": cube.astype(float).tolist(),
                    "grasp_stage": _stage(grasped_now, lifted_now, near_goal, step_success),
                    "grasped": grasped_now,
                    "lifted": lifted_now,
                    "near_goal": near_goal,
                    "native_success": step_success,
                    "entities": entities,
                    "obstacle_collision": step_collision,
                    "condition_drift_m": condition_drift_m(replan_obs, obs),
                    "tcp_geometry_clearance_m": clearance["clearance_m"],
                    "cube_geometry_clearance_m": _geometry(cube[:2], center, radius)["clearance_m"],
                    "is_surface_clearance": False,
                })
                termination = classify_termination(
                    step_success, step_terminated, step_truncated, elapsed, max_steps)
                if termination is not None:
                    break
            if termination is not None:
                break
        if termination is None:
            termination = classify_termination(native_success, terminated, truncated, elapsed, max_steps)
        return {
            "scene_index": index,
            "scene_id": scene["metadata"]["scene_id"],
            "seed": int(scene["metadata"]["seed"]),
            "status": "evaluated",
            "restored_state_sha256": scene["metadata"]["state_sha256"],
            "controller": controller,
            "first_contact": first_contact,
            "native_success_step": success_step,
            "action_abs_max": action_abs_max,
            "action_clipped": False,
            "obs_tcp_error_m": obs_tcp_error,
            "replans": replans,
            "steps": steps,
            "is_surface_clearance": False,
            "result": {
                "termination_reason": termination,
                "elapsed_steps": elapsed,
                "terminated": terminated,
                "truncated": truncated,
                "native_success": native_success,
                "safe_success": bool(native_success and not collision and not crossing),
                "obstacle_contact": collision,
                "xy_forbidden_crossing": crossing,
                "phase": ("task_success" if native_success else "no_grasp" if not grasped
                          else "grasp_no_lift" if not lifted else "lift_no_transport"
                          if not transported else "transport_or_place_failed"),
            },
        }
    finally:
        env.close()


def replay_expert(scene_set, index):
    import torch
    from examples.baselines.constrain.diagnose_pickcube_contact import observe_obstacle_contacts
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env

    scene = scene_set.load(index)
    geometry = scene["metadata"]["cylinder"]
    with __import__("h5py").File(scene_set.directory / "scenes.h5", "r") as handle:
        actions = np.asarray(handle[scene["metadata"]["scene_id"]]["expert_actions"])
    env = make_pickcube_env(obstacle_radius=geometry["radius"], obstacle_height=geometry["height"])
    try:
        scene_set.restore(index, env)
        center = np.asarray(geometry["center_xy"], dtype=np.float64)
        radius = float(geometry["radius"])
        initial_cube_z = float(env.unwrapped.cube.pose.p[0, 2].detach().cpu())
        goal_xy = env.unwrapped.goal_site.pose.p[0, :2].detach().cpu().numpy()
        steps = []
        physx = False
        for action in actions:
            _, _, _, _, info = env.step(action[None])
            tcp = env.unwrapped.agent.tcp.pose.p[0].detach().cpu().numpy()
            cube = env.unwrapped.cube.pose.p[0].detach().cpu().numpy()
            grasped = bool(np.asarray(info["is_grasped"]).reshape(-1)[0])
            lifted = bool(cube[2] > initial_cube_z + 0.04)
            near_goal = bool(np.linalg.norm(cube[:2] - goal_xy) < 0.10)
            success = bool(np.asarray(info["native_success"]).reshape(-1)[0])
            entities = [hit["entity"] for hit in observe_obstacle_contacts(env)]
            collision = bool(np.asarray(info["obstacle_collision"]).reshape(-1)[0])
            physx = physx or collision or bool(entities)
            steps.append({
                "step": len(steps) + 1,
                "tcp_xyz": tcp.astype(float).tolist(),
                "tcp_xy": tcp[:2].astype(float).tolist(),
                "cube_xyz": cube.astype(float).tolist(),
                "grasp_stage": _stage(grasped, lifted, near_goal, success),
                "entities": entities,
                "tcp_geometry_clearance_m": _geometry(tcp[:2], center, radius)["clearance_m"],
                "cube_geometry_clearance_m": _geometry(cube[:2], center, radius)["clearance_m"],
                "is_surface_clearance": False,
            })
            if success:
                break
        retained = (scene["metadata"].get("expert") or {}).get("clearance") or {}
        return {
            "steps": steps,
            "physx_contact": physx,
            "min_tcp_geometry_clearance_m": min(step["tcp_geometry_clearance_m"] for step in steps),
            "min_cube_geometry_clearance_m": min(step["cube_geometry_clearance_m"] for step in steps),
            "retained_vertex_gap_m": retained.get("geometric_vertex_clearance_m"),
            "retained_vertex_gap_is_surface_clearance": False,
            "is_surface_clearance": False,
        }
    finally:
        env.close()


def _nearest_expert(contact_step, expert):
    pool = [step for step in expert["steps"] if step["grasp_stage"] == contact_step["grasp_stage"]]
    if not pool:
        return None, False
    point = np.asarray(contact_step["tcp_xyz"][:2], dtype=np.float64)
    distances = [float(np.linalg.norm(np.asarray(step["tcp_xy"]) - point)) for step in pool]
    chosen = pool[int(np.argmin(distances))]
    return {
        "distance_m": min(distances),
        "expert_step": chosen["step"],
        "expert_tcp_xyz": chosen["tcp_xyz"],
        "expert_geometry_clearance_m": chosen["tcp_geometry_clearance_m"],
        "same_phase": True,
        "is_surface_clearance": False,
    }, True


def assemble_contact(traced, expert):
    if not traced.get("first_contact"):
        raise ValueError("contact episode has no first contact")
    contact_step = traced["steps"][traced["first_contact"]["step"] - 1]
    replan = next(item for item in traced["replans"] if item["step"] == contact_step["replan_step"])
    nearest, found = _nearest_expert(contact_step, expert)
    drop = float(replan["tcp_geometry_clearance_m"]) - float(contact_step["tcp_geometry_clearance_m"])
    return {
        "step": contact_step["step"],
        "queue_index": contact_step["queue_index"],
        "entities": contact_step["entities"] or [traced["first_contact"].get("link")],
        "tcp_xyz": contact_step["tcp_xyz"],
        "cube_xyz": contact_step["cube_xyz"],
        "grasp_stage": contact_step["grasp_stage"],
        "condition_drift_m": contact_step["condition_drift_m"],
        "clearance_drop_m": drop,
        "expert_tcp_distance_m": None if nearest is None else nearest["distance_m"],
        "expert_physx_contact": bool(expert["physx_contact"]),
        "queue_heads_toward_obstacle": bool(replan["queue_heads_toward_obstacle"]),
        "is_surface_clearance": False,
        "replan_step": replan["step"],
        "kinematic_endpoint": replan["kinematic_endpoint"],
        "executed_tcp_xyz": contact_step["tcp_xyz"],
        "nearest_expert": nearest,
        "same_phase_expert": found,
        "link": traced["first_contact"].get("link"),
    }


def verify_executed_slice(adapter, queued):
    import torch
    from examples.baselines.constrain.pickcube_constrain import pickcube_relative_condition

    torch.manual_seed(12345)
    try:
        got = adapter.get_action(queued).detach().cpu()
    except ValueError as exc:
        return {
            "matches_slice_1_9": False,
            "differs_from_slice_0_8": False,
            "executed_slice": list(EXECUTED_SLICE),
            "error": str(exc),
        }
    torch.manual_seed(12345)
    obs = torch.as_tensor(queued, dtype=torch.float32, device=adapter.device)
    if obs.ndim == 2:
        obs = obs[None]
    hint = pickcube_relative_condition(obs, adapter.norm)[:, None, :].expand(-1, 16, -1)
    sample = torch.randn(obs.shape[0], 16, 7, device=adapter.device)
    cond = obs[..., :42].flatten(start_dim=obs.ndim - 2)
    with torch.no_grad():
        for timestep in adapter.scheduler.timesteps:
            prediction = adapter.denoiser(sample, timestep, global_cond=cond, hint=hint, enabled=True)
            sample = adapter.scheduler.step(prediction, timestep, sample).prev_sample
    sliced = sample[:, 1:9].detach().cpu()
    other = sample[:, 0:8].detach().cpu()
    return {
        "matches_slice_1_9": bool(torch.equal(got, sliced)),
        "differs_from_slice_0_8": bool(not torch.equal(sliced, other)),
        "executed_slice": list(EXECUTED_SLICE),
    }


def _normalization_matches(adapter, registration):
    expected = adapter.norm.to_dict()
    return expected["relative_xy_scale_m"] == 0.5 and expected["radius_scale_m"] == 0.08 and expected["clip"] is True


def run_diagnosis(registration, device):
    import torch
    from examples.baselines.constrain.pickcube_learning_pilot import PilotBaseAdapter, PilotConstrainAdapter
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet

    if _sha_file(registration["checkpoint"]) != EXPOSURE_CKPT_SHA:
        raise RuntimeError("checkpoint hash does not match the frozen 90k EMA")
    if _sha_file(registration["base_checkpoint"]) != BASE_CKPT_SHA:
        raise RuntimeError("base checkpoint hash changed")
    if _sha_file(registration["dataset"]) != OBSTACLE300_H5_SHA:
        raise RuntimeError("frozen 300 dataset hash changed")
    exposure = json.loads(Path(registration["exposure_registration"]).read_text())
    published_dir = Path(registration["published_eval"])
    output = Path(registration["paths"]["diagnosis"])
    output.mkdir(parents=True, exist_ok=False)
    log_path = Path(registration["paths"]["exp"]) / "logs" / "diagnosis.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = TransportDetourSceneSet(registration["dev_scenes"])
    adapter = PilotConstrainAdapter(registration["checkpoint"], exposure, device)
    if adapter.metadata["checkpoint_sha256"] != EXPOSURE_CKPT_SHA:
        raise RuntimeError("loaded checkpoint is not the frozen 90k EMA")
    base = PilotBaseAdapter(registration["base_checkpoint"], exposure, device)
    scene = catalog.load(CONTACT_INDICES[0])
    queued = np.repeat(np.asarray(scene["observation"], dtype=np.float32)[None, :], 2, axis=0)
    slice_report = verify_executed_slice(adapter, queued)
    torch.manual_seed(12345)
    base_action = base.get_action(queued)
    base_shape = list(base_action.shape[-2:])
    contract_payload = {
        "action_shape": [8, 7] if slice_report["matches_slice_1_9"] else [0, 0],
        "action_abs_max": 0.0,
        "action_clipped": False,
        "success_step": 1,
        "elapsed_steps": 1,
        "native_success": True,
        "success_first_step_ok": True,
        "controller_matches": True,
        "obs_tcp_error_m": 0.0,
        "normalization_matches": _normalization_matches(adapter, exposure) and slice_report["matches_slice_1_9"],
        "executed_slice": slice_report["executed_slice"] if slice_report["matches_slice_1_9"] else [0, 8],
        "base_action_shape": base_shape,
        "base_action_width": int(base_action.shape[-1]),
        "slice_differs_from_0_8": slice_report["differs_from_slice_0_8"],
    }
    if base_shape != [8, 7]:
        contract_payload["action_shape"] = base_shape
    episodes = []
    started = time.monotonic()
    for index in CONTACT_INDICES:
        published = json.loads((published_dir / f"episode_{index:03d}.json").read_text())
        previous = signal.signal(signal.SIGALRM, _deadline)
        signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_SECONDS)
        episode_started = time.monotonic()
        try:
            traced = instrument_episode(adapter, catalog, index)
            expert = replay_expert(catalog, index)
            assert_reproduction(traced, published)
            contact = assemble_contact(traced, expert)
            contract_payload["action_abs_max"] = max(contract_payload["action_abs_max"], traced["action_abs_max"])
            contract_payload["obs_tcp_error_m"] = max(contract_payload["obs_tcp_error_m"], traced["obs_tcp_error_m"])
            contract_payload["controller_matches"] = contract_payload["controller_matches"] and traced["controller"]["matches"]
            if traced["result"]["native_success"] and traced["native_success_step"] != traced["result"]["elapsed_steps"]:
                contract_payload["success_first_step_ok"] = False
                contract_payload["success_step"] = traced["native_success_step"]
                contract_payload["elapsed_steps"] = traced["result"]["elapsed_steps"]
            episode = {
                "scene_index": index,
                "scene_id": traced["scene_id"],
                "seed": traced["seed"],
                "checkpoint_sha256": EXPOSURE_CKPT_SHA,
                "reproduction": {"matched": True},
                "is_surface_clearance": False,
                "evaluator_mismatch": False,
                "controller": traced["controller"],
                "first_contact": traced["first_contact"],
                "result": traced["result"],
                "contact": contact,
                "replans": traced["replans"],
                "steps": traced["steps"],
                "expert_summary": {
                    "physx_contact": expert["physx_contact"],
                    "min_tcp_geometry_clearance_m": expert["min_tcp_geometry_clearance_m"],
                    "min_cube_geometry_clearance_m": expert["min_cube_geometry_clearance_m"],
                    "retained_vertex_gap_m": expert["retained_vertex_gap_m"],
                    "retained_vertex_gap_is_surface_clearance": False,
                    "step_count": len(expert["steps"]),
                },
                "expert_steps": expert["steps"],
                "wall_seconds": round(time.monotonic() - episode_started, 3),
            }
            episode["contact"]["mechanism"] = classify_contact_mechanism(contact)["mechanism"]
            _write_json(output / f"episode_{index:03d}.json", episode)
            episodes.append(episode)
            line = {
                "scene_index": index, "status": "reproduced",
                "mechanism": episode["contact"]["mechanism"],
                "queue_index": contact["queue_index"],
                "drift_m": contact["condition_drift_m"],
                "wall_seconds": episode["wall_seconds"],
            }
            with log_path.open("a") as stream:
                stream.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
        except Exception as exc:
            failure = {"scene_index": index, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            _write_json(output / f"episode_{index:03d}.failure.json", failure)
            with log_path.open("a") as stream:
                stream.write(json.dumps(failure) + "\n")
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    contract = evaluator_contract(contract_payload)
    if not contract["passed"] or contract_payload["base_action_width"] != 7:
        for episode in episodes:
            episode["evaluator_mismatch"] = True
    index = build_diagnosis_index(episodes, registration["protocol"], registration["source_identity"], contract)
    index["elapsed_seconds"] = round(time.monotonic() - started, 3)
    index["base_action_shape"] = base_shape
    index["slice_report"] = slice_report
    _write_json(output / "index.json", index)
    print(json.dumps({"decision": index["decision"], "contract": contract["passed"]}), flush=True)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preregister", "diagnose"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "preregister":
        from examples.baselines.constrain.pickcube_learning_pilot import write_preregistration

        exp = args.registration.parent
        payload = build_diagnosis_preregistration({
            "exp": str(exp),
            "diagnosis": str(exp / "diagnosis"),
        })
        payload["source_identity"] = diagnosis_source_identity()
        write_preregistration(args.registration, payload)
        print(args.registration, flush=True)
        return payload
    registration = json.loads(args.registration.read_text())
    current = diagnosis_source_identity()
    frozen = (registration.get("source_identity") or {}).get("files_sha256")
    if current["files_sha256"] != frozen:
        raise RuntimeError("source identity does not match preregistration")
    return run_diagnosis(registration, args.device)


if __name__ == "__main__":
    main()
