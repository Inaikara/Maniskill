"""Replay a frozen existing PickCube catalog pool and append native episodes.

This tool never samples layouts. Registration freezes source identities and the
deterministic catalog order before ``run`` may construct a PhysX environment.
"""

import argparse
import hashlib
import importlib.util
import json
import importlib.metadata
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.constrain.audit_pickcube_existing_pool import (
    RETAINED_CATALOGS,
    SOURCES,
    full_scene_duplicate,
    task_duplicate,
)


OLD_NAME = "20261024_213325_obstacle300"
DIRECT_SOURCE = "demos/PickCube-transport-detour/20261024_104348_expansion"
DEV_MANIFEST = "outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes/manifest.json"
TICKET03_MANIFEST = "outputs/exp/20261023_165956_pickcube_transport_detour/manifest.json"
OLD_GEOMETRY_SOURCES = (
    Path("outputs/exp/20261024_104348_pickcube_data_expansion/retained_scenes/manifest.json"),
    Path("outputs/exp/20261024_104348_pickcube_data_expansion/pair_scenes/merged/manifest.json"),
    Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/train_scenes/manifest.json"),
)
TARGET = 200
PER_CANDIDATE_WALL_S = 160
OUTER_WALL_S = 20000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_tree_hash(directory):
    rows = []
    for path in sorted(Path(directory).rglob("*")):
        if path.is_file():
            rows.append([path.relative_to(directory).as_posix(), path.stat().st_size, sha(path)])
    return canonical_hash(rows)


def read_tree(node):
    return ({key: read_tree(value) for key, value in node.items()}
            if isinstance(node, h5py.Group) else np.asarray(node[()]))


def restore_like(value, template):
    if isinstance(template, dict):
        if value.keys() != template.keys():
            raise ValueError("catalog native state key set differs from runtime")
        return {key: restore_like(value[key], template[key]) for key in template}
    if torch.is_tensor(template):
        return torch.as_tensor(value, dtype=template.dtype, device=template.device).reshape(template.shape)
    return np.asarray(value, dtype=np.asarray(template).dtype).reshape(np.asarray(template).shape)


def state_equal(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    return np.array_equal(left, right)


def clone_tree(value):
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def controller_contract(controller):
    targets = {}
    for name, subcontroller in getattr(controller, "controllers", {}).items():
        config = getattr(subcontroller, "config", None)
        value = getattr(config, "use_target", False)
        if torch.is_tensor(value):
            value = bool(value.reshape(-1)[0].item())
        targets[name] = {"attribute_present": hasattr(config, "use_target"),
                         "use_target": bool(value)}
    return {"state": jsonable(controller.get_state()), "controllers": targets}


def state_hash(value):
    digest = hashlib.sha256()
    def visit(node, name=""):
        if isinstance(node, dict):
            for key in sorted(node):
                visit(node[key], f"{name}/{key}")
        else:
            a = np.ascontiguousarray(node.detach().cpu().numpy() if torch.is_tensor(node) else node)
            digest.update(name.encode()); digest.update(a.dtype.str.encode())
            digest.update(np.asarray(a.shape, dtype=np.int64).tobytes()); digest.update(a.tobytes())
    visit(value)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def source_identity(root, old_root):
    paths = [old_root / "trajectory.h5", old_root / "trajectory.json",
             root / "trajectory.h5", root / "trajectory.json",
             Path(DEV_MANIFEST), Path(TICKET03_MANIFEST)]
    identities = {str(path): sha(path) for path in paths}
    for catalog in RETAINED_CATALOGS:
        identities[str(catalog / "manifest.json")] = sha(catalog / "manifest.json")
        identities[str(catalog / "scenes.h5")] = sha(catalog / "scenes.h5")
    for source in OLD_GEOMETRY_SOURCES:
        identities[str(source)] = sha(source)
    for module_name in (
        "examples.baselines.constrain.pickcube_expert",
        "examples.baselines.constrain.pickcube_obstacle",
        "mani_skill.envs.tasks.tabletop.pick_cube",
    ):
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot_resolve_runtime_source:{module_name}")
        identities[spec.origin] = sha(spec.origin)
    return identities


def external_task_identities():
    external = []
    for filename in (DEV_MANIFEST, TICKET03_MANIFEST):
        payload = json.loads(Path(filename).read_text())
        rows = payload.get("scenes", payload.get("episodes", []))
        for i, row in enumerate(rows):
            layout = row.get("layout", row)
            external.append({"episode_id": row.get("episode_id", row.get("scene_id", i)),
                             "kind": "obstacle", "layout": {
                                 "seed": layout.get("seed"), "start_xy": layout.get("start_xy"),
                                 "goal_xy": layout.get("goal_xy")}})
    return external


def select_pool(old_manifest, external):
    catalog_rows = []
    for path in RETAINED_CATALOGS:
        catalog_rows.extend(json.loads((path / "manifest.json").read_text())["scenes"])
    for path in OLD_GEOMETRY_SOURCES:
        catalog_rows.extend(json.loads(path.read_text())["scenes"])
    by_seed = {int(row["seed"]): row for row in catalog_rows}
    old_obs = []
    for source in old_manifest["episodes"]:
        if source["kind"] != "obstacle":
            continue
        old = json.loads(json.dumps(source))
        geometry = by_seed.get(int(source["layout"]["seed"]))
        if geometry is None or any(
            abs(float(a) - float(b)) > 1e-7
            for key in ("start_xy", "goal_xy")
            for a, b in zip(source["layout"][key], geometry[key])
        ):
            raise ValueError(f"old_obstacle_geometry_unresolved:{source['episode_id']}")
        old["cylinder"] = geometry["cylinder"]
        old_obs.append(old)
    accepted = []
    candidates = []
    for directory in RETAINED_CATALOGS:
        rows = json.loads((directory / "manifest.json").read_text())["scenes"]
        for row in rows:
            candidate = {"episode_id": row["scene_id"], "kind": "obstacle", "layout": {
                "seed": row["seed"], "start_xy": row["start_xy"], "goal_xy": row["goal_xy"]},
                "cylinder": row["cylinder"]}
            expert = row.get("expert", {})
            if not (expert.get("native_success") is True and expert.get("held_at_success") is True
                    and expert.get("safe_success") is True and expert.get("obstacle_collision") is False
                    and expert.get("obstacle_crossing") is False
                    and expert.get("direct_transport_blocked") is True
                    and row["cylinder"].get("radius") == 0.03):
                continue
            if any(task_duplicate(candidate, old) for old in external):
                continue
            if any(full_scene_duplicate(candidate, old) for old in old_obs):
                continue
            if any(full_scene_duplicate(candidate, previous["identity"]) for previous in accepted):
                continue
            accepted.append({"identity": candidate, "row": row,
                             "catalog": directory.as_posix()})
    # Pin underlying action/initial-state/observation bytes into the ordered pool.
    for candidate in accepted:
        with h5py.File(Path(candidate["catalog"]) / "scenes.h5", "r") as handle:
            group = handle[candidate["row"]["scene_id"]]
            candidate["expert_actions_sha256"] = hashlib.sha256(
                np.ascontiguousarray(group["expert_actions"][:]).tobytes()).hexdigest()
            candidate["catalog_observations_sha256"] = hashlib.sha256(
                np.ascontiguousarray(group["observations"][:]).tobytes()).hexdigest()
            candidate["catalog_state_sha256"] = candidate["row"]["state_sha256"]
    return accepted


def preregister(registration_path, old_root, direct_root, exp_dir):
    old_manifest = json.loads((old_root / "trajectory.json").read_text())
    pool = select_pool(old_manifest, external_task_identities())
    if len(pool) < TARGET:
        raise RuntimeError(f"frozen full-scene pool has only {len(pool)} eligible candidates")
    registration = {
        "protocol": "pickcube-fullscene500-native-replay-v1",
        "target": {"direct": 1000, "obstacle": 500, "new_native_obstacle": TARGET},
        "identity": {
            "task_conflict": "same seed OR (start_xy and goal_xy componentwise within 0.02 m)",
            "full_scene_conflict_train": "task conflict AND cylinder center XY componentwise within 0.005 m AND height <=0.005 m AND radius <=0.001 m",
            "external_dev_ticket03": "strict task conflict regardless of cylinder geometry",
            "selection": "catalog path order and manifest order; first full-scene-unique candidate retained; no outcomes",
        },
        "execution": {"backend": "PhysX CPU", "max_steps": 500,
                      "per_candidate_wall_seconds": PER_CANDIDATE_WALL_S,
                      "outer_wall_seconds": OUTER_WALL_S,
                      "max_candidates": len(pool), "replace_only_from_frozen_pool": True,
                      "stop_at_first_native_held_success": True,
                      "admit_only_native_success_held_safe_contact_free": True,
                      "require_initial_full_state_controller_and_45d_observation_identity": True,
                      "require_exact_first_transition_restore": True},
        "runtime_versions": {name: importlib.metadata.version(name) for name in
                              ("gymnasium", "mani-skill", "torch", "sapien")},
        "inputs": {"old_obstacle300": str(old_root), "direct_source": str(direct_root),
                   "catalogs": [p.as_posix() for p in RETAINED_CATALOGS],
                   "sha256": source_identity(direct_root, old_root)},
        "replay_implementation": {"path": str(Path(__file__)), "sha256": sha(Path(__file__)),
                                  "identity_helper": "examples/baselines/constrain/audit_pickcube_existing_pool.py",
                                  "identity_helper_sha256": sha("examples/baselines/constrain/audit_pickcube_existing_pool.py")},
        "pool_count": len(pool),
        "pool": [{"catalog": c["catalog"], "scene_id": c["row"]["scene_id"], "row": c["row"],
                  "seed": c["row"]["seed"], "start_xy": c["row"]["start_xy"],
                  "goal_xy": c["row"]["goal_xy"], "cylinder": c["row"]["cylinder"],
                  "waypoint_xy": c["row"].get("expert", {}).get("waypoint_xy"),
                  "state_sha256": c["catalog_state_sha256"],
                  "expert_actions_sha256": c["expert_actions_sha256"],
                  "catalog_observations_sha256": c["catalog_observations_sha256"]}
                 for c in pool],
    }
    registration["registration_sha256"] = canonical_hash(registration)
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"registration": str(registration_path), "pool_count": len(pool),
                      "registration_sha256": registration["registration_sha256"]}, indent=2))


def obs_vector(base, info):
    from mani_skill.utils.common import flatten_state_dict
    named = base._get_obs_state_dict(info)
    return flatten_state_dict(named, use_torch=True)[0].detach().cpu().numpy().astype(np.float32).copy()


def replay_candidate(candidate, env):
    from examples.baselines.constrain.pickcube_expert import _set_state_dict_exact
    from examples.baselines.constrain.pickcube_expert import _reset_elapsed_steps

    catalog = Path(candidate["catalog"])
    row = candidate["row"]
    geometry = row["cylinder"]
    with h5py.File(catalog / "scenes.h5", "r") as handle:
        group = handle[row["scene_id"]]
        actions = np.asarray(group["expert_actions"], dtype=np.float32)
        catalog_obs = np.asarray(group["observations"], dtype=np.float32)
        catalog_state = read_tree(group["state"])
        controller_json = json.loads(group.attrs["controller_json"])
    if (hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
            != candidate["expert_actions_sha256"]):
        raise ValueError("frozen_expert_actions_hash_mismatch")
    if hashlib.sha256(np.ascontiguousarray(catalog_obs).tobytes()).hexdigest() != candidate["catalog_observations_sha256"]:
        raise ValueError("frozen_catalog_observation_hash_mismatch")
    if state_hash(catalog_state) != row["state_sha256"]:
        raise ValueError("catalog_full_initial_state_fingerprint_mismatch")
    if state_hash(controller_json) != row["controller_sha256"]:
        raise ValueError("catalog_controller_fingerprint_mismatch")
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) > 500 or not np.isfinite(actions).all():
        raise ValueError("invalid_native_expert_action_array")
    options = {"obstacle_enabled": True, "obstacle_center": geometry["center_xy"],
               "cube_xy": row["start_xy"], "goal_pos": [*row["goal_xy"], 0.12]}
    obs, info = env.reset(seed=int(row["seed"]), options=options)
    base = env.unwrapped
    expected = restore_like(catalog_state, base.get_state_dict())
    _set_state_dict_exact(base, expected)
    controller = base.agent.controller
    template_controller = controller.get_state()
    if controller_json != jsonable(template_controller):
        raise ValueError("catalog_controller_snapshot_does_not_match_runtime")
    controller.set_state(controller_json)
    _reset_elapsed_steps(env, base)
    initial_state = clone_tree(base.get_state_dict())
    initial_controller = clone_tree(controller.get_state())
    initial_controller_contract = controller_contract(controller)
    if initial_controller != {} or any(
        item["attribute_present"] and item["use_target"]
        for item in initial_controller_contract["controllers"].values()
    ):
        raise ValueError("native_controller_reset_contract_mismatch")
    initial_raw = np.asarray(base.get_obs(), dtype=np.float32).reshape(-1).copy()
    initial_info = jsonable(base.get_info())
    initial_policy = obs_vector(base, base.get_info())
    if not state_equal(expected, initial_state):
        raise ValueError("initial_full_state_restore_mismatch")
    if not np.array_equal(initial_raw, catalog_obs[0]):
        raise ValueError("catalog_initial_45d_observation_mismatch")
    if not np.array_equal(initial_raw[:42], initial_policy[:42]):
        raise ValueError("policy_observation_prefix_mismatch")
    states = [clone_tree(initial_state)]
    raw_obs = [initial_raw.copy()]
    policy_obs = [initial_policy[:42].copy()]
    controllers = [clone_tree(initial_controller)]
    controller_contracts = [initial_controller_contract]
    infos = [initial_info]
    terminated, truncated = [False], [False]
    executed = []
    safety_trace = []
    first_transition = None
    success_at = None
    for index, action in enumerate(actions):
        obs, _, term, trunc, info = env.step(torch.as_tensor(action, dtype=torch.float32, device=base.device))
        native_info = base.get_info()
        current_info = jsonable(native_info)
        current_state = clone_tree(base.get_state_dict())
        current_raw = np.asarray(base.get_obs(), dtype=np.float32).reshape(-1).copy()
        current_policy = obs_vector(base, native_info)[:42]
        current_controller = clone_tree(controller.get_state())
        current_controller_contract = controller_contract(controller)
        executed.append(action.copy())
        states.append(current_state); raw_obs.append(current_raw); policy_obs.append(current_policy)
        controllers.append(current_controller); controller_contracts.append(current_controller_contract)
        infos.append(current_info)
        terminated.append(bool(np.asarray(term).reshape(-1)[0]))
        truncated.append(bool(np.asarray(trunc).reshape(-1)[0]))
        collision = bool(np.asarray(current_info["obstacle_collision"]).reshape(-1)[0])
        crossing = bool(np.asarray(current_info["obstacle_crossing"]).reshape(-1)[0])
        native = bool(np.asarray(current_info["native_success"]).reshape(-1)[0])
        held = bool(np.asarray(current_info["is_grasped"]).reshape(-1)[0])
        safety_trace.append({"native_success": native, "held": held,
                             "obstacle_collision": collision, "obstacle_crossing": crossing})
        if index == 0:
            first_transition = {"state": clone_tree(current_state), "raw": current_raw.copy(),
                                "policy": current_policy.copy(), "controller": clone_tree(current_controller),
                                "controller_contract": current_controller_contract,
                                "info": current_info, "terminated": terminated[-1], "truncated": truncated[-1]}
        if collision or crossing:
            raise ValueError(f"unsafe_prefix_at_action_{index + 1}")
        if native and held:
            success_at = index + 1
            break
        if terminated[-1] or truncated[-1]:
            break
    if success_at is None:
        raise ValueError("stored_expert_did_not_reach_native_held_success_within_500")
    if not first_transition:
        raise ValueError("empty_expert_actions")

    # Independent fresh-environment exact initial + first-action restore check.
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    audit_env = make_pickcube_env(obstacle_radius=float(geometry["radius"]),
                                  obstacle_height=float(geometry["height"]))
    try:
        audit_base = audit_env.unwrapped
        audit_env.reset(seed=int(row["seed"]), options=options)
        audit_expected = restore_like(catalog_state, audit_base.get_state_dict())
        _set_state_dict_exact(audit_base, audit_expected)
        audit_controller_obj = audit_base.agent.controller
        audit_controller_obj.set_state(controller_json)
        _reset_elapsed_steps(audit_env, audit_base)
        if not state_equal(audit_expected, audit_base.get_state_dict()):
            raise ValueError("restore_audit_initial_state_mismatch")
        audit_obs = np.asarray(audit_base.get_obs(), dtype=np.float32).reshape(-1).copy()
        if not np.array_equal(audit_obs, initial_raw):
            mismatch = np.flatnonzero(audit_obs != initial_raw)
            raise ValueError(f"restore_audit_initial_observation_mismatch:max_abs={float(np.max(np.abs(audit_obs-initial_raw)))}:indices={mismatch[:12].tolist()}")
        _, _, audit_term, audit_trunc, _ = audit_env.step(
            torch.as_tensor(executed[0], dtype=torch.float32, device=audit_base.device))
        audit_state = clone_tree(audit_base.get_state_dict())
        audit_raw = np.asarray(audit_base.get_obs(), dtype=np.float32).reshape(-1).copy()
        audit_controller = clone_tree(audit_controller_obj.get_state())
        audit_native_info = jsonable(audit_base.get_info())
        if not state_equal(audit_state, first_transition["state"]):
            raise ValueError("restore_audit_first_transition_state_mismatch")
        if not np.array_equal(audit_raw, first_transition["raw"]):
            raise ValueError("restore_audit_first_transition_observation_mismatch")
        if not state_equal(audit_controller, first_transition["controller"]):
            raise ValueError("restore_audit_first_transition_controller_mismatch")
        if controller_contract(audit_controller_obj) != first_transition["controller_contract"]:
            raise ValueError("restore_audit_first_transition_use_target_mismatch")
        if bool(np.asarray(audit_term).reshape(-1)[0]) != first_transition["terminated"] or bool(np.asarray(audit_trunc).reshape(-1)[0]) != first_transition["truncated"]:
            raise ValueError("restore_audit_first_transition_flags_mismatch")
        if audit_native_info != first_transition["info"]:
            raise ValueError("restore_audit_first_transition_info_mismatch")
    finally:
        audit_env.close()
    return {"row": row, "catalog": candidate["catalog"], "actions": np.asarray(executed),
            "raw_observations": np.asarray(raw_obs), "policy_observations": np.asarray(policy_obs),
            "states": states, "controllers": controllers, "infos": [jsonable(v) for v in infos],
            "controller_contracts": controller_contracts,
            "terminated": np.asarray(terminated, dtype=np.uint8),
            "truncated": np.asarray(truncated, dtype=np.uint8),
            "native_success_step": success_at, "catalog_action_count": len(actions),
            "safety_trace": safety_trace, "restore_audit": {"passed": True,
                "initial_state_sha256": state_hash(initial_state),
                "controller_snapshot": jsonable(initial_controller), "initial_observation_exact": True,
                "first_transition_state_exact": True, "first_transition_controller_exact": True,
                "first_transition_observation_exact": True, "first_transition_info_and_flags_exact": True}}


def write_tree(group, name, value):
    if isinstance(value[0], dict):
        child = group.create_group(name)
        for key in value[0]:
            write_tree(child, key, [item[key] for item in value])
        return
    arrays = [item.detach().cpu().numpy() if torch.is_tensor(item) else np.asarray(item)
              for item in value]
    group.create_dataset(name, data=np.stack(arrays), compression="gzip", compression_opts=1)


def write_replay_artifact(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as handle:
        group = handle.create_group("traj")
        group.attrs["kind"] = "obstacle"
        group.attrs["metadata_json"] = json.dumps({
            "catalog": record["catalog"], "scene_id": record["row"]["scene_id"],
            "seed": record["row"]["seed"], "native_success_step": record["native_success_step"],
            "catalog_action_count": record["catalog_action_count"],
            "actions_truncated_at_first_native_held_success": record["native_success_step"] < record["catalog_action_count"],
            "safety_trace": record["safety_trace"], "restore_audit": record["restore_audit"]},
            sort_keys=True)
        group.create_dataset("actions", data=record["actions"], compression="gzip", compression_opts=1)
        group.create_dataset("obs", data=record["raw_observations"], compression="gzip", compression_opts=1)
        group.create_dataset("policy_obs", data=record["policy_observations"], compression="gzip", compression_opts=1)
        group.create_dataset("success", data=np.asarray([False] * len(record["actions"]) + [True], dtype=np.uint8))
        group.create_dataset("terminated", data=record["terminated"])
        group.create_dataset("truncated", data=record["truncated"])
        strings = h5py.string_dtype("utf-8")
        group.create_dataset("infos_json", data=np.asarray([json.dumps(x, sort_keys=True) for x in record["infos"]], dtype=strings))
        group.create_dataset("controllers_json", data=np.asarray([json.dumps(jsonable(x), sort_keys=True) for x in record["controllers"]], dtype=strings))
        group.create_dataset("controller_contracts_json", data=np.asarray([json.dumps(x, sort_keys=True) for x in record["controller_contracts"]], dtype=strings))
        states_group = group.create_group("env_states")
        for key in record["states"][0]:
            write_tree(states_group, key, [state[key] for state in record["states"]])


def copy_and_append(old_root, direct_root, data_dir, accepted, registration):
    data_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((old_root / "trajectory.json").read_text())
    old_episodes = manifest["episodes"]
    if len(old_episodes) != 1300:
        raise ValueError("expected exactly 1,000 direct + 300 obstacle source episodes")
    geometry_by_seed = {}
    for path in (*RETAINED_CATALOGS, *OLD_GEOMETRY_SOURCES):
        manifest_path = path / "manifest.json" if path.is_dir() else path
        for scene in json.loads(manifest_path.read_text()).get("scenes", []):
            geometry_by_seed[int(scene["seed"])] = scene
    new_manifest = []
    for episode in old_episodes:
        item = dict(episode)
        item["source_identity"] = {"path": str(old_root), "episode_id": episode["episode_id"],
                                   "arrays_copied_unchanged": True}
        if episode["kind"] == "obstacle":
            scene = geometry_by_seed.get(int(episode["layout"]["seed"]))
            if scene is None:
                raise ValueError(f"source_obstacle_geometry_unresolved:{episode['episode_id']}")
            item["cylinder"] = scene["cylinder"]
            item["layout"] = {**item["layout"], "cylinder": scene["cylinder"]}
        new_manifest.append(item)
    with h5py.File(old_root / "trajectory.h5", "r") as source, \
            h5py.File(data_dir / "trajectory.h5", "w") as target:
        for row in old_episodes:
            source.copy(source[f"traj_{row['episode_id']}"], target, name=f"traj_{row['episode_id']}")
        for offset, record in enumerate(accepted):
            episode_id = 1300 + offset
            with h5py.File(record["artifact"], "r") as replay:
                replay.copy(replay["traj"], target, name=f"traj_{episode_id}")
                group = target[f"traj_{episode_id}"]
                artifact_meta = json.loads(group.attrs["metadata_json"])
                del group.attrs["metadata_json"]
            row = record["row"]
            cylinder = row["cylinder"]
            new_manifest.append({"episode_id": episode_id, "kind": "obstacle", "seed": int(row["seed"]),
                "scene_id": row["scene_id"], "layout": {"seed": int(row["seed"]),
                    "start_xy": row["start_xy"], "goal_xy": row["goal_xy"],
                    "center_xy": cylinder["center_xy"], "cylinder": cylinder},
                "cylinder": cylinder, "waypoint_xy": row["expert"].get("waypoint_xy"),
                "native_success": True, "held_at_success": True, "obstacle_collision": False,
                "obstacle_crossing": False, "native_success_step": artifact_meta["native_success_step"],
                "catalog_action_count": artifact_meta["catalog_action_count"],
                "actions_truncated_at_first_native_held_success": artifact_meta["actions_truncated_at_first_native_held_success"],
                "native_safety_trace_through_success": artifact_meta["safety_trace"],
                "source": {"catalog": record["catalog"], "scene_id": row["scene_id"],
                           "state_sha256": row["state_sha256"],
                           "expert_actions_sha256": record["expert_actions_sha256"]},
                "restore_audit": artifact_meta["restore_audit"]})
    new_manifest = {"format": "pickcube-fullscene500-v1",
        "preregistration_sha256": registration["registration_sha256"],
        "source_implementation": registration["replay_implementation"],
        "dataset_contract": {"task_conflict": registration["identity"]["task_conflict"],
            "full_scene_conflict_train": registration["identity"]["full_scene_conflict_train"],
            "external_dev_ticket03": registration["identity"]["external_dev_ticket03"],
            "selected_obstacle_count_means_distinct_real_native_demonstrations": True},
        "episodes": new_manifest, "counts": {"direct": 1000, "obstacle": 500},
        "class_ids": {"direct": [row["episode_id"] for row in new_manifest if row["kind"] == "direct"],
                      "obstacle": [row["episode_id"] for row in new_manifest if row["kind"] == "obstacle"]},
        "source_obstacle300": {"path": str(old_root), "trajectory_sha256": sha(old_root / "trajectory.h5"),
                               "manifest_sha256": sha(old_root / "trajectory.json")},
        "direct_source": {"path": str(direct_root), "trajectory_sha256": sha(direct_root / "trajectory.h5"),
                          "manifest_sha256": sha(direct_root / "trajectory.json")}}
    new_manifest["trajectory_sha256"] = sha(data_dir / "trajectory.h5")
    (data_dir / "trajectory.json").write_text(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n")
    # Every old array is compared after h5py group-copy; append cannot mutate source bytes.
    with h5py.File(old_root / "trajectory.h5", "r") as src, \
            h5py.File(direct_root / "trajectory.h5", "r") as direct_src, \
            h5py.File(data_dir / "trajectory.h5", "r") as dst:
        for row in old_episodes:
            sid = f"traj_{row['episode_id']}"
            for key in src[sid]:
                if key not in dst[sid] or not np.array_equal(src[sid][key][:], dst[sid][key][:]):
                    raise ValueError(f"old_episode_array_changed:{sid}/{key}")
            if row["kind"] == "direct":
                for key in src[sid]:
                    if key not in direct_src[sid] or not np.array_equal(src[sid][key][:], direct_src[sid][key][:]):
                        raise ValueError(f"original_direct_array_changed:{sid}/{key}")


def canary(registration_path, output_dir):
    registration = json.loads(registration_path.read_text())
    expected = registration.pop("registration_sha256")
    if canonical_hash(registration) != expected:
        raise ValueError("registration checksum mismatch")
    registration["registration_sha256"] = expected
    if sha(registration["replay_implementation"]["path"]) != registration["replay_implementation"]["sha256"]:
        raise ValueError("replay implementation changed after registration")
    if sha(registration["replay_implementation"]["identity_helper"]) != registration["replay_implementation"]["identity_helper_sha256"]:
        raise ValueError("identity helper changed after registration")
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for index, candidate in enumerate(registration["pool"][:3]):
        geometry = candidate["cylinder"]
        env = make_pickcube_env(obstacle_radius=float(geometry["radius"]),
                                obstacle_height=float(geometry["height"]))
        started = time.monotonic()
        try:
            record = replay_candidate(candidate, env)
            results.append({"pool_index": index, "scene_id": candidate["scene_id"],
                            "catalog": candidate["catalog"], "success": True,
                            "native_success_step": record["native_success_step"],
                            "restore_audit": record["restore_audit"],
                            "elapsed_seconds": time.monotonic() - started})
        except Exception as error:
            results.append({"pool_index": index, "scene_id": candidate["scene_id"],
                            "catalog": candidate["catalog"], "success": False,
                            "reason": f"{type(error).__name__}:{error}",
                            "elapsed_seconds": time.monotonic() - started})
        finally:
            env.close()
    (output_dir / "canary.json").write_text(json.dumps({"registration_sha256": expected,
        "attempted": len(results), "results": results}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))


def run(registration_path, data_dir, exp_dir):
    registration = json.loads(registration_path.read_text())
    expected = registration.pop("registration_sha256")
    if canonical_hash(registration) != expected:
        raise ValueError("registration checksum mismatch")
    registration["registration_sha256"] = expected
    for filename, expected_hash in registration["inputs"]["sha256"].items():
        if sha(filename) != expected_hash:
            raise ValueError(f"frozen input changed: {filename}")
    if sha(registration["replay_implementation"]["path"]) != registration["replay_implementation"]["sha256"]:
        raise ValueError("replay implementation changed after registration")
    if sha(registration["replay_implementation"]["identity_helper"]) != registration["replay_implementation"]["identity_helper_sha256"]:
        raise ValueError("identity helper changed after registration")
    exp_dir.mkdir(parents=True, exist_ok=False)
    (exp_dir / "registration.json").write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n")
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    accepted, rejected = [], []
    journal = exp_dir / "attempts.jsonl"
    start = time.monotonic()
    env = None
    env_geometry = None
    try:
        for i, candidate in enumerate(registration["pool"]):
            if len(accepted) == TARGET:
                break
            elapsed = time.monotonic() - start
            if elapsed >= OUTER_WALL_S:
                break
            begin = time.monotonic()
            outcome = {"pool_index": i, "catalog": candidate["catalog"],
                       "scene_id": candidate["scene_id"], "seed": candidate["seed"]}
            try:
                cyl = candidate["cylinder"]
                geometry_key = (float(cyl["radius"]), float(cyl["height"]))
                if env is None or env_geometry != geometry_key:
                    if env is not None:
                        env.close()
                    env = make_pickcube_env(obstacle_radius=float(cyl["radius"]),
                                            obstacle_height=float(cyl["height"]))
                    env_geometry = geometry_key
                # Each native episode has a hard wall; terminate the complete run
                # externally with GNU timeout if a simulator call wedges.
                record = replay_candidate(candidate, env)
                artifact = exp_dir / "replays" / f"replay_{len(accepted):03d}.h5"
                write_replay_artifact(artifact, record)
                accepted_row = {"row": record["row"], "catalog": record["catalog"],
                                "artifact": str(artifact),
                                "expert_actions_sha256": candidate["expert_actions_sha256"],
                                "native_success_step": record["native_success_step"]}
                accepted.append(accepted_row)
                outcome.update({"accepted": True, "native_success_step": record["native_success_step"],
                                "restore_audit": record["restore_audit"]})
            except Exception as error:
                outcome.update({"accepted": False, "reason": f"{type(error).__name__}:{error}"})
                rejected.append(outcome.copy())
                if env is not None:
                    env.close(); env = None; env_geometry = None
            outcome["elapsed_seconds"] = time.monotonic() - begin
            outcome["accepted_count"] = len(accepted)
            with journal.open("a") as handle:
                handle.write(json.dumps(outcome, sort_keys=True) + "\n")
            if outcome["elapsed_seconds"] > PER_CANDIDATE_WALL_S:
                break
    finally:
        if env is not None:
            env.close()
    report = {"registration_sha256": expected, "target": TARGET, "accepted": len(accepted),
              "rejected": len(rejected), "attempted": len(accepted) + len(rejected),
              "pool_remaining": len(registration["pool"]) - len(accepted) - len(rejected),
              "elapsed_seconds": time.monotonic() - start,
              "source_identity_after_run": registration["inputs"]["sha256"],
              "accepted_scene_ids": [{"catalog": row["catalog"], "scene_id": row["row"]["scene_id"],
                                      "seed": row["row"]["seed"]} for row in accepted],
              "rejections": rejected}
    (exp_dir / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if len(accepted) != TARGET:
        raise RuntimeError(f"native replay admitted {len(accepted)}/{TARGET}; no dataset assembled")
    copy_and_append(Path(registration["inputs"]["old_obstacle300"]),
                    Path(registration["inputs"]["direct_source"]), data_dir, accepted, registration)
    report["dataset"] = str(data_dir)
    report["trajectory_sha256"] = sha(data_dir / "trajectory.h5")
    report["manifest_sha256"] = sha(data_dir / "trajectory.json")
    (exp_dir / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("accepted", "rejected", "attempted", "elapsed_seconds", "trajectory_sha256", "manifest_sha256")}, indent=2))


def audit_dataset(data_dir, output_path):
    root = Path(data_dir)
    manifest_path, h5_path = root / "trajectory.json", root / "trajectory.h5"
    manifest = json.loads(manifest_path.read_text())
    episodes = manifest["episodes"]
    if sorted(int(row["episode_id"]) for row in episodes) != list(range(1500)):
        raise ValueError("manifest_episode_ids_must_cover_0_through_1499")
    by_id = {int(row["episode_id"]): row for row in episodes}
    classes = {kind: [row for row in episodes if row["kind"] == kind]
               for kind in ("direct", "obstacle")}
    if {key: len(value) for key, value in classes.items()} != {"direct": 1000, "obstacle": 500}:
        raise ValueError("dataset_class_counts_mismatch")
    with h5py.File(h5_path, "r") as handle:
        if len(handle) != 1500:
            raise ValueError("hdf5_must_contain_exactly_1500_episodes")
        old_h5 = h5py.File(manifest["source_obstacle300"]["path"] + "/trajectory.h5", "r")
        direct_h5 = h5py.File(manifest["direct_source"]["path"] + "/trajectory.h5", "r")
        try:
            action_rows = {kind: [] for kind in classes}
            lengths = {kind: [] for kind in classes}
            windows = {kind: 0 for kind in classes}
            new_scene_rows = []
            for episode_id in range(1500):
                row = by_id[episode_id]
                group = handle[f"traj_{episode_id}"]
                if group.attrs["kind"] != row["kind"]:
                    raise ValueError(f"episode_{episode_id}_class_attribute_mismatch")
                actions, obs = group["actions"][:], group["obs"][:]
                if actions.ndim != 2 or actions.shape[1] != 7 or obs.shape != (len(actions) + 1, 45):
                    raise ValueError(f"episode_{episode_id}_action_observation_shape_mismatch")
                if not np.isfinite(actions).all() or not np.isfinite(obs).all():
                    raise ValueError(f"episode_{episode_id}_contains_nonfinite_values")
                if not bool(group["success"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"episode_{episode_id}_missing_native_success_or_truncated")
                action_rows[row["kind"]].append(actions)
                lengths[row["kind"]].append(len(actions))
                windows[row["kind"]] += max(0, len(actions) - 1)
                if episode_id < 1300:
                    source = old_h5[f"traj_{episode_id}"]
                    for key in source:
                        if not np.array_equal(group[key][:], source[key][:]):
                            raise ValueError(f"old_source_array_changed:{episode_id}/{key}")
                    if row["kind"] == "direct":
                        source_direct = direct_h5[f"traj_{episode_id}"]
                        for key in source_direct:
                            if not np.array_equal(group[key][:], source_direct[key][:]):
                                raise ValueError(f"direct_source_array_changed:{episode_id}/{key}")
                else:
                    T = len(actions)
                    for key in ("policy_obs", "infos_json", "controllers_json", "controller_contracts_json"):
                        if len(group[key]) != T + 1:
                            raise ValueError(f"episode_{episode_id}_{key}_not_T_plus_1")
                    if group["policy_obs"].shape != (T + 1, 42):
                        raise ValueError(f"episode_{episode_id}_policy_shape_mismatch")
                    if not np.array_equal(obs[:, :42], group["policy_obs"][:]):
                        raise ValueError(f"episode_{episode_id}_policy_raw_prefix_mismatch")
                    def assert_state_rows(node):
                        if isinstance(node, h5py.Dataset):
                            if len(node) != T + 1:
                                raise ValueError(f"episode_{episode_id}_state_not_T_plus_1:{node.name}")
                        else:
                            for child in node.values():
                                assert_state_rows(child)
                    assert_state_rows(group["env_states"])
                    if json.loads(group["controllers_json"][-1]) != {}:
                        raise ValueError(f"episode_{episode_id}_controller_snapshot_contract_changed")
                    contracts = [json.loads(value) for value in group["controller_contracts_json"][:]]
                    if any(any(item["attribute_present"] and item["use_target"]
                               for item in contract["controllers"].values()) for contract in contracts):
                        raise ValueError(f"episode_{episode_id}_controller_use_target_true")
                    if row["restore_audit"]["passed"] is not True:
                        raise ValueError(f"episode_{episode_id}_restore_audit_missing")
                    if len(row["native_safety_trace_through_success"]) != T or any(
                        step["obstacle_collision"] or step["obstacle_crossing"]
                        or (step["native_success"] and not step["held"])
                        for step in row["native_safety_trace_through_success"]
                    ):
                        raise ValueError(f"episode_{episode_id}_unsafe_or_incomplete_native_prefix")
                    if not row["native_safety_trace_through_success"][-1]["native_success"] or not row["native_safety_trace_through_success"][-1]["held"]:
                        raise ValueError(f"episode_{episode_id}_does_not_end_at_first_held_native_success")
                    new_scene_rows.append(row)
            full_unique = []
            task_unique = []
            for row in classes["obstacle"]:
                identity = {"kind": "obstacle", "layout": {"seed": row.get("seed", row["layout"].get("seed")),
                    "start_xy": row["layout"]["start_xy"], "goal_xy": row["layout"]["goal_xy"]},
                    "cylinder": row.get("cylinder", {})}
                if not any(full_scene_duplicate(identity, prior) for prior in full_unique):
                    full_unique.append(identity)
                if not any(task_duplicate(identity, prior) for prior in task_unique):
                    task_unique.append(identity)
            stats = {}
            for kind, chunks in action_rows.items():
                all_actions = np.concatenate(chunks, axis=0)
                episode_rms = np.stack([np.sqrt(np.mean(chunk.astype(np.float64) ** 2, axis=0))
                                        for chunk in chunks])
                stats[kind] = {"episodes": len(chunks), "length_min_median_p90_p95_max":
                    [int(np.min(lengths[kind])), float(np.median(lengths[kind])),
                     float(np.percentile(lengths[kind], 90)), float(np.percentile(lengths[kind], 95)),
                     int(np.max(lengths[kind]))], "training_windows_obs2_pred16_before1_after14": windows[kind],
                    "action_channel_min": np.min(all_actions, axis=0).tolist(),
                    "action_channel_max": np.max(all_actions, axis=0).tolist(),
                    "episode_rms_quantiles_0_05_50_95_100": np.quantile(episode_rms, [0, .05, .5, .95, 1], axis=0).tolist()}
        finally:
            old_h5.close(); direct_h5.close()
    result = {"audit": "pickcube-fullscene500-readonly-v1", "counts": {k: len(v) for k, v in classes.items()},
        "episode_ids": {k: sorted(int(row["episode_id"]) for row in v) for k, v in classes.items()},
        "obstacle_task_unique": len(task_unique), "obstacle_full_scene_unique": len(full_unique),
        "direct_arrays_match_original_source": True, "old_1300_episode_arrays_match_source": True,
        "new_obstacle_restore_audits_passed": len(new_scene_rows), "native_controller_contract":
        "T+1 snapshots are {}, use_target remains false wherever the runtime exposes it",
        "action_statistics": stats, "trajectory_sha256": sha(h5_path),
        "manifest_sha256": sha(manifest_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("counts", "obstacle_task_unique",
          "obstacle_full_scene_unique", "new_obstacle_restore_audits_passed",
          "trajectory_sha256", "manifest_sha256")}, indent=2))


def assemble_from_replays(registration_path, replay_dir, data_dir):
    registration = json.loads(registration_path.read_text())
    expected = registration.pop("registration_sha256")
    if canonical_hash(registration) != expected:
        raise ValueError("registration checksum mismatch")
    registration["registration_sha256"] = expected
    for filename, expected_hash in registration["inputs"]["sha256"].items():
        if sha(filename) != expected_hash:
            raise ValueError(f"frozen input changed: {filename}")
    lookup = {(candidate["catalog"], candidate["scene_id"]): candidate
              for candidate in registration["pool"]}
    accepted = []
    for path in sorted(Path(replay_dir).glob("replay_*.h5")):
        with h5py.File(path, "r") as handle:
            metadata = json.loads(handle["traj"].attrs["metadata_json"])
        candidate = lookup[(metadata["catalog"], metadata["scene_id"])]
        if metadata["restore_audit"]["passed"] is not True:
            raise ValueError(f"replay artifact failed restore audit: {path}")
        accepted.append({"row": candidate["row"], "catalog": candidate["catalog"],
                         "artifact": str(path),
                         "expert_actions_sha256": candidate["expert_actions_sha256"],
                         "native_success_step": metadata["native_success_step"]})
    if len(accepted) != TARGET:
        raise ValueError(f"expected exactly {TARGET} persisted native replays, found {len(accepted)}")
    copy_and_append(Path(registration["inputs"]["old_obstacle300"]),
                    Path(registration["inputs"]["direct_source"]), data_dir, accepted, registration)


class PickCubeFullScene500Dataset(Dataset):
    """Opt-in manifest-pinned PickCube window reader for the 1,000/500 format."""

    def __init__(self, data_path, manifest_path, *, expected_h5_sha256,
                 expected_manifest_sha256, kind, obs_horizon=2, pred_horizon=16,
                 include_obstacle=False):
        self.data_path, self.manifest_path = Path(data_path), Path(manifest_path)
        if kind not in ("direct", "obstacle"):
            raise ValueError("kind must be direct or obstacle")
        self.kind = kind
        if sha(self.data_path) != expected_h5_sha256 or sha(self.manifest_path) != expected_manifest_sha256:
            raise ValueError("fullscene500 dataset identity mismatch")
        manifest = json.loads(self.manifest_path.read_text())
        if manifest.get("format") != "pickcube-fullscene500-v1" or manifest.get("trajectory_sha256") != expected_h5_sha256:
            raise ValueError("unsupported or unpinned fullscene500 manifest")
        episodes = manifest["episodes"]
        ids = manifest["class_ids"][kind]
        if len(ids) != (1000 if kind == "direct" else 500):
            raise ValueError("fullscene500 class ID count mismatch")
        by_id = {int(row["episode_id"]): row for row in episodes}
        if len(by_id) != 1500 or set(ids) != {int(row["episode_id"]) for row in episodes if row["kind"] == kind}:
            raise ValueError("fullscene500 explicit class IDs disagree with episode manifest")
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.include_obstacle = bool(include_obstacle)
        self.observations, self.actions, self.windows = [], [], []
        before, after = obs_horizon - 1, pred_horizon - obs_horizon
        with h5py.File(self.data_path, "r") as handle:
            if len(handle) != 1500:
                raise ValueError("fullscene500 HDF5 episode count mismatch")
            for episode_id in ids:
                row, group = by_id[int(episode_id)], handle[f"traj_{int(episode_id)}"]
                if row["kind"] != kind or group.attrs["kind"] != kind:
                    raise ValueError(f"episode {episode_id}: manifest/HDF5 class mismatch")
                obs, actions = np.asarray(group["obs"][:], dtype=np.float32), np.asarray(group["actions"][:], dtype=np.float32)
                if obs.shape != (len(actions) + 1, 45) or actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"episode {episode_id}: unexpected action/observation schema")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError(f"episode {episode_id}: nonfinite arrays")
                if not bool(group["success"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"episode {episode_id}: missing native successful endpoint")
                index = len(self.actions)
                self.observations.append(torch.as_tensor(obs))
                self.actions.append(torch.as_tensor(actions))
                T = len(actions)
                self.windows.extend((index, start, start + pred_horizon)
                                    for start in range(-before, T - pred_horizon + after))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode, start, end = self.windows[index]
        obs, actions = self.observations[episode], self.actions[episode]
        length = len(actions)
        obs_window = obs[max(0, start):start + self.obs_horizon]
        action_window = actions[max(0, start):end]
        if start < 0:
            obs_window = torch.cat((obs_window[0].repeat(-start, 1), obs_window))
            action_window = torch.cat((action_window[0].repeat(-start, 1), action_window))
        if end > length:
            padding = torch.cat((torch.zeros(6), action_window[-1, 6:7]))
            action_window = torch.cat((action_window, padding.repeat(end - length, 1)))
        if obs_window.shape != (self.obs_horizon, 45) or action_window.shape != (self.pred_horizon, 7):
            raise AssertionError("fullscene500 official window/padding invariant failed")
        observations = obs_window if self.include_obstacle and self.kind == "obstacle" else obs_window[..., :42]
        return {"observations": observations, "actions": action_window}


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("preregister")
    p.add_argument("--registration", type=Path, required=True)
    p.add_argument("--old-root", type=Path, required=True)
    p.add_argument("--direct-root", type=Path, required=True)
    p.add_argument("--exp-dir", type=Path, required=True)
    c = commands.add_parser("canary")
    c.add_argument("--registration", type=Path, required=True)
    c.add_argument("--output-dir", type=Path, required=True)
    r = commands.add_parser("run")
    r.add_argument("--registration", type=Path, required=True)
    r.add_argument("--data-dir", type=Path, required=True)
    r.add_argument("--exp-dir", type=Path, required=True)
    a = commands.add_parser("audit")
    a.add_argument("--data-dir", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    m = commands.add_parser("assemble")
    m.add_argument("--registration", type=Path, required=True)
    m.add_argument("--replay-dir", type=Path, required=True)
    m.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preregister":
        preregister(args.registration, args.old_root, args.direct_root, args.exp_dir)
    elif args.command == "canary":
        canary(args.registration, args.output_dir)
    elif args.command == "audit":
        audit_dataset(args.data_dir, args.output)
    elif args.command == "assemble":
        assemble_from_replays(args.registration, args.replay_dir, args.data_dir)
    else:
        run(args.registration, args.data_dir, args.exp_dir)


if __name__ == "__main__":
    main()
