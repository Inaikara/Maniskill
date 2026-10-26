"""Independent saved-artifact audit for the matched PlaceSphere collection."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.placesphere_matched_collection import (
    assert_preregistration,
    _greedy_task_identity,
)
from examples.baselines.constrain.placesphere_matched_terminal import (
    NATIVE_ANG_THRESH_RAD_S,
    NATIVE_LIN_THRESH_M_S,
    NATIVE_XY_M,
    RADIUS_M,
    positive_direct_accepted,
    positive_obstacle_accepted,
)
from examples.baselines.constrain.placesphere_pilot import _actions_are_legal


def _check_tree_lengths(node, expected, label):
    for name, item in node.items():
        if isinstance(item, h5py.Group):
            _check_tree_lengths(item, expected, f"{label}/{name}")
        elif item.shape[0] != expected:
            raise ValueError(f"{label}/{name} does not have T+1 rows")


def _layout_key(layout):
    return (layout["catalog"], int(layout["index"]), int(layout["seed"]))


def _flag(info, name):
    return bool(np.asarray(info[name]).reshape(-1)[0])


def validate_collection(dataset_dir, experiment_dir, registration_path):
    dataset, experiment = Path(dataset_dir), Path(experiment_dir)
    registration = assert_preregistration(registration_path)
    report = json.loads((dataset / "report.json").read_text())
    manifest = json.loads((dataset / "trajectory.json").read_text())["episodes"]
    expected = registration["targets"]
    counts = {kind: sum(row["kind"] == kind for row in manifest) for kind in ("direct", "obstacle")}
    if not report.get("complete") or counts != {"direct": int(expected["direct"]),
                                                  "obstacle": int(expected["obstacle"])}:
        raise ValueError(f"dataset counts are incomplete: {counts}")
    if len(manifest) != sum(counts.values()):
        raise ValueError("trajectory manifest episode count mismatch")
    raw_schema = registration["runtime_contract"]["raw_observation_schema"]
    policy_schema = registration["runtime_contract"]["policy_observation_schema"]
    raw_dim = registration["runtime_contract"]["raw_observation_dimension"]
    policy_dim = registration["runtime_contract"]["policy_observation_dimension"]
    if (registration["runtime_contract"]["action_dimension"] != 7 or raw_dim != 42 or policy_dim != 39
            or sum(field["flattened_dimension"] for field in raw_schema) != raw_dim
            or sum(field["flattened_dimension"] for field in policy_schema) != policy_dim):
        raise ValueError("frozen named action/observation contract is invalid")
    policy_columns, offset = [], 0
    for field in raw_schema:
        width = int(field["flattened_dimension"])
        if field["path"] != "/extra/placesphere_obstacle":
            policy_columns.extend(range(offset, offset + width))
        offset += width
    if len(policy_columns) != policy_dim:
        raise ValueError("raw-to-policy field selection does not produce 39 columns")

    episode_checks, direct_layouts, obstacle_layouts = [], [], []
    layout_id_mapping = []
    phase_action_counts = {"direct": Counter(), "obstacle": Counter()}
    restore_passed = 0
    paired_direct = {}
    with h5py.File(dataset / "trajectory.h5", "r") as h5:
        if len(h5) != len(manifest):
            raise ValueError("HDF5 and JSON episode counts differ")
        for index, episode in enumerate(manifest):
            key = f"traj_{index}"
            group = h5[key]
            kind = episode["kind"]
            metadata = json.loads(group.attrs["metadata_json"])
            record = metadata["expert_record"]
            layout = episode["layout"]
            (direct_layouts if kind == "direct" else obstacle_layouts).append(layout)
            layout_id_mapping.append({"episode_id": index, "kind": kind, "source_catalog": layout["catalog"],
                                      "source_scene_id": layout["scene_id"], "source_index": int(layout["index"]),
                                      "seed": int(layout["seed"]), "start_xy": layout["start_xy"],
                                      "container_xy": layout["container_xy"]})
            actions = np.asarray(group["actions"])
            policy = np.asarray(group["obs"])
            raw = np.asarray(group["raw_obs"])
            T = len(actions)
            if (actions.shape != (T, 7) or policy.shape != (T + 1, 39) or raw.shape != (T + 1, 42)
                    or not np.isfinite(actions).all() or not np.isfinite(policy).all() or not np.isfinite(raw).all()
                    or not _actions_are_legal(actions) or not np.array_equal(raw[:, policy_columns], policy)):
                raise ValueError(f"{key}: action/native observation contract failed")
            if (len(group["success"]) != T + 1 or len(group["terminated"]) != T + 1
                    or len(group["truncated"]) != T + 1 or not group["success"][-1]
                    or not group["terminated"][-1] or group["truncated"][:].any()):
                raise ValueError(f"{key}: first-native-success T+1 evidence is invalid")
            _check_tree_lengths(group["env_states"], T + 1, f"{key}/env_states")
            _check_tree_lengths(group["controller_states"], T + 1, f"{key}/controller_states")
            infos = episode["infos"]
            if len(infos) != T + 1:
                raise ValueError(f"{key}: native info record is not T+1")
            final = infos[-1]
            if not (_flag(final, "native_success") and _flag(final, "is_obj_on_bin")
                    and _flag(final, "is_obj_static") and not _flag(final, "is_obj_grasped")):
                raise ValueError(f"{key}: terminal info is not native static ungrasped placement")
            if record.get("observation_contract_ok") is not True:
                raise ValueError(f"{key}: runtime named observation contract was not proven")
            audit = metadata.get("t_plus_one_restore_audit") or {}
            if not audit.get("passed"):
                raise ValueError(f"{key}: exact initial/first-transition restore audit failed")
            restore_passed += 1
            if record.get("radius_m") != RADIUS_M or record.get("native_xy_m") != NATIVE_XY_M:
                raise ValueError(f"{key}: radius/native threshold changed")
            if not (record.get("native_success") and record.get("on_bin")
                    and record.get("static") and record.get("ungrasped")):
                raise ValueError(f"{key}: saved expert record lacks native success")
            if kind == "direct":
                if not positive_direct_accepted(record):
                    raise ValueError(f"{key}: positive DIRECT admission predicate failed")
                paired_direct[_layout_key(layout)] = metadata
            else:
                if not positive_obstacle_accepted(record):
                    raise ValueError(f"{key}: positive OBSTACLE full-prefix admission predicate failed")
                for info in infos:
                    if _flag(info, "obstacle_collision") or _flag(info, "obstacle_crossing"):
                        raise ValueError(f"{key}: obstacle collision/crossing in native info sequence")
                if not metadata["matched_admission"].get("collection_ready"):
                    raise ValueError(f"{key}: paired role admission failed")
            for phase in metadata["step_phase"]:
                phase_action_counts[kind][phase] += 1
            episode_checks.append({"episode_id": index, "kind": kind, "T": T,
                                   "T_plus_1": T + 1, "restore_audit": audit,
                                   "full_prefix_safe": kind != "obstacle" or record["from_execution_start_safe"]})

    obstacle_keys = {_layout_key(row) for row in obstacle_layouts}
    if not obstacle_keys.issubset(set(paired_direct)):
        raise ValueError("an OBSTACLE training layout lacks its regenerated paired DIRECT trajectory")
    if len(obstacle_keys) != int(expected["obstacle"]):
        raise ValueError("OBSTACLE training identities are not unique by source identity")
    task_identity = {
        "direct": _greedy_task_identity(direct_layouts),
        "obstacle": _greedy_task_identity(obstacle_layouts),
        "external_dev_ticket04_conflicts": report["task_identity"]["external_dev_ticket04_conflicts"],
    }
    if task_identity["obstacle"]["greedy_unique"] != int(expected["obstacle"]):
        raise ValueError("OBSTACLE task-layout dedup audit did not retain 300 unique layouts")
    attempts = [json.loads(line) for line in (dataset / "attempts.jsonl").read_text().splitlines() if line]
    if len(attempts) != report["attempted_layouts"]:
        raise ValueError("attempt journal denominator does not equal attempted layout count")
    paired_attempts = [row for row in attempts if row.get("accepted") and row.get("terminal_replay") is not None]
    if len(paired_attempts) != int(expected["obstacle"]):
        raise ValueError("paired negative/replay proof count differs from obstacle target")
    if any(row.get("negative_blocked_in_transport") is not True for row in paired_attempts):
        raise ValueError("a paired negative probe failed to prove the expected transport block")
    if report.get("source_identity") != registration["source_identity"]:
        raise ValueError("collection source identity differs from preregistration")

    mapping_path = experiment / "layout_id_mapping.json"
    mapping_path.write_text(json.dumps(layout_id_mapping, indent=2, sort_keys=True) + "\n")
    artifact_hashes = {}
    for name in ("trajectory.h5", "trajectory.json", "attempts.jsonl", "report.json",
                 "frozen_contract.json", "collection_config.json"):
        artifact_hashes[name] = hashlib.sha256((dataset / name).read_bytes()).hexdigest()
    artifact_hashes["registration.json"] = hashlib.sha256(Path(registration_path).read_bytes()).hexdigest()
    artifact_hashes["layout_id_mapping.json"] = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    source_counts = {
        kind: dict(Counter(row["source_catalog"] for row in layout_id_mapping if row["kind"] == kind))
        for kind in ("direct", "obstacle")
    }
    result = {
        "complete": True,
        "class_counts": counts,
        "episode_count": len(episode_checks),
        "t_plus_one_restore_audits_passed": restore_passed,
        "paired_layouts_with_negative_and_replay_proofs": len(paired_attempts),
        "obstacle_task_identity": task_identity["obstacle"],
        "direct_task_identity": task_identity["direct"],
        "source_layout_counts_by_role": source_counts,
        "matched_direct_obstacle_layout_count": len(obstacle_keys),
        "external_dev_ticket04_conflicts": task_identity["external_dev_ticket04_conflicts"],
        "action_dimension": 7,
        "policy_observation_dimension": policy_dim,
        "raw_observation_dimension": raw_dim,
        "native_thresholds_changed": False,
        "obstacle_geometry_changed": False,
        "phase_action_counts": {kind: dict(counts) for kind, counts in phase_action_counts.items()},
        "episodes": episode_checks,
        "artifact_sha256": artifact_hashes,
        "collection_identity": registration["source_identity"],
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    experiment.mkdir(parents=True, exist_ok=True)
    (experiment / "dataset_validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (experiment / "dataset_identity.json").write_text(json.dumps({
        "artifact_sha256": artifact_hashes,
        "validation_sha256": hashlib.sha256((experiment / "dataset_validation.json").read_bytes()).hexdigest(),
        "collection_identity": registration["source_identity"],
        "auditor_sha256": result["auditor_sha256"],
    }, indent=2, sort_keys=True) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args(argv)
    result = validate_collection(args.dataset_dir, args.experiment_dir, args.registration)
    print(json.dumps({"complete": result["complete"], "class_counts": result["class_counts"],
                      "restore_audits": result["t_plus_one_restore_audits_passed"],
                      "obstacle_unique": result["obstacle_task_identity"]["greedy_unique"]}, sort_keys=True))


if __name__ == "__main__":
    main()
