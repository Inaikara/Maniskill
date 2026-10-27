"""Recover four task-identity exclusions from the already-frozen R26 pool."""

import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.placesphere_obstacle1000_expansion import (
    frozen_candidate_layouts,
    sha256,
    task_layout_conflict,
)
from examples.baselines.constrain.placesphere_matched_collection import (
    _pair,
    _restore_one_step,
    _write_episode,
    _json_infos,
)
from examples.baselines.constrain import placesphere_matched_terminal as matched


ROOT = Path(__file__).resolve().parents[3]


def _layout_identity(row):
    layout = row.get("layout", row)
    return {"seed": int(row["seed"]), "start_xy": layout["start_xy"],
            "container_xy": layout["container_xy"]}


def run_replacements(registration_path, pool_manifest, existing_manifest, output_dir,
                     *, target=4, episode_wall_seconds=160, outer_wall_seconds=2400):
    registration_path, pool_manifest = Path(registration_path), Path(pool_manifest)
    existing_manifest, output_dir = Path(existing_manifest), Path(output_dir)
    reg = json.loads(registration_path.read_text())
    if reg.get("created_before_replay") is not True:
        raise ValueError("replacement replay lacks a before-run registration")
    if sha256(pool_manifest) != reg["pool_manifest_sha256"]:
        raise ValueError("frozen candidate pool manifest changed")
    if any(sha256(path) != digest for path, digest in reg["candidate_h5_sha256"].items()):
        raise ValueError("a frozen candidate source HDF5 changed")
    if any(sha256(ROOT / path) != digest for path, digest in reg["source_hashes"].items()):
        raise ValueError("replacement replay source identity changed")
    existing = json.loads(existing_manifest.read_text())["episodes"]
    accepted_identities = [_layout_identity(row) for row in existing
                           if int(row["episode_id"]) >= 1300
                           and int(row["episode_id"]) - 1300 not in reg["excluded_candidate_ids"]]
    layouts = frozen_candidate_layouts(pool_manifest)
    suffix = [row for row in layouts if row["original_id"] >= reg["first_remaining_candidate_id"]]
    expected = [row["original_id"] for row in suffix]
    if expected != reg["remaining_candidate_ids"]:
        raise ValueError("remaining candidate order differs from replacement preregistration")
    external = []
    for catalog in reg["external_manifests"]:
        document = json.loads(Path(catalog).read_text())
        if "episodes" in document:
            external.extend(_layout_identity(row) for row in document["episodes"]
                            if row.get("kind") == "obstacle")
        else:
            external.extend(_layout_identity(row) for row in document["scenes"])
    output_dir.mkdir(parents=True, exist_ok=False)
    results, admitted = [], []
    started = time.monotonic()
    with h5py.File(output_dir / "replacement_obstacles.h5", "x") as train_h5:
        with (output_dir / "attempts.jsonl").open("x") as journal:
            for row in suffix:
                if len(admitted) == target or time.monotonic() - started >= outer_wall_seconds:
                    break
                identity = _layout_identity(row)
                if (any(task_layout_conflict(identity, other) for other in external + accepted_identities)
                        or any(task_layout_conflict(identity, other["identity"]) for other in admitted)):
                    result = {"candidate_id": row["original_id"], "seed": row["seed"],
                              "admitted": False, "failure": "task_identity_duplicate_external_preflight",
                              "native_replay_attempted": False}
                else:
                    attempt_started = time.monotonic()
                    direct, obstacle, replay, negative, common, admission = _pair(
                        row, episode_wall_seconds, 8)
                    audits = {}
                    if admission.get("collection_ready"):
                        audits = {"direct": _restore_one_step(row, direct),
                                  "obstacle": _restore_one_step(row, obstacle)}
                    passed = bool(admission.get("collection_ready")
                                  and all(item.get("passed") is True for item in audits.values()))
                    attempt_path = output_dir / f"candidate_{row['original_id']:03d}.h5"
                    attempt_meta = {"candidate_id": row["original_id"], "seed": row["seed"],
                                    "admitted": passed, "admission": admission,
                                    "restore_audits": audits, "terminal_suffix_proof": replay,
                                    "elapsed_seconds": time.monotonic() - attempt_started}
                    with h5py.File(attempt_path, "x") as attempt_h5:
                        attempt_h5.attrs["metadata_json"] = json.dumps(attempt_meta, sort_keys=True, default=str)
                        for role, record in (("direct", direct), ("obstacle", obstacle), ("negative", negative)):
                            raw = (record or {}).get("_raw")
                            if not raw:
                                continue
                            group = attempt_h5.create_group(role)
                            for name in ("actions", "policy_observations", "raw_observations"):
                                values = np.asarray(raw[name], dtype=np.float32)
                                group.create_dataset(name, data=values)
                                group.attrs[f"{name}_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
                    result = {"candidate_id": row["original_id"], "seed": row["seed"],
                              "admitted": passed,
                              "failure": None if passed else admission.get("reasons", ["t_plus_one_restore_mismatch"]),
                              "native_replay_attempted": True, "attempt_h5": str(attempt_path),
                              "attempt_h5_sha256": sha256(attempt_path),
                              "admission": admission, "restore_audits": audits,
                              "terminal_suffix_proof": replay,
                              "elapsed_seconds": attempt_meta["elapsed_seconds"]}
                    if passed:
                        candidate_index = len(admitted)
                        final_id = 1496 + candidate_index
                        public = matched._public_record(obstacle)
                        metadata = {"episode_id": final_id, "kind": "obstacle",
                                    "source_candidate_id": row["original_id"], "source_seed": row["seed"],
                                    "layout": row["candidate_scene"], "matched_admission": admission,
                                    "t_plus_one_restore_audit": audits["obstacle"],
                                    "paired_direct_restore_audit": audits["direct"],
                                    "negative_transport_block_proof": bool(negative.get("blocked_in_transport")),
                                    "terminal_suffix_proof": replay,
                                    "common_release_measurements": None if common is None else common.get("release_target_measurements"),
                                    "step_phase": obstacle["_raw"]["trace"]["step_phase"],
                                    "sphere_linear_speed_m_s": obstacle["_raw"]["trace"]["sphere_linear_speed_m_s"],
                                    "sphere_angular_speed_rad_s": obstacle["_raw"]["trace"]["sphere_angular_speed_rad_s"],
                                    "expert_record": public,
                                    "source_candidate_h5_sha256": sha256(row["candidate_h5"])}
                        _write_episode(train_h5.create_group(f"traj_{candidate_index}"), obstacle,
                                       "obstacle", metadata)
                        train_h5.flush()
                        admitted.append({"candidate_index": candidate_index, "final_episode_id": final_id,
                                         "candidate_id": row["original_id"], "seed": row["seed"],
                                         "kind": "obstacle", "layout": row["candidate_scene"],
                                         "infos": _json_infos(obstacle["_raw"]["infos"]),
                                         "identity": identity})
                results.append(result)
                journal.write(json.dumps(result, sort_keys=True, default=str) + "\n")
                journal.flush()
    manifest = {"format": "placesphere_r26_obstacle_replacements_v1", "episodes": admitted,
                "attempts": results, "count": len(admitted),
                "registration_sha256": sha256(registration_path),
                "pool_manifest_sha256": sha256(pool_manifest)}
    (output_dir / "replacement_obstacles.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    report = {"target": target, "accepted": len(admitted), "attempts": len(results),
              "replayed_candidate_ids": [r["candidate_id"] for r in results if r["native_replay_attempted"]],
              "preflight_rejected_candidate_ids": [r["candidate_id"] for r in results if not r["native_replay_attempted"]],
              "rejected_candidate_ids": [r["candidate_id"] for r in results if not r["admitted"]],
              "unused_suffix_candidate_ids": [r["original_id"] for r in suffix[len(results):]],
              "elapsed_seconds": time.monotonic() - started,
              "replacement_h5_sha256": sha256(output_dir / "replacement_obstacles.h5"),
              "replacement_manifest_sha256": sha256(output_dir / "replacement_obstacles.json"),
              "complete": len(admitted) == target}
    (output_dir / "replacement_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
