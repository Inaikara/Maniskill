"""Audits and freezes the opt-in Round-25 PlaceSphere 1k/1k cohort.

This module never changes the trusted 1k/300 package.  Collection is performed
as a separate expert-positive cohort; this utility appends only its 700
OBSTACLE episodes, reusing the original 1,000 DIRECT and 300 OBSTACLE groups
byte-for-byte at the HDF5 dataset level.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_HOLD8_OBSTACLE500_VERSION,
)


SOURCE_DATA = Path("demos/PlaceSphere-transport-detour/20261026_184500_matched_terminal_hold8_r23")
TARGET_COUNTS = {"direct": 1000, "obstacle": 500}
OLD_COUNTS = {"direct": 1000, "obstacle": 300}
ADD_OBSTACLE = 200
MAX_FROZEN_CANDIDATES = 239
TASK_IDENTITY_TOLERANCE_M = 0.02
ROOT = Path(__file__).resolve().parents[3]
TRAIN_BASE = Path("outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt")
TRAIN_CONTROL = Path("outputs/ckpt/20261026_195000_placesphere_hold8_constrain_90k_r23/final.pt")
SCENES = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
PRIMARY = Path("outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval")

EXPERIMENT_SOURCES = (
    "examples/baselines/constrain/placesphere_obstacle1000_expansion.py",
    "examples/baselines/constrain/placesphere_matched_collection.py",
    "examples/baselines/constrain/placesphere_matched_terminal.py",
    "examples/baselines/constrain/placesphere_transport_detour.py",
    "examples/baselines/constrain/placesphere_pilot.py",
    "examples/baselines/constrain/placesphere_round20_tplusone.py",
    "examples/baselines/constrain/placesphere_manifest_dataset.py",
    "examples/baselines/constrain/placesphere_constrain.py",
    "examples/baselines/constrain/evaluate_placesphere_matched.py",
    "examples/baselines/constrain/placesphere_base_dp.py",
    "examples/baselines/constrain/tests/test_placesphere_obstacle1000_expansion.py",
)
NATIVE_SOURCES = (
    "mani_skill/envs/tasks/tabletop/place_sphere.py",
    "mani_skill/examples/motionplanning/panda/solutions/place_sphere.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _h5_group_digest(group):
    digest = hashlib.sha256()
    for key in sorted(group.attrs):
        value = group.attrs[key]
        digest.update(key.encode())
        digest.update(value if isinstance(value, bytes) else str(value).encode())
    for name in sorted(group):
        node = group[name]
        digest.update(name.encode())
        if isinstance(node, h5py.Group):
            digest.update(bytes.fromhex(_h5_group_digest(node)))
        else:
            digest.update(np.asarray(node).dtype.str.encode())
            digest.update(np.asarray(node).shape.__repr__().encode())
            digest.update(np.ascontiguousarray(node[:]).tobytes())
            for key in sorted(node.attrs):
                digest.update(key.encode())
                digest.update(str(node.attrs[key]).encode())
    return digest.hexdigest()


def task_layout_conflict(left, right, tolerance=TASK_IDENTITY_TOLERANCE_M):
    if int(left["seed"]) == int(right["seed"]):
        return True
    return bool(
        np.max(np.abs(np.asarray(left["start_xy"]) - np.asarray(right["start_xy"]))) <= tolerance
        and np.max(np.abs(np.asarray(left["container_xy"]) - np.asarray(right["container_xy"]))) <= tolerance
    )


def unique_layout_audit(rows):
    rows = [{**row.get("layout", {}), **row} for row in rows]
    accepted = []
    conflicts = []
    for row in rows:
        if any(task_layout_conflict(row, prior) for prior in accepted):
            conflicts.append(int(row["episode_id"]))
        else:
            accepted.append(row)
    return {"count": len(rows), "greedy_unique": len(accepted), "repeats": len(conflicts),
            "repeated_episode_ids": conflicts,
            "rule": "same seed, or start and container each within 0.02 m componentwise"}


def frozen_pool_selection(rows, *, limit=MAX_FROZEN_CANDIDATES, target=ADD_OBSTACLE):
    """Select the preregistered deterministic prefix; never proposes new layouts."""
    if len(rows) != MAX_FROZEN_CANDIDATES:
        raise ValueError(f"frozen candidate pool must contain exactly {MAX_FROZEN_CANDIDATES} layouts")
    ordered = sorted(rows, key=lambda row: (int(row["seed"]), int(row["original_id"])))
    if unique_layout_audit(ordered)["greedy_unique"] != len(ordered):
        raise ValueError("frozen candidate pool contains duplicate task identities")
    return {"pool_count": len(ordered), "one_pass_limit": len(ordered),
            "selection_target": int(target), "ordered_candidates": ordered,
            "ordering": "ascending(seed, original_id); accept the first 200 passing the frozen actual geometry/expert admission; one pass, no replacement or resampling"}


def freeze_existing_pool_selection(pool_manifest, output_path):
    """Persist source-pinned first-200 ordering and all 39 unused candidate IDs."""
    pool_path, output_path = Path(pool_manifest), Path(output_path)
    if output_path.exists():
        raise FileExistsError(output_path)
    pool = json.loads(pool_path.read_text())
    rows = []
    for original_id, scene in enumerate(pool.get("scenes", [])):
        expert = scene.get("expert", {})
        if (expert.get("native_success") is not True or expert.get("safe_success") is not True
                or expert.get("obstacle_collision") is not False
                or expert.get("obstacle_crossing") is not False):
            raise ValueError(f"frozen candidate {original_id} lacks source native-positive layout evidence")
        rows.append({"original_id": original_id, **scene})
    selection = frozen_pool_selection(rows)
    payload = {
        "format": "placesphere_r26_frozen_candidate_replay_selection_v1",
        "constraint": "1,000 direct + 500 obstacle; no new layout sampling",
        "pool_manifest": str(pool_path), "pool_manifest_sha256": sha256(pool_path),
        "pool_sources": pool.get("recovery_sources"),
        "candidate_h5_sha256": {str(Path(row["path"]) / "scenes.h5"):
                                sha256(Path(row["path"]) / "scenes.h5")
                                for row in pool.get("recovery_sources", [])},
        "implementation_sources": {
            name: sha256(ROOT / name) for name in (
                "examples/baselines/constrain/placesphere_obstacle1000_expansion.py",
                "examples/baselines/constrain/placesphere_matched_terminal.py",
                "examples/baselines/constrain/placesphere_matched_collection.py",
                "examples/baselines/constrain/placesphere_round20_tplusone.py",
                "examples/baselines/constrain/placesphere_transport_detour.py",
                "examples/baselines/constrain/placesphere_manifest_dataset.py",
            )
        },
        "pool_count": len(rows), "ordered_candidate_ids": [r["original_id"] for r in sorted(rows, key=lambda r: (r["seed"], r["original_id"]))],
        "ordered_candidate_ids": [r["original_id"] for r in selection["ordered_candidates"]],
        "selection_target": 200,
        "selection_rule": selection["ordering"], "replay_order": "one pass through all 239 in frozen order; stop immediately after 200 admissions",
        "unused_candidate_ids": "all non-admitted candidates after the 200th admission; if the pool is exhausted, report shortfall and do not sample or duplicate",
        "admission": "actual positive direct and obstacle native-safety, transport-block negative and shared-terminal suffix proof; policy performance is not an admission criterion",
        "training_additions_cap": 200,
        "final_class_counts": {"direct": 1000, "obstacle": 500},
        "training_plan": {"base": "reuse frozen Base EMA 92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c",
                          "control_updates": 150000, "batch_size": 1024, "learning_rate": 0.0001,
                          "warmup_updates": 500, "seed": 1, "schedule": "cosine",
                          "ema_power": 0.75, "epsilon_objective": "unweighted", "checkpoint": "final EMA"},
        "evaluation_plan": {"baseline": "outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval",
                            "fixed_dev_episodes": 30, "stride": 2, "max_control_steps": 200,
                            "episode_wall_seconds": 300, "no_scene_dropping": True},
        "outputs": {"dataset": "demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26",
                    "exp": "outputs/exp/20261027_140000_placesphere_hold8_500_r26"},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def frozen_candidate_layouts(pool_manifest):
    """Resolve recovered pool rows to their immutable original HDF5 state groups."""
    pool_path = Path(pool_manifest)
    pool = json.loads(pool_path.read_text())
    sources = pool["recovery_sources"]
    first = Path(sources[0]["path"])
    continuation = Path(sources[1]["path"])
    rows = []
    for original_id, scene in enumerate(pool["scenes"]):
        if original_id < int(sources[0]["accepted"]):
            source_h5, local_id = first / "scenes.h5", original_id
        else:
            source_h5 = continuation / "scenes.h5"
            local_id = original_id - int(sources[0]["accepted"])
        rows.append({"original_id": original_id, "seed": int(scene["seed"]),
                     "start_xy": scene["start_xy"], "container_xy": scene["container_xy"],
                     "candidate_h5": str(source_h5), "candidate_group": f"placesphere-{local_id:03d}",
                     "candidate_scene": scene})
    return sorted(rows, key=lambda row: (row["seed"], row["original_id"]))


def collect_frozen_candidate_matches(selection_path, pool_manifest, dataset_dir, experiment_dir,
                                    *, episode_wall_seconds=160, outer_wall_seconds=12000):
    """One-pass matched replay of the frozen 239 layouts; writes only admitted obstacle positives."""
    import time

    from examples.baselines.constrain import placesphere_matched_collection as collector
    from examples.baselines.constrain import placesphere_matched_terminal as matched

    selection_path, pool_manifest = Path(selection_path), Path(pool_manifest)
    dataset, experiment = Path(dataset_dir), Path(experiment_dir)
    registration = json.loads(selection_path.read_text())
    if registration["pool_manifest_sha256"] != sha256(pool_manifest):
        raise ValueError("frozen candidate pool manifest changed")
    if any(sha256(path) != digest for path, digest in registration["candidate_h5_sha256"].items()):
        raise ValueError("a frozen candidate source HDF5 changed")
    if any(sha256(ROOT / path) != digest for path, digest in registration["implementation_sources"].items()):
        raise ValueError("matched replay implementation changed after preregistration")
    dataset.mkdir(parents=True, exist_ok=False)
    replay_dir = experiment / "candidate_replays"
    replay_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    accepted = []
    outcomes = []
    layouts = frozen_candidate_layouts(pool_manifest)
    expected_order = registration["ordered_candidate_ids"]
    if [row["original_id"] for row in layouts] != expected_order:
        raise ValueError("runtime frozen candidate order differs from preregistration")
    with h5py.File(dataset / "candidate_obstacles.h5", "x") as training_h5:
        journal = (replay_dir / "attempts.jsonl").open("x")
        try:
            for attempt_index, layout in enumerate(layouts):
                if len(accepted) >= ADD_OBSTACLE or time.monotonic() - started >= outer_wall_seconds:
                    break
                attempt_started = time.monotonic()
                direct, obstacle, replay, negative, common, admission = collector._pair(
                    layout, episode_wall_seconds, 8)
                audits = {}
                if admission.get("collection_ready"):
                    audits = {"direct": collector._restore_one_step(layout, direct),
                              "obstacle": collector._restore_one_step(layout, obstacle)}
                admitted = bool(admission.get("collection_ready") and
                                all(row.get("passed") is True for row in audits.values()))
                failure = None if admitted else admission.get("reasons", [])
                if admission.get("collection_ready") and not admitted:
                    failure = ["t_plus_one_restore_mismatch"]
                attempt_h5 = replay_dir / f"candidate_{layout['original_id']:03d}.h5"
                attempt_meta = {"candidate_id": layout["original_id"], "seed": layout["seed"],
                                "admitted": admitted, "failure": failure,
                                "matched_admission": admission, "restore_audits": audits,
                                "terminal_replay": replay, "candidate_layout": layout["candidate_scene"],
                                "elapsed_seconds": time.monotonic() - attempt_started}
                with h5py.File(attempt_h5, "x") as attempt_file:
                    attempt_file.attrs["metadata_json"] = json.dumps(attempt_meta, sort_keys=True, default=str)
                    for role, record in (("direct", direct), ("obstacle", obstacle), ("negative", negative)):
                        raw = (record or {}).get("_raw")
                        if not raw:
                            continue
                        role_group = attempt_file.create_group(role)
                        for name, values in (("actions", raw["actions"]),
                                             ("policy_observations", raw["policy_observations"]),
                                             ("raw_observations", raw["raw_observations"])):
                            arr = np.asarray(values, dtype=np.float32)
                            role_group.create_dataset(name, data=arr)
                            role_group.attrs[f"{name}_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
                    attempt_file.attrs["terminal_suffix_json"] = json.dumps(
                        replay, sort_keys=True, default=str)
                if admitted:
                    candidate_id = len(accepted)
                    episode_id = 1300 + candidate_id
                    expert_record = matched._public_record(obstacle)
                    metadata = {"episode_id": episode_id, "kind": "obstacle",
                                "source_candidate_id": layout["original_id"], "source_seed": layout["seed"],
                                "layout": layout["candidate_scene"], "matched_admission": admission,
                                "t_plus_one_restore_audit": audits["obstacle"],
                                "paired_direct_restore_audit": audits["direct"],
                                "negative_transport_block_proof": negative.get("blocked_in_transport"),
                                "terminal_suffix_proof": replay,
                                "common_release_measurements": None if common is None else common.get("release_target_measurements"),
                                "step_phase": obstacle["_raw"]["trace"]["step_phase"],
                                "sphere_linear_speed_m_s": obstacle["_raw"]["trace"]["sphere_linear_speed_m_s"],
                                "sphere_angular_speed_rad_s": obstacle["_raw"]["trace"]["sphere_angular_speed_rad_s"],
                                "expert_record": expert_record,
                                "source_candidate_h5_sha256": sha256(layout["candidate_h5"])}
                    collector._write_episode(training_h5.create_group(f"traj_{candidate_id}"),
                                             obstacle, "obstacle", metadata)
                    accepted.append({"episode_id": episode_id, "candidate_id": layout["original_id"],
                                     "seed": layout["seed"], "kind": "obstacle",
                                     "layout": layout["candidate_scene"],
                                     "infos": collector._json_infos(obstacle["_raw"]["infos"])})
                    training_h5.flush()
                result = {"candidate_id": layout["original_id"], "seed": layout["seed"],
                          "admitted": admitted, "failure": failure,
                          "attempt_h5": str(attempt_h5), "attempt_h5_sha256": sha256(attempt_h5),
                          "training_episode_id": 1300 + len(accepted) - 1 if admitted else None,
                          "elapsed_seconds": attempt_meta["elapsed_seconds"]}
                outcomes.append(result)
                journal.write(json.dumps(result, sort_keys=True) + "\n")
                journal.flush()
                progress = {"target": ADD_OBSTACLE, "accepted": len(accepted),
                            "attempted": len(outcomes), "last_candidate_id": layout["original_id"],
                            "remaining_pool": len(layouts) - len(outcomes),
                            "elapsed_seconds": time.monotonic() - started}
                (replay_dir / "progress.json").write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n")
        finally:
            journal.close()
    manifest = {"format": "placesphere_r26_frozen_matched_obstacle_candidates_v1",
                "episodes": accepted, "attempts": len(outcomes), "training_count": len(accepted),
                "selection_sha256": sha256(selection_path), "candidate_pool_sha256": sha256(pool_manifest),
                "attempt_outcomes": outcomes}
    (dataset / "candidate_obstacles.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    report = {"target": ADD_OBSTACLE, "accepted": len(accepted), "attempts": len(outcomes),
              "unused_candidate_ids": [r["original_id"] for r in layouts[len(outcomes):]],
              "rejected_candidate_ids": [r["candidate_id"] for r in outcomes if not r["admitted"]],
              "complete": len(accepted) == ADD_OBSTACLE,
              "elapsed_seconds": time.monotonic() - started,
              "candidate_obstacle_h5_sha256": sha256(dataset / "candidate_obstacles.h5"),
              "candidate_manifest_sha256": sha256(dataset / "candidate_obstacles.json"),
              "admitted_obstacles": accepted}
    (dataset / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def next_unseen_seed(*journal_paths):
    """Return one past the maximum attempted seed, while measuring historical overlap."""
    per_journal = []
    all_seeds = []
    for path in journal_paths:
        seeds = []
        with Path(path).open() as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                if "seed" not in row:
                    raise ValueError(f"{path}:{line_no} has no attempted seed")
                seeds.append(int(row["seed"]))
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{path} repeats attempt seeds internally")
        per_journal.append(set(seeds))
        all_seeds.extend(seeds)
    if not all_seeds:
        raise ValueError("cannot resume from empty attempt journals")
    occurrence_counts = {}
    for seeds in per_journal:
        for seed in seeds:
            occurrence_counts[seed] = occurrence_counts.get(seed, 0) + 1
    overlap = {seed for seed, count in occurrence_counts.items() if count > 1}
    return {"next_unseen_seed": max(all_seeds) + 1, "max_attempted_seed": max(all_seeds),
            "unique_attempt_seeds": len(set(all_seeds)), "journal_rows": len(all_seeds),
            "cross_journal_seed_overlap": len(overlap),
            "overlapping_seeds": sorted(overlap)}


def preview_candidate_layout(seed, base_xy):
    """Reproduce the generator's cheap start/goal proposal without opening PhysX."""
    from examples.baselines.constrain.placesphere_pilot import (
        PlaceSphereLayoutConfig, validate_placesphere_layout,
    )

    rng = np.random.default_rng(int(seed))
    config = PlaceSphereLayoutConfig()
    for _ in range(6):
        start = np.array([rng.uniform(-0.19, -0.08), rng.uniform(-0.13, 0.13)])
        goal = np.array([rng.uniform(0.04, 0.15), rng.uniform(-0.13, 0.13)])
        valid, _, _, _ = validate_placesphere_layout(
            start, goal, np.asarray(base_xy, dtype=np.float64), config,
            container_half_extent=0.025)
        if valid:
            return {"seed": int(seed), "start_xy": start.tolist(), "container_xy": goal.tolist()}
    return None


def recover_candidate_catalog(first_catalog, interrupted_catalog, output_dir):
    """Recover a journal-consistent accepted prefix into a new catalog; never edit inputs."""
    first, partial, output = Path(first_catalog), Path(interrupted_catalog), Path(output_dir)
    first_manifest = json.loads((first / "manifest.json").read_text())
    attempt_rows = [json.loads(line) for line in (partial / "attempts.jsonl").read_text().splitlines()]
    accepted_attempts = [row for row in attempt_rows if row.get("accepted") is True]
    with h5py.File(partial / "scenes.h5", "r") as h5:
        expected_groups = {f"placesphere-{i:03d}" for i in range(len(accepted_attempts))}
        if set(h5.keys()) != expected_groups:
            raise ValueError(f"interrupted accepted prefix/HDF5 mismatch: {len(accepted_attempts)} journal accepts, {len(h5)} groups")
        recovered = []
        for episode_id, attempt in enumerate(accepted_attempts):
            group = h5[f"placesphere-{episode_id:03d}"]
            metadata = json.loads(group.attrs["metadata_json"])
            if (int(metadata.get("seed", -1)) != int(attempt["seed"])
                    or metadata.get("expert", {}).get("native_success") is not True
                    or metadata.get("expert", {}).get("safe_success") is not True
                    or metadata.get("expert", {}).get("obstacle_collision") is not False
                    or metadata.get("expert", {}).get("obstacle_crossing") is not False):
                raise ValueError(f"interrupted accepted episode {episode_id} lacks intact positive safety metadata")
            if (len(group["expert_actions"]) + 1 != len(group["raw_observations"])
                    or group["raw_observations"].shape[1] != 42
                    or group["policy_observations"].shape[1] != 39
                    or not np.isfinite(group["expert_actions"][:]).all()):
                raise ValueError(f"interrupted accepted episode {episode_id} has invalid native arrays")
            recovered.append(metadata)
    combined = list(first_manifest["scenes"]) + recovered
    task_rows = [{"episode_id": i, "seed": int(row["seed"]),
                  "start_xy": row["start_xy"], "container_xy": row["container_xy"]}
                 for i, row in enumerate(combined)]
    audit = unique_layout_audit(task_rows)
    if audit["greedy_unique"] != len(combined):
        raise ValueError("recovered accepted scenes duplicate an existing accepted task layout")
    output.mkdir(parents=True, exist_ok=False)
    result = {"format": "placesphere-r25-recovered-expansion-catalog-v1",
              "scene_count": len(combined), "scenes": combined,
              "task_identity_audit": audit,
              "recovery_sources": [
                  {"path": str(first), "h5_sha256": sha256(first / "scenes.h5"),
                   "manifest_sha256": sha256(first / "manifest.json"),
                   "attempts_sha256": sha256(first / "attempts.jsonl"),
                   "accepted": len(first_manifest["scenes"])},
                  {"path": str(partial), "h5_sha256": sha256(partial / "scenes.h5"),
                   "attempts_sha256": sha256(partial / "attempts.jsonl"),
                   "accepted_incomplete_prefix": len(recovered),
                   "attempt_rows": len(attempt_rows),
                   "first_attempt_seed": attempt_rows[0]["seed"],
                   "last_attempt_seed": attempt_rows[-1]["seed"]}],
              "incomplete_inputs_mutated": False}
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def preregister_candidate_resume(path, *, first_catalog, interrupted_catalog, recovered_catalog,
                                first_journal, interrupted_journal, timestamp):
    """The R25 continuation/canary path is superseded by the frozen R26 pool."""
    raise RuntimeError("R25 candidate resume/canary is superseded; use only the frozen R26 candidate pool")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    cursor = next_unseen_seed(first_journal, interrupted_journal)
    sources = {name: sha256(ROOT / name) for name in EXPERIMENT_SOURCES + NATIVE_SOURCES}
    return_payload = {
        "format": "placesphere_r25_resume_cursor_and_canary_preregistration_v1",
        "created_before_native_canary": True,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": sources,
        "source_identity_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "inputs": {"first_catalog": str(first_catalog), "first_h5_sha256": sha256(Path(first_catalog) / "scenes.h5"),
                   "first_manifest_sha256": sha256(Path(first_catalog) / "manifest.json"),
                   "interrupted_catalog": str(interrupted_catalog),
                   "interrupted_h5_sha256": sha256(Path(interrupted_catalog) / "scenes.h5"),
                   "recovered_catalog": str(recovered_catalog),
                   "recovered_manifest_sha256": sha256(Path(recovered_catalog) / "manifest.json"),
                   "first_journal": str(first_journal), "first_journal_sha256": sha256(first_journal),
                   "interrupted_journal": str(interrupted_journal),
                   "interrupted_journal_sha256": sha256(interrupted_journal)},
        "seed_recovery": cursor,
        "proposal_prefilter": "same NumPy default_rng(seed) start/goal proposal and six geometry validations as _attempt_body, using recorded fixed Panda base pose; task-identity duplicate and geometry exhaustion candidates are journaled before environment creation; all nonduplicate valid proposals run unchanged original expert and post-execution safety admission",
        "acceptance_criteria_unchanged": True,
        "canary": {"attempt_budget": 8, "target": 9, "wall_seconds": 300,
                   "per_layout_wall_seconds": 90, "workers": 1,
                   "output_directory": f"outputs/exp/{timestamp}_placesphere_r25_resume_canary",
                   "purpose": "verify the seed cursor is strictly beyond all attempted inputs, preflight avoids known task duplicates before PhysX, checkpoints persist each attempt, and each accepted record passes the original expert and fresh restore audit"},
        "full_continuation_after_review_only": {"remaining_after_recovery": 0,
            "attempt_seed_must_start_at": cursor["next_unseen_seed"],
            "attempt_budget_basis": "preregister a fresh finite budget from the canary/unique-seed yield; preserve current 2,941+3,197 attempt rows and all 239 accepted scene groups"},
        "outputs_are_additive": True,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(return_payload, indent=2, sort_keys=True) + "\n")
    return return_payload


def run_candidate_canary(registration_path, recovered_catalog, output_dir):
    """New candidate proposals are prohibited under the active 1,000/500 contract."""
    raise RuntimeError("new candidate layout sampling is prohibited by the frozen 1,000/500 contract")
    from examples.baselines.constrain import placesphere_transport_detour as generator

    reg = json.loads(Path(registration_path).read_text())
    if reg.get("created_before_native_canary") is not True:
        raise ValueError("native canary lacks a before-run preregistration")
    for name, expected in reg["source_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"canary source identity mismatch: {name}")
    pinned_inputs = (
        (Path(reg["inputs"]["first_catalog"]) / "scenes.h5", reg["inputs"]["first_h5_sha256"]),
        (Path(reg["inputs"]["first_catalog"]) / "manifest.json", reg["inputs"]["first_manifest_sha256"]),
        (Path(reg["inputs"]["interrupted_catalog"]) / "scenes.h5", reg["inputs"]["interrupted_h5_sha256"]),
        (Path(reg["inputs"]["first_journal"]), reg["inputs"]["first_journal_sha256"]),
        (Path(reg["inputs"]["interrupted_journal"]), reg["inputs"]["interrupted_journal_sha256"]),
        (Path(recovered_catalog) / "manifest.json", reg["inputs"]["recovered_manifest_sha256"]),
    )
    for name, expected in pinned_inputs:
        if sha256(name) != expected:
            raise ValueError(f"canary input changed after preregistration: {name}")
    if (str(recovered_catalog) != reg["inputs"]["recovered_catalog"]
            or sha256(Path(recovered_catalog) / "manifest.json") != reg["inputs"]["recovered_manifest_sha256"]):
        raise ValueError("recovered candidate catalog differs from preregistration")
    plan = reg["canary"]
    if str(output_dir) != plan["output_directory"]:
        raise ValueError("canary output directory differs from preregistration")
    known = json.loads((Path(recovered_catalog) / "manifest.json").read_text())["scenes"]
    accepted = []
    base_xy = known[0]["base_xy"]
    all_layouts = list(known)

    def preflight(seed):
        layout = preview_candidate_layout(seed, base_xy)
        if layout is None:
            return "geometry_retry_exhausted_preflight"
        if any(task_layout_conflict(layout, prior) for prior in all_layouts + accepted):
            return "task_identity_duplicate_preflight"
        return None

    def reject(metadata):
        layout = {"seed": int(metadata["seed"]), "start_xy": metadata["start_xy"],
                  "container_xy": metadata["container_xy"]}
        return "task_identity_duplicate_postcheck" if any(
            task_layout_conflict(layout, prior) for prior in all_layouts + accepted) else None

    def remember(metadata):
        accepted.append(metadata)

    result = generator.generate(
        plan["output_directory"], seed=int(reg["seed_recovery"]["next_unseen_seed"]) - 1,
        target=int(plan["target"]), max_attempts=int(plan["attempt_budget"]),
        max_wall_seconds=float(plan["wall_seconds"]),
        per_attempt_deadline=float(plan["per_layout_wall_seconds"]),
        preflight_reject=preflight, reject_metadata=reject, note_accepted=remember)
    result["resume_preregistration_sha256"] = sha256(registration_path)
    result["first_proposal_seed"] = int(reg["seed_recovery"]["next_unseen_seed"])
    result["accepted_canary_count"] = len(accepted)
    Path(plan["output_directory"], "canary_outcome.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def preregister_full_candidate_resume(path, *, recovered_catalog, first_journal,
                                     interrupted_journal, canary_journal):
    """The superseded 461-layout continuation plan is out of scope."""
    raise RuntimeError("the 461-layout continuation is superseded; no candidate plan may be resumed")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    cursor = next_unseen_seed(first_journal, interrupted_journal, canary_journal)
    first_rows = sum(1 for _ in Path(first_journal).open())
    interrupted_rows = sum(1 for _ in Path(interrupted_journal).open())
    canary = [json.loads(line) for line in Path(canary_journal).read_text().splitlines()]
    accepted = 239 + sum(bool(row.get("accepted")) for row in canary)
    if accepted > ADD_OBSTACLE:
        raise ValueError("candidate cohort exceeds the registered 700 additions")
    remaining = ADD_OBSTACLE - accepted
    new_seed_count = (interrupted_rows - cursor["cross_journal_seed_overlap"] + len(canary))
    unique_seed_yield = 16 / max(1, new_seed_count)
    estimated_attempts = int(np.ceil(remaining / unique_seed_yield))
    attempt_budget = estimated_attempts + 1200
    sources = {name: sha256(ROOT / name) for name in EXPERIMENT_SOURCES + NATIVE_SOURCES}
    payload = {
        "format": "placesphere_r25_full_candidate_resume_preregistration_v1",
        "created_after_canary_before_full_continuation": True,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": sources,
        "source_identity_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "recovered_catalog": {"path": str(recovered_catalog),
                              "manifest_sha256": sha256(Path(recovered_catalog) / "manifest.json"),
                              "accepted_task_unique_layouts": 239},
        "journals": {str(p): sha256(p) for p in (first_journal, interrupted_journal, canary_journal)},
        "cursor": {**cursor, "first_proposal_seed": cursor["next_unseen_seed"],
                   "generator_seed_argument": cursor["next_unseen_seed"] - 1,
                   "old_journal_rows": first_rows, "interrupted_journal_rows": interrupted_rows,
                   "preflight_canary_rows": len(canary), "historical_overlap_attempts": 2940,
                   "resume_seed_overlap_forbidden": True},
        "remaining_unique_layout_target": remaining,
        "yield_evidence": {"fresh_interrupted_unique_seed_attempts": new_seed_count - len(canary),
                           "fresh_interrupted_accepted_layouts": 16,
                           "canary_attempts": len(canary), "canary_accepted": sum(bool(r.get("accepted")) for r in canary),
                           "conservative_observed_unique_seed_yield": unique_seed_yield,
                           "estimated_attempts": estimated_attempts},
        "finite_budget": {"attempt_budget": attempt_budget,
                          "wall_seconds": 65000,
                          "per_layout_wall_seconds": 160,
                          "buffer_attempts": 1200,
                          "workers": 1},
        "preflight_contract": "replay the frozen per-seed uniform start/goal sampler exactly; reject only task-identity duplicates or geometry exhaustion before PhysX, then use unchanged original full physics candidate criteria; post-physics layout dedup remains as independent verification",
        "acceptance_and_distribution_unchanged": True,
        "progress_recovery": "persist attempt journal and atomic recovery_checkpoint.json each attempt; resume strictly at next_unseen_seed after validating prior journal hashes, accepted HDF5 groups and source identity",
        "source_training_gate": "candidate scenes are geometry proposals only; all 461 additional training obstacle demonstrations still require matched direct/negative/suffix proofs, shared 0.015m above-release hold-8 contract and per-episode T+1 restore before assembly",
        "full_run_output": "outputs/exp/20261027_180000_placesphere_obstacle1000_r25/candidate_source_batch2",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def preregister(path, *, timestamp="20261027_000000"):
    """The old expansion plan is superseded by the frozen 1,000/500 contract."""
    raise RuntimeError("R25 1,000-obstacle preregistration is superseded by the active frozen 1,000/500 plan")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    source_data = SOURCE_DATA / "trajectory.h5"
    source_manifest = SOURCE_DATA / "trajectory.json"
    source_validation = SOURCE_DATA / "training_identity.json"
    old_identity = json.loads(source_validation.read_text())
    if (old_identity.get("class_counts") != OLD_COUNTS
            or old_identity.get("h5_sha256") != sha256(source_data)
            or old_identity.get("manifest_sha256") != sha256(source_manifest)
            or old_identity.get("admitted_for_training") is not True):
        raise ValueError("current hold-8 source is not the qualified 1k/300 training package")
    artifacts = [source_data, source_manifest, source_validation,
                 TRAIN_BASE, TRAIN_CONTROL, SCENES / "manifest.json", SCENES / "scenes.h5"]
    external_catalogs = [Path("outputs/exp/20261025_085101_placesphere_obstacle300_90k/obstacle_scenes/catalog_full"),
                         Path("outputs/exp/20261024_155244_placesphere_data_expansion/pair_scenes/merged")]
    artifacts.extend(catalog / "manifest.json" for catalog in external_catalogs)
    hashes = {str(p): sha256(p) for p in artifacts}
    sources = {name: sha256(ROOT / name) for name in EXPERIMENT_SOURCES + NATIVE_SOURCES
               if (ROOT / name).is_file()}
    if len(sources) != len(EXPERIMENT_SOURCES) + len(NATIVE_SOURCES):
        raise ValueError("a preregistered implementation or native source is missing")
    if sha256(TRAIN_BASE) != "92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c":
        raise ValueError("matched Base EMA identity mismatch")
    if sha256(TRAIN_CONTROL) != "c8d1404f68cf153176035c8d7d95d032517838f9df877b295bbefd5102f57ada":
        raise ValueError("prior Constrain EMA identity mismatch")
    scene_manifest_sha = sha256(SCENES / "manifest.json")
    if scene_manifest_sha != "9e1d8ee4786d2336f01e07d664ca37c3dcc7ca455e7876176def65c18b63c119":
        raise ValueError("frozen 30-scene identity mismatch")
    payload = {
        "format": "placesphere_obstacle1000_expansion_preregistration_v1",
        "round": 25,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "created_before_collection": True,
        "training": False,
        "source_artifacts": hashes,
        "experiment_source_sha256": sources,
        "source_identity_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "runtime": {"python": platform.python_version(),
                    "packages": {p: importlib.metadata.version(p) for p in
                                 ("numpy", "torch", "gymnasium", "mani-skill", "mplib", "h5py")}},
        "dataset_plan": {
            "source_dataset": str(SOURCE_DATA), "source_class_counts": OLD_COUNTS,
            "source_h5_sha256": hashes[str(source_data)],
            "source_manifest_sha256": hashes[str(source_manifest)],
            "preserve_all_original_arrays": True,
            "add": {"kind": "obstacle", "count": ADD_OBSTACLE,
                    "ids_in_final_package": [1300, 1499],
                    "new_direct_positive_and_negative_block_and_terminal_suffix_records_are_probes_only": True},
            "final_training_class_counts": TARGET_COUNTS,
            "generator": "existing frozen PlaceSphere transport-detour geometry/expert candidate generator; same 0.03m radius, measured per-scene height, same-side waypoint, above-release 0.015m shared-terminal expert contract",
            "distribution": "broad existing uniform start/container and midsegment blocking-cylinder proposal; no scene-25 target selection, no policy metric filter, no threshold relaxation",
            "task_identity": {"rule": "same seed, or start and container each within 0.02 m componentwise",
                              "tolerance_m": TASK_IDENTITY_TOLERANCE_M,
                              "excluded": ["all 300 source obstacle task layouts", "frozen dev30", "Ticket04 training layouts", "each other"]},
            "attempt_budget": 10000, "candidate_wall_seconds": 21600,
            "candidate_seed": 2026102711,
            "per_layout_wall_seconds": 160, "workers": 1,
            "candidate_rejections": "append every geometry, expert, task-identity, direct-block, negative-block, suffix, restore, timeout, exception, and cleanup outcome to durable JSONL; never discard or silently continue an accepted episode",
            "expert_admission": "positive obstacle is native first-success, T actions/T+1 39/42 state, native 7D action, full-prefix contact/crossing free, 0.015m common above-release terminal; matched direct positive, negative direct transport block and terminal suffix proof accompany each layout; only obstacle positive enters training",
        },
        "training_plan": {
            "mode": "scratch from qualified matched Base final EMA; do not resume old 90k cosine/optimizer",
            "base_checkpoint": str(TRAIN_BASE), "base_checkpoint_sha256": hashes[str(TRAIN_BASE)],
            "constrain_initialization": "frozen Base EMA 92ac831b...; only control copy/hint/control residual trainable; no new direct Base training",
            "dataset_profile": MATCHED_HOLD8_OBSTACLE1000_VERSION,
            "class_counts": TARGET_COUNTS, "selected_class": "obstacle", "uniform_official_window_sampling": True,
            "total_updates": 300000, "batch_size": 1024, "learning_rate": 0.0001,
            "warmup": 500, "optimizer": "AdamW betas=(0.95,0.999), weight_decay=1e-6",
            "scheduler": "cosine with exact 300000 update horizon", "ema_power": 0.75, "seed": 1,
            "objective": "unweighted epsilon MSE; no phase/proximity/release weights",
            "selected_checkpoint": "final EMA fixed before training; no checkpoint/seed selection",
            "wall_budget_seconds": 15000,
            "exposure_note": "approximately 10x the old 300-obstacle cohort at equal updates; updates also rise 90k→300k, so data-volume and total-exposure changes are combined, not isolated",
        },
        "evaluation_plan": {
            "scene_directory": str(SCENES), "scene_manifest_sha256": scene_manifest_sha,
            "primary_baseline": str(PRIMARY), "primary_baseline_stride": 2,
            "secondary_baseline": "Round-22 matched hold-0 stride-8, explicitly confounded comparison",
            "stride": 2, "prediction_horizon": 16, "executed_slice": [1, 9],
            "episodes_per_policy": 30, "fixed_denominator": 30,
            "max_control_steps": 200, "episode_wall_seconds": 300, "overall_wall_seconds": 18000,
            "same_saved_scene_state_controller_and_seed": True,
            "native_first_success_termination": True, "runtime_errors_stay_in_denominator": True,
            "no_checkpoint_or_scene_selection": True,
            "gate": {"task_min": 27, "safe_min": 26, "contact_max": 3, "xy_crossing_max": 3},
        },
        "outputs": {"dataset": f"demos/PlaceSphere-transport-detour/{timestamp}_obstacle1000_hold8_r25",
                    "exp": f"outputs/exp/{timestamp}_placesphere_obstacle1000_r25",
                    "ckpt": f"outputs/ckpt/{timestamp}_placesphere_obstacle1000_constrain_300k_r25"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_obstacle_candidate(path, manifest_path, *, expected_count=ADD_OBSTACLE):
    """Validate all candidate IDs/labels, positive safety, native endpoints and uniqueness."""
    manifest = json.loads(Path(manifest_path).read_text())
    records = manifest.get("episodes")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"candidate manifest must contain exactly {expected_count} episodes")
    if [int(row.get("episode_id", -1)) for row in records] != list(range(expected_count)):
        raise ValueError("candidate episode IDs must be ordered and contiguous from zero")
    rows = []
    with h5py.File(path, "r") as h5:
        if set(h5.keys()) != {f"traj_{i}" for i in range(expected_count)}:
            raise ValueError("candidate HDF5 groups do not match the manifest IDs")
        for row in records:
            eid = int(row["episode_id"])
            group = h5[f"traj_{eid}"]
            if row.get("kind") != "obstacle" or group.attrs.get("kind") != "obstacle":
                raise ValueError(f"candidate {eid} is not a positive obstacle training episode")
            actions, obs, raw = group["actions"][:], group["obs"][:], group["raw_obs"][:]
            if (actions.ndim != 2 or actions.shape[1] != 7 or obs.shape != (len(actions) + 1, 39)
                    or raw.shape != (len(actions) + 1, 42) or not np.isfinite(actions).all()
                    or not np.isfinite(obs).all() or not np.isfinite(raw).all()):
                raise ValueError(f"candidate {eid} violates finite T/T+1 native 7/39/42 contract")
            metadata = json.loads(group.attrs.get("metadata_json", "{}"))
            expert = metadata.get("expert_record", {})
            if (not expert.get("native_success") or not expert.get("terminated")
                    or expert.get("truncated") or not expert.get("from_execution_start_safe")
                    or expert.get("obstacle_contact") or expert.get("obstacle_crossing")
                    or not expert.get("on_bin") or not expert.get("static") or not expert.get("ungrasped")):
                raise ValueError(f"candidate {eid} lacks first-native-success full-prefix safety proof")
            if metadata.get("t_plus_one_restore_audit", {}).get("passed") is not True:
                raise ValueError(f"candidate {eid} lacks T+1 restore audit")
            layout = metadata.get("layout")
            if not isinstance(layout, dict):
                layout = row.get("layout")
            if not isinstance(layout, dict):
                raise ValueError(f"candidate {eid} lacks task geometry for deduplication")
            rows.append({"episode_id": eid, "seed": int(row["seed"]),
                         "start_xy": layout["start_xy"], "container_xy": layout["container_xy"],
                         "layout": layout, "length": len(actions)})
    audit = unique_layout_audit(rows)
    if audit["greedy_unique"] != expected_count:
        raise ValueError(f"candidate obstacle task layouts are not unique: {audit}")
    return {"count": expected_count, "task_identity": audit,
            "lengths": {"median": float(np.median([r["length"] for r in rows])),
                        "p90": float(np.percentile([r["length"] for r in rows], 90)),
                        "p95": float(np.percentile([r["length"] for r in rows], 95)),
                        "max": max(r["length"] for r in rows),
                        "above_200": sum(r["length"] > 200 for r in rows)},
            "candidate_h5_sha256": sha256(path),
            "candidate_manifest_sha256": sha256(manifest_path), "rows": rows}


def generate_candidate_catalog(registration_path, output_dir):
    """Candidate generation is frozen off under the active 1,000/500 contract."""
    raise RuntimeError("new candidate layout sampling is prohibited by the frozen 1,000/500 contract")
    from examples.baselines.constrain import placesphere_transport_detour as generator

    registration = json.loads(Path(registration_path).read_text())
    if registration.get("created_before_collection") is not True:
        raise ValueError("candidate generation requires the pre-collection registration")
    for name, expected in registration["experiment_source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"candidate generator source changed since registration: {name}")
    for name, expected in registration["source_artifacts"].items():
        if sha256(name) != expected:
            raise ValueError(f"candidate generator source artifact changed since registration: {name}")
    forbidden = []
    with h5py.File(SOURCE_DATA / "trajectory.h5", "r") as h5:
        for eid in range(1300):
            if h5[f"traj_{eid}"].attrs["kind"] != "obstacle":
                continue
            metadata = json.loads(h5[f"traj_{eid}"].attrs["metadata_json"])
            layout = metadata["layout"]
            forbidden.append({"seed": metadata["expert_record"]["seed"],
                              "start_xy": layout["start_xy"],
                              "container_xy": layout["container_xy"]})
    for catalog in (Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes"),
                    Path("outputs/exp/20261025_085101_placesphere_obstacle300_90k/obstacle_scenes/catalog_full"),
                    Path("outputs/exp/20261024_155244_placesphere_data_expansion/pair_scenes/merged")):
        manifest = json.loads((catalog / "manifest.json").read_text())
        forbidden.extend(manifest["scenes"])
    accepted = []

    def reject(metadata):
        row = {"seed": int(metadata["seed"]), "start_xy": metadata["start_xy"],
               "container_xy": metadata["container_xy"]}
        if any(task_layout_conflict(row, other) for other in forbidden + accepted):
            return "task_identity_duplicate"
        return None

    def remember(metadata):
        accepted.append({"seed": int(metadata["seed"]), "start_xy": metadata["start_xy"],
                         "container_xy": metadata["container_xy"]})

    result = generator.generate(
        output_dir, seed=int(registration["dataset_plan"]["candidate_seed"]),
        target=ADD_OBSTACLE, max_attempts=int(registration["dataset_plan"]["attempt_budget"]),
        max_wall_seconds=float(registration["dataset_plan"]["candidate_wall_seconds"]),
        per_attempt_deadline=float(registration["dataset_plan"]["per_layout_wall_seconds"]),
        reject_metadata=reject, note_accepted=remember)
    if not result["complete"]:
        raise RuntimeError(f"candidate catalog incomplete; journals retained: {result}")
    return result


def continue_candidate_catalog(registration_path, previous_catalog, output_dir):
    """Candidate continuation is prohibited by the active frozen 1,000/500 contract."""
    raise RuntimeError("candidate layout continuation is prohibited; only existing R26 pool layouts may be replayed")
    from examples.baselines.constrain import placesphere_transport_detour as generator

    prereg = json.loads(Path(registration_path).read_text())
    plan = prereg["continuation"]
    previous = Path(previous_catalog)
    old_manifest_path, old_h5 = previous / "manifest.json", previous / "scenes.h5"
    if sha256(old_manifest_path) != plan["previous_manifest_sha256"] or sha256(old_h5) != plan["previous_h5_sha256"]:
        raise ValueError("incomplete candidate source changed after continuation was preregistered")
    old_manifest = json.loads(old_manifest_path.read_text())
    prior_accepted = [{"seed": int(r["seed"]), "start_xy": r["start_xy"],
                       "container_xy": r["container_xy"]} for r in old_manifest["scenes"]]
    if len(prior_accepted) != plan["already_accepted"]:
        raise ValueError("prior accepted candidate count differs from extension preregistration")
    for name, expected in prereg["source_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"continuation source identity changed: {name}")
    forbidden = []
    with h5py.File(SOURCE_DATA / "trajectory.h5", "r") as h5:
        for eid in range(1300):
            if h5[f"traj_{eid}"].attrs["kind"] == "obstacle":
                meta = json.loads(h5[f"traj_{eid}"].attrs["metadata_json"])
                layout = meta["layout"]
                forbidden.append({"seed": meta["expert_record"]["seed"],
                                  "start_xy": layout["start_xy"],
                                  "container_xy": layout["container_xy"]})
    for catalog in (Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes"),
                    Path("outputs/exp/20261025_085101_placesphere_obstacle300_90k/obstacle_scenes/catalog_full"),
                    Path("outputs/exp/20261024_155244_placesphere_data_expansion/pair_scenes/merged")):
        forbidden.extend(json.loads((catalog / "manifest.json").read_text())["scenes"])
    accepted = []

    def reject(metadata):
        row = {"seed": int(metadata["seed"]), "start_xy": metadata["start_xy"],
               "container_xy": metadata["container_xy"]}
        return "task_identity_duplicate" if any(
            task_layout_conflict(row, other) for other in forbidden + prior_accepted + accepted) else None

    def remember(metadata):
        accepted.append({"seed": int(metadata["seed"]), "start_xy": metadata["start_xy"],
                         "container_xy": metadata["container_xy"]})

    result = generator.generate(
        output_dir, seed=int(plan["candidate_seed"]), target=int(plan["remaining"]),
        max_attempts=int(plan["attempt_budget"]),
        max_wall_seconds=float(plan["candidate_wall_seconds"]),
        per_attempt_deadline=float(plan["per_layout_wall_seconds"]),
        reject_metadata=reject, note_accepted=remember)
    result["continuation_registration_sha256"] = sha256(registration_path)
    result["previous_candidate_manifest_sha256"] = sha256(old_manifest_path)
    result["previous_candidate_h5_sha256"] = sha256(old_h5)
    (Path(output_dir) / "continuation_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["complete"]:
        raise RuntimeError(f"candidate continuation incomplete; journals retained: {result}")
    return result


def merge_candidate_catalogs(first, second, output_dir, *, expected_count=ADD_OBSTACLE):
    """Merge immutable candidate records by appending scenes, never renumbering evidence."""
    first, second, output = Path(first), Path(second), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    left = json.loads((first / "manifest.json").read_text())["scenes"]
    right = json.loads((second / "manifest.json").read_text())["scenes"]
    scenes = [dict(row) for row in left + right]
    if len(scenes) != expected_count:
        raise ValueError(f"candidate catalog merge has {len(scenes)}, expected {expected_count}")
    identities = [{"episode_id": i, "seed": int(row["seed"]),
                   "start_xy": row["start_xy"], "container_xy": row["container_xy"]}
                  for i, row in enumerate(scenes)]
    if unique_layout_audit(identities)["greedy_unique"] != expected_count:
        raise ValueError("merged candidate pool has duplicate task layouts")
    def catalog_record(path):
        path = Path(path)
        scenes_h5 = path / "scenes.h5"
        return {"path": str(path), "manifest_sha256": sha256(path / "manifest.json"),
                "scenes_h5_sha256": sha256(scenes_h5) if scenes_h5.is_file() else None}

    manifest = {"format": "placesphere-r25-task-unique-expansion-catalog-v1",
                "scene_count": len(scenes), "scenes": scenes,
                "source_catalogs": [catalog_record(first), catalog_record(second)],
                "task_identity_audit": unique_layout_audit(identities)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"scene_count": len(scenes), "manifest_sha256": sha256(output / "manifest.json"),
            "task_identity_audit": manifest["task_identity_audit"]}


def run_registered_candidate_batch(registration_path, recovered_catalog):
    """The superseded 461-layout continuation is intentionally disabled."""
    raise RuntimeError("the old 461-layout continuation is superseded; no candidate sampling is authorized")
    from examples.baselines.constrain import placesphere_transport_detour as generator

    reg = json.loads(Path(registration_path).read_text())
    if reg.get("created_after_canary_before_full_continuation") is not True:
        raise ValueError("full candidate batch lacks its frozen pre-run registration")
    for name, expected in reg["source_hashes"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"full continuation source identity mismatch: {name}")
    for name, expected in reg["journals"].items():
        if sha256(name) != expected:
            raise ValueError(f"a prior journal changed after full continuation registration: {name}")
    if (sha256(Path(recovered_catalog) / "manifest.json")
            != reg["recovered_catalog"]["manifest_sha256"]):
        raise ValueError("recovered candidate prefix changed after registration")
    known = json.loads((Path(recovered_catalog) / "manifest.json").read_text())["scenes"]
    base_xy = known[0]["base_xy"]
    external = []
    with h5py.File(SOURCE_DATA / "trajectory.h5", "r") as h5:
        for eid in range(1300):
            if h5[f"traj_{eid}"].attrs["kind"] == "obstacle":
                meta = json.loads(h5[f"traj_{eid}"].attrs["metadata_json"])
                layout = meta["layout"]
                external.append({"seed": int(meta["expert_record"]["seed"]),
                                 "start_xy": layout["start_xy"],
                                 "container_xy": layout["container_xy"]})
    for catalog in (Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes"),
                    Path("outputs/exp/20261025_085101_placesphere_obstacle300_90k/obstacle_scenes/catalog_full"),
                    Path("outputs/exp/20261024_155244_placesphere_data_expansion/pair_scenes/merged")):
        external.extend(json.loads((catalog / "manifest.json").read_text())["scenes"])
    accepted = []

    def preflight(seed):
        proposal = preview_candidate_layout(seed, base_xy)
        if proposal is None:
            return "geometry_retry_exhausted_preflight"
        if any(task_layout_conflict(proposal, row) for row in external + known + accepted):
            return "task_identity_duplicate_preflight"
        return None

    def reject(metadata):
        proposal = {"seed": int(metadata["seed"]), "start_xy": metadata["start_xy"],
                    "container_xy": metadata["container_xy"]}
        if any(task_layout_conflict(proposal, row) for row in external + known + accepted):
            return "task_identity_duplicate_postcheck"
        return None

    def remember(metadata):
        accepted.append(metadata)

    plan, budget = reg["cursor"], reg["finite_budget"]
    output = reg["full_run_output"]
    result = generator.generate(
        output, seed=int(plan["generator_seed_argument"]),
        target=int(reg["remaining_unique_layout_target"]),
        max_attempts=int(budget["attempt_budget"]),
        max_wall_seconds=float(budget["wall_seconds"]),
        per_attempt_deadline=float(budget["per_layout_wall_seconds"]),
        preflight_reject=preflight, reject_metadata=reject, note_accepted=remember)
    result["resume_preregistration_sha256"] = sha256(registration_path)
    result["first_proposal_seed"] = plan["first_proposal_seed"]
    result["remaining_unique_layout_target"] = reg["remaining_unique_layout_target"]
    Path(output, "resume_outcome.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["complete"]:
        raise RuntimeError(f"finite registered candidate batch incomplete; all progress retained: {result}")
    return result


def preregister_continuation(path, *, initial_registration, previous_catalog,
                             candidate_seed=2026102712):
    """Freeze a bounded extension only after measuring positive candidate yield."""
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    base = json.loads(Path(initial_registration).read_text())
    report = json.loads((Path(previous_catalog) / "report.json").read_text())
    if report.get("complete") or report.get("accepted", 0) <= 0:
        raise ValueError("continuation requires incomplete work with demonstrated ongoing accepted yield")
    remaining = ADD_OBSTACLE - int(report["accepted"])
    rate = float(report["accepted"]) / float(report["attempts"])
    mean_attempt = float(report["elapsed_seconds"]) / float(report["attempts"])
    estimated_attempts = int(np.ceil(remaining / rate))
    sources = {name: sha256(ROOT / name) for name in EXPERIMENT_SOURCES + NATIVE_SOURCES}
    payload = {
        "format": "placesphere_r25_candidate_budget_extension_preregistration_v1",
        "created_before_continuation": True,
        "initial_registration_sha256": sha256(initial_registration),
        "initial_source_identity_sha256": base["source_identity_sha256"],
        "source_hashes": sources,
        "source_identity_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "prior_attempt_ledger": {
            "report_path": str(Path(previous_catalog) / "report.json"),
            "attempts_path": str(Path(previous_catalog) / "attempts.jsonl"),
            "h5_sha256": sha256(Path(previous_catalog) / "scenes.h5"),
            "manifest_sha256": sha256(Path(previous_catalog) / "manifest.json"),
            "attempts": report["attempts"], "accepted": report["accepted"],
            "rejections": report["failures"], "ongoing_acceptance_rate": rate,
            "mean_wall_seconds_per_attempt": mean_attempt,
        },
        "continuation": {
            "remaining": remaining,
            "already_accepted": int(report["accepted"]),
            "expected_attempts_at_observed_yield": estimated_attempts,
            "attempt_budget": max(estimated_attempts + 1200, 7500),
            "candidate_wall_seconds": 55000,
            "per_layout_wall_seconds": 160,
            "candidate_seed": int(candidate_seed),
            "previous_h5_sha256": sha256(Path(previous_catalog) / "scenes.h5"),
            "previous_manifest_sha256": sha256(Path(previous_catalog) / "manifest.json"),
            "no_acceptance_rule_or_geometry_relaxation": True,
            "retained_previous_accepted_and_rejected_records": True,
        },
        "budget_basis": {
            "estimated_seconds_at_observed_mean": estimated_attempts * mean_attempt,
            "finite_extension_wall_seconds": 55000,
            "attempt_budget_buffer": "1,200 additional candidate attempts above the point estimate, capped by the 55,000 s wall budget",
            "failure_action": "if the extension stops incomplete, preserve all artifacts and recompute a new finite preregistration from the observed continuation yield before any further attempt",
        },
        "created_before_continuation": True,
        "output_directory": "outputs/exp/20261027_103000_placesphere_obstacle1000_r25/candidate_source_continuation",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def extract_obstacle_candidates(collection_h5, collection_manifest, output_h5, output_manifest):
    """Extract only positive OBSTACLE demonstrations; direct episodes remain probes."""
    records = json.loads(Path(collection_manifest).read_text())["episodes"]
    obstacles = [row for row in records if row.get("kind") == "obstacle"]
    if len(obstacles) != ADD_OBSTACLE:
        raise ValueError(f"expected {ADD_OBSTACLE} obstacle positives, got {len(obstacles)}")
    target_manifest = []
    with h5py.File(collection_h5, "r") as src, h5py.File(output_h5, "x") as dst:
        for new_id, row in enumerate(obstacles):
            source_id = int(row["episode_id"])
            src.copy(f"traj_{source_id}", dst, name=f"traj_{new_id}")
            copied = dict(row)
            copied["source_collection_episode_id"] = source_id
            copied["episode_id"] = new_id
            target_manifest.append(copied)
    Path(output_manifest).write_text(json.dumps({"episodes": target_manifest}, indent=2, sort_keys=True) + "\n")


def final_dataset_audit(h5_path, manifest_path, *, source_h5, candidate_h5):
    """Before training: verify package IDs, source equality, windows, and per-action speed coverage."""
    manifest = json.loads(Path(manifest_path).read_text())
    rows = manifest.get("episodes", [])
    total = sum(TARGET_COUNTS.values())
    if len(rows) != total or {row["kind"] for row in rows} != {"direct", "obstacle"}:
        raise ValueError("final package must contain exactly 1,000 direct + 500 obstacle episodes")
    by_id = {int(row["episode_id"]): row for row in rows}
    if len(by_id) != total or set(by_id) != set(range(total)):
        raise ValueError(f"final package must cover unique episode IDs 0..{total - 1}")
    counts = {kind: sum(row["kind"] == kind for row in rows) for kind in TARGET_COUNTS}
    if counts != TARGET_COUNTS:
        raise ValueError(f"wrong final class counts {counts}")
    lengths, windows = {kind: [] for kind in counts}, {kind: 0 for kind in counts}
    action_counts = {kind: 0 for kind in counts}
    actions_over_200 = {kind: 0 for kind in counts}
    open_near_zero = {kind: 0 for kind in counts}
    win_stats = {kind: {"all": 0, "action_only": 0, "linear_static": 0,
                        "angular_static": 0, "both_native_static": 0} for kind in counts}
    with h5py.File(source_h5, "r") as old, h5py.File(candidate_h5, "r") as new, h5py.File(h5_path, "r") as final:
        if set(final.keys()) != {f"traj_{i}" for i in range(total)}:
            raise ValueError("final HDF5 group IDs differ from the exact package range")
        # Exact payload equality for the retained source IDs and appended obstacle IDs.
        for eid in range(1300):
            left, right = old[f"traj_{eid}"], final[f"traj_{eid}"]
            if _h5_group_digest(left) != _h5_group_digest(right):
                raise ValueError(f"retained source HDF5 group bytes/attrs changed for episode {eid}")
        for offset in range(ADD_OBSTACLE):
            left, right = new[f"traj_{offset}"], final[f"traj_{1300 + offset}"]
            if _h5_group_digest(left) != _h5_group_digest(right):
                raise ValueError(f"new candidate HDF5 group bytes/attrs changed during append for {offset}")
        for eid, row in by_id.items():
            group, kind = final[f"traj_{eid}"], row["kind"]
            if group.attrs.get("kind") != kind:
                raise ValueError(f"manifest/HDF5 label mismatch at {eid}")
            actions = np.asarray(group["actions"])
            metadata = json.loads(group.attrs["metadata_json"])
            phases = metadata["step_phase"]
            linear = np.asarray(metadata["sphere_linear_speed_m_s"], dtype=float)
            angular = np.asarray(metadata["sphere_angular_speed_rad_s"], dtype=float)
            if len(phases) != len(actions) or len(linear) != len(actions) or len(angular) != len(actions):
                raise ValueError(f"episode {eid} is missing a same-index phase/speed record")
            if not np.isfinite(actions).all() or not np.isfinite(linear).all() or not np.isfinite(angular).all():
                raise ValueError(f"episode {eid} has nonfinite action or kinematic data")
            length = len(actions)
            lengths[kind].append(length)
            windows[kind] += len(range(-1, length - 2))
            action_counts[kind] += length
            actions_over_200[kind] += max(0, length - 200)
            open_near_zero[kind] += int(np.sum((actions[:, 6] > 0) & (np.linalg.norm(actions[:, :6], axis=1) <= .02)))
            for start in range(0, length - 7, 8):
                chunk = actions[start:start + 8]
                action_only = bool(np.all(chunk[:, 6] > 0) and np.max(np.linalg.norm(chunk[:, :6], axis=1)) <= .02)
                stat = win_stats[kind]
                stat["all"] += 1
                if action_only:
                    stat["action_only"] += 1
                    lin_ok = float(np.max(linear[start:start + 8])) <= .01
                    ang_ok = float(np.max(angular[start:start + 8])) <= .5
                    stat["linear_static"] += int(lin_ok)
                    stat["angular_static"] += int(ang_ok)
                    stat["both_native_static"] += int(lin_ok and ang_ok)
    summary = {"complete": True, "class_counts": counts,
               "h5_sha256": sha256(h5_path), "manifest_sha256": sha256(manifest_path),
               "training_windows": windows, "episode_lengths": {},
               "action_volume": {}, "stride8_action_kinematic_windows": win_stats,
               "retained_old_groups_byte_values_preserved": True,
               "direct_layouts": unique_layout_audit([row for row in rows if row["kind"] == "direct"]),
               "obstacle_layouts": unique_layout_audit([row for row in rows if row["kind"] == "obstacle"])}
    for kind in counts:
        summary["episode_lengths"][kind] = {
            "count": len(lengths[kind]), "median": float(np.median(lengths[kind])),
            "p90": float(np.percentile(lengths[kind], 90)), "p95": float(np.percentile(lengths[kind], 95)),
            "max": max(lengths[kind]), "above_200": sum(x > 200 for x in lengths[kind]),
        }
        summary["action_volume"][kind] = {
            "actions": action_counts[kind], "actions_beyond_200_fraction": actions_over_200[kind] / action_counts[kind],
            "open_near_zero_actions": open_near_zero[kind],
            "open_near_zero_fraction": open_near_zero[kind] / action_counts[kind],
        }
    return summary


def append_obstacles(candidate_h5, candidate_manifest, output_dir, *, registration_path,
                     expected_old_h5, expected_old_manifest):
    """Build an immutable new HDF5 version and a class-ID manifest after auditing.

    `registration_path` must have been written before any new collection begins
    and pins both original source files plus the generator/runtime identities.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    old_data, old_manifest = SOURCE_DATA / "trajectory.h5", SOURCE_DATA / "trajectory.json"
    if sha256(old_data) != expected_old_h5 or sha256(old_manifest) != expected_old_manifest:
        raise ValueError("trusted hold-8 source package identity mismatch")
    registration = json.loads(Path(registration_path).read_text())
    if registration.get("created_before_collection") is not True or registration.get("training") is not False:
        raise ValueError("collection registration is not a pre-collection freeze")
    for path, digest in registration["source_artifacts"].items():
        if sha256(path) != digest:
            raise ValueError(f"preregistered source artifact changed: {path}")
    candidate = verify_obstacle_candidate(candidate_h5, candidate_manifest)
    base_manifest = json.loads(old_manifest.read_text())
    if len(base_manifest["episodes"]) != sum(OLD_COUNTS.values()):
        raise ValueError("trusted source manifest episode count changed")
    target_h5, target_manifest = out / "trajectory.h5", out / "trajectory.json"
    with h5py.File(old_data, "r") as source, h5py.File(candidate_h5, "r") as addition, h5py.File(target_h5, "x") as dest:
        if set(source.keys()) != {f"traj_{i}" for i in range(1300)}:
            raise ValueError("trusted source group IDs are not exactly 0..1299")
        for eid in range(1300):
            source.copy(f"traj_{eid}", dest, name=f"traj_{eid}")
        for offset in range(ADD_OBSTACLE):
            addition.copy(f"traj_{offset}", dest, name=f"traj_{1300 + offset}")
    episodes = list(base_manifest["episodes"])
    candidate_manifest_json = json.loads(Path(candidate_manifest).read_text())
    for offset, record in enumerate(candidate_manifest_json["episodes"]):
        item = dict(record)
        item["episode_id"] = 1300 + offset
        item["kind"] = "obstacle"
        item["expansion_source_episode_id"] = offset
        episodes.append(item)
    manifest = {"format": "placesphere_matched_terminal_hold8_r25_v1", "episodes": episodes,
                "source_manifest_sha256": sha256(old_manifest),
                "candidate_manifest_sha256": sha256(candidate_manifest),
                "registration_sha256": sha256(registration_path)}
    target_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    class_counts = {kind: sum(row["kind"] == kind for row in episodes) for kind in TARGET_COUNTS}
    if class_counts != TARGET_COUNTS:
        raise ValueError(f"final class count mismatch: {class_counts}")
    return {"dataset_version": MATCHED_HOLD8_OBSTACLE500_VERSION,
            "h5_sha256": sha256(target_h5), "manifest_sha256": sha256(target_manifest),
            "class_counts": class_counts, "old_h5_sha256": sha256(old_data),
            "candidate_audit": {k: v for k, v in candidate.items() if k != "rows"},
            "old_arrays_preserved_at_dataset_value_level": True,
            "training_direct_ids": [int(r["episode_id"]) for r in episodes if r["kind"] == "direct"],
            "training_obstacle_ids": [int(r["episode_id"]) for r in episodes if r["kind"] == "obstacle"],
            "validation_complete": False, "admitted_for_training": False}


def assemble_and_validate(candidate_h5, candidate_manifest, output_dir, *, registration_path,
                          expected_old_h5, expected_old_manifest):
    identity = append_obstacles(candidate_h5, candidate_manifest, output_dir,
                                 registration_path=registration_path,
                                 expected_old_h5=expected_old_h5,
                                 expected_old_manifest=expected_old_manifest)
    target = Path(output_dir)
    validation = final_dataset_audit(
        target / "trajectory.h5", target / "trajectory.json",
        source_h5=SOURCE_DATA / "trajectory.h5", candidate_h5=candidate_h5)
    validation.update({"registration_sha256": sha256(registration_path),
                       "candidate_source_h5_sha256": sha256(candidate_h5),
                       "candidate_source_manifest_sha256": sha256(candidate_manifest),
                       "external_task_identity_filter": "pre-collection candidate generator rejects source 300, dev30, Ticket04 catalogs and within-cohort repeats using the frozen seed-or-two-endpoints rule"})
    validation_path = target / "dataset_validation.json"
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    identity.update({"release_hold_steps": 8,
                     "episode_ids": {kind: [int(row["episode_id"]) for row in json.loads((target / "trajectory.json").read_text())["episodes"] if row["kind"] == kind]
                                     for kind in TARGET_COUNTS},
                     "registration_sha256": sha256(registration_path),
                     "dataset_validation_sha256": sha256(validation_path),
                     "validation_complete": True, "admitted_for_training": True,
                     "native_thresholds_changed": False, "geometry_changed": False})
    identity_path = target / "training_identity.json"
    identity["format"] = "placesphere_matched_training_identity_v1"
    identity_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    return identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-candidate")
    audit.add_argument("--h5", type=Path, required=True)
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    pre = sub.add_parser("preregister")
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--timestamp", default="20261027_000000")
    continuation = sub.add_parser("preregister-continuation")
    continuation.add_argument("--output", type=Path, required=True)
    continuation.add_argument("--registration", type=Path, required=True)
    continuation.add_argument("--previous-catalog", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--candidate-h5", type=Path, required=True)
    assemble.add_argument("--candidate-manifest", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--registration", type=Path, required=True)
    assemble.add_argument("--old-h5-sha256", required=True)
    assemble.add_argument("--old-manifest-sha256", required=True)
    recover = sub.add_parser("recover-candidates")
    recover.add_argument("--first-catalog", type=Path, required=True)
    recover.add_argument("--interrupted-catalog", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    resume_prereg = sub.add_parser("preregister-resume")
    resume_prereg.add_argument("--first-catalog", type=Path, required=True)
    resume_prereg.add_argument("--interrupted-catalog", type=Path, required=True)
    resume_prereg.add_argument("--recovered-catalog", type=Path, required=True)
    resume_prereg.add_argument("--first-journal", type=Path, required=True)
    resume_prereg.add_argument("--interrupted-journal", type=Path, required=True)
    resume_prereg.add_argument("--output", type=Path, required=True)
    resume_prereg.add_argument("--timestamp", required=True)
    canary = sub.add_parser("candidate-canary")
    canary.add_argument("--registration", type=Path, required=True)
    canary.add_argument("--recovered-catalog", type=Path, required=True)
    canary.add_argument("--output", type=Path, required=True)
    full_prereg = sub.add_parser("preregister-full-resume")
    full_prereg.add_argument("--recovered-catalog", type=Path, required=True)
    full_prereg.add_argument("--first-journal", type=Path, required=True)
    full_prereg.add_argument("--interrupted-journal", type=Path, required=True)
    full_prereg.add_argument("--canary-journal", type=Path, required=True)
    full_prereg.add_argument("--output", type=Path, required=True)
    full_run = sub.add_parser("run-registered-candidate-batch")
    full_run.add_argument("--registration", type=Path, required=True)
    full_run.add_argument("--recovered-catalog", type=Path, required=True)
    frozen_replay = sub.add_parser("collect-frozen-matches")
    frozen_replay.add_argument("--selection", type=Path, required=True)
    frozen_replay.add_argument("--pool-manifest", type=Path, required=True)
    frozen_replay.add_argument("--dataset-dir", type=Path, required=True)
    frozen_replay.add_argument("--experiment-dir", type=Path, required=True)
    frozen_replay.add_argument("--episode-wall-s", type=float, default=160)
    frozen_replay.add_argument("--outer-wall-s", type=float, default=12000)
    args = parser.parse_args(argv)
    if args.command == "preregister":
        preregister(args.output, timestamp=args.timestamp)
        return
    if args.command == "preregister-continuation":
        preregister_continuation(args.output, initial_registration=args.registration,
                                 previous_catalog=args.previous_catalog)
        return
    if args.command == "assemble":
        identity = assemble_and_validate(
            args.candidate_h5, args.candidate_manifest, args.output_dir,
            registration_path=args.registration, expected_old_h5=args.old_h5_sha256,
            expected_old_manifest=args.old_manifest_sha256)
        print(json.dumps({k: identity[k] for k in ("dataset_version", "class_counts", "h5_sha256", "manifest_sha256")}, sort_keys=True))
        return
    if args.command == "recover-candidates":
        result = recover_candidate_catalog(args.first_catalog, args.interrupted_catalog, args.output)
        print(json.dumps({"scene_count": result["scene_count"],
                          "greedy_unique": result["task_identity_audit"]["greedy_unique"],
                          "recovered_accepted": result["recovery_sources"][1]["accepted_incomplete_prefix"]}, sort_keys=True))
        return
    if args.command == "preregister-resume":
        result = preregister_candidate_resume(
            args.output, first_catalog=args.first_catalog,
            interrupted_catalog=args.interrupted_catalog, recovered_catalog=args.recovered_catalog,
            first_journal=args.first_journal, interrupted_journal=args.interrupted_journal,
            timestamp=args.timestamp)
        print(json.dumps({"next_unseen_seed": result["seed_recovery"]["next_unseen_seed"],
                          "cross_journal_seed_overlap": result["seed_recovery"]["cross_journal_seed_overlap"]}, sort_keys=True))
        return
    if args.command == "candidate-canary":
        result = run_candidate_canary(args.registration, args.recovered_catalog, args.output)
        print(json.dumps({"attempts": result["attempts"], "accepted": result["accepted"],
                          "preflight_rejections": result["failures"].get("task_identity_duplicate_preflight", 0),
                          "complete": result["complete"]}, sort_keys=True))
        return
    if args.command == "preregister-full-resume":
        result = preregister_full_candidate_resume(
            args.output, recovered_catalog=args.recovered_catalog,
            first_journal=args.first_journal, interrupted_journal=args.interrupted_journal,
            canary_journal=args.canary_journal)
        print(json.dumps({"first_proposal_seed": result["cursor"]["first_proposal_seed"],
                          "remaining": result["remaining_unique_layout_target"],
                          "attempt_budget": result["finite_budget"]["attempt_budget"],
                          "wall_seconds": result["finite_budget"]["wall_seconds"]}, sort_keys=True))
        return
    if args.command == "run-registered-candidate-batch":
        result = run_registered_candidate_batch(args.registration, args.recovered_catalog)
        print(json.dumps({"complete": result["complete"], "accepted": result["accepted"],
                          "attempts": result["attempts"],
                          "preflight_duplicates": result["failures"].get("task_identity_duplicate_preflight", 0)}, sort_keys=True))
        return
    if args.command == "collect-frozen-matches":
        result = collect_frozen_candidate_matches(
            args.selection, args.pool_manifest, args.dataset_dir, args.experiment_dir,
            episode_wall_seconds=args.episode_wall_s, outer_wall_seconds=args.outer_wall_s)
        print(json.dumps({k: result[k] for k in (
            "complete", "accepted", "attempts", "unused_candidate_ids", "rejected_candidate_ids",
            "candidate_obstacle_h5_sha256", "candidate_manifest_sha256")}, sort_keys=True))
        return
    result = verify_obstacle_candidate(args.h5, args.manifest)
    result["rows"] = [{k: v for k, v in row.items() if k != "layout"} for row in result["rows"]]
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
