"""Fixed-contract native PlugCharger data collection and immutable packaging.

Collection is deliberately opt-in through ``collect_fixed_dataset``; importing
this module has no simulator or RNG side effects.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain import plug_orientation

ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = Path(__file__).with_name("plug_orientation_contract.json")


@contextmanager
def _collection_deadline(seconds):
    """Hard wall bound for a native callback (production runs on the main thread)."""
    if seconds <= 0:
        raise TimeoutError("registered collection wall-clock cap exhausted")
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("hard collection deadline requires the main thread and SIGALRM")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(_signum, _frame):
        raise TimeoutError("registered collection wall-clock cap exhausted during native callback")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
DATA_H5 = "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
DATA_JSON = "trajectory.state.pd_ee_delta_pose.physx_cpu.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@contextmanager
def _term_as_timeout(message):
    previous = signal.getsignal(signal.SIGTERM)
    def handle_term(_signum, _frame): raise TimeoutError(message)
    signal.signal(signal.SIGTERM, handle_term)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _native_episode_ok(path, case, replay=False):
    """Validate raw native HDF5 payload and the teacher's recorded official result."""
    if not path.is_file():
        return False, "missing_hdf5"
    with h5py.File(path, "r") as handle:
        if "traj_0" not in handle:
            return False, "missing_traj_0"
        traj = handle["traj_0"]
        if not all(key in traj for key in ("actions", "obs", "env_states")):
            return False, "missing_native_fields"
        actions = np.asarray(traj["actions"][:])
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0 or len(actions) > 600 or not np.isfinite(actions).all():
            return False, "invalid_actions"
        if np.max(np.abs(actions), initial=0) > 1.00001:
            return False, "action_out_of_range"
        # Recorder stores observations/states as nested structures. Their first
        # dimension is the authoritative T+1 alignment; verify all leaves.
        def leaves(node):
            if isinstance(node, h5py.Dataset):
                yield np.asarray(node[:])
            else:
                for child in node.values():
                    yield from leaves(child)
        obs_arrays = list(leaves(traj["obs"]))
        state_arrays = list(leaves(traj["env_states"]))
        if not obs_arrays or not state_arrays:
            return False, "missing_obs_or_states"
        if any(a.shape[0] != len(actions) + 1 or not np.isfinite(a).all() for a in obs_arrays + state_arrays):
            return False, "invalid_t_plus_one_or_nonfinite"
        final_success = _audit_final_pose(traj["env_states"], expected_yaw=float(case.get("target_yaw", math.nan)))
    outcome = case.get("replay" if replay else "expert", {})
    declared = outcome.get("success") is True
    official_declared = (outcome.get("success") is True if replay
                         else outcome.get("official_success") is True)
    if not declared or not official_declared or not final_success:
        return False, "official_success_false_or_missing"
    return True, None


def _audit_final_pose(states, expected_yaw=0.0):
    """Independently re-evaluate the frozen full-pose gate from saved native state."""
    from transforms3d.euler import euler2quat
    charger = np.asarray(states["actors/charger"][-1], dtype=np.float64)
    targets = np.asarray(states["plug_orientation_target"][:], dtype=np.float64)
    target = targets[-1]
    if (charger.shape[0] < 7 or target.shape != (4,) or not np.isfinite(charger).all()
            or not np.isfinite(targets).all() or not math.isfinite(expected_yaw)
            or not np.allclose(targets[:, 0], expected_yaw, atol=1e-7, rtol=0)
            or not np.allclose(targets[:, 1:4], targets[0, 1:4], atol=1e-8, rtol=0)):
        return False
    pos_error = float(np.linalg.norm(charger[:3] - target[1:4]))
    q_norm = float(np.linalg.norm(charger[3:7]))
    if not math.isfinite(q_norm) or q_norm <= 1e-12: return False
    q = charger[3:7] / q_norm
    goal_q = np.asarray(euler2quat(0, 0, float(target[0])), dtype=np.float64)
    dot = min(1.0, max(0.0, abs(float(np.dot(q, goal_q)))))
    angle = 2.0 * math.acos(dot)
    return pos_error <= 0.005 and angle <= 0.2


def _pair_native_match(source_path, replay_path):
    with h5py.File(source_path, "r") as source, h5py.File(replay_path, "r") as replay:
        left, right = source["traj_0"], replay["traj_0"]
        if not np.array_equal(left["actions"][:], right["actions"][:]): return False
        def same_initial(a, b):
            if isinstance(a, h5py.Group):
                return isinstance(b, h5py.Group) and set(a) == set(b) and all(same_initial(a[k], b[k]) for k in a)
            return np.array_equal(a[0], b[0])
        return (same_initial(left["env_states"], right["env_states"])
                and same_initial(left["obs"], right["obs"]))


def consolidate_attempts(dataset_dir, attempts, expected_count=1000):
    """Consolidate the first valid ordered cases, retaining source attempt IDs."""
    dataset_dir = Path(dataset_dir)
    if expected_count < 1:
        raise ValueError("expected_count must be positive")
    selected = []
    for row in attempts:
        if row.get("accepted"):
            selected.append(row)
            if len(selected) == expected_count:
                break
    if len(selected) != expected_count:
        raise ValueError(f"need exactly {expected_count} eligible ordered episodes; got {len(selected)}")
    json_rows = []
    output_h5 = dataset_dir / DATA_H5
    output_json = dataset_dir / DATA_JSON
    if output_h5.exists() or output_json.exists():
        raise FileExistsError("refusing to overwrite consolidated training data")
    with h5py.File(output_h5, "x") as target:
        for episode_id, row in enumerate(selected):
            source_path = Path(row["source_h5"]).resolve()
            replay_path = Path(row["replay_h5"]).resolve()
            if not source_path.is_file() or not replay_path.is_file():
                raise FileNotFoundError(f"missing source or replay for candidate {row['seed']}")
            with h5py.File(source_path, "r") as source:
                if "traj_0" not in source:
                    raise ValueError(f"source candidate {row['seed']} has no traj_0")
                source.copy("traj_0", target, name=f"traj_{episode_id}")
            case = json.loads(Path(row["case_json"]).read_text())
            expert_ok, expert_reason = _native_episode_ok(source_path, case)
            replay_ok, replay_reason = _native_episode_ok(replay_path, case, replay=True)
            if not (expert_ok and replay_ok and _pair_native_match(source_path, replay_path)):
                raise ValueError(f"candidate changed after admission: {row['seed']}: {expert_reason or replay_reason}")
            with h5py.File(source_path, "r") as h5:
                actions = np.asarray(h5["traj_0/actions"][:])
                env_states = h5["traj_0/env_states"]
                initial_charger = np.asarray(env_states["actors/charger"][0], dtype=np.float64)
                initial_receptacle = np.asarray(env_states["actors/receptacle"][0], dtype=np.float64)
                target_state = np.asarray(env_states["plug_orientation_target"][0], dtype=np.float64)
                obs_nodes = []
                # Stored native flattened policy observations are included in sidecar
                # JSON by the runner; HDF5 obs remains the exact recorder source.
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset) and name.startswith("obs/"):
                        obs_nodes.append(np.asarray(obj[:]))
                h5["traj_0"].visititems(visit)
                if isinstance(h5["traj_0/obs"], h5py.Dataset):
                    obs_nodes.append(np.asarray(h5["traj_0/obs"][:]))
                if not obs_nodes or not any(a.shape[-1:] == (46,) and np.isfinite(a).all() for a in obs_nodes):
                    raise ValueError(f"candidate {row['seed']} lacks finite native 46-D observations")
            json_rows.append({
                "episode_id": episode_id, "episode_seed": int(row["seed"]),
                "source_candidate_seed": int(row["seed"]),
                "source_episode_id": 0, "source_h5": str(source_path),
                "source_h5_sha256": sha256_file(source_path),
                "replay_h5": str(replay_path), "replay_h5_sha256": sha256_file(replay_path),
                "case_json": str(Path(row["case_json"]).resolve()),
                "case_json_sha256": sha256_file(row["case_json"]),
                "source_actions_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
                "charger_initial_pose": initial_charger[:7].tolist(),
                "receptacle_initial_pose": initial_receptacle[:7].tolist(),
                "goal_position_m": target_state[1:4].tolist(),
                "target_yaw_rad": float(target_state[0]),
            })
    payload = {"env_id": plug_orientation.ENV_ID, "robot_uid": "panda_wristcam",
               "obs_mode": "state", "control_mode": "pd_ee_delta_pose",
               "sim_backend": "physx_cpu", "observation_dim": 46,
               "action_dim": 7, "episodes": json_rows}
    _write_json(output_json, payload)
    return payload


def collect_fixed_dataset(dataset_dir, exp_dir, *, contract_path=CONTRACT_PATH,
                          run_case_fn=None, expected_count=1000, max_attempts=1500,
                          _test_mode=False):
    """Run the frozen ordered seed design, journaling each attempt before continuing.

    ``run_case_fn`` is an internal test seam. Production defaults to the exact
    sealed v4 teacher. Only the narrowly identified MPLib screw-planning failure
    is recoverable; all other exceptions are journaled and re-raised.
    """
    dataset_dir, exp_dir = Path(dataset_dir), Path(exp_dir)
    dataset_dir.mkdir(parents=True, exist_ok=False)
    exp_dir.mkdir(parents=True, exist_ok=False)
    contract_bytes = Path(contract_path).read_bytes()
    contract = json.loads(contract_bytes)
    spec = contract["data_collection"]
    seed_lo, seed_hi = spec["candidate_attempt_seed_range_inclusive"]
    if max_attempts < 1 or max_attempts > spec["candidate_attempt_limit"]:
        raise ValueError("max_attempts cannot exceed the frozen candidate limit")
    if expected_count != spec["accepted_episodes"] and not _test_mode:
        raise ValueError("production collection count is frozen at 1000")
    if run_case_fn is None:
        from examples.baselines.constrain.plug_orientation_ee_expert import run_case
        run_case_fn = run_case
    started = time.monotonic()
    archive = exp_dir / "source_archive"
    archive.mkdir()
    archived = {}
    from examples.baselines.constrain.run_plug_ee_lift_v4 import SOURCE_PATHS as TEACHER_SOURCE_PATHS
    source_files = list(TEACHER_SOURCE_PATHS) + [
        "examples/baselines/constrain/plug_orientation_data.py",
        "examples/baselines/constrain/plug_orientation_contract.json",
    ]
    for relative in source_files:
        source = ROOT / relative
        name = relative
        dest = archive / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        archived[relative] = sha256_file(source)
        if sha256_file(dest) != archived[name]:
            raise RuntimeError(f"source archive mismatch: {relative}")
    prereg = {"protocol": "plug_orientation_fixed_1000_v1",
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
              "contract_path": str(Path(contract_path).resolve()),
              "source_sha256": archived, "candidate_seed_range": [seed_lo, seed_hi],
               "attempt_limit": max_attempts, "accepted_episode_count": expected_count,
               "target_yaw_rad": 0.0, "teacher": "native_ee_teacher_v4",
              "teacher_configuration": {"algorithm": "native_ee_stream_final_target_hold_v4_object_pose_lift_tracking",
                  "max_controls": 600, "align_controls": 180, "lift_controls": 80,
                  "native_control_mode": "pd_ee_delta_pose", "success_gate": contract["base_policy"]["official_success"]},
              "per_attempt_control_cap": spec["per_attempt_control_cap"],
              "wall_clock_cap_seconds": spec["wall_clock_cap_seconds"],
              "runtime_versions": __import__("examples.baselines.constrain.run_plug_ee_lift_v4",
                  fromlist=["_runtime_versions"])._runtime_versions(),
              "source_archive": "source_archive",
              "dataset_dir": str(dataset_dir.resolve()),
              "eligibility": spec["eligibility"]}
    _write_json(exp_dir / "preregistration.json", prereg)
    prereg_hash = sha256_file(exp_dir / "preregistration.json")
    attempts, expert_controls, replay_controls = [], 0, 0
    unknown_expert_control_attempts, unknown_replay_control_attempts = 0, 0
    jsonl = dataset_dir / "attempts.jsonl"
    _write_json(dataset_dir / "attempt_index.json", {"attempts": []})
    with jsonl.open("x") as journal:
        for index in range(max_attempts):
            if time.monotonic() - started >= spec["wall_clock_cap_seconds"]:
                break
            seed = seed_lo + index
            if seed > seed_hi or sha256_file(exp_dir / "preregistration.json") != prereg_hash:
                raise RuntimeError("seed bound or preregistration identity violated")
            if any(sha256_file(archive / name) != digest or sha256_file(ROOT / name) != digest
                   for name, digest in archived.items()):
                raise RuntimeError("live or archived source identity changed")
            attempt_dir = dataset_dir / f"candidate_{index + 1:04d}_seed_{seed}"
            row = {"candidate_index": index + 1, "seed": seed, "target_yaw": 0.0,
                   "status": "running", "accepted": False, "attempt_dir": str(attempt_dir)}
            try:
                remaining = spec["wall_clock_cap_seconds"] - (time.monotonic() - started)
                with _term_as_timeout("SIGTERM received during native candidate"), _collection_deadline(remaining):
                    case = run_case_fn(seed, attempt_dir, target_yaw=0.0,
                                       deadline=started + spec["wall_clock_cap_seconds"])
                if time.monotonic() - started >= spec["wall_clock_cap_seconds"]:
                    raise TimeoutError("registered collection wall-clock cap exhausted after native callback")
                source = attempt_dir / "expert.h5"
                replay = attempt_dir / "replay" / "replay.h5"
                row.update(source_h5=str(source), replay_h5=str(replay),
                           case_json=str(attempt_dir / "case.json"),
                           expert_controls=int(case.get("expert_controls", 0)),
                           replay_controls=int(case.get("replay_controls", 0)))
                expert_controls += row["expert_controls"]
                replay_controls += row["replay_controls"]
                if max(row["expert_controls"], row["replay_controls"]) > spec["per_attempt_control_cap"]:
                    raise RuntimeError("per-attempt control cap exceeded")
                expert_ok, expert_reason = _native_episode_ok(source, case)
                replay_ok, replay_reason = _native_episode_ok(replay, case, replay=True)
                pair_match = expert_ok and replay_ok and _pair_native_match(source, replay)
                if time.monotonic() - started >= spec["wall_clock_cap_seconds"]:
                    raise TimeoutError("registered collection wall-clock cap exhausted before episode admission")
                row.update(status="completed", expert_eligible=expert_ok,
                           replay_eligible=replay_ok, raw_replay_pair_match=bool(pair_match),
                           rejection_reason=expert_reason or replay_reason or (None if pair_match else "raw_replay_mismatch"),
                           accepted=bool(expert_ok and replay_ok and pair_match))
                row["source_sha256"] = sha256_file(source) if source.is_file() else None
                row["replay_sha256"] = sha256_file(replay) if replay.is_file() else None
            except BaseException as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                partial = attempt_dir / "partial_failure.json"
                known_planning_failure = (type(exc) is RuntimeError
                    and str(exc) == "native teacher exceeded 600-control episode cap")
                known_planning_failure = known_planning_failure or (
                    type(exc) is RuntimeError and str(exc) == "MPLib screw planning failed")
                row["failure_class"] = "known_planning_failure" if known_planning_failure else "unexpected_exception"
                row["partial_failure"] = str(partial) if partial.exists() else None
                row["source_h5"] = str(attempt_dir / "expert.h5")
                row["replay_h5"] = str(attempt_dir / "replay" / "replay.h5")
                row["source_sha256"] = sha256_file(row["source_h5"]) if Path(row["source_h5"]).is_file() else None
                row["replay_sha256"] = sha256_file(row["replay_h5"]) if Path(row["replay_h5"]).is_file() else None
                try:
                    partial_payload = json.loads(partial.read_text()) if partial.exists() else {}
                    partial_count = partial_payload.get("expert_controls", row.get("expert_controls"))
                except (OSError, ValueError, TypeError) as metadata_exc:
                    partial_payload, partial_count = {}, None
                    row["partial_metadata_error"] = f"{type(metadata_exc).__name__}: {metadata_exc}"
                try:
                    row["expert_controls"] = int(partial_count) if partial_count is not None else None
                except (TypeError, ValueError, OverflowError) as metadata_exc:
                    row["expert_controls"] = None
                    row["partial_metadata_error"] = f"{type(metadata_exc).__name__}: {metadata_exc}"
                row["replay_controls"] = (0 if known_planning_failure else
                                           row.get("replay_controls"))
                if row["expert_controls"] is None: unknown_expert_control_attempts += 1
                else: expert_controls += row["expert_controls"]
                if row["replay_controls"] is None: unknown_replay_control_attempts += 1
                else: replay_controls += row["replay_controls"]
                attempts.append(row)
                journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                journal.flush(); os.fsync(journal.fileno())
                _write_json(dataset_dir / "attempt_index.json", {"attempts": attempts})
                _write_json(dataset_dir / "dataset_manifest.json", {
                    "dataset": dataset_dir.name, "protocol": prereg["protocol"], "status": "in_progress",
                    "contract_sha256": prereg["contract_sha256"], "preregistration_sha256": prereg_hash,
                    "collection_experiment_dir": str(exp_dir.resolve()),
                    "collection_preregistration_path": str((exp_dir / "preregistration.json").resolve()),
                    "collection_source_archive": str(archive.resolve()),
                    "candidate_attempt_count": len(attempts),
                    "accepted_count": sum(bool(item.get("accepted")) for item in attempts),
                    "expert_controls_known": expert_controls, "replay_controls_known": replay_controls,
                    "unknown_expert_control_attempts": unknown_expert_control_attempts,
                    "unknown_replay_control_attempts": unknown_replay_control_attempts,
                    "attempts_sha256": sha256_file(jsonl),
                    "attempt_index_sha256": sha256_file(dataset_dir / "attempt_index.json"),
                    "attempts": attempts})
                if not known_planning_failure:
                    raise
                if row["expert_controls"] is None:
                    raise RuntimeError("known planning failure omitted its expert control count; cannot certify budget")
                if row["expert_controls"] > spec["per_attempt_control_cap"]:
                    raise RuntimeError("failed native attempt exceeded the per-attempt control cap")
                if (expert_controls > spec["max_candidate_expert_controls"]
                        or replay_controls > spec["max_candidate_replay_controls"]):
                    raise RuntimeError("aggregate native-control contract exceeded")
                continue
            attempts.append(row)
            journal.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            journal.flush(); os.fsync(journal.fileno())
            _write_json(dataset_dir / "attempt_index.json", {"attempts": attempts})
            _write_json(dataset_dir / "dataset_manifest.json", {
                "dataset": dataset_dir.name, "protocol": prereg["protocol"], "status": "in_progress",
                "contract_sha256": prereg["contract_sha256"], "preregistration_sha256": prereg_hash,
                "collection_experiment_dir": str(exp_dir.resolve()),
                "collection_preregistration_path": str((exp_dir / "preregistration.json").resolve()),
                "collection_source_archive": str(archive.resolve()),
                "candidate_attempt_count": len(attempts),
                "accepted_count": sum(bool(item.get("accepted")) for item in attempts),
                "expert_controls_known": expert_controls, "replay_controls_known": replay_controls,
                "unknown_expert_control_attempts": unknown_expert_control_attempts,
                "unknown_replay_control_attempts": unknown_replay_control_attempts,
                "attempts_sha256": sha256_file(jsonl),
                "attempt_index_sha256": sha256_file(dataset_dir / "attempt_index.json"),
                "attempts": attempts})
            if sum(bool(item.get("accepted")) for item in attempts) >= expected_count:
                break
            if expert_controls > spec["max_candidate_expert_controls"] or replay_controls > spec["max_candidate_replay_controls"]:
                raise RuntimeError("aggregate native-control contract exceeded")
    consolidated = None
    if (sum(bool(r.get("accepted")) for r in attempts) == expected_count
            and time.monotonic() - started < spec["wall_clock_cap_seconds"]):
        consolidated = consolidate_attempts(dataset_dir, attempts, expected_count)
    coverage = [] if consolidated is None else [{"seed": row["episode_seed"],
        "target_yaw_rad": row["target_yaw_rad"], "goal_position_m": row["goal_position_m"],
        "charger_initial_position_m": row["charger_initial_pose"][:3],
        "receptacle_initial_pose": row["receptacle_initial_pose"]}
        for row in consolidated["episodes"]]
    manifest = {"dataset": dataset_dir.name, "protocol": prereg["protocol"],
                "status": "complete" if consolidated is not None else "incomplete",
                "started_utc": prereg["created_utc"],
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "elapsed_seconds": time.monotonic() - started,
                "collection_experiment_dir": str(exp_dir.resolve()),
                "collection_preregistration_path": str((exp_dir / "preregistration.json").resolve()),
                "collection_source_archive": str(archive.resolve()),
                "preregistration_sha256": prereg_hash, "contract_sha256": prereg["contract_sha256"],
                "candidate_attempt_count": len(attempts), "accepted_count": sum(r["accepted"] for r in attempts),
                "attempts_sha256": sha256_file(jsonl), "attempt_index_sha256": sha256_file(dataset_dir / "attempt_index.json"),
                "failure_classes": {kind: sum((r.get("failure_class") or r.get("rejection_reason")) == kind
                                               for r in attempts)
                                    for kind in sorted({r.get("failure_class") or r.get("rejection_reason")
                                                        for r in attempts
                                                        if r.get("failure_class") or r.get("rejection_reason")})},
                "acceptance_fraction": (sum(bool(r.get("accepted")) for r in attempts) / len(attempts)
                                        if attempts else 0.0),
                "expert_controls": expert_controls, "replay_controls": replay_controls,
                "unknown_expert_control_attempts": unknown_expert_control_attempts,
                "unknown_replay_control_attempts": unknown_replay_control_attempts,
                "selection_rule": "first 1000 eligible in candidate seed order",
                "controller_contract": {"control_mode": "pd_ee_delta_pose", "action_dim": 7,
                                        "action_bounds": [-1.0, 1.0], "observation_dim": 46,
                                        "target_yaw_rad": (attempts[0].get("target_yaw", 0.0) if len({r.get("target_yaw", 0.0) for r in attempts}) == 1 else None),
                                        "target_yaw_policy": "per_episode_actual_goal", "teacher": "native_ee_teacher_v4"},
                "training_coverage": coverage,
                "teacher_configuration": prereg["teacher_configuration"],
                "source_sha256": archived,
                "attempts": attempts}
    _write_json(dataset_dir / "dataset_manifest.json", manifest)
    if consolidated is not None:
        manifest.update(training_h5_sha256=sha256_file(dataset_dir / DATA_H5),
                        training_json_sha256=sha256_file(dataset_dir / DATA_JSON),
                        selected_seeds=[row["episode_seed"] for row in consolidated["episodes"]])
        _write_json(dataset_dir / "dataset_manifest.json", manifest)
    return manifest
