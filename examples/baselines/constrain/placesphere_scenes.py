"""Construct, fingerprint, and restore fixed PlaceSphere obstacle scenes.

This module uses the native PlaceSphere expert runner from ``placesphere_pilot``;
it never loads a policy or demonstration trajectory.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain import placesphere_obstacle
from examples.baselines.constrain.placesphere_pilot import (
    PlaceSpherePilotConfig,
    _attempt,
    _clone_tree,
    _json,
    _obs_vector,
    _policy_observation,
    _set_state_dict_exact,
    make_env,
)

CONTRACT = Path("demos/PlaceSphere-obstacle/20261022_184558_formal/frozen_contract.json")
FORMAT = "placesphere-obstacle-scenes-v1"
STATE_ATOL = 1e-7
OBS_ATOL = 1e-7


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha(path):
    return _sha_bytes(Path(path).read_bytes())


def _tree_hash(tree):
    digest = hashlib.sha256()

    def visit(value, path=""):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}/{key}")
            return
        array = np.ascontiguousarray(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
        digest.update(path.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    visit(tree)
    return digest.hexdigest()


def _write_tree(group, tree):
    for key, value in tree.items():
        if isinstance(value, dict):
            _write_tree(group.create_group(key), value)
        else:
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            group.create_dataset(key, data=array)


def _read_tree(group):
    return {key: (_read_tree(value) if isinstance(value, h5py.Group) else np.asarray(value[()]))
            for key, value in group.items()}


def _controller_to_runtime(saved, template):
    if isinstance(template, dict):
        if not isinstance(saved, dict) or saved.keys() != template.keys():
            raise ValueError("controller state schema differs from runtime")
        return {key: _controller_to_runtime(saved[key], template[key]) for key in template}
    if torch.is_tensor(template):
        value = torch.as_tensor(saved, dtype=template.dtype, device=template.device)
        if value.shape != template.shape:
            raise ValueError("controller state shape differs from runtime")
        return value
    if isinstance(template, np.ndarray):
        value = np.asarray(saved, dtype=template.dtype)
        if value.shape != template.shape:
            raise ValueError("controller state shape differs from runtime")
        return value
    return type(template)(saved)


def _tree_max_abs(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return float("inf")
        return max((_tree_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, dict) or isinstance(right, dict):
        return float("inf")
    a = np.asarray(left.detach().cpu() if torch.is_tensor(left) else left)
    b = np.asarray(right.detach().cpu() if torch.is_tensor(right) else right)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0))


def _next_attempt_number(*, attempts, accepted, target, max_attempts, elapsed, max_wall_seconds):
    if accepted >= target or attempts >= max_attempts or elapsed >= max_wall_seconds:
        return None
    return attempts + 1


def _snapshot_sources(root, contract_path, commit):
    """Persist exact contract and actual source bytes before environment setup."""
    source_dir = root / "source_snapshot"
    source_dir.mkdir()
    paths = ["examples/baselines/constrain/placesphere_scenes.py",
             "examples/baselines/constrain/placesphere_pilot.py",
             "examples/baselines/constrain/placesphere_obstacle.py",
             "examples/baselines/constrain/layout_geometry.py",
             "examples/baselines/constrain/pickcube_expert.py",
             "examples/baselines/constrain/placesphere_formal.py",
             "mani_skill/envs/tasks/tabletop/place_sphere.py",
             "mani_skill/examples/motionplanning/panda/solutions/place_sphere.py"]
    repo = Path(__file__).resolve().parents[3]
    files = {}
    for relative in paths:
        payload = (repo / relative).read_bytes()
        archive = source_dir / relative
        archive.parent.mkdir(parents=True, exist_ok=True)
        with archive.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        files[relative] = {"sha256": _sha_bytes(payload), "snapshot_path": archive.relative_to(root).as_posix()}
    contract_bytes = contract_path.read_bytes()
    contract_copy = root / "frozen_contract.json"
    with contract_copy.open("xb") as stream:
        stream.write(contract_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    frozen = json.loads(contract_bytes)
    manifest = {"captured_before_first_attempt": True, "git_commit": commit,
                "frozen_contract_sha256": _sha_bytes(contract_bytes),
                "contract_source_fingerprints": frozen["local_source_sha256_before_collection"],
                "contract_native_source_fingerprints": frozen["native_worktree_source_sha256"],
                "actual_source_files": files}
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def generate(output_dir, *, seed=40017, target=100, max_attempts=1000,
             max_wall_seconds=3600.0, attempt_runner=None):
    """Generate an immutable all-or-incomplete scene set in a fresh exp folder."""
    if (isinstance(target, bool) or target < 1 or isinstance(max_attempts, bool)
            or max_attempts < 1 or not np.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("target, attempt budget, and finite wall budget must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    config = PlaceSpherePilotConfig(seed=seed, target_per_class=target,
        obstacle_target=target, direct_target=target, max_attempts_per_class=max_attempts,
        max_steps=200, max_wall_seconds=max_wall_seconds, obstacle_radius=.03,
        obstacle_height=.08, safety_margin=.005)
    import subprocess
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    contract_path = repo / CONTRACT
    provenance = _snapshot_sources(root, contract_path, commit)
    contract = json.loads(contract_path.read_text())
    if contract["obstacle"]["radius_m"] != .03 or contract["obstacle"]["height_m"] != .08:
        raise ValueError("frozen PlaceSphere obstacle geometry differs from required cylinder")
    if contract["sampling"]["same_base_side_only"] is not True:
        raise ValueError("frozen contract does not require same-base-side waypoints")
    config_path = root / "config.json"
    config_payload = {"generator": "placesphere_scenes-v1", "seed": seed,
        "target": target, "max_attempts": max_attempts, "max_wall_seconds": max_wall_seconds,
        "observation_mode": "state", "robot_uid": "panda", "control_mode": "pd_ee_delta_pose",
        "physics_backend": "physx_cpu", "observation_dimension": 39,
        "raw_observation_dimension": 42, "contract_sha256": provenance["frozen_contract_sha256"],
        "source_snapshot_manifest_sha256": _sha(root / "provenance.json"),
        "source_fingerprints": provenance["actual_source_files"],
        "contract_source_fingerprints": provenance["contract_source_fingerprints"],
        "tolerances": {"full_state_atol": STATE_ATOL, "raw_observation_atol": OBS_ATOL,
                       "policy_observation_atol": OBS_ATOL, "controller_atol": OBS_ATOL}}
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
    (root / "attempts.jsonl").touch()
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    rows, failures, phase_seconds, totals = [], {}, {}, {}
    runner = attempt_runner or _attempt
    report = {"complete": False, "target": target, "accepted": 0, "attempts": 0,
              "actual_runner_invocations": 0, "failures": {}, "phase_elapsed_seconds": {},
              "elapsed_seconds": 0.0, "generation_seed": seed,
              "attempt_budget": max_attempts, "wall_budget_seconds": max_wall_seconds,
              "frozen_contract_sha256": provenance["frozen_contract_sha256"]}
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with h5py.File(root / "scenes.h5", "x") as h5:
        attempts = 0
        while (number := _next_attempt_number(attempts=attempts, accepted=len(rows), target=target,
                max_attempts=max_attempts, elapsed=time.monotonic()-started,
                max_wall_seconds=max_wall_seconds)) is not None:
            attempts = number
            scene_seed = int(rng.integers(0, 2**31-1))
            attempt_start = time.monotonic()
            try:
                result, reason = runner("obstacle", scene_seed, rng, config)
            except Exception as error:
                result, reason = {"failure": "runner_exception", "diagnostics": {"error": repr(error)}}, "runner_exception"
            duration = time.monotonic() - attempt_start
            result = result or {}
            accepted = "actions" in result and result.get("native_success") and result.get("safe_success")
            row = {"attempt": attempts, "seed": scene_seed, "elapsed_seconds": duration,
                   "accepted": bool(accepted), "failure": None if accepted else reason or result.get("failure", "unspecified")}
            for key, value in result.get("counters", {}).items():
                totals[key] = totals.get(key, 0) + int(value)
            for key, value in result.get("phase_timings", {}).items():
                phase_seconds[key] = phase_seconds.get(key, 0.0) + float(value)
            if accepted:
                index = len(rows)
                sid = f"placesphere-{index:03d}"
                group = h5.create_group(sid)
                state = result["states"][0]
                controller = result["controller_states"][0]
                raw_obs = np.asarray(result["raw_obs"][0], dtype=np.float32)
                policy_obs = np.asarray(result["obs"][0], dtype=np.float32)
                actions = np.asarray(result["actions"], dtype=np.float32)
                _write_tree(group.create_group("state"), state)
                _write_tree(group.create_group("controller"), controller)
                group.create_dataset("initial_raw_observation", data=raw_obs)
                group.create_dataset("initial_policy_observation", data=policy_obs)
                group.create_dataset("expert_actions", data=actions)
                group.create_dataset("expert_raw_observations", data=result["raw_obs"])
                group.create_dataset("expert_policy_observations", data=result["obs"])
                group.attrs["task_contract_json"] = json.dumps(config_payload, sort_keys=True)
                trace = result["trace"]
                layout = result["layout"]
                cylinder = {"center_xy": trace["obstacle_center"], "radius": float(trace["actual_cylinder_radius_m"]), "height": float(trace["actual_cylinder_height_m"]),
                            "actual_collision_radius": float(trace["actual_cylinder_radius_m"]), "actual_collision_height": float(trace["actual_cylinder_height_m"]),
                            "shape": "static_vertical_cylinder"}
                metadata = {"scene_id": sid, "seed": scene_seed,
                    "start_xy": layout["start_xy"], "container_xy": layout["container_xy"],
                    "base_xy": trace["base_xy"], "base_side": int(np.sign(np.cross(
                        np.asarray(layout["container_xy"])-np.asarray(layout["start_xy"]),
                        np.asarray(trace["base_xy"])-np.asarray(layout["start_xy"])))),
                    "waypoint_xy": trace["waypoint_xy"], "cylinder": cylinder,
                    "expert": {"native_success": True, "safe_success": True,
                        "obstacle_collision": False, "obstacle_crossing": False,
                        "on_container": True, "static": True, "ungrasped": True,
                        "release_phase_reached": any(p.get("phase") == "release_open_and_retreat" and p.get("reached")
                                                      for p in trace["phases"]),
                        "post_release_safety": trace.get("post_release_safety"),
                        "settle_phase": next((p for p in trace["phases"] if p.get("phase") == "settle"), None),
                        "post_release_control_steps": trace.get("post_release_control_steps"),
                        "first_obstacle_contact": trace.get("first_obstacle_contact"),
                        "action_count": len(actions), "trace": trace},
                    "state_sha256": _tree_hash(state), "controller_sha256": _tree_hash(controller),
                    "raw_observation_sha256": _sha_bytes(raw_obs.tobytes()),
                    "policy_observation_sha256": _sha_bytes(policy_obs.tobytes()),
                    "actions_sha256": _sha_bytes(actions.tobytes()),
                    "raw_observation_shape": list(raw_obs.shape), "policy_observation_shape": list(policy_obs.shape),
                    "contract_sha256": provenance["frozen_contract_sha256"]}
                proof = metadata["expert"].get("post_release_safety") or {}
                if not metadata["expert"]["release_phase_reached"] or not all(proof.get(key) is True for key in
                        ("checked_through_native_success", "all_ungrasped", "no_obstacle_contact",
                         "no_forbidden_crossing", "terminal_on_bin", "terminal_static", "terminal_ungrasped")):
                    raise ValueError("expert acceptance missing complete post-release safety proof")
                if (abs(cylinder["radius"]-.03)>1e-7 or abs(cylinder["height"]-.08)>1e-7
                        or abs(cylinder["actual_collision_radius"]-cylinder["radius"])>1e-7
                        or abs(cylinder["actual_collision_height"]-cylinder["height"])>1e-7):
                    raise ValueError("constructed cylinder geometry differs from frozen scene geometry")
                rows.append(metadata)
                row.update(scene_id=sid, counters=result.get("counters", {}),
                           phase_timings=result.get("phase_timings", {}))
            else:
                failure = row["failure"]
                failures[failure] = failures.get(failure, 0) + 1
                row.update(counters=result.get("counters", {}), phase_timings=result.get("phase_timings", {}),
                           diagnostics=result.get("diagnostics", {}))
            with (root / "attempts.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            h5.flush()
            report.update(accepted=len(rows), attempts=attempts, actual_runner_invocations=attempts,
                failures=failures, counters=totals, phase_elapsed_seconds=phase_seconds,
                elapsed_seconds=time.monotonic()-started)
            (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report.update(complete=len(rows) == target, accepted=len(rows), attempts=attempts,
        actual_runner_invocations=attempts, failures=failures, counters=totals,
        phase_elapsed_seconds=phase_seconds, elapsed_seconds=time.monotonic()-started,
        acceptance_rate=len(rows)/attempts if attempts else 0.0,
        complete_attempt_budget_exhausted=attempts == max_attempts and len(rows) < target,
        wall_budget_exhausted=time.monotonic()-started >= max_wall_seconds)
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    scenes_path = root / "scenes.h5"
    manifest = {"format": FORMAT, "scene_count": len(rows), "scenes_file": scenes_path.name,
        "scenes_file_sha256": _sha(scenes_path), "config_sha256": _sha(config_path),
        "frozen_contract_sha256": provenance["frozen_contract_sha256"],
        "provenance_sha256": _sha(root / "provenance.json"),
        "attempt_journal_sha256": _sha(root / "attempts.jsonl"),
        "generation_report_sha256": _sha(root / "report.json"),
        "generation": report, "tolerances": config_payload["tolerances"], "scenes": rows}
    manifest["manifest_identity_sha256"] = _sha_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode())
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "checksums.json").write_text(json.dumps({name: _sha(root/name) for name in
        ("config.json", "frozen_contract.json", "provenance.json", "attempts.jsonl",
         "report.json", "scenes.h5", "manifest.json")}, indent=2, sort_keys=True) + "\n")
    if not report["complete"]:
        raise RuntimeError(f"incomplete PlaceSphere scene set: {report}")
    return report


class PlaceSphereSceneSet:
    """Verified immutable scene loader with true cylinder reconstruction on restore."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != FORMAT:
            raise ValueError("unsupported PlaceSphere scene format")
        identity = {key: value for key, value in self.manifest.items() if key != "manifest_identity_sha256"}
        if _sha_bytes(json.dumps(identity, sort_keys=True, indent=2).encode()) != self.manifest["manifest_identity_sha256"]:
            raise ValueError("scene manifest identity mismatch")
        for name, expected in (("scenes.h5", self.manifest["scenes_file_sha256"]),
                               ("config.json", self.manifest["config_sha256"]),
                               ("provenance.json", self.manifest["provenance_sha256"]),
                               ("attempts.jsonl", self.manifest["attempt_journal_sha256"]),
                               ("report.json", self.manifest["generation_report_sha256"])):
            if _sha(self.directory / name) != expected:
                raise ValueError(f"scene artifact checksum mismatch: {name}")
        self.path = self.directory / "scenes.h5"
        self.config = json.loads((self.directory / "config.json").read_text())
        self.tolerances = self.manifest["tolerances"]
        self.provenance = json.loads((self.directory / "provenance.json").read_text())
        if _sha(self.directory / "frozen_contract.json") != self.manifest["frozen_contract_sha256"]:
            raise ValueError("frozen contract checksum mismatch")
        for relative, record in self.provenance["actual_source_files"].items():
            if _sha(self.directory / record["snapshot_path"]) != record["sha256"]:
                raise ValueError(f"source snapshot checksum mismatch: {relative}")
        expected = self.provenance["contract_source_fingerprints"]
        current = json.loads((self.directory / "frozen_contract.json").read_text())
        if current["local_source_sha256_before_collection"] != expected:
            raise ValueError("frozen contract source identity mismatch")

    def __len__(self):
        return self.manifest["scene_count"]

    def load(self, index):
        row = self.manifest["scenes"][index]
        with h5py.File(self.path, "r") as h5:
            group = h5[row["scene_id"]]
            scene_contract = json.loads(group.attrs["task_contract_json"])
            if scene_contract != self.config or row["contract_sha256"] != self.manifest["frozen_contract_sha256"]:
                raise ValueError("scene task contract identity mismatch")
            scene = {"state": _read_tree(group["state"]), "controller": _read_tree(group["controller"]),
                "raw_observation": np.asarray(group["initial_raw_observation"]),
                "policy_observation": np.asarray(group["initial_policy_observation"]),
                "actions": np.asarray(group["expert_actions"]), "metadata": row}
        if _tree_hash(scene["state"]) != row["state_sha256"]:
            raise ValueError("scene state checksum mismatch")
        if _tree_hash(scene["controller"]) != row["controller_sha256"]:
            raise ValueError("scene controller checksum mismatch")
        for key, digest in (("raw_observation", "raw_observation_sha256"),
                             ("policy_observation", "policy_observation_sha256"),
                             ("actions", "actions_sha256")):
            if _sha_bytes(scene[key].tobytes()) != row[digest]:
                raise ValueError(f"scene {key} checksum mismatch")
        exp = row["expert"]
        if not all(exp.get(key) is True for key in ("native_success", "safe_success", "on_container", "static", "ungrasped", "release_phase_reached")):
            raise ValueError("scene expert acceptance evidence incomplete")
        if exp.get("obstacle_collision") is not False or exp.get("obstacle_crossing") is not False:
            raise ValueError("scene expert acceptance evidence unsafe")
        proof = exp.get("post_release_safety") or {}
        if not all(proof.get(key) is True for key in ("checked_through_native_success", "all_ungrasped",
                "no_obstacle_contact", "no_forbidden_crossing", "terminal_on_bin", "terminal_static", "terminal_ungrasped")):
            raise ValueError("scene post-release safety proof missing or unsafe")
        state_obstacle = np.asarray(scene["state"][placesphere_obstacle.OBSTACLE_KEY]).reshape(-1)
        cylinder = row["cylinder"]
        if state_obstacle[3] != 1 or np.any(state_obstacle[4:7] != 0):
            raise ValueError("unsafe obstacle state flags")
        if abs(float(state_obstacle[2]) - cylinder["radius"]) > 1e-7:
            raise ValueError("scene obstacle radius differs from recorded cylinder")
        if scene["raw_observation"].shape != (42,) or scene["policy_observation"].shape != (39,):
            raise ValueError("scene observation dimensions differ from frozen task contract")
        return scene

    def restore(self, index, env=None):
        scene = self.load(index)
        row = scene["metadata"]
        cylinder = row["cylinder"]
        owned = env is None
        if owned:
            env = make_env(config=PlaceSpherePilotConfig(obstacle_radius=cylinder["radius"], obstacle_height=cylinder["height"]))
        try:
            base = env.unwrapped
            if (abs(float(base.obstacle_collision_radius) - cylinder["radius"]) > 1e-7
                    or abs(float(base.obstacle_collision_height) - cylinder["height"]) > 1e-7):
                raise ValueError("constructed cylinder geometry differs from scene")
            env.reset(seed=row["seed"], options={"sphere_xy": row["start_xy"],
                "container_xy": row["container_xy"], "obstacle_enabled": True,
                "obstacle_center": cylinder["center_xy"]})
            template = base.get_state_dict()
            def convert(saved, like):
                if isinstance(like, dict):
                    if not isinstance(saved, dict) or saved.keys() != like.keys():
                        raise ValueError("native task state schema differs from runtime")
                    return {key: convert(saved[key], like[key]) for key in like}
                value = torch.as_tensor(saved, dtype=like.dtype, device=like.device)
                if value.shape != like.shape:
                    raise ValueError("native task state shape differs from runtime")
                return value
            _set_state_dict_exact(base, convert(scene["state"], template))
            controller_template = base.agent.controller.get_state()
            base.agent.controller.set_state(_controller_to_runtime(scene["controller"], controller_template))
            for wrapper in (env, base):
                if hasattr(wrapper, "_elapsed_steps"):
                    wrapper._elapsed_steps = 0
            if torch.is_tensor(getattr(base, "_elapsed_steps", None)):
                base._elapsed_steps.zero_()
            base._first_obstacle_contact = None
            actual_state = base.get_state_dict()
            actual_controller = base.agent.controller.get_state()
            info = base.evaluate()
            raw = _obs_vector(base.get_obs(info=info, unflattened=True))
            policy = _policy_observation(base, info)
            norms = {"full_native_state_max_abs": _tree_max_abs(scene["state"], actual_state),
                "controller_state_max_abs": _tree_max_abs(scene["controller"], actual_controller),
                "raw_observation_max_abs": float(np.max(np.abs(scene["raw_observation"]-raw), initial=0)),
                "policy_observation_max_abs": float(np.max(np.abs(scene["policy_observation"]-policy), initial=0))}
            limits = {"full_native_state_max_abs": self.tolerances["full_state_atol"],
                "controller_state_max_abs": self.tolerances["controller_atol"],
                "raw_observation_max_abs": self.tolerances["raw_observation_atol"],
                "policy_observation_max_abs": self.tolerances["policy_observation_atol"]}
            if any(not np.isfinite(value) or value > limits[key] for key, value in norms.items()):
                raise ValueError(f"fresh-environment scene restoration mismatch: {norms}; limits={limits}")
            return env, {"raw_observation": raw, "policy_observation": policy,
                         "state": actual_state, "controller": actual_controller, "norms": norms}
        except BaseException:
            if owned:
                env.close()
            raise


def audit(directory):
    catalog = PlaceSphereSceneSet(directory)
    rows = []
    audit_source_sha256 = _sha(Path(__file__).resolve())
    for index in range(len(catalog)):
        scene = catalog.load(index)
        env, restored = catalog.restore(index)
        try:
            first = {"state": _tree_hash(restored["state"]),
                     "controller": _tree_hash(restored["controller"]),
                     "raw_observation": _sha_bytes(restored["raw_observation"].tobytes()),
                     "policy_observation": _sha_bytes(restored["policy_observation"].tobytes())}
            cylinder_geometry = {"radius": float(env.unwrapped.obstacle_collision_radius),
                "height": float(env.unwrapped.obstacle_collision_height),
                "center_xy": env.unwrapped._obstacle_center[0].cpu().tolist()}
        finally:
            env.close()
        second_env, second = catalog.restore(index)
        try:
            actual = {"state": _tree_hash(second["state"]),
                      "controller": _tree_hash(second["controller"]),
                      "raw_observation": _sha_bytes(second["raw_observation"].tobytes()),
                      "policy_observation": _sha_bytes(second["policy_observation"].tobytes())}
        finally:
            second_env.close()
        rows.append({"scene_id": scene["metadata"]["scene_id"], "passed": True,
            "recorded_state_sha256": scene["metadata"]["state_sha256"],
            "recorded_controller_sha256": scene["metadata"]["controller_sha256"],
            "first_fresh_restore_fingerprints": first, "second_fresh_restore_fingerprints": actual,
            "fresh_restore_fingerprints_exactly_reproducible": first == actual,
            "first_restore_norms": restored["norms"], "second_restore_norms": second["norms"],
            "actual_cylinder": cylinder_geometry})
        if first != actual:
            raise RuntimeError(f"fresh native restore fingerprints differ for {scene['metadata']['scene_id']}")
    result = {"checked": len(rows), "all_passed": len(rows) == len(catalog), "scenes": rows,
              "audit_source_sha256": audit_source_sha256,
              "tolerances": catalog.tolerances}
    path = Path(directory) / "restore_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (Path(directory) / "restore_audit.sha256").write_text(_sha(path) + "  restore_audit.json\n")
    if not result["all_passed"]:
        raise RuntimeError("PlaceSphere scene restore audit incomplete")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=40017)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--max-wall-seconds", type=float, default=3600.0)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    if args.audit:
        result = audit(args.output_dir)
        if not result["all_passed"]:
            raise RuntimeError("scene audit incomplete")
    else:
        report = generate(args.output_dir, seed=args.seed, target=args.target,
            max_attempts=args.max_attempts, max_wall_seconds=args.max_wall_seconds)
        if not report["complete"]:
            raise RuntimeError(f"scene generation incomplete: {report}")


if __name__ == "__main__":
    main()
