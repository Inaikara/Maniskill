"""Generate and strictly restore frozen-contract PickCube obstacle scenes."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from examples.baselines.constrain.pickcube_expert import (
    _set_state_dict_exact,
    make_pickcube_env,
)
from examples.baselines.constrain.pickcube_pilot import (
    PickCubePilotConfig,
    _default_attempt,
    _jsonable,
)


CONTRACT = Path("demos/PickCube-obstacle/20261022_132753_formal/frozen_contract.json")
RUNTIME_CONTRACT = Path(
    "outputs/exp/20261022_132753_pickcube_formal/validation_full_contract/runtime_contract_resolution.json"
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_hash(value):
    digest = hashlib.sha256()
    def visit(node, name=""):
        if isinstance(node, dict):
            for key in sorted(node):
                visit(node[key], f"{name}/{key}")
        else:
            array = np.ascontiguousarray(node.detach().cpu().numpy() if torch.is_tensor(node) else node)
            digest.update(name.encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes())
    visit(value)
    return digest.hexdigest()


def _write_tree(group, value):
    for key, item in value.items():
        if isinstance(item, dict):
            _write_tree(group.create_group(key), item)
        else:
            if torch.is_tensor(item):
                item = item.detach().cpu().numpy()
            group.create_dataset(key, data=np.asarray(item))


def _read_tree(group):
    return {key: (_read_tree(item) if isinstance(item, h5py.Group) else np.asarray(item[()]))
            for key, item in group.items()}


class PickCubeSceneSet:
    """Tamper-evident immutable scenes with explicit index load and restore."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != "pickcube-obstacle-scenes-v1":
            raise ValueError("unsupported PickCube scene format")
        payload = {k: v for k, v in self.manifest.items() if k != "manifest_sha256"}
        if _sha_bytes(json.dumps(payload, sort_keys=True, indent=2).encode()) != self.manifest["manifest_sha256"]:
            raise ValueError("scene manifest checksum mismatch")
        self.path = self.directory / self.manifest["scenes_file"]
        if _sha(self.path) != self.manifest["scenes_file_sha256"]:
            raise ValueError("scene data checksum mismatch")
        if _sha(self.directory / "config.json") != self.manifest["config_sha256"]:
            raise ValueError("scene config checksum mismatch")
        if _sha(CONTRACT) != self.manifest["frozen_contract_sha256"]:
            raise ValueError("frozen source contract checksum mismatch")
        if _sha(RUNTIME_CONTRACT) != self.manifest["runtime_contract_sha256"]:
            raise ValueError("frozen native runtime contract checksum mismatch")

    def __len__(self):
        return len(self.manifest["scenes"])

    def load(self, index):
        row = self.manifest["scenes"][index]
        with h5py.File(self.path, "r") as handle:
            group = handle[row["scene_id"]]
            state = _read_tree(group["state"])
            obs = np.asarray(group["initial_observation"])
            controller = json.loads(group.attrs["controller_json"])
        if _tree_hash(state) != row["state_sha256"]:
            raise ValueError("scene state checksum mismatch")
        if hashlib.sha256(obs.tobytes()).hexdigest() != row["observation_sha256"]:
            raise ValueError("scene observation checksum mismatch")
        expert = row.get("expert", {})
        if (expert.get("native_success") is not True
                or expert.get("safe_success", True) is not True
                or expert.get("obstacle_collision") is not False
                or expert.get("obstacle_crossing") is not False):
            raise ValueError("scene expert acceptance evidence is unsafe or incomplete")
        flags = state["pickcube_obstacle"].reshape(-1)
        if flags[3] != 1 or np.any(flags[4:7] != 0):
            raise ValueError("unsafe obstacle state flags in scene")
        return {"metadata": row, "state": state, "observation": obs, "controller": controller}

    def restore(self, index, env=None):
        scene = self.load(index)
        owned = env is None
        if owned:
            geometry = scene["metadata"]["cylinder"]
            env = make_pickcube_env(obstacle_radius=geometry["radius"], obstacle_height=geometry["height"])
        try:
            base = env.unwrapped
            geometry = scene["metadata"]["cylinder"]
            if (abs(float(base.obstacle_collision_radius) - geometry["radius"]) > 1e-7
                    or abs(float(base.obstacle_height) - geometry["height"]) > 1e-7):
                raise ValueError("constructed cylinder geometry differs from scene")
            template = base.get_state_dict()
            env.reset(seed=scene["metadata"]["seed"], options={
                "obstacle_enabled": True, "obstacle_center": geometry["center_xy"],
                "cube_xy": scene["metadata"]["start_xy"],
                "goal_pos": [*scene["metadata"]["goal_xy"], 0.12],
            })
            def convert(value, like):
                if isinstance(like, dict):
                    if set(value) != set(like):
                        raise ValueError("scene native state fields differ from runtime")
                    return {k: convert(value[k], like[k]) for k in like}
                return torch.as_tensor(value, dtype=like.dtype, device=like.device).reshape(like.shape)
            _set_state_dict_exact(base, convert(scene["state"], base.get_state_dict()))
            controller_template = base.agent.controller.get_state()
            base.agent.controller.set_state(_restore_controller(scene["controller"], controller_template))
            for wrapper in (env, base):
                if hasattr(wrapper, "_elapsed_steps"):
                    wrapper._elapsed_steps = 0
            if torch.is_tensor(getattr(base, "_elapsed_steps", None)):
                base._elapsed_steps.zero_()
            actual = base.get_obs()
            flat = np.asarray(actual.detach().cpu() if torch.is_tensor(actual) else actual).reshape(-1)
            if not np.array_equal(flat, scene["observation"]):
                raise ValueError("restored initial observation mismatch")
            return env, actual
        except BaseException:
            if owned:
                env.close()
            raise


def _restore_controller(value, template):
    if isinstance(template, dict):
        return {k: _restore_controller(value[k], v) for k, v in template.items()}
    if torch.is_tensor(template):
        return torch.as_tensor(value, dtype=template.dtype, device=template.device).reshape(template.shape)
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype).reshape(template.shape)
    if isinstance(template, (list, tuple)):
        return type(template)(value)
    return type(template)(value)


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def generate(output_dir, *, seed=20017, target=100, max_attempts=1000, max_wall_seconds=3600):
    """Bounded fresh-scene expert generation, journaling every attempt in exp/."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    if target < 1 or max_attempts < 1 or max_wall_seconds <= 0 or not np.isfinite(max_wall_seconds):
        raise ValueError("target, attempt budget, and finite wall budget must be positive")
    contract = json.loads(CONTRACT.read_text())
    # Source/config identity and the attempt journal exist before physics begins.
    from examples.baselines.constrain.pickcube_provenance import current_source_fingerprint
    source = current_source_fingerprint(Path(__file__).resolve().parents[3])
    expected = contract["source_fingerprints"]
    for name, digest in expected["files"].items():
        if source["files"].get(name) != digest:
            raise ValueError(f"frozen source mismatch: {name}")
    runtime_contract = json.loads(RUNTIME_CONTRACT.read_text())
    for name, digest in runtime_contract["source_sha256"].items():
        if _sha(name) != digest:
            raise ValueError(f"frozen native runtime source mismatch: {name}")
    runtime_env = make_pickcube_env(obstacle_radius=contract["obstacle"]["radius_m"],
                                    obstacle_height=contract["obstacle"]["height_m"])
    try:
        arm = runtime_env.unwrapped.agent.controller.controllers["arm"].config
        resolved = {
            "control_mode": runtime_env.unwrapped.control_mode,
            "pos_lower": np.broadcast_to(arm.pos_lower, 3).astype(float).tolist(),
            "pos_upper": np.broadcast_to(arm.pos_upper, 3).astype(float).tolist(),
            "rot_lower": np.broadcast_to(arm.rot_lower, 3).astype(float).tolist(),
            "rot_upper": np.broadcast_to(arm.rot_upper, 3).astype(float).tolist(),
            "controller_state_snapshot_observed": _jsonable(
                runtime_env.unwrapped.agent.controller.get_state()),
        }
    finally:
        runtime_env.close()
    for key, value in resolved.items():
        if value != runtime_contract[key]:
            raise ValueError(f"live native runtime contract mismatch: {key}")
    config = PickCubePilotConfig(seed=seed, target_per_class=target,
        max_attempts_per_class=max_attempts, max_steps=500, max_wall_seconds=max_wall_seconds,
        waypoint_offset_range=tuple(contract["sampling"]["waypoint_offset_range_m"]),
        obstacle_radius=contract["obstacle"]["radius_m"], obstacle_height=contract["obstacle"]["height_m"],
        obstacle_target=target, obstacle_max_attempts=max_attempts)
    config_data = {"config": config.__dict__, "contract": contract,
                   "runtime_contract": runtime_contract,
                   "seed": seed, "target": target, "max_attempts": max_attempts,
                   "max_wall_seconds": max_wall_seconds, "source_fingerprints": source}
    (output / "config.json").write_text(json.dumps(config_data, indent=2, sort_keys=True) + "\n")
    journal = output / "attempts.jsonl"
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    rows, failures, phases = [], {}, {}
    initial_report = {"target": target, "accepted": 0, "attempts": 0, "complete": False,
                      "failures": {}, "phase_elapsed_seconds": {}, "elapsed_seconds": 0.0,
                      "acceptance_rate": 0.0, "generation_seed": seed,
                      "wall_budget_seconds": max_wall_seconds, "source_fingerprints": source,
                      "config_sha256": _sha(output / "config.json"),
                      "frozen_contract_sha256": _sha(CONTRACT)}
    (output / "report.json").write_text(json.dumps(initial_report, indent=2, sort_keys=True) + "\n")
    journal.touch()
    scenes_path = output / "scenes.h5"
    with h5py.File(scenes_path, "w") as h5:
        for attempt in range(1, max_attempts + 1):
            if len(rows) >= target or time.monotonic() - started >= max_wall_seconds:
                break
            attempt_seed = int(rng.integers(0, 2**31 - 1))
            at = time.monotonic()
            result, reason = _default_attempt("obstacle", attempt_seed, rng, config)
            elapsed = time.monotonic() - at
            record = {"attempt": attempt, "seed": attempt_seed, "elapsed_seconds": elapsed,
                      "accepted": False, "failure": reason}
            attempt_phases = result.get("phase_timings", {}) if isinstance(result, dict) else {}
            for phase, value in attempt_phases.items():
                phases[phase] = phases.get(phase, 0.0) + float(value)
            if isinstance(result, dict) and "actions" in result:
                state = result["states"][0]
                identity = result["trace"]["runtime_initial_identity"]
                controller = identity["controller_state_snapshot"]
                obs = np.asarray(result["obs"][0])
                layout = result["layout"]
                cylinder = {"center_xy": result["trace"]["obstacle_center"],
                            "radius": float(config.obstacle_radius), "height": float(config.obstacle_height)}
                sid = f"pickcube-{len(rows):03d}"
                group = h5.create_group(sid)
                _write_tree(group.create_group("state"), state)
                group.create_dataset("initial_observation", data=obs)
                group.create_dataset("expert_actions", data=np.asarray(result["actions"], dtype=np.float32))
                group.attrs["controller_json"] = json.dumps(controller, sort_keys=True)
                trace = result["trace"]
                row = {"scene_id": sid, "seed": attempt_seed, "start_xy": layout["start_xy"],
                       "goal_xy": layout["goal_xy"], "base_xy": layout["base_xy"],
                       "cylinder": cylinder, "expert": {"native_success": True,
                       "obstacle_collision": False, "obstacle_crossing": False,
                       "safe_success": True, "steps": int(len(result["actions"])),
                       "waypoint_xy": trace.get("waypoint_xy"),
                       "cylinder_candidates_tested": trace.get("cylinder_candidates_tested"),
                       "cube_clearance_m": trace.get("cylinder_cube_clearance_m"),
                       "robot_sweep_clearance_m": trace.get("cylinder_robot_sweep_clearance_m"),
                       "first_obstacle_contact": trace.get("first_obstacle_contact"),
                       "runtime_initial_identity": trace["runtime_initial_identity"]},
                       "state_sha256": _tree_hash(state),
                       "observation_sha256": hashlib.sha256(obs.tobytes()).hexdigest(),
                       "observation_shape": list(obs.shape), "observation_dtype": str(obs.dtype),
                       "controller_sha256": _tree_hash(controller)}
                rows.append(row)
                record.update(accepted=True, scene_id=sid, phase_timings=result["phase_timings"],
                              counters=result["counters"])
            else:
                reason = result.get("failure", reason) if isinstance(result, dict) else reason
                record["failure"] = reason or "unspecified"
                failures[record["failure"]] = failures.get(record["failure"], 0) + 1
                if isinstance(result, dict):
                    record.update({k: result[k] for k in ("counters", "phase_timings", "diagnostics") if k in result})
            with journal.open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            h5.flush()
            initial_report.update(accepted=len(rows), attempts=attempt,
                failures=failures, phase_elapsed_seconds=phases,
                elapsed_seconds=time.monotonic() - started,
                acceptance_rate=len(rows) / attempt)
            (output / "report.json").write_text(json.dumps(initial_report, indent=2, sort_keys=True) + "\n")
    report = {"target": target, "accepted": len(rows), "attempts": attempt if max_attempts else 0,
              "complete": len(rows) == target, "failures": failures, "phase_elapsed_seconds": phases,
              "elapsed_seconds": time.monotonic() - started,
              "acceptance_rate": len(rows) / attempt if attempt else 0.0,
              "generation_seed": seed, "wall_budget_seconds": max_wall_seconds}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    manifest = {"format": "pickcube-obstacle-scenes-v1", "scene_count": len(rows),
                "scenes_file": scenes_path.name, "scenes": rows,
                "generation": report, "config_sha256": _sha(output / "config.json"),
                "frozen_contract_sha256": _sha(CONTRACT), "source_fingerprints": source,
                "runtime_contract_sha256": _sha(RUNTIME_CONTRACT),
                "scenes_file_sha256": _sha(scenes_path)}
    manifest["manifest_sha256"] = _sha_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report


def audit(directory):
    catalog = PickCubeSceneSet(directory)
    evidence = []
    for index in range(len(catalog)):
        env = make_pickcube_env(obstacle_radius=catalog.manifest["scenes"][index]["cylinder"]["radius"],
                                obstacle_height=catalog.manifest["scenes"][index]["cylinder"]["height"])
        try:
            catalog.restore(index, env)
            scene = catalog.load(index)
            evidence.append({"scene_id": scene["metadata"]["scene_id"], "roundtrip": True,
                             "state_sha256": scene["metadata"]["state_sha256"]})
        finally:
            env.close()
    path = Path(directory) / "restore_audit.json"
    path.write_text(json.dumps({"checked": len(evidence), "all_restored": len(evidence) == len(catalog),
                                "scenes": evidence}, indent=2) + "\n")
    if len(evidence) != len(catalog):
        raise RuntimeError("restore audit incomplete")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20017)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--max-wall-seconds", type=float, default=3600)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit(args.output_dir)
    else:
        report = generate(args.output_dir, seed=args.seed, target=args.target,
                          max_attempts=args.max_attempts, max_wall_seconds=args.max_wall_seconds)
        if not report["complete"]:
            raise RuntimeError(f"scene target incomplete: {report}")


if __name__ == "__main__":
    main()
