"""Bounded native-EE v4 pair and preregistered 20-case validation runner."""

import argparse
import hashlib
import json
import math
import time
import traceback
import platform
import importlib.metadata
import shlex
import sys
from pathlib import Path

from examples.baselines.constrain.plug_orientation_ee_expert import (
    MAX_ALIGN_CONTROLS, MAX_CONTROLS, run_case,
)

PAIR_SEEDS = (2026113003, 2026113001)
FULL_SEEDS = tuple(range(2026113000, 2026113010))
RANDOM_YAWS = tuple(float(x) for x in __import__("numpy").linspace(-math.pi / 8, math.pi / 8, 10))
PAIR_CONTROLS = 3000
FULL_EXPERT_CONTROLS = 12000
FULL_REPLAY_CONTROLS = 12000
FULL_CONTROLS = 24000
PAIR_SECONDS = 1800
FULL_EXPERT_SECONDS = 3600
FULL_REPLAY_SECONDS = 3600
FULL_SECONDS = 9000
OUTER_REQUIRED_TIMEOUT_SECONDS = 9000
SOURCE_PATHS = (
    "examples/baselines/constrain/run_plug_ee_lift_v4.py",
    "examples/baselines/constrain/plug_orientation_ee_expert.py",
    "examples/baselines/constrain/peg_orientation_near_hole_perturb.py",
    "examples/baselines/constrain/peg_orientation_takeover.py",
    "examples/baselines/constrain/peg_orientation_preflight.py",
    "examples/baselines/constrain/plug_orientation.py",
    "mani_skill/envs/tasks/tabletop/plug_charger.py",
    "mani_skill/examples/motionplanning/panda/motionplanner.py",
    "mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py",
    "mani_skill/examples/motionplanning/base_motionplanner/utils.py",
    "mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py",
    "mani_skill/agents/base_agent.py",
    "mani_skill/agents/robots/panda/panda_wristcam.py",
    "mani_skill/agents/robots/panda/panda.py",
    "mani_skill/agents/controllers/pd_ee_pose.py",
    "mani_skill/agents/controllers/utils/kinematics.py",
    "mani_skill/agents/controllers/base_controller.py",
    "mani_skill/utils/wrappers/record.py",
    "mani_skill/utils/structs/pose.py",
    "mani_skill/utils/structs/actor.py",
    "mani_skill/utils/structs/articulation.py",
    "mani_skill/utils/structs/base.py",
    "mani_skill/utils/registration.py",
    "mani_skill/utils/common.py",
    "mani_skill/utils/building/ground.py",
    "mani_skill/agents/controllers/pd_joint_pos.py",
    "mani_skill/utils/scene_builder/table/scene_builder.py",
    "mani_skill/envs/sapien_env.py",
)


def _runtime_versions():
    versions = {"python": platform.python_version()}
    for package in ("numpy", "torch", "gymnasium", "sapien", "mplib", "h5py", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _seal_preregistration(mode, exp, demos, cases, started_utc):
    prereg_path = exp / "preregistration.json"
    archive = exp / "source_archive"
    if prereg_path.exists() or archive.exists() or (exp / "native_attempt_001.json").exists():
        raise FileExistsError(f"refusing to overwrite sealed or prior outputs in {exp}")
    root = Path(__file__).resolve().parents[3]
    archived = {}
    for relative in SOURCE_PATHS:
        source = root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required provenance source missing: {relative}")
        digest = _sha(source)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if _sha(destination) != digest or _sha(source) != digest:
            raise RuntimeError(f"source changed while archiving: {relative}")
        archived[relative] = digest
    yaw_values = [yaw for _, yaw, profile in cases if profile == "random"]
    bins = [0] * 5
    for yaw in yaw_values:
        index = min(4, max(0, int((yaw + math.pi / 8) / (math.pi / 20))))
        bins[index] += 1
    registration = {
        "protocol": "plug_native_ee_teacher_v4", "mode": mode,
        "created_utc": started_utc, "planned_cases": [
            {"seed": seed, "target_yaw": yaw, "profile": profile}
            for seed, yaw, profile in cases],
        "random_target_yaws": yaw_values, "planned_random_yaw_bin_counts": bins,
        "execution": {"environment_id": "PlugChargerOrientation-v1",
                      "robot_uid": "panda_wristcam", "obs_mode": "state",
                      "control_mode": "pd_ee_delta_pose", "sim_backend": "physx_cpu",
                      "reward_mode": "sparse", "max_episode_steps": 1000,
                      "observation_dim": 46, "action_dim": 7,
                      "target_yaw_override_applied_at_reset": True},
        "control_caps": {"per_episode": MAX_CONTROLS, "align": MAX_ALIGN_CONTROLS,
                         "pair_total": PAIR_CONTROLS if mode == "pair" else None,
                         "full_expert": FULL_EXPERT_CONTROLS if mode == "full" else None,
                         "full_replay": FULL_REPLAY_CONTROLS if mode == "full" else None,
                         "full_total": FULL_CONTROLS if mode == "full" else None},
        "wall_seconds": {"pair": PAIR_SECONDS if mode == "pair" else None,
                         "expert": FULL_EXPERT_SECONDS if mode == "full" else None,
                         "replay": FULL_REPLAY_SECONDS if mode == "full" else None,
                         "full": FULL_SECONDS if mode == "full" else None,
                         "outer_required_timeout_seconds": OUTER_REQUIRED_TIMEOUT_SECONDS},
        "official_gate": {"position_m": 0.005, "quaternion_angle_rad": 0.2},
        "acceptance": {"fixed_expert_minimum": 8, "random_expert_minimum": 8,
                       "replay_fraction_each_profile": 0.95,
                       "random_accepted_minimum": 4, "random_bins": 5,
                       "random_accepted_includes_endpoint": True},
        "historical_numeric_results_viewed": True,
        "blinding": "nonblind preflight confirmation; not a new blind test",
        "outer_command": "timeout --signal=TERM --kill-after=20s 9000s " +
                         " ".join(shlex.quote(value) for value in (
                             sys.executable,
                             str(root / "examples/baselines/constrain/run_plug_ee_lift_v4.py"),
                             mode, "--experiment-dir", str(exp), "--demo-dir", str(demos))),
        "runtime_versions": _runtime_versions(), "source_file_count": len(archived),
        "source_sha256": archived,
        "source_archive": "source_archive", "demo_dir": str(demos),
    }
    _write(prereg_path, registration)
    prereg_bytes = prereg_path.read_bytes()
    return prereg_path, hashlib.sha256(prereg_bytes).hexdigest(), archived, root


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(mode, experiment_dir, demo_dir):
    exp, demos = Path(experiment_dir), Path(demo_dir)
    exp.mkdir(parents=True, exist_ok=True)
    if any(exp.iterdir()):
        raise FileExistsError(f"experiment directory is not empty: {exp}")
    demos.mkdir(parents=True, exist_ok=True)
    if any(demos.iterdir()):
        raise FileExistsError(f"demo directory is not empty: {demos}")
    started = time.monotonic()
    utc_started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cases = ([(seed, 0., "fixed_pair") for seed in PAIR_SEEDS] if mode == "pair" else
             [(seed, 0., "fixed") for seed in FULL_SEEDS] +
             [(seed, yaw, "random") for seed, yaw in zip(FULL_SEEDS, RANDOM_YAWS)])
    report = {"protocol": "plug_native_ee_teacher_v4", "mode": mode,
              "target_yaws": [yaw for _, yaw, _ in cases], "cases": [],
              "align_control_cap": MAX_ALIGN_CONTROLS,
              "episode_control_cap": MAX_CONTROLS,
              "started_utc": utc_started, "started_monotonic": started}
    summary_path = exp / "native_attempt_001.json"
    prereg_path, prereg_sha, source_hashes, source_root = _seal_preregistration(
        mode, exp, demos, cases, utc_started)
    report["preregistration_sha256"] = prereg_sha
    report["source_sha256"] = source_hashes
    try:
        for index, (seed, yaw, profile) in enumerate(cases):
            if hashlib.sha256(prereg_path.read_bytes()).hexdigest() != prereg_sha:
                raise RuntimeError("sealed preregistration changed before physical case")
            for relative, digest in source_hashes.items():
                if _sha(source_root / relative) != digest or _sha(exp / "source_archive" / relative) != digest:
                    raise RuntimeError(f"sealed source identity changed before case: {relative}")
            if time.monotonic() - started > (PAIR_SECONDS if mode == "pair" else FULL_SECONDS):
                raise TimeoutError("registered whole-stage wall-time budget exhausted")
            case_dir = demos / f"{profile}_seed_{seed}"
            row = {"seed": seed, "target_yaw": yaw, "profile": profile,
                   "started_monotonic": time.monotonic()}
            report["cases"].append(row)
            try:
                result = run_case(seed, case_dir, target_yaw=yaw)
                source_controls = int(result["expert_controls"])
                replay_controls = int(result["replay_controls"])
                if source_controls > MAX_CONTROLS or replay_controls > MAX_CONTROLS:
                    raise RuntimeError("per-episode 600-control cap exceeded")
                row.update({"status": "completed", "expert_controls": source_controls,
                            "replay_controls": replay_controls,
                            "expert_elapsed_seconds": float(result["expert_elapsed_seconds"]),
                            "replay_elapsed_seconds": float(result["replay_elapsed_seconds"]),
                            "expert_success": bool(result["expert"]["success"]),
                            "expert_official_success": bool(result["expert"]["official_success"]),
                            "replay_success": bool(result["replay"]["success"]),
                            "case_json": str(case_dir / "case.json"),
                            "finished_monotonic": time.monotonic()})
                report.setdefault("expert_controls", 0)
                report.setdefault("replay_controls", 0)
                report["expert_controls"] += source_controls
                report["replay_controls"] += replay_controls
                report["controls"] = report["expert_controls"] + report["replay_controls"]
                report["expert_seconds"] = sum(r.get("expert_elapsed_seconds", 0.)
                                                for r in report["cases"])
                report["replay_seconds"] = sum(r.get("replay_elapsed_seconds", 0.)
                                                for r in report["cases"])
                if mode == "pair" and report["controls"] > PAIR_CONTROLS:
                    raise RuntimeError("pair control envelope exceeded")
                if mode == "full":
                    if report["expert_controls"] > FULL_EXPERT_CONTROLS:
                        raise RuntimeError("aggregate expert controls exceeded")
                    if report["replay_controls"] > FULL_REPLAY_CONTROLS:
                        raise RuntimeError("aggregate replay controls exceeded")
                    if report["controls"] > FULL_CONTROLS:
                        raise RuntimeError("whole-run control envelope exceeded")
                    if report["expert_seconds"] > FULL_EXPERT_SECONDS:
                        raise RuntimeError("aggregate expert wall-time exceeded")
                    if report["replay_seconds"] > FULL_REPLAY_SECONDS:
                        raise RuntimeError("aggregate replay wall-time exceeded")
                _write(summary_path, report)
            except BaseException as exc:
                row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "finished_monotonic": time.monotonic()})
                partial = case_dir / "partial_failure.json"
                row["partial_failure_path"] = str(partial) if partial.exists() else None
                _write(summary_path, report)
                raise
        if mode == "full":
            report["fixed_expert_successes"] = sum(
                r["profile"] == "fixed" and r.get("expert_success", False) for r in report["cases"])
            report["random_expert_successes"] = sum(
                r["profile"] == "random" and r.get("expert_success", False) for r in report["cases"])
            report["random_target_yaws"] = [r["target_yaw"] for r in report["cases"] if r["profile"] == "random"]
            def yaw_bin(yaw):
                return min(4, max(0, int((yaw + math.pi / 8) / (math.pi / 20))))
            report["random_yaw_bin_counts_planned"] = [
                sum(yaw_bin(y) == i for y in report["random_target_yaws"]) for i in range(5)]
            accepted_yaws = [r["target_yaw"] for r in report["cases"]
                             if r["profile"] == "random" and r.get("expert_success") and r.get("replay_success")]
            report["random_accepted_yaw_bin_counts"] = [
                sum(yaw_bin(y) == i for y in accepted_yaws) for i in range(5)]
            report["random_accepted_endpoint_covered"] = any(
                math.isclose(y, -math.pi / 8, abs_tol=1e-9) or
                math.isclose(y, math.pi / 8, abs_tol=1e-9) for y in accepted_yaws)
            report["fixed_gate_pass"] = report["fixed_expert_successes"] >= 8
            report["random_gate_pass"] = report["random_expert_successes"] >= 8
            report["note"] = "Replay acceptance and random coverage require independent audit; no automatic admission."
        report["exit_code"] = 0
    except BaseException:
        report.setdefault("exit_code", 1)
        raise
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report["elapsed_seconds"] = time.monotonic() - started
        _write(summary_path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pair", "full"))
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--demo-dir", required=True)
    args = parser.parse_args()
    run(args.mode, args.experiment_dir, args.demo_dir)


if __name__ == "__main__":
    main()
