"""Single bounded supervisor for the four-stage PlugCharger preflight."""
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

from examples.baselines.constrain import plug_orientation_preflight as preflight


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _terminate(proc, grace=20):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.communicate()
    try:
        return proc.communicate(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return proc.communicate()


def run_pipeline(output, dataset_root, *, popen_factory=subprocess.Popen,
                 whole_seconds=preflight.MAX_WHOLE_SECONDS,
                 expert_grasp_tilt_deg=None):
    output, dataset_root = Path(output), Path(dataset_root)
    output.mkdir(parents=True, exist_ok=False)
    dataset_root.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).resolve()
    worker = Path(preflight.__file__).resolve()
    started = time.monotonic()
    deadline = started + min(whole_seconds, preflight.MAX_WHOLE_SECONDS)
    manifest = {
        "status": "preregistered", "planned_denominator_per_profile": preflight.PLANNED_SCENES,
        "fixed_scenes": preflight.scene_plan("fixed"),
        "random_scenes": preflight.scene_plan("random"),
        "max_attempts_per_profile": preflight.MAX_ATTEMPTS,
        "max_controls_per_attempt": preflight.MAX_CONTROLS_PER_ATTEMPT,
        "max_expert_controls_combined": preflight.MAX_STAGE_CONTROLS,
        "max_replay_controls_combined": preflight.MAX_STAGE_CONTROLS,
        "max_expert_seconds_combined": 3600, "max_replay_seconds_combined": 3600,
        "max_whole_seconds": min(whole_seconds, preflight.MAX_WHOLE_SECONDS),
        "scripts": {str(p): sha256(p) for p in (script, worker)},
        "runtime": {"python": platform.python_version(), **{
            name: _version(name) for name in ("numpy", "torch", "sapien", "gymnasium", "mplib", "h5py")}},
        "dataset_root": str(dataset_root.resolve()),
        "planner_seed_policy": "scene_seed_plus_one_before_reset_and_planner_creation",
        "expert_grasp_tilt_deg": expert_grasp_tilt_deg,
        "expert_method": "official_solver" if expert_grasp_tilt_deg is None
        else "experiment_side_official_copy_parameterized_tilt",
    }
    manifest_path = output / "preregistration.json"
    with manifest_path.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    archive = output / "source_archive"
    archive.mkdir()
    closure = {"own/plug_orientation_pipeline.py": script,
               "own/plug_orientation_preflight.py": worker,
               "own/plug_orientation_expert_candidate.py": Path(worker).with_name("plug_orientation_expert_candidate.py"),
               "own/plug_orientation.py": Path(preflight.plug_orientation.__file__).resolve(),
               "own/peg_orientation_preflight.py": Path(__file__).with_name("peg_orientation_preflight.py").resolve()}
    # Capture the exact native task and official solver entrypoint, preserving package paths.
    native_modules = (
        "mani_skill.envs.tasks.tabletop.plug_charger",
        "mani_skill.examples.motionplanning.panda.solutions.plug_charger",
        "mani_skill.examples.motionplanning.panda.motionplanner",
        "mani_skill.examples.motionplanning.two_finger_gripper.motionplanner",
        "mani_skill.examples.motionplanning.base_motionplanner.motionplanner",
        "mani_skill.examples.motionplanning.base_motionplanner.utils",
        "mani_skill.trajectory.replay_trajectory",
        "mani_skill.trajectory.utils.actions.conversion",
        "mani_skill.utils.wrappers.record",
        "mani_skill.agents.robots.panda.panda_wristcam",
        "mani_skill.agents.robots.panda.panda",
        "mani_skill.agents.controllers.pd_ee_pose",
    )
    for module_name in native_modules:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin and Path(spec.origin).is_file():
            closure[f"native/{module_name.replace('.', '/')}.py"] = Path(spec.origin).resolve()
    hashes, current_sources = {}, {}
    for relative, path in closure.items():
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[relative] = sha256(target)
        current_sources[relative] = path
    manifest["source_closure"] = hashes
    manifest["source_count"] = len(hashes)
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    _validate_source_seal(manifest_path, archive, hashes, current_sources)
    phases = []
    expert_controls = replay_controls = 0
    expert_started = replay_started = None
    expert_deadline = replay_deadline = None
    try:
        for stage, profile in (("collect", "fixed"), ("collect", "random"),
                               ("replay", "fixed"), ("replay", "random")):
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError("whole-pipeline deadline exhausted")
            if stage == "collect":
                expert_started = expert_started or now
                expert_deadline = expert_deadline or min(deadline, expert_started + 3600)
                left = min(expert_deadline - now, deadline - now)
                budget = preflight.MAX_STAGE_CONTROLS - expert_controls
                command = [sys.executable, "-m", "examples.baselines.constrain.plug_orientation_preflight",
                           "--distribution", profile, "--output-dir", str(dataset_root / f"expert_{profile}"),
                           "--wall-seconds", str(max(1, int(left))), "--max-stage-controls", str(budget)]
                if expert_grasp_tilt_deg is not None:
                    command.extend(["--expert-grasp-tilt-deg", str(expert_grasp_tilt_deg)])
                report_file = dataset_root / f"expert_{profile}" / "collection_report.json"
                if profile == "random":
                    budget = preflight.MAX_STAGE_CONTROLS - expert_controls
                    command[command.index("--max-stage-controls") + 1] = str(budget)
            else:
                replay_started = replay_started or now
                replay_deadline = replay_deadline or min(deadline, replay_started + 3600)
                left = min(replay_deadline - now, deadline - now)
                if profile == "fixed":
                    source = dataset_root / "expert_fixed" / "expert.h5"
                else:
                    source = dataset_root / "expert_random" / "expert.h5"
                budget = preflight.MAX_STAGE_CONTROLS - replay_controls
                command = [sys.executable, "-m", "examples.baselines.constrain.plug_orientation_preflight",
                           "--distribution", profile, "--output-dir", str(dataset_root / f"expert_{profile}"),
                            "--replay-source", str(source), "--replay-output-dir", str(dataset_root / f"replay_{profile}"),
                           "--wall-seconds", str(max(1, int(left))), "--max-stage-controls", str(budget)]
                report_file = dataset_root / f"replay_{profile}" / "replay_report.json"
            phase = {"stage": stage, "profile": profile, "command": command,
                     "seconds_remaining_at_launch": left, "launched": False,
                     "controls_status": "unknown_if_no_report"}
            phases.append(phase)
            proc = popen_factory(command, start_new_session=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            phase["launched"] = True
            try:
                remaining = min(left, deadline - time.monotonic())
                stdout, _ = proc.communicate(timeout=max(0.001, remaining))
            except subprocess.TimeoutExpired:
                # The final communicate after TERM/KILL drains and returns buffered output.
                stdout, _ = _terminate(proc)
                (output / f"{stage}_{profile}.log").write_text(stdout or "")
                phase.update(returncode=proc.returncode, error="TimeoutError",
                             termination_returncode=proc.returncode)
                phase["controls_status"] = "unknown_if_no_report"
                manifest.update(status="partial", phases=phases,
                                measured_completed_expert_controls=expert_controls,
                                measured_completed_replay_controls=replay_controls,
                                partial_total_status="incomplete; timed-out phase controls unknown",
                                incomplete_phase_count=sum(bool(p.get("returncode") is None or p.get("error")) for p in phases),
                                elapsed_seconds=time.monotonic()-started)
                (output / "pipeline_report.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
                raise TimeoutError(f"{stage}/{profile} deadline exceeded")
            (output / f"{stage}_{profile}.log").write_text(stdout or "")
            phase.update(returncode=proc.returncode)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, command)
            report = json.loads(report_file.read_text())
            if stage == "collect":
                if not report.get("complete_plan"):
                    raise RuntimeError(f"{stage}/{profile} did not complete its fixed denominator")
                expert_controls += report["stage_controls"]
            else:
                if not report.get("complete"):
                    raise RuntimeError(f"{stage}/{profile} replay did not finish every source episode")
                replay_controls += report["stage_controls"]
        manifest.update(status="complete", phases=phases, expert_controls=expert_controls,
                        measured_completed_expert_controls=expert_controls,
                        measured_completed_replay_controls=replay_controls,
                        replay_controls=replay_controls, elapsed_seconds=time.monotonic()-started)
        (output / "pipeline_report.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        return manifest
    except BaseException as exc:
        manifest.update(status="partial", phases=phases,
                        error=f"{type(exc).__name__}: {exc}",
                        partial_total_status="incomplete",
                        measured_completed_expert_controls=expert_controls,
                        measured_completed_replay_controls=replay_controls,
                        incomplete_phase_count=sum(bool(p.get("returncode") is None or p.get("error")) for p in phases),
                        elapsed_seconds=time.monotonic()-started)
        (output / "pipeline_report.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
        raise


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _validate_source_seal(manifest_path, archive, hashes, current_sources):
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in hashes.items():
        source = current_sources[relative]
        archived = archive / relative
        if not archived.is_file() or sha256(archived) != expected:
            raise RuntimeError(f"source archive seal mismatch: {relative}")
        if sha256(source) != expected:
            raise RuntimeError(f"current source seal mismatch: {source}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=("run",))
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--expert-grasp-tilt-deg", type=float,
                        help="explicitly use experiment-side solver copy; official remains default")
    args = parser.parse_args()
    run_pipeline(args.output, args.dataset_root,
                 expert_grasp_tilt_deg=args.expert_grasp_tilt_deg)


if __name__ == "__main__":
    main()
