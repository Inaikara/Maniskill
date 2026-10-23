"""Keep the selected three-task artifacts; archive a hash inventory before pruning."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = (
    "20260930_base_dp_stackcube_980/best_eval_success_at_end.pt",
    "20261021_constrainnet_wristcam_seed1/final.pt",
    "20261022_144000_pickcube_base_dp/final.pt",
    "20261022_144835_pickcube_constrain/final.pt",
    "20261022_193900_placesphere_base_dp/final.pt",
    "20261022_201100_placesphere_constrain/final.pt",
)
EXPERIMENTS = (
    "20260930_base_dp_stackcube_980_seed1",
    "20261021_constrainnet_wristcam_seed1",
    "20261021_wristcam_scenes",
    "20261021_wristcam_paired_eval",
    "20261022_132753_pickcube_formal",
    "20261022_144000_pickcube_base_dp",
    "20261022_144835_pickcube_constrain",
    "20261022_152157_pickcube_scenes",
    "20261022_162002_pickcube_paired_eval",
    "20261022_184558_placesphere_formal",
    "20261022_193900_placesphere_base_dp",
    "20261022_201100_placesphere_constrain",
    "20261022_205500_placesphere_scenes",
    "20261022_213500_placesphere_paired_eval",
    "20261022_221641_multitask_obstacle_completion_audit",
)
DATASETS = {
    "PickCube-obstacle": "20261022_132753_formal",
    "PlaceSphere-obstacle": "20261022_184558_formal",
}
EVALUATIONS = {
    "StackCube": "20261021_wristcam_paired_eval",
    "PickCube": "20261022_162002_pickcube_paired_eval",
    "PlaceSphere": "20261022_213500_placesphere_paired_eval",
}


def relative(path):
    return str(path.relative_to(ROOT))


def files(path):
    if path.is_symlink():
        raise ValueError(f"Refusing symlink: {path}")
    if path.is_file():
        return [path]
    result = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"Refusing symlink: {child}")
        if child.is_file():
            result.append(child)
    return result


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": relative(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def build_plan():
    keep = [ROOT / "outputs/ckpt" / name for name in WEIGHTS]
    keep += [ROOT / "outputs/exp" / name for name in EXPERIMENTS]
    keep += [path for path in (ROOT / "outputs/exp").iterdir()
             if path.is_dir() and path.name.endswith("_final_artifact_cleanup")]
    keep += [ROOT / "demos" / name / run for name, run in DATASETS.items()]
    # Native source/replay files are protected by the repository's traceability contract.
    keep += [ROOT / "demos/StackCube-obstacle", ROOT / "demos/StackCube-v1"]
    for path in keep:
        if not path.exists():
            raise FileNotFoundError(f"Required selected artifact missing: {path}")
    remove = []
    selected_weights = {ROOT / "outputs/ckpt" / name for name in WEIGHTS}
    for directory in sorted((ROOT / "outputs/ckpt").iterdir()):
        selected = [path for path in selected_weights if path.parent == directory]
        if selected:
            remove += [path for path in directory.iterdir() if path not in selected]
        else:
            remove.append(directory)
    remove += [path for path in sorted((ROOT / "outputs/exp").iterdir())
               if path not in keep]
    for dataset, run in DATASETS.items():
        remove += [path for path in sorted((ROOT / "demos" / dataset).iterdir())
                   if path.name != run]
    # Resolve every candidate against the explicit mutable artifact roots before deleting anything.
    allowed = (ROOT / "outputs/ckpt", ROOT / "outputs/exp",
               ROOT / "demos/PickCube-obstacle", ROOT / "demos/PlaceSphere-obstacle")
    for path in remove:
        if path.is_symlink() or not any(path.resolve().is_relative_to(base) for base in allowed):
            raise ValueError(f"Deletion outside artifact scope: {path}")
    return keep, remove


def verify_results():
    results = {}
    for task, name in EVALUATIONS.items():
        run = ROOT / "outputs/exp" / name
        summary = json.loads((run / "paired_summary.json").read_text())
        if task == "PickCube":
            summary = json.loads((run / "termination_correction.json").read_text())[
                "corrected_policy_summaries"]
        elif task == "PlaceSphere":
            summary = summary["policies"]
        results[task] = {}
        for policy in ("base_dp", "constrain_dp"):
            records = [json.loads(path.read_text())
                       for path in sorted((run / policy).glob("episode_*.json"))]
            if len(records) != 100 or any(row["status"] != "evaluated" for row in records):
                raise RuntimeError(f"Incomplete final evaluation: {task}/{policy}")
            flags = [row["result"] for row in records]
            if task == "StackCube":
                names = ("stack_success", "safe_success", "obstacle_collision", "obstacle_crossing")
                keys = ("stack_success_count", "safe_success_count", "contact_count", "crossing_count")
            else:
                names = ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
                keys = ("task_success_count", "safe_success_count", "obstacle_contact_count",
                        "xy_crossing_count" if task == "PickCube" else "xy_forbidden_crossing_count")
            totals = [sum(bool(row[key]) for row in flags) for key in names]
            if totals != [summary[policy][key] for key in keys]:
                raise RuntimeError(f"Summary differs from episode records: {task}/{policy}")
            video = run / policy / "representative.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise RuntimeError(f"Missing final video: {video}")
            results[task][policy] = dict(zip(
                ("task_success", "safe_success", "contact", "xy_crossing"), totals))
            results[task][policy].update(denominator=100, runtime_errors=0,
                                         video_sha256=identity(video)["sha256"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report:
        if args.apply:
            parser.error("--verify-report cannot be combined with --apply")
        report = json.loads(args.verify_report.read_text())
        if report["status"] != "complete":
            raise ValueError("Cleanup report is not complete")
        migration_file = args.verify_report.parent / "path_migrations.json"
        migrations = {}
        if migration_file.exists():
            entries = json.loads(migration_file.read_text())["files"]
            migrations = {entry["from"]: entry for entry in entries}
        for before in report["retained_file_identities"]:
            migration = migrations.get(before["path"])
            current_path = before["path"]
            if migration:
                if migration["sha256"] != before["sha256"]:
                    raise ValueError(f"Migration identity differs: {current_path}")
                current_path = migration["to"]
            after = identity(ROOT / current_path)
            after["path"] = before["path"]
            if after != before:
                raise RuntimeError(f"Retained artifact differs: {before['path']}")
        for name in report["delete_roots"]:
            if (ROOT / name).exists():
                raise RuntimeError(f"Deleted artifact still present: {name}")
        verification = {"status": "verified",
                        "retained_files": len(report["retained_file_identities"]),
                        "deleted_entries_absent": len(report["delete_roots"]),
                        "path_migrations": list(migrations.values()),
                        "episode_records_checked": 600, "results": verify_results()}
        destination = args.verify_report.parent / "final_verification.json"
        destination.write_text(json.dumps(verification, indent=2) + "\n")
        print(json.dumps(verification, indent=2))
        return
    keep, remove = build_plan()
    removed_files = [identity(item) for path in remove for item in files(path)]
    kept_files = [identity(item) for path in keep for item in files(path)]
    summary = {
        "selected_weight_files": len(WEIGHTS),
        "selected_experiment_directories": len(EXPERIMENTS),
        "remove_entries": len(remove),
        "remove_files": len(removed_files),
        "remove_bytes": sum(item["bytes"] for item in removed_files),
        "kept_files": len(kept_files),
        "kept_bytes": sum(item["bytes"] for item in kept_files),
    }
    if not args.apply:
        print(json.dumps({"dry_run": True, **summary,
                          "remove": [relative(path) for path in remove]}, indent=2))
        return
    if args.report_dir is None:
        parser.error("--apply requires --report-dir")
    report_dir = args.report_dir.resolve()
    if report_dir.parent != ROOT / "outputs/exp":
        raise ValueError("Report must be a new direct child of outputs/exp")
    if not report_dir.name.endswith("_final_artifact_cleanup"):
        raise ValueError("Report name must end with _final_artifact_cleanup")
    report_dir.mkdir(exist_ok=False)
    shutil.copyfile(ROOT / "examples/baselines/constrain/OUTPUTS.md",
                    report_dir / "previous_OUTPUTS.md")
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "planned", "selection": "Final valid paired run per task; safe success primary, retain Base DP comparator",
        "summary": summary, "retained_roots": [relative(path) for path in keep],
        "delete_roots": [relative(path) for path in remove],
        "retained_file_identities": kept_files, "deleted_file_identities": removed_files,
        "completed_deletions": [],
        "historical_references": "Deleted paths in original manifests remain historical; this inventory records their identities.",
    }
    manifest = report_dir / "cleanup_manifest.json"

    def save():
        temporary = manifest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(manifest)

    save()
    try:
        for path in remove:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            report["completed_deletions"].append(relative(path))
            save()
        for before in kept_files:
            if identity(ROOT / before["path"]) != before:
                raise RuntimeError(f"Selected artifact changed: {before['path']}")
        remaining = sorted(relative(path) for path in (ROOT / "outputs/ckpt").rglob("*")
                           if path.is_file())
        expected = sorted("outputs/ckpt/" + name for name in WEIGHTS)
        if remaining != expected:
            raise RuntimeError("Remaining checkpoint files differ from selected six weights")
        report.update(status="complete", retained_hashes_verified=len(kept_files),
                      remaining_checkpoint_files=remaining)
        save()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    print(json.dumps({"report": relative(manifest), **summary,
                      "retained_hashes_verified": len(kept_files)}, indent=2))


if __name__ == "__main__":
    main()
