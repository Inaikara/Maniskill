"""Single deterministic zero-command hold control on the frozen near-hole bank."""

import hashlib
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from examples.baselines.constrain import peg_orientation_near_hole_eval as study
from examples.baselines.constrain.peg_orientation_base_dp import _video_frame


CONTROL_CAP, STAGE_SECONDS, WHOLE_SECONDS = 13000, 900, 1800
SUFFIX_CAP = 200


class MinimalConstantHoldAdapter:
    """Native 39d history adapter returning only zero pose deltas and closed grip."""

    metadata = {"kind": "model-free", "action": [0, 0, 0, 0, 0, 0, -1]}

    def get_action(self, history):
        if np.asarray(history).shape != (1, 2, 39):
            raise ValueError("hold adapter requires history shape (1, 2, 39)")
        actions = np.zeros((1, 8, 7), dtype=np.float32)
        actions[:, :, 6] = -1
        return actions


class _NativeBudget(list):
    def __init__(self, started):
        super().__init__([0, started])
        self.stage_started = started

    def __getitem__(self, index):
        if index == 0 and (super().__getitem__(0) >= CONTROL_CAP or
                           time.monotonic() - self.stage_started >= STAGE_SECONDS):
            raise TimeoutError("hold native-control/stage deadline exhausted before action")
        return super().__getitem__(index)


def _hash(path):
    return study.sha256(path)


def _bank_paths(split):
    return study.BANK / f"{split}_bank" / ("branches_reconciled.jsonl" if split == "random" else "branches.jsonl")


def prepare_output_directory(output):
    """Accept only a precreated workspace, never an earlier experiment attempt."""
    output = Path(output).resolve()
    if not output.exists():
        output.mkdir(parents=True)
    if not output.is_dir():
        raise NotADirectoryError(output)
    prior_artifacts = (
        "preregistration.json", "results.json", "arms.jsonl", "failure.json",
        "setup_failure.json", "source_archive", "source_closure.json",
        "outer_exit.json", "outer_timing.json", "post_execution_audit.json",
        "model_metadata.json",
    )
    found = [name for name in prior_artifacts if (output / name).exists()]
    found.extend(path.name for path in output.glob("*_hold.mp4"))
    if found:
        raise FileExistsError(f"refusing to overwrite prior run artifact(s): {sorted(found)}")
    if not (output / "attempt_001.sh").is_file():
        raise FileNotFoundError("frozen attempt_001.sh wrapper must exist before runner starts")
    return output


def run_hold(output):
    output = prepare_output_directory(output)
    started_utc, started = datetime.now(timezone.utc).isoformat(), time.monotonic()
    rows, groups, envs = [], {}, []
    banks, catalogs, sources = {}, {}, {}
    for split in ("random", "fixed"):
        catalogs[split] = study.PegSceneCatalog(study.CATALOG_ROOT /
            ("validation_random_scenes" if split == "random" else "validation_fixed_scenes"))
        banks[split] = study._bank_rows(split)
        sources[split] = study.DATA_DIRS[split]
    eligible = {split: [r for r in banks[split] if r["eligible"]] for split in banks}
    if [len(eligible["random"]), len(eligible["fixed"])] != [15, 11]:
        raise ValueError("frozen eligible branch counts changed; refusing substitution")
    if any(int(row["prefix_controls"]) > 300 for split in eligible for row in eligible[split]):
        raise ValueError("a frozen prefix exceeds its 300-control budget")
    input_hashes = {
        "bank_files": {str(p.relative_to(study.ROOT)): _hash(p) for p in sorted(study.BANK.rglob("*")) if p.is_file()},
        "catalog_manifests": {s: _hash(catalogs[s].directory / "manifest.json") for s in banks},
        "sources": {s: {str(p.relative_to(study.ROOT)): _hash(p) for p in sorted(sources[s].iterdir()) if p.is_file()} for s in banks},
    }
    if input_hashes["bank_files"][str(_bank_paths("random").relative_to(study.ROOT))] != "ea7976771d7fe02ddf2e14affacdf5926e0dbc948c9a292972e4bba88ab1ba99":
        raise ValueError("random bank identity mismatch")
    if input_hashes["bank_files"][str(_bank_paths("fixed").relative_to(study.ROOT))] != "6e60934f3d91c29f0aa887730ca878b7d9df7e3a3306051fe10b7a2014a8f5a8":
        raise ValueError("fixed bank identity mismatch")
    prereg = {
        "objective": "One model-free deterministic hold control on the exact 26 eligible round-50 near-hole prefixes.",
        "selection": {s: [int(r["catalog_index"]) for r in eligible[s]] for s in eligible},
        "exclusions": {s: [int(r["catalog_index"]) for r in banks[s] if not r["eligible"]]
                       for s in banks},
        "source_provenance": "Round-46 bank; random recovered from journals; fixed amendment retrospective and historical source bytes unavailable.",
        "inputs": input_hashes,
        "historical_models_not_loaded": {
            "base_sha256": "067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea",
            "constrain_sha256": "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df"},
        "action": {"native_mode": "pd_ee_delta_pose", "normalized": [0, 0, 0, 0, 0, 0, -1],
                   "semantics": "zero translation/rotation, gripper closed; controller may actively hold its target"},
        "budget": {"prefix_controls_per_branch": 300, "eligible_prefix_controls_max": 7800,
                   "suffix_controls_per_branch": SUFFIX_CAP,
                   "native_controls_max": CONTROL_CAP, "suffix_time_limit": 600,
                   "stage_seconds": STAGE_SECONDS, "whole_seconds": WHOLE_SECONDS},
        "success": "Official info.success only; near-hole geometry is not success.",
        "videos": "First admitted index in each group, one pass only.",
        "interpretation": "A control for this controller hold only; not a stationary-start or general causal test.",
        "outer_wrapper": {"path": str((output / "attempt_001.sh").relative_to(study.ROOT)),
                          "sha256": _hash(output / "attempt_001.sh")},
        "raw_logs": {"stdout": str((output / "attempt_001.stdout.log").relative_to(study.ROOT)),
                     "stderr": str((output / "attempt_001.stderr.log").relative_to(study.ROOT))},
        "created_utc": started_utc,
    }
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    closure = {rel: _hash(study.ROOT / rel) for rel in study.SOURCE_PATHS}
    closure.update({"examples/baselines/constrain/peg_orientation_near_hole_hold.py": _hash(__file__),
                    "examples/baselines/constrain/tests/test_peg_orientation_near_hole_hold.py":
                    _hash(study.ROOT / "examples/baselines/constrain/tests/test_peg_orientation_near_hole_hold.py")})
    archive = output / "source_archive"
    for rel in closure:
        target = archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((study.ROOT / rel).read_bytes())
    wrapper = output / "attempt_001.sh"
    if not wrapper.is_file():
        raise FileNotFoundError("frozen outer wrapper must exist before runner starts")
    (archive / "outer_attempt_001.sh").write_bytes(wrapper.read_bytes())
    closure["outer_attempt_001.sh"] = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    (output / "source_closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
    (output / "preregistration.sha256").write_text(_hash(output / "preregistration.json") + "  preregistration.json\n")

    budget = _NativeBudget(started)
    adapter = MinimalConstantHoldAdapter()
    try:
        for split in ("random", "fixed"):
            groups[split] = {"eligible": len(eligible[split]), "successes": 0, "rows": []}
            for bank_row in eligible[split]:
                if time.monotonic() - started >= WHOLE_SECONDS:
                    raise TimeoutError("whole hold stage deadline exceeded")
                idx = int(bank_row["catalog_index"])
                scene = catalogs[split].load(idx)
                actions = study._source_actions(sources[split] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5",
                                                bank_row["source_episode_id"])
                pair = {"group": split, "catalog_index": idx, "prefix_controls": int(bank_row["prefix_controls"]),
                        "status": "preparing", "started_utc": datetime.now(timezone.utc).isoformat(),
                        "controls_before": int(budget[0])}
                env = study._make_env(split)
                envs.append(env)
                writer = None
                try:
                    arm = study._prepare_arm(env, scene, actions, int(bank_row["prefix_controls"]), bank_row, budget)
                    arm["prefix_controls"] = int(bank_row["prefix_controls"])
                    pair["prefix_verification"] = {
                        "physical_and_history_errors": arm["errors"],
                        "public_controller": arm["controller"],
                        "head_in_hole_xyz_m": arm["head_in_hole_xyz_m"],
                        "relative_yaw_rad": arm["relative_yaw_rad"],
                        "grasped": arm["grasped"], "inserted": arm["inserted"],
                        "history_sha256": hashlib.sha256(np.asarray(arm["history"]).tobytes()).hexdigest(),
                        "prefix_action_sha256": arm["prefix_action_sha256"],
                    }
                    if idx == eligible[split][0]["catalog_index"]:
                        import imageio.v2 as imageio
                        writer = imageio.get_writer(str(output / f"{split}_index{idx:02d}_hold.mp4"),
                                                    fps=20, codec="libx264", macro_block_size=1)
                    result = study._run_suffix(env, adapter, arm, 1 + idx, budget, writer)
                    if not all(np.array_equal(np.asarray(t["action"][:6]), np.zeros(6)) and t["action"][6] == -1
                               for t in result["trace"]):
                        raise AssertionError("actual outgoing hold action differed from registered vector")
                    pair.update({"status": "complete", "result": result,
                                 "outgoing_action_verified": True,
                                 "ended_utc": datetime.now(timezone.utc).isoformat(),
                                 "controls_after": int(budget[0])})
                    groups[split]["successes"] += int(result["native_success"])
                    groups[split]["rows"].append({"catalog_index": idx,
                        "native_success": result["native_success"], "suffix_controls": result["suffix_controls"],
                        "failure": result["native_failure"], "initial_head_xyz_m": result["initial_head_xyz_m"],
                        "final_head_xyz_m": result["trace"][-1]["head_in_hole_xyz_m"] if result["trace"] else result["initial_head_xyz_m"],
                        "final_grasped": result["trace"][-1]["grasped"] if result["trace"] else True,
                        "controls": result["trace"]})
                    study.append_pair_journal(output / "arms.jsonl", pair)
                    rows.append(pair)
                except Exception as exc:
                    pair.update({"status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)},
                                 "controls_at_failure": int(budget[0])})
                    study.append_pair_journal(output / "arms.jsonl", pair)
                    rows.append(pair)
                    raise
                finally:
                    if writer is not None:
                        writer.close()
                    env.close()
                    envs.remove(env)
        result = {"groups": groups, "branches_completed": len(rows),
                  "native_controls_including_prefix": int(budget[0]),
                  "elapsed_seconds": time.monotonic() - started,
                  "started_utc": started_utc,
                  "interpretation": "26 unique conditional branches, one deterministic hold each; no pooling with three-draw policy rows."}
        (output / "results.json").write_text(json.dumps(study.jsonable(result), indent=2, sort_keys=True) + "\n")
        return result
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc),
            "controls": int(budget[0]), "completed_rows": len(rows), "elapsed_seconds": time.monotonic() - started},
            indent=2, sort_keys=True) + "\n")
        raise
    finally:
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_hold(args.output)


if __name__ == "__main__":
    main()
