"""Preregister and run one paired R39 residual-guidance PickCube trial."""

import argparse
import hashlib
import json
import shutil
import signal
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_residual_guidance import (
    GUIDANCE_KIND, REGISTERED_SCALE, ResidualGuidanceAdapter,
)

ROOT = Path(".")
EXP = Path("outputs/exp/20261029_000001_pickcube_residual_guidance_r39")
R38 = Path("outputs/exp/20261028_221700_pickcube_position_units_r38")
BASE_CKPT = Path("outputs/ckpt/20261028_221700_pickcube_position_units_r38_base/final.pt")
CONTROL_CKPT = Path("outputs/ckpt/20261028_221700_pickcube_position_units_r38_constrain/final.pt")
STATS = R38 / "position_encoding.json"
SCENES = Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes")
GATE = {"task_gte": 27, "safe_gte": 26, "contact_lte": 3, "xy_lte": 3}
SOURCE_FILES = (
    "examples/baselines/constrain/pickcube_residual_guidance.py",
    "examples/baselines/constrain/pickcube_residual_guidance_r39.py",
    "examples/baselines/constrain/pickcube_position_units.py",
    "examples/baselines/constrain/pickcube_position_units_round38.py",
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/pickcube_base_dp.py",
    "examples/baselines/constrain/pickcube_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/pickcube_learning_pilot.py",
    "examples/baselines/constrain/pickcube_transport_detour.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "mani_skill/envs/tasks/tabletop/pick_cube.py",
)
EPISODE_WALL_SECONDS = 600
OVERALL_WALL_SECONDS = 36000


class EpisodeTimeout(TimeoutError):
    pass


def _alarm(_signum, _frame):
    raise EpisodeTimeout(f"episode exceeded {EPISODE_WALL_SECONDS}s")


def _json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _source_identity():
    files = {name: sha256_file(name) for name in SOURCE_FILES}
    return {"files": files, "files_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def register():
    """Pin inputs, source, and R38 analysis rows before any R39 rollout."""
    from examples.baselines.constrain.pickcube_position_units_round38 import _install_profile
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet

    _install_profile()
    EXP.mkdir(parents=True, exist_ok=False)
    snapshots = EXP / "r38_snapshot"
    snapshots.mkdir()
    for policy in ("base_dp", "constrain_dp"):
        src = R38 / "paired" / policy
        dst = snapshots / policy
        dst.mkdir()
        for index in range(30):
            shutil.copyfile(src / f"episode_{index:03d}.json", dst / f"episode_{index:03d}.json")
    catalog = TransportDetourSceneSet(SCENES)
    if len(catalog) != 30:
        raise ValueError("R39 requires the existing frozen 30-scene R36 development set")
    prereg = {
        "round": 39, "status": "registered_before_native_evaluation",
        "hypothesis": "epsilon-space learned control residual extrapolation may strengthen detour/safety behavior",
        "formal_performance_claim": False,
        "guidance": {"kind": GUIDANCE_KIND, "scale": REGISTERED_SCALE,
                     "formula": "eps_base + 1.5 * (eps_control - eps_base)",
                     "base": "same Control checkpoint frozen base, enabled=False",
                     "control": "same Control checkpoint EMA, enabled=True",
                     "same_sample_timestep_global_condition_raw_hint": True,
                     "not_classifier_free_guidance": True,
                     "base_comparator_guidance": None},
        "inputs": {
            "base_checkpoint": str(BASE_CKPT), "base_checkpoint_sha256": sha256_file(BASE_CKPT),
            "control_checkpoint": str(CONTROL_CKPT), "control_checkpoint_sha256": sha256_file(CONTROL_CKPT),
            "normalizer": str(STATS), "normalizer_file_sha256": sha256_file(STATS),
            "normalizer_profile_sha256": "8bcb1df3252e86b085d51507f72ce4d7722eadea7cc46d676d6da7d33b3d7fef",
            "frozen_h5_sha256": "0c74b88eebe165a03592f7c64d186d1cc550d9fa013f51a2b54d41c349349f4f",
            "frozen_manifest_sha256": "6b9d2c75aaaef02753c002eef817e8325478d2160af34bdf800e604c76e5c486",
            "validation_sha256": "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f",
            "scene_manifest_sha256": sha256_file(SCENES / "manifest.json"),
            "scene_h5_sha256": sha256_file(SCENES / "scenes.h5"),
            "r38_base_episode_sha256": [sha256_file(snapshots / "base_dp" / f"episode_{i:03d}.json") for i in range(30)],
            "r38_control_episode_sha256": [sha256_file(snapshots / "constrain_dp" / f"episode_{i:03d}.json") for i in range(30)],
        },
        "protocol": {"episodes_per_policy": 30, "control_steps": 500,
                     "wall_seconds_per_episode": EPISODE_WALL_SECONDS,
                     "overall_wall_seconds": OVERALL_WALL_SECONDS,
                     "policy_order": ["base_dp", "guided_control"],
                     "sampler": {"scheduler": "DDPMScheduler", "train_timesteps": 100,
                                 "inference_timesteps": 100, "beta_schedule": "squaredcos_cap_v2",
                                 "prediction_type": "epsilon", "clip_sample": True,
                                 "slice": [1, 9], "execute_stride": 8},
                     "seed_rule": "torch.manual_seed(original_scene_seed) after model setup per episode",
                     "action_path": "native 7d pd_ee_delta_pose identity; original scheduler clipping only",
                     "gate": GATE},
        "source_identity_before_native": _source_identity(),
        "r38_analysis_snapshot_sha256": sha256_file(R38 / "paired" / "analysis.json"),
        "runtime_seconds_cap_choice": "600 seconds each policy episode; fixed before execution due two epsilon predictions per denoising step",
    }
    prereg["registration_sha256"] = hashlib.sha256(
        json.dumps(prereg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _json(EXP / "preregistration.json", prereg)
    return prereg


def _exact_base_match(new_row, old_row):
    keys = ("scene_id", "seed", "restored_state_sha256", "restored_controller_state_sha256",
            "restored_observation_sha256", "first_grasp_step", "first_lift_step",
            "first_transport_step", "first_native_success_step", "first_contact",
            "step_trace", "result")
    return all(new_row.get(k) == old_row.get(k) for k in keys)


def _transitions(old_rows, new_rows, key, violation=False):
    old = [bool(r.get("result", {}).get(key, False)) for r in old_rows]
    new = [bool(r.get("result", {}).get(key, False)) for r in new_rows]
    table = {"both": sum(a and b for a, b in zip(old, new)),
             "neither": sum(not a and not b for a, b in zip(old, new)),
             "old_marginal": sum(old), "new_marginal": sum(new)}
    if violation:
        table.update(cleared_old_only=sum(a and not b for a, b in zip(old, new)),
                     introduced_new_only=sum(not a and b for a, b in zip(old, new)))
        cells = ("cleared_old_only", "introduced_new_only", "both", "neither")
    else:
        table.update(improved_new_only=sum(not a and b for a, b in zip(old, new)),
                     regressed_old_only=sum(a and not b for a, b in zip(old, new)))
        cells = ("improved_new_only", "regressed_old_only", "both", "neither")
    if sum(table[k] for k in cells) != len(old):
        raise ValueError("paired marginal transition cells do not sum to denominator")
    return table


def evaluate(registration_path, device="cuda"):
    from examples.baselines.constrain import pickcube_position_units_round38 as profile
    from examples.baselines.constrain import pickcube_zscore_round37 as engine
    from examples.baselines.constrain.pickcube_learning_pilot import run_pilot_episode, summarize_pilot
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet

    profile._install_profile()
    reg = json.loads(Path(registration_path).read_text())
    check = dict(reg); expected = check.pop("registration_sha256", None)
    if expected != hashlib.sha256(json.dumps(check, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise ValueError("R39 preregistration checksum mismatch")
    if reg["source_identity_before_native"] != _source_identity():
        raise ValueError("registered R39 inference/evaluator source has changed")
    paths = reg["inputs"]
    for path, sha_key in ((BASE_CKPT, "base_checkpoint_sha256"),
                          (CONTROL_CKPT, "control_checkpoint_sha256"), (STATS, "normalizer_file_sha256"),
                          (SCENES / "manifest.json", "scene_manifest_sha256"),
                          (SCENES / "scenes.h5", "scene_h5_sha256")):
        if sha256_file(path) != paths[sha_key]:
            raise ValueError(f"R39 frozen input hash mismatch: {path}")
    scene_set = TransportDetourSceneSet(SCENES)
    base = engine.ZScoreBaseAdapter(BASE_CKPT, STATS, device)
    control = ResidualGuidanceAdapter(engine.ZScoreConstrainAdapter(
        CONTROL_CKPT, STATS, BASE_CKPT, device, control_enabled=True), REGISTERED_SCALE)
    output = EXP / "paired"
    output.mkdir(exist_ok=False)
    started_all = time.monotonic()
    rows = {}
    for policy, adapter in (("base_dp", base), ("guided_control", control)):
        directory = output / policy
        directory.mkdir()
        rows[policy] = []
        for index in range(30):
            if time.monotonic() - started_all > OVERALL_WALL_SECONDS:
                raise TimeoutError("R39 overall evaluation budget exceeded")
            scene = scene_set.load(index)
            start = time.monotonic()
            previous = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_SECONDS)
            try:
                row = run_pilot_episode(adapter, scene_set, index, device, max_steps=500)
            except Exception as exc:
                row = {"scene_index": index, "scene_id": scene["metadata"]["scene_id"],
                       "seed": int(scene["metadata"]["seed"]), "status": "runtime_error",
                       "error": f"{type(exc).__name__}: {exc}"}
                if isinstance(exc, EpisodeTimeout):
                    row["error_kind"] = "wall_timeout"
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            row["wall_seconds"] = round(time.monotonic() - start, 3)
            row["checkpoint_sha256"] = adapter.metadata["checkpoint_sha256"]
            row["guidance_kind"] = adapter.metadata.get("guidance_kind")
            row["guidance_scale"] = adapter.metadata.get("guidance_scale")
            row["guidance_operation"] = adapter.metadata.get("guidance_semantics")
            _json(directory / f"episode_{index:03d}.json", row)
            rows[policy].append(row)
        _json(directory / "summary.json", summarize_pilot(rows[policy], 30))
        if policy == "base_dp":
            old = [json.loads((EXP / "r38_snapshot/base_dp" / f"episode_{i:03d}.json").read_text())
                   for i in range(30)]
            mismatches = [i for i, (new, prior) in enumerate(zip(rows[policy], old))
                          if new.get("status") != "evaluated" or not _exact_base_match(new, prior)]
            if mismatches:
                _json(output / "base_identity_failure.json", {"mismatches": mismatches,
                    "control_not_run": True, "reason": "R39 base no longer matches R38 scene-by-scene"})
                raise RuntimeError(f"base identity mismatch before guided evaluation at indexes {mismatches}")

    snapshots = {policy: [json.loads((EXP / "r38_snapshot" / policy / f"episode_{i:03d}.json").read_text())
                          for i in range(30)] for policy in ("base_dp", "constrain_dp")}
    old_control = snapshots["constrain_dp"]
    summaries = {policy: summarize_pilot(value, 30) for policy, value in rows.items()}
    paired = all(a.get("scene_id") == b.get("scene_id")
                 and a.get("restored_state_sha256") == b.get("restored_state_sha256")
                 and a.get("restored_controller_state_sha256") == b.get("restored_controller_state_sha256")
                 and a.get("restored_observation_sha256") == b.get("restored_observation_sha256")
                 for a, b in zip(rows["base_dp"], rows["guided_control"]))
    metrics = {"task_success": "native_success", "safe_success": "safe_success",
               "obstacle_contact": "obstacle_contact", "xy_forbidden_crossing": "xy_forbidden_crossing"}
    transitions = {name: _transitions(old_control, rows["guided_control"], key,
                                      violation=name in ("obstacle_contact", "xy_forbidden_crossing"))
                  for name, key in metrics.items()}
    control_summary = summaries["guided_control"]
    gate = {"task": control_summary["task_success_count"] >= 27,
            "safe": control_summary["safe_success_count"] >= 26,
            "contact": control_summary["obstacle_contact_count"] <= 3,
            "xy": control_summary["xy_crossing_count"] <= 3,
            "complete": all(x["complete"] for x in summaries.values()), "paired": paired}
    analysis = {"complete": gate["complete"], "paired_scene_identity": paired,
                "policy_metrics": summaries, "r38_control_vs_r39_guided": transitions,
                "base_exactly_matches_r38": True, "gate": {"criteria": gate, "passed": all(gate.values())},
                "formal_performance_claim": False,
                "guidance_interpretation": "exploratory learned conditional residual extrapolation; no guaranteed safety/route preservation",
                "wall_seconds": {k: [r.get("wall_seconds") for r in v] for k, v in rows.items()},
                "wall_summary_seconds": {k: {"min": min(x), "median": statistics.median(x),
                    "p90": float(np.percentile(x, 90)), "max": max(x)}
                    for k, v in rows.items() if (x := [r["wall_seconds"] for r in v])},
                "r39_source_identity_after_native": _source_identity(),
                "overall_elapsed_seconds": time.monotonic() - started_all}
    _json(output / "analysis.json", analysis)
    _json(output / "status.json", {"status": "complete" if gate["complete"] else "complete_with_errors",
                                     "gate_passed": all(gate.values())})
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "evaluate"))
    parser.add_argument("--registration", type=Path, default=EXP / "preregistration.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    print(json.dumps(register() if args.command == "register" else evaluate(args.registration, args.device),
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
