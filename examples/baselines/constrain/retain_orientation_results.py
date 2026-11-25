"""Retain the best Peg/Plug orientation results and render their experiment videos.

Selection follows the completed comparisons, not a new test search. Weights are
hardlinked so the retained file keeps the original bytes. Plug videos replay the
recorded native actions; they do not resample the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RETAINED = ROOT / "outputs/ckpt/retained_orientation"
VIDEO = ROOT / "outputs/video"
PEG_COMPARISON = ROOT / "outputs/exp/20261118_061124_peg_orientation_random500_comparison"
PLUG_HELDOUT = ROOT / "outputs/exp/20261123_070724_plug_random500_heldout"
PLUG_TABLE = ROOT / "outputs/exp/20261123_070724_plug_random500_heldout_envelope/episode_table.jsonl"

WEIGHTS = {
    "PegInsertionSide/BaseDP": {
        "source": "outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt",
        "sha256": "067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea",
        "result": "random 3/100, fixed 52/100 in the paired random-500 comparison",
    },
    "PegInsertionSide/ConstrainNet": {
        "source": "outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_25000.pt",
        "sha256": "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df",
        "result": "validation-selected 25k; nonblind paired random 23/100 vs Base 3/100, fixed 54/100 vs 52/100",
    },
    "PlugCharger/BaseDP": {
        "source": "outputs/ckpt/20261119_194656_plug_orientation_base/checkpoint_080000.pt",
        "sha256": "ceeb68b4838ced6e6ffa43646bcc3cfc47538469e02264c4bc80a2542a33d130",
        "result": "frozen v1 80k; authorized heldout random 22/100, fixed 53/100",
    },
    "PlugCharger/ConstrainNet": {
        "source": "outputs/ckpt/20261122_042359_plug_orientation_constrain_random500/checkpoint_030000.pt",
        "sha256": "911bd5d7be828ae06ff87ab66e6fbb67cb1a5d77a173668f9dda581acf4746a1",
        "result": "validation-selected 30k; authorized heldout random correct 41/100 vs Base 22/100, fixed 54/100 vs 53/100",
    },
}

PEG_VIDEOS = {
    "PegInsertionSide_BaseDP/random_scene000.mp4": PEG_COMPARISON / "random_test_base/rollouts.mp4",
    "PegInsertionSide_BaseDP/fixed_scene000.mp4": PEG_COMPARISON / "fixed_retention_base/rollouts.mp4",
    "PegInsertionSide_ConstrainNet/random_scene000.mp4": PEG_COMPARISON / "random_test_correct/rollouts.mp4",
    "PegInsertionSide_ConstrainNet/fixed_scene000.mp4": PEG_COMPARISON / "fixed_retention_correct/rollouts.mp4",
    "PegInsertionSide_ConstrainNet/random_fixed_standard_scene000.mp4": PEG_COMPARISON / "random_test_fixed_standard/rollouts.mp4",
    "PegInsertionSide_ConstrainNet/random_shuffled_scene000.mp4": PEG_COMPARISON / "random_test_shuffled/rollouts.mp4",
}

PLUG_COHORT_FILES = {
    ("random_final_test", "base"): "random_final_test_base",
    ("random_final_test", "constrain_correct"): "random_final_test_constrain_correct",
    ("random_final_test", "constrain_fixed_standard"): "random_final_test_constrain_fixed_standard",
    ("random_final_test", "constrain_shuffled"): "random_final_test_constrain_shuffled",
    ("fixed_base_test", "base"): "fixed_base_test_base",
    ("fixed_base_test", "constrain_correct"): "fixed_base_test_constrain_correct",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def representative_index(rows):
    """Lowest successful scene index, otherwise the lowest index. No visual selection."""
    return indices_with_outcome(rows, True, 1)[0] if any(row.get("success") is True for row in rows) else indices_with_outcome(rows, False, 1)[0]


def indices_with_outcome(rows, success, count):
    """Lowest scene indices with the requested recorded outcome. No visual selection."""
    def scene_index(row):
        return int(row["index"] if "index" in row else row["scene_index"])

    ordered = sorted(rows, key=scene_index)
    chosen = [scene_index(row) for row in ordered if bool(row.get("success")) is bool(success)]
    if len(chosen) < count:
        raise ValueError(f"need {count} scenes with success={success}, found {len(chosen)}")
    return chosen[:count]


def recorded_actions(path):
    actions = []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("event") == "control":
            action = np.asarray(event["action"], dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"invalid recorded action in {path}")
            actions.append(action)
    if not actions:
        raise ValueError(f"no recorded actions in {path}")
    return actions


def retain_weights(root=RETAINED):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    retained = {}
    for name, spec in WEIGHTS.items():
        source = ROOT / spec["source"]
        if sha256_file(source) != spec["sha256"]:
            raise ValueError(f"best-weight identity mismatch: {source}")
        destination = root / name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != spec["sha256"]:
                raise ValueError(f"retained weight already differs: {destination}")
        else:
            os.link(source, destination)
        retained[name] = {"source": spec["source"], "retained": str(destination.relative_to(ROOT)),
                          "sha256": spec["sha256"], "result": spec["result"]}
    return retained


def copy_peg_videos(video_root=VIDEO):
    copied = {}
    for relative, source in PEG_VIDEOS.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = Path(video_root) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        copied[relative] = {"source": str(source.relative_to(ROOT)), "sha256": sha256_file(destination)}
    return copied


def render_plug_videos(video_root=VIDEO, table_path=PLUG_TABLE, journal_root=PLUG_HELDOUT):
    import gymnasium as gym
    import imageio.v2 as imageio

    from examples.baselines.constrain.plug_orientation import ENV_ID
    from examples.baselines.constrain.plug_orientation_scenes import PlugSceneCatalog, restore_plug_scene

    rows = [json.loads(line) for line in Path(table_path).read_text().splitlines() if line]
    catalogs = {
        "random_final_test": PlugSceneCatalog(ROOT / "outputs/exp/20261119_184307_plug_orientation_catalogs/random_final_test"),
        "fixed_base_test": PlugSceneCatalog(ROOT / "outputs/exp/20261119_184307_plug_orientation_catalogs/fixed_base_test"),
    }
    rendered = {}
    env = gym.make(ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", render_mode="rgb_array", max_episode_steps=1000)
    try:
        for (split, cohort), prefix in PLUG_COHORT_FILES.items():
            cohort_rows = [row for row in rows if row["split"] == split and row["cohort"] == cohort]
            index = representative_index(cohort_rows)
            chosen = next(row for row in cohort_rows if int(row["index"]) == index)
            actions = recorded_actions(journal_root / f"{prefix}_episode_{index:03d}.jsonl")
            if len(actions) != int(chosen["controls"]):
                raise ValueError(f"{prefix} scene {index}: action count differs from the audited result")
            method = "BaseDP" if cohort == "base" else "ConstrainNet"
            label = "success" if chosen["success"] else "failure"
            condition = "" if cohort in {"base", "constrain_correct"} else f"_{cohort.removeprefix('constrain_')}"
            relative = f"PlugCharger_{method}/{split}_{label}_scene{index:03d}{condition}.mp4"
            destination = Path(video_root) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.mp4")
            restore_plug_scene(env, catalogs[split].load(index))
            with imageio.get_writer(str(temporary), fps=20, codec="libx264", macro_block_size=1) as writer:
                writer.append_data(_frame(env))
                final_success = False
                for action in actions:
                    _obs, _reward, _terminated, _truncated, info = env.step(action)
                    final_success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    writer.append_data(_frame(env))
            if final_success != bool(chosen["success"]):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"{relative}: replay success {final_success} differs from recorded {chosen['success']}")
            temporary.replace(destination)
            rendered[relative] = {"scene_index": index, "success": bool(chosen["success"]),
                                  "controls": len(actions), "sha256": sha256_file(destination)}
    finally:
        env.close()
    return rendered


def _frame(env):
    frame = env.render()
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    return np.asarray(frame, dtype=np.uint8)


def replay_plug_indices(split, cohort, indices, video_root=VIDEO, table_path=PLUG_TABLE, journal_root=PLUG_HELDOUT, limit=3):
    import gymnasium as gym
    import imageio.v2 as imageio

    from examples.baselines.constrain.plug_orientation import ENV_ID
    from examples.baselines.constrain.plug_orientation_scenes import PlugSceneCatalog, restore_plug_scene

    prefix = PLUG_COHORT_FILES[(split, cohort)]
    rows = [json.loads(line) for line in Path(table_path).read_text().splitlines() if line]
    by_index = {int(row["index"]): row for row in rows if row["split"] == split and row["cohort"] == cohort}
    catalog = PlugSceneCatalog(ROOT / "outputs/exp/20261119_184307_plug_orientation_catalogs" / split)
    rendered = {}
    env = gym.make(ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                   sim_backend="physx_cpu", render_mode="rgb_array", max_episode_steps=1000)
    try:
        for index in indices:
            chosen = by_index[index]
            actions = recorded_actions(journal_root / f"{prefix}_episode_{index:03d}.jsonl")
            if len(actions) != int(chosen["controls"]):
                raise ValueError(f"{prefix} scene {index}: action count differs from the audited result")
            method = "BaseDP" if cohort == "base" else "ConstrainNet"
            label = "success" if chosen["success"] else "failure"
            condition = "" if cohort in {"base", "constrain_correct"} else f"_{cohort.removeprefix('constrain_')}"
            relative = f"PlugCharger_{method}/{split}_{label}_scene{index:03d}{condition}.mp4"
            destination = Path(video_root) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp.mp4")
            restore_plug_scene(env, catalog.load(index))
            with imageio.get_writer(str(temporary), fps=20, codec="libx264", macro_block_size=1) as writer:
                writer.append_data(_frame(env))
                final_success = False
                for action in actions:
                    _obs, _reward, _terminated, _truncated, info = env.step(action)
                    final_success = bool(np.asarray(info.get("success", False)).reshape(-1)[0])
                    writer.append_data(_frame(env))
            if destination.exists():
                rendered[relative] = {"scene_index": index, "success": bool(chosen["success"]),
                                      "controls": len(actions), "sha256": sha256_file(destination),
                                      "source": "existing_recorded_replay"}
                continue
            if final_success != bool(chosen["success"]):
                temporary.unlink(missing_ok=True)
                rendered[f"{relative}#replay_mismatch"] = {
                    "scene_index": index, "recorded_success": bool(chosen["success"]),
                    "replay_success": final_success, "source": "replay_mismatch_discarded"}
                continue
            temporary.replace(destination)
            rendered[relative] = {"scene_index": index, "success": bool(chosen["success"]),
                                  "controls": len(actions), "sha256": sha256_file(destination),
                                  "source": "recorded_action_replay"}
            if sum(item.get("source") != "replay_mismatch_discarded" for item in rendered.values()) >= limit:
                break
    finally:
        env.close()
    return rendered


def render_peg_outcome_videos(video_root=VIDEO, work_root=None, count=3):
    """Render additional Peg videos with the original checkpoint, catalog, and seed.

    Peg journals did not retain per-control actions, so these are same-seed rerenders,
    not exact replays. The returned record states whether the new outcome matches.
    """
    from examples.baselines.constrain.peg_orientation_base_dp import evaluate
    from examples.baselines.constrain.peg_orientation_constrain import PegConstrainAdapter

    catalogs = {
        "random": ROOT / "outputs/exp/20261111_172604_peg_orientation_adaptation/random_test_scenes",
        "fixed": ROOT / "outputs/exp/20261111_172604_peg_orientation_adaptation/fixed_retention_scenes",
    }
    cohorts = {
        ("random", "base", False): (PEG_COMPARISON / "random_test_base/summary.json", True),
        ("random", "constrain", True): (PEG_COMPARISON / "random_test_correct/summary.json", False),
        ("fixed", "base", False): (PEG_COMPARISON / "fixed_retention_base/summary.json", True),
        ("fixed", "constrain", True): (PEG_COMPARISON / "fixed_retention_correct/summary.json", False),
    }
    work_root = Path(work_root or ROOT / "outputs/exp/peg_outcome_video_rerenders")
    rendered = {}
    for (split, role, want_success), (summary_path, base_eval) in cohorts.items():
        summary = json.loads(summary_path.read_text())
        indices = indices_with_outcome(summary["records"], want_success, count)
        for index in indices:
            output = work_root / f"{split}_{role}_scene{index:03d}"
            if output.exists():
                raise FileExistsError(output)
            checkpoint = ROOT / WEIGHTS["PegInsertionSide/BaseDP" if base_eval else "PegInsertionSide/ConstrainNet"]["source"]
            if base_eval:
                evaluate(checkpoint, catalogs[split], output, device="cuda", video=True,
                         max_episode_steps=300, env_time_limit=300, seed=1, scene_indices=[index])
            else:
                evaluate(checkpoint, catalogs[split], output, device="cuda", video=True,
                         max_episode_steps=300, env_time_limit=300, seed=1, scene_indices=[index],
                         policy_factory=lambda task, checkpoint=checkpoint: PegConstrainAdapter(
                             checkpoint, device="cuda", task=task, hint_mode="correct"),
                         hint_mode="correct")
            record = json.loads((output / f"episode_{index:03d}.json").read_text())
            original = next(row for row in summary["records"] if int(row["scene_index"]) == index)
            method = "BaseDP" if base_eval else "ConstrainNet"
            label = "failure" if not want_success else "success"
            relative = f"PegInsertionSide_{method}/{split}_{label}_scene{index:03d}.mp4"
            destination = Path(video_root) / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / "rollouts.mp4", destination)
            rendered[relative] = {
                "scene_index": index,
                "requested_success": want_success,
                "rerender_success": bool(record["success"]),
                "original_success": bool(original["success"]),
                "outcome_matches_original": bool(record["success"]) is bool(original["success"]),
                "sha256": sha256_file(destination),
                "source": "same_seed_cuda_rerender",
            }
    return rendered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--outcome-videos", action="store_true")
    parser.add_argument("--outcome-count", type=int, default=3)
    args = parser.parse_args(argv)
    if args.outcome_videos:
        record = Path(args.selection)
        if record.exists():
            raise FileExistsError(record)
        payload = {"format": "orientation-outcome-videos-v1", "plug_recorded_replays": {}}
        if args.outcome_count:
            table = [json.loads(line) for line in PLUG_TABLE.read_text().splitlines() if line]
            for split, cohort, success in (
                ("random_final_test", "base", False),
                ("random_final_test", "constrain_correct", True),
                ("fixed_base_test", "base", False),
                ("fixed_base_test", "constrain_correct", True),
            ):
                rows = [row for row in table if row["split"] == split and row["cohort"] == cohort]
                pool = indices_with_outcome(rows, success, min(8, sum(bool(row.get("success")) is success for row in rows)))
                payload["plug_recorded_replays"].update(replay_plug_indices(
                    split, cohort, pool, limit=args.outcome_count))
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(payload, indent=2) + "\n")
        return payload
    selection = Path(args.selection)
    if selection.exists():
        raise FileExistsError(selection)
    payload = {"format": "orientation-best-results-v1", "weights": retain_weights(),
               "peg_recorded_scene0_videos": copy_peg_videos()}
    if args.videos:
        payload["plug_replay_videos"] = render_plug_videos()
    selection.parent.mkdir(parents=True, exist_ok=True)
    selection.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    main()
