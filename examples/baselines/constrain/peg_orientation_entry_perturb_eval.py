"""Paired frozen-policy evaluation on the admitted outside-entry pose-window bank."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import peg_orientation_near_hole_eval as near
from examples.baselines.constrain import peg_orientation_near_hole_perturb as near_perturb

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "outputs/exp/20261119_164500_peg_entry_pose_window_initialization_v2"
SEEDS = (1, 1001, 2001)
CONDITIONS = ("aligned", "lateral_holeY_plus5mm", "lateral_holeY_minus5mm",
              "yaw_worldZ_plus5deg", "yaw_worldZ_minus5deg")
MAX_NATIVE_CONTROLS, STAGE_SECONDS, WHOLE_SECONDS = 223200, 10800, 14400
ENV_TIME_LIMIT = 1000


def bank_rows(path=BANK / "journal.jsonl"):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    admitted = [r for r in rows if r.get("result", {}).get("admitted") is True]
    if len(rows) != 60 or len(admitted) != 60:
        raise ValueError("outside-entry bank must contain exactly 60 admitted cases")
    seen = set()
    for row in admitted:
        result = row["result"]
        key = (row["profile"], row["condition"], int(row["catalog_index"]))
        if row["condition"] not in CONDITIONS or result.get("admission_mode") != "pose_window":
            raise ValueError("bank row is not a pose-window-admitted registered condition")
        if result.get("official_inserted") or not result.get("native_grasped"):
            raise ValueError("bank admission state is inserted or not grasped")
        if not .019 <= float(result["actual_entry_clearance_m"]) <= .021:
            raise ValueError("bank row is not at the registered outside-entry clearance")
        if key in seen:
            raise ValueError("duplicate bank case")
        seen.add(key)
        if (result.get("full_action_sha256") != hashlib.sha256(
                np.concatenate((result["executed_prefix_actions"],
                                result["executed_initializer_actions"])).astype(np.float32).tobytes()).hexdigest()):
            raise ValueError("bank full replay action hash mismatch")
        if int(result["controls"]) != len(result["executed_prefix_actions"]) + len(result["executed_initializer_actions"]):
            raise ValueError("bank full replay action count mismatch")
    return admitted


def _seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _state_matches(expected, actual, tolerance=near.TOLERANCE):
    errors = near._max_error(expected, actual)
    if not errors or not np.isfinite(list(errors.values())).all() or max(errors.values()) > tolerance:
        raise ValueError(f"replayed physical state differs from bank: {errors}")
    return errors


def source_episode_for_scene(profile, catalog_index, catalog):
    """Resolve the source H5 trajectory by the shared reset seed, not catalog position."""
    metadata_path = near.DATA_DIRS[profile] / "trajectory.state.pd_ee_delta_pose.physx_cpu.json"
    metadata = json.loads(metadata_path.read_text())
    scene_seed = int(catalog.manifest["scenes"][catalog_index]["seed"])
    matches = [int(episode["episode_id"]) for episode in metadata["episodes"]
               if int(episode["episode_seed"]) == scene_seed]
    if len(matches) != 1:
        raise ValueError(f"expected one source episode for {profile} catalog scene {catalog_index} seed {scene_seed}, found {len(matches)}")
    return matches[0]


def _prepare_output(output):
    output = Path(output)
    allowed = {"attempt_001.sh", "attempt_001.stdout.log", "attempt_001.stderr.log"}
    if not output.is_dir():
        raise FileNotFoundError("output directory and attempt_001 wrapper must be precreated")
    unexpected = {p.name for p in output.iterdir()} - allowed
    if unexpected or not (output / "attempt_001.sh").is_file():
        raise FileExistsError(f"output directory contains preregistration/results or lacks its wrapper: {sorted(unexpected)}")
    wrapper = output / "attempt_001.sh"
    wrapper_bytes = wrapper.read_bytes()
    if not wrapper_bytes.strip():
        raise ValueError("precreated wrapper is empty")
    archive = output / "source_archive"
    archive.mkdir()
    source = Path(__file__).resolve()
    archived = archive / source.name
    archived.write_bytes(source.read_bytes())
    return {"wrapper_path": str(wrapper), "wrapper_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
            "runner_path": str(source), "runner_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "archived_runner_sha256": hashlib.sha256(archived.read_bytes()).hexdigest()}


def _prepare(env, scene, row, budget, started, *, max_native_controls=MAX_NATIVE_CONTROLS,
             stage_seconds=STAGE_SECONDS):
    result = row["result"]
    obs = near.restore_peg_scene(env, scene)
    history = [np.asarray(obs.detach().cpu() if isinstance(obs, torch.Tensor) else obs).reshape(-1).copy()]
    actions = np.concatenate((result["executed_prefix_actions"], result["executed_initializer_actions"])).astype(np.float32)
    digest = hashlib.sha256()
    for action in actions:
        if budget[0] >= max_native_controls or time.monotonic() - started >= stage_seconds:
            raise TimeoutError("entry evaluation native budget exhausted before replay action")
        digest.update(action.tobytes())
        obs, _, terminated, truncated, info = env.step(action)
        budget[0] += 1
        history.append(np.asarray(obs.detach().cpu() if isinstance(obs, torch.Tensor) else obs).reshape(-1).copy())
        if near._native_success(info):
            raise ValueError("outside-entry prefix became officially inserted")
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            raise ValueError("outside-entry replay terminated before policy handoff")
    if digest.hexdigest() != result["full_action_sha256"]:
        raise ValueError("replayed full action digest differs from bank")
    task = env.unwrapped
    if str(task.control_mode) != result["controller_mode"] or str(task.control_mode) != "pd_ee_delta_pose":
        raise ValueError("replayed controller mode differs from bank or frozen EE mode")
    physical = near.capture_physical_state(task)
    errors = _state_matches(result["physical_state"], physical)
    expected_history = np.asarray(result["last_two_observations"], dtype=np.float64)
    actual_history = np.asarray(history[-2:], dtype=np.float64)
    if actual_history.shape != expected_history.shape or not np.isfinite(actual_history).all() or np.max(np.abs(actual_history - expected_history)) > near.TOLERANCE:
        raise ValueError("replayed latest-two actual observation history differs from bank")
    window = result.get("last_pose_window_candidate") or []
    if len(window) != 11 or not near_perturb.minimal_window_admission(window, result["control_timestep_s"]):
        raise ValueError("bank pose-window provenance failed prospective admission check")
    if not near.capture_physical_state(task) or not near.native_grasped(task):
        raise ValueError("replayed handoff is not grasped")
    if bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any()):
        raise ValueError("replayed outside-entry handoff is already inserted")
    head = task.box_hole_pose.inv() * task.peg_head_pose
    head_xyz = np.asarray(head.p[0].detach().cpu(), dtype=np.float64)
    half_length = float(task.peg_half_sizes[0, 0].detach().cpu())
    clearance = -half_length - float(head_xyz[0])
    if abs(clearance - float(result["actual_entry_clearance_m"])) > near.TOLERANCE or not .019 <= clearance <= .021:
        raise ValueError("replayed entry clearance differs from the registered outside-entry handoff")
    return {"history": actual_history, "physical_state": physical, "errors": errors,
            "controller": near._controller_snapshot(env), "prefix_controls": len(actions),
            "head_in_hole_xyz_m": head_xyz.tolist(),
            "relative_yaw_rad": near._relative_yaw(task)}


def run_comparison(output, *, adapters=None, env_factory=None, catalog_factory=None,
                   source_action_loader=None, bank=None):
    """Run all fixed pairs; injectable dependencies exist for deterministic orchestration tests."""
    output = Path(output)
    setup = _prepare_output(output)
    rows = bank if bank is not None else bank_rows()
    if not rows or (bank is None and len(rows) != 60):
        raise ValueError("expected full 60-case bank")
    started, budget = time.monotonic(), [0, time.monotonic()]
    # Real dependencies are loaded together before any per-pair RNG handoff.
    if adapters is None:
        from examples.baselines.constrain.peg_orientation_base_dp import PegBaseAdapter
        from examples.baselines.constrain.peg_orientation_constrain import PegConstrainAdapter
        adapters = {"base": PegBaseAdapter(near.BASE_CKPT, device="cuda"),
                    "constrain": PegConstrainAdapter(near.CN_CKPT, device="cuda", task=None, record_all_hints=True)}
    if env_factory is None:
        from examples.baselines.constrain.peg_orientation import ENV_ID, RANDOM_ENV_ID
        import gymnasium as gym
        env_factory = lambda profile: gym.make(RANDOM_ENV_ID if profile == "random" else ENV_ID,
            obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
            reconfiguration_freq=1, max_episode_steps=ENV_TIME_LIMIT)
    if catalog_factory is None:
        from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog
        catalog_factory = lambda profile: PegSceneCatalog(near.CATALOG_ROOT / ("validation_random_scenes" if profile == "random" else "validation_fixed_scenes"))
    if source_action_loader is None:
        source_action_loader = lambda profile, episode: near._source_actions(near.DATA_DIRS[profile] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", episode)
    catalogs = {p: catalog_factory(p) for p in ("random", "fixed")}
    journal, completed = output / "pairs.jsonl", []
    all_pairs = [(r, seed_i, seed + int(r["catalog_index"])) for r in rows
                 for seed_i, seed in enumerate(SEEDS)]
    try:
        for row, replicate, seed in all_pairs:
            profile, idx, result = row["profile"], int(row["catalog_index"]), row["result"]
            scene = catalogs[profile].load(idx)
            source_episode = source_episode_for_scene(profile, idx, catalogs[profile])
            source_actions = source_action_loader(profile, source_episode)
            if not np.array_equal(np.asarray(source_actions[:int(result["source_prefix_controls"])], dtype=np.float32),
                                  np.asarray(result["executed_prefix_actions"], dtype=np.float32)):
                raise ValueError("bank source prefix does not match registered source trajectory")
            pair = {"profile": profile, "condition": row["condition"], "catalog_index": idx,
                    "replicate": replicate, "seed": seed, "arms": {}, "status": "preparing"}
            envs = []
            try:
                envs = [env_factory(profile), env_factory(profile)]
                base_arm, cn_arm = (_prepare(e, scene, row, budget, started) for e in envs)
                ctl = near.assert_controller_match(*envs)
                if not np.array_equal(base_arm["history"], cn_arm["history"]):
                    raise ValueError("paired actual histories differ before inference")
                pair["handoff"] = {"controller": ctl, "base_errors": base_arm["errors"],
                    "constrain_errors": cn_arm["errors"], "history": base_arm["history"].tolist(),
                    "extent": "physical state, latest-two actual observations, and public controller state only",
                    "bank_actual_entry_clearance_m": float(result["actual_entry_clearance_m"]),
                    "replayed_prefix_controls": base_arm["prefix_controls"],
                    "base_head_in_hole_xyz_m": base_arm["head_in_hole_xyz_m"],
                    "constrain_head_in_hole_xyz_m": cn_arm["head_in_hole_xyz_m"],
                    "base_relative_yaw_rad": base_arm["relative_yaw_rad"],
                    "constrain_relative_yaw_rad": cn_arm["relative_yaw_rad"]}
                near.append_pair_journal(journal, pair)
                adapters["constrain"].task = envs[1].unwrapped
                manifest = catalogs[profile].manifest["scenes"][idx]
                for adapter in adapters.values():
                    if hasattr(adapter, "set_episode_context"):
                        adapter.set_episode_context(idx, manifest)
                for name, env, arm in (("base", envs[0], base_arm), ("constrain", envs[1], cn_arm)):
                    if budget[0] >= MAX_NATIVE_CONTROLS or time.monotonic() - started >= WHOLE_SECONDS:
                        raise TimeoutError("entry evaluation physical-control or whole-stage wall budget exhausted")
                    arm["prefix_controls"] = arm["prefix_controls"]
                    _seed(seed)
                    arm_result = near._run_suffix(env, adapters[name], arm, seed, budget,
                        max_native_controls=MAX_NATIVE_CONTROLS, stage_seconds=STAGE_SECONDS)
                    pair["arms"][name] = arm_result
                    pair["status"] = "partial"
                    near.append_pair_journal(journal, pair)
                pair["status"] = "complete"
                completed.append(pair)
                near.append_pair_journal(journal, pair)
            except Exception as exc:
                pair.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                            physical_controls_at_failure=budget[0])
                near.append_pair_journal(journal, pair)
                (output / "failure.json").write_text(json.dumps(near.jsonable({"failure": pair["failure"], "physical_controls": budget[0]}), allow_nan=False, indent=2))
                raise
            finally:
                for env in envs:
                    env.close()
                adapters["constrain"].task = None
        per_condition, handoff_geometry = {}, {}
        for profile in ("random", "fixed"):
            per_condition[profile] = {}
            for condition in CONDITIONS:
                selected = [x for x in completed if x["profile"] == profile and x["condition"] == condition]
                both = sum(x["arms"]["base"]["native_success"] and x["arms"]["constrain"]["native_success"] for x in selected)
                base_only = sum(x["arms"]["base"]["native_success"] and not x["arms"]["constrain"]["native_success"] for x in selected)
                cn_only = sum(not x["arms"]["base"]["native_success"] and x["arms"]["constrain"]["native_success"] for x in selected)
                neither = sum(not x["arms"]["base"]["native_success"] and not x["arms"]["constrain"]["native_success"] for x in selected)
                per_condition[profile][condition] = {"unique_origins": len({x["catalog_index"] for x in selected}), "draws_per_origin": 3,
                    "draws_completed": len(selected), "base_successes": sum(x["arms"]["base"]["native_success"] for x in selected),
                    "constrain_successes": sum(x["arms"]["constrain"]["native_success"] for x in selected),
                    "paired": {"both": both, "base_only": base_only, "constrain_only": cn_only, "neither": neither}}
                heads = [x["handoff"]["base_head_in_hole_xyz_m"] for x in selected]
                yaws = [x["handoff"]["base_relative_yaw_rad"] for x in selected]
                handoff_geometry[f"{profile}/{condition}"] = {
                    "rows": len(selected),
                    "head_in_hole_xyz_min_m": np.min(heads, axis=0).tolist() if heads else None,
                    "head_in_hole_xyz_max_m": np.max(heads, axis=0).tolist() if heads else None,
                    "relative_yaw_range_rad": [float(min(yaws)), float(max(yaws))] if yaws else None,
                    "actual_entry_clearance_range_m": ([min(x["handoff"]["bank_actual_entry_clearance_m"] for x in selected),
                        max(x["handoff"]["bank_actual_entry_clearance_m"] for x in selected)] if selected else None)}
        summary = {"pairs_completed": len(completed), "native_controls_including_replay": budget[0],
                   "setup": setup,
                   "per_condition": per_condition, "handoff_geometry": handoff_geometry,
                   "interpretation": "Conditional outside-entry pose-window paired draws; three draws per six admitted origins/condition are repeated, not new scenes.",
                   "native_rollouts_started": bool(completed)}
        (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    except Exception:
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_comparison(args.output)


if __name__ == "__main__":
    main()
