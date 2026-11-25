"""Paired native comparison from frozen expert-reachable Peg near-hole branches."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from examples.baselines.constrain import peg_orientation
from examples.baselines.constrain.peg_orientation_base_dp import PegBaseAdapter, _video_frame
from examples.baselines.constrain.peg_orientation_constrain import PegConstrainAdapter
from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog, restore_peg_scene
from examples.baselines.constrain.peg_orientation_near_hole import near_hole_eligible
from examples.baselines.constrain.peg_orientation_takeover import capture_physical_state, native_grasped

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "outputs/exp/20261118_122546_peg_near_hole_initialization"
CATALOG_ROOT = ROOT / "outputs/exp/20261111_172604_peg_orientation_adaptation"
BASE_CKPT = ROOT / "outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt"
CN_CKPT = ROOT / "outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_25000.pt"
BASE_SHA = "067a4f5cda35ce91151b12b8095c136673d6d9f3afd6f742c7aa164f8b62beea"
CN_SHA = "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df"
DATA_DIRS = {
    "random": ROOT / "demos/PegInsertionSide-near-hole-diagnostic-random_20261118_122546",
    "fixed": ROOT / "demos/PegInsertionSide-near-hole-diagnostic-fixed_20261118_122546",
}
SEEDS = (1, 1001, 2001)
SUFFIX_CAP, PAIR_CAP, STAGE_CAP, WHOLE_STAGE_CAP = 200, 500, 7200, 10800
TOLERANCE = 1e-5
VIDEO_CASES = {"random": 0, "fixed": 0}
SOURCE_PATHS = (
    "examples/baselines/constrain/peg_orientation_near_hole_eval.py",
    "examples/baselines/constrain/peg_orientation_near_hole.py",
    "examples/baselines/constrain/peg_orientation_takeover.py",
    "examples/baselines/constrain/peg_orientation_data.py",
    "examples/baselines/constrain/peg_orientation.py",
    "examples/baselines/constrain/peg_orientation_base_dp.py",
    "examples/baselines/constrain/peg_orientation_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/tests/test_peg_orientation.py",
    "examples/baselines/constrain/tests/test_base_dp.py",
    "examples/baselines/constrain/tests/test_peg_orientation_preflight.py",
    "examples/baselines/constrain/tests/test_peg_orientation_data.py",
    "examples/baselines/constrain/tests/test_peg_orientation_base_dp.py",
    "examples/baselines/constrain/tests/test_peg_official_dp.py",
    "examples/baselines/constrain/tests/test_peg_orientation_constrain.py",
    "examples/baselines/constrain/tests/test_peg_orientation_failure_analysis.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover_study.py",
    "examples/baselines/constrain/tests/test_peg_orientation_near_hole.py",
    "examples/baselines/constrain/tests/test_peg_orientation_near_hole_eval.py",
    "mani_skill/agents/base_agent.py",
    "mani_skill/agents/controllers/base_controller.py",
    "mani_skill/agents/controllers/pd_ee_pose.py",
    "mani_skill/agents/controllers/pd_joint_pos.py",
    "mani_skill/agents/robots/panda/panda.py",
    "mani_skill/agents/robots/panda/panda_wristcam.py",
    "mani_skill/envs/sapien_env.py",
    "mani_skill/envs/tasks/tabletop/peg_insertion_side.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_metadata():
    packages = {}
    for name in ("gymnasium", "torch", "numpy", "h5py", "imageio", "diffusers", "mani_skill"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed-as-distribution"
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def jsonable(value):
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def append_pair_journal(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def assert_controller_match(base_env, cn_env, tolerance=TOLERANCE):
    left = base_env.unwrapped.agent.controller.get_state()
    right = cn_env.unwrapped.agent.controller.get_state()
    if set(left) != set(right):
        raise ValueError("controller state mismatch: public state keys differ")
    for key in left:
        a, b = np.asarray(jsonable(left[key])), np.asarray(jsonable(right[key]))
        if a.shape != b.shape or (a.size and np.max(np.abs(a - b)) > tolerance):
            raise ValueError(f"controller state mismatch at {key}")
    if base_env.unwrapped.control_mode != cn_env.unwrapped.control_mode:
        raise ValueError("controller state mismatch: control modes differ")
    return {"public_state": jsonable(left), "public_state_keys": sorted(left),
            "state_extent": "empty public controller state" if not left else "all public get_state fields",
            "control_mode": str(base_env.unwrapped.control_mode)}


def _max_error(expected, actual):
    errors = {}
    for key, value in expected.items():
        a, b = np.asarray(value), np.asarray(actual[key])
        if a.shape != b.shape:
            errors[key] = float("inf")
        else:
            errors[key] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return errors


def _actual_history(observations):
    return [np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x).reshape(-1).copy()
            for x in observations]


def _native_success(info):
    value = info.get("success", False)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bool(np.asarray(value).any())


def _controller_snapshot(env):
    controller = env.unwrapped.agent.controller
    state = controller.get_state()
    return {"type": f"{type(controller).__module__}.{type(controller).__qualname__}",
            "state": jsonable(state), "state_keys": sorted(state),
            "mode": str(env.unwrapped.control_mode)}


def _relative_yaw(task):
    from examples.baselines.constrain.peg_orientation_data import _pose_yaw
    peg_yaw = _pose_yaw(task.peg.pose)
    hole_yaw = _pose_yaw(task.box_hole_pose)
    return float(np.arctan2(np.sin(peg_yaw - hole_yaw), np.cos(peg_yaw - hole_yaw)))


def _source_actions(group, source_episode):
    with h5py.File(group, "r") as handle:
        return np.asarray(handle[f"traj_{source_episode}"]["actions"])


def _make_env(split):
    env_id = peg_orientation.RANDOM_ENV_ID if split == "random" else peg_orientation.ENV_ID
    return gym.make(env_id, obs_mode="state", control_mode="pd_ee_delta_pose",
                    sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=600,
                    render_mode="rgb_array")


def _prepare_arm(env, scene, actions, prefix_controls, bank_row, budget):
    obs = restore_peg_scene(env, scene)
    history = _actual_history([obs])
    digest = hashlib.sha256()
    for action in actions[:prefix_controls]:
        if budget[0] >= 78000 or time.monotonic() - budget[1] >= STAGE_CAP:
            raise TimeoutError("stage native-control or wall budget exhausted before prefix action")
        digest.update(np.asarray(action).tobytes())
        obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        budget[0] += 1
        history.append(np.asarray(obs.detach().cpu()).reshape(-1).copy())
        if _native_success(info):
            raise ValueError("admitted bank branch is already officially inserted during prefix replay")
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            raise ValueError("prefix terminated before frozen near-hole branch")
    task = env.unwrapped
    if digest.hexdigest() != bank_row["prefix_actions_sha256"]:
        raise ValueError("prefix action SHA does not match bank")
    if str(task.control_mode) != bank_row["controller_mode"]:
        raise ValueError("replayed controller mode differs from bank")
    if not native_grasped(task):
        raise ValueError("replayed branch is not natively grasped")
    inserted = bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any())
    head = task.box_hole_pose.inv() * task.peg_head_pose
    head_xyz = np.asarray(head.p[0].detach().cpu(), dtype=np.float64)
    if not near_hole_eligible(task):
        raise ValueError("replayed branch no longer meets frozen near-hole eligibility")
    physical = capture_physical_state(task)
    physical_errors = _max_error(bank_row["physical_state"], physical)
    history_array = np.asarray(history[-2:])
    history_errors = {"observation_history": float(np.max(np.abs(
        history_array - np.asarray(bank_row["observation_history"]))))}
    errors = {**physical_errors, **history_errors,
              "head_in_hole_xyz_m": float(np.max(np.abs(head_xyz - bank_row["head_in_hole_xyz_m"])))}
    if max(errors.values()) > TOLERANCE:
        raise ValueError(f"replayed native branch differs from bank by >{TOLERANCE}: {errors}")
    return {"history": history[-2:], "physical_state": physical, "errors": errors,
            "controller": _controller_snapshot(env), "head_in_hole_xyz_m": head_xyz.tolist(),
            "relative_yaw_rad": _relative_yaw(task),
            "grasped": True, "inserted": inserted, "prefix_action_sha256": digest.hexdigest()}


def _seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _run_suffix(env, adapter, arm, seed, budget, video_writer=None, *, max_native_controls=78000, stage_seconds=STAGE_CAP):
    _seed(seed)
    arm_started = time.monotonic()
    history = [x.copy() for x in arm["history"]]
    initial = arm["head_in_hole_xyz_m"]
    rows, frames = [], 0
    if video_writer is not None:
        video_writer.append_data(_video_frame(env)); frames += 1
    success_control = None
    for control in range(SUFFIX_CAP):
        if budget[0] >= max_native_controls or time.monotonic() - budget[1] >= stage_seconds:
            raise TimeoutError("stage native-control or wall budget exhausted before suffix action")
        if control % 8 == 0:
            action_chunk = adapter.get_action(np.asarray(history[-2:], dtype=np.float32)[None])
            if isinstance(action_chunk, torch.Tensor):
                action_chunk = action_chunk.detach().cpu().numpy()
            action_chunk = np.asarray(action_chunk)[0]
        action = action_chunk[control % 8]
        obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        budget[0] += 1
        history.append(np.asarray(obs.detach().cpu()).reshape(-1).copy())
        task = env.unwrapped
        head = task.box_hole_pose.inv() * task.peg_head_pose
        head_xyz = np.asarray(head.p[0].detach().cpu(), dtype=np.float64)
        success = _native_success(info)
        if success and success_control is None:
            success_control = control + 1
        rows.append({"suffix_control": control + 1, "head_in_hole_xyz_m": head_xyz.tolist(),
                     "head_distance_m": float(np.linalg.norm(head_xyz)),
                     "grasped": native_grasped(task),
                     "contact_loss": bool(rows[-1]["grasped"] and not native_grasped(task)) if rows else False,
                     "official_inserted": bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any()),
                     "native_success": success, "terminal": bool(np.asarray(terminated).any() or np.asarray(truncated).any()),
                     "action": np.asarray(action).reshape(-1).tolist()})
        if video_writer is not None:
            video_writer.append_data(_video_frame(env)); frames += 1
        if success or rows[-1]["terminal"]:
            break
    return {"native_success": success_control is not None, "success_suffix_control": success_control,
            "elapsed_seconds": time.monotonic() - arm_started,
            "seconds_to_native_success": (time.monotonic() - arm_started if success_control is not None else None),
            "prefix_controls": int(arm["prefix_controls"]), "suffix_controls": len(rows),
            "total_controls": int(arm["prefix_controls"]) + len(rows),
            "initial_head_xyz_m": initial, "trace": rows,
            "native_failure": None if success_control is not None else
                ("contact_loss" if any(r["contact_loss"] for r in rows) else
                 "terminal" if rows and rows[-1]["terminal"] else "control_cap"),
            "video_frames": frames}


def summarize_pairs(rows, group, eligible_count):
    selected = [row for row in rows if row["group"] == group]
    seeds = {str(base): {"scenes": 0, "base_successes": 0, "constrain_successes": 0,
                         "paired": {"both": 0, "base_only": 0, "constrain_only": 0, "neither": 0}}
             for base in SEEDS}
    for row in selected:
        seed = str(SEEDS[int(row["replicate"])])
        entry = seeds[seed]
        entry["scenes"] += 1
        b, c = row["base"]["native_success"], row["constrain"]["native_success"]
        entry["base_successes"] += int(b)
        entry["constrain_successes"] += int(c)
        entry["paired"]["both" if b and c else "base_only" if b else "constrain_only" if c else "neither"] += 1
    pooled = {"base_successes": sum(item["base_successes"] for item in seeds.values()),
              "constrain_successes": sum(item["constrain_successes"] for item in seeds.values()),
              "paired": {key: sum(item["paired"][key] for item in seeds.values())
                         for key in ("both", "base_only", "constrain_only", "neither")}}
    return {"eligible_scenes": int(eligible_count), "replicate_draws_per_scene": 3,
            "pooled_descriptive_draws": 3 * int(eligible_count),
            "pooled_denominator_per_policy": 3 * int(eligible_count),
            "per_seed": seeds, "pooled_descriptive": pooled,
            "scene_replicates_are_not_independent_scenes": True}


def _bank_rows(split):
    filename = "branches_reconciled.jsonl" if split == "random" else "branches.jsonl"
    path = BANK / f"{split}_bank" / filename
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_comparison(output, device="cpu"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    catalogs, banks, sources = {}, {}, {}
    all_rows = []
    budget = [0, time.monotonic()]
    envs = []
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    try:
        for split in ("random", "fixed"):
            catalogs[split] = PegSceneCatalog(CATALOG_ROOT / ("validation_random_scenes" if split == "random" else "validation_fixed_scenes"))
            banks[split] = _bank_rows(split)
            source = DATA_DIRS[split]
            sources[split] = {"h5": source / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5",
                              "metadata": source / "trajectory.state.pd_ee_delta_pose.physx_cpu.json"}
        input_hashes = {"base_checkpoint": sha256(BASE_CKPT), "constrain_checkpoint": sha256(CN_CKPT)}
        if input_hashes != {"base_checkpoint": BASE_SHA, "constrain_checkpoint": CN_SHA}:
            raise ValueError("frozen checkpoint identity mismatch")
        source_identities = {}
        for split in ("random", "fixed"):
            source_identities[split] = {str(path.relative_to(ROOT)): sha256(path)
                                        for path in sorted(DATA_DIRS[split].iterdir()) if path.is_file()}
        manifest_hashes = {split: sha256(catalogs[split].directory / "manifest.json") for split in catalogs}
        bank_hashes = {split: sha256(BANK / f"{split}_bank" / ("branches_reconciled.jsonl" if split == "random" else "branches.jsonl")) for split in banks}
        bank_files = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(BANK.rglob("*")) if path.is_file()}
        prereg = {"objective": "Paired frozen Base vs Constrain comparison on validation near-hole expert branches; descriptive conditional benchmark only.",
                  "catalog_indices": list(range(20)), "eligible_counts": {k: sum(bool(r["eligible"]) for r in v) for k,v in banks.items()},
                  "bank_hashes": bank_hashes, "catalog_manifest_hashes": manifest_hashes,
                  "complete_bank_and_provenance_file_hashes": bank_files,
                  "source_h5_json_hashes": source_identities, "model_hashes": input_hashes,
                  "models": {"base": {"adapter": "PegBaseAdapter", "checkpoint": str(BASE_CKPT), "action_dim": 7, "obs_dim": 39},
                             "constrain": {"adapter": "PegConstrainAdapter", "checkpoint": str(CN_CKPT), "action_dim": 7, "obs_dim": 39, "hint": "live current task hint"}},
                  "seeds": list(SEEDS), "seed_rule": "seed=1000*k+1+catalog_index for k=0,1,2, applied after adapters and both independently replayed prefixes are constructed",
                  "prefix": "independent native EE envs, catalog restore, exact converted H5 actions[:bank prefix_controls], no intermediate restore/source observations",
                  "controller": "both arms remain pd_ee_delta_pose; compare public get_state and physical state/history before inference; empty public controller state is reported as such",
                  "budget": {"suffix_controls_per_arm": SUFFIX_CAP, "physical_controls_including_replay_max": 78000, "env_time_limit": 600, "native_stage_wall_seconds": STAGE_CAP, "whole_stage_wall_seconds": WHOLE_STAGE_CAP},
                  "native_success": "official info.success, stop immediately",
                  "video_cases": {"rule": "first eligible catalog index in each split, replicate 0, both policies; captured in-pass",
                                  "selected_before_outcomes": VIDEO_CASES},
                  "exclusions": "No replacements. Bank-ineligible rows remain in initial 20-scene denominator, not policy failure rows; policy outcome denominator eligible branches only.",
                  "provenance": "bank random recovered from journals; fixed amendment retrospective/historical source bytes unavailable; bank catalog viewed; not blinded or end-to-end.",
                  "formal_gate": False, "plug": False, "created_utc": started_utc}
        prereg["runtime"] = runtime_metadata()
        wrapper = output / "attempt_001.sh"
        wrapper_bytes = wrapper.read_bytes()
        wrapper_sha = hashlib.sha256(wrapper_bytes).hexdigest()
        prereg["outer_command"] = f"bash {output.relative_to(ROOT)}/attempt_001.sh"
        prereg["outer_wrapper"] = {"path": str(wrapper.relative_to(ROOT)), "sha256": wrapper_sha}
        prereg["attempt_stdout"] = str(output.relative_to(ROOT) / "attempt_001.stdout.log")
        prereg["attempt_stderr"] = str(output.relative_to(ROOT) / "attempt_001.stderr.log")
        prereg_path = output / "preregistration.json"
        prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
        archived = output / "source_archive"
        archived.mkdir()
        closure = {rel: sha256(ROOT / rel) for rel in SOURCE_PATHS}
        for rel in SOURCE_PATHS:
            target = archived / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
        (archived / "outer_attempt_001.sh").write_bytes(wrapper_bytes)
        closure["outer_attempt_001.sh"] = wrapper_sha
        (output / "source_closure.json").write_text(json.dumps(closure, indent=2, sort_keys=True) + "\n")
        prereg["source_closure_sha256"] = sha256(output / "source_closure.json")
        prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
        (output / "preregistration.sha256").write_text(sha256(prereg_path) + "  preregistration.json\n")
        # Load both real adapters before any policy RNG seed is set.
        adapters = {"base": PegBaseAdapter(BASE_CKPT, device=device),
                    "constrain": PegConstrainAdapter(CN_CKPT, device=device, task=None, record_all_hints=True)}
        metadata = {name: adapter.metadata for name, adapter in adapters.items()}
        (output / "model_metadata.json").write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True) + "\n")
        for split in ("random", "fixed"):
            source_json = json.loads(sources[split]["metadata"].read_text())
            env_id = peg_orientation.RANDOM_ENV_ID if split == "random" else peg_orientation.ENV_ID
            for bank_row in banks[split]:
                if not bank_row["eligible"]:
                    continue
                idx = int(bank_row["catalog_index"])
                scene = catalogs[split].load(idx)
                actions = _source_actions(sources[split]["h5"], bank_row["source_episode_id"])
                for replicate, base_seed in enumerate(SEEDS):
                    seed = int(base_seed + idx)
                    pair = {"group": split, "catalog_index": idx, "replicate": replicate,
                            "seed": seed, "arms": {}, "status": "preparing",
                            "started_utc": datetime.now(timezone.utc).isoformat(),
                            "started_monotonic": time.monotonic(),
                            "physical_controls_at_start": int(budget[0])}
                    journal_path = output / "pairs.jsonl"
                    video_writers = {}
                    try:
                        # Both separate environments and prefixes exist before either model is queried.
                        envs = [_make_env(split), _make_env(split)]
                        base_env, cn_env = envs
                        base_arm = _prepare_arm(base_env, scene, actions, int(bank_row["prefix_controls"]), bank_row, budget)
                        cn_arm = _prepare_arm(cn_env, scene, actions, int(bank_row["prefix_controls"]), bank_row, budget)
                        controller_match = assert_controller_match(base_env, cn_env)
                        controller_match["base"] = base_arm["controller"]
                        controller_match["constrain"] = cn_arm["controller"]
                        if not np.array_equal(np.asarray(base_arm["history"]), np.asarray(cn_arm["history"])):
                            raise ValueError("paired actual observation histories differ")
                        pair["prefix"] = {"physical_errors_base": base_arm["errors"], "physical_errors_constrain": cn_arm["errors"],
                                           "controller": controller_match, "history_sha256": hashlib.sha256(np.asarray(base_arm["history"]).tobytes()).hexdigest(),
                                           "near_initial": {"base_head_in_hole_xyz_m": base_arm["head_in_hole_xyz_m"],
                                                            "constrain_head_in_hole_xyz_m": cn_arm["head_in_hole_xyz_m"],
                                                            "base_relative_yaw_rad": base_arm["relative_yaw_rad"],
                                                            "constrain_relative_yaw_rad": cn_arm["relative_yaw_rad"]}}
                        # Both controllers and branches have been checked before the first policy inference.
                        adapters["constrain"].task = cn_env.unwrapped
                        for adapter in adapters.values():
                            if hasattr(adapter, "set_episode_context"):
                                adapter.set_episode_context(idx, catalogs[split].manifest["scenes"][idx])
                        initial_inserted = [bool(np.asarray(e.unwrapped.has_peg_inserted()[0].detach().cpu()).any()) for e in envs]
                        if any(initial_inserted):
                            raise ValueError("admitted branch is already officially inserted")
                        chosen_video = (idx == next(i for i, r in enumerate(banks[split]) if r["eligible"])
                                        and replicate == 0)
                        if chosen_video:
                            for arm_name in ("base", "constrain"):
                                video_writers[arm_name] = imageio.get_writer(
                                    str(output / f"{split}_index{idx:02d}_replicate{replicate}_{arm_name}.mp4"),
                                    fps=20, codec="libx264", macro_block_size=1)
                        for arm_name, env, arm in (("base", base_env, base_arm), ("constrain", cn_env, cn_arm)):
                            if time.monotonic() - started >= WHOLE_STAGE_CAP:
                                raise TimeoutError("near-hole policy stage wall budget exceeded")
                            arm["prefix_controls"] = int(bank_row["prefix_controls"])
                            result = _run_suffix(env, adapters[arm_name], arm, seed, budget,
                                                 video_writers.get(arm_name))
                            result["model_metadata"] = adapters[arm_name].metadata
                            if arm_name == "constrain":
                                result["hint_samples"] = adapters[arm_name].hint_samples
                            pair["arms"][arm_name] = result
                            pair["status"] = "partial"
                            append_pair_journal(journal_path, pair)
                            if arm_name == "base":
                                # Same per-pair seed, independent arm RNG; no shared simulator state.
                                _seed(seed)
                        pair["status"] = "complete"
                        pair["ended_utc"] = datetime.now(timezone.utc).isoformat()
                        pair["elapsed_seconds"] = time.monotonic() - pair["started_monotonic"]
                        all_rows.append({"group": split, "catalog_index": idx, "replicate": replicate,
                                         "seed": seed, "prefix": pair["prefix"],
                                         "base": pair["arms"]["base"],
                                         "constrain": pair["arms"]["constrain"]})
                        append_pair_journal(journal_path, pair)
                    except Exception as exc:
                        pair["status"] = "failed"
                        pair["failure"] = {"type": type(exc).__name__, "message": str(exc)}
                        pair["physical_controls_at_failure"] = int(budget[0])
                        append_pair_journal(output / "pairs.jsonl", pair)
                        raise
                    finally:
                        for writer in video_writers.values():
                            writer.close()
                        for env in envs:
                            env.close()
                        envs = []
                        # Restore policy task before next arm's live-hint sampling.
                        adapters["constrain"].task = None
        geometry = {}
        for split in ("random", "fixed"):
            selected = [row for row in all_rows if row["group"] == split]
            heads = np.asarray([row["prefix"]["near_initial"]["base_head_in_hole_xyz_m"] for row in selected])
            yaws = np.asarray([row["prefix"]["near_initial"]["base_relative_yaw_rad"] for row in selected])
            geometry[split] = {"measured_pair_initial_head_xyz_min_m": heads.min(axis=0).tolist(),
                               "measured_pair_initial_head_xyz_max_m": heads.max(axis=0).tolist(),
                               "measured_pair_initial_relative_yaw_range_rad": [float(yaws.min()), float(yaws.max())],
                               "replicate_rows": len(selected)}
        result = {"groups": {split: summarize_pairs(all_rows, split,
                                                   sum(bool(row["eligible"]) for row in banks[split]))
                              for split in ("random", "fixed")},
                  "measured_initial_geometry": geometry,
                  "pairs_completed": len(all_rows), "native_controls_including_prefix_replay": budget[0],
                  "elapsed_seconds": time.monotonic() - started, "started_utc": started_utc,
                  "interpretation": "Conditional paired validation evidence only; three stochastic draws per same eligible branch, not independent scenes; no end-to-end or formal gate claim."}
        (output / "results.json").write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
        return result
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc),
                   "physical_controls_including_prefix_replay": int(budget[0]),
                   "elapsed_seconds": time.monotonic() - started,
                   "failed_utc": datetime.now(timezone.utc).isoformat()}
        (output / "failure.json").write_text(json.dumps(jsonable(failure), indent=2, sort_keys=True) + "\n")
        raise
    finally:
        for env in envs:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_comparison(args.output, args.device)


if __name__ == "__main__":
    main()
