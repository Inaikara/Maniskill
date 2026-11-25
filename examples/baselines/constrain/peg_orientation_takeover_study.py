"""Preregistered bidirectional Peg policy/expert takeover diagnostic."""

import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

from examples.baselines.constrain import peg_orientation
from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog, restore_peg_scene
from examples.baselines.constrain.peg_orientation_takeover import (
    capture_physical_state, native_grasped, run_live_expert_continuation,
    run_official_geometry_continuation, build_official_planner,
)
from examples.baselines.constrain.peg_orientation_constrain import PegConstrainAdapter

ROOT = Path(__file__).resolve().parents[3]
VALIDATION_CATALOG = ROOT / "outputs/exp/20261111_172604_peg_orientation_adaptation/validation_random_scenes"
DATA = ROOT / "demos/PegInsertionSide-orientation-random-500"
H5 = DATA / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
META = DATA / "trajectory.state.pd_ee_delta_pose.physx_cpu.json"
CHECKPOINT = ROOT / "outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_25000.pt"
CHECKPOINT_SHA256 = "73834286ff5e94fc3bf157d602174522365a9cb3bbb638c6e05579806b82b7df"
TRAIN_IDS = (2, 73, 144, 215, 286, 357, 428, 499)
VALIDATION_IDS = tuple(range(12))
PREFIX_CAP, SUFFIX_CAP, TOTAL_CONTROL_CAP = 296, 200, 24000
TRAIN_PREFIX_CAP = 300
WALL_CAP = 3600
TOLERANCE = 1e-5
SEED = 10000
SOURCE_PATHS = (
    "examples/baselines/constrain/peg_orientation_takeover_study.py",
    "examples/baselines/constrain/peg_orientation_takeover.py",
    "examples/baselines/constrain/peg_orientation_takeover_experiment.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover.py",
    "examples/baselines/constrain/tests/test_peg_orientation_takeover_study.py",
    "examples/baselines/constrain/peg_orientation_preflight.py",
    "examples/baselines/constrain/peg_orientation.py",
    "examples/baselines/constrain/peg_orientation_data.py",
    "examples/baselines/constrain/peg_orientation_constrain.py",
    "examples/baselines/constrain/constrain_denoiser.py",
    "examples/baselines/constrain/peg_orientation_base_dp.py",
    "mani_skill/agents/base_agent.py",
    "mani_skill/agents/controllers/base_controller.py",
    "mani_skill/agents/controllers/pd_ee_pose.py",
    "mani_skill/agents/controllers/pd_joint_pos.py",
    "mani_skill/agents/robots/panda/panda.py",
    "mani_skill/agents/robots/panda/panda_wristcam.py",
    "mani_skill/envs/sapien_env.py",
    "mani_skill/envs/tasks/tabletop/peg_insertion_side.py",
    "mani_skill/examples/motionplanning/base_motionplanner/motionplanner.py",
    "mani_skill/examples/motionplanning/panda/motionplanner.py",
    "mani_skill/examples/motionplanning/two_finger_gripper/motionplanner.py",
    "mani_skill/examples/motionplanning/panda/solutions/peg_insertion_side.py",
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return _jsonable(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic): return _jsonable(x.item())
    if isinstance(x, dict): return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_jsonable(v) for v in x]
    return x


def _canonical_hash(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _env():
    return gym.make(peg_orientation.RANDOM_ENV_ID, obs_mode="state", control_mode="pd_ee_delta_pose",
                    sim_backend="physx_cpu", reconfiguration_freq=1, max_episode_steps=600)


def _head(task):
    pose = task.box_hole_pose.inv() * task.peg_head_pose
    return np.asarray(pose.p[0].detach().cpu(), dtype=np.float64)


def _obs(x):
    return np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x).reshape(-1).copy()


def _success(info):
    v = info.get("success", False)
    if isinstance(v, torch.Tensor): v = v.detach().cpu().numpy()
    return bool(np.asarray(v).any())


def _charge(budget):
    if budget[0] >= TOTAL_CONTROL_CAP:
        raise RuntimeError("aggregate native control budget exhausted before action")
    if len(budget) > 1 and time.monotonic() - budget[1] >= WALL_CAP:
        raise TimeoutError("aggregate native-stage deadline exceeded before action")
    budget[0] += 1


def _step(env, action, budget):
    _charge(budget)
    return env.step(np.asarray(action, dtype=np.float32))


class _BudgetEnv:
    """Count planner calls too, including its control-mode action dictionary."""
    def __init__(self, env, budget): self._env, self._budget, self.trace = env, budget, []
    @property
    def unwrapped(self): return self._env.unwrapped
    def __getattr__(self, name): return getattr(self._env, name)
    def step(self, action):
        _charge(self._budget)
        result = self._env.step(action)
        self.trace.append(_trace(self._env, len(self.trace) + 1, result[-1]))
        return result


def _trigger(task):
    p = _head(task)
    return native_grasped(task) and not bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any()) and -0.10 <= p[0] < -0.015


def _seed_policy(adapter, scene_id, base_seed=SEED):
    seed = base_seed + int(scene_id)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    adapter_seed = {"torch_cpu": torch.get_rng_state().clone(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    return seed, adapter_seed


def _infer(adapter, history):
    action = adapter.get_action(np.asarray(history[-2:], dtype=np.float32))
    if isinstance(action, torch.Tensor): action = action.detach().cpu().numpy()
    return np.asarray(action)[0]


def _record_prefix(env, actions, cap, budget, policy=None, history=None, adapter=None):
    frames = [] if history is None else list(history)
    controls = 0
    pending = None
    action_bank = []
    trace = []
    success = False
    for i in range(cap):
        if policy is None:
            action = actions[i]
        else:
            if i % 8 == 0:
                pending = _infer(adapter, frames[-2:])
            action = pending[i % 8]
        action_bank.append(np.asarray(action).copy())
        observation, _, terminated, truncated, info = _step(env, action, budget)
        controls += 1
        trace.append(_trace(env, controls, info))
        frames.append(_obs(observation))
        success = _success(info)
        if success or bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            break
        if policy is not None and i % 8 == 7:
            continue
    return {"controls": controls, "history": frames[-2:], "success": success,
            "terminal": success or (controls < cap), "next_chunk": pending, "frames": frames,
            "actions": action_bank, "trace": trace}


def _match(a, b):
    errors = {}
    for k in a["physical_state"]:
        errors[k] = float(np.max(np.abs(np.asarray(a["physical_state"][k]) - np.asarray(b["physical_state"][k]))))
    errors["observation_history"] = float(np.max(np.abs(np.asarray(a["observation_history"]) - np.asarray(b["observation_history"]))))
    errors["prefix_action"] = 0.0 if a["prefix_action_sha256"] == b["prefix_action_sha256"] else float("inf")
    return errors


def _snapshot(env, history, prefix_hash, controls):
    task = env.unwrapped
    relative = task.agent.tcp.pose.inv() * task.peg.pose
    from examples.baselines.constrain.peg_orientation_data import _pose_yaw
    return {"physical_state": capture_physical_state(task), "observation_history": history[-2:],
            "prefix_action_sha256": prefix_hash, "controls": controls,
            "head_in_hole_xyz_m": _head(task).tolist(), "grasped": native_grasped(task),
            "peg_to_tcp_pose": np.asarray(relative.raw_pose[0].detach().cpu()).tolist(),
            "peg_yaw_rad": _pose_yaw(task.peg.pose),
            "inserted": bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any()),
            "controller_state_serialized": False}


def _trace(env, control, info):
    task = env.unwrapped
    tcp = np.asarray(task.agent.tcp.pose.raw_pose[0].detach().cpu()).tolist()
    return {"control": int(control), "head_in_hole_xyz_m": _head(task).tolist(),
            "tcp_pose": tcp, "grasped": native_grasped(task),
            "inserted": bool(np.asarray(task.has_peg_inserted()[0].detach().cpu()).any()),
            "native_success": _success(info)}


def _expert_suffix(env, actions, start, budget):
    used, success, trace = 0, False, []
    for action in actions[start:start + SUFFIX_CAP]:
        result = _step(env, action, budget); used += 1
        trace.append(_trace(env, used, result[-1]))
        if _success(result[-1]): success = True; break
        if bool(np.asarray(result[2]).any()) or bool(np.asarray(result[3]).any()): break
    return {"controls": used, "native_success": success, "trace": trace}


def _expert_live(env, budget):
    def deadline(_signum, _frame): raise TimeoutError("per-arm expert planning/continuation exceeded 120 seconds")
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 120)
    proxy = _BudgetEnv(env, budget)
    try:
        result = run_live_expert_continuation(proxy, build_official_planner,
            run_official_geometry_continuation, max_controls=SUFFIX_CAP)
        result["trace"] = proxy.trace
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _continue_policy(env, adapter, history, budget, cap=SUFFIX_CAP):
    used, success, rows = 0, False, []
    while used < cap:
        chunk = _infer(adapter, history[-2:])
        for act in chunk:
            if used >= cap: break
            result = _step(env, act, budget); used += 1
            history.append(_obs(result[0]))
            p = _head(env.unwrapped)
            rows.append({"control": used, "head_in_hole_xyz_m": p.tolist(),
                         "grasped": native_grasped(env.unwrapped), "inserted": bool(np.asarray(env.unwrapped.has_peg_inserted()[0].detach().cpu()).any()),
                         "success": _success(result[-1]),
                         "terminal": bool(np.asarray(result[2]).any()) or bool(np.asarray(result[3]).any())})
            if rows[-1]["success"]: success = True; break
            if bool(np.asarray(result[2]).any()) or bool(np.asarray(result[3]).any()): break
        if success or (rows and rows[-1]["success"]): break
        if rows and rows[-1].get("terminal", False): break
    return {"controls": used, "native_success": success, "trace": rows}


def _load_training_episode(index):
    meta = json.loads(META.read_text())["episodes"][index]
    with h5py.File(H5, "r") as f:
        g = f[f"traj_{index}"]
        return meta, np.asarray(g["obs"]), np.asarray(g["actions"]), np.asarray(g["success"])


def _reset_expert_prefix(env, meta, obs, actions, budget, cap=TRAIN_PREFIX_CAP):
    actual, _ = env.reset(seed=int(meta["episode_seed"]), options=meta["reset_kwargs"].get("options"))
    if np.max(np.abs(_obs(actual) - obs[0])) > TOLERANCE:
        raise RuntimeError("training initial observation mismatch")
    history = [np.asarray(obs[0]).copy()]
    trace = []
    for i, action in enumerate(actions[:cap]):
        result = _step(env, action, budget); history.append(_obs(result[0]))
        trace.append(_trace(env, i + 1, result[-1]))
        if _trigger(env.unwrapped):
            return i + 1, history, _success(result[-1]), trace
        if _success(result[-1]) or bool(np.asarray(result[2]).any()) or bool(np.asarray(result[3]).any()):
            return i + 1, history, True, trace
    return None, history, False, trace


def _run_train_episode(index, budget, on_arm=None):
    meta, obs, actions, _ = _load_training_episode(index)
    arms = []
    reference = None
    for label in ("recorded_expert_suffix", "live_expert_continuation", "frozen_policy"):
        env = _env()
        try:
            n, hist, terminal, prefix_trace = _reset_expert_prefix(env, meta, obs, actions, budget)
            prefix_hash = hashlib.sha256(actions[:n or len(hist)-1].tobytes()).hexdigest()
            branch = _snapshot(env, hist, prefix_hash, n or len(hist)-1)
            row = {"episode_id": index, "arm": label, "prefix_controls": n or len(hist)-1,
                   "triggered": n is not None and _trigger(env.unwrapped), "prefix_terminal": terminal,
                   "branch": branch, "prefix_trace": prefix_trace}
            if reference is None: reference = branch
            else:
                errors = _match(reference, branch); row["matched_branch_errors"] = errors
                if max(errors.values()) > TOLERANCE: raise RuntimeError(f"training branches mismatch {index}: {errors}")
            if row["triggered"]:
                if label == "recorded_expert_suffix": row.update(_expert_suffix(env, actions, n, budget))
                elif label == "live_expert_continuation": row["expert"] = _expert_live(env, budget)
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    adapter = PegConstrainAdapter(CHECKPOINT, device=device, task=env.unwrapped)
                    seed, rng = _seed_policy(adapter, index)
                    row["policy_seed"], row["policy_rng_after_seed"] = seed, rng
                    policy = _continue_policy(env, adapter, hist, budget)
                    row.update(policy)
            else: row["status"] = "no_eligible_trigger"
            continuation_controls = row.get("controls", row.get("expert", {}).get("controls", 0))
            row["native_controls_total"] = row.get("prefix_controls", 0) + continuation_controls
            arms.append(row)
            if on_arm is not None: on_arm(row)
        finally: env.close()
    return arms


def _run_validation_episode(index, catalog, budget, on_update=None):
    scene = catalog.load(index); metadata = scene[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = _env()
    try:
        adapter = PegConstrainAdapter(CHECKPOINT, device=device, task=env.unwrapped)
        adapter.set_episode_context(index, metadata)
        observation = restore_peg_scene(env, scene)
        seed, rng = _seed_policy(adapter, index, base_seed=1)
        first = _obs(observation)
        history = [first.copy(), first.copy()]
        prefix = _record_prefix(env, None, PREFIX_CAP, budget, policy=True, history=history, adapter=adapter)
        base = {"episode_id": index, "prefix_controls": prefix["controls"], "policy_seed": seed,
                "policy_rng_after_prefix": {"torch_cpu": torch.get_rng_state().clone(),
                  "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                "prefix_success": prefix["success"], "branch": _snapshot(env, prefix["history"],
                  hashlib.sha256(np.asarray(prefix["actions"]).tobytes()).hexdigest(), prefix["controls"])}
        base["prefix_trace"] = prefix["trace"]
        if on_update is not None: on_update({**base, "status": "prefix_complete", "arms": []})
        if prefix["success"]:
            return {**base, "status": "early_policy_success_ineligible", "arms": []}
        if prefix["terminal"]:
            return {**base, "status": "terminal_before_296_ineligible", "arms": []}
        if not native_grasped(env.unwrapped):
            return {**base, "status": "unheld_at_296_ineligible", "arms": []}
        # Arm 1 continues in the exact prefix environment and RNG stream.
        policy = _continue_policy(env, adapter, prefix["history"], budget)
        policy_arm = {"arm": "unchanged_policy_continuation", **policy}
        if on_update is not None: on_update({**base, "status": "expert_replay_pending", "arms": [policy_arm]})
        # Independent initial restore and exact saved native action trace.
        env2 = _env()
        try:
            restore_peg_scene(env2, scene)
            adapter2 = PegConstrainAdapter(CHECKPOINT, device=device, task=env2.unwrapped)
            adapter2.set_episode_context(index, metadata)
            torch.set_rng_state(base["policy_rng_after_prefix"]["torch_cpu"])
            if torch.cuda.is_available() and base["policy_rng_after_prefix"]["torch_cuda"]:
                torch.cuda.set_rng_state_all(base["policy_rng_after_prefix"]["torch_cuda"])
            replay_history = [_obs(env2.unwrapped.get_obs())]
            replay_history.append(replay_history[0].copy())
            for action in prefix["actions"]:
                result = _step(env2, action, budget); replay_history.append(_obs(result[0]))
            replay_branch = _snapshot(env2, replay_history, base["branch"]["prefix_action_sha256"], prefix["controls"])
            errors = _match(base["branch"], replay_branch)
            if max(errors.values()) > TOLERANCE:
                return {**base, "status": "matched_prefix_mismatch", "matched_branch_errors": errors,
                        "arms": [policy_arm]}
            expert = _expert_live(env2, budget)
            expert_arm = {"arm": "live_expert_continuation", **expert,
                          "matched_branch_errors": errors, "branch": replay_branch}
            final = {**base, "status": "completed", "arms": [policy_arm, expert_arm]}
            if on_update is not None: on_update(final)
            return final
        finally: env2.close()
    finally: env.close()


def _sources():
    return {p: _sha(ROOT / p) for p in SOURCE_PATHS}


def _input_hashes():
    return {"training_h5": _sha(H5), "training_metadata": _sha(META),
            "checkpoint": _sha(CHECKPOINT), "validation_scenes": _sha(VALIDATION_CATALOG / "scenes.h5"),
            "validation_manifest": _sha(VALIDATION_CATALOG / "manifest.json")}


def _analysis(rows):
    train = [row for row in rows if "arm" in row and row.get("arm") != "validation_pair"]
    train_summary = {}
    for label in ("recorded_expert_suffix", "live_expert_continuation", "frozen_policy"):
        selected = [r for r in train if r["arm"] == label]
        def succeeded(r):
            value = r.get("native_success", False)
            if "expert" in r: value = r["expert"].get("status") == "native_success"
            return bool(value)
        train_summary[label] = {"n_selected": len(selected), "n_eligible": sum(r.get("triggered", False) for r in selected),
                                "n_success": sum(succeeded(r) for r in selected),
                                "n_no_eligible_trigger": sum(not r.get("triggered", False) for r in selected)}
    val = [r for r in rows if r.get("status") in ("completed", "early_policy_success_ineligible", "unheld_at_296_ineligible", "terminal_before_296_ineligible", "matched_prefix_mismatch")]
    eligible = [r for r in val if r.get("status") == "completed"]
    def arm_success(row, name):
        for arm in row.get("arms", []):
            if arm.get("arm") == name:
                if "native_success" in arm:
                    return bool(arm["native_success"])
                if "expert" in arm:
                    return arm["expert"].get("status") == "native_success"
                return arm.get("status") == "native_success"
        return False
    both = sum(arm_success(r, "unchanged_policy_continuation") and arm_success(r, "live_expert_continuation") for r in eligible)
    expert_only = sum(arm_success(r, "live_expert_continuation") and not arm_success(r, "unchanged_policy_continuation") for r in eligible)
    policy_only = sum(arm_success(r, "unchanged_policy_continuation") and not arm_success(r, "live_expert_continuation") for r in eligible)
    neither = len(eligible) - both - expert_only - policy_only
    return {"training_near_branch": train_summary,
            "validation_policy_296": {"denominator_all_12": len(val), "eligible_matched": len(eligible),
              "early_success_ineligible": sum(r.get("status") == "early_policy_success_ineligible" for r in val),
              "unheld_ineligible": sum(r.get("status") == "unheld_at_296_ineligible" for r in val),
              "terminal_before_296_ineligible": sum(r.get("status") == "terminal_before_296_ineligible" for r in val),
              "mismatched_branch": sum(r.get("status") == "matched_prefix_mismatch" for r in val),
              "paired_outcomes": {"both": both, "expert_only": expert_only, "policy_only": policy_only, "neither": neither}},
            "interpretation_limits": ["Training arms are selected training demos and differ from validation scenes/handoff stages.",
              "Expert continuation changes from pd_ee_delta_pose to pd_joint_pos and resets controller targets; this is a confound.",
              "Expert rescue only supports feasibility under this oracle; does not establish irreversibility or off-manifold cause.",
              "A single policy sample per branch is not a robustness estimate; not formal evaluation or an OOD test."]}


def prepare(outdir):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=False)
    catalog = PegSceneCatalog(VALIDATION_CATALOG)
    if len(catalog) < 12: raise ValueError("validation random catalog must contain at least 12 scenes")
    validation_manifest = json.loads((VALIDATION_CATALOG / "manifest.json").read_text())
    if validation_manifest.get("split") != "validation_random":
        raise ValueError("validation catalog manifest must declare split validation_random")
    if _sha(CHECKPOINT) != CHECKPOINT_SHA256: raise ValueError("frozen checkpoint identity mismatch")
    registration = {"format": "peg-bidirectional-takeover-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
      "train_episode_ids": list(TRAIN_IDS), "validation_catalog_indices": list(VALIDATION_IDS),
      "validation_catalog_path": str(VALIDATION_CATALOG.relative_to(ROOT)),
      "validation_split": validation_manifest["split"],
      "train_h5_sha256": _sha(H5), "train_metadata_sha256": _sha(META),
      "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": CHECKPOINT_SHA256,
      "validation_scenes_sha256": _sha(catalog.path), "validation_manifest_sha256": _sha(VALIDATION_CATALOG / "manifest.json"),
      "trigger": "first postaction native grasp, not inserted, official peg-head x in [-0.10,-0.015) m",
      "trigger_tolerance": TOLERANCE,
      "ineligibility": ["preserve selected ID with no replacement", "training: no trigger within 300 or trigger only after terminal", "validation: official success before 296, terminal before 296, or unheld at 296"],
      "branch_match": "physical state and last-two actual observation history max abs error <=1e-5; exact prefix action SHA256; mismatch invalidates pair",
      "raw_state_fields": ["robot_qpos", "robot_qvel", "robot_root_pose", "tcp_pose", "peg_pose", "hole_pose", "head_in_hole_xyz_m", "peg_to_tcp_pose", "peg_yaw_rad", "native_grasped", "has_peg_inserted"],
      "controller_state": "recorded public-controller serialization not used; controller_state_serialized=false; mode switch and controller.reset are explicit confounds",
      "arms": {"training": ["recorded expert suffix", "live official expert", "frozen checkpoint policy"],
               "validation": ["policy prefix through 296 then unchanged policy 200", "matched replay prefix then live expert 200"]},
      "policy": {"training_seed": "10000+episode_id", "validation_seed": "1+catalog_index",
                  "training_history": "last two actual observations at takeover",
                  "validation_history": "two restored initial observations; then last two actual observations, continuous through control 296 and policy continuation",
                  "act_chunk": 8,
                  "checkpoint_update": 25000, "ema": True, "hint": "live correct, native 7d, DDPM 100"},
      "budgets": {"training_prefix": TRAIN_PREFIX_CAP, "validation_prefix": PREFIX_CAP, "suffix": SUFFIX_CAP, "aggregate_controls": TOTAL_CONTROL_CAP,
                   "aggregate_wall_seconds": WALL_CAP, "per_expert_arm_wall_seconds": 120, "env_time_limit": 600},
      "confounds": ["live expert uses pd_joint_pos after public controller reset; explicit controller-mode confound", "diagnostic only, not formal evaluation"],
      "normalization": "numpy/tensor result rows recursively normalized once before canonical result hashing and persistence",
      "visual_evidence": "no video capture; all arms retain per-control head-in-hole/grasp/insertion/success traces",
      "attempt_arm_order": "training IDs ascending; per ID recorded suffix, live expert, frozen policy; validation IDs ascending; prefix then policy continuation then matched expert replay",
      "inputs_sha256": _input_hashes(),
      "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                  "gymnasium": __import__("gymnasium").__version__,
                  "mani_skill_git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=20).stdout.strip()},
      "source_closure_sha256": _sources()}
    registration["registration_sha256"] = _canonical_hash(registration)
    (outdir / "preregistration.json").write_text(json.dumps(registration, indent=2, sort_keys=True)+"\n")
    archive = outdir / "registered_sources"
    for p in SOURCE_PATHS:
        dst = archive / p; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/p, dst)
    (outdir / "runtime_conda_explicit.txt").write_text(subprocess.run(["conda", "list", "--explicit"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=20).stdout)
    (outdir / "runtime_versions.json").write_text(json.dumps({"python": sys.version, "numpy": np.__version__, "torch": torch.__version__}, indent=2)+"\n")
    print(json.dumps({"outdir": str(outdir), "registration_sha256": registration["registration_sha256"]}))


def run(outdir, attempt_id):
    outdir = Path(outdir); reg = json.loads((outdir/"preregistration.json").read_text())
    digest = reg.pop("registration_sha256")
    if _canonical_hash(reg) != digest: raise RuntimeError("preregistration checksum mismatch")
    if _sources() != reg["source_closure_sha256"]: raise RuntimeError("registered source closure changed")
    attempt = outdir / f"attempt_{attempt_id}"; attempt.mkdir()
    start = time.monotonic(); budget = [0, start]; rows = []
    ledger_path = attempt/"execution_ledger.json"
    ledger = {"registration_sha256": digest, "started_utc": datetime.now(timezone.utc).isoformat(), "arms": []}
    ledger_path.write_text(json.dumps(ledger, indent=2)+"\n")
    journal_number = 0

    def persist_progress(kind, episode_id, arm_name, row):
        nonlocal journal_number
        journal_number += 1
        normalized = _jsonable(row)
        name = f"progress_{journal_number:04d}.json"
        payload = _jsonable({"sequence": journal_number, "kind": kind, "episode_id": episode_id,
                             "arm": arm_name, "row": normalized})
        encoded = (json.dumps(payload, indent=2, sort_keys=True)+"\n").encode()
        journal_path = attempt / name
        journal_path.write_bytes(encoded)
        return normalized, {"path": name, "sha256": hashlib.sha256(encoded).hexdigest()}

    try:
        for index in TRAIN_IDS:
            def save_train_arm(row):
                normalized, journal = persist_progress("training_arm", index, row["arm"], row)
                rows.append(normalized)
                ledger["arms"].append({"episode_id":index,"arm":normalized["arm"],
                                        "status":normalized.get("status","completed"),
                                        "controls":normalized["native_controls_total"], "journals":[journal]})
                ledger_path.write_text(json.dumps(_jsonable(ledger), indent=2)+"\n")
            _run_train_episode(index, budget, on_arm=save_train_arm)
        catalog = PegSceneCatalog(VALIDATION_CATALOG)
        for index in VALIDATION_IDS:
            def save_validation_update(row):
                wrapped, journal = persist_progress("validation_progress", index, "validation_pair",
                                                    {"episode_id": index, "arm": "validation_pair", **row})
                position = next((i for i, saved in enumerate(rows)
                                 if saved.get("episode_id") == index and saved.get("arm") == "validation_pair"), None)
                if position is None: rows.append(wrapped)
                else: rows[position] = wrapped
                counts = {name: next((a.get("controls", 0) for a in row.get("arms", []) if a.get("arm") == name), 0)
                          for name in ("unchanged_policy_continuation", "live_expert_continuation")}
                ledger_row = next((a for a in ledger["arms"] if a.get("episode_id") == index and a.get("arm") == "validation_pair"), None)
                update = {"episode_id": index, "arm": "validation_pair", "status": row["status"],
                          "prefix_controls": row.get("prefix_controls", 0), **counts}
                if ledger_row is None: ledger["arms"].append(update)
                else: ledger_row.update(update)
                ledger_row = next(a for a in ledger["arms"]
                                  if a.get("episode_id") == index and a.get("arm") == "validation_pair")
                ledger_row.setdefault("journals", []).append(journal)
                ledger_path.write_text(json.dumps(_jsonable(ledger), indent=2)+"\n")
            row = _run_validation_episode(index, catalog, budget, on_update=save_validation_update)
            if row["status"] != "completed": save_validation_update(row)
            if time.monotonic()-start > WALL_CAP: raise TimeoutError("aggregate native-stage deadline exceeded")
        inputs_after = _input_hashes()
        if inputs_after != reg["inputs_sha256"]: raise RuntimeError("frozen input identity changed during experiment")
        sources_after = _sources()
        if sources_after != reg["source_closure_sha256"]: raise RuntimeError("registered source changed during experiment")
        result = _jsonable({"status":"completed","elapsed_seconds":time.monotonic()-start,"native_controls_total":budget[0],"registration_sha256":digest,"inputs_sha256_before":reg["inputs_sha256"],"inputs_sha256_after":inputs_after,"source_closure_sha256_before":reg["source_closure_sha256"],"source_closure_sha256_after":sources_after,"rows":rows})
        result["results_sha256"] = _canonical_hash(result)
        (attempt/"results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
        (attempt/"analysis.json").write_text(json.dumps(_analysis(rows),indent=2,sort_keys=True)+"\n")
        ledger.update(status="completed",native_controls_total=budget[0],elapsed_seconds=time.monotonic()-start)
        ledger_path.write_text(json.dumps(_jsonable(ledger),indent=2)+"\n")
    except BaseException as exc:
        failure = _jsonable({"status":"failed_or_interrupted","error_type":type(exc).__name__,"error":str(exc),"native_controls_total":budget[0],"rows_completed_or_partial":rows})
        (attempt/"attempt_failure.json").write_text(json.dumps(failure,indent=2,sort_keys=True)+"\n")
        ledger.update(status="failed_or_interrupted",error=str(exc),native_controls_total=budget[0]); ledger_path.write_text(json.dumps(_jsonable(ledger),indent=2)+"\n")
        raise


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    a=s.add_parser("prepare"); a.add_argument("--outdir",required=True)
    b=s.add_parser("run"); b.add_argument("--outdir",required=True); b.add_argument("--attempt-id",required=True)
    x=p.parse_args(); prepare(x.outdir) if x.cmd=="prepare" else run(x.outdir,x.attempt_id)


if __name__ == "__main__": main()
