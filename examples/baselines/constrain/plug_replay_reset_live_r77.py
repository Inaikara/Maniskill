"""Bounded reset-vs-state-vs-live-expert PlugCharger diagnosis (R77)."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common

from examples.baselines.constrain import plug_orientation

SOURCE = Path("demos/PlugCharger-orientation-preflight-compat_v2_20261119_105049/expert_fixed/expert.h5")
SOURCE_JSON = SOURCE.with_suffix(".json")
CASES = ((0, "pass", 2026113001), (2, "fail", 2026113003))
MAX_CASE_CONTROLS = 600
MAX_TOTAL_CONTROLS = 6000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def snapshot(env):
    u = env.unwrapped
    state = jsonable(u.get_state_dict())
    agent = u.agent
    ctrl = agent.controller.controllers["arm"]
    cfg = jsonable(ctrl.config.__dict__) if hasattr(ctrl.config, "__dict__") else str(ctrl.config)
    return {
        "state": state,
        "charger_pose": jsonable(u.charger.pose.raw_pose),
        "receptacle_pose": jsonable(u.receptacle.pose.raw_pose),
        "tcp_pose": jsonable(agent.tcp.pose.raw_pose),
        "robot_qpos": jsonable(agent.robot.get_qpos()),
        "robot_qvel": jsonable(agent.robot.get_qvel()),
        "controller_config": cfg,
        "controller_start_qpos": jsonable(getattr(ctrl, "_start_qpos", None)),
        "controller_target_qpos": jsonable(getattr(ctrl, "_target_qpos", None)),
        "controller_step": jsonable(getattr(ctrl, "_step", None)),
        "sim_timestep": float(u.sim_config.sim_freq_inv) if hasattr(u, "sim_config") and hasattr(u.sim_config, "sim_freq_inv") else None,
        "sim_config": str(getattr(u, "sim_config", None)),
    }


def _get_sources():
    meta = json.loads(SOURCE_JSON.read_text())
    rows = {}
    with h5py.File(SOURCE, "r") as h:
        for eid, _, seed in CASES:
            group = h[f"traj_{eid}"]
            rows[eid] = (np.asarray(group["actions"], dtype=np.float32),
                         trajectory_utils.dict_to_list_of_dicts(group["env_states"])[0])
    return meta, rows


def run_case(eid, label, seed, condition, out, action_rows, state0, counter):
    out.mkdir(parents=True, exist_ok=False)
    env = gym.make(plug_orientation.ENV_ID, obs_mode="state", control_mode="pd_joint_pos",
                   sim_backend="physx_cpu", reward_mode="sparse", max_episode_steps=600)
    start = time.monotonic()
    traces = []
    try:
        env.reset(seed=seed, options={"target_yaw": 0.0})
        fresh = snapshot(env)
        if condition == "restore_once":
            env.unwrapped.set_state_dict(common.batch(state0))
        initial = snapshot(env)
        n = 0
        for i, action in enumerate(action_rows[:MAX_CASE_CONTROLS]):
            if counter["controls"] >= MAX_TOTAL_CONTROLS:
                raise RuntimeError("registered aggregate native-control cap reached")
            counter["controls"] += 1
            n += 1
            result = env.step(action)
            traces.append({"index": i, "submitted_action": action.tolist(),
                           "terminated": bool(result[2]), "truncated": bool(result[3]),
                           "charger_pose": jsonable(env.unwrapped.charger.pose.raw_pose),
                           "tcp_pose": jsonable(env.unwrapped.agent.tcp.pose.raw_pose)})
            if result[2] or result[3]:
                break
        terminal = jsonable(env.unwrapped.evaluate())
        report = {"condition": condition, "source_episode_id": eid, "label": label,
                  "seed": seed, "source_h5_sha256": sha(SOURCE),
                  "source_json_sha256": sha(SOURCE_JSON), "source_action_count": len(action_rows),
                  "controls": n, "initial_state_restores": int(condition == "restore_once"),
                  "fresh_reset_snapshot": fresh, "first_control_snapshot": initial,
                  "terminal": terminal, "trace": traces, "elapsed_seconds": time.monotonic()-start}
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        (out / "trace.jsonl").write_text("".join(json.dumps(t, allow_nan=False)+"\n" for t in traces))
        return report
    finally:
        env.close()


def run_live(eid, label, seed, out, counter):
    from examples.baselines.constrain.peg_orientation_preflight import install_mplib_base_pose_compat
    install_mplib_base_pose_compat()
    from mani_skill.examples.motionplanning.panda.solutions.plug_charger import solve
    from mani_skill.utils.wrappers.record import RecordEpisode
    out.mkdir(parents=True, exist_ok=False)
    env = gym.make(plug_orientation.ENV_ID, obs_mode="state", control_mode="pd_joint_pos",
                   sim_backend="physx_cpu", reward_mode="sparse", max_episode_steps=200)
    recorder = RecordEpisode(env, output_dir=str(out), trajectory_name="live_expert",
                             save_trajectory=True, save_video=False, save_on_reset=False,
                             record_env_state=True, source_type="motionplanning",
                             source_desc="R77 fresh official planner diagnostic")
    class Bounded:
        def __init__(self): self.controls = 0; self.captures = []
        def __getattr__(self, key): return getattr(recorder, key)
        def reset(self, *, seed=None, options=None, **kwargs):
            opts = dict(options or {}); opts["target_yaw"] = 0.0
            result = recorder.reset(seed=seed, options=opts, **kwargs)
            self.captures.append(snapshot(recorder.unwrapped)); return result
        def step(self, action):
            if self.controls >= MAX_CASE_CONTROLS or counter["controls"] >= MAX_TOTAL_CONTROLS:
                raise RuntimeError("registered live expert control cap reached")
            self.controls += 1; counter["controls"] += 1
            return recorder.step(action)
    bounded = Bounded(); start = time.monotonic()
    try:
        result = solve(bounded, seed=seed, debug=False, vis=False)
        info = result[-1] if result != -1 else {}
        recorder.flush_trajectory(save=True)
        report = {"condition":"live_official_expert", "source_episode_id":eid, "label":label,
                  "seed":seed, "controls":bounded.controls, "terminal":jsonable(info),
                  "reset_capture":bounded.captures[0] if bounded.captures else None,
                  "elapsed_seconds":time.monotonic()-start}
        (out / "live_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
    finally:
        recorder.close()


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--output", required=True); a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True, exist_ok=False)
    meta, rows=_get_sources(); counter={"controls":0}; reports=[]
    archive=root/"source_archive"; archive.mkdir()
    closure={"experiment/plug_replay_reset_live_r77.py":Path(__file__).resolve(),
             "experiment/plug_orientation.py":Path(plug_orientation.__file__).resolve(),
             "experiment/plug_orientation_preflight.py":Path(__import__("examples.baselines.constrain.plug_orientation_preflight",fromlist=["x"]).__file__).resolve(),
             "experiment/peg_orientation_preflight.py":Path(__import__("examples.baselines.constrain.peg_orientation_preflight",fromlist=["x"]).__file__).resolve()}
    for name in ("mani_skill.envs.tasks.tabletop.plug_charger",
                 "mani_skill.examples.motionplanning.panda.solutions.plug_charger",
                 "mani_skill.examples.motionplanning.panda.motionplanner",
                 "mani_skill.trajectory.replay_trajectory",
                 "mani_skill.utils.wrappers.record",
                 "mani_skill.agents.controllers.pd_joint_pos"):
        spec=importlib.util.find_spec(name)
        if spec and spec.origin and Path(spec.origin).is_file(): closure[f"native/{name.replace('.', '/')}.py"]=Path(spec.origin)
    source_hashes={}
    for rel,src in closure.items():
        dst=archive/rel; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(src.read_bytes()); source_hashes[rel]=sha(dst)
    registration={"protocol":"plug_reset_state_live_r77", "source_h5":str(SOURCE),
      "source_h5_sha256":sha(SOURCE), "source_json_sha256":sha(SOURCE_JSON),
      "old_source_env_kwargs":meta["env_info"]["env_kwargs"], "cases":[{"episode_id":i,"label":l,"seed":s} for i,l,s in CASES],
      "order":["fresh_reset_source_actions","fresh_reset_restore_source_state0_source_actions","fresh_live_official_expert"],
      "per_case_control_cap":MAX_CASE_CONTROLS,"aggregate_control_cap":MAX_TOTAL_CONTROLS,
      "no_intermediate_state_restore":True,"source_archive_sha256":source_hashes,
      "started_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    (root/"preregistration.json").write_text(json.dumps(registration,indent=2)+"\n")
    for eid,label,seed in CASES:
        acts,state0=rows[eid]
        for cond in ("fresh_reset","restore_once"):
            reports.append(run_case(eid,label,seed,cond,root/f"episode_{eid}_{label}_{cond}",acts,state0,counter))
        reports.append(run_live(eid,label,seed,root/f"episode_{eid}_{label}_live",counter))
    summary={"reports":reports,"total_controls":counter["controls"],"control_cap":MAX_TOTAL_CONTROLS,
             "source_h5_sha256_after":sha(SOURCE),"source_json_sha256_after":sha(SOURCE_JSON)}
    (root/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"total_controls":counter["controls"],"cases":len(reports)},sort_keys=True))

if __name__ == "__main__": main()
