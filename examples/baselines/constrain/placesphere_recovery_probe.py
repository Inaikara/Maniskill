"""Feasibility of resuming the existing expert from a near-terminal train state.

A closed-grip descent back to the same container target is not a recovery demo.
"""

import json
from pathlib import Path

import numpy as np

PROBE_OFFSET_M = 0.02
DESCEND_DZ = -0.05


def _descent(actions):
    rows = [np.asarray(action, dtype=np.float64).reshape(7) for action in list(actions)[:8]]
    if len(rows) < 8:
        raise ValueError("probe needs eight continuation actions")
    return all(float(row[6]) < 0 for row in rows) and any(float(row[2]) < DESCEND_DZ for row in rows)


def probe_decision(nominal_actions, perturbed_actions, *, perturbed_contact, endpoint_open):
    nominal = _descent(nominal_actions)
    perturbed = _descent(perturbed_actions)
    same = nominal and perturbed
    collect = not same and not perturbed_contact and bool(endpoint_open)
    return {
        "nominal_closed_descent": nominal,
        "perturbed_closed_descent": perturbed,
        "same_closed_descent": same,
        "perturbed_contact": bool(perturbed_contact),
        "endpoint_open_observed": bool(endpoint_open),
        "collect_this_expert": collect,
        "reward_claim": False,
        "uses_dev_scenes": False,
        "reason": "same_closed_descent" if same else ("perturbed_contact" if perturbed_contact else "different_action"),
    }


def _json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_probe(output_dir, *, scene_dir, scene_index=0, device="cpu"):
    import signal
    import time
    import h5py
    import torch
    from examples.baselines.constrain.diagnose_placesphere_failure import _child_pids, _terminate_new_children
    from examples.baselines.constrain.placesphere_learning_pilot import FROZEN_DEV_SCENES, _bool_info
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
    from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    catalog = TransportDetourSceneSet(scene_dir)
    row = catalog.manifest["scenes"][scene_index]
    dev = json.loads((Path(FROZEN_DEV_SCENES) / "manifest.json").read_text())
    dev_seeds = {int(item["seed"]) for item in dev["scenes"]}
    if int(row["seed"]) in dev_seeds:
        raise RuntimeError("probe scene is a dev scene")
    target = np.asarray(row["expert"]["fork_targets"]["container_approach_xyz"], dtype=np.float64)
    started = time.monotonic()
    before = set(_child_pids())

    def deadline(_signum, _frame):
        raise TimeoutError("probe wall exceeded")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, 180)
    env = make_env(config=PlaceSpherePilotConfig(
        max_steps=320, obstacle_radius=float(row["cylinder"]["radius"]),
        obstacle_height=float(row["cylinder"]["height"])))
    try:
        env, restored = catalog.restore(scene_index, env)
        if not restored.get("bit_exact"):
            raise RuntimeError("train scene restore is not bit-exact")
        with h5py.File(catalog.directory / catalog.manifest["scenes_file"], "r") as handle:
            actions = np.asarray(handle[row["scene_id"]]["expert_actions"], dtype=np.float32)
        base = env.unwrapped
        stop = None
        for step, action in enumerate(actions):
            tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
            if stop is None and float(action[6]) < 0 and float(tcp[2]) > float(target[2]) + 0.03:
                if float(np.linalg.norm(tcp[:2] - target[:2])) < 0.05:
                    stop = step
                    break
            env.step(np.asarray(action, dtype=np.float32)[None])
        if stop is None:
            raise RuntimeError("near-terminal closed-grip state was not reached")
        nominal = []
        contact = False
        for _ in range(8):
            tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
            delta = target - tcp
            command = np.zeros(7, dtype=np.float32)
            command[:3] = np.clip(delta / 0.1, -0.35, 0.35)
            command[6] = -1.0
            nominal.append(command.copy())
            _, _, _, _, info = env.step(command[None])
            contact = contact or _bool_info(info, "obstacle_collision")
        nominal_contact = contact
        env.close()
        env = make_env(config=PlaceSpherePilotConfig(
            max_steps=320, obstacle_radius=float(row["cylinder"]["radius"]),
            obstacle_height=float(row["cylinder"]["height"])))
        env, restored = catalog.restore(scene_index, env)
        base = env.unwrapped
        for action in actions[:stop]:
            env.step(np.asarray(action, dtype=np.float32)[None])
        lateral = np.zeros(7, dtype=np.float32)
        lateral[0] = PROBE_OFFSET_M / 0.1
        lateral[6] = -1.0
        env.step(lateral[None])
        perturbed = []
        contact = False
        for _ in range(8):
            tcp = base.agent.tcp.pose.p[0].detach().cpu().numpy()
            delta = target - tcp
            command = np.zeros(7, dtype=np.float32)
            command[:3] = np.clip(delta / 0.1, -0.35, 0.35)
            command[6] = -1.0
            perturbed.append(command.copy())
            _, _, _, _, info = env.step(command[None])
            contact = contact or _bool_info(info, "obstacle_collision")
        decision = probe_decision(nominal, perturbed, perturbed_contact=contact, endpoint_open=False)
        decision["nominal_contact"] = nominal_contact
        decision["scene_id"] = row["scene_id"]
        decision["seed"] = int(row["seed"])
        decision["stop_step"] = int(stop)
        decision["elapsed_s"] = time.monotonic() - started
        decision["perturbation_m"] = PROBE_OFFSET_M
        decision["perturbation_is_a_packaged_demo"] = False
        _json_atomic(output / "probe.json", {
            "decision": decision,
            "nominal_actions": [row.tolist() for row in nominal],
            "perturbed_actions": [row.tolist() for row in perturbed],
        })
        return decision
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        env.close()
        _terminate_new_children(before)
