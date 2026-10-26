"""Measure native spin decay beyond the fixed Round-20 feedback window."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

R19 = Path("outputs/exp/20261026_100200_placesphere_matched_terminal_r19")
CASES = ("placesphere-003", "placesphere-019")


def _case_rows(scene_id):
    document = json.loads((R19 / "dev30_validation.json").read_text())
    return next(row for row in document["rows"] if row["layout"]["scene_id"] == scene_id)


def run_case(scene_id, *, continuation_steps=2000, episode_wall_s=160):
    import torch
    from examples.baselines.constrain.placesphere_matched_terminal import (
        NATIVE_ANG_THRESH_RAD_S, NATIVE_LIN_THRESH_M_S, _child_pids,
        _kill_new_children, park_execution_snapshot,
    )
    from examples.baselines.constrain.pickcube_expert import restore_execution_state, snapshot_execution_state
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env
    from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet

    row = _case_rows(scene_id)
    layout = row["layout"]
    actions = np.asarray(row["positive_direct"]["actions"], dtype=np.float32)
    scene_manifest = json.loads((Path(layout["catalog"]) / "manifest.json").read_text())
    scene = scene_manifest["scenes"][int(layout["index"])]
    before = set(_child_pids())
    env = None
    start = time.monotonic()
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("episode wall exceeded")))
    signal.setitimer(signal.ITIMER_REAL, float(episode_wall_s))
    try:
        env = make_env(config=PlaceSpherePilotConfig(
            max_steps=len(actions) + continuation_steps + 2,
            obstacle_radius=float(scene["cylinder"]["radius"]),
            obstacle_height=float(scene["cylinder"]["height"])))
        catalog = TransportDetourSceneSet(layout["catalog"])
        env, restored = catalog.restore(int(layout["index"]), env)
        if not restored.get("bit_exact"):
            raise RuntimeError("scene restore is not bit-exact")
        base = env.unwrapped
        initial, _ = park_execution_snapshot(snapshot_execution_state(env))
        restore_execution_state(env, initial)
        terminated = truncated = False
        for action in actions:
            _, _, terminated, truncated, _ = env.step(torch.as_tensor(action, device=base.device))
            if terminated or truncated:
                break
        def sample(index):
            info = base.evaluate()
            linear = float(torch.linalg.norm(base.obj.linear_velocity, dim=1)[0].item())
            angular = float(torch.linalg.norm(base.obj.angular_velocity, dim=1)[0].item())
            return {
                "continuation_step": int(index), "linear_speed_m_s": linear,
                "angular_speed_rad_s": angular,
                "linear_threshold_m_s": NATIVE_LIN_THRESH_M_S,
                "angular_threshold_rad_s": NATIVE_ANG_THRESH_RAD_S,
                "on_bin": bool(info["is_obj_on_bin"][0].item()),
                "grasped": bool(info["is_obj_grasped"][0].item()),
                "native_success": bool(info["native_success"][0].item()),
                "obstacle_contact_latched": bool(base._obstacle_collision[0].item()),
                "obstacle_crossing_latched": bool(base._forbidden_crossing[0].item()),
            }
        trace = [sample(0)]
        first_native = 0 if trace[0]["native_success"] else None
        for index in range(1, continuation_steps + 1):
            _, _, terminated, truncated, _ = env.step(torch.tensor(
                [0, 0, 0, 0, 0, 0, 1], dtype=torch.float32, device=base.device))
            current = sample(index)
            if index % 10 == 0 or current["native_success"] or terminated or truncated:
                trace.append(current)
            if current["native_success"]:
                first_native = index
                break
            if terminated or truncated:
                break
        return {
            "scene_id": scene_id, "catalog": layout["catalog"], "index": layout["index"],
            "seed": layout["seed"], "restore_bit_exact": True,
            "saved_direct_actions": len(actions),
            "saved_direct_actions_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
            "continuation_budget_steps": continuation_steps,
            "first_native_success_continuation_step": first_native,
            "trace": trace, "actual_cylinder": False,
            "contact_latch_cleared": False, "full_prefix_started_at_episode_start": True,
            "native_thresholds_changed": False,
            "termination_at_end": bool(terminated), "truncation_at_end": bool(truncated),
            "elapsed_s": time.monotonic() - start,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if env is not None:
            env.close()
        _kill_new_children(before)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--continuation-steps", type=int, default=2000)
    parser.add_argument("--episode-wall-s", type=int, default=160)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).read_bytes()
    prereg = {
        "kind": "bounded_long_neutral_settle_measurement",
        "parent_phase_b_identity": "b437c3f2d2bb426936459921de248c5e0bb32a459e4dcfaa03865e560135993b",
        "probe_code_sha256": hashlib.sha256(code).hexdigest(),
        "cases": list(CASES), "selection_use": False, "training_use": False,
        "episode_wall_s": args.episode_wall_s,
        "continuation_steps": args.continuation_steps,
        "argv": __import__("sys").argv,
        "scope": "Diagnostic only; open-neutral continuation from retained full positive trace; no motion or threshold change.",
    }
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    results = []
    for scene_id in CASES:
        result = run_case(scene_id, continuation_steps=args.continuation_steps,
                          episode_wall_s=args.episode_wall_s)
        results.append(result)
        (output / "partial.json").write_text(json.dumps({"results": results}, indent=2, sort_keys=True) + "\n")
    payload = {"complete": len(results) == len(CASES), "results": results,
               "selection_use": False, "training_use": False}
    (output / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": payload["complete"], "cases": len(results), "output": args.output_dir}))


if __name__ == "__main__":
    main()
