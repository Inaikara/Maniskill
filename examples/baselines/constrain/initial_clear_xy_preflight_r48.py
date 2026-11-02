"""Zero-step native restoration audit for the R48 initial-clear catalogs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(task, scenes_dir):
    import gymnasium as gym
    original_make = gym.make
    gym.make = lambda *args, **kwargs: original_make(*args, obstacle_mode="render_only", **kwargs)
    try:
        if task == "pickcube":
            from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
            catalog = TransportDetourSceneSet(scenes_dir)
        else:
            from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
            catalog = TransportDetourSceneSet(scenes_dir)
        rows = []
        for index in range(len(catalog)):
            scene = catalog.load(index)
            env, restored = catalog.restore(index)
            try:
                base = env.unwrapped
                tcp_actor = getattr(base.agent, "tcp", None)
                if tcp_actor is None:
                    tcp_actor = base.agent.robot.links_map["panda_hand_tcp"]
                tcp = np.asarray(tcp_actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
                center = np.asarray(base._obstacle_center).reshape(-1, 2)[0].astype(float)
                cylinder = scene["metadata"]["cylinder"]
                radius = float(base.obstacle_collision_radius)
                visual_actor = base.obstacle._objs[0]
                visible_component = visual_actor.find_component_by_type(
                    __import__("sapien").render.RenderBodyComponent)
                visible_center = np.asarray(visual_actor.pose.p).reshape(-1, 3)[0, :2]
                controller = base.agent.controller.get_state()
                from examples.baselines.constrain.pickcube_scenes import _tree_hash
                state_hash = _tree_hash(base.get_state_dict())
                controller_hash = _tree_hash(controller)
                admission = scene["metadata"]["initial_tcp_admission"]
                gap = float(np.linalg.norm(tcp-center)-radius-.005)
                metadata = scene["metadata"]
                start = np.asarray(metadata["start_xy"], dtype=float)
                goal = np.asarray(metadata.get("goal_xy", metadata.get("container_xy")), dtype=float)
                fraction = float(metadata["sampling_fraction"])
                direct_goal = metadata["direct_goal_xyz"]
                if task == "pickcube":
                    layout_valid = (-.18 <= start[0] <= -.02 and -.06 <= start[1] <= .12
                        and .02 <= goal[0] <= .18 and -.06 <= goal[1] <= .12
                        and np.asarray(direct_goal)[2] == .12 and radius == .03
                        and metadata["cylinder"]["height"] == .14)
                else:
                    extent = max(float(v) for v in base.block_half_size[1:])
                    layout_valid = (-.19 <= start[0] <= -.08 and -.13 <= start[1] <= .13
                        and .04 <= goal[0] <= .15 and -.13 <= goal[1] <= .13
                        and -.25 + extent <= goal[0] <= .25 - extent
                        and -.25 + extent <= goal[1] <= .25 - extent
                        and radius == .03 and metadata["cylinder"]["height"] == .08)
                vector = goal - start
                projected = float(np.dot(center-start, vector) / np.dot(vector, vector))
                layout_valid = bool(layout_valid and .4 <= fraction <= .6
                    and np.linalg.norm(goal-start) >= .14 and .4 <= projected <= .6)
                row = {"scene_index": index, "scene_id": scene["metadata"]["scene_id"],
                    "seed": scene["metadata"]["seed"], "tcp_xy": tcp.tolist(),
                    "center_xy": center.tolist(), "visible_center_xy": visible_center.tolist(),
                    "radius_m": radius, "profile_margin_m": .005,
                    "signed_initial_gap_m": gap, "state_sha256": state_hash,
                    "expected_state_sha256": scene["metadata"]["state_sha256"],
                    "controller_sha256": controller_hash,
                    "expected_controller_sha256": scene["metadata"]["controller_sha256"],
                    "center_matches_metadata": bool(np.allclose(center, cylinder["center_xy"], atol=1e-6, rtol=0)),
                    "visible_center_matches": bool(np.allclose(center, visible_center, atol=1e-6, rtol=0)),
                    "state_match": state_hash == scene["metadata"]["state_sha256"],
                    "controller_match": controller_hash == scene["metadata"]["controller_sha256"],
                    "initial_admission_hash_match": bool(np.allclose(tcp, admission["tcp_xy"], atol=1e-7, rtol=0)),
                    "collision_shapes": sum(len(body.collision_shapes) for body in base.obstacle._bodies),
                    "layout_contract_version": metadata.get("layout_contract_version"),
                    "layout_bounds_and_goal_height_match": layout_valid,
                    "restore_norms": restored.get("norms", {}) if isinstance(restored, dict) else {},
                    "zero_step": True}
                valid = (row["state_match"] and row["controller_match"] and row["center_matches_metadata"]
                         and row["visible_center_matches"] and row["initial_admission_hash_match"]
                         and row["collision_shapes"] == 0 and gap > 0 and layout_valid
                         and row["layout_contract_version"] == "transport_detour_training_bounds_v1")
                row["passed"] = bool(valid)
                rows.append(row)
                if not valid:
                    raise ValueError(f"native restore preflight failed for {task} scene {index}: {row}")
            finally:
                env.close()
        return rows
    finally:
        gym.make = original_make


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pickcube", "placesphere"), required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    prereg = args.output.parent / f"{args.task}_preflight_registration_r52.json"
    if prereg.exists():
        raise FileExistsError(prereg)
    source_paths = [Path(__file__), Path(__file__).with_name("initial_clear_xy_scenes.py"),
                    Path(__file__).with_name("render_only_xy_eval.py"),
                    Path("mani_skill/agents/robots/panda/panda.py")]
    snapshot_dir = args.output.parent / f"{args.task}_preflight_sources_r52"
    if snapshot_dir.exists():
        raise FileExistsError(snapshot_dir)
    snapshot_dir.mkdir(parents=True)
    source_records = {}
    for path in source_paths:
        snapshot = snapshot_dir / path.name
        shutil.copyfile(path, snapshot)
        source_records[str(path)] = {"sha256": _sha(path), "snapshot": str(snapshot),
                                     "snapshot_sha256": _sha(snapshot)}
    registration = {"format": "initial_clear_xy_restore_preflight_r52_v1", "task": args.task,
        "catalog": str(args.scenes), "sources": source_records,
        "inputs": {str(path): _sha(path) for path in (args.scenes/"manifest.json", args.scenes/"scenes.h5")},
        "command": f"python -m examples.baselines.constrain.initial_clear_xy_preflight_r48 --task {args.task} --scenes {args.scenes} --output {args.output}",
        "policy_rollouts": 0, "steps": 0}
    registration["registration_sha256"] = hashlib.sha256(json.dumps(registration, sort_keys=True).encode()).hexdigest()
    prereg.write_text(json.dumps(registration, indent=2, sort_keys=True)+"\n")
    rows = audit(args.task, args.scenes)
    report = {"task": args.task, "catalog": str(args.scenes), "catalog_manifest_sha256": _sha(args.scenes/"manifest.json"),
              "catalog_h5_sha256": _sha(args.scenes/"scenes.h5"), "checked": len(rows),
              "all_passed": all(row["passed"] for row in rows),
              "minimum_initial_profile_gap_m": min(row["signed_initial_gap_m"] for row in rows),
              "scenes": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "scenes"}, sort_keys=True), flush=True)
    if not report["all_passed"]:
        raise RuntimeError(f"restore preflight failed: {args.task}")


if __name__ == "__main__":
    main()
