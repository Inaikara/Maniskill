"""No-step audit of actual restored TCP/cylinder geometry for R47."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


PICK_SCENES = Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes")
SPHERE_SCENES = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")


def audit_task(task):
    import gymnasium as gym
    original_make = gym.make
    def render_only_make(*args, **kwargs):
        kwargs["obstacle_mode"] = "render_only"
        return original_make(*args, **kwargs)
    gym.make = render_only_make
    if task == "pickcube":
        from examples.baselines.constrain.pickcube_scenes import _tree_hash as native_tree_hash
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
        from examples.baselines.constrain.evaluate_pickcube_paired import _native_env
        catalog = TransportDetourSceneSet(PICK_SCENES)
        make_env = _native_env
    else:
        from examples.baselines.constrain.placesphere_scenes import _tree_hash as native_tree_hash
        from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
        from examples.baselines.constrain.evaluate_placesphere_paired import _make_env
        catalog = TransportDetourSceneSet(SPHERE_SCENES)
        make_env = _make_env

    rows = []
    for index in range(len(catalog)):
        scene = catalog.load(index)
        env = make_env(scene)
        try:
            restored = catalog.restore(index, env)
            env = restored[0] if isinstance(restored, tuple) else env
            base = env.unwrapped
            state = base.get_state_dict()
            controller = base.agent.controller.get_state()
            meta = scene["metadata"]
            center = np.asarray(base._obstacle_center, dtype=np.float64).reshape(-1, 2)[0]
            expected = np.asarray(meta["cylinder"]["center_xy"], dtype=np.float64)
            radius = float(base.obstacle_collision_radius)
            tcp_actor = getattr(base.agent, "tcp", None)
            if tcp_actor is None:
                tcp_actor = base.agent.robot.links_map["panda_hand_tcp"]
            tcp = np.asarray(tcp_actor.pose.p, dtype=np.float64).reshape(-1, 3)[0, :2]
            state_key = "state_sha256"
            actual_state_hash = native_tree_hash(state)
            controller_hash = native_tree_hash(controller)
            visual_actor = base.obstacle._objs[0]
            visible = visual_actor.find_component_by_type(__import__("sapien").render.RenderBodyComponent)
            visible_center = np.asarray(visual_actor.pose.p, dtype=np.float64).reshape(-1, 3)[0, :2]
            collision_shapes = sum(len(body.collision_shapes) for body in base.obstacle._bodies)
            margin = float(getattr(base, "safety_margin", 0.005))
            distance = float(np.linalg.norm(tcp - center))
            row = {
                "task": task, "scene_index": index,
                "scene_state_sha256": meta[state_key], "restored_state_sha256": actual_state_hash,
                "scene_controller_sha256": meta["controller_sha256"],
                "restored_controller_sha256": controller_hash,
                "state_match": actual_state_hash == meta[state_key],
                "controller_match": controller_hash == meta["controller_sha256"],
                "tcp_xy": tcp.tolist(), "center_xy_task_state": center.tolist(),
                "center_xy_scene_metadata": expected.tolist(), "center_xy_visible_actor": visible_center.tolist(),
                "center_matches": bool(np.allclose(center, expected, atol=1e-6, rtol=0)
                                        and np.allclose(center, visible_center, atol=1e-6, rtol=0)),
                "cylinder_radius_m": radius, "tcp_only_margin_m": margin,
                "tcp_only_forbidden_radius_m": radius + margin,
                "distance_to_center_m": distance,
                "signed_gap_to_actual_cylinder_m": distance - radius,
                "signed_gap_to_tcp_only_profile_m": distance - radius - margin,
                "visible_cylinder": bool(visible is not None and visible.visibility > 0),
                "collision_shapes": collision_shapes,
                "zero_step": True,
            }
            if not row["state_match"] or not row["controller_match"] or not row["center_matches"]:
                raise ValueError(f"restored scene mismatch: {task} {index}: {row}")
            rows.append(row)
        finally:
            env.close()
    gym.make = original_make
    return rows


def _tree_hash(value):
    digest = hashlib.sha256()
    def visit(node):
        if isinstance(node, dict):
            for key in sorted(node):
                digest.update(str(key).encode())
                visit(node[key])
        elif hasattr(node, "detach"):
            visit(node.detach().cpu().numpy())
        elif isinstance(node, np.ndarray):
            digest.update(str(node.dtype).encode())
            digest.update(np.asarray(node.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(node).tobytes())
        else:
            digest.update(json.dumps(node, sort_keys=True).encode())
    visit(value)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = {task: audit_task(task) for task in ("pickcube", "placesphere")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({task: {"count": len(rows), "initial_actual_disk_violations": [
        row["scene_index"] for row in rows if row["signed_gap_to_actual_cylinder_m"] < 0],
        "inside_optional_margin_only": [row["scene_index"] for row in rows
            if row["signed_gap_to_actual_cylinder_m"] >= 0 and row["signed_gap_to_tcp_only_profile_m"] < 0]}
        for task, rows in results.items()}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
