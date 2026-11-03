"""Replay retained selected-profile actions into inspection videos.

This does not resample a policy. It restores each frozen scene in render-only
mode and executes the saved native actions from the retained result rows.
"""

import argparse
import hashlib
import json
import signal
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _frame(env):
    image = env.render()
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image).squeeze()
    if image.ndim == 4:
        image = image[0]
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(info, key):
    value = info.get(key, False)
    return bool(np.asarray(value).reshape(-1)[0])


def _make_env(task, scene):
    import gymnasium as gym

    cylinder = scene["metadata"]["cylinder"]
    if task == "pickcube":
        from examples.baselines.constrain.pickcube_expert import make_pickcube_env

        original = make_pickcube_env

        def make(render_mode=None, *, obstacle_radius=0.03, obstacle_height=0.25):
            return gym.make(
                "PickCubeObstacle-v1", num_envs=1, obs_mode="state", sim_backend="physx_cpu",
                robot_uids="panda", robot_init_qpos_noise=0.02, control_mode="pd_ee_delta_pose",
                max_episode_steps=500, render_mode=render_mode, obstacle_radius=obstacle_radius,
                obstacle_height=obstacle_height, obstacle_mode="render_only",
            )

        return make(render_mode="rgb_array", obstacle_radius=cylinder["radius"],
                    obstacle_height=cylinder["height"]), original
    from examples.baselines.constrain.placesphere_pilot import PlaceSpherePilotConfig, make_env

    config = PlaceSpherePilotConfig(
        max_steps=200, obstacle_radius=cylinder["radius"], obstacle_height=cylinder["height"],
        safety_margin=0.005,
    )
    original_make = gym.make

    def render_only_make(*args, **kwargs):
        kwargs["obstacle_mode"] = "render_only"
        return original_make(*args, **kwargs)

    gym.make = render_only_make
    try:
        return make_env("rgb_array", config), original_make
    except BaseException:
        gym.make = original_make
        raise


def render(task, rows_path, scenes, output, episode_seconds=600):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows = json.loads(Path(rows_path).read_text())
    if task == "pickcube":
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
    else:
        from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
    catalog = TransportDetourSceneSet(scenes)
    records = []
    for row in rows:
        index = int(row["scene_index"])
        scene = catalog.load(index)
        env, original = _make_env(task, scene)
        previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("episode render deadline")))
        signal.alarm(episode_seconds)
        video = output / f"scene_{index:03d}.mp4"
        temporary = output / f".scene_{index:03d}.tmp.mp4"
        writer = imageio.get_writer(str(temporary), fps=20, codec="libx264", macro_block_size=1)
        try:
            catalog.restore(index, env)
            base = env.unwrapped
            if any(len(body.collision_shapes) != 0 for body in base.obstacle._bodies):
                raise RuntimeError("render-only replay unexpectedly has collision shapes")
            writer.append_data(_frame(env))
            native = False
            trace = row.get("actual_tcp_trace") or []
            actions = [step["filtered_action"] for step in trace] if trace else row["actions_native_pd_ee_delta_pose"]
            if len(actions) != len(row["actions_native_pd_ee_delta_pose"]):
                raise ValueError(f"scene {index}: executed trace length differs from recorded actions")
            for action in actions:
                _obs, _reward, _terminated, _truncated, info = env.step(np.asarray(action, dtype=np.float32)[None])
                native = native or _bool(info, "native_success")
                writer.append_data(_frame(env))
            writer.close()
            writer = None
            temporary.replace(video)
            records.append({
                "scene_index": index, "scene_id": row.get("scene_id"), "video": video.name,
                "steps": len(actions), "recorded_native_success": bool(row.get("native_success")),
                "replay_native_success": native, "matches_recorded_success": native == bool(row.get("native_success")),
                "video_sha256": _sha(video),
            })
            print(json.dumps(records[-1]), flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            if writer is not None:
                writer.close()
            if temporary.exists():
                temporary.unlink()
            env.close()
            if task == "placesphere":
                import gymnasium as gym
                gym.make = original
    report = {"task": task, "method": rows[0].get("method_label"), "rows": str(rows_path),
              "rows_sha256": _sha(rows_path), "scenes": str(scenes), "episodes": records,
              "complete": len(records) == len(rows) and all(item["matches_recorded_success"] for item in records)}
    (output / "replay_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise RuntimeError("retained-action replay did not match recorded native success")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pickcube", "placesphere"), required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.task, args.rows, args.scenes, args.output)


if __name__ == "__main__":
    main()
