"""Frozen selected-policy evaluation with a measured-TCP, per-step XY filter."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.baselines.constrain.inference_xy_constraint_r45 import XYConstraint, validate_tcp_trace

_ACTIVE_TRACES = []

def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _radius(base, task):
    cylinder = float(base.obstacle_collision_radius)
    margin = float(getattr(base, "safety_margin", 0.005))
    if task == "pickcube":
        return cylinder + float(base._cube_xy_radius) + margin
    return cylinder + float(base.radius) + margin


def instrument_env(env, task, *, control_enabled=True):
    """Wrap the native step seam; filter each just-in-time action using live TCP."""
    base = env.unwrapped
    if getattr(base, "obstacle_mode", None) != "render_only":
        raise ValueError("XY evaluation requires an explicitly render_only environment")
    obstacle = base.obstacle
    if any(len(body.collision_shapes) != 0 for body in obstacle._bodies):
        raise ValueError("render_only obstacle unexpectedly has collision shapes")
    import sapien
    visual = obstacle._objs[0].find_component_by_type(sapien.render.RenderBodyComponent)
    if visual is None or visual.visibility <= 0:
        raise ValueError("render_only obstacle visual is absent or invisible")
    actor = getattr(base.agent, "tcp", None)
    if actor is None:
        actor = base.agent.robot.links_map["panda_hand_tcp"]
    trace = []
    original_step = env.step
    center = _array(base._obstacle_center).reshape(-1, 2)[0].astype(float)
    radius = _radius(base, task)
    constraint = XYConstraint(translation_scale_m=0.1, action_limit=0.35)

    def step(action):
        requested = _array(action).reshape(1, -1)[0].astype(float)
        tcp = _array(actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
        filtered = constraint.filter_action(requested, tcp, center, radius)
        result = original_step(filtered.reshape(1, 7))
        info = result[-1]
        tcp_after = _array(actor.pose.p).reshape(-1, 3)[0, :2].astype(float)
        trace.append({"step": len(trace) + 1, "tcp_xy_before": tcp.tolist(),
                      "tcp_xy_after": tcp_after.tolist(), "requested_action": requested.tolist(),
                      "filtered_action": filtered.tolist(),
                      "native_success": bool(np.asarray(info.get("native_success", False)).reshape(-1)[0]),
                      "info": _jsonable(info)})
        return result

    env.step = step
    env._render_only_xy_trace = trace
    env._render_only_xy_center = center.tolist()
    env._render_only_xy_radius = float(radius)
    _ACTIVE_TRACES.append((trace, center.tolist(), float(radius)))
    return trace


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    value = _array(value)
    return value.item() if value.ndim == 0 else value.tolist()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def freeze_registration(path, task, indices, sources, inputs=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    body = {"format": "render_only_xy_eval_r46_v1", "task": task,
            "method": "ConstrainNet+XY constraint", "indices": [int(i) for i in indices],
            "constraint_profile": {"controller": "pd_ee_delta_pose", "frame": "world XY",
                "translation_scale_m_per_action_unit": 0.1, "xy_action_limit": 0.35,
                "non_xy_channels": "pass through", "pickcube_radius": "cylinder + sqrt(2)*cube_half_size + safety_margin",
                "placesphere_radius": "cylinder + sphere_radius + safety_margin",
                "body_collision_gate": False, "obstacle_mode": "render_only"},
            "formal_or_generalization_claim": False,
            "inputs": inputs or {},
            "sources": {name: {"path": str(Path(src)), "sha256": sha256(src)}
                        for name, src in sorted(sources.items())}}
    body["registration_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def validate_registration(path):
    reg = json.loads(Path(path).read_text())
    body = dict(reg)
    recorded = body.pop("registration_sha256", None)
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if recorded != actual:
        raise ValueError("registration checksum mismatch")
    for name, item in reg["sources"].items():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"registered source mismatch: {name}")
    return reg


def summarize_rows(rows, requested):
    complete = [r for r in rows if r.get("status") == "evaluated"]
    return {"denominator": int(requested), "evaluated": len(complete),
            "runtime_errors": int(requested) - len(complete),
            "native_success_count": sum(bool(r.get("native_success")) for r in complete),
            "xy_zero_count": sum(bool(r.get("xy_zero")) for r in complete)}


def _install_render_only_step_wrapper(task):
    import gymnasium as gym
    original_make = gym.make

    def make(*args, **kwargs):
        kwargs["obstacle_mode"] = "render_only"
        env = original_make(*args, **kwargs)
        instrument_env(env, task)
        return env
    gym.make = make
    return original_make


def _evaluate_one(task, index, device):
    _ACTIVE_TRACES.clear()
    if task == "pickcube":
        from examples.baselines.constrain import pickcube_body_hint_round41 as r41
        from examples.baselines.constrain import pickcube_xy_gauge_round40 as r40
        from examples.baselines.constrain import pickcube_zscore_round37 as engine
        from examples.baselines.constrain.pickcube_clearance_eval_r44 import (
            CONTROL, STATS_PATH, BASE, DEV_SCENES,
        )
        from examples.baselines.constrain.pickcube_clearance_train_r44 import (
            TRAIN_FORMAT, MODEL_KIND, TOTAL, EMA_POWER,
        )
        from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG, body_hint_from_raw
        from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
        from examples.baselines.constrain.evaluate_pickcube_paired import PickCubePairedEvaluator
        import torch
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        from diffusers.training_utils import EMAModel
        r41._install_profile(); r40._install_evaluation_loaders()
        ck = torch.load(CONTROL, map_location="cpu", weights_only=False)
        cfg = ck.get("config", {})
        if (ck.get("format") != TRAIN_FORMAT or cfg.get("model_kind") != MODEL_KIND
                or cfg.get("horizon") != 8 or ck.get("step") != TOTAL
                or ck.get("ema") is None or cfg.get("selected_weights") != "ema"):
            raise ValueError("R44 selected EMA checkpoint profile mismatch")
        model = engine.build_constrain_denoiser(CONTROL_CONFIG, device="cpu")
        model.load_state_dict(ck["model"], strict=True)
        ema = EMAModel(parameters=model.trainable_parameters(), power=EMA_POWER)
        ema.load_state_dict(ck["ema"]); ema.copy_to(model.trainable_parameters())
        model.to(device).eval()
        stats = engine.load_stats(STATS_PATH)
        scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
                                  clip_sample=True, prediction_type="epsilon")
        scheduler.set_timesteps(100)
        class Adapter:
            def __init__(self):
                self.device = torch.device(device); self.model = model; self.stats = stats
                self.scheduler = scheduler
                self.metadata = {"execution_horizon": 8}
            def base_cond(self, raw):
                raw = torch.as_tensor(raw, dtype=torch.float32, device=self.device)
                if raw.ndim == 2: raw = raw[None]
                if raw.shape[-2:] != (2, 45): raise ValueError("R44 actor requires (B,2,45)")
                return raw
            def get_action(self, raw):
                cond = engine.normalize_observation(raw[..., :42], self.stats).flatten(1)
                hint = body_hint_from_raw(raw)[:, None].expand(-1, 16, -1)
                sample = torch.randn((len(raw), 16, 7), device=self.device)
                with torch.no_grad():
                    for timestep in self.scheduler.timesteps:
                        sample = self.scheduler.step(
                            self.model(sample, timestep, global_cond=cond, hint=hint),
                            timestep, sample).prev_sample
                return sample[:, 1:9]
            @staticmethod
            def map_action_ee_delta_pose(action): return torch.as_tensor(action)
        catalog = TransportDetourSceneSet(DEV_SCENES)
        scene = catalog.load(index)
        runner = PickCubePairedEvaluator(Adapter(), catalog, "constrain_dp", ".", device, max_steps=500)
        row = runner.run_episode(index, scene)
    else:
        from examples.baselines.constrain import placesphere_base_dp as base_mod
        from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
        from examples.baselines.constrain.evaluate_placesphere_paired import PlaceSpherePairedEvaluator
        ckpt = Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt")
        base = Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base_verified/final.pt")
        adapter = PlaceSphereConstrainAdapter(ckpt, device=device, control_enabled=True,
                                              base_checkpoint=base, sampler="ddim", eta=0.0)
        from examples.baselines.constrain.placesphere_action_codec import CONTROLLER_RANGE
        from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet
        transport_catalog = TransportDetourSceneSet("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
        class Catalog:
            def __len__(self): return len(transport_catalog)
            def load(self, i): return transport_catalog.load(i)
            def restore(self, i, env):
                env, result = transport_catalog.restore(i, env)
                scene = transport_catalog.load(i)
                return env, {"raw_observation": scene["raw_observation"],
                             "bit_exact": result["bit_exact"], "norms": result["norms"]}
        catalog = Catalog()
        scene = catalog.load(index)
        runner = PlaceSpherePairedEvaluator(adapter, catalog, ".", device)
        row = runner.run_episode(index, scene)
    traces = [item for item in _ACTIVE_TRACES if item[0]]
    trace, center, radius = traces[-1] if traces else ([], [0, 0], 0.03)
    points = ([trace[0]["tcp_xy_before"]] + [step["tcp_xy_after"] for step in trace]) if trace else []
    report = validate_tcp_trace(points, center, radius,
                                first_success_step=next((i for i, step in enumerate(trace, 1)
                                                         if step["native_success"]), None)) if points else {
        "zero_xy_violations": False, "violating_samples": [], "violating_segments": []}
    row["native_success"] = bool(row.get("result", {}).get("native_success", False))
    row["actual_tcp_trace"] = trace
    row["xy_constraint"] = {"center_xy": center, "forbidden_radius_m": radius,
                             "controller_translation_scale_m": 0.1,
                             "controller_pose_action_limit": 0.35,
                             "visual_only_obstacle": True, "collision_shapes": 0}
    row["xy_validation"] = report
    row["xy_zero"] = bool(report["zero_xy_violations"])
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pickcube", "placesphere"), required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    source_paths = {"runner": Path(__file__),
                    "filter": Path(__file__).with_name("inference_xy_constraint_r45.py"),
                    "controller": Path("mani_skill/agents/robots/panda/panda.py")}
    if args.task == "pickcube":
        from examples.baselines.constrain.pickcube_clearance_eval_r44 import CONTROL, DEV_SCENES, STATS_PATH
        checkpoint = CONTROL
        scenes = Path(DEV_SCENES)
        source_paths.update({"task": Path("examples/baselines/constrain/pickcube_obstacle.py"),
                             "actor": Path("examples/baselines/constrain/pickcube_clearance_train_r44.py"),
                             "rollout": Path("examples/baselines/constrain/evaluate_pickcube_paired.py")})
        input_paths = {"checkpoint": checkpoint, "normalizer": STATS_PATH,
                       "scene_manifest": scenes / "manifest.json", "scene_h5": scenes / "scenes.h5"}
        from examples.baselines.constrain.pickcube_clearance_train_r44 import TRAIN_REG
        from examples.baselines.constrain import pickcube_zscore_round37 as engine
        input_paths.update({"training_registration": TRAIN_REG,
                            "data_h5": Path(engine.DATA_H5),
                            "data_manifest": Path(engine.DATA_MANIFEST)})
    else:
        checkpoint = Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt")
        scenes = Path("outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes")
        source_paths.update({"task": Path("examples/baselines/constrain/placesphere_obstacle.py"),
                             "actor": Path("examples/baselines/constrain/placesphere_constrain.py"),
                             "rollout": Path("examples/baselines/constrain/evaluate_placesphere_paired.py"),
                             "scene_loader": Path("examples/baselines/constrain/placesphere_transport_detour.py")})
        input_paths = {"checkpoint": checkpoint,
                       "base_checkpoint": Path("outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base_verified/final.pt"),
                       "scene_manifest": scenes / "manifest.json", "scene_h5": scenes / "scenes.h5"}
        input_paths.update({"training_registration": Path("outputs/exp/20261027_224727_placesphere_action_codec_r30/training_preregistration.json"),
                            "data_h5": Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/trajectory.h5"),
                            "data_manifest": Path("demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3/manifest.json"),
                            "checkpoint_correction": Path("outputs/exp/20261027_224727_placesphere_action_codec_r30/checkpoint_metadata_correction.json")})
    if args.register:
        if args.output.exists():
            raise FileExistsError(args.output)
        freeze_registration(args.registration, args.task, args.indices, source_paths,
                            {name: {"path": str(path), "sha256": sha256(path)}
                             for name, path in input_paths.items()})
        return
    reg = validate_registration(args.registration)
    if (reg.get("task") != args.task or reg.get("indices") != args.indices
            or any(sha256(path) != item["sha256"] for name, item in reg["inputs"].items()
                   for path in [item["path"]])):
        raise ValueError("registered task/indices/input identity mismatch")
    original_make = _install_render_only_step_wrapper(args.task)
    rows = []
    try:
        for index in args.indices:
            try:
                rows.append(_evaluate_one(args.task, index, args.device))
            except BaseException as exc:
                rows.append({"scene_index": index, "status": "runtime_error",
                             "error": f"{type(exc).__name__}: {exc}"})
    finally:
        import gymnasium as gym
        gym.make = original_make
    # Integration recording is completed per scene in the evaluators' native traces.
    args.output.mkdir(parents=True, exist_ok=False)
    summary = summarize_rows(rows, len(args.indices))
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2, default=_jsonable) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
