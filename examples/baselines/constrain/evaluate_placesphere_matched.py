"""Evaluate matched-r21b PlaceSphere policies on the frozen 30-scene dev set."""

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import torch

from examples.baselines.constrain.placesphere_base_dp import PlaceSphereBaseDPAdapter, sha256_file
from examples.baselines.constrain.placesphere_constrain import PlaceSphereConstrainAdapter
from examples.baselines.constrain.placesphere_learning_pilot import (
    EpisodeDeadline, _episode_deadline, run_pilot_episode, summarize_pilot,
)
from examples.baselines.constrain.placesphere_transport_detour import TransportDetourSceneSet


def _read_rows(directory, count=30):
    directory = Path(directory)
    return [json.loads((directory / f"episode_{i:03d}.json").read_text()) for i in range(count)]


def _validate_primary_baseline(directory, base_sha, constrain_sha):
    base = _read_rows(Path(directory) / "base_dp")
    constrain = _read_rows(Path(directory) / "constrain_dp")
    for index, (left, right) in enumerate(zip(base, constrain)):
        if (left.get("scene_index") != index or right.get("scene_index") != index
                or left.get("scene_id") != right.get("scene_id")
                or left.get("seed") != right.get("seed")
                or not left.get("restored_bit_exact") or not right.get("restored_bit_exact")
                or left.get("checkpoint_sha256") != base_sha
                or right.get("checkpoint_sha256") != constrain_sha):
            raise ValueError(f"primary baseline identity/restore mismatch at scene {index}")
        left_trace, right_trace = left.get("closed_loop_trace", []), right.get("closed_loop_trace", [])
        if (not left_trace or not right_trace
                or len(left_trace[0].get("policy_observation", [])) != 42
                or len(right_trace[0].get("policy_observation", [])) != 42
                or left_trace[0]["policy_observation"] != right_trace[0]["policy_observation"]):
            raise ValueError(f"primary baseline initial observation mismatch at scene {index}")
    return {"episodes_per_policy": 30, "matching_scene_ids_seeds_and_initial_observations": True,
            "all_restored_bit_exact": True, "observation_dimension": 42}


def _validate_primary_scene_pairing(directory):
    base = _read_rows(Path(directory) / "base_dp")
    constrain = _read_rows(Path(directory) / "constrain_dp")
    for index, (left, right) in enumerate(zip(base, constrain)):
        if (left.get("scene_index") != index or right.get("scene_index") != index
                or left.get("scene_id") != right.get("scene_id")
                or left.get("seed") != right.get("seed")
                or not left.get("restored_bit_exact") or not right.get("restored_bit_exact")):
            raise ValueError(f"primary scene identity/restore mismatch at scene {index}")
        left_trace, right_trace = left.get("closed_loop_trace", []), right.get("closed_loop_trace", [])
        if (not left_trace or not right_trace
                or len(left_trace[0].get("policy_observation", [])) != 42
                or left_trace[0]["policy_observation"] != right_trace[0]["policy_observation"]):
            raise ValueError(f"primary initial observation mismatch at scene {index}")
    return {"episodes_per_policy": 30, "matching_scene_ids_seeds_and_initial_observations": True,
            "all_restored_bit_exact": True, "checkpoint_comparison": "new codec checkpoints intentionally differ"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _synchronize(adapter):
    if str(getattr(adapter, "device", "cpu")).startswith("cuda"):
        torch.cuda.synchronize(adapter.device)


def _transitions(old, new, metric):
    left = [bool(row.get("result", {}).get(metric, False)) for row in old]
    right = [bool(row.get("result", {}).get(metric, False)) for row in new]
    if metric in ("obstacle_contact", "xy_forbidden_crossing"):
        return {"cleared": sum(a and not b for a, b in zip(left, right)),
                "introduced": sum(not a and b for a, b in zip(left, right)),
                "both": sum(a and b for a, b in zip(left, right)),
                "neither": sum(not a and not b for a, b in zip(left, right))}
    return {"improved": sum(not a and b for a, b in zip(left, right)),
            "regressed": sum(a and not b for a, b in zip(left, right)),
            "both": sum(a and b for a, b in zip(left, right)),
            "neither": sum(not a and not b for a, b in zip(left, right))}


class _BasePolicy:
    def __init__(self, adapter):
        self.adapter = adapter
        self.metadata = adapter.metadata
        self.inference_calls = 0
        self.max_inference_wall_seconds = 0.0

    def get_action(self, history):
        queued = torch.as_tensor(history, dtype=torch.float32, device=self.adapter.device)[None]
        _synchronize(self.adapter)
        started = time.perf_counter()
        try:
            return self.adapter.get_action(queued[..., :39])
        finally:
            _synchronize(self.adapter)
            elapsed = time.perf_counter() - started
            self.inference_calls += 1
            self.max_inference_wall_seconds = max(self.max_inference_wall_seconds, elapsed)


class _ConstrainPolicy:
    def __init__(self, adapter, checkpoint_path):
        self.adapter = adapter
        self.metadata = dict(adapter.metadata)
        self.metadata["checkpoint_sha256"] = sha256_file(checkpoint_path)
        self.inference_calls = 0
        self.max_inference_wall_seconds = 0.0

    def get_action(self, history):
        queued = torch.as_tensor(history, dtype=torch.float32, device=self.adapter.device)[None]
        _synchronize(self.adapter)
        started = time.perf_counter()
        try:
            return self.adapter.get_action(self.adapter.base_cond(queued))
        finally:
            _synchronize(self.adapter)
            elapsed = time.perf_counter() - started
            self.inference_calls += 1
            self.max_inference_wall_seconds = max(self.max_inference_wall_seconds, elapsed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--constrain-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-baseline", type=Path, required=True)
    parser.add_argument("--secondary-baseline", type=Path)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    parser.add_argument("--ensemble-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--inference-preregistration", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--replan-stride", type=int, choices=(2, 8), default=8)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.ensemble_size == 4 and args.sampler != "ddim":
        raise ValueError("ensemble size 4 is registered only for DDIM")
    catalog = TransportDetourSceneSet(args.scene_dir)
    if len(catalog) != 30:
        raise ValueError(f"expected frozen 30-scene development set, found {len(catalog)}")
    preregistration = None
    preregistration_sha = None
    if args.sampler == "ddim":
        if args.inference_preregistration is None:
            raise ValueError("DDIM requires the frozen inference preregistration")
        preregistration = json.loads(args.inference_preregistration.read_text())
        preregistration_sha = _sha(args.inference_preregistration)
        source_paths = {
            "evaluate_placesphere_matched.py": Path(__file__),
            "placesphere_base_dp.py": Path(__file__).with_name("placesphere_base_dp.py"),
            "placesphere_constrain.py": Path(__file__).with_name("placesphere_constrain.py"),
            "placesphere_inference_sampler.py": Path(__file__).with_name("placesphere_inference_sampler.py"),
            "placesphere_learning_pilot.py": Path(__file__).with_name("placesphere_learning_pilot.py"),
            "placesphere_pilot.py": Path(__file__).with_name("placesphere_pilot.py"),
            "placesphere_transport_detour.py": Path(__file__).with_name("placesphere_transport_detour.py"),
            "placesphere_scenes.py": Path(__file__).with_name("placesphere_scenes.py"),
            "placesphere_obstacle.py": Path(__file__).with_name("placesphere_obstacle.py"),
            "diagnose_placesphere_failure.py": Path(__file__).with_name("diagnose_placesphere_failure.py"),
            "constrain_denoiser.py": Path(__file__).with_name("constrain_denoiser.py"),
            "constrain_ckpt.py": Path(__file__).with_name("constrain_ckpt.py"),
            "placesphere_action_codec.py": Path(__file__).with_name("placesphere_action_codec.py"),
        }
        actual_sources = {name: _sha(path) for name, path in source_paths.items()}
        registration_format = preregistration.get("format")
        if registration_format == "placesphere_r30_action_codec_ddim100_preregistration_v1":
            required_paths = {
                "scene_dir": "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes",
                "base_checkpoint": "outputs/ckpt/20261027_224727_placesphere_action_codec_r30_base_verified/final.pt",
                "constrain_checkpoint": "outputs/ckpt/20261027_224727_placesphere_action_codec_r30_constrain_verified_150k/final.pt",
                "primary_baseline": "outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval",
                "output_dir": "outputs/exp/20261027_224727_placesphere_action_codec_r30/paired_eval",
                "dataset": "demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3",
            }
            data_dir = Path(preregistration["data"]["dataset"])
            expected = preregistration["inputs"]
            base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
            constrain_payload = torch.load(args.constrain_checkpoint, map_location="cpu", weights_only=False)
            base_profile = base_payload.get("action_encoding")
            control_profile = constrain_payload.get("config", {}).get("action_encoding")
            if (str(args.scene_dir) != required_paths["scene_dir"]
                    or str(args.base_checkpoint) != required_paths["base_checkpoint"]
                    or str(args.constrain_checkpoint) != required_paths["constrain_checkpoint"]
                    or str(args.primary_baseline) != required_paths["primary_baseline"]
                    or str(args.output_dir) != required_paths["output_dir"]
                    or str(data_dir) != required_paths["dataset"]
                    or args.replan_stride != 2 or preregistration.get("protocol", {}).get("ensemble_size") != 1
                    or preregistration.get("protocol", {}).get("sampler") != "DDIMScheduler eta=0; 100 steps"
                    or preregistration.get("protocol", {}).get("primary_gate") != {"task_min": 27, "safe_min": 26, "contact_max": 3, "xy_max": 3}
                    or preregistration.get("source_sha256") != actual_sources
                    or sha256_file(preregistration.get("checkpoint_metadata_correction", {}).get("path", "")) != preregistration.get("checkpoint_metadata_correction", {}).get("sha256")
                    or _sha(args.base_checkpoint) != expected.get("base_checkpoint_sha256")
                    or _sha(args.constrain_checkpoint) != expected.get("constrain_checkpoint_sha256")
                    or base_profile != preregistration.get("action_encoding")
                    or control_profile != base_profile
                    or constrain_payload.get("config", {}).get("data_identity", {}).get("h5_sha256") != expected.get("data_h5_sha256")
                    or constrain_payload.get("config", {}).get("data_identity", {}).get("manifest_sha256") != expected.get("data_manifest_sha256")
                    or constrain_payload.get("config", {}).get("data_identity", {}).get("training_identity_sha256") != expected.get("training_identity_sha256")
                    or sha256_file(data_dir / "trajectory.h5") != expected.get("data_h5_sha256")
                    or sha256_file(data_dir / "trajectory.json") != expected.get("data_manifest_sha256")
                    or sha256_file(data_dir / "training_identity.json") != expected.get("training_identity_sha256")
                    or sha256_file(args.scene_dir / "manifest.json") != expected.get("scene_manifest_sha256")
                    or sha256_file(args.scene_dir / "scenes.h5") != expected.get("scene_h5_sha256")
                    or sha256_file(args.primary_baseline / "paired_summary.json") != expected.get("primary_summary_sha256")):
                raise ValueError("R30 action-codec evaluation preregistration identity mismatch")
            primary_pairing = _validate_primary_scene_pairing(args.primary_baseline)
            if (args.sampler != "ddim" or args.ensemble_size != 1
                    or preregistration.get("protocol", {}).get("stride") != 2
                    or preregistration.get("protocol", {}).get("max_control_steps") != 200
                    or preregistration.get("protocol", {}).get("episode_wall_seconds") != 300
                    or preregistration.get("protocol", {}).get("fixed_denominator_per_policy") != 30
                    or preregistration.get("budgets", {}).get("outer_wall_seconds") != 18000):
                raise ValueError("R30 evaluation protocol differs from the fixed registration")
            preregistration_sha = _sha(args.inference_preregistration)
            # Continue through the common strict checkpoint adapters and episode runner.
            inference_sampler, eta = "ddim", 0.0
            ddim_r30_registered = True
        else:
            ddim_r30_registered = False
        frozen_inputs = {
            "base_checkpoint": "92ac831b7dc1b1cdad4ddc9d7003813e174cb329507bdeb019e23f74d3aac19c",
            "constrain_checkpoint": "ff388efb87e4fde6acaaf21778aaee25b93883af990b99ad510fa82f97ccbf71",
            "data_h5": "796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961",
            "data_manifest": "aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9",
            "training_identity": "648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78",
            "scene_manifest": "9e1d8ee4786d2336f01e07d664ca37c3dcc7ca455e7876176def65c18b63c119",
            "primary_summary": ("ef24daa752f99ce31d5063325397c1276f3e6bc3d085dcaeafa824904378346e"
                                if args.ensemble_size == 4 else
                                "3c537c1ba646e5d483e2fd4075524ba36bc0ea08eae4356cc4e0231276c2b0cd"),
        }
        expected_paths = {
            "scene_dir": "outputs/exp/20261024_014459_placesphere_learning_pilot/dev_scenes",
            "base_checkpoint": "outputs/ckpt/20261026_195000_placesphere_hold8_base_dp_r23/final.pt",
            "constrain_checkpoint": "outputs/ckpt/20261027_140000_placesphere_hold8_500_r26_constrain_150k/final.pt",
            "primary_baseline": ("outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval"
                                 if args.ensemble_size == 4 else
                                 "outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval"),
        }
        registered_sampler = preregistration.get("sampler", {})
        if not ddim_r30_registered and (preregistration.get("format") != "placesphere_frozen_ddim100_eta0_v1"
                or registered_sampler.get("name") != "DDIMScheduler"
                or registered_sampler.get("inference_steps") != 100
                or registered_sampler.get("eta") != 0.0
                or registered_sampler.get("source_config") != "each policy's training-time 100-step DDPMScheduler config"
                or registered_sampler.get("prediction_type") != "epsilon"
                or registered_sampler.get("beta_schedule") != "squaredcos_cap_v2"
                or registered_sampler.get("clip_sample") is not True
                or registered_sampler.get("training_scheduler_unchanged") is not True
                or {name: preregistration.get("checkpoints", {}).get(name, {}).get("sha256")
                    for name in ("base_dp", "constrain_dp")} != {
                    "base_dp": _sha(args.base_checkpoint), "constrain_dp": _sha(args.constrain_checkpoint)}
                or preregistration.get("scenes", {}).get("manifest_sha256") != _sha(args.scene_dir / "manifest.json")
                or preregistration.get("scenes", {}).get("scenes_h5_sha256") != _sha(args.scene_dir / "scenes.h5")
                or preregistration.get("source_sha256") != actual_sources
                or preregistration.get("protocol", {}).get("primary_baseline_summary_sha256") != _sha(args.primary_baseline / "paired_summary.json")):
            raise ValueError("DDIM preregistration identity mismatch")
        if not ddim_r30_registered and (str(args.scene_dir) != preregistration.get("scenes", {}).get("path")
                or str(args.base_checkpoint) != expected_paths["base_checkpoint"]
                or str(args.constrain_checkpoint) != expected_paths["constrain_checkpoint"]
                or str(args.primary_baseline) != preregistration.get("protocol", {}).get("primary_baseline")
                or str(args.output_dir) != preregistration.get("protocol", {}).get("output_dir")
                or expected_paths["scene_dir"] != preregistration.get("scenes", {}).get("path")
                or args.replan_stride != 2
                or preregistration.get("protocol", {}).get("stride") != 2
                or preregistration.get("protocol", {}).get("prediction_horizon") != 16
                or preregistration.get("protocol", {}).get("executed_action_slice") != [1, 9]
                or preregistration.get("protocol", {}).get("max_control_steps") != 200
                or preregistration.get("protocol", {}).get("episode_wall_seconds") != 300
                or preregistration.get("protocol", {}).get("fixed_denominator_per_policy") != 30
                or preregistration.get("budgets", {}).get("outer_wall_seconds") != 18000
                or preregistration.get("budgets", {}).get("per_policy_episodes") != 30
                or preregistration.get("budgets", {}).get("total_policy_episodes") != 60
                or preregistration.get("data", {}).get("direct_episodes") != 1000
                or preregistration.get("data", {}).get("obstacle_episodes") != 500
                or _sha(args.base_checkpoint) != frozen_inputs["base_checkpoint"]
                or _sha(args.constrain_checkpoint) != frozen_inputs["constrain_checkpoint"]
                or preregistration.get("data", {}).get("h5_sha256") != frozen_inputs["data_h5"]
                or preregistration.get("data", {}).get("manifest_sha256") != frozen_inputs["data_manifest"]
                or preregistration.get("data", {}).get("training_identity_sha256") != frozen_inputs["training_identity"]
                or preregistration.get("scenes", {}).get("manifest_sha256") != frozen_inputs["scene_manifest"]
                or preregistration.get("protocol", {}).get("primary_baseline_summary_sha256") != frozen_inputs["primary_summary"]):
            raise ValueError("DDIM trial does not match the frozen r26 artifacts")
        if not ddim_r30_registered:
            primary_pairing = _validate_primary_baseline(
                args.primary_baseline, frozen_inputs["base_checkpoint"],
                frozen_inputs["constrain_checkpoint"])
        registered_ensemble = preregistration.get("protocol", {}).get("ensemble_size", 1)
        if not ddim_r30_registered and registered_ensemble != args.ensemble_size:
            raise ValueError("DDIM ensemble size does not match preregistration")
        if not ddim_r30_registered:
            constrain_ck = torch.load(args.constrain_checkpoint, map_location="cpu", weights_only=False)
            data = constrain_ck["config"]["data_identity"]
            frozen_data_dir = Path(preregistration["data"]["dataset"])
            actual_data_identity = _sha(frozen_data_dir / "training_identity.json")
            if (data.get("h5_sha256") != preregistration.get("data", {}).get("h5_sha256")
                    or data.get("manifest_sha256") != preregistration.get("data", {}).get("manifest_sha256")
                    or data.get("training_identity_sha256") != preregistration.get("data", {}).get("training_identity_sha256")
                    or _sha(frozen_data_dir / "trajectory.h5") != preregistration["data"]["h5_sha256"]
                    or _sha(frozen_data_dir / "trajectory.json") != preregistration["data"]["manifest_sha256"]
                    or actual_data_identity != preregistration["data"]["training_identity_sha256"]
                    or str(frozen_data_dir) != "demos/PlaceSphere-transport-detour/20261027_140000_hold8_500_r26/training_package_r3"):
                raise ValueError("frozen training data identity mismatch")
    inference_sampler = args.sampler
    eta = 0.0 if inference_sampler == "ddim" else None
    base_adapter = PlaceSphereBaseDPAdapter(args.base_checkpoint, args.device, weights="ema",
                                            sampler=inference_sampler, eta=eta,
                                            ensemble_size=args.ensemble_size)
    constrain_adapter = PlaceSphereConstrainAdapter(
        args.constrain_checkpoint, args.device, base_checkpoint=str(args.base_checkpoint),
        sampler=inference_sampler, eta=eta, ensemble_size=args.ensemble_size)
    policies = {"base_dp": _BasePolicy(base_adapter),
                "constrain_dp": _ConstrainPolicy(constrain_adapter, args.constrain_checkpoint)}
    args.output_dir.mkdir(parents=True)
    frozen = {"scene_dir": str(args.scene_dir),
              "scene_manifest_sha256": sha256_file(args.scene_dir / "manifest.json"),
              "scene_count": 30,
              "checkpoints": {"base_dp": {"path": str(args.base_checkpoint), "sha256": sha256_file(args.base_checkpoint)},
                              "constrain_dp": {"path": str(args.constrain_checkpoint), "sha256": sha256_file(args.constrain_checkpoint)}},
              "primary_baseline": str(args.primary_baseline),
              "secondary_baseline": str(args.secondary_baseline),
               "device": args.device, "max_steps": 200, "episode_wall_seconds": 300,
               "execution_stride": args.replan_stride,
                 "prediction_horizon": 16, "executed_action_slice": [1, 9],
               "ensemble_size": args.ensemble_size,
                "action_encoding": {"base_dp": base_adapter.metadata.get("action_encoding"),
                                    "constrain_dp": constrain_adapter.metadata.get("action_encoding")},
                "sampler": {"training": "DDPMScheduler squaredcos_cap_v2 epsilon clip_sample=True; 100 train timesteps",
                            "inference": "DDIMScheduler eta=0; 100 timesteps" if inference_sampler == "ddim" else
                                         "DDPMScheduler squaredcos_cap_v2; 100 timesteps",
                            "reverse_variance_draws": "removed for DDIM; subsequent initial-noise RNG positions therefore differ" if inference_sampler == "ddim" else "DDPM reverse variance draws retained",
                            "weights_shared_across_policies": True},
                 "noise_sampling": "same torch.manual_seed(scene seed) reset per policy; independent initial plans per ensemble member" if args.ensemble_size == 4 else "same torch.manual_seed(scene seed) reset per policy; DDIM eta=0 removes reverse variance draws",
               "inference_preregistration_sha256": preregistration_sha,
               "primary_baseline_pairing_audit": primary_pairing if args.sampler == "ddim" else None,
              "selected_weights": "final EMA fixed before results", "runtime_errors_remain_in_denominator": True}
    (args.output_dir / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    rows_by_policy = {}
    for name, adapter in policies.items():
        out = args.output_dir / name
        out.mkdir()
        rows = []
        for index in range(30):
            call_start = adapter.inference_calls
            adapter.max_inference_wall_seconds = 0.0
            previous = signal.signal(signal.SIGALRM, _episode_deadline)
            signal.setitimer(signal.ITIMER_REAL, 300)
            started = time.monotonic()
            trace = []
            try:
                row = run_pilot_episode(adapter, catalog, index, args.device, max_steps=200,
                                        replan_stride=args.replan_stride,
                                        trace_callback=trace.append)
            except Exception as exc:
                row = {"scene_index": index, "status": "runtime_error",
                       "error_kind": "wall_timeout" if isinstance(exc, EpisodeDeadline) else "unexpected",
                        "error": f"{type(exc).__name__}: {exc}",
                        "closed_loop_trace": trace,
                        "checkpoint_sha256": adapter.metadata["checkpoint_sha256"]}
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
            row["wall_seconds"] = time.monotonic() - started
            row["inference_calls"] = adapter.inference_calls - call_start
            row["max_single_inference_wall_seconds"] = adapter.max_inference_wall_seconds
            row["closed_loop_trace"] = trace
            (out / f"episode_{index:03d}.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
        rows_by_policy[name] = rows
        (out / "summary.json").write_text(json.dumps(summarize_pilot(rows, 30), indent=2) + "\n")

    metric_map = {"task": "native_success", "safe": "safe_success",
                  "contact": "obstacle_contact", "xy": "xy_forbidden_crossing"}
    primary_label = ("r27_primary_constrain" if args.inference_preregistration is not None
                     and preregistration.get("format") == "placesphere_r30_action_codec_ddim100_preregistration_v1"
                     else "primary_constrain_vs_stride2_r24")
    base_label = ("r27_primary_base" if primary_label == "r27_primary_constrain"
                  else "base_vs_stride2_r24_base")
    comparisons = [(primary_label, args.primary_baseline / "constrain_dp", "constrain_dp"),
                   (base_label, args.primary_baseline / "base_dp", "base_dp")]
    if args.secondary_baseline is not None:
        comparisons.append(("secondary_constrain_vs_100_obstacle", args.secondary_baseline / "constrain_dp", "constrain_dp"))
    comparison = {}
    for name, baseline, new_name in comparisons:
        old = _read_rows(baseline)
        if any(a.get("seed") != b.get("seed") or a.get("scene_id") != b.get("scene_id")
               for a, b in zip(old, rows_by_policy[new_name])):
            raise ValueError(f"scene pairing mismatch for {name}")
        comparison[name] = {metric: _transitions(old, rows_by_policy[new_name], field)
                            for metric, field in metric_map.items()}
    report = {"protocol_version": "placesphere_action_codec_r30_primary_r27_v1" if primary_label == "r27_primary_constrain" else None,
              "primary_baseline_role": "immutable R27 DDIM100 single-plan reference; different weights are expected" if primary_label == "r27_primary_constrain" else None,
              "complete": all(summarize_pilot(rows, 30)["complete"] for rows in rows_by_policy.values()),
              "denominator": 30,
              "summaries": {name: summarize_pilot(rows, 30) for name, rows in rows_by_policy.items()},
              "comparisons": comparison,
              "gate": {"task_success_min": .90, "safe_success_min": .85,
                       "contact_max": .10, "xy_crossing_max": .10,
                       "passed": False}}
    # The result gate is computed from the new ConstrainNet, without checkpoint selection.
    s = report["summaries"]["constrain_dp"]
    report["gate"]["passed"] = bool(report["complete"] and
        s["task_success_count"] >= 27 and s["safe_success_count"] >= 26 and
        s["obstacle_contact_count"] <= 3 and s["xy_crossing_count"] <= 3)
    (args.output_dir / "paired_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["complete"]:
        raise RuntimeError("paired evaluation has runtime errors; rows retained in denominator")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
