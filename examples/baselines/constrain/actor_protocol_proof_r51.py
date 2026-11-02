"""Bounded, no-environment-step R51 comparison against the retained R50 actor."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import pickcube_body_hint_round41 as r41
from examples.baselines.constrain import pickcube_xy_gauge_round40 as r40
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_clearance_train_r44 import (
    Horizon8ClearanceActor, TRAIN_FORMAT, MODEL_KIND, TOTAL, EMA_POWER,
    STATS_PATH, PRECHECK, FIT, BASE,
)
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet


ROOT = Path("outputs/exp/20261102_130536_xy_runtime_diagnosis_r50")
CATALOG = Path("outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/pickcube")
CHECKPOINT = Path("outputs/ckpt/20261029_160000_pickcube_horizon_clearance_r44_control/final.pt")
STATS = Path("outputs/exp/20261029_031500_pickcube_bodyhint_r41/position_encoding.json")
OUT = Path("outputs/exp/20261102_200000_xy_protocol_buffer_r51/actor_protocol_proof.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_adapter_action(actor, raw):
    """R50's selected evaluator adapter protocol using explicitly audited components."""
    cond = engine.normalize_observation(raw[..., :42], actor.stats).flatten(1)
    from examples.baselines.constrain.pickcube_body_hint import body_hint_from_raw

    hint = body_hint_from_raw(raw)[:, None].expand(-1, 16, -1)
    sample = torch.randn((len(raw), 16, 7), device=actor.device)
    with torch.no_grad():
        for timestep in actor.scheduler.timesteps:
            eps = actor.model(sample, timestep, global_cond=cond, hint=hint)
            sample = actor.scheduler.step(eps, timestep, sample).prev_sample
    return sample[:, 1:9]


def main():
    registration = json.loads((ROOT / "pickcube_registration.json").read_text())
    for name, entry in registration["sources"].items():
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"R50 source changed: {name}")
    for name, entry in registration["source_snapshots"].items():
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"R50 immutable snapshot changed: {name}")
    original_source = Path(registration["source_snapshots"]["actor"]["path"])
    current_source = Path(registration["sources"]["actor"]["path"])
    if digest(original_source) != digest(current_source):
        raise ValueError("R50 original actor bytes differ from current actor source")

    # Constructor's historical loader fails its strict metadata gate in this
    # checkout. Validate each registered byte/input and typed EMA component
    # explicitly, then bind the unchanged original method to this validated object.
    for name, entry in registration["inputs"].items():
        if digest(entry["path"]) != entry["sha256"]:
            raise ValueError(f"R50 input changed: {name}")
    r41._install_profile()
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    if (ck.get("format") != TRAIN_FORMAT or cfg.get("format") != TRAIN_FORMAT
            or cfg.get("model_kind") != MODEL_KIND or cfg.get("horizon") != 8
            or ck.get("step") != TOTAL or ck.get("ema") is None
            or cfg.get("selected_weights") != "ema"
            or cfg.get("normalization", {}).get("observation_stats_sha256") != engine.load_stats(STATS).get("sha256")
            or cfg.get("normalization", {}).get("observation_stats") != engine.load_stats(STATS)
            or cfg.get("precheck_sha256") != sha256_file(PRECHECK)
            or cfg.get("response_fit_sha256") != sha256_file(FIT)
            or cfg.get("base_weights", {}).get("sha256") != sha256_file(BASE)):
        raise ValueError("R50 checkpoint failed explicit typed checkpoint/model audit")
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG
    from diffusers.training_utils import EMAModel
    model = engine.build_constrain_denoiser(CONTROL_CONFIG, device="cpu")
    model.load_state_dict(ck["model"], strict=True)
    ema = EMAModel(parameters=model.trainable_parameters(), power=EMA_POWER)
    ema.load_state_dict(ck["ema"]); ema.copy_to(model.trainable_parameters())
    model.to("cuda").eval()
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

    def actor_instance():
        actor = Horizon8ClearanceActor.__new__(Horizon8ClearanceActor)
        actor.device, actor.model, actor.ck, actor.stats = torch.device("cuda"), model, ck, engine.load_stats(STATS)
        actor.scheduler = DDPMScheduler(num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2", clip_sample=True, prediction_type="epsilon")
        actor.scheduler.set_timesteps(100)
        return actor

    original, current = actor_instance(), actor_instance()
    scenes = TransportDetourSceneSet(CATALOG)
    results = []
    for index in (0, 1, 3):
        observation = np.asarray(scenes.load(index)["observation"], dtype=np.float32)
        raw = torch.as_tensor(np.repeat(observation[None, None, :], 2, axis=1),
                              device="cuda")
        # Same initial random tensor and scheduler-internal random draws.
        torch.manual_seed(918273 + index)
        torch.cuda.manual_seed_all(918273 + index)
        expected = Horizon8ClearanceActor.get_action(original, raw)
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all()
        torch.manual_seed(918273 + index)
        torch.cuda.manual_seed_all(918273 + index)
        actual = selected_adapter_action(current, raw)
        rng_exact = torch.equal(rng_cpu, torch.get_rng_state()) and all(
            torch.equal(a, b) for a, b in zip(rng_cuda, torch.cuda.get_rng_state_all())
        )
        results.append({"scene_index": index, "max_abs_action_error": float((expected-actual).abs().max()),
                        "exact_action": bool(torch.equal(expected, actual)), "rng_state_exact": rng_exact})

    # Profile installers are process-global; verify the real evaluator order and
    # reverse/repeated order leave the runtime profile values identical.
    def profile_snapshot():
        return {"engine_paths": [str(engine.ROUND37_EXP), str(engine.ROUND37_BASE_CKPT),
                                 str(engine.ROUND37_CONTROL_CKPT), str(engine.R36_DEV)],
                "source_files": list(engine.SOURCE_FILES),
                "identity": engine.source_identity()}

    before = profile_snapshot()
    for _ in range(2):
        r41._install_profile(); r40._install_evaluation_loaders()
    after = profile_snapshot()
    r40._install_evaluation_loaders(); r41._install_profile()
    reversed_after = profile_snapshot()
    report = {"format": "r51_original_actor_protocol_proof_v1", "environment_steps": 0,
        "original_actor_source_sha256": digest(original_source),
        "current_actor_source_sha256": digest(current_source),
        "checkpoint_sha256": digest(CHECKPOINT), "normalizer_sha256": digest(STATS),
        "original_method": "Horizon8ClearanceActor.get_action directly called on __new__ instance with explicitly audited model/checkpoint/normalizer/scheduler properties",
        "historical_constructor_result": "rejected by historical load_actor metadata gate; constructor equivalence not claimed",
        "constructor_equivalence_claim": False,
        "adapter_protocol": "R50 selected Adapter.get_action normalization/hint/DDPM body",
        "comparisons": results, "maximum_absolute_error": max(x["max_abs_action_error"] for x in results),
        "all_actions_exact": all(x["exact_action"] for x in results),
        "all_rng_states_exact": all(x["rng_state_exact"] for x in results),
        "profile_installers": {"repeat_same_order_unchanged": before == after,
                                "reverse_order_unchanged": before == reversed_after}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
