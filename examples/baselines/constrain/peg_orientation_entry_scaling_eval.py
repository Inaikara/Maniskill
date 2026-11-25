"""Six-arm descriptive checkpoint comparison on the frozen outside-entry bank."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import peg_orientation_entry_perturb_eval as paired
from examples.baselines.constrain import peg_orientation_near_hole_eval as near

ROOT = Path(__file__).resolve().parents[3]
BANK = paired.BANK
SEEDS = paired.SEEDS
CONDITIONS = paired.CONDITIONS
BASE = "base"
ARMS = {
    BASE: "outputs/ckpt/20261116_123901_peg_orientation_base_official_config/update_100000.pt",
    "cn100_15k": "outputs/ckpt/20261117_003640_peg_orientation_replay_exploratory/metadata_complete/update_15000.pt",
    "cn100_30k": "outputs/ckpt/20261117_003640_peg_orientation_replay_exploratory/metadata_complete/update_30000.pt",
    "cn200_30k": "outputs/ckpt/20261117_211300_peg_orientation_random200_correction/update_30000.pt",
    "cn500_25k": "outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_25000.pt",
    "cn500_30k": "outputs/ckpt/20261118_035943_peg_orientation_random500_study/update_30000.pt",
}
PRIMARY = (BASE, "cn100_30k", "cn200_30k", "cn500_30k")
SECONDARY = (BASE, "cn100_15k", "cn200_30k", "cn500_25k")
MAX_NATIVE_CONTROLS, STAGE_SECONDS, WHOLE_SECONDS = 669600, 32400, 43200
ENV_TIME_LIMIT = 1000


def validate_arm_result(name, result):
    """Reject malformed adapter output at this evaluator's narrow public seam."""
    actions = result.detach().cpu().numpy() if isinstance(result, torch.Tensor) else np.asarray(result)
    if actions.shape != (1, 8, 7) or not np.isfinite(actions).all():
        raise ValueError(f"{name} adapter must return finite (1, 8, 7) actions, got {actions.shape}")
    return actions


def create_adapters(device="cuda"):
    from examples.baselines.constrain.peg_orientation_base_dp import PegBaseAdapter
    from examples.baselines.constrain.peg_orientation_constrain import PegConstrainAdapter
    adapters = {BASE: PegBaseAdapter(ROOT / ARMS[BASE], device=device)}
    for name in ARMS:
        if name == BASE:
            continue
        adapters[name] = PegConstrainAdapter(ROOT / ARMS[name], device=device, task=None, record_all_hints=True)
    return adapters


def _prepare_output(output, additional_sources=()):
    output = Path(output)
    allowed = {"attempt_001.sh", "attempt_001.stdout.log", "attempt_001.stderr.log"}
    if not output.is_dir() or not (output / "attempt_001.sh").is_file():
        raise FileNotFoundError("output directory and attempt_001 wrapper must be precreated")
    unexpected = {p.name for p in output.iterdir()} - allowed
    if unexpected:
        raise FileExistsError(f"output directory already contains artifacts: {sorted(unexpected)}")
    wrapper = output / "attempt_001.sh"
    wrapper_hash = hashlib.sha256(wrapper.read_bytes()).hexdigest()
    if not wrapper.read_bytes().strip():
        raise ValueError("precreated wrapper is empty")
    archived = output / "source_archive" / Path(__file__).name
    archived.parent.mkdir()
    archived.write_bytes(Path(__file__).read_bytes())
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    archived_sources = {}
    for source in additional_sources:
        source = Path(source).resolve()
        target = archived.parent / source.name
        target.write_bytes(source.read_bytes())
        archived_sources[source.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return {"wrapper_path": str(wrapper), "wrapper_sha256": wrapper_hash,
            "runner_path": str(Path(__file__).resolve()), "runner_sha256": source_hash,
            "archived_runner_sha256": hashlib.sha256(archived.read_bytes()).hexdigest(),
            "additional_source_sha256": archived_sources}


def _set_context(adapters, envs, row, catalog):
    manifest = catalog.manifest["scenes"][int(row["catalog_index"])]
    for name, adapter in adapters.items():
        if name != BASE:
            adapter.task = envs[name].unwrapped
        if hasattr(adapter, "set_episode_context"):
            adapter.set_episode_context(int(row["catalog_index"]), manifest)


def _aggregate(completed, primary=PRIMARY, secondary=SECONDARY):
    def paired(rows, edges):
        output = {}
        for left, right in edges:
            output[f"{left}->{right}"] = {
                "improved": sum(not x["arms"][left]["native_success"] and x["arms"][right]["native_success"] for x in rows),
                "regressed": sum(x["arms"][left]["native_success"] and not x["arms"][right]["native_success"] for x in rows),
                "both": sum(x["arms"][left]["native_success"] and x["arms"][right]["native_success"] for x in rows),
                "neither": sum(not x["arms"][left]["native_success"] and not x["arms"][right]["native_success"] for x in rows)}
        return output

    def monotonicity(rows, models):
        patterns = {}
        for row in rows:
            pattern = "".join("1" if row["arms"][name]["native_success"] else "0" for name in models)
            patterns[pattern] = patterns.get(pattern, 0) + 1
        values = list(patterns)
        return {"models": list(models), "draws": len(rows), "patterns": patterns,
            "nondecreasing_draws": sum(count for pattern, count in patterns.items() if "10" not in pattern),
            "nonincreasing_draws": sum(count for pattern, count in patterns.items() if "01" not in pattern)}

    models = tuple(completed[0]["arms"]) if completed else tuple(ARMS)
    curves = {"primary": tuple(primary), "secondary": tuple(secondary)}
    edges = []
    for curve in curves.values():
        curve_edges = list(zip(curve, curve[1:]))
        curve_edges.extend((curve[0], name) for name in curve[1:])
        for edge in curve_edges:
            if edge not in edges:
                edges.append(edge)
    edges = tuple(edges)
    profiles = {}
    for profile in ("random", "fixed"):
        profiles[profile] = {}
        for condition in (*CONDITIONS, "_all_conditions"):
            rows = [r for r in completed if r["profile"] == profile and r["condition"] == condition]
            if condition == "_all_conditions":
                rows = [r for r in completed if r["profile"] == profile]
            by_arm = {name: sum(x["arms"][name]["native_success"] for x in rows) for name in models}
            per_seed = {}
            for replicate in range(len(SEEDS)):
                seed_rows = [x for x in rows if x["replicate"] == replicate]
                per_seed[str(SEEDS[replicate])] = {
                    "draws": len(seed_rows), "successes": {name: sum(x["arms"][name]["native_success"] for x in seed_rows) for name in models},
                    "denominator": len(seed_rows), "paired_transitions": paired(seed_rows, edges),
                    "monotonicity": {key: monotonicity(seed_rows, curve[1:]) for key, curve in curves.items()}}
            profiles[profile][condition] = {"draws": len(rows), "unique_origins": len({x["catalog_index"] for x in rows}),
                "successes": by_arm, "primary": list(curves["primary"]), "secondary": list(curves["secondary"]),
                "per_seed": per_seed, "paired_transitions": paired(rows, edges),
                "monotonicity": {key: monotonicity(rows, curve[1:]) for key, curve in curves.items()}}
    return profiles


def run_comparison(output, *, adapters=None, env_factory=None, catalog_factory=None,
                   source_action_loader=None, bank=None, arms=None, primary=None, secondary=None,
                   max_native_controls=None, stage_seconds=None, whole_seconds=None,
                   source_archive_files=()):
    output = Path(output)
    roster = dict(ARMS if arms is None else arms)
    primary = tuple(PRIMARY if primary is None else primary)
    secondary = tuple(SECONDARY if secondary is None else secondary)
    max_native_controls = MAX_NATIVE_CONTROLS if max_native_controls is None else max_native_controls
    stage_seconds = STAGE_SECONDS if stage_seconds is None else stage_seconds
    whole_seconds = WHOLE_SECONDS if whole_seconds is None else whole_seconds
    setup = _prepare_output(output, source_archive_files)
    rows = bank if bank is not None else paired.bank_rows()
    if not rows or (bank is None and len(rows) != 60):
        raise ValueError("expected full 60-case bank")
    adapters = create_adapters() if adapters is None else adapters
    if set(adapters) != set(roster) or len(adapters) != len(roster):
        raise ValueError(f"one adapter required for each roster arm: {tuple(roster)}")
    if env_factory is None:
        import gymnasium as gym
        from examples.baselines.constrain.peg_orientation import ENV_ID, RANDOM_ENV_ID
        env_factory = lambda profile: gym.make(RANDOM_ENV_ID if profile == "random" else ENV_ID,
            obs_mode="state", control_mode="pd_ee_delta_pose", sim_backend="physx_cpu",
            reconfiguration_freq=1, max_episode_steps=ENV_TIME_LIMIT)
    if catalog_factory is None:
        from examples.baselines.constrain.peg_orientation_data import PegSceneCatalog
        catalog_factory = lambda profile: PegSceneCatalog(near.CATALOG_ROOT / ("validation_random_scenes" if profile == "random" else "validation_fixed_scenes"))
    if source_action_loader is None:
        source_action_loader = lambda profile, episode: near._source_actions(near.DATA_DIRS[profile] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5", episode)
    catalogs = {p: catalog_factory(p) for p in ("random", "fixed")}
    source_files = {}
    for profile in ("random", "fixed"):
        source_h5 = near.DATA_DIRS[profile] / "trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
        source_files[profile] = {"path": str(source_h5),
            "sha256": near.sha256(source_h5) if source_h5.is_file() else None}
    started, budget = time.monotonic(), [0, time.monotonic()]
    journal, completed = output / "pairs.jsonl", []
    try:
        for row in rows:
            profile, idx, result = row["profile"], int(row["catalog_index"]), row["result"]
            catalog = catalogs[profile]
            scene = catalog.load(idx)
            source_episode = paired.source_episode_for_scene(profile, idx, catalog)
            source_actions = source_action_loader(profile, source_episode)
            prefix = int(result["source_prefix_controls"])
            source_prefix = np.asarray(source_actions[:prefix], np.float32)
            registered_prefix = np.asarray(result["executed_prefix_actions"], np.float32)
            if not np.array_equal(source_prefix, registered_prefix):
                raise ValueError("bank source prefix does not match registered source trajectory")
            for replicate, seed_base in enumerate(SEEDS):
                seed = int(seed_base) + idx
                pair = {"profile": profile, "condition": row["condition"], "catalog_index": idx,
                        "source_scene_id": row.get("source_scene_id"), "replicate": replicate,
                        "source_episode_id": source_episode,
                        "source_episode_seed": int(catalog.manifest["scenes"][idx]["seed"]),
                        "source_hdf5": source_files[profile],
                        "source_prefix_action_sha256": hashlib.sha256(source_prefix.tobytes()).hexdigest(),
                        "bank_full_action_sha256": result["full_action_sha256"],
                        "bank_prefix_controls": prefix,
                        "seed": seed, "arms": {}, "status": "preparing"}
                envs = {}
                try:
                    # Construct all roster histories/controllers before any query.
                    envs = {name: env_factory(profile) for name in roster}
                    prepared = {name: paired._prepare(env, scene, row, budget, started,
                        max_native_controls=max_native_controls, stage_seconds=stage_seconds)
                        for name, env in envs.items()}
                    reference = prepared[BASE]
                    for name in roster:
                        near.assert_controller_match(envs[BASE], envs[name])
                        if not np.array_equal(reference["history"], prepared[name]["history"]):
                            raise ValueError(f"{name} handoff observations differ from Base")
                        errors = near._max_error(reference["physical_state"], prepared[name]["physical_state"])
                        if not errors or not np.isfinite(list(errors.values())).all() or max(errors.values()) > 1e-5:
                            raise ValueError(f"{name} physical handoff differs from Base: {errors}")
                    pair["handoff"] = {"controller": reference["controller"], "history": reference["history"].tolist(),
                        "physical_state": near.jsonable(reference["physical_state"]), "max_state_error_vs_base": {
                        name: near._max_error(reference["physical_state"], prepared[name]["physical_state"]) for name in roster}}
                    near.append_pair_journal(journal, pair)
                    _set_context(adapters, envs, row, catalog)
                    for name in roster:
                        arm = prepared[name]
                        if budget[0] >= max_native_controls or time.monotonic() - started >= whole_seconds:
                            raise TimeoutError("scaling comparison physical-control or wall budget exhausted")
                        paired._seed(seed)
                        class CheckedAdapter:
                            def get_action(self, observations):
                                return validate_arm_result(name, adapters[name].get_action(observations))
                        outcome = near._run_suffix(envs[name], CheckedAdapter(), arm, seed, budget,
                            max_native_controls=max_native_controls, stage_seconds=stage_seconds)
                        pair["arms"][name] = near.jsonable(outcome)
                        pair["status"] = "partial"
                        near.append_pair_journal(journal, pair)
                    pair["status"] = "complete"
                    completed.append(pair)
                    near.append_pair_journal(journal, pair)
                except Exception as exc:
                    pair.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc)},
                                physical_controls_at_failure=budget[0])
                    near.append_pair_journal(journal, pair)
                    (output / "failure.json").write_text(json.dumps(near.jsonable({"failure": pair["failure"], "physical_controls": budget[0]}), allow_nan=False, indent=2))
                    raise
                finally:
                    for env in envs.values(): env.close()
                    for name in roster:
                        if name != BASE: adapters[name].task = None
        summary = {"pairs_completed": len(completed), "arm_rows": len(completed) * len(roster),
            "model_roster": list(roster),
            "native_controls_including_replay": budget[0], "setup": setup,
            "per_condition": _aggregate(completed, primary, secondary),
            "primary_curve": list(primary), "secondary_curve": list(secondary),
            "shared_base_evaluated_once_per_draw": True,
            "independent_scene_origins": len({int(r["catalog_index"]) for r in rows}),
            "condition_draws_are_repeated_origins": True,
            "interpretation": "Descriptive paired checkpoint comparison; repeated draws are not independent scenes.",
            "native_rollouts_started": bool(completed)}
        (output / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        return summary
    except Exception:
        raise
