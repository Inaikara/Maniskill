"""Read-only correction analysis for the R27 DDIM trial's matched R26 primary."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
R26_RUN = Path("outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2")
R27_RUN = Path("outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval")
R26_PRIMARY = "outputs/exp/20261027_140000_placesphere_hold8_500_r26/paired_eval_stride2"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha(directory):
    root = Path(directory)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha(path).encode())
        digest.update(b"\n")
    return len(files), sum(path.stat().st_size for path in files), digest.hexdigest()


def verify_registered_inputs(registration, *, root=ROOT):
    for key, item in registration["inputs"].items():
        if _sha(Path(root) / item["path"]) != item["sha256"]:
            raise ValueError(f"registered analysis input hash mismatch: {key}")
    return True


def validate_primary_configs(candidate, primary, *, primary_run_path=R26_PRIMARY):
    if candidate.get("primary_baseline") != R26_PRIMARY or primary_run_path != R26_PRIMARY:
        raise ValueError("the R26 primary must be the same-checkpoint R26 primary path")
    for policy in ("base_dp", "constrain_dp"):
        if candidate.get("checkpoints", {}).get(policy, {}).get("sha256") != primary.get("checkpoints", {}).get(policy, {}).get("sha256"):
            raise ValueError(f"{policy} checkpoint mismatch; this cannot be the same-checkpoint primary")
    fields = ("scene_dir", "scene_manifest_sha256", "scene_count", "max_steps",
              "episode_wall_seconds", "execution_stride", "prediction_horizon",
              "executed_action_slice", "runtime_errors_remain_in_denominator")
    mismatches = [field for field in fields if candidate.get(field) != primary.get(field)]
    if mismatches:
        raise ValueError(f"primary rollout contract mismatch: {mismatches}")
    return True


def _rows(run, policy):
    directory = ROOT / run / policy
    return [json.loads((directory / f"episode_{index:03d}.json").read_text())
            for index in range(30)]


def _transition(old_rows, new_rows, field, safety=False):
    old = [bool(row["result"][field]) for row in old_rows]
    new = [bool(row["result"][field]) for row in new_rows]
    if safety:
        return {"cleared": sum(a and not b for a, b in zip(old, new)),
                "introduced": sum(not a and b for a, b in zip(old, new)),
                "both": sum(a and b for a, b in zip(old, new)),
                "neither": sum(not a and not b for a, b in zip(old, new))}
    return {"improved": sum(not a and b for a, b in zip(old, new)),
            "regressed": sum(a and not b for a, b in zip(old, new)),
            "both": sum(a and b for a, b in zip(old, new)),
            "neither": sum(not a and not b for a, b in zip(old, new))}


def _verify_pairs(r26_run, r27_run, policy, scene_manifest_sha256):
    old_rows, new_rows = _rows(r26_run, policy), _rows(r27_run, policy)
    for index, (old, new) in enumerate(zip(old_rows, new_rows)):
        for row in (old, new):
            if (row.get("status") != "evaluated" or not row.get("restored_bit_exact")
                    or row.get("scene_index") != index):
                raise ValueError(f"{policy} scene {index}: incomplete or non-bit-exact episode")
        if (old.get("scene_id"), old.get("seed")) != (new.get("scene_id"), new.get("seed")):
            raise ValueError(f"{policy} scene {index}: scene/seed mismatch")
        old_trace, new_trace = old.get("closed_loop_trace", []), new.get("closed_loop_trace", [])
        if not old_trace or not new_trace or old_trace[0].get("policy_observation") != new_trace[0].get("policy_observation"):
            raise ValueError(f"{policy} scene {index}: initial controller observation mismatch")
        if old.get("checkpoint_sha256") != new.get("checkpoint_sha256"):
            raise ValueError(f"{policy} scene {index}: episode checkpoint SHA mismatch")
    return old_rows, new_rows


def analyze(registration_path):
    registration_path = Path(registration_path)
    registration = json.loads(registration_path.read_text())
    if registration.get("format") != "placesphere_r27_primary_correction_analysis_v1":
        raise ValueError("unexpected correction-analysis registration format")
    analysis_source = ROOT / "examples/baselines/constrain/placesphere_r27_primary_analysis.py"
    test_source = ROOT / "examples/baselines/constrain/tests/test_placesphere_r27_primary_analysis.py"
    if registration.get("analysis_source_sha256") != _sha(analysis_source):
        raise ValueError("correction analysis source hash mismatch")
    if registration.get("test_source_sha256") != _sha(test_source):
        raise ValueError("correction analysis test hash mismatch")

    paths = registration["inputs"]
    verify_registered_inputs(registration)
    for key, item in registration.get("evaluation_trees", {}).items():
        count, size, digest = _tree_sha(ROOT / item["path"])
        if (count, size, digest) != (item["files"], item["bytes"], item["sha256"]):
            raise ValueError(f"registered episode-tree identity mismatch: {key}")
    r26_run, r27_run = paths["r26_frozen_config"]["path"].rsplit("/", 1)[0], R27_RUN.as_posix()
    r26_config = json.loads((ROOT / paths["r26_frozen_config"]["path"]).read_text())
    r27_config = json.loads((ROOT / paths["r27_frozen_config"]["path"]).read_text())
    corrected_r27_primary = dict(r27_config)
    corrected_r27_primary["primary_baseline"] = R26_PRIMARY
    validate_primary_configs(corrected_r27_primary, r26_config, primary_run_path=r26_run)

    r26_prereg = json.loads((ROOT / paths["r26_preregistration"]["path"]).read_text())
    r26_execution = json.loads((ROOT / paths["r26_execution_record"]["path"]).read_text())
    r27_prereg = json.loads((ROOT / paths["r27_original_preregistration"]["path"]).read_text())
    if (r26_prereg.get("format") != "placesphere_r26_hold8_500_stride2_paired_evaluation_preregistration_v1"
            or r26_prereg.get("outputs", {}).get("paired_eval") != R26_PRIMARY
            or r26_execution.get("exact_commands", {}).get("evaluation", "").find("--sampler") >= 0
            or r26_execution.get("phase_outcomes", {}).get("paired_evaluation", {}).get("stride") != 2):
        raise ValueError("R26 source registration does not establish the original DDPM100 stride-2 run")
    if r27_prereg.get("protocol", {}).get("primary_baseline") != "outputs/exp/20261026_235900_placesphere_stride2_r24/paired_eval":
        raise ValueError("the original R27 registration deviation is not the recorded R24 baseline")
    expected_old_sources = {
        "examples/baselines/constrain/evaluate_placesphere_matched.py": "7e14cba4472d5d5f049159d310fa1da1745c05cb503bdc23b979b4b57d47a79b",
        "examples/baselines/constrain/placesphere_base_dp.py": "e655dcbcf14e84c668cea1e616aa35e870d470a21de67330e6fb2d7c7a0680e0",
        "examples/baselines/constrain/placesphere_constrain.py": "f2e25ffddcb71ab01ab39eadf9d90dfaedf7c9c018ca658da8fba605e74fcc6d",
    }
    for source, expected in expected_old_sources.items():
        if r26_prereg.get("source_sha256", {}).get(source) != expected:
            raise ValueError(f"R26 registered source identity mismatch: {source}")

    scene_manifest_sha = r26_config["scene_manifest_sha256"]
    if scene_manifest_sha != registration["comparison_contract"]["same_scene_manifest_sha256"]:
        raise ValueError("scene manifest differs from correction registration")
    scene_h5_sha = _sha(ROOT / r26_config["scene_dir"] / "scenes.h5")
    if scene_h5_sha != registration["comparison_contract"]["same_scene_h5_sha256"]:
        raise ValueError("scene state/controller/observation HDF5 differs from correction registration")

    results = {}
    for policy in ("base_dp", "constrain_dp"):
        old_rows, new_rows = _verify_pairs(r26_run, r27_run, policy, scene_manifest_sha)
        results[policy] = {
            "r26_ddpm100": {
                metric: sum(bool(row["result"][metric]) for row in old_rows)
                for metric in ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
            },
            "r27_ddim100_eta0": {
                metric: sum(bool(row["result"][metric]) for row in new_rows)
                for metric in ("native_success", "safe_success", "obstacle_contact", "xy_forbidden_crossing")
            },
            "transitions": {
                "task": _transition(old_rows, new_rows, "native_success"),
                "safe": _transition(old_rows, new_rows, "safe_success"),
                "contact": _transition(old_rows, new_rows, "obstacle_contact", safety=True),
                "xy": _transition(old_rows, new_rows, "xy_forbidden_crossing", safety=True),
            },
            "r26_failure_indices": [row["scene_index"] for row in old_rows if not row["result"]["native_success"]],
            "r27_failure_indices": [row["scene_index"] for row in new_rows if not row["result"]["native_success"]],
            "paired_initial_observations_exact": True,
            "bit_exact_restores": 30,
        }

    r26_count, r26_bytes, r26_tree = _tree_sha(ROOT / r26_run)
    r27_count, r27_bytes, r27_tree = _tree_sha(ROOT / r27_run)
    return {
        "format": "placesphere_r27_primary_correction_analysis_result_v1",
        "registration": registration_path.as_posix(),
        "registration_sha256": _sha(registration_path),
        "analysis_source_sha256": _sha(analysis_source),
        "test_source_sha256": _sha(test_source),
        "primary": "R26 DDPM100 stride-2; same Base EMA and Constrain EMA checkpoint SHA as R27 DDIM",
        "secondary": "R24 remains a secondary, differently weighted/data comparison; it is not causal for this sampler trial",
        "compatibility": {
            "base_and_constrain_checkpoint_hashes_match": True,
            "scene_manifest_sha256": scene_manifest_sha,
            "scene_h5_sha256": scene_h5_sha,
            "scene_count": 30,
            "stride": 2,
            "prediction_horizon": 16,
            "executed_slice": [1, 9],
            "control_step_limit": 200,
            "episode_wall_limit_seconds": 300,
            "scene_seed_and_id_pairs_match_per_policy": True,
            "all_restores_bit_exact": True,
            "initial_policy_observations_match_per_policy": True,
            "r26_sampler_evidence": {
                "name": "DDPM",
                "steps": 100,
                "evidence": "original R26 invocation omitted a sampler override; its preregistered evaluator/base/constrain source hashes and 100-step DDPM training/runtime contracts are pinned in R26 evaluation_preregistration.json"
            },
            "r27_sampler": "DDIM eta=0, 100 steps",
        },
        "policies": results,
        "artifact_trees": {
            "r26": {"path": r26_run, "files": r26_count, "bytes": r26_bytes, "sha256": r26_tree},
            "r27": {"path": r27_run, "files": r27_count, "bytes": r27_bytes, "sha256": r27_tree},
        },
        "interpretation": "Only the verified R26 same-checkpoint comparison is the primary sampler comparison. R24 figures previously stored in R27 remain secondary and confounded by different weights/data.",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.registration), indent=2, sort_keys=True))
