"""Register immutable paired inputs and analyze the R30 codec dev run."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from examples.baselines.constrain.placesphere_action_codec import (
    CONTROLLER_RANGE, encode_native,
)
from examples.baselines.constrain.placesphere_base_dp import sha256_file


ROOT = Path(__file__).resolve().parents[3]
RUN = Path("outputs/exp/20261027_224727_placesphere_action_codec_r30/paired_eval")
PRIMARY = Path("outputs/exp/20261027_204500_placesphere_ddim100_r27/paired_eval")
ANALYSIS_DIR = Path("outputs/exp/20261027_224727_placesphere_action_codec_r30")
METRICS = {"task": "native_success", "safe": "safe_success",
           "contact": "obstacle_contact", "xy": "xy_forbidden_crossing"}


def _tree_identity(directory):
    root = ROOT / directory
    entries = sorted(path for path in root.rglob("*") if path.is_file())
    digest = __import__("hashlib").sha256()
    for path in entries:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return {"path": directory.as_posix(), "files": len(entries),
            "bytes": sum(path.stat().st_size for path in entries),
            "sha256": digest.hexdigest()}


def _rows(root, policy):
    directory = ROOT / root / policy
    return [json.loads((directory / f"episode_{i:03d}.json").read_text()) for i in range(30)]


def _transitions(old, new, field, inverse=False):
    left = [bool(row["result"][field]) for row in old]
    right = [bool(row["result"][field]) for row in new]
    return ({"cleared": sum(a and not b for a, b in zip(left, right)),
            "introduced": sum(not a and b for a, b in zip(left, right)),
            "both": sum(a and b for a, b in zip(left, right)),
            "neither": sum(not a and not b for a, b in zip(left, right))}
            if inverse else
            {"improved": sum(not a and b for a, b in zip(left, right)),
             "regressed": sum(a and not b for a, b in zip(left, right)),
             "both": sum(a and b for a, b in zip(left, right)),
             "neither": sum(not a and not b for a, b in zip(left, right))})


def register(registration_path):
    registration_path = Path(registration_path)
    if registration_path.exists():
        raise FileExistsError(registration_path)
    input_dir = ANALYSIS_DIR / "analysis_inputs"
    primary_snapshot = input_dir / "r27_primary_snapshot"
    result_snapshot = input_dir / "r30_result_snapshot"
    if input_dir.exists():
        if not primary_snapshot.is_dir() or not result_snapshot.is_dir():
            raise ValueError("existing analysis inputs are incomplete")
    else:
        input_dir.mkdir(parents=True)
        shutil.copytree(ROOT / PRIMARY, ROOT / primary_snapshot)
        shutil.copytree(ROOT / RUN, ROOT / result_snapshot)
    registrations = {
        "training": "outputs/exp/20261027_224727_placesphere_action_codec_r30/training_preregistration.json",
        "evaluation": "outputs/exp/20261027_224727_placesphere_action_codec_r30/evaluation_preregistration_r3.json",
        "metadata_correction": "outputs/exp/20261027_224727_placesphere_action_codec_r30/checkpoint_metadata_correction.json",
    }
    registration_hashes = {name: sha256_file(ROOT / path) for name, path in registrations.items()}
    result = {
        "format": "placesphere_r30_action_codec_analysis_registration_v1",
        "created_after_one_fixed_evaluation": True,
        "primary_reference": PRIMARY.as_posix(),
        "candidate_evaluation": RUN.as_posix(),
        "immutable_inputs": {
            "primary_r27": _tree_identity(primary_snapshot),
            "candidate_r30": _tree_identity(result_snapshot),
        },
        "registrations": registrations,
        "registration_sha256": registration_hashes,
        "source_sha256": sha256_file(Path(__file__)),
        "interpretation": "R27 is the primary scene/seed reference; policy weights intentionally differ because both Base and Constrain were retrained for the fixed codec. This is an end-to-end pipeline comparison, not pure Constrain attribution.",
    }
    registration_path.parent.mkdir(parents=True, exist_ok=True)
    registration_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def analyze(registration_path, output_path):
    registration = json.loads(Path(registration_path).read_text())
    if (registration.get("format") != "placesphere_r30_action_codec_analysis_registration_v1"
            or registration.get("source_sha256") != sha256_file(Path(__file__))):
        raise ValueError("R30 analysis registration/source mismatch")
    if any(sha256_file(ROOT / path) != registration["registration_sha256"][name]
           for name, path in registration["registrations"].items()):
        raise ValueError("a preregistration or checkpoint-correction record changed after analysis freeze")
    roots = {}
    for key, item in registration["immutable_inputs"].items():
        root = Path(item["path"])
        identity = _tree_identity(root)
        if (identity["files"], identity["bytes"], identity["sha256"]) != (item["files"], item["bytes"], item["sha256"]):
            raise ValueError(f"immutable {key} snapshot identity mismatch")
        roots[key] = root
    policies = {}
    for policy in ("base_dp", "constrain_dp"):
        old, new = _rows(roots["primary_r27"], policy), _rows(roots["candidate_r30"], policy)
        for index, (a, b) in enumerate(zip(old, new)):
            if (a.get("scene_index") != index or b.get("scene_index") != index
                    or (a.get("scene_id"), a.get("seed")) != (b.get("scene_id"), b.get("seed"))
                    or a.get("status") != "evaluated" or b.get("status") != "evaluated"
                    or a.get("restored_bit_exact") is not True or b.get("restored_bit_exact") is not True
                    or a.get("closed_loop_trace", [{}])[0].get("policy_observation")
                    != b.get("closed_loop_trace", [{}])[0].get("policy_observation")):
                raise ValueError(f"{policy} paired scene identity/restore mismatch at index {index}")
        counts = {name: sum(bool(row["result"][field]) for row in new)
                  for name, field in METRICS.items()}
        successful_steps = [int(row["result"]["elapsed_steps"]) for row in new
                            if row["result"].get("native_success")]
        channel_native, channel_encoded, transitions = [], [], {}
        for row in new:
            for trace in row["closed_loop_trace"]:
                if trace.get("action_encoding_profile") != CONTROLLER_RANGE["name"]:
                    raise ValueError("R30 trace is missing the registered codec profile")
                native = np.asarray(trace["predicted_native_action"], dtype=np.float64)
                encoded = np.asarray(trace["encoded_action"], dtype=np.float64)
                expected = np.asarray(encode_native(native, CONTROLLER_RANGE), dtype=np.float64)
                if not np.allclose(encoded, expected, atol=1e-6, rtol=1e-6):
                    raise ValueError("encoded/native action trace pair is inconsistent")
                channel_native.append(native)
                channel_encoded.append(encoded)
        for name, field in METRICS.items():
            transitions[name] = _transitions(old, new, field, inverse=name in ("contact", "xy"))
        policies[policy] = {
            "episodes": 30, "runtime_errors": sum(row.get("status") != "evaluated" for row in new),
            "counts": counts,
            "failed_scene_indices": [row["scene_index"] for row in new if not row["result"].get("native_success")],
            "failed_scenes": [{"index": row["scene_index"], "scene_id": row["scene_id"],
                               "termination_reason": row["result"].get("termination_reason"),
                               "steps": row["result"].get("elapsed_steps"),
                               "release": row.get("release"),
                               "first_contact": row.get("first_contact"),
                               "crossing": row.get("crossing")}
                              for row in new if not row["result"].get("native_success")],
            "successful_steps": {"count": len(successful_steps),
                                 "median": float(np.median(successful_steps)) if successful_steps else None,
                                 "p90_nearest_rank": (sorted(successful_steps)[int(np.ceil(.9 * len(successful_steps))) - 1]
                                                       if successful_steps else None),
                                 "max": max(successful_steps) if successful_steps else None},
            "transitions_vs_R27": transitions,
            "action_observability": {
                "traced_executed_actions": len(channel_native),
                "predicted_native_action_mean": np.mean(channel_native, axis=0).tolist(),
                "encoded_action_mean": np.mean(channel_encoded, axis=0).tolist(),
                "all_encoded_native_pairs_verified": True,
            },
        }
    c = policies["constrain_dp"]["counts"]
    passed = (policies["constrain_dp"]["runtime_errors"] == 0
              and c["task"] >= 27 and c["safe"] >= 26
              and c["contact"] <= 3 and c["xy"] <= 3)
    result = {
        "format": "placesphere_r30_action_codec_authoritative_analysis_v1",
        "registration": Path(registration_path).as_posix(),
        "registration_sha256": sha256_file(registration_path),
        "primary": "R27 DDIM100 eta0 single plan; frozen canonical 30 scene IDs/seeds",
        "codec": CONTROLLER_RANGE,
        "denominator": {"per_policy": 30, "runtime_errors_retained": True,
                        "runtime_errors": {p: v["runtime_errors"] for p, v in policies.items()}},
        "policies": policies,
        "gate": {"requirements": {"task_min": 27, "safe_min": 26,
                                   "contact_max": 3, "xy_max": 3},
                 "observed": c, "passed": passed,
                 "formal08_preparation": "parent review required" if passed else "stop"},
        "comparison_scope": "both policies' weights changed through matched re-training; end-to-end codec pipeline comparison, not control-only attribution",
    }
    Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.register:
        print(json.dumps(register(args.registration), indent=2), flush=True)
    else:
        print(json.dumps(analyze(args.registration, args.output), indent=2), flush=True)


if __name__ == "__main__":
    main()
