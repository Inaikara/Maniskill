"""Bounded non-selecting engineering feedback on repeated Round-20 dev failures."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from examples.baselines.constrain.placesphere_matched_terminal import (
    RELEASE_HOLD_STEPS_CANDIDATES,
    dev_layouts,
    freeze_identity,
    run_role_bundle,
)

TARGETS = ("placesphere-000", "placesphere-003", "placesphere-019")
VARIANTS = (8, 16)


def run(output_dir, *, episode_wall_s=160, phase_wall_s=3000):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    layouts = {row["scene_id"]: row for row in dev_layouts()}
    parent_identity = freeze_identity()["files_sha256"]
    module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    prereg = {
        "kind": "non_selecting_engineering_feedback",
        "parent_phase_b_identity": parent_identity,
        "probe_module_sha256": module_sha,
        "cases": list(TARGETS),
        "variants": list(VARIANTS),
        "candidate_set_matches_parent_preregistration": list(VARIANTS) == list(RELEASE_HOLD_STEPS_CANDIDATES[1:]),
        "selection_use": False,
        "training_use": False,
        "independent_dev_claim": False,
        "episode_wall_s": episode_wall_s,
        "phase_wall_s": phase_wall_s,
        "per_episode_action_budget": 200,
        "argv": __import__("sys").argv,
        "scope": "Repeated frozen-dev cases only; reports engineering feedback; does not re-pick the frozen calibration candidate.",
    }
    (output / "preregistration.json").write_text(json.dumps(prereg, indent=2, sort_keys=True) + "\n")
    deadline = time.monotonic() + phase_wall_s
    rows = []
    for scene_id in TARGETS:
        for hold in VARIANTS:
            if time.monotonic() >= deadline:
                rows.append({"scene_id": scene_id, "hold_steps": hold, "complete": False,
                             "error": "phase_wall_exceeded"})
                break
            started = time.monotonic()
            bundle = run_role_bundle(
                layouts[scene_id], margin_m=0.015, release_hold_steps=hold,
                episode_wall_s=episode_wall_s)
            rows.append({"scene_id": scene_id, "hold_steps": hold,
                         "elapsed_s": time.monotonic() - started,
                         "bundle": bundle, "complete": True})
            (output / "partial.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n")
    result = {
        "complete": len(rows) == len(TARGETS) * len(VARIANTS) and all(row["complete"] for row in rows),
        "selection_use": False, "training_use": False, "independent_dev_claim": False,
        "parent_phase_b_identity": parent_identity, "probe_module_sha256": module_sha,
        "rows": rows,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episode-wall-s", type=int, default=160)
    parser.add_argument("--phase-wall-s", type=int, default=3000)
    args = parser.parse_args()
    result = run(args.output_dir, episode_wall_s=args.episode_wall_s, phase_wall_s=args.phase_wall_s)
    print(json.dumps({"complete": result["complete"], "rows": len(result["rows"]),
                      "selection_use": False, "output": args.output_dir}, sort_keys=True))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
