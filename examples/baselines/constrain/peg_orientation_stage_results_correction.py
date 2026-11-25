"""Recompute stage-diagnosis minima from retained native episode traces."""

import argparse
import json
from pathlib import Path


def correct_results(experiment_dir):
    """Update minima using all measured pre/post snapshots, including terminal."""
    root = Path(experiment_dir)
    result_path = root / "results.json"
    results = json.loads(result_path.read_text())
    for profile, profile_result in results["profiles"].items():
        episode_dir = root / ("base_indices_0_5" if profile == "Base" else "constrain_indices_0_5")
        for row in profile_result["rows"]:
            episode = json.loads((episode_dir / f"episode_{row['scene_index']:03d}.json").read_text())
            distances_3d = []
            distances_xy = []
            abs_yaws = []
            for trace in episode["phase_trace"]:
                for stage, xy_key in (("preaction", "preaction_observation_peg_hole_xy_m"),
                                      ("postaction", "observation_peg_hole_xy_m")):
                    geometry = trace[f"stage_geometry_{stage}"]
                    distances_3d.append((geometry["peg_center_to_hole_center_distance_m"], trace["control"]))
                    distances_xy.append((trace[xy_key], trace["control"]))
                    abs_yaws.append((abs(geometry["wrapped_peg_plus_x_minus_box_plus_x_yaw_rad"]), trace["control"]))
            distance_3d, control_3d = min(distances_3d)
            distance_xy, _ = min(distances_xy)
            abs_yaw, _ = min(abs_yaws)
            row["minimum_center_3d_m"] = distance_3d
            row["minimum_center_3d_control"] = control_3d
            row["minimum_center_xy_m"] = distance_xy
            row["minimum_abs_wrapped_yaw_rad"] = abs_yaw
    results["aggregation_semantics"] = {
        "minimum_center_3d_m": "Minimum pose-derived peg-center to hole-center distance over every available pre-action and post-action stage_geometry snapshot; final post-action snapshot is terminal.",
        "minimum_center_xy_m": "Minimum measured XY distance over preaction_observation_peg_hole_xy_m and observation_peg_hole_xy_m in phase_trace. Not inferred from 3D.",
        "minimum_abs_wrapped_yaw_rad": "Minimum absolute wrapped peg-axis-minus-hole-axis yaw over pre-action and post-action stage_geometry snapshots; terminal signed yaw is separately reported.",
    }
    result_path.write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    correct_results(args.experiment_dir)


if __name__ == "__main__":
    main()
