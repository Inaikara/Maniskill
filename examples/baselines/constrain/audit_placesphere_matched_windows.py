"""Non-selecting audit of action-only and separate native motion window criteria."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.placesphere_base_dp import sha256_file
from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_H5_SHA256,
)

LINEAR_STATIC_M_S = 0.01
ANGULAR_STATIC_RAD_S = 0.5
ACTION_NORM_MAX = 0.02
WINDOW = 8


def audit(data_path):
    if sha256_file(data_path) != MATCHED_H5_SHA256:
        raise ValueError("matched HDF5 identity mismatch")
    roles = {"direct": defaultdict(Counter), "obstacle": defaultdict(Counter)}
    total_windows = Counter()
    with h5py.File(data_path, "r") as handle:
        for key in sorted(handle, key=lambda item: int(item.split("_")[1])):
            group = handle[key]
            role = str(group.attrs["kind"])
            actions = np.asarray(group["actions"])
            metadata = json.loads(group.attrs["metadata_json"])
            phases = metadata["step_phase"]
            linear = metadata["sphere_linear_speed_m_s"]
            angular = metadata["sphere_angular_speed_rad_s"]
            if not (len(phases) == len(linear) == len(angular) == len(actions)):
                raise ValueError(f"{key}: current-action phase/speed evidence length mismatch")
            for begin in range(0, len(actions) - WINDOW + 1, WINDOW):
                action_window = actions[begin:begin + WINDOW]
                if len(action_window) != WINDOW:
                    continue
                total_windows[role] += 1
                action_only = bool(np.all(action_window[:, 6] > 0) and
                                   np.max(np.linalg.norm(action_window[:, :6], axis=1)) <= ACTION_NORM_MAX)
                if not action_only:
                    continue
                phase = phases[begin]
                current_linear = np.asarray(linear[begin:begin + WINDOW])
                current_angular = np.asarray(angular[begin:begin + WINDOW])
                if len(current_linear) != WINDOW or len(current_angular) != WINDOW:
                    raise ValueError(f"{key}: incomplete current-state velocity window")
                linear_ok = bool(np.max(current_linear) <= LINEAR_STATIC_M_S)
                angular_ok = bool(np.max(current_angular) <= ANGULAR_STATIC_RAD_S)
                counts = roles[role][phase]
                counts["action_only_open_near_zero_windows"] += 1
                counts["linear_static_windows"] += int(linear_ok)
                counts["angular_static_windows"] += int(angular_ok)
                counts["both_linear_and_angular_static_windows"] += int(linear_ok and angular_ok)
                counts["linear_static_angular_only_failure_windows"] += int(linear_ok and not angular_ok)

    result = {"h5_sha256": MATCHED_H5_SHA256, "window_actions": WINDOW,
              "window_stride_actions": WINDOW,
              "phase_label": "current executed phase at window first action; no future-state label",
              "action_only_predicate": "all 8 action[6] > 0 and max(norm(action[:6])) <= 0.02",
              "linear_static_threshold_m_s": LINEAR_STATIC_M_S,
              "angular_static_threshold_rad_s": ANGULAR_STATIC_RAD_S,
              "motion_criteria_are_independent": True, "roles": {}}
    for role, phases in roles.items():
        totals = Counter()
        by_phase = {}
        for phase, counter in sorted(phases.items()):
            by_phase[phase] = dict(counter)
            totals.update(counter)
        n = totals["action_only_open_near_zero_windows"]
        result["roles"][role] = {"fixed_stride_8_training_trajectory_windows":
            total_windows[role], "action_only_open_near_zero_windows": n,
            "action_only_open_near_zero_fraction": n / total_windows[role] if total_windows[role] else 0.0,
            "linear_static_within_action_only_windows": totals["linear_static_windows"],
            "angular_static_within_action_only_windows": totals["angular_static_windows"],
            "both_native_static_within_action_only_windows": totals["both_linear_and_angular_static_windows"],
            "linear_static_but_angular_only_failure_windows": totals["linear_static_angular_only_failure_windows"],
            "per_phase": by_phase}
        if n:
            for name in ("linear_static_windows", "angular_static_windows",
                         "both_linear_and_angular_static_windows",
                         "linear_static_angular_only_failure_windows"):
                result["roles"][role][name + "_fraction_of_action_only"] = totals[name] / n
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
