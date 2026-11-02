"""Read-only comparison of frozen R48 PickCube reset features to frozen demos."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet


CATALOG = Path("outputs/exp/20261102_160000_initial_clear_xy_r48_frozen/pickcube")
DEMO = Path("demos/PickCube-transport-detour/20261028_164500_frozen500")
OLD_SCENES = Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes")
OUTPUT = Path("outputs/exp/20261102_200000_xy_protocol_buffer_r51/pickcube_feature_distribution_diagnosis.json")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    return {"min": np.min(values, axis=0).tolist(),
            "p05": np.percentile(values, 5, axis=0).tolist(),
            "median": np.median(values, axis=0).tolist(),
            "p95": np.percentile(values, 95, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist()}


def main():
    catalog = TransportDetourSceneSet(CATALOG)
    r48 = np.stack([catalog.load(i)["observation"] for i in range(len(catalog))]).astype(np.float64)
    demo_manifest = json.loads((DEMO / "trajectory.json").read_text())
    train, kinds = [], []
    with h5py.File(DEMO / "trajectory.h5", "r") as handle:
        for episode in demo_manifest["episodes"]:
            train.append(np.asarray(handle[f"traj_{episode['episode_id']}"]["obs"][0], dtype=np.float64))
            kinds.append(episode["kind"])
    train = np.stack(train)
    old_manifest = json.loads((OLD_SCENES / "manifest.json").read_text())
    fields = {"goal_xyz": (26, 29), "cube_xyz_quat": (29, 36),
              "cube_quaternion": (32, 36), "robot_qpos9": (0, 9),
              "goal_to_cube_xyz": (36, 39)}
    report = {"format": "r51_pickcube_task_feature_contract_diagnosis_v1", "steps": 0,
        "inputs": {str(path): sha256(path) for path in
                   (CATALOG / "manifest.json", CATALOG / "scenes.h5",
                    DEMO / "trajectory.json", DEMO / "trajectory.h5")},
        "cohort_counts": {"r48_eval": len(r48), "training_direct": kinds.count("direct"),
                           "training_obstacle": kinds.count("obstacle")},
        "r48_vs_frozen_training_first_observation": {},
        "r48_goals_metadata_vs_observation": {},
        "old_scene_contract": {"manifest_sha256": sha256(OLD_SCENES / "manifest.json"),
            "generation_seed": old_manifest.get("generation_seed"),
            "height_rule": old_manifest.get("height_rule"),
            "goal_xy_range": distribution(np.asarray([row["goal_xy"] for row in old_manifest["scenes"]])),
            "goal_z_from_frozen_demo_initials": distribution(train[:, 28:29])}}
    for name, (start, end) in fields.items():
        r, d = r48[:, start:end], train[:, start:end]
        report["r48_vs_frozen_training_first_observation"][name] = {
            "r48": distribution(r), "direct_and_obstacle_frozen": distribution(d),
            "r48_minus_training_median": (np.median(r, axis=0)-np.median(d, axis=0)).tolist()}
    report["r48_goals_metadata_vs_observation"] = {
        "metadata_goal_z": "not stored separately; raw saved observation contains the native-reset value",
        "goal_z_raw_observation": distribution(r48[:, 28:29]),
        "goal_z_unique_count": int(np.unique(r48[:, 28]).size),
        "goal_z_at_or_below_0_08_m": int((r48[:, 28] <= .08).sum()),
        "goal_z_above_0_16_m": int((r48[:, 28] > .16).sum()),
        "fixed_training_goal_z_m": float(np.median(train[:, 28])),
        "all_frozen_training_initial_goal_z_is_0_12": bool(np.allclose(train[:, 28], .12, atol=1e-7)),
        "goal_xy_raw": distribution(r48[:, 26:28]),
        "cube_quaternion_norm_range": [float(np.linalg.norm(r48[:, 32:36], axis=1).min()),
                                       float(np.linalg.norm(r48[:, 32:36], axis=1).max())],
        "qpos_scene0_vs_train_median": {"scene0": r48[0, :9].tolist(),
            "training_median": np.median(train[:, :9], axis=0).tolist(),
            "abs_delta": np.abs(r48[0, :9]-np.median(train[:, :9], axis=0)).tolist()}}
    report["source_contract_evidence"] = {
        "R48_sampler": "first native reset reads goal actor pose; second reset explicitly reuses its sampled goal_pos; initial_clear_xy_scenes.py sample_task lines 195-212",
        "historical_train_contract": "all 1,500 immutable demonstration initial observations encode extra/goal_pos z=0.12; old R36 scene manifest records only goal_xy and scene observation contract fixes target through existing task/layout configuration",
        "interpretation": "Native-reset goal-height variability in the R48 catalog is not represented in the frozen training demonstrations. It is a task-distribution mismatch requiring contract resolution before full native trials; it does not justify resampling or filtering scenes."}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT),
        "goal_z": report["r48_goals_metadata_vs_observation"]["goal_z_raw_observation"],
        "goal_xy": report["r48_goals_metadata_vs_observation"]["goal_xy_raw"],
        "goal_z_outside_0_08_to_0_16": report["r48_goals_metadata_vs_observation"]["goal_z_at_or_below_0_08_m"]+report["r48_goals_metadata_vs_observation"]["goal_z_above_0_16_m"]},
        sort_keys=True))


if __name__ == "__main__":
    main()
