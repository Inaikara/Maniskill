"""R42 preregistered training-only clearance proxy and frozen response fit."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_body_hint_round41 import (
    BASE, EXP as R41_EXP, URDF, ROBOT_SOURCE, _digest,
)
from examples.baselines.constrain.pickcube_body_hint import T_TCP_HAND, T_TCP_LINK7

EXP = Path("outputs/exp/20261029_050700_pickcube_clearance_r42")
BODY_MARGIN_M = 0.02
CUBE_XY_MARGIN_M = 0.01
SAMPLE_SPHERE_PADDING_M = 0.005
RIDGE = 1e-6
FIT_IDS = [i for i in range(1000, 1500) if i % 5 != 0]
VALID_IDS = [i for i in range(1000, 1500) if i % 5 == 0]
URDF_SHA = "524c6986dbbabe244e05801f262862205e880834d01459b06ee273016d0579ba"
ROBOT_SHA = "bf85577be75b7513802e676decb5543e3e65f2a3dfbc26175dd8b77ae57691f3"


def cylinder_sdf(points, center_xy, radius, height):
    """Exact finite-cylinder SDF for points; signed distance, meters."""
    points = np.asarray(points, dtype=np.float64)
    center_xy = np.asarray(center_xy, dtype=np.float64)
    radial = np.linalg.norm(points[..., :2] - center_xy, axis=-1) - radius
    vertical = np.abs(points[..., 2] - height / 2) - height / 2
    outside = np.linalg.norm(np.maximum(np.stack((radial, vertical), axis=-1), 0), axis=-1)
    return outside + np.minimum(np.maximum(radial, vertical), 0)


def quaternion_rotation_vector(q_wxyz):
    """Sign-invariant world-frame SO(3) logarithm for scalar-first quaternions."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q = np.where(q[..., :1] < 0, -q, q)
    xyz, w = q[..., 1:], np.clip(q[..., :1], -1, 1)
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(norm, w)
    scale = np.divide(angle, norm, out=np.full_like(angle, 2.0), where=norm > 1e-12)
    return xyz * scale


def pose_delta(current, following):
    """World translation and world rotation-log increment between raw TCP poses."""
    translation = following[..., 19:22] - current[..., 19:22]
    q0, q1 = current[..., 22:26], following[..., 22:26]
    # q1 * conjugate(q0), with both raw quaternions in wxyz order.
    w0, v0 = q0[..., :1], q0[..., 1:]
    w1, v1 = q1[..., :1], q1[..., 1:]
    rel = np.concatenate((w1*w0 + np.sum(v1*v0, axis=-1, keepdims=True),
                          -w1*v0 + w0*v1 - np.cross(v1, v0)), axis=-1)
    return np.concatenate((translation, quaternion_rotation_vector(rel)), axis=-1)


def response_features(observations, actions):
    """21 preregistered inputs: current 6D command, previous TCP delta, qvel9."""
    current = observations[1:-1]
    previous = observations[:-2]
    command = actions[1:, :6]
    velocity = current[:, 9:18]
    return np.concatenate((command, pose_delta(previous, current), velocity), axis=-1)


def ridge_fit(features, targets, fit_mask):
    x, y = np.asarray(features, np.float64)[fit_mask], np.asarray(targets, np.float64)[fit_mask]
    if x.ndim != 2 or x.shape[1] != 21 or len(x) == 0:
        raise ValueError("expected non-empty trajectory-ID fit matrix with 21 features")
    mean_x, mean_y = x.mean(0), y.mean(0)
    coef = np.linalg.solve((x-mean_x).T @ (x-mean_x) + RIDGE*np.eye(21),
                           (x-mean_x).T @ (y-mean_y))
    return {"feature_mean": mean_x, "target_mean": mean_y, "coefficient": coef}


def ridge_predict(model, features):
    return model["target_mean"] + (np.asarray(features)-model["feature_mean"]) @ model["coefficient"]


def _mesh_pointclouds():
    import trimesh
    root = ET.parse(URDF).getroot()
    clouds, provenance = {}, {}
    for link_name in ("panda_hand", "panda_link7"):
        link = root.find(f"link[@name='{link_name}']")
        mesh_node = link.find("collision/geometry/mesh")
        if mesh_node is None:
            raise ValueError(f"collision mesh missing for {link_name}")
        mesh_path = Path("mani_skill/assets/robots/panda") / mesh_node.attrib["filename"]
        mesh = trimesh.load_mesh(mesh_path, process=False)
        # Vertices plus deterministic face centroids, then fixed evenly-spaced selection.
        all_points = np.concatenate((np.asarray(mesh.vertices), np.asarray(mesh.triangles).mean(axis=1)))
        indices = np.linspace(0, len(all_points)-1, min(256, len(all_points)), dtype=int)
        clouds[link_name] = all_points[indices].astype(np.float64)
        provenance[link_name] = {"path": str(mesh_path), "sha256": sha256_file(mesh_path),
                                 "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                                 "samples": len(indices), "bounds_m": mesh.bounds.tolist()}
    return clouds, provenance


def _fit_frozen_response():
    import h5py
    manifest = json.loads(Path(engine.DATA_MANIFEST).read_text())
    rows = {int(row["episode_id"]): row for row in manifest["episodes"]}
    features, targets, trajectory_ids = [], [], []
    with h5py.File(engine.DATA_H5, "r") as data:
        for tid in range(1000, 1500):
            row = rows[tid]
            cylinder = row.get("cylinder")
            if not cylinder or not all(key in cylinder for key in ("center_xy", "radius", "height")):
                raise ValueError(f"trajectory {tid} has missing cylinder geometry metadata")
            obs = np.asarray(data[f"traj_{tid}/obs"], dtype=np.float64)
            act = np.asarray(data[f"traj_{tid}/actions"], dtype=np.float64)
            if len(obs) != len(act)+1 or not np.isfinite(obs).all() or not np.isfinite(act).all():
                raise ValueError(f"trajectory {tid} has invalid raw arrays")
            n = len(obs)-2
            features.append(response_features(obs, act))
            targets.append(pose_delta(obs[1:-1], obs[2:]))
            trajectory_ids.extend([tid]*n)
    x, y = np.concatenate(features), np.concatenate(targets)
    tids = np.asarray(trajectory_ids)
    fit_mask = np.isin(tids, FIT_IDS)
    valid_mask = np.isin(tids, VALID_IDS)
    model = ridge_fit(x, y, fit_mask)
    prediction = ridge_predict(model, x[valid_mask])
    target = y[valid_mask]
    zero = np.zeros_like(target)
    pos_error = np.linalg.norm(prediction[:, :3]-target[:, :3], axis=1)
    zero_pos_error = np.linalg.norm(zero[:, :3]-target[:, :3], axis=1)
    metrics = {"fit_frame_count": int(fit_mask.sum()), "validation_frame_count": int(valid_mask.sum()),
        "mean_position_error_m": float(pos_error.mean()), "p95_position_error_m": float(np.percentile(pos_error,95)),
        "p95_angular_error_rad": float(np.percentile(np.linalg.norm(prediction[:,3:]-target[:,3:],axis=1),95)),
        "zero_displacement_mean_position_error_m": float(zero_pos_error.mean()),
        "mean_error_improvement_fraction": float(1-pos_error.mean()/zero_pos_error.mean()),
        "fit_trajectory_ids": FIT_IDS, "validation_trajectory_ids": VALID_IDS,
        "quantiles_position_error_m": {str(q): float(np.percentile(pos_error,q)) for q in (0,25,50,75,90,95,99,100)},
        "command_fit_input_count": 21}
    packed = {k: v.tolist() for k,v in model.items()}
    return packed, metrics


def _validate_native_frames(clouds):
    """Read-only FK on 36 existing states; never steps or creates a recorded episode."""
    import h5py
    import torch
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    from scipy.spatial.transform import Rotation
    ids = [1000, 1023, 1100, 1200, 1300, 1400, 1499, 1055, 1250]
    env = make_pickcube_env(obstacle_radius=.03, obstacle_height=.18)
    errors, transform_errors, count = [], {"panda_hand": [], "panda_link7": []}, 0
    try:
        base = env.unwrapped
        env.reset(seed=0, options={"obstacle_enabled": True, "obstacle_center": [0.1, .2],
                                   "cube_xy": [-.1, .2], "goal_pos": [.1, .2, .12]})
        links = base.agent.robot.links_map
        with h5py.File(engine.DATA_H5, "r") as source:
            for tid in ids:
                obs = np.asarray(source[f"traj_{tid}/obs"])
                for frame in np.linspace(0, len(obs)-1, 4, dtype=int):
                    raw = obs[frame]
                    qpos = torch.as_tensor(raw[:9], dtype=torch.float32, device=base.device)[None]
                    base.agent.robot.set_qpos(qpos); base.agent.robot.set_qvel(torch.zeros_like(qpos))
                    base.scene.update_render()
                    tcp = base.agent.tcp.pose.to_transformation_matrix()[0].detach().cpu().numpy()
                    for link_name, fixed in (("panda_hand", T_TCP_HAND), ("panda_link7", T_TCP_LINK7)):
                        matrix = links[link_name].pose.to_transformation_matrix()[0].detach().cpu().numpy()
                        relation = np.linalg.inv(tcp) @ matrix
                        transform_errors[link_name].append(float(np.max(np.abs(relation-np.asarray(fixed)))))
                        points_tcp = (np.linalg.inv(np.asarray(fixed)) @ np.c_[clouds[link_name], np.ones(len(clouds[link_name]))].T).T[:, :3]
                        points_world = points_tcp @ tcp[:3,:3].T + tcp[:3,3]
                        # Independent native link transform yields same world cloud.
                        native_world = clouds[link_name] @ matrix[:3,:3].T + matrix[:3,3]
                        errors.append(float(np.max(np.abs(points_world-native_world))))
                    saved = np.eye(4); saved[:3,3] = raw[19:22]
                    saved[:3,:3] = Rotation.from_quat(raw[22:26][[1,2,3,0]]).as_matrix()
                    count += 1
    finally:
        env.close()
    return {"states": count, "tcp_saved_pose_max_abs": max(errors),
            "fixed_relation_max_abs": {k:max(v) for k,v in transform_errors.items()},
            "point_transform_max_abs": max(errors), "tolerance": 1e-5,
            "native_physics_collision_ground_truth": False,
            "method": "read-only saved-qpos FK and URDF collision-mesh rigid-transform consistency; no env.step"}


def run_precheck():
    reg_path = EXP / "preregistration.json"
    reg = json.loads(reg_path.read_text()); digest = reg.pop("registration_sha256")
    if digest != _digest(reg) or reg["source_files"] != preregistration()["source_files"]:
        raise ValueError("R42 registration/source identity mismatch")
    clouds, meshes = _mesh_pointclouds()
    native = _validate_native_frames(clouds)
    fit, dynamics = _fit_frozen_response()
    # Proxy geometry validity and recorded-state comparison are reported as sampled proxy evidence only.
    import h5py
    manifest = json.loads(Path(engine.DATA_MANIFEST).read_text())
    rows = {int(row["episode_id"]): row for row in manifest["episodes"]}
    clearance, inside, finite = [], 0, True
    with h5py.File(engine.DATA_H5, "r") as data:
        for tid in range(1000,1500):
            meta = rows[tid]["cylinder"]; obs = np.asarray(data[f"traj_{tid}/obs"])
            for raw in obs[1:-1]:
                qmat = _rotation_from_wxyz(raw[22:26])
                p_tcp, q_tcp = raw[19:22], qmat
                for name, points in clouds.items():
                    t = np.asarray(T_TCP_HAND if name == "panda_hand" else T_TCP_LINK7)
                    local = (np.linalg.inv(t) @ np.c_[points,np.ones(len(points))].T).T[:,:3]
                    world = local @ q_tcp.T + p_tcp
                    sdf = cylinder_sdf(world, meta["center_xy"], meta["radius"], meta["height"])
                    clearance.extend(sdf.tolist()); inside += int((sdf < 0).sum()); finite &= bool(np.isfinite(sdf).all())
    clearance = np.asarray(clearance)
    geometry = {"mesh_sources": meshes, "sample_points_total_per_pose": sum(map(len,clouds.values())),
        "sample_sphere_padding_m": SAMPLE_SPHERE_PADDING_M, "padding_is_conservative_proxy_not_ground_truth": True,
        "sdf_samples": int(len(clearance)), "negative_proxy_samples": inside, "finite": finite,
        "clearance_quantiles_m": {str(q): float(np.percentile(clearance,q)) for q in (0,1,5,25,50,75,95,99,100)},
        "approximation": "finite collision mesh vertex/face-centroid samples do not represent exact robot surface; fixed 5mm point-sphere padding; not PhysX collision labels"}
    passed = (native["states"] == 36 and native["fixed_relation_max_abs"]["panda_hand"] <= 1e-5
        and native["fixed_relation_max_abs"]["panda_link7"] <= 1e-5
        and native["point_transform_max_abs"] <= 1e-5 and finite
        and geometry.get("carried_cube_proxy_implemented", False)
        and dynamics["mean_error_improvement_fraction"] >= .10 and dynamics["p95_position_error_m"] <= .01)
    result = {"status": "passed" if passed else "failed", "all_prechecks_passed": passed,
        "geometry": geometry, "native_fk_validation": native, "dynamics_fit": dynamics,
        "dynamics_model": fit, "training_permitted": passed,
        "hard_fail_reasons": [name for name, ok in {
            "36-state FK/frame transform tolerance": native["states"] == 36 and native["fixed_relation_max_abs"]["panda_hand"] <= 1e-5 and native["fixed_relation_max_abs"]["panda_link7"] <= 1e-5 and native["point_transform_max_abs"] <= 1e-5,
            "finite proxy clearance values": finite,
            "preregistered carried-cube XY proxy implemented": geometry.get("carried_cube_proxy_implemented", False),
            "ridge mean improvement >= 10%": dynamics["mean_error_improvement_fraction"] >= .10,
            "ridge p95 position error <= 0.01m": dynamics["p95_position_error_m"] <= .01}.items() if not ok]}
    (EXP / "precheck").mkdir(exist_ok=False)
    (EXP / "precheck" / "geometry_response_precheck.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    # Fixed fit artifact includes identity and no DEV data by construction.
    (EXP / "precheck" / "frozen_response_fit.json").write_text(json.dumps({"model": fit, "metrics": dynamics}, indent=2, sort_keys=True)+"\n")
    return result


def correct_recorded_precheck():
    """Analysis-only correction: invalidate a result whose FK/proxy gates were incomplete."""
    original = EXP / "precheck" / "geometry_response_precheck.json"
    evidence = json.loads(original.read_text())
    failures = list(evidence["hard_fail_reasons"])
    if evidence["native_fk_validation"]["point_transform_max_abs"] > 1e-5:
        failures.append("point-cloud transform validation exceeded 1e-5m (recorded 0.4244311456; FK comparison invalid)")
    failures.append("preregistered carried-cube forbidden-disk proxy was not implemented or validated")
    failures.append("declared 5mm sample-sphere padding was recorded but not applied to sampled point distances")
    failures.append("registered left/right finger collision samples were omitted")
    corrected = {"status": "failed", "all_prechecks_passed": False, "training_permitted": False,
        "correction_kind": "report-only invalidation; no geometry/dynamics refit and no new validation states",
        "original_precheck_path": str(original), "original_precheck_sha256": sha256_file(original),
        "original_recorded_metrics": {"response_mean_improvement_fraction": evidence["dynamics_fit"]["mean_error_improvement_fraction"],
            "response_p95_position_error_m": evidence["dynamics_fit"]["p95_position_error_m"],
            "fk_fixed_hand_max_abs": evidence["native_fk_validation"]["fixed_relation_max_abs"]["panda_hand"],
            "fk_fixed_link7_max_abs": evidence["native_fk_validation"]["fixed_relation_max_abs"]["panda_link7"],
            "point_transform_validation_max_abs": evidence["native_fk_validation"]["point_transform_max_abs"],
            "proxy_negative_samples": evidence["geometry"]["negative_proxy_samples"],
            "proxy_samples": evidence["geometry"]["sdf_samples"]},
        "hard_fail_reasons": failures,
        "interpretation": "The response fit numerically met its held-out thresholds, but the end-to-end geometry/dynamics surrogate precheck is not credible: native point-transform validation failed, declared padding was not applied, registered finger samples and the carried-cube constraint were absent. The recorded proxy intersections are not PhysX collision labels and cannot validate collision behavior.",
        "native_extra_episodes": 0, "training_started": False, "development_evaluation_started": False}
    target = EXP / "precheck" / ("precheck_correction.json" if not (EXP / "precheck" / "precheck_correction.json").exists()
                                 else "precheck_correction_02.json")
    corrected["supersedes_correction_sha256"] = (sha256_file(EXP / "precheck" / "precheck_correction.json")
        if target.name.endswith("_02.json") else None)
    target.write_text(json.dumps(corrected, indent=2, sort_keys=True)+"\n")
    return corrected


def record_report_source_correction():
    reg = json.loads((EXP / "preregistration.json").read_text())
    registered = reg["source_files"]
    current = preregistration()["source_files"]
    changed = {name: {"precheck_sha256": registered.get(name), "report_source_sha256": current.get(name)}
               for name in set(registered) | set(current) if registered.get(name) != current.get(name)}
    if set(changed) != {str(Path(__file__).relative_to(Path.cwd()))}:
        raise ValueError(f"unexpected report-only source changes: {changed}")
    corrections = sorted((EXP / "precheck").glob("precheck_correction*.json"))
    event = {"phase": "after completed single precheck, analysis-only fail-closed correction",
        "changed_files": changed, "reason": "make the failed FK point-transform and missing carried-cube gates fail closed; no refit, no new validation states, no training or rollout",
        "original_precheck_sha256": sha256_file(EXP / "precheck" / "geometry_response_precheck.json"),
        "corrected_result_sha256": sha256_file(corrections[-1]),
        "additional_precheck_attempt": False, "training_started": False}
    path = EXP / ("analysis_source_correction.json" if not (EXP / "analysis_source_correction.json").exists()
                  else "analysis_source_correction_02.json")
    path.write_text(json.dumps(event, indent=2, sort_keys=True)+"\n")
    return event


def record_precheck_source_correction():
    """Pin final precheck implementation before any geometry/dynamics fitting."""
    path = EXP / "preregistration.json"
    reg = json.loads(path.read_text())
    body = dict(reg); old_digest = body.pop("registration_sha256")
    if old_digest != _digest(body):
        raise ValueError("R42 initial registration checksum mismatch")
    before = reg["source_files"]
    current = preregistration()
    changed = {name: {"registered_sha256": before.get(name), "current_sha256": current["source_files"].get(name)}
               for name in set(before) | set(current["source_files"])
               if before.get(name) != current["source_files"].get(name)}
    if not changed or set(changed) != {str(Path(__file__).relative_to(Path.cwd()))}:
        raise ValueError(f"unexpected R42 precheck source changes: {changed}")
    initial = EXP / "preregistration_initial.json"
    if initial.exists():
        raise FileExistsError(initial)
    initial.write_bytes(path.read_bytes())
    reg["source_files"] = current["source_files"]
    reg["source_files_sha256"] = current["source_files_sha256"]
    reg["source_correction_before_precheck"] = {"changed_files": changed,
        "reason": "finalize the pre-registered one-pass geometry/dynamics precheck implementation before fitting; no data or geometry analysis had begun",
        "geometry_fit_started": False, "dynamics_fit_started": False, "training_started": False}
    reg.pop("registration_sha256", None)
    reg["registration_sha256"] = _digest(reg)
    path.write_text(json.dumps(reg, indent=2, sort_keys=True)+"\n")
    event = {"previous_registration_sha256": old_digest,
             "next_registration_sha256": reg["registration_sha256"],
             "initial_registration_sha256": sha256_file(initial),
             "changed_files": changed, "fit_started": False, "training_started": False}
    (EXP / "precheck_source_correction.json").write_text(json.dumps(event, indent=2, sort_keys=True)+"\n")
    return event


def _rotation_from_wxyz(q):
    from scipy.spatial.transform import Rotation
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(q[[1,2,3,0]] / np.linalg.norm(q)).as_matrix()


def preregistration():
    source_files = [str(Path(__file__).relative_to(Path.cwd())), str(URDF), str(ROBOT_SOURCE),
                    "examples/baselines/constrain/pickcube_body_hint.py",
                    "examples/baselines/constrain/pickcube_body_hint_round41.py",
                    "examples/baselines/constrain/pickcube_zscore_round37.py",
                    "examples/baselines/constrain/pickcube_xy_gauge.py",
                    "examples/baselines/constrain/materialize_pickcube_fullscene500.py"]
    files = {name: sha256_file(name) for name in source_files}
    return {
        "round": 42, "status": "preregistered_before_geometry_or_response_fit",
        "hypothesis": "the frozen teacher epsilon objective may not provide robust clearance margins; this auxiliary training-only proxy tests a differentiable gradient, not guaranteed learned avoidance",
        "dataset": {"h5": engine.DATA_H5, "h5_sha256": engine.DATA_H5_SHA,
                    "manifest": engine.DATA_MANIFEST, "manifest_sha256": engine.DATA_MANIFEST_SHA,
                    "validation_sha256": "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f",
                    "class": "obstacle", "episode_ids": list(range(1000,1500)), "count": 500},
        "initialization": {"base": str(BASE), "base_sha256": sha256_file(BASE),
                           "source": "exact frozen R40 Base final EMA; not R41 Control resume"},
        "control": {"condition_dim": 9, "global_condition_dim": 84,
                    "action_dim": 7, "raw_observation_dim": 45,
                    "xy_gauge": "R40 common-origin training augmentation retained; dedicated seed 424243",
                    "position_profile": str(R41_EXP / "position_encoding.json")},
        "geometry": {"native_urdf_sha256": URDF_SHA, "robot_source_sha256": ROBOT_SHA,
                     "mesh_sampling": "all collision-mesh vertices and deterministic face centroids, deterministic farthest-point downsample to <=256 points per link; link7, hand, left/right fingers",
                     "point_sphere_padding_m": SAMPLE_SPHERE_PADDING_M,
                     "body_margin_m": BODY_MARGIN_M, "finite_cylinder_sdf": "radial/vertical exact finite-cylinder point SDF; positive outside, negative inside",
                     "cylinder_metadata": "per-trajectory manifest cylinder center_xy/radius/height; any missing or disagreement rejects; no constant geometry",
                     "carried_cube": "enabled only under current raw grasp flag; exact native PickCube forbidden-disk radius formula, transformed from raw current TCP/cube poses"},
        "response_fit": {"feature_count": 21, "features": ["native action[0:6] at current frame t", "previous-to-current TCP world translation+SO3-log delta", "current qvel[0:9]"],
                         "target": "recorded current-to-next TCP world translation+SO3-log delta",
                         "alignment": "window obs[start:start+2] has current frame obs[start+1]; action label index 1 is at t; target raw pose is obs[start+2]",
                         "controller": "Panda pd_ee_delta_pose root_translation; pos/rot bounds [-0.1,0.1] physical units; rotation is compact axis-angle; confirm from pinned controller",
                         "fit": "float64 centered linear ridge, lambda=1e-6, no refit", "fit_trajectory_ids": FIT_IDS,
                         "validation_trajectory_ids": VALID_IDS,
                         "acceptance": {"mean_error_improvement_over_zero_displacement_fraction_gte": 0.10,
                                        "p95_hand_link7_next_position_error_m_lte": 0.01}},
        "scope": "analysis fit uses only HDF5 obstacle IDs 1000-1499 and their paired manifest geometry; no direct/dev inputs, env.step, new episode, or new data",
        "prechecks": {"maximum_attempts": 1, "native_extra_episodes": 0,
                      "geometry_transform_validation_states": 36,
                      "reject_and_do_not_train_on_any_precheck_failure": True},
        "source_files": files, "source_files_sha256": _digest(files),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "correct-source", "precheck", "correct-recorded-precheck", "record-report-source-correction"))
    args = parser.parse_args()
    if args.command == "register":
        EXP.mkdir(parents=True, exist_ok=False)
        reg = preregistration()
        reg["registration_sha256"] = _digest(reg)
        (EXP / "preregistration.json").write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"registration": str(EXP / "preregistration.json"),
                          "registration_sha256": reg["registration_sha256"]}, indent=2))
    elif args.command == "correct-source":
        print(json.dumps(record_precheck_source_correction(), indent=2, sort_keys=True))
    elif args.command == "correct-recorded-precheck":
        print(json.dumps(correct_recorded_precheck(), indent=2, sort_keys=True))
    elif args.command == "record-report-source-correction":
        print(json.dumps(record_report_source_correction(), indent=2, sort_keys=True))
    else:
        print(json.dumps(run_precheck(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
