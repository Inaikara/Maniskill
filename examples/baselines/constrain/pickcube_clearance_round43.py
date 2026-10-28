"""R43: independent geometry/frame repair and one fail-closed frozen-data precheck."""

import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_body_hint import T_TCP_HAND, T_TCP_LINK7
from examples.baselines.constrain.pickcube_body_hint_round41 import BASE, ROBOT_SOURCE, URDF
from examples.baselines.constrain.pickcube_clearance_proxy_r43 import (
    BODY_MARGIN_M, BODY_PREDICTION_P95_M, CUBE_XY_MARGIN_M, GEOMETRY_LINKS,
    GEOMETRY_TOLERANCE_M, MAX_POINTS_PER_LINK, PRECHECK_SCHEMA,
    RESPONSE_IMPROVEMENT, SAMPLE_SPHERE_PADDING_M, TRAINING_SCHEMA,
    carried_cube_xy_gap, cylinder_sdf, native_forbidden_radius,
    padded_cylinder_sdf, transform_link_points_to_world,
)

EXP = Path("outputs/exp/20261029_052324_pickcube_clearance_r43")
CKPT = Path("outputs/ckpt/20261029_052324_pickcube_clearance_r43_constrain")
DATA = {"h5_sha256": engine.DATA_H5_SHA, "manifest_sha256": engine.DATA_MANIFEST_SHA,
        "validation_sha256": "f4756f697540fdd21904d0cdd3c9fca83e35bec089f292271aa91695d46d5b8f",
        "episode_ids": list(range(1000, 1500)), "class": "obstacle", "count": 500}
SOURCE_NAMES = (
    "examples/baselines/constrain/pickcube_clearance_proxy_r43.py",
    "examples/baselines/constrain/pickcube_clearance_round43.py",
    "examples/baselines/constrain/pickcube_clearance_train_r43.py",
    "examples/baselines/constrain/pickcube_clearance_proxy_r42.py",
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/pickcube_xy_gauge.py",
    "examples/baselines/constrain/pickcube_body_hint.py",
    "examples/baselines/constrain/pickcube_body_hint_round41.py",
    "examples/baselines/constrain/materialize_pickcube_fullscene500.py",
    str(URDF), str(ROBOT_SOURCE),
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_files():
    return {name: sha256_file(name) for name in SOURCE_NAMES}


def registered_assets():
    root = ET.parse(URDF).getroot()
    paths = set()
    mesh_root = Path("mani_skill/assets/robots/panda")
    for name in GEOMETRY_LINKS:
        link = root.find(f"link[@name='{name}']")
        for node in link.findall("collision/geometry/mesh"):
            paths.add(str(mesh_root / node.attrib["filename"]))
    return {"urdf": sha256_file(URDF), "robot_source": sha256_file(ROBOT_SOURCE),
            "meshes": {path: sha256_file(path) for path in sorted(paths)},
            "collision_sampling_config": {"links": list(GEOMETRY_LINKS),
                "max_points_per_link": MAX_POINTS_PER_LINK, "point_padding_m": SAMPLE_SPHERE_PADDING_M}}


def _origin_matrix(node):
    if node is None:
        return np.eye(4)
    xyz = np.fromstring(node.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(node.attrib.get("rpy", "0 0 0"), sep=" ")
    from scipy.spatial.transform import Rotation
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    matrix[:3, 3] = xyz
    return matrix


def _sample_clouds():
    """Deterministic collision-shape samples in each link frame, capped at 256."""
    import trimesh
    root = ET.parse(URDF).getroot()
    clouds, metadata = {}, {}
    mesh_root = Path("mani_skill/assets/robots/panda")
    for name in GEOMETRY_LINKS:
        link = root.find(f"link[@name='{name}']")
        if link is None:
            raise ValueError(f"registered collision link is absent from URDF: {name}")
        points = []
        sources = []
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh_node, box_node = geometry.find("mesh"), geometry.find("box")
            origin = _origin_matrix(collision.find("origin"))
            if mesh_node is not None:
                path = mesh_root / mesh_node.attrib["filename"]
                mesh = trimesh.load_mesh(path, process=False)
                scale = np.fromstring(mesh_node.attrib.get("scale", "1 1 1"), sep=" ")
                local = np.asarray(mesh.vertices) * scale
                candidates = np.concatenate((local, np.asarray(mesh.triangles).mean(axis=1)))
                indices = np.linspace(0, len(candidates)-1, min(512, len(candidates)), dtype=int)
                local = candidates[indices]
                points.append(local @ origin[:3, :3].T + origin[:3, 3])
                sources.append({"path": str(path), "sha256": sha256_file(path),
                                "vertices": len(mesh.vertices), "faces": len(mesh.faces),
                                "sample_indices": indices.tolist(), "kind": "mesh_vertices_face_centroids"})
            elif box_node is not None:
                half = np.fromstring(box_node.attrib["size"], sep=" ") / 2
                local = np.array([[x, y, z] for x in (-half[0], half[0])
                                  for y in (-half[1], half[1]) for z in (-half[2], half[2])])
                points.append(local @ origin[:3, :3].T + origin[:3, 3])
                sources.append({"kind": "exact_box_corners", "half_size_m": half.tolist(),
                                "origin": origin.tolist()})
            else:
                raise ValueError(f"unsupported registered collision geometry on {name}")
        cloud = np.concatenate(points, axis=0)
        index = np.linspace(0, len(cloud)-1, min(MAX_POINTS_PER_LINK, len(cloud)), dtype=int)
        clouds[name] = cloud[index]
        metadata[name] = {"source_shapes": sources, "all_points_before_cap": len(cloud),
                          "selection_indices": index.tolist(), "samples": len(index)}
    return clouds, metadata


def _native_pose_validation(clouds, manifest_rows):
    """Restore saved qpos into one read-only native fixture; never call env.step."""
    import h5py
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    ids = [1000, 1023, 1100, 1200, 1300, 1400, 1499, 1055, 1250]
    env = make_pickcube_env(obstacle_radius=.03, obstacle_height=.18)
    records = []
    frame_count = 0
    finger_rows = {name: [] for name in ("panda_leftfinger", "panda_rightfinger")}
    try:
        base = env.unwrapped
        env.reset(seed=0, options={"obstacle_enabled": True, "obstacle_center": [0.1, .2],
                                   "cube_xy": [-.1, .2], "goal_pos": [.1, .2, .12]})
        links = base.agent.robot.links_map
        with h5py.File(engine.DATA_H5, "r") as h5:
            for tid in ids:
                observations = np.asarray(h5[f"traj_{tid}/obs"])
                for frame in np.linspace(0, len(observations)-1, 4, dtype=int):
                    raw = observations[frame]
                    qpos = torch.as_tensor(raw[:9], dtype=torch.float32, device=base.device)[None]
                    base.agent.robot.set_qpos(qpos)
                    base.agent.robot.set_qvel(torch.zeros_like(qpos))
                    base.scene.update_render()
                    def matrix(link):
                        return link.pose.to_transformation_matrix()[0].detach().cpu().numpy()
                    tcp = matrix(base.agent.tcp)
                    record = {"raw": raw.copy(), "tcp": tcp.copy(), "native": {}}
                    for name in GEOMETRY_LINKS:
                        native_link = matrix(links[name])
                        record["native"][name] = native_link
                        if name in ("panda_leftfinger", "panda_rightfinger"):
                            finger_rows[name].append((raw[7 if name.endswith("leftfinger") else 8],
                                                      np.linalg.inv(tcp) @ native_link))
                    records.append(record)
                    frame_count += 1
    finally:
        env.close()
    finger_model = {}
    for name, rows in finger_rows.items():
        q = np.asarray([r[0] for r in rows], dtype=np.float64)
        transforms = np.asarray([r[1] for r in rows])
        calibration_pair = next(((i, j) for i in range(len(q)) for j in range(i+1, len(q))
                                 if abs(q[i]-q[j]) > 1e-6), None)
        if calibration_pair is None:
            raise ValueError(f"existing states do not identify prismatic motion for {name}")
        i, j = calibration_pair
        q_ref = float(q[i]); q_diff = float(q[j]-q[i])
        axis = (transforms[j, :3, 3]-transforms[i, :3, 3]) / q_diff
        rotation_spread = float(np.max(np.abs(transforms[:, :3, :3]-transforms[i, :3, :3])))
        translation_residuals = []
        for k in range(len(q)):
            predicted_t = transforms[i, :3, 3] + axis*(q[k]-q_ref)
            translation_residuals.append(float(np.max(np.abs(predicted_t-transforms[k, :3, 3]))))
        translation_residual = max(translation_residuals)
        finger_model[name] = {"qpos_index": 7 if name.endswith("leftfinger") else 8,
            "calibration_state_indices": [i, j], "q_reference_m": q_ref,
            "T_tcp_finger_at_reference": transforms[i].tolist(),
            "axis_tcp_m_per_q": axis.tolist(), "rotation_max_abs_spread": rotation_spread,
            "translation_max_abs_residual_m": translation_residual}
    errors = {name: [] for name in GEOMETRY_LINKS}
    for record in records:
        tcp = record["tcp"]
        raw = record["raw"]
        for name in GEOMETRY_LINKS:
            if name == "panda_hand":
                derived = tcp @ np.asarray(T_TCP_HAND)
            elif name == "panda_link7":
                derived = tcp @ np.asarray(T_TCP_LINK7)
            else:
                model = finger_model[name]
                derived = np.asarray(model["T_tcp_finger_at_reference"]).copy()
                derived[:3, 3] += np.asarray(model["axis_tcp_m_per_q"]) * (
                    raw[model["qpos_index"]]-model["q_reference_m"])
                derived = tcp @ derived
                if max(model["rotation_max_abs_spread"], model["translation_max_abs_residual_m"]) > GEOMETRY_TOLERANCE_M:
                    errors[name].append(float("inf")); continue
            native_link = record["native"][name]
            native_points = clouds[name] @ native_link[:3, :3].T + native_link[:3, 3]
            derived_points = clouds[name] @ derived[:3, :3].T + derived[:3, 3]
            errors[name].append(float(np.max(np.abs(derived_points-native_points))))
    return {"states": frame_count, "point_max_abs_by_link_m": {k: max(v) for k, v in errors.items()},
            "finger_models": finger_model, "max_abs_tolerance_m": GEOMETRY_TOLERANCE_M,
            "method": "saved qpos read-only FK; independent native-link pose point comparison; no env.step"}


def _rotation_wxyz(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _tcp_matrix(raw):
    value = np.eye(4); value[:3, :3] = _rotation_wxyz(raw[22:26]); value[:3, 3] = raw[19:22]
    return value


def _body_points(raw, clouds, finger_models, *, translation=None, orientation_raw=None):
    tcp = _tcp_matrix(raw)
    if translation is not None:
        tcp[:3, 3] = translation
    if orientation_raw is not None:
        tcp[:3, :3] = _rotation_wxyz(orientation_raw[22:26])
    result = []
    for name, cloud in clouds.items():
        if name == "panda_hand": relation = np.asarray(T_TCP_HAND)
        elif name == "panda_link7": relation = np.asarray(T_TCP_LINK7)
        else:
            model = finger_models[name]
            relation = np.asarray(model["T_tcp_finger_at_reference"]).copy()
            relation[:3, 3] += np.asarray(model["axis_tcp_m_per_q"]) * (raw[model["qpos_index"]]-model["q_reference_m"])
        result.append(transform_link_points_to_world(cloud, tcp, relation))
    return np.concatenate(result)


def _precheck_once(registration_path=None, result_directory="precheck"):
    import h5py
    from examples.baselines.constrain.pickcube_clearance_proxy_r43 import transform_link_points_to_world
    reg_path = Path(registration_path) if registration_path else EXP / "preregistration.json"
    reg = json.loads(reg_path.read_text())
    body = dict(reg); digest = body.pop("registration_sha256", None)
    if digest != _digest(body) or reg["source_files"] != source_files():
        raise ValueError("R43 frozen registration/source identity mismatch")
    if reg.get("asset_hashes") != registered_assets():
        raise ValueError("R43 frozen geometry asset identity mismatch")
    actual_data = {**DATA, "h5_sha256": sha256_file(engine.DATA_H5),
                   "manifest_sha256": sha256_file(engine.DATA_MANIFEST)}
    if actual_data != DATA:
        raise ValueError("R43 frozen dataset source identity mismatch")
    clouds, cloud_metadata = _sample_clouds()
    manifest = json.loads(Path(engine.DATA_MANIFEST).read_text())
    rows = {int(row["episode_id"]): row for row in manifest["episodes"]}
    assets = registered_assets()
    native = _native_pose_validation(clouds, rows)
    from examples.baselines.constrain.pickcube_clearance_proxy_r42 import response_features, ridge_predict
    sdf_values, physical_sdf_values, raw_points_count, all_finite = [], [], 0, True
    radius_errors, center_errors, height_errors = [], [], []
    cube_gaps, cube_active = [], 0
    # Compute recorded physical centers and separately inflated training proxy values.
    with h5py.File(engine.DATA_H5, "r") as h5:
        for tid in range(1000, 1500):
            row = rows[tid]
            cylinder = row.get("cylinder")
            if not cylinder or not all(k in cylinder for k in ("center_xy", "radius", "height")):
                raise ValueError(f"missing recorded cylinder for trajectory {tid}")
            center = np.asarray(cylinder["center_xy"], dtype=np.float64)
            observations = np.asarray(h5[f"traj_{tid}/obs"], dtype=np.float64)
            for raw in observations:
                q = raw[22:26]
                w, x, y, z = q / np.linalg.norm(q)
                rotation = np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                tcp = np.eye(4); tcp[:3,:3] = rotation; tcp[:3,3] = raw[19:22]
                for name, local in clouds.items():
                    if name == "panda_hand": relation = np.asarray(T_TCP_HAND)
                    elif name == "panda_link7": relation = np.asarray(T_TCP_LINK7)
                    else:
                        model = native["finger_models"][name]
                        relation = np.asarray(model["T_tcp_finger_at_reference"]).copy()
                        relation[:3,3] += np.asarray(model["axis_tcp_m_per_q"]) * (raw[model["qpos_index"]]-model["q_reference_m"])
                    world = transform_link_points_to_world(local, tcp, relation)
                    # This is a training proxy SDF. Physical-center values remain separately defined.
                    points_tensor = torch.as_tensor(world)
                    physical_sdf = cylinder_sdf(points_tensor, center, cylinder["radius"], cylinder["height"]).numpy()
                    sdf = physical_sdf - SAMPLE_SPHERE_PADDING_M
                    physical_sdf_values.extend(physical_sdf.tolist())
                    sdf_values.extend(sdf.tolist()); raw_points_count += len(sdf)
                    all_finite &= bool(np.isfinite(sdf).all())
                radius_errors.append(abs(float(raw[44])-float(cylinder["radius"])))
                center_errors.append(float(np.max(np.abs(raw[42:44]-center))))
                radius = native_forbidden_radius(float(raw[44]), 0.02, 0.005)
                cube_gap = carried_cube_xy_gap(torch.as_tensor(raw), raw[42:44], radius)
                if torch.isfinite(cube_gap):
                    cube_gaps.append(float(cube_gap)); cube_active += 1
            if row.get("layout", {}).get("cylinder") != cylinder:
                raise ValueError(f"trajectory {tid} cylinder manifest copies disagree")
            height_errors.append(abs(float(row["layout"]["cylinder"]["height"])-float(cylinder["height"])))
    # Response coefficients remain the frozen R42 asset: no fitting/refit in R43.
    old_fit = Path("outputs/exp/20261029_050700_pickcube_clearance_r42/precheck/frozen_response_fit.json")
    if not old_fit.is_file():
        raise FileNotFoundError(old_fit)
    fit_artifact = json.loads(old_fit.read_text())
    dyn = fit_artifact["metrics"]
    response_model = fit_artifact["model"]
    body_errors, zero_body_errors = [], []
    with h5py.File(engine.DATA_H5, "r") as h5:
        for tid in range(1000, 1500):
            if tid % 5:
                continue
            observations = np.asarray(h5[f"traj_{tid}/obs"], dtype=np.float64)
            actions = np.asarray(h5[f"traj_{tid}/actions"], dtype=np.float64)
            features = response_features(observations, actions)
            predicted = ridge_predict(response_model, features)
            # feature row 0 is current obs[1], teacher action[1], and target obs[2].
            for index, delta in enumerate(predicted):
                current, following = observations[index+1], observations[index+2]
                pred_xyz = current[19:22] + delta[:3]
                predicted_points = _body_points(current, clouds, native["finger_models"],
                                                translation=pred_xyz, orientation_raw=current)
                zero_points = _body_points(current, clouds, native["finger_models"], orientation_raw=current)
                actual_points = _body_points(following, clouds, native["finger_models"])
                body_errors.extend(np.linalg.norm(predicted_points-actual_points, axis=1).tolist())
                zero_body_errors.extend(np.linalg.norm(zero_points-actual_points, axis=1).tolist())
    body_errors = np.asarray(body_errors)
    zero_body_errors = np.asarray(zero_body_errors)
    body_mean_improvement = float(1-body_errors.mean()/zero_body_errors.mean())
    sdf_values = np.asarray(sdf_values)
    metrics = {
        "native_point_max_abs_m": max(max(native["point_max_abs_by_link_m"].values()),
                                       native["point_max_abs_by_link_m"]["panda_hand"],
                                       native["point_max_abs_by_link_m"]["panda_link7"]),
        "native_point_max_abs_by_link_m": native["point_max_abs_by_link_m"],
        "native_states": native["states"], "all_four_links_covered": set(clouds) == set(GEOMETRY_LINKS),
        "padding_applied": True, "point_padding_m": SAMPLE_SPHERE_PADDING_M,
        "proxy_points": raw_points_count, "proxy_sdf_finite": all_finite,
        "physical_unpadded_center_sdf_quantiles_m": {str(q): float(np.percentile(physical_sdf_values,q)) for q in (0,1,5,25,50,75,90,95,99,100)},
        "proxy_sdf_quantiles_m": {str(q): float(np.percentile(sdf_values,q)) for q in (0,1,5,25,50,75,90,95,99,100)},
        "manifest_radius_max_abs_error_m": max(radius_errors), "manifest_center_max_abs_error_m": max(center_errors),
        "manifest_height_max_abs_error_m": max(height_errors),
        "cube_xy_kernel_exercised": cube_active > 0, "cube_xy_active_grasp_frames": cube_active,
        "cube_xy_gap_quantiles_m": {str(q): float(np.percentile(cube_gaps,q)) for q in (0,1,5,25,50,75,95,99,100)},
        "response_improvement_fraction": dyn["mean_error_improvement_fraction"],
        "response_mean_position_error_m": dyn["mean_position_error_m"],
        "response_p95_position_error_m": dyn["p95_position_error_m"],
        "heldout_body_prediction_mean_m": float(body_errors.mean()),
        "heldout_body_prediction_p95_m": float(np.percentile(body_errors,95)),
        "heldout_body_zero_motion_mean_m": float(zero_body_errors.mean()),
        "heldout_body_mean_improvement_fraction": body_mean_improvement,
        "response_model_sha256": sha256_file(old_fit),
        "response_interpretation": "R42 registered 21-feature frozen ridge; response metric alone does not validate full-body next-state prediction",
    }
    # No admission based on flags: all independent measurements must meet fixed constants.
    passed = (native["states"] == 36 and metrics["native_point_max_abs_m"] <= GEOMETRY_TOLERANCE_M
              and all_finite and metrics["all_four_links_covered"] and metrics["cube_xy_kernel_exercised"]
              and metrics["heldout_body_prediction_p95_m"] <= BODY_PREDICTION_P95_M
              and body_mean_improvement >= RESPONSE_IMPROVEMENT
              and metrics["response_improvement_fraction"] >= RESPONSE_IMPROVEMENT
              and dyn["p95_position_error_m"] <= BODY_PREDICTION_P95_M
              and metrics["manifest_radius_max_abs_error_m"] <= GEOMETRY_TOLERANCE_M
              and metrics["manifest_height_max_abs_error_m"] <= GEOMETRY_TOLERANCE_M)
    source = source_files()
    result = {"schema": PRECHECK_SCHEMA, "status": "passed" if passed else "failed",
        "all_prechecks_passed": passed, "training_permitted": passed,
        "registration_sha256": digest, "registration_file_sha256": sha256_file(reg_path),
        "source_files": source, "data": DATA, "asset_hashes": assets,
        "geometry_assets": cloud_metadata, "native_fk_validation": native,
        "metrics": metrics, "hard_fail_reasons": [key for key, ok in {
            "native geometry center transforms <= 1e-5": metrics["native_point_max_abs_m"] <= GEOMETRY_TOLERANCE_M,
            "finite padded proxy SDF": all_finite,
            "all four collision links covered": metrics["all_four_links_covered"],
            "carried-cube XY component implemented and exercised": metrics["cube_xy_kernel_exercised"],
            "response mean improvement >= 10%": metrics["response_improvement_fraction"] >= RESPONSE_IMPROVEMENT,
            "held-out full-body next-point p95 <= 1 cm": metrics["heldout_body_prediction_p95_m"] <= BODY_PREDICTION_P95_M,
            "held-out full-body mean improves >= 10%": body_mean_improvement >= RESPONSE_IMPROVEMENT,
            "manifest/raw center/radius and height agreement": max(metrics["manifest_radius_max_abs_error_m"], metrics["manifest_center_max_abs_error_m"], metrics["manifest_height_max_abs_error_m"]) <= GEOMETRY_TOLERANCE_M}.items() if not ok],
        "no_env_step": True, "new_episodes": 0, "training_started": False, "native_trial_started": False}
    result_path = EXP / result_directory / "geometry_response_precheck.json"
    result_path.parent.mkdir(parents=True, exist_ok=False)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def register():
    if EXP.exists() or CKPT.exists():
        raise FileExistsError("R43 output paths must be new")
    EXP.mkdir(parents=True, exist_ok=False)
    reg = {"round": 43, "schema": "pickcube_clearance_preregistration_r43_v1",
        "status": "frozen_before_precheck", "source_files": source_files(), "data": DATA,
        "asset_hashes": registered_assets(),
        "base_checkpoint": str(BASE), "base_checkpoint_sha256": sha256_file(BASE),
        "geometry": {"links": list(GEOMETRY_LINKS), "max_points_per_link": MAX_POINTS_PER_LINK,
                     "point_padding_m": SAMPLE_SPHERE_PADDING_M, "body_margin_m": BODY_MARGIN_M,
                     "cube_xy_margin_m": CUBE_XY_MARGIN_M,
                     "forbidden_radius": "obstacle_collision_radius+sqrt(2)*cube_half_size+safety_margin"},
        "precheck_criteria": {"native_point_max_abs_m_lte": GEOMETRY_TOLERANCE_M,
            "heldout_body_prediction_p95_m_lte": BODY_PREDICTION_P95_M,
            "response_improvement_fraction_gte": RESPONSE_IMPROVEMENT,
            "response_fit_source": "immutable R42 frozen response fit; no refit"},
        "scope": "existing frozen obstacle IDs 1000-1499 only; read-only native FK; no new data, layouts, or env.step",
        "precheck_attempt_limit": 1}
    reg["registration_sha256"] = _digest(reg)
    (EXP / "preregistration.json").write_text(json.dumps(reg, indent=2, sort_keys=True)+"\n")
    return reg


def register_final_precheck():
    """Freeze all final kernels/loaders/actors before the fresh acceptance pass."""
    attempt=len(list(EXP.glob("preregistration_final*.json")))+1
    suffix="" if attempt==1 else f"_{attempt:02d}"
    registration_path=EXP/f"preregistration_final{suffix}.json"
    result_directory=f"precheck_final{suffix}"
    if registration_path.exists() or (EXP/result_directory).exists():
        raise FileExistsError("R43 final precheck registration/attempt already exists")
    previous = json.loads((EXP / "preregistration.json").read_text())
    reg = {"round":43,"schema":"pickcube_clearance_preregistration_r43_v1",
        "status":"final_source_frozen_before_precheck","attempt":attempt,
        "supersedes_registration_sha256":previous["registration_sha256"],
        "source_files":source_files(),"data":DATA,"asset_hashes":registered_assets(),
        "base_checkpoint":str(BASE),"base_checkpoint_sha256":sha256_file(BASE),
        "geometry":{"links":list(GEOMETRY_LINKS),"max_points_per_link":MAX_POINTS_PER_LINK,
            "point_padding_m":SAMPLE_SPHERE_PADDING_M,"body_margin_m":BODY_MARGIN_M,
            "cube_xy_margin_m":CUBE_XY_MARGIN_M,
            "forbidden_radius":"obstacle_collision_radius+sqrt(2)*cube_half_size+safety_margin",
            "swept_fractions":[.25,.5,.75,1.0]},
        "precheck_criteria":{"native_point_max_abs_m_lte":GEOMETRY_TOLERANCE_M,
            "heldout_body_prediction_p95_m_lte":BODY_PREDICTION_P95_M,
            "response_improvement_fraction_gte":RESPONSE_IMPROVEMENT,
            "response_fit_source":"immutable R42 frozen response fit; no refit"},
        "scope":"frozen IDs 1000-1499; saved-state native FK only; no new data/layouts/env.step",
        "precheck_attempt_limit":1}
    reg["registration_sha256"]=_digest(reg)
    registration_path.write_text(json.dumps(reg,indent=2,sort_keys=True)+"\n")
    return reg


def run_final_precheck():
    registrations=sorted(EXP.glob("preregistration_final*.json"))
    if not registrations: raise FileNotFoundError("R43 final source registration missing")
    registration_path=registrations[-1]
    suffix=registration_path.stem.removeprefix("preregistration_final")
    return _precheck_once(registration_path,"precheck_final"+suffix)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("register", "precheck", "register-final", "precheck-final", "register-final-next", "precheck-final-next"))
    args = parser.parse_args()
    if args.command == "register": value=register()
    elif args.command == "precheck": value=_precheck_once()
    elif args.command in ("register-final","register-final-next"): value=register_final_precheck()
    elif args.command=="precheck-final": value=_precheck_once(EXP/"preregistration_final.json","precheck_final")
    else: value=run_final_precheck()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
