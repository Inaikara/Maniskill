"""R44 preregistered 8-step SE(3) response precheck on frozen PickCube data."""

import hashlib
import json
from pathlib import Path

import numpy as np

from examples.baselines.constrain import pickcube_clearance_round43 as r43
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_clearance_proxy_r43 import (
    GEOMETRY_TOLERANCE_M, GEOMETRY_LINKS, BODY_MARGIN_M,
)
from examples.baselines.constrain import pickcube_zscore_round37 as engine

EXP = Path("outputs/exp/20261029_160000_pickcube_horizon_clearance_r44")
CKPT = Path("outputs/ckpt/20261029_160000_pickcube_horizon_clearance_r44_control")
HORIZON = 8
RIDGE = 1e-6
MODEL_KIND = "SE3_horizon8_clearance"
FIT = EXP / "response_fit.json"
REG = EXP / "preregistration.json"
PRECHECK = EXP / "precheck/response_precheck.json"
CLOSED_REG = EXP / "preregistration_sourceclosed_08.json"
CLOSED_PRECHECK = EXP / "precheck/sourceclosed_08/response_precheck.json"
DATA = dict(r43.DATA)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _unit(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return np.concatenate((aw*bw-(av*bv).sum(-1, keepdims=True),
        aw*bv+bw*av+np.cross(av, bv)), axis=-1)


def quat_conj(q):
    return np.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def quat_log(q):
    q = _unit(q)
    q = np.where(q[..., :1] < 0, -q, q)  # sign-invariant shortest representative
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(n, np.maximum(q[..., :1], 0.0))
    scale = np.where(n > 1e-12, angle / np.maximum(n, 1e-12), 2.0)
    return v * scale


def quat_exp(v):
    v = np.asarray(v, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    half = angle / 2
    scale = np.where(angle > 1e-12, np.sin(half) / np.maximum(angle, 1e-12), .5)
    return _unit(np.concatenate((np.cos(half), v * scale), axis=-1))


def controller_rotation_prior(native_actions):
    """Cumulative Panda root-aligned XYZ command rotations for up to 8 actions."""
    result=[]
    cumulative=np.array([1.0,0.0,0.0,0.0])
    for action in np.asarray(native_actions,dtype=np.float64):
        rot=action[3:6].copy();norm=np.linalg.norm(rot)
        if norm>1: rot=rot/norm
        x,y,z=rot*(-.1)  # Panda PDEEPoseControllerConfig.rot_lower
        qx=np.array([np.cos(x/2),np.sin(x/2),0,0])
        qy=np.array([np.cos(y/2),0,np.sin(y/2),0])
        qz=np.array([np.cos(z/2),0,0,np.sin(z/2)])
        cumulative=_unit(quat_mul(quat_mul(quat_mul(qx,qy),qz),cumulative))
        result.append(cumulative.copy())
    return np.asarray(result)


def rotation_matrix(q):
    w, x, y, z = _unit(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def action_to_physical(action):
    """Panda root translation + signed root-aligned XYZ Euler controller units."""
    a = np.asarray(action, dtype=np.float64)
    rotation=a[...,3:6].copy()
    norm=np.linalg.norm(rotation,axis=-1,keepdims=True)
    rotation=rotation/np.maximum(norm,1.0)
    return np.concatenate((a[..., :3] * .1, rotation * -.1), axis=-1)


def make_features(previous, current, actions8):
    """72 inputs: native physical planned commands, current q/v, previous TCP delta."""
    a = action_to_physical(actions8).reshape(-1)
    qprev, qcur = _unit(previous[22:26]), _unit(current[22:26])
    delta_p = current[19:22] - previous[19:22]
    delta_r = quat_log(quat_mul(qcur, quat_conj(qprev)))
    return np.concatenate((a, current[:9], current[9:18], delta_p, delta_r))


def make_targets(current, future8):
    q0 = _unit(current[22:26])
    out = []
    for future in future8:
        out.extend(future[19:22] - current[19:22])
        qh = _unit(future[22:26])
        out.extend(quat_log(quat_mul(qh, quat_conj(q0))))
    return np.asarray(out, dtype=np.float64)


def _rows():
    import h5py
    manifest = json.loads(Path(engine.DATA_MANIFEST).read_text())
    manifest_rows = {int(row["episode_id"]): row for row in manifest["episodes"]}
    train_x, train_y, val_x, val_y = [], [], [], []
    excluded = {"not_grasped": 0, "incomplete_future": 0, "eligible_windows": 0}
    with h5py.File(engine.DATA_H5, "r") as h5:
        for episode_id in range(1000, 1500):
            obs = np.asarray(h5[f"traj_{episode_id}/obs"], dtype=np.float64)
            actions = np.asarray(h5[f"traj_{episode_id}/actions"], dtype=np.float64)
            # Observation row t is state t. Index zero has no preceding observed frame.
            for t in range(1, len(actions)):
                if len(actions) - t < HORIZON or len(obs) - (t + HORIZON) < 0:
                    excluded["incomplete_future"] += 1
                    continue
                current = obs[t]
                if current[18] <= .5:
                    excluded["not_grasped"] += 1
                    continue
                x = make_features(obs[t-1], current, actions[t:t+HORIZON, :6])
                y = make_targets(current, obs[t+1:t+HORIZON+1])
                excluded["eligible_windows"] += 1
                (val_x if episode_id % 5 == 0 else train_x).append(x)
                (val_y if episode_id % 5 == 0 else train_y).append(y)
    return tuple(np.asarray(v, dtype=np.float64) for v in (train_x, train_y, val_x, val_y)), excluded


def register():
    if EXP.exists() or CKPT.exists():
        raise FileExistsError("R44 experiment/checkpoint destination already exists")
    EXP.mkdir(parents=True)
    import xml.etree.ElementTree as ET
    controller = Path("mani_skill/agents/controllers/pd_ee_pose.py")
    robot = Path("mani_skill/agents/robots/panda/panda.py")
    r37 = Path("examples/baselines/constrain/pickcube_zscore_round37.py")
    body = {
        "round": 44, "schema": "pickcube_horizon8_se3_preregistration_v1",
        "model_kind": MODEL_KIND, "status": "frozen_before_fit",
        "data": {**DATA, "h5_sha256": sha256_file(engine.DATA_H5),
                 "manifest_sha256": sha256_file(engine.DATA_MANIFEST)},
        "base_checkpoint_sha256": sha256_file(r43.BASE),
        "assets": r43.registered_assets(),
        "controller_contract": {"controller_source_sha256": sha256_file(controller),
            "robot_source_sha256": sha256_file(robot), "frame": "root_translation:root_aligned_body_rotation",
            "translation_native_scale_m": .1, "rotation_euler_xyz_native_scale_rad": .1,
            "rotation_bounds_rad": [-.1, .1]},
        "fit": {"feature_dim": 72, "target_dim": 48, "horizon": HORIZON,
            "train_episode_ids": "1000..1499 where id % 5 != 0",
            "validation_episode_ids": "1000..1499 where id % 5 == 0",
            "eligibility": "current raw is_grasped and eight recorded future actions and T+1 observations",
            "feature_order": ["8 physical native 6D planned commands", "current qpos9", "current qvel9",
                "previous-to-current TCP world delta position3", "previous-to-current world rotation log3"],
            "target_order": "for h=1..8, TCP world dp3 and log(q(t+h)*inverse(q(t)))3",
            "scaling": "feature column mean/std fitted on train IDs only; std floor 1e-12",
            "solver": "float64 centered ridge, lambda=1e-6, single fixed fit"},
        "geometry": {"links": list(GEOMETRY_LINKS), "body_margin_m": BODY_MARGIN_M,
            "point_padding_m": .005, "finger_pose": "current finger q held for forecast; actual future finger q used only for error measurement",
            "body_p95_overall_and_each_horizon_lte_m": .01,
            "mean_body_error_improvement_gte": .10, "native_point_transform_max_abs_lte_m": GEOMETRY_TOLERANCE_M},
        "limitations": ["current grasped carry windows only; grasp approach coverage excluded",
            "8 discrete forecast poses are not continuous swept collision proof",
            "no fitted-policy rollouts or future observations are inputs"],
        "source_files": {str(r37): sha256_file(r37), str(controller): sha256_file(controller),
            str(robot): sha256_file(robot), "examples/baselines/constrain/pickcube_clearance_round44.py": sha256_file(__file__)},
    }
    body["registration_sha256"] = digest(body)
    REG.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def fit():
    reg = json.loads(REG.read_text())
    if reg.get("registration_sha256") != digest({k: v for k, v in reg.items() if k != "registration_sha256"}):
        raise ValueError("R44 preregistration integrity mismatch")
    source = {"examples/baselines/constrain/pickcube_zscore_round37.py": sha256_file("examples/baselines/constrain/pickcube_zscore_round37.py"),
        "mani_skill/agents/controllers/pd_ee_pose.py": sha256_file("mani_skill/agents/controllers/pd_ee_pose.py"),
        "mani_skill/agents/robots/panda/panda.py": sha256_file("mani_skill/agents/robots/panda/panda.py"),
        "examples/baselines/constrain/pickcube_clearance_round44.py": sha256_file(__file__)}
    if source != reg["source_files"]:
        raise ValueError("R44 fit source differs from pre-fit registration")
    (x, y, vx, vy), excluded = _rows()
    if x.shape[1:] != (72,) or y.shape[1:] != (48,) or vx.shape[1:] != (72,) or vy.shape[1:] != (48,):
        raise ValueError("R44 recorded-sequence feature/target dimensions are inconsistent")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-12)
    z = (x-mean)/scale
    coefficient = np.linalg.solve(z.T @ z + RIDGE*np.eye(z.shape[1]), z.T @ (y-y.mean(axis=0)))
    prediction = (vx-mean)/scale @ coefficient + y.mean(axis=0)
    artifact = {"schema": "pickcube_se3_response_horizon8_v1", "model_kind": MODEL_KIND,
        "registration_sha256": reg["registration_sha256"], "fit_episode_ids": "1000..1499; train id%5!=0",
        "validation_episode_ids": "1000..1499; id%5==0", "ridge": RIDGE,
        "feature_mean": mean.tolist(), "feature_scale": scale.tolist(),
        "target_mean": y.mean(axis=0).tolist(), "coefficient": coefficient.tolist(),
        "training_windows": len(x), "validation_windows": len(vx), "excluded_counts": excluded,
        "fit_source_files": source}
    artifact["model_sha256"] = digest(artifact)
    FIT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def _source_closure():
    from examples.baselines.constrain import pickcube_clearance_train_r44 as train44
    from examples.baselines.constrain import pickcube_clearance_eval_r44 as eval44
    from examples.baselines.constrain import pickcube_clearance_round43 as geometry43
    names=set(train44._source_files())|set(eval44.SOURCES)|set(geometry43.SOURCE_NAMES)
    names.update(("examples/baselines/constrain/pickcube_clearance_analysis_r43.py",
        "examples/baselines/constrain/constrain_ckpt.py","examples/baselines/constrain/constrain_denoiser.py",
        "examples/baselines/constrain/train_constrain.py","examples/baselines/constrain/pickcube_base_dp.py",
        "examples/baselines/constrain/pickcube_constrain.py","examples/baselines/constrain/pickcube_observation_norm.py",
        "examples/baselines/constrain/pickcube_clearance_round44.py"))
    return {name:sha256_file(name) for name in sorted(names)}


def register_source_closure():
    if CLOSED_REG.exists() or CLOSED_PRECHECK.parent.exists():
        raise FileExistsError("R44 source-closed precheck registration already exists")
    fit=json.loads(FIT.read_text());old=json.loads(REG.read_text())
    if fit.get("registration_sha256")!=old.get("registration_sha256") or fit.get("model_sha256")!=digest({k:v for k,v in fit.items() if k!="model_sha256"}):
        raise ValueError("fixed R44 fit coefficients or original preregistration identity mismatch")
    prior=EXP/"preregistration_sourceclosed_07.json"
    body={"schema":"pickcube_horizon8_sourceclosure_v1","attempt":8,"status":"frozen_before_fixed_model_precheck",
        "original_registration_sha256":old["registration_sha256"],"fit_sha256":sha256_file(FIT),"fit_model_sha256":fit["model_sha256"],
        "data":{"h5_sha256":sha256_file(engine.DATA_H5),"manifest_sha256":sha256_file(engine.DATA_MANIFEST),
            "validation_sha256":DATA["validation_sha256"],"episode_ids":DATA["episode_ids"]},
        "base_checkpoint_sha256":sha256_file(r43.BASE),"asset_hashes":r43.registered_assets(),
        "supersedes_sourceclosure_sha256":json.loads(prior.read_text())["closure_sha256"],
        "source_files":_source_closure(),
        "rotation_action_prior":"registered root-aligned XYZ Euler from panda.py, rot_lower=-0.1 and norm<=1; fit coefficients remain unchanged",
        "acceptance":"unchanged fixed fit; same thresholds; no refit or tuning"}
    body["closure_sha256"]=digest(body)
    CLOSED_REG.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    return body


def precheck(closure_path=None):
    """Measure all horizons against held-out recorded poses and frozen body points."""
    import h5py
    from examples.baselines.constrain.pickcube_clearance_round43 import _sample_clouds, _native_pose_validation, _body_points
    from examples.baselines.constrain.pickcube_clearance_proxy_r43 import cylinder_sdf
    reg, fit_artifact = json.loads(REG.read_text()), json.loads(FIT.read_text())
    if fit_artifact.get("model_kind") != MODEL_KIND or fit_artifact.get("registration_sha256") != reg["registration_sha256"]:
        raise ValueError("R44 response fit type/registration mismatch")
    if fit_artifact.get("model_sha256") != digest({k:v for k,v in fit_artifact.items() if k!="model_sha256"}):
        raise ValueError("R44 response fit coefficient hash mismatch")
    clouds, cloud_metadata = _sample_clouds()
    manifest = json.loads(Path(engine.DATA_MANIFEST).read_text())
    rows = {int(row["episode_id"]): row for row in manifest["episodes"]}
    native = _native_pose_validation(clouds, rows)
    mean = np.asarray(fit_artifact["feature_mean"]); scale = np.asarray(fit_artifact["feature_scale"])
    target_mean = np.asarray(fit_artifact["target_mean"]); coef = np.asarray(fit_artifact["coefficient"])
    tcp_errors = [[] for _ in range(HORIZON)]
    body_errors = [[] for _ in range(HORIZON)]
    zero_errors = [[] for _ in range(HORIZON)]
    eligible = 0
    with h5py.File(engine.DATA_H5, "r") as h5:
        for episode_id in range(1000, 1500):
            if episode_id % 5:
                continue
            obs = np.asarray(h5[f"traj_{episode_id}/obs"], dtype=np.float64)
            actions = np.asarray(h5[f"traj_{episode_id}/actions"], dtype=np.float64)
            for t in range(1, len(actions)-HORIZON+1):
                current = obs[t]
                if current[18] <= .5:
                    continue
                feature = (make_features(obs[t-1], current, actions[t:t+HORIZON, :6])-mean)/scale
                prediction = feature @ coef + target_mean
                eligible += 1
                for h in range(HORIZON):
                    actual = obs[t+h+1]
                    dp = prediction[6*h:6*h+3]
                    rot = prediction[6*h+3:6*h+6]
                    p = current[19:22] + dp
                    prior = controller_rotation_prior(actions[t:t+HORIZON,:6])
                    q = quat_mul(quat_mul(prior[h],quat_exp(rot)), _unit(current[22:26]))
                    predicted_raw = current.copy(); predicted_raw[19:22] = p; predicted_raw[22:26] = q
                    predicted_points = _body_points(predicted_raw, clouds, native["finger_models"])
                    actual_points = _body_points(actual, clouds, native["finger_models"])
                    zero_points = _body_points(current, clouds, native["finger_models"])
                    tcp_errors[h].append(float(np.linalg.norm(p-actual[19:22])))
                    body_errors[h].extend(np.linalg.norm(predicted_points-actual_points, axis=1).tolist())
                    zero_errors[h].extend(np.linalg.norm(zero_points-actual_points, axis=1).tolist())
    per_horizon = {}
    for h in range(HORIZON):
        e, z = np.asarray(body_errors[h]), np.asarray(zero_errors[h])
        per_horizon[str(h+1)] = {"tcp_position_mean_m": float(np.mean(tcp_errors[h])),
            "tcp_position_p95_m": float(np.percentile(tcp_errors[h],95)),
            "body_point_mean_m": float(e.mean()), "body_point_p95_m": float(np.percentile(e,95)),
            "zero_motion_body_mean_m": float(z.mean()),
            "mean_improvement_fraction": float(1-e.mean()/z.mean())}
    all_body, all_zero = np.concatenate([np.asarray(x) for x in body_errors]), np.concatenate([np.asarray(x) for x in zero_errors])
    metrics = {"eligible_validation_windows": eligible,
        "raw_tcp_position_error_quantiles_m": {str(q): float(np.percentile(np.concatenate(tcp_errors),q)) for q in (50,95,99,100)},
        "body_point_error_quantiles_m": {str(q): float(np.percentile(all_body,q)) for q in (50,95,99,100)},
        "overall_body_point_p95_m": float(np.percentile(all_body,95)),
        "overall_zero_motion_body_mean_m": float(all_zero.mean()),
        "overall_body_mean_m": float(all_body.mean()),
        "overall_mean_improvement_fraction": float(1-all_body.mean()/all_zero.mean()),
        "per_horizon": per_horizon,
        "native_point_max_abs_m": max(native["point_max_abs_by_link_m"].values()),
        "native_point_max_abs_by_link_m": native["point_max_abs_by_link_m"],
        "all_four_links_covered": set(clouds)==set(GEOMETRY_LINKS),
        "geometry_asset_metadata": cloud_metadata}
    criteria = {"native_geometry_max_error": metrics["native_point_max_abs_m"] <= GEOMETRY_TOLERANCE_M,
        "all_four_links": metrics["all_four_links_covered"],
        "overall_body_p95": metrics["overall_body_point_p95_m"] <= .01,
        "each_horizon_body_p95": all(v["body_point_p95_m"] <= .01 for v in per_horizon.values()),
        "body_mean_improvement": metrics["overall_mean_improvement_fraction"] >= .10}
    output_path=PRECHECK; closure=None
    if closure_path is not None:
        closure=json.loads(Path(closure_path).read_text())
        body={k:v for k,v in closure.items() if k!="closure_sha256"}
        if closure.get("closure_sha256")!=digest(body) or closure.get("source_files")!=_source_closure():
            raise ValueError("R44 source-closed preregistration/source integrity mismatch")
        if closure.get("fit_sha256")!=sha256_file(FIT) or closure.get("fit_model_sha256")!=fit_artifact["model_sha256"]:
            raise ValueError("R44 source closure does not pin exact fixed coefficients")
        if closure.get("original_registration_sha256")!=reg["registration_sha256"]:
            raise ValueError("R44 source closure belongs to another original registration")
        output_path=CLOSED_PRECHECK
    result = {"schema": "pickcube_se3_horizon8_precheck_v1", "model_kind": MODEL_KIND,
        "horizon": HORIZON, "status": "passed" if all(criteria.values()) else "failed",
        "training_permitted": all(criteria.values()), "registration_sha256":reg["registration_sha256"],
        "source_closure_sha256":None if closure is None else closure["closure_sha256"],
        "source_files":None if closure is None else closure["source_files"],
        "native_fk_validation":native,
        "fit_sha256":sha256_file(FIT), "data":DATA, "metrics":metrics, "criteria":criteria,
        "carry_window_scope":"current is_grasped true; grasp approach excluded",
        "finger_forecast":"current finger q held; actual next finger q used for error only",
        "rotation_action_prior":"exact registered Panda root-aligned XYZ Euler conversion with native norm limit and rot_lower=-0.1; learned residual ridge unchanged",
        "discrete_samples_only":True, "continuous_sweep_proof":False,
        "no_env_step":True, "new_episodes":0, "training_started":False, "native_trial_started":False}
    result["report_sha256"] = digest(result)
    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("register","fit","precheck","register-source-closed","precheck-source-closed"))
    args=parser.parse_args()
    value=register() if args.command=="register" else fit() if args.command=="fit" else register_source_closure() if args.command=="register-source-closed" else precheck(CLOSED_REG) if args.command=="precheck-source-closed" else precheck()
    print(json.dumps(value,indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
