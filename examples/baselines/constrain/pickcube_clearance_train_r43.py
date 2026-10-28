"""Typed R43 clearance-objective training and inference-only actor loader."""

import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import pickcube_body_hint_round41 as r41
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_body_hint import T_TCP_HAND, T_TCP_LINK7, body_hint_from_raw
from examples.baselines.constrain.pickcube_clearance_proxy_r43 import (
    BODY_MARGIN_M, CUBE_XY_MARGIN_M, GEOMETRY_LINKS, PRECHECK_SCHEMA,
    SAMPLE_SPHERE_PADDING_M, TRAINING_SCHEMA, native_forbidden_radius,
    swept_clearance_components, verify_training_admission,
)
from examples.baselines.constrain.pickcube_clearance_round43 import (
    DATA, EXP, CKPT, _digest, _sample_clouds, registered_assets, source_files,
)

TRAIN_FORMAT = "pickcube_clearance_proxy9_constrain_r43_v1"
TRAIN_REGISTRATION = EXP / "train_preregistration.json"
PRECHECK_REPORT = EXP / "precheck_final_02/geometry_response_precheck.json"
BENCHMARK_PATH = EXP / "train_benchmark_10_real_batches_02.json"
BASE = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt")
R40_EXP = Path("outputs/exp/20261029_004500_pickcube_xy_gauge_r40")
R41_EXP = Path("outputs/exp/20261029_031500_pickcube_bodyhint_r41")
TOTAL_UPDATES = 150000
BATCH_SIZE = 1024
SEED = 1
GAUGE_SEED = 424243
LR = 1e-4
WARMUP = 500
EMA_POWER = 0.75
PROXY_WEIGHT = 0.01
PROXY_WARMUP = 10000
DEVICE = "cuda"
RUN_DIR = EXP / "train_constrain"
CHECKPOINT_DIR = CKPT
RESPONSE_FIT = Path("outputs/exp/20261029_050700_pickcube_clearance_r42/precheck/frozen_response_fit.json")
TRAIN_SOURCE_NAMES = (
    "examples/baselines/constrain/pickcube_clearance_train_r43.py",
    "examples/baselines/constrain/pickcube_clearance_proxy_r43.py",
    "examples/baselines/constrain/pickcube_clearance_proxy_r42.py",
    "examples/baselines/constrain/pickcube_clearance_round43.py",
    "examples/baselines/constrain/pickcube_body_hint.py",
    "examples/baselines/constrain/pickcube_body_hint_round41.py",
    "examples/baselines/constrain/pickcube_xy_gauge.py",
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/materialize_pickcube_fullscene500.py",
)


def train_source_files():
    return {p: sha256_file(p) for p in TRAIN_SOURCE_NAMES}


def _rotation_wxyz(q):
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def _rotation_delta(previous_q, current_q):
    p = previous_q / torch.linalg.vector_norm(previous_q, dim=-1, keepdim=True).clamp_min(1e-12)
    c = current_q / torch.linalg.vector_norm(current_q, dim=-1, keepdim=True).clamp_min(1e-12)
    pw, pv = p[..., :1], p[..., 1:]
    cw, cv = c[..., :1], c[..., 1:]
    w = cw*pw + (cv*pv).sum(-1, keepdim=True)
    v = -cw*pv + pw*cv - torch.cross(cv, pv, dim=-1)
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w, v = w*sign, v*sign
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2*torch.atan2(norm, w.clamp(-1, 1))
    scale = torch.where(norm > 1e-12, angle/norm.clamp_min(1e-12), torch.full_like(norm, 2.0))
    return v*scale


def _response_delta(current_raw, previous_raw, predicted_action6, model, device):
    feature = torch.cat((predicted_action6,
        current_raw[:, 19:22]-previous_raw[:, 19:22],
        _rotation_delta(previous_raw[:, 22:26], current_raw[:, 22:26]),
        current_raw[:, 9:18]), dim=-1)
    mean = torch.as_tensor(model["feature_mean"], dtype=feature.dtype, device=device)
    target_mean = torch.as_tensor(model["target_mean"], dtype=feature.dtype, device=device)
    coef = torch.as_tensor(model["coefficient"], dtype=feature.dtype, device=device)
    return (target_mean + (feature-mean) @ coef)[:, :3]


def _body_clouds_torch(clouds, finger_models, raw, translation_delta):
    pos = raw[:, 19:22]
    rotation = _rotation_wxyz(raw[:, 22:26])
    local_models = {
        "panda_hand": (np.asarray(T_TCP_HAND)[:3, :3], np.asarray(T_TCP_HAND)[:3, 3]),
        "panda_link7": (np.asarray(T_TCP_LINK7)[:3, :3], np.asarray(T_TCP_LINK7)[:3, 3]),
    }
    point_chunks = []
    for name in GEOMETRY_LINKS:
        if name in local_models:
            rel_r, rel_t = local_models[name]
        else:
            profile = finger_models[name]
            rel_r = np.asarray(profile["T_tcp_finger_at_reference"])[:3, :3]
            rel_t = np.asarray(profile["T_tcp_finger_at_reference"])[:3, 3]
        points = torch.as_tensor(clouds[name], dtype=raw.dtype, device=raw.device)
        rel_r = torch.as_tensor(rel_r, dtype=raw.dtype, device=raw.device)
        rel_t = torch.as_tensor(rel_t, dtype=raw.dtype, device=raw.device)
        if name in local_models:
            link_local = points @ rel_r.T + rel_t
            finger_offset = 0
        else:
            profile = finger_models[name]
            qi = profile["qpos_index"]
            axis = torch.as_tensor(profile["axis_tcp_m_per_q"], dtype=raw.dtype, device=raw.device)
            finger_offset = (raw[:, qi]-float(profile["q_reference_m"]))[:, None] * axis
            link_local = points @ rel_r.T + rel_t
        transformed = torch.einsum("bij,nj->bni", rotation, link_local)
        if name not in local_models:
            transformed = transformed + torch.einsum("bij,bj->bi", rotation, finger_offset)[:, None, :]
        point_chunks.append(pos[:, None, :] + translation_delta[:, None, :] + transformed)
    return torch.cat(point_chunks, dim=1)


def _cube_center_xy(raw):
    rotation = _rotation_wxyz(raw[:, 22:26])
    offset_world = raw[:, 29:32]-raw[:, 19:22]
    offset_local = torch.einsum("bji,bj->bi", rotation, offset_world)
    predicted_tcp = raw[:, 19:22]
    predicted_cube = predicted_tcp + torch.einsum("bij,bj->bi", rotation, offset_local)
    return predicted_cube[:, :2]


def clearance_proxy_from_x0(x0, raw, previous_raw, response_model, clouds, finger_models,
                            cylinders, active_mask):
    """Production proxy: clipped x0 action[1], frozen one-step response, swept body/cube costs."""
    active = active_mask.bool()
    if not bool(active.any()):
        zero = x0.sum()*0.0
        return zero, zero, {"active_rows": torch.zeros((), device=x0.device)}
    x0 = x0[active]
    current = raw[active, 1]
    previous = previous_raw[active, 0]
    geom = {key: value[active] for key, value in cylinders.items()}
    delta = _response_delta(current, previous, x0[:, 1, :6], response_model, x0.device)
    body_points = _body_clouds_torch(clouds, finger_models, current, torch.zeros_like(delta))
    current_cube = _cube_center_xy(current)
    radii = geom["radius"] + math.sqrt(2.0)*0.02 + 0.005
    body_cost, cube_cost, stats = swept_clearance_components(
        body_points, delta, current_cube, geom["center"], geom["radius"], geom["height"],
        current[:, 18] > .5, radii)
    stats["active_rows"] = active.sum()
    stats["tcp_translation_delta_rms_m"] = torch.sqrt((delta*delta).mean()).detach()
    return body_cost, cube_cost, stats


def training_registration_payload(precheck_sha256, precheck_registration_sha256, benchmark):
    if (benchmark.get("status")!="complete" or benchmark.get("benchmark_steps")!=10
            or benchmark.get("batch_size")!=BATCH_SIZE
            or max(benchmark.get("disabled_control_epsilon_max_abs",float("inf")),
                   benchmark.get("disabled_control_ddpm100_sample_max_abs",float("inf")))>1e-6
            or not all(np.isfinite([row["epsilon_mse"],row["body"],row["cube"]]).all()
                       for row in benchmark.get("values",[]))):
        raise ValueError("R43 ten-real-batch benchmark/equivalence evidence failed")
    if benchmark.get("projected_150k_seconds_p90",float("inf"))>15000:
        raise ValueError("R43 measured p90 training projection exceeds preregistered 15000s budget")
    return {"schema": TRAINING_SCHEMA, "phase": "single_clearance_objective_control_training",
        "precheck_sha256": precheck_sha256,
        "precheck_registration_sha256": precheck_registration_sha256,
        "source_files": source_files(), "training_source_files": train_source_files(),
        "precheck_source_files": source_files(),
        "data": DATA, "asset_hashes": registered_assets(),
        "response_fit_sha256": sha256_file(RESPONSE_FIT),
        "base_checkpoint_sha256": sha256_file(BASE),
        "profile": {"format": TRAIN_FORMAT, "hint_dim": 9, "global_condition_dim": 84,
            "raw_observation_dim": 45, "state_dim": 42, "action_dim": 7,
            "normalizer_sha256": sha256_file(R41_EXP/"position_encoding.json"),
            "xy_gauge_seed": GAUGE_SEED},
        "objective": {"epsilon_mse": 1.0, "clearance_weight_final": PROXY_WEIGHT,
            "clearance_warmup_steps": PROXY_WARMUP, "active_diffusion_timesteps": [10,30],
            "x0_clip": [-1,1], "executed_action_index": 1,
            "swept_fractions": [0.25,0.5,0.75,1.0], "point_padding_m": SAMPLE_SPHERE_PADDING_M,
            "body_margin_m": BODY_MARGIN_M, "cube_xy_margin_m": CUBE_XY_MARGIN_M,
            "loss_definition": "eps_mse + 0.01*min(step/10000,1)*(body+cube normalized active-row hinge costs)"},
        "benchmark_sha256":sha256_file(BENCHMARK_PATH),
        "training": {"updates": TOTAL_UPDATES, "batch_size": BATCH_SIZE, "seed": SEED,
            "optimizer": "AdamW", "lr": LR, "warmup": WARMUP, "schedule": "cosine",
            "ema_power": EMA_POWER, "base_updates": 0, "budget_seconds": benchmark["budget_seconds"],
            "benchmark": benchmark},
        "selection": "final EMA fixed before training; no dev selection",
        "registration_status": "frozen_before_training"}


def register_training(benchmark):
    if TRAIN_REGISTRATION.exists() or RUN_DIR.exists() or CKPT.exists():
        raise FileExistsError("R43 training paths already exist; no retry/resume is authorized")
    precheck = json.loads(PRECHECK_REPORT.read_text())
    if precheck.get("status") != "passed" or precheck.get("training_permitted") is not True:
        raise ValueError("fresh R43 precheck did not pass")
    registration = training_registration_payload(sha256_file(PRECHECK_REPORT),
        precheck["registration_sha256"], benchmark)
    body = dict(registration); registration["registration_sha256"] = _digest(body)
    TRAIN_REGISTRATION.write_text(json.dumps(registration, indent=2, sort_keys=True)+"\n")
    return registration


def _verify_training_registration():
    reg = json.loads(TRAIN_REGISTRATION.read_text())
    if reg.get("registration_sha256") != _digest({k:v for k,v in reg.items() if k!="registration_sha256"}):
        raise ValueError("R43 training registration checksum mismatch")
    if reg.get("training_source_files") != train_source_files() or reg.get("precheck_source_files") != source_files():
        raise ValueError("R43 train/precheck source identity changed after phase freeze")
    verify_training_admission(TRAIN_REGISTRATION, PRECHECK_REPORT,
        source_files(), DATA, registered_assets())
    if reg.get("response_fit_sha256") != sha256_file(RESPONSE_FIT) or reg.get("base_checkpoint_sha256") != sha256_file(BASE):
        raise ValueError("R43 frozen response or R40 Base hash mismatch")
    if reg.get("profile",{}).get("normalizer_sha256")!=sha256_file(R41_EXP/"position_encoding.json"):
        raise ValueError("R43 R41 position-unit normalizer hash mismatch")
    if reg.get("benchmark_sha256")!=sha256_file(BENCHMARK_PATH):
        raise ValueError("R43 benchmark evidence file hash mismatch")
    return reg


def benchmark_ten_real_batches(device="cuda"):
    """Bounded disposable 10-step benchmark through the exact production loss path."""
    if device!="cuda" or not torch.cuda.is_available(): raise RuntimeError("benchmark requires CUDA")
    r41._install_profile()
    from examples.baselines.constrain.pickcube_xy_gauge import augment_window_observations,make_generator
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    stats=engine.load_stats(R41_EXP/"position_encoding.json")
    dataset=engine.native_dataset("obstacle",include_obstacle=True)
    if len(dataset)!=45840: raise ValueError("benchmark frozen window count mismatch")
    rows={int(row["episode_id"]):row for row in json.loads(Path(engine.DATA_MANIFEST).read_text())["episodes"]}
    windows=dataset.windows
    clouds,_=_sample_clouds()
    finger_models=json.loads(PRECHECK_REPORT.read_text())["native_fk_validation"]["finger_models"]
    response=json.loads(RESPONSE_FIT.read_text())["model"]
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    checkpoint=torch.load(BASE,map_location="cpu",weights_only=False)
    base=engine.build_agent("cpu");base.load_state_dict(checkpoint["weights"]["ema"],strict=True);base.to(device).eval()
    model=engine.build_constrain_denoiser(CONTROL_CONFIG,device="cpu")
    model.base_net.load_state_dict(base.noise_pred_net.state_dict(),strict=True);model.to(device).train()
    optimizer=torch.optim.AdamW(model.trainable_parameters(),lr=LR,betas=(.95,.999),weight_decay=1e-6)
    scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    scheduler.alphas_cumprod=scheduler.alphas_cumprod.to(device)
    gauge=make_generator(SEED)
    # Frozen-base/zero-Control equality through the exact DDPM-100 sampler.
    first=[dataset[i] for i in range(2)]
    raw2=torch.stack([x["observations"] for x in first]).to(device)
    cond2=engine.normalize_observation(raw2[...,:42],stats).flatten(1)
    hint2=body_hint_from_raw(raw2)[:,None,:].expand(-1,16,-1)
    probe=torch.randn((2,16,7),device=device);probe_control=probe.clone()
    scheduler_a=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    scheduler_b=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    scheduler_a.set_timesteps(100);scheduler_b.set_timesteps(100)
    gen_a=torch.Generator(device=device).manual_seed(424243);gen_b=torch.Generator(device=device).manual_seed(424243)
    eps_errors=[]
    with torch.no_grad():
        for timestep in scheduler_a.timesteps:
            eps_a=base.noise_pred_net(probe,timestep,global_cond=cond2)
            eps_b=model(probe_control,timestep,global_cond=cond2,hint=hint2,enabled=False)
            eps_errors.append(float((eps_a-eps_b).abs().max()))
            probe=scheduler_a.step(eps_a,timestep,probe,generator=gen_a).prev_sample
            probe_control=scheduler_b.step(eps_b,timestep,probe_control,generator=gen_b).prev_sample
    torch.cuda.synchronize()
    zero_control_epsilon_max=max(eps_errors)
    zero_control_sample_max=float((probe-probe_control).abs().max())
    if max(zero_control_epsilon_max,zero_control_sample_max)>1e-6:
        raise RuntimeError("R43 disabled-Control DDPM-100 does not match independent R40 Base")
    model.train()
    index=torch.randperm(len(dataset))[:10*BATCH_SIZE].tolist()
    elapsed=[]; vals=[]
    for batch_i in range(10):
        ids=index[batch_i*BATCH_SIZE:(batch_i+1)*BATCH_SIZE]
        rows_batch=[dataset[i] for i in ids]
        raw=torch.stack([r["observations"] for r in rows_batch]).to(device)
        actions=torch.stack([r["actions"] for r in rows_batch]).to(device)
        start=time.perf_counter()
        shifted,_=augment_window_observations(raw.clone(),gauge)
        cond=engine.normalize_observation(shifted[...,:42],stats).flatten(1)
        hint=body_hint_from_raw(raw)[:,None,:].expand(-1,16,-1)
        noise=torch.randn_like(actions);t=torch.randint(0,100,(BATCH_SIZE,),device=device)
        noised=scheduler.add_noise(actions,noise,t)
        eps=model(noised,t,global_cond=cond,hint=hint)
        epsloss=torch.nn.functional.mse_loss(eps,noise)
        abar=scheduler.alphas_cumprod[t].reshape(-1,1,1)
        x0=((noised-(1-abar).sqrt()*eps)/abar.sqrt()).clamp(-1,1)
        window_rows=[windows[i] for i in ids]
        geo=[rows[1000+ep]["cylinder"] for ep,_,_ in window_rows]
        cylinders={"center":torch.tensor([x["center_xy"] for x in geo],device=device),
                    "radius":torch.tensor([x["radius"] for x in geo],device=device),
                    "height":torch.tensor([x["height"] for x in geo],device=device)}
        body,cube,counts=clearance_proxy_from_x0(x0,raw,raw,response,clouds,finger_models,cylinders,(t>=10)&(t<=30))
        loss=epsloss+PROXY_WEIGHT*body+PROXY_WEIGHT*cube
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
        vals.append({"epsilon_mse":float(epsloss.detach()),"body":float(body.detach()),"cube":float(cube.detach()),
                     "active":int(counts["active_rows"])})
        del raw,actions,shifted,cond,hint,noise,noised,eps,epsloss,abar,x0,body,cube,loss
    median=float(np.median(elapsed));p90=float(np.percentile(elapsed,90))
    payload={"status":"complete","benchmark_steps":10,"batch_size":BATCH_SIZE,
        "elapsed_seconds":elapsed,"median_seconds":median,"p90_seconds":p90,
        "projected_150k_seconds_median":median*TOTAL_UPDATES,
        "projected_150k_seconds_p90":p90*TOTAL_UPDATES,
        "budget_seconds":15000,"gpu_name":torch.cuda.get_device_name(0),
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(0),"values":vals,
        "disabled_control_epsilon_max_abs":zero_control_epsilon_max,
        "disabled_control_ddpm100_sample_max_abs":zero_control_sample_max,
        "checkpoint_written":False,"policy_training_started":False}
    path=BENCHMARK_PATH
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload


def _save_checkpoint(path, model, ema, optimizer, scheduler, step, rng, config):
    from examples.baselines.constrain.constrain_ckpt import save_constrain_checkpoint
    return save_constrain_checkpoint(path, model, ema, optimizer, scheduler, step, rng, config,
                                     checkpoint_format=TRAIN_FORMAT)


def _rng_state(gauge_generator, permutation, cursor):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
        "gauge": gauge_generator.get_state(), "sampler_permutation": permutation,
        "sampler_cursor": cursor}


def run_training(device=DEVICE):
    reg = _verify_training_registration()
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R43 training preregistration requires the verified CUDA device")
    r41._install_profile()
    import h5py
    from examples.baselines.constrain.pickcube_xy_gauge import METADATA as GAUGE_METADATA, augment_window_observations, make_generator
    from examples.baselines.constrain.pickcube_clearance_round43 import _sample_clouds
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG, body_hint_from_raw
    from examples.baselines.constrain.pickcube_clearance_proxy_r43 import clearance_hinge
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    from torch.utils.data import RandomSampler

    stats_path = R41_EXP / "position_encoding.json"
    stats = engine.load_stats(stats_path)
    dataset = engine.native_dataset("obstacle", include_obstacle=True)
    if len(dataset) != 45840 or len(dataset.windows) != 45840:
        raise ValueError("frozen obstacle window contract mismatch")
    manifest = json.loads(Path(engine.DATA_MANIFEST).read_text())
    rows = {int(row["episode_id"]): row for row in manifest["episodes"]}
    by_id = [rows[i] for i in range(1000,1500)]
    clouds, cloudmeta = _sample_clouds()
    report = json.loads(PRECHECK_REPORT.read_text())
    finger_models = report["native_fk_validation"]["finger_models"]
    response_model = json.loads(RESPONSE_FIT.read_text())["model"]
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    base_ck = torch.load(BASE, map_location="cpu", weights_only=False)
    base_agent = engine.build_agent("cpu"); base_agent.load_state_dict(base_ck["weights"]["ema"], strict=True)
    model = engine.build_constrain_denoiser(CONTROL_CONFIG, device="cpu")
    model.base_net.load_state_dict(base_agent.noise_pred_net.state_dict(), strict=True)
    model.to(device); model.train()
    frozen_sha = engine.state_sha(model.base_net.state_dict())
    params = model.trainable_parameters()
    if any(p.requires_grad for p in model.base_net.parameters()):
        raise RuntimeError("R43 matched R40 Base must remain frozen")
    optimizer = torch.optim.AdamW(params, lr=LR, betas=(0.95,0.999), weight_decay=1e-6)
    lr_scheduler = engine._make_schedule(TOTAL_UPDATES, WARMUP, optimizer)
    ema = EMAModel(parameters=params, power=EMA_POWER)
    diffusion = DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",
                              clip_sample=True,prediction_type="epsilon")
    diffusion.alphas_cumprod = diffusion.alphas_cumprod.to(device)
    gauge_generator = make_generator(SEED)
    permutation = torch.randperm(len(dataset)); cursor = 0
    RUN_DIR.mkdir(parents=True, exist_ok=False); CHECKPOINT_DIR.mkdir(parents=True, exist_ok=False)
    protocol = reg["objective"]
    config = {"format": TRAIN_FORMAT, "profile": engine.PROFILE,
        "model": CONTROL_CONFIG, "policy": engine.POLICY, "diffusion": dict(engine.DIFFUSION),
        "normalization": {"observation_profile": engine.PROFILE, "observation_stats": stats,
            "observation_stats_sha256": stats["sha256"], "action": "identity native seven-channel"},
        "typed_hint_metadata": r41.body_hint_metadata(), "training_augmentation": GAUGE_METADATA,
        "data_identity": {"selected_class":"obstacle","selected_episode_ids":list(range(1000,1500)),
            "episodes":500,"windows":45840,"h5_sha256":DATA["h5_sha256"],
            "manifest_sha256":DATA["manifest_sha256"]},
        "base_weights":{"path":str(BASE),"sha256":sha256_file(BASE),"subkey":"weights.ema",
            "updates":30000,"state_sha256":frozen_sha},
        "clearance_objective": protocol, "clearance_registration_sha256":reg["registration_sha256"],
        "precheck_sha256":reg["precheck_sha256"],"geometry_asset_hashes":reg["asset_hashes"],
        "response_fit_sha256":reg["response_fit_sha256"],"training":reg["training"],
        "selected_weights":"ema","outputs":{"run_dir":str(RUN_DIR),"checkpoint_dir":str(CHECKPOINT_DIR)}}
    # Replace the descriptive hint marker with exact typed metadata before any update.
    from examples.baselines.constrain.pickcube_body_hint import body_hint_metadata
    config["normalization"]["hint"] = body_hint_metadata()
    config["normalization"]["observation_profile"] = engine.PROFILE
    (RUN_DIR/"run_config.json").write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
    if config["clearance_objective"] != protocol or not params:
        raise ValueError("typed R43 model metadata failed roundtrip before training")
    preflight_dir=EXP/"preflight"
    preflight_dir.mkdir(exist_ok=True)
    fixture=preflight_dir/"typed_checkpoint_export_step000000.pt"
    if fixture.exists(): raise FileExistsError(fixture)
    _save_checkpoint(fixture,model,ema,optimizer,lr_scheduler,0,
        _rng_state(gauge_generator,permutation,cursor),config)
    fixture_payload=torch.load(fixture,map_location="cpu",weights_only=False)
    if (fixture_payload.get("format")!=TRAIN_FORMAT or fixture_payload.get("step")!=0
            or fixture_payload.get("config")!=config or fixture_payload.get("ema") is None
            or fixture_payload.get("optimizer") is None or fixture_payload.get("lr_scheduler") is None):
        raise RuntimeError("R43 typed checkpoint export/metadata roundtrip failed before training")
    reloaded=engine.build_constrain_denoiser(CONTROL_CONFIG,device="cpu")
    reloaded.load_state_dict(fixture_payload["model"],strict=True)
    restored_ema=EMAModel(parameters=reloaded.trainable_parameters(),power=EMA_POWER)
    restored_ema.load_state_dict(fixture_payload["ema"])
    checkpoint_preflight={"status":"passed","fixture_path":str(fixture),
        "fixture_sha256":sha256_file(fixture),"format":fixture_payload["format"],
        "step":fixture_payload["step"],"strict_model_load":True,
        "ema_roundtrip":True,"optimizer_state_present":True,"scheduler_state_present":True,
        "metadata_exact":fixture_payload["config"]==config,
        "updates_not_started":True}
    (preflight_dir/"typed_checkpoint_export.json").write_text(json.dumps(checkpoint_preflight,indent=2,sort_keys=True)+"\n")
    loader = Path(engine.DATA_H5)
    with h5py.File(loader,"r") as h5:
        # Dataset's preloaded windows map directly to IDs 1000..1499 by episode index.
        pass
    statslog=(RUN_DIR/"loss.jsonl").open("x")
    started=time.monotonic(); losses=[]
    for step in range(1,TOTAL_UPDATES+1):
        if cursor+BATCH_SIZE>len(permutation):
            permutation=torch.randperm(len(dataset)); cursor=0
        indices=permutation[cursor:cursor+BATCH_SIZE].tolist(); cursor+=BATCH_SIZE
        examples=[dataset[i] for i in indices]
        raw=torch.stack([v["observations"] for v in examples]).to(device)
        actions=torch.stack([v["actions"] for v in examples]).to(device)
        raw_gauge,_=augment_window_observations(raw.clone(),gauge_generator)
        global_cond=engine.normalize_observation(raw_gauge[...,:42],stats).flatten(1)
        hint=body_hint_from_raw(raw)[:,None,:].expand(-1,16,-1)
        noise=torch.randn_like(actions); t=torch.randint(0,100,(BATCH_SIZE,),device=device)
        noised=diffusion.add_noise(actions,noise,t)
        eps=model(noised,t,global_cond=global_cond,hint=hint)
        eps_loss=torch.nn.functional.mse_loss(eps,noise)
        selected=(t>=10)&(t<=30)
        alpha=diffusion.alphas_cumprod[t].reshape(-1,1,1)
        x0=((noised-(1-alpha).sqrt()*eps)/alpha.sqrt()).clamp(-1,1)
        geom_rows=[dataset.windows[i] for i in indices]
        centers=[]; radii=[]; heights=[]
        for episode,_,_ in geom_rows:
            row=by_id[episode]["cylinder"]
            centers.append(row["center_xy"]);radii.append(row["radius"]);heights.append(row["height"])
        cylinders={"center":torch.as_tensor(centers,dtype=raw.dtype,device=device),
            "radius":torch.as_tensor(radii,dtype=raw.dtype,device=device),
            "height":torch.as_tensor(heights,dtype=raw.dtype,device=device)}
        body_cost,cube_cost,proxy_stats=clearance_proxy_from_x0(x0,raw,raw, response_model,
            clouds,finger_models,cylinders,selected)
        proxy=body_cost+cube_cost
        weight=PROXY_WEIGHT*min(step/PROXY_WARMUP,1.0)
        loss=eps_loss+weight*proxy
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite R43 loss at step {step}")
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grads=[p.grad for p in params if p.grad is not None]
        gradients_finite=all(bool(torch.isfinite(g).all()) for g in grads)
        if not gradients_finite: raise FloatingPointError(f"nonfinite R43 gradient at step {step}")
        grad_norm=float(torch.sqrt(sum((g.detach().float().square().sum() for g in grads),torch.zeros((),device=device))))
        optimizer.step();lr_scheduler.step();ema.step(params)
        row={"step":step,"loss":float(loss.detach()),"epsilon_mse":float(eps_loss.detach()),
            "clearance_proxy":float(proxy.detach()),"body_clearance":float(body_cost.detach()),
            "cube_clearance":float(cube_cost.detach()),"clearance_weight":weight,
            "active_diffusion_rows":int(selected.sum()),
            "proxy_active_rows":int(proxy_stats["active_rows"]),
            "body_active_rows":int(proxy_stats["body_active_rows"]),
            "cube_active_rows":int(proxy_stats["cube_active_rows"]),
            "clearance_active_fraction":float(proxy_stats["active_rows"])/BATCH_SIZE,
            "action1_abs_gt_0_35_fraction":float((x0[:,1,:6].detach().abs()>.35).float().mean()),
            "gradients_finite":gradients_finite,"trainable_grad_norm":grad_norm,
            "body_min_gap_m":float(proxy_stats["body_min_gap_m"]),
            "cube_min_gap_m":float(proxy_stats["cube_min_gap_m"]),
            "tcp_translation_delta_rms_m":float(proxy_stats["tcp_translation_delta_rms_m"]),
            "lr":lr_scheduler.get_last_lr()[0]}
        statslog.write(json.dumps(row)+"\n"); losses.append(row)
        if step%100==0: statslog.flush()
        if step%10000==0 or step==TOTAL_UPDATES:
            statslog.flush()
            if engine.state_sha(model.base_net.state_dict())!=frozen_sha:
                raise RuntimeError(f"frozen R40 Base changed at update {step}")
            _save_checkpoint(CHECKPOINT_DIR/f"recovery_{step:06d}.pt",model,ema,optimizer,lr_scheduler,
                step,_rng_state(gauge_generator,permutation,cursor),config)
            print(json.dumps({"step":step,"loss":row["loss"],"epsilon_mse":row["epsilon_mse"],
                              "clearance_proxy":row["clearance_proxy"]}),flush=True)
    statslog.close()
    final=CHECKPOINT_DIR/"final.pt"
    _save_checkpoint(final,model,ema,optimizer,lr_scheduler,TOTAL_UPDATES,
        _rng_state(gauge_generator,permutation,cursor),config)
    if engine.state_sha(model.base_net.state_dict())!=frozen_sha:
        raise RuntimeError("R43 frozen R40 Base changed")
    result={"status":"complete","updates":TOTAL_UPDATES,"first_loss":losses[0],
        "last_loss":losses[-1],"all_losses_finite":all(np.isfinite(x["loss"]) and np.isfinite(x["epsilon_mse"]) and np.isfinite(x["clearance_proxy"]) for x in losses),
        "elapsed_seconds":time.monotonic()-started,"checkpoint":str(final),
        "checkpoint_sha256":sha256_file(final),"frozen_base_state_sha256":frozen_sha,
        "training_registration_sha256":reg["registration_sha256"],
        "precheck_sha256":reg["precheck_sha256"],"source_files_after":train_source_files()}
    (RUN_DIR/"train_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("benchmark","register-training","train"))
    parser.add_argument("--device",default=DEVICE)
    args=parser.parse_args()
    if args.command=="benchmark": result=benchmark_ten_real_batches(args.device)
    elif args.command=="register-training":
        benchmark=json.loads(BENCHMARK_PATH.read_text())
        result=register_training(benchmark)
    else:
        phase_path=EXP/"phase_runtime_control_training.json"
        if phase_path.exists(): raise FileExistsError("R43 control-training attempt already recorded")
        reg=_verify_training_registration()
        runtime={"phase":"single_r43_clearance_control_training","status":"running",
            "argv":sys.argv,"started_at_unix":time.time(),"budget_seconds":reg["training"]["budget_seconds"],
            "registration_sha256":reg["registration_sha256"],"precheck_sha256":reg["precheck_sha256"],
            "source_files":train_source_files(),"data_h5_sha256":sha256_file(engine.DATA_H5),
            "manifest_sha256":sha256_file(engine.DATA_MANIFEST),"base_sha256":sha256_file(BASE),
            "gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        phase_path.write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
        old_handler=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit(143)))
        started=time.monotonic()
        try:
            result=run_training(args.device)
            runtime.update(status="complete",training_result=result)
        except BaseException as exc:
            runtime.update(status="failed",error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            signal.signal(signal.SIGTERM,old_handler)
            runtime.update(elapsed_seconds=time.monotonic()-started,finished_at_unix=time.time(),
                source_files_after=train_source_files(),data_h5_sha256_after=sha256_file(engine.DATA_H5),
                manifest_sha256_after=sha256_file(engine.DATA_MANIFEST),base_sha256_after=sha256_file(BASE))
            phase_path.write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True,default=str))


def load_actor(checkpoint, stats_path, device="cpu"):
    """Strict typed actor loader. It never imports or calls a clearance cost."""
    admission = _verify_training_registration()
    r41._install_profile()
    stats=engine.load_stats(stats_path)
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    cfg=ck.get("config",{})
    required=(ck.get("format")==TRAIN_FORMAT and cfg.get("model")==r41.CONTROL_CONFIG
        and cfg.get("clearance_objective",{}).get("clearance_weight_final")==PROXY_WEIGHT
        and cfg.get("typed_hint_metadata")==r41.body_hint_metadata()
        and cfg.get("normalization",{}).get("observation_stats_sha256")==stats["sha256"]
        and cfg.get("data_identity",{}).get("h5_sha256")==DATA["h5_sha256"]
        and cfg.get("data_identity",{}).get("manifest_sha256")==DATA["manifest_sha256"]
        and cfg.get("base_weights",{}).get("sha256")==sha256_file(BASE)
        and cfg.get("selected_weights")=="ema" and ck.get("step")==TOTAL_UPDATES and ck.get("ema") is not None)
    if not required: raise ValueError("R43 typed clearance checkpoint contract mismatch")
    if (cfg.get("normalization",{}).get("observation_stats")!=stats
            or cfg.get("normalization",{}).get("action")!="identity native seven-channel"
            or cfg.get("training_augmentation")!=r41.r40.METADATA
            or cfg.get("clearance_registration_sha256") is None
            or cfg.get("precheck_sha256")!=sha256_file(PRECHECK_REPORT)
            or cfg.get("clearance_registration_sha256")!=admission["registration_sha256"]
            or cfg.get("training",{}).get("updates")!=TOTAL_UPDATES):
        raise ValueError("R43 clearance profile/data/training metadata mismatch")
    model=engine.build_constrain_denoiser(r41.CONTROL_CONFIG,device=device)
    model.load_state_dict(ck["model"],strict=True)
    from diffusers.training_utils import EMAModel
    ema=EMAModel(parameters=model.trainable_parameters(),power=EMA_POWER);ema.load_state_dict(ck["ema"])
    ema.copy_to(model.trainable_parameters());model.eval()
    return model,ck


class ClearanceActor:
    """Native action actor; geometry and clearance functions are absent at inference."""
    def __init__(self, checkpoint, stats_path, device="cuda"):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.device=torch.device(device)
        self.stats=engine.load_stats(stats_path)
        self.model,self.ck=load_actor(checkpoint,stats_path,device)
        self.scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",
            clip_sample=True,prediction_type="epsilon")
        self.scheduler.set_timesteps(100)
        self.metadata={"format":TRAIN_FORMAT,"profile":engine.PROFILE,
            "checkpoint_sha256":sha256_file(checkpoint),"weights":"ema",
            "hint_dim":9,"inference_clearance_cost_calls":0,"execution_horizon":8}

    def base_cond(self, raw_queued):
        raw=torch.as_tensor(raw_queued,dtype=torch.float32,device=self.device)
        if raw.ndim==2: raw=raw[None]
        if raw.shape[-2:]!=(2,45): raise ValueError("R43 actor requires native raw (B,2,45) history")
        return raw

    def get_action(self, raw_queued):
        raw=self.base_cond(raw_queued)
        global_cond=engine.normalize_observation(raw[...,:42],self.stats).flatten(1)
        hint=body_hint_from_raw(raw)[:,None,:].expand(-1,16,-1)
        sample=torch.randn((len(raw),16,7),device=self.device)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                eps=self.model(sample,timestep,global_cond=global_cond,hint=hint)
                sample=self.scheduler.step(eps,timestep,sample).prev_sample
        return sample[:,1:9]

    @staticmethod
    def map_action_ee_delta_pose(action):
        value=torch.as_tensor(action)
        if value.shape[-1]!=7: raise ValueError("R43 actor emits native seven-channel actions")
        return value


if __name__=="__main__":
    main()
