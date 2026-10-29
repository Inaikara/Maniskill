"""R44 typed horizon-8 training-only clearance objective and Control trainer."""

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from examples.baselines.constrain import pickcube_body_hint_round41 as r41
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_body_hint import body_hint_from_raw
from examples.baselines.constrain.pickcube_clearance_proxy_r43 import (
    GEOMETRY_LINKS, SAMPLE_SPHERE_PADDING_M, BODY_MARGIN_M, CUBE_XY_MARGIN_M,
    padded_cylinder_sdf,
)
from examples.baselines.constrain.pickcube_clearance_round43 import (
    DATA, EXP as R43_EXP, _sample_clouds, registered_assets, source_files as r43_sources,
)
from examples.baselines.constrain.pickcube_clearance_round44 import (
    EXP, CKPT, FIT, REG as PHASE_A_REG, MODEL_KIND, HORIZON, digest, quat_mul,
)

TRAIN_SCHEMA = "pickcube_se3_horizon8_training_admission_v1"
TRAIN_FORMAT = "pickcube_se3_horizon8_clearance_constrain_v1"
BASE = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt")
STATS_PATH = Path("outputs/exp/20261029_031500_pickcube_bodyhint_r41/position_encoding.json")
R40_EXP = Path("outputs/exp/20261029_004500_pickcube_xy_gauge_r40")
BENCHMARK = EXP / "train_benchmark_10_batches.json"
TRAIN_REG = EXP / "train_preregistration.json"
PRECHECK = EXP / "precheck/sourceclosed_08/response_precheck.json"
TOTAL = 150000
BATCH = 1024
SEED = 1
GAUGE_SEED = 424243
LR = 1e-4
WARMUP = 500
EMA_POWER = .75
LAMBDA = .01
LAMBDA_WARMUP = 10000
CONTROL_WALL_CAP_SECONDS = 15000
RUNTIME_DIR = EXP / "train_constrain"
CHECKPOINT_DIR = CKPT


def _torch_qmul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av,bv,dim=-1)), dim=-1)


def _torch_qnormalize(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)


def _torch_qexp(v):
    angle = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    half = angle * .5
    scale = torch.where(angle > 1e-8, torch.sin(half)/angle.clamp_min(1e-8), .5- angle.square()/48)
    return _torch_qnormalize(torch.cat((torch.cos(half), v*scale), dim=-1))


def _torch_qmatrix(q):
    w,x,y,z = _torch_qnormalize(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(q.shape[:-1]+(3,3))


def _native_rotation_deltas(planned8):
    """Panda's exact root-aligned XYZ Euler command map, including norm clipping."""
    native=planned8[...,3:6]
    norm=torch.linalg.vector_norm(native,dim=-1,keepdim=True)
    native=native/norm.clamp_min(1.0)
    x,y,z=(native*(-.1)).unbind(-1)  # Panda config rot_lower=-0.1 (negative scale)
    qx=torch.stack((torch.cos(x/2),torch.sin(x/2),torch.zeros_like(x),torch.zeros_like(x)),dim=-1)
    qy=torch.stack((torch.cos(y/2),torch.zeros_like(y),torch.sin(y/2),torch.zeros_like(y)),dim=-1)
    qz=torch.stack((torch.cos(z/2),torch.zeros_like(z),torch.zeros_like(z),torch.sin(z/2)),dim=-1)
    # ManiSkill's euler_angles_to_matrix(..., "XYZ") is Rx @ Ry @ Rz.
    return _torch_qnormalize(_torch_qmul(_torch_qmul(qx,qy),qz))


def _planned_world_rotation(prior_deltas):
    cumulative=[]; q=torch.zeros_like(prior_deltas[:,0]);q[:,0]=1
    for h in range(prior_deltas.shape[1]):
        q=_torch_qnormalize(_torch_qmul(prior_deltas[:,h],q));cumulative.append(q)
    return torch.stack(cumulative,dim=1)


def _response_features(previous, current, planned8):
    qprev, qcur = _torch_qnormalize(previous[:,22:26]), _torch_qnormalize(current[:,22:26])
    conjugate = torch.cat((qprev[:,:1],-qprev[:,1:]),dim=-1)
    relative = _torch_qnormalize(_torch_qmul(qcur,conjugate))
    relative = torch.where(relative[:,:1] < 0, -relative, relative)
    v = relative[:,1:]
    n = torch.linalg.vector_norm(v,dim=-1,keepdim=True)
    theta = 2*torch.atan2(n,relative[:,:1].clamp_min(0))
    log = v*torch.where(n>1e-8,theta/n.clamp_min(1e-8),torch.full_like(n,2.0))
    position=planned8[...,:3]*.1
    native_rot=planned8[...,3:6]
    native_norm=torch.linalg.vector_norm(native_rot,dim=-1,keepdim=True)
    physical_rotation=(native_rot/native_norm.clamp_min(1.0))*(-.1)
    physical=torch.cat((position,physical_rotation),dim=-1)
    return torch.cat((physical.flatten(1),current[:,:9],current[:,9:18],
        current[:,19:22]-previous[:,19:22],log),dim=-1)


def _predict_response(previous, current, planned8, model):
    feature=_response_features(previous,current,planned8)
    mean=torch.as_tensor(model["feature_mean"],device=feature.device,dtype=feature.dtype)
    scale=torch.as_tensor(model["feature_scale"],device=feature.device,dtype=feature.dtype)
    target=torch.as_tensor(model["target_mean"],device=feature.device,dtype=feature.dtype)
    coef=torch.as_tensor(model["coefficient"],device=feature.device,dtype=feature.dtype)
    return ((feature-mean)/scale)@coef+target


def _body_points_pose(raw, position, orientation, clouds, finger_models):
    rotation=_torch_qmatrix(orientation)
    chunks=[]
    fixed={
        "panda_hand":(np.asarray(__import__("examples.baselines.constrain.pickcube_body_hint",fromlist=["T_TCP_HAND"]).T_TCP_HAND)[:3,:3],
                      np.asarray(__import__("examples.baselines.constrain.pickcube_body_hint",fromlist=["T_TCP_HAND"]).T_TCP_HAND)[:3,3]),
        "panda_link7":(np.asarray(__import__("examples.baselines.constrain.pickcube_body_hint",fromlist=["T_TCP_LINK7"]).T_TCP_LINK7)[:3,:3],
                       np.asarray(__import__("examples.baselines.constrain.pickcube_body_hint",fromlist=["T_TCP_LINK7"]).T_TCP_LINK7)[:3,3]),
    }
    for name in GEOMETRY_LINKS:
        points=torch.as_tensor(clouds[name],device=raw.device,dtype=raw.dtype)
        if name in fixed:
            rel_r,rel_t=fixed[name]
            local=points@torch.as_tensor(rel_r,device=raw.device,dtype=raw.dtype).T+torch.as_tensor(rel_t,device=raw.device,dtype=raw.dtype)
        else:
            profile=finger_models[name]
            rel=torch.as_tensor(profile["T_tcp_finger_at_reference"],device=raw.device,dtype=raw.dtype)
            local=points@rel[:3,:3].T+rel[:3,3]
            qi=profile["qpos_index"]
            axis=torch.as_tensor(profile["axis_tcp_m_per_q"],device=raw.device,dtype=raw.dtype)
            local=local[None]+(raw[:,qi]-float(profile["q_reference_m"]))[:,None,None]*axis[None,None,:]
            chunks.append(position[:,None,:]+torch.einsum("bij,bnj->bni",rotation,local))
            continue
        chunks.append(position[:,None,:]+torch.einsum("bij,nj->bni",rotation,local))
    return torch.cat(chunks,dim=1)


def horizon8_clearance_components(x0, raw, response_model, clouds, finger_models,
                                  cylinders, active_mask):
    """Differentiable sum of eight body and grasped-cube clearance penalties."""
    selected=active_mask.bool() & (raw[:,1,18]>.5)
    if not bool(selected.any()):
        zero=x0.sum()*0
        return zero,zero,{"eligible_rows":selected.sum(),"body_active_fraction":zero.detach(),
                          "cube_active_fraction":zero.detach()}
    raw=raw[selected]; planned=x0[selected,1:9,:6]
    geom={k:v[selected] for k,v in cylinders.items()}
    current=raw[:,1]; previous=raw[:,0]
    response=_predict_response(previous,current,planned,response_model).reshape(-1,8,6)
    q0=_torch_qnormalize(current[:,22:26]); r0=_torch_qmatrix(q0)
    planned_rotation=_planned_world_rotation(_native_rotation_deltas(planned))
    cube_local=torch.einsum("bij,bj->bi",r0.transpose(1,2),current[:,29:32]-current[:,19:22])
    body_parts=[]; cube_parts=[]; body_active=[]; cube_active=[]
    for h in range(8):
        delta=response[:,h,:3]
        residual_q=_torch_qexp(response[:,h,3:])
        qh=_torch_qnormalize(_torch_qmul(_torch_qmul(planned_rotation[:,h],residual_q),q0))
        ph=current[:,19:22]+delta
        points=_body_points_pose(current,ph,qh,clouds,finger_models)
        sdf=padded_cylinder_sdf(points,geom["center"][:,None,:],geom["radius"][:,None],geom["height"][:,None])
        bh=F.relu((BODY_MARGIN_M-sdf)/BODY_MARGIN_M).square().mean(dim=1)
        cube=ph+torch.einsum("bij,bj->bi",_torch_qmatrix(qh),cube_local)
        radius=geom["radius"]+math.sqrt(2)*.02+.005
        gap=torch.linalg.vector_norm(cube[:,:2]-geom["center"],dim=-1)-radius
        ch=F.relu((CUBE_XY_MARGIN_M-gap)/CUBE_XY_MARGIN_M).square()
        body_parts.append(bh);cube_parts.append(ch)
        body_active.append((bh>0).float().mean());cube_active.append((ch>0).float().mean())
    body=torch.stack(body_parts,dim=1).sum(dim=1).mean()
    cube=torch.stack(cube_parts,dim=1).sum(dim=1).mean()
    return body,cube,{"eligible_rows":selected.sum(),"body_active_fraction":torch.stack(body_active).mean().detach(),
        "cube_active_fraction":torch.stack(cube_active).mean().detach(),
        "body_cost_by_horizon":torch.stack(body_parts,dim=1).mean(dim=0).detach(),
        "cube_cost_by_horizon":torch.stack(cube_parts,dim=1).mean(dim=0).detach()}


def eligible_window_mask(dataset):
    """Mask only real grasped windows with eight future actions and T+1 states."""
    masks=[]
    for episode,start,_end in dataset.windows:
        obs,actions=dataset.observations[episode],dataset.actions[episode]
        current=start+1
        masks.append(bool(start>=0 and current<len(actions) and current+8<len(obs)
                          and obs[current,18]>.5))
    return torch.tensor(masks,dtype=torch.bool)


def _source_files():
    names=("examples/baselines/constrain/pickcube_clearance_train_r44.py",
        "examples/baselines/constrain/pickcube_clearance_eval_r44.py",
        "examples/baselines/constrain/pickcube_clearance_round44.py",
        "examples/baselines/constrain/pickcube_clearance_round43.py",
        "examples/baselines/constrain/pickcube_clearance_proxy_r43.py",
        "examples/baselines/constrain/pickcube_clearance_train_r43.py",
        "examples/baselines/constrain/pickcube_clearance_eval_r43.py",
        "examples/baselines/constrain/pickcube_zscore_round37.py",
        "examples/baselines/constrain/pickcube_body_hint.py",
        "examples/baselines/constrain/pickcube_body_hint_round41.py",
        "examples/baselines/constrain/pickcube_xy_gauge.py",
        "examples/baselines/constrain/pickcube_xy_gauge_round40.py",
        "examples/baselines/constrain/materialize_pickcube_fullscene500.py",
        "mani_skill/agents/robots/panda/panda.py","mani_skill/agents/controllers/pd_ee_pose.py",
        "mani_skill/assets/robots/panda/panda_v2.urdf")
    return {n:sha256_file(n) for n in names}


def _phase_a_report():
    report=json.loads(PRECHECK.read_text())
    closure_path=EXP/"preregistration_sourceclosed_08.json"
    closure=json.loads(closure_path.read_text())
    from examples.baselines.constrain.pickcube_clearance_round44 import _source_closure
    closure_body={k:v for k,v in closure.items() if k!="closure_sha256"}
    if (report.get("status")!="passed" or report.get("model_kind")!=MODEL_KIND or report.get("horizon")!=8
            or not report.get("source_closure_sha256") or not report.get("source_files")
            or closure.get("closure_sha256")!=digest(closure_body)
            or closure.get("source_files")!=_source_closure()
            or report.get("source_closure_sha256")!=closure.get("closure_sha256")
            or report.get("source_files")!=closure.get("source_files")):
        raise ValueError("R44 exact horizon-8 response precheck did not pass")
    metrics=report["metrics"]
    if (metrics["overall_body_point_p95_m"]>.01
        or any(metrics["per_horizon"][str(h)]["body_point_p95_m"]>.01 for h in range(1,9))
        or metrics["overall_mean_improvement_fraction"]<.1):
        raise ValueError("R44 per-horizon response admission failed")
    return report


def registration_payload(benchmark):
    report=_phase_a_report()
    if benchmark.get("status")!="complete" or len(benchmark.get("batches",[]))!=10:
        raise ValueError("R44 exact-path ten-batch benchmark is incomplete")
    durations=np.asarray(benchmark.get("elapsed_seconds",[]),dtype=np.float64)
    if len(durations)!=10 or not np.isfinite(durations).all() or np.any(durations<=0):
        raise ValueError("R44 measured ten-batch durations are invalid")
    steady_p90=float(np.percentile(durations[1:],90))
    projected=float(durations[0]+steady_p90*(TOTAL-1))
    if projected>CONTROL_WALL_CAP_SECONDS:
        raise ValueError(f"R44 cold-start plus steady p90 projection {projected:.1f}s exceeds finite {CONTROL_WALL_CAP_SECONDS}s cap")
    if benchmark.get("disabled_base_epsilon_max_abs",float("inf"))>1e-6 or benchmark.get("disabled_base_sample_max_abs",float("inf"))>1e-6:
        raise ValueError("R44 disabled Control equivalence failed")
    return {"schema":TRAIN_SCHEMA,"model_kind":MODEL_KIND,"horizon":8,
        "precheck_sha256":sha256_file(PRECHECK),"fit_sha256":sha256_file(FIT),
        "phase_a_registration_sha256":report["registration_sha256"],
        "phase_a_source_closure":report["source_files"],"source_files":_source_files(),
        "data":DATA,"asset_hashes":registered_assets(),
        "base_checkpoint":{"path":str(BASE),"sha256":sha256_file(BASE),"state_source":"R40 final EMA; Control only"},
        "profile":{"hint_dim":9,"global_condition_dim":84,"state_dim":42,"raw_dim":45,
            "action_dim":7,"normalizer_sha256":sha256_file(STATS_PATH),"gauge_seed":GAUGE_SEED},
        "objective":{"epsilon_mse":1.0,"proxy_weight":.01,"warmup_steps":10000,
            "timesteps_inclusive":[10,30],"prediction":"scheduler x0 clipped [-1,1]",
            "planned_native_actions":"x0[:,1:9,:6]", "pose_points":8,
            "per_pose":"body normalized squared hinge + carried cube XY normalized squared hinge",
            "aggregation":"sum eight forecasts then mean eligible rows",
            "held_mask":"current raw grasped and eight recorded future actions/states available",
            "rotation_command_prior":"exact root-aligned XYZ Euler controller map, rot_lower=-0.1, native norm-limited; fitted ridge residual",
            "body_margin_m":BODY_MARGIN_M,"cube_xy_margin_m":CUBE_XY_MARGIN_M,
            "sample_padding_m":SAMPLE_SPHERE_PADDING_M,"continuous_sweep":False},
        "training":{"updates":TOTAL,"batch_size":BATCH,"seed":SEED,"lr":LR,"warmup":WARMUP,
            "schedule":"cosine","ema_power":EMA_POWER,"optimizer":"AdamW betas .95,.999 weight_decay 1e-6",
            "base_updates":0,"budget_seconds":CONTROL_WALL_CAP_SECONDS,
            "benchmark_projection":{"measured_batch_seconds":durations.tolist(),"cold_first_batch_seconds":float(durations[0]),
                "steady_state_batch_count":9,"steady_state_p90_seconds":steady_p90,
                "projected_total_seconds_including_one_cold_batch":projected,
                "method":"retain cold first batch once; project per-update p90 from nine steady-state batches"},
            "benchmark":benchmark},
        "selection":"final EMA fixed before dev; no dev selection","status":"frozen_before_training"}


def _audit_sources(reg):
    if reg.get("schema")!=TRAIN_SCHEMA or reg.get("model_kind")!=MODEL_KIND or reg.get("horizon")!=8:
        raise ValueError("wrong R44 typed training contract")
    if reg.get("source_files")!=_source_files(): raise ValueError("R44 training runtime source hash mismatch")
    if reg.get("data")!=DATA or reg.get("asset_hashes")!=registered_assets(): raise ValueError("R44 data/asset identity mismatch")
    if reg.get("fit_sha256")!=sha256_file(FIT) or reg.get("precheck_sha256")!=sha256_file(PRECHECK):
        raise ValueError("R44 frozen response/precheck changed")
    if reg.get("base_checkpoint",{}).get("sha256")!=sha256_file(BASE): raise ValueError("R40 Base identity mismatch")
    payload={k:v for k,v in reg.items() if k!="registration_sha256"}
    if reg.get("registration_sha256")!=digest(payload): raise ValueError("R44 phase-B registration checksum mismatch")
    return reg


def benchmark_ten_batches(device="cuda"):
    if device!="cuda" or not torch.cuda.is_available(): raise RuntimeError("R44 benchmark requires CUDA")
    report=_phase_a_report()
    r41._install_profile()
    from examples.baselines.constrain.pickcube_xy_gauge import make_generator,augment_window_observations
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    stats=engine.load_stats(STATS_PATH); dataset=engine.native_dataset("obstacle",include_obstacle=True)
    if len(dataset)!=45840: raise ValueError("R44 requires all existing 45,840 obstacle training windows")
    eligible=eligible_window_mask(dataset); rows={int(r["episode_id"]):r for r in json.loads(Path(engine.DATA_MANIFEST).read_text())["episodes"]}
    clouds,_=_sample_clouds(); fingers=report["native_fk_validation"]["finger_models"]
    response=json.loads(FIT.read_text()); baseck=torch.load(BASE,map_location="cpu",weights_only=False)
    base=engine.build_agent("cpu");base.load_state_dict(baseck["weights"]["ema"],strict=True);base.to(device).eval()
    model=engine.build_constrain_denoiser(CONTROL_CONFIG,device="cpu");model.base_net.load_state_dict(base.noise_pred_net.state_dict(),strict=True);model.to(device).train()
    diff=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon");diff.alphas_cumprod=diff.alphas_cumprod.to(device)
    # Explicit independently seeded disabled-Control/Base sampler equivalence.
    examples=[dataset[i] for i in (0,1)];raw2=torch.stack([e["observations"] for e in examples]).to(device)
    cond=engine.normalize_observation(raw2[...,:42],stats).flatten(1);hint=body_hint_from_raw(raw2)[:,None].expand(-1,16,-1)
    aa=engine.build_agent("cpu");aa.load_state_dict(baseck["weights"]["ema"],strict=True);aa.to(device).eval()
    sam=torch.randn((2,16,7),device=device);sam2=sam.clone(); sch=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon");sch.set_timesteps(100);sch.alphas_cumprod=sch.alphas_cumprod.to(device);diff.set_timesteps(100)
    gen_a=torch.Generator(device=device).manual_seed(424243);gen_b=torch.Generator(device=device).manual_seed(424243)
    with torch.no_grad():
        for t in diff.timesteps:
            e1=aa.noise_pred_net(sam,t,global_cond=cond);e2=model(sam2,t,global_cond=cond,hint=hint,enabled=False)
            sam=diff.step(e1,t,sam,generator=gen_a).prev_sample;sam2=sch.step(e2,t,sam2,generator=gen_b).prev_sample
    epsmatch=float((e1-e2).abs().max()); samplematch=float((sam-sam2).abs().max())
    if max(epsmatch,samplematch)>1e-6: raise ValueError(f"R44 disabled Control != exact R40 Base: epsilon={epsmatch}, sample={samplematch}")
    from examples.baselines.constrain.pickcube_xy_gauge import augment_window_observations,make_generator
    gauge=make_generator(SEED);indices=torch.randperm(len(dataset))[:10*BATCH].tolist();durations=[];batch_logs=[]
    opt=torch.optim.AdamW(model.trainable_parameters(),lr=LR,betas=(.95,.999),weight_decay=1e-6)
    for bi in range(10):
        ids=indices[bi*BATCH:(bi+1)*BATCH]; examples=[dataset[i] for i in ids]
        raw=torch.stack([e["observations"] for e in examples]).to(device); actions=torch.stack([e["actions"] for e in examples]).to(device)
        aug,_=augment_window_observations(raw.clone(),gauge); gcond=engine.normalize_observation(aug[...,:42],stats).flatten(1); hints=body_hint_from_raw(raw)[:,None].expand(-1,16,-1)
        noise=torch.randn_like(actions); t=torch.randint(0,100,(BATCH,),device=device); noised=diff.add_noise(actions,noise,t)
        begin=time.perf_counter();pred=model(noised,t,global_cond=gcond,hint=hints);eps=F.mse_loss(pred,noise)
        abar=diff.alphas_cumprod[t].reshape(-1,1,1);x0=((noised-(1-abar).sqrt()*pred)/abar.sqrt()).clamp(-1,1)
        window_meta=[dataset.windows[i] for i in ids]; cylinders={k:torch.tensor([rows[1000+ep]["cylinder"][key] for ep,_,_ in window_meta],device=device,dtype=raw.dtype) for k,key in (("center","center_xy"),("radius","radius"),("height","height"))}
        mask=torch.tensor([bool(eligible[i]) for i in ids],device=device)&(t>=10)&(t<=30)
        body,cube,counts=horizon8_clearance_components(x0,raw,response,clouds,fingers,cylinders,mask)
        audit=_proxy_gradient_audit(model,model.trainable_parameters(),eps,body,cube,x0) if bi==0 else None
        if audit is not None and (audit["planned_xyz_gradient_norm"]<=0 or audit["planned_rotation_gradient_norm"]<=0
                or audit["planned_gripper_gradient_norm"]!=0):
            raise RuntimeError(f"R44 ten-batch benchmark real-minibatch gradient audit failed: {audit}")
        loss=eps+.01*(body+cube);opt.zero_grad(set_to_none=True);loss.backward();opt.step();torch.cuda.synchronize()
        if not all(np.isfinite([float(eps.detach()),float(body.detach()),float(cube.detach()),float(loss.detach())])):
            raise FloatingPointError(f"R44 benchmark batch {bi} produced nonfinite objective components")
        durations.append(time.perf_counter()-begin)
        batch_logs.append({"epsilon":float(eps.detach()),"body":float(body.detach()),"cube":float(cube.detach()),"eligible_low_t_rows":int(counts["eligible_rows"]),"all_finite":bool(torch.isfinite(loss)),"gradient_audit":audit})
    med=float(np.median(durations));p90=float(np.percentile(durations,90))
    payload={"status":"complete","batches":batch_logs,"elapsed_seconds":durations,"median_seconds":med,"p90_seconds":p90,
        "projected_150k_seconds_median":med*TOTAL,"projected_150k_seconds_p90":p90*TOTAL,"budget_seconds":18000,
        "gpu_name":torch.cuda.get_device_name(0),"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "disabled_base_epsilon_max_abs":epsmatch,"disabled_base_sample_max_abs":samplematch,
        "weights_written":False,"production_training_started":False}
    if BENCHMARK.exists(): raise FileExistsError(BENCHMARK)
    BENCHMARK.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload


def register_training():
    benchmark=json.loads(BENCHMARK.read_text()); body=registration_payload(benchmark)
    if TRAIN_REG.exists(): raise FileExistsError(TRAIN_REG)
    body["registration_sha256"]=digest(body);TRAIN_REG.write_text(json.dumps(body,indent=2,sort_keys=True)+"\n")
    return body


def _proxy_gradient_audit(model,params,eps,body,cube,x0):
    grads={}
    for name,component in (("epsilon",eps),("body",body),("cube",cube)):
        values=torch.autograd.grad(component,params,retain_graph=True,allow_unused=True)
        grads[name]=float(torch.sqrt(sum((g.detach().float().square().sum() for g in values if g is not None),torch.zeros((),device=x0.device))))
    xgrad=torch.autograd.grad(body+cube,x0,retain_graph=True)[0]
    return {"component_parameter_gradient_norms":grads,
        "planned_xyz_gradient_norm":float(xgrad[:,1:9,:3].norm()),
        "planned_rotation_gradient_norm":float(xgrad[:,1:9,3:6].norm()),
        "planned_gripper_gradient_norm":float(xgrad[:,1:9,6].norm())}


def run_training(device="cuda",resume=None):
    reg=_audit_sources(json.loads(TRAIN_REG.read_text()))
    if not torch.cuda.is_available() or device!="cuda": raise RuntimeError("R44 Control requires CUDA")
    r41._install_profile()
    from examples.baselines.constrain.pickcube_xy_gauge import METADATA as GAUGE,make_generator,augment_window_observations
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    from examples.baselines.constrain.constrain_ckpt import save_constrain_checkpoint
    stats=engine.load_stats(STATS_PATH);dataset=engine.native_dataset("obstacle",include_obstacle=True)
    eligible=eligible_window_mask(dataset);rows={int(r["episode_id"]):r for r in json.loads(Path(engine.DATA_MANIFEST).read_text())["episodes"]}
    clouds,cloudmeta=_sample_clouds();fingers=json.loads(PRECHECK.read_text())["native_fk_validation"]["finger_models"];response=json.loads(FIT.read_text())
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);np.random.seed(SEED);random.seed(SEED)
    baseck=torch.load(BASE,map_location="cpu",weights_only=False);base=engine.build_agent("cpu");base.load_state_dict(baseck["weights"]["ema"],strict=True)
    model=engine.build_constrain_denoiser(CONTROL_CONFIG,device="cpu");model.base_net.load_state_dict(base.noise_pred_net.state_dict(),strict=True);model.to(device).train()
    params=model.trainable_parameters(); frozen=engine.state_sha(model.base_net.state_dict())
    opt=torch.optim.AdamW(params,lr=LR,betas=(.95,.999),weight_decay=1e-6);lr_sched=engine._make_schedule(TOTAL,WARMUP,opt)
    ema=EMAModel(parameters=params,power=EMA_POWER);diff=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon");diff.alphas_cumprod=diff.alphas_cumprod.to(device)
    gauge=make_generator(SEED);permutation=torch.randperm(len(dataset));cursor=0;start_step=0
    config={"format":TRAIN_FORMAT,"model_kind":MODEL_KIND,"horizon":8,"profile":engine.PROFILE,"model":CONTROL_CONFIG,
        "policy":engine.POLICY,"diffusion":dict(engine.DIFFUSION),"normalization":{"observation_stats":stats,"observation_stats_sha256":stats["sha256"],"action":"identity native seven-channel"},
        "typed_hint_metadata":r41.body_hint_metadata(),"training_augmentation":GAUGE,"data_identity":{"class":"obstacle","episode_ids":list(range(1000,1500)),"windows":45840,"h5_sha256":DATA["h5_sha256"],"manifest_sha256":DATA["manifest_sha256"]},
        "base_weights":{"path":str(BASE),"sha256":sha256_file(BASE),"frozen_state_sha256":frozen},"response_fit_sha256":sha256_file(FIT),
        "precheck_sha256":sha256_file(PRECHECK),"training_registration_sha256":reg["registration_sha256"],"clearance_objective":reg["objective"],"selected_weights":"ema"}
    from examples.baselines.constrain.pickcube_body_hint import body_hint_metadata
    config["normalization"]["hint"]=body_hint_metadata()
    runconfig=RUNTIME_DIR/"run_config.json"
    if resume is None:
        CHECKPOINT_DIR.mkdir(parents=True,exist_ok=False);RUNTIME_DIR.mkdir(parents=True,exist_ok=False)
        runconfig.write_text(json.dumps(config,indent=2,sort_keys=True)+"\n")
        # Strict step-zero export/load validates typed metadata before optimizer updates.
        fixture=RUNTIME_DIR/"typed_checkpoint_step000000.pt"
        save_constrain_checkpoint(fixture,model,ema,opt,lr_sched,0,{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"gauge":gauge.get_state(),"permutation":permutation,"cursor":cursor,"cumulative_wall_seconds":0.0},config,checkpoint_format=TRAIN_FORMAT)
        loaded=torch.load(fixture,map_location="cpu",weights_only=False)
        if loaded["format"]!=TRAIN_FORMAT or loaded["config"]!=config or loaded["step"]!=0: raise RuntimeError("R44 typed checkpoint export roundtrip failed")
        log_mode="x"
    else:
        checkpoint=torch.load(resume,map_location="cpu",weights_only=False)
        if (checkpoint.get("format")!=TRAIN_FORMAT or checkpoint.get("config")!=config
                or checkpoint.get("step",0)>=TOTAL or checkpoint.get("optimizer") is None
                or checkpoint.get("lr_scheduler") is None or checkpoint.get("ema") is None):
            raise ValueError("R44 resume checkpoint contract/state is incomplete")
        model.load_state_dict(checkpoint["model"],strict=True);ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["optimizer"]);lr_sched.load_state_dict(checkpoint["lr_scheduler"])
        rng=checkpoint.get("rng",{});elapsed_before=float(rng.get("cumulative_wall_seconds",0.0));random.setstate(rng["python"]);np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"]);torch.cuda.set_rng_state_all(rng["cuda"]);gauge.set_state(rng["gauge"])
        permutation=rng["permutation"];cursor=int(rng["cursor"]);start_step=int(checkpoint["step"])
        if json.loads(runconfig.read_text())!=config: raise ValueError("R44 run config changed across resume")
        log_mode="a"
    if resume is None: elapsed_before=0.0
    log=(RUNTIME_DIR/"loss.jsonl").open(log_mode);started=time.monotonic();last_final_written=False
    try:
        for step in range(start_step+1,TOTAL+1):
            if elapsed_before+time.monotonic()-started>CONTROL_WALL_CAP_SECONDS:
                raise TimeoutError(f"R44 registered finite {CONTROL_WALL_CAP_SECONDS}-second Control training cap reached")
            if cursor+BATCH>len(permutation): permutation=torch.randperm(len(dataset));cursor=0
            ids=permutation[cursor:cursor+BATCH].tolist();cursor+=BATCH;examples=[dataset[i] for i in ids]
            raw=torch.stack([e["observations"] for e in examples]).to(device);actions=torch.stack([e["actions"] for e in examples]).to(device)
            aug,_=augment_window_observations(raw.clone(),gauge);gcond=engine.normalize_observation(aug[...,:42],stats).flatten(1);hint=body_hint_from_raw(raw)[:,None].expand(-1,16,-1)
            noise=torch.randn_like(actions);t=torch.randint(0,100,(BATCH,),device=device);noised=diff.add_noise(actions,noise,t);pred=model(noised,t,global_cond=gcond,hint=hint);eps=F.mse_loss(pred,noise)
            alpha=diff.alphas_cumprod[t].reshape(-1,1,1);x0=((noised-(1-alpha).sqrt()*pred)/alpha.sqrt()).clamp(-1,1)
            metas=[dataset.windows[i] for i in ids];cyl={k:torch.tensor([rows[1000+ep]["cylinder"][key] for ep,_,_ in metas],device=device,dtype=raw.dtype) for k,key in (("center","center_xy"),("radius","radius"),("height","height"))}
            mask=torch.tensor([bool(eligible[i]) for i in ids],device=device)&(t>=10)&(t<=30)
            body,cube,counts=horizon8_clearance_components(x0,raw,response,clouds,fingers,cyl,mask)
            # One real minibatch gradient audit before the first optimizer update.
            grad_audit=_proxy_gradient_audit(model,params,eps,body,cube,x0) if step==1 else None
            if grad_audit is not None and (grad_audit["planned_xyz_gradient_norm"]<=0
                    or grad_audit["planned_rotation_gradient_norm"]<=0
                    or grad_audit["planned_gripper_gradient_norm"]!=0
                    or not all(np.isfinite(list(grad_audit["component_parameter_gradient_norms"].values())))):
                raise RuntimeError(f"R44 real minibatch proxy gradient audit failed: {grad_audit}")
            weight=LAMBDA*min(step/LAMBDA_WARMUP,1.0);loss=eps+weight*(body+cube)
            if not torch.isfinite(loss): raise FloatingPointError(f"R44 nonfinite loss step {step}")
            opt.zero_grad(set_to_none=True);loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params): raise FloatingPointError(f"R44 nonfinite gradient step {step}")
            gradnorm=float(torch.sqrt(sum((p.grad.detach().float().square().sum() for p in params if p.grad is not None),torch.zeros((),device=device))))
            opt.step();lr_sched.step();ema.step(params)
            row={"step":step,"loss":float(loss.detach()),"epsilon_mse":float(eps.detach()),"body_clearance":float(body.detach()),"cube_clearance":float(cube.detach()),
                "lambda_eff":weight,"eligible_low_t_rows":int(counts["eligible_rows"]),"body_active_fraction":float(counts["body_active_fraction"]),
                "cube_active_fraction":float(counts["cube_active_fraction"]),"x0_action_abs_gt_0_35_fraction":float((x0[:,1:9,:6].detach().abs()>.35).float().mean()),
                "grad_norm":gradnorm,"gradient_audit":grad_audit,"lr":lr_sched.get_last_lr()[0]}
            log.write(json.dumps(row)+"\n")
            if step%100==0: log.flush()
            if step%10000==0 or step==TOTAL:
                log.flush()
                if engine.state_sha(model.base_net.state_dict())!=frozen: raise RuntimeError("R40 frozen Base changed")
                save_constrain_checkpoint(CHECKPOINT_DIR/f"recovery_{step:06d}.pt",model,ema,opt,lr_sched,step,
                    {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"gauge":gauge.get_state(),"permutation":permutation,"cursor":cursor,"cumulative_wall_seconds":elapsed_before+time.monotonic()-started},config,checkpoint_format=TRAIN_FORMAT)
                if step==TOTAL:
                    # Persist final EMA immediately before any post-training audit.
                    save_constrain_checkpoint(CHECKPOINT_DIR/"final.pt",model,ema,opt,lr_sched,step,
                        {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"gauge":gauge.get_state(),"permutation":permutation,"cursor":cursor,"cumulative_wall_seconds":elapsed_before+time.monotonic()-started},config,checkpoint_format=TRAIN_FORMAT)
                    last_final_written=True
        log.close()
    except BaseException:
        log.flush();log.close()
        raise
    summary={"status":"complete","updates":TOTAL,"elapsed_seconds":time.monotonic()-started,"checkpoint":str(CHECKPOINT_DIR/"final.pt"),
        "checkpoint_sha256":sha256_file(CHECKPOINT_DIR/"final.pt"),"frozen_base_state_sha256":frozen,"registration_sha256":reg["registration_sha256"],
        "precheck_sha256":sha256_file(PRECHECK),"response_fit_sha256":sha256_file(FIT),"source_files":_source_files(),"final_saved_before_postaudit":last_final_written}
    (RUNTIME_DIR/"train_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    return summary


def load_actor(checkpoint,stats_path=STATS_PATH,device="cpu"):
    """Typed horizon-8 EMA loader; actor path does not import or call proxy costs."""
    reg=_audit_sources(json.loads(TRAIN_REG.read_text()))
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False)
    cfg=ck.get("config",{})
    if (ck.get("format")!=TRAIN_FORMAT or cfg.get("format")!=TRAIN_FORMAT
        or cfg.get("model_kind")!=MODEL_KIND or cfg.get("horizon")!=8
        or ck.get("step")!=TOTAL or ck.get("ema") is None
        or cfg.get("training_registration_sha256")!=reg["registration_sha256"]
        or cfg.get("precheck_sha256")!=sha256_file(PRECHECK)
        or cfg.get("response_fit_sha256")!=sha256_file(FIT)
        or cfg.get("base_weights",{}).get("sha256")!=sha256_file(BASE)
        or cfg.get("normalization",{}).get("observation_stats_sha256")!=sha256_file(stats_path)):
        raise ValueError("R44 checkpoint rejected: typed horizon-8 metadata mismatch")
    stats=engine.load_stats(stats_path)
    if cfg.get("normalization",{}).get("observation_stats")!=stats or cfg.get("typed_hint_metadata")!=r41.body_hint_metadata():
        raise ValueError("R44 actor observation/hint profile mismatch")
    r41._install_profile()
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG
    from diffusers.training_utils import EMAModel
    model=engine.build_constrain_denoiser(CONTROL_CONFIG,device="cpu")
    model.load_state_dict(ck["model"],strict=True)
    ema=EMAModel(parameters=model.trainable_parameters(),power=EMA_POWER);ema.load_state_dict(ck["ema"]);ema.copy_to(model.trainable_parameters())
    model.to(device).eval()
    return model,ck,stats


class Horizon8ClearanceActor:
    """Unmodified native DDPM actor; no clearance cost/projection in inference."""
    def __init__(self,checkpoint,stats_path=STATS_PATH,device="cuda"):
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        self.device=torch.device(device);self.model,self.ck,self.stats=load_actor(checkpoint,stats_path,device)
        self.scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
        self.scheduler.set_timesteps(100)
        self.metadata={"format":TRAIN_FORMAT,"model_kind":MODEL_KIND,"horizon":8,"checkpoint_sha256":sha256_file(checkpoint),
            "weights":"ema","hint_dim":9,"condition_dim":84,"inference_clearance_cost_calls":0,"execution_horizon":8}

    def base_cond(self,raw):
        raw=torch.as_tensor(raw,dtype=torch.float32,device=self.device)
        if raw.ndim==2: raw=raw[None]
        if raw.shape[-2:]!=(2,45): raise ValueError("R44 actor requires raw native (B,2,45) history")
        return raw

    def get_action(self,raw):
        raw=self.base_cond(raw);gcond=engine.normalize_observation(raw[...,:42],self.stats).flatten(1)
        hint=body_hint_from_raw(raw)[:,None].expand(-1,16,-1);sample=torch.randn((len(raw),16,7),device=self.device)
        with torch.no_grad():
            for timestep in self.scheduler.timesteps:
                eps=self.model(sample,timestep,global_cond=gcond,hint=hint)
                sample=self.scheduler.step(eps,timestep,sample).prev_sample
        return sample[:,1:9]

    @staticmethod
    def map_action_ee_delta_pose(action):
        value=torch.as_tensor(action)
        if value.shape[-1]!=7: raise ValueError("R44 actor outputs native seven-channel actions")
        return value
