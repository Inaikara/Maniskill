"""Offline R44 evidence assembly; intentionally separate from runtime actor sources."""

import json
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_clearance_round44 import EXP, FIT, DATA
from examples.baselines.constrain.pickcube_clearance_train_r44 import (
    BASE, CKPT, PRECHECK, STATS_PATH, TRAIN_REG, TRAIN_FORMAT,
    eligible_window_mask, horizon8_clearance_components, _sample_clouds,
    _proxy_gradient_audit, _source_files,
)

PAIR=EXP/"paired_native_r44"
R41=Path("outputs/exp/20261029_031500_pickcube_bodyhint_r41/paired")
R43=Path("outputs/exp/20261029_052324_pickcube_clearance_r43/paired_native_r43")
FINAL=EXP/"native_analysis_final.json"
INPUTS=EXP/"native_analysis_inputs_r44.json"
AUDIT=EXP/"final_ema_gradient_audit_02.json"
FINAL_V2=EXP/"native_analysis_final_02.json"


def _rows(root,policy):
    return [json.loads((root/policy/f"episode_{i:03d}.json").read_text()) for i in range(30)]


def _transitions(old,new,field):
    a=[bool(r.get("result",{}).get(field,False)) for r in old]
    b=[bool(r.get("result",{}).get(field,False)) for r in new]
    positive=field in ("native_success","safe_success")
    counts={"both":sum(x and y for x,y in zip(a,b)),"neither":sum(not x and not y for x,y in zip(a,b)),
        "old_marginal":sum(a),"new_marginal":sum(b)}
    if positive:
        counts.update(improved_new_only=sum(not x and y for x,y in zip(a,b)),regressed_old_only=sum(x and not y for x,y in zip(a,b)))
    else:
        counts.update(cleared_old_only=sum(x and not y for x,y in zip(a,b)),introduced_new_only=sum(not x and y for x,y in zip(a,b)))
    return counts


def final_ema_gradient_audit(device="cuda"):
    """A fresh deterministic real minibatch component-gradient audit at tau=20."""
    from examples.baselines.constrain.pickcube_clearance_actor_r44 import load_actor
    from examples.baselines.constrain import pickcube_body_hint_round41 as r41
    from examples.baselines.constrain.pickcube_clearance_round43 import _sample_clouds
    from examples.baselines.constrain.pickcube_body_hint_round41 import body_hint_from_raw
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    r41._install_profile()
    model,ck,stats=load_actor(CKPT/"final.pt",STATS_PATH,device)
    model.train()
    dataset=engine.native_dataset("obstacle",include_obstacle=True)
    eligible=eligible_window_mask(dataset)
    candidates=torch.where(eligible)[0]
    generator=torch.Generator(device="cpu").manual_seed(1)
    selected=candidates[torch.randperm(len(candidates),generator=generator)[:1024]].tolist()
    samples=[dataset[int(i)] for i in selected]
    raw=torch.stack([s["observations"] for s in samples]).to(device)
    actions=torch.stack([s["actions"] for s in samples]).to(device)
    geom_rows=[dataset.windows[int(i)] for i in selected]
    manifest={int(r["episode_id"]):r for r in json.loads(Path(engine.DATA_MANIFEST).read_text())["episodes"]}
    cylinders={k:torch.tensor([manifest[1000+ep]["cylinder"][key] for ep,_,_ in geom_rows],device=device,dtype=raw.dtype)
        for k,key in (("center","center_xy"),("radius","radius"),("height","height"))}
    clouds,_=_sample_clouds();finger=json.loads(PRECHECK.read_text())["native_fk_validation"]["finger_models"]
    torch.manual_seed(1);torch.cuda.manual_seed_all(1)
    diffusion=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon")
    diffusion.alphas_cumprod=diffusion.alphas_cumprod.to(device)
    noise=torch.randn_like(actions);t=torch.full((len(raw),),20,device=device,dtype=torch.long)
    noised=diffusion.add_noise(actions,noise,t)
    gcond=engine.normalize_observation(raw[...,:42],stats).flatten(1)
    hint=body_hint_from_raw(raw)[:,None].expand(-1,16,-1)
    prediction=model(noised,t,global_cond=gcond,hint=hint)
    epsilon=torch.nn.functional.mse_loss(prediction,noise)
    alpha=diffusion.alphas_cumprod[t].reshape(-1,1,1)
    x0=((noised-(1-alpha).sqrt()*prediction)/alpha.sqrt()).clamp(-1,1)
    body,cube,counts=horizon8_clearance_components(x0,raw,json.loads(FIT.read_text()),clouds,finger,cylinders,torch.ones(len(raw),device=device,dtype=torch.bool))
    audit=_proxy_gradient_audit(model,model.trainable_parameters(),epsilon,body,cube,x0)
    actual_actions=torch.cat(dataset.actions,dim=0).numpy().astype(np.float64)
    predicted_actions=x0[:,1:9,:].detach().cpu().numpy().astype(np.float64).reshape(-1,7)
    action_stats=lambda a:{"min_by_native_channel":a.min(axis=0).tolist(),"median_by_native_channel":np.median(a,axis=0).tolist(),
        "mean_by_native_channel":a.mean(axis=0).tolist(),"max_by_native_channel":a.max(axis=0).tolist(),
        "fraction_abs_gt_0_35_by_channel":(np.abs(a)>.35).mean(axis=0).tolist()}
    result={"status":"passed" if audit["planned_xyz_gradient_norm"]>0 and audit["planned_rotation_gradient_norm"]>0 and audit["planned_gripper_gradient_norm"]==0 else "failed",
        "phase":"post_training_final_ema","model_kind":"SE3_horizon8_clearance","checkpoint_sha256":sha256_file(CKPT/"final.pt"),
        "training_registration_sha256":json.loads(TRAIN_REG.read_text())["registration_sha256"],"diffusion_timestep":20,
        "selected_real_eligible_windows":len(selected),"source_episode_classes":"frozen obstacle IDs 1000-1499 only",
        "component_losses":{"epsilon":float(epsilon.detach()),"body":float(body.detach()),"cube":float(cube.detach())},
        "gradient_audit":audit,"proxy_stats":{"eligible_rows":int(counts["eligible_rows"]),
            "body_active_fraction":float(counts["body_active_fraction"]),"cube_active_fraction":float(counts["cube_active_fraction"]),
            "body_cost_by_horizon":counts["body_cost_by_horizon"].cpu().tolist(),
            "cube_cost_by_horizon":counts["cube_cost_by_horizon"].cpu().tolist()},
        "no_optimizer_update":True,"actor_inference_cost_calls":0}
    result["action_support_calibration"]={"frozen_obstacle_native_action_labels":action_stats(actual_actions),
        "final_ema_predicted_x0_planned_actions":action_stats(predicted_actions),
        "interpretation_threshold_abs_native_command":.35}
    AUDIT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    if result["status"]!="passed": raise RuntimeError(f"post-training EMA gradient audit failed: {audit}")
    return result


def analyze():
    if not (EXP/"native_trial_preregistration.json").is_file(): raise FileNotFoundError("R44 native preregistration missing")
    trial=json.loads((EXP/"native_trial_preregistration.json").read_text())
    trial_body={k:v for k,v in trial.items() if k!="trial_registration_sha256"}
    from examples.baselines.constrain.pickcube_clearance_round44 import digest
    if trial.get("trial_registration_sha256")!=digest(trial_body): raise ValueError("R44 trial preregistration checksum mismatch")
    if trial.get("source_files")!={k:sha256_file(k) for k in trial["source_files"]}: raise ValueError("R44 evaluation source identity changed")
    if sha256_file(CKPT/"final.pt")!=trial["checkpoint_sha256"]: raise ValueError("R44 final EMA hash differs from trial registration")
    rows={p:_rows(PAIR,p) for p in ("base_dp","constrain_dp")}
    if any(r.get("status")!="evaluated" or r.get("scene_index")!=i for p in rows for i,r in enumerate(rows[p])):
        raise ValueError("R44 fixed 30-row denominator incomplete/order mismatch")
    r41={p:_rows(R41,p) for p in rows};r43={p:_rows(R43,p) for p in rows}
    base_exact=all(a.get("step_trace")==b.get("step_trace") and a.get("result")==b.get("result") for a,b in zip(rows["base_dp"],r43["base_dp"]))
    if not base_exact: raise ValueError("R44 R40 Base traces differ from registered R43/R40 Base")
    fields=("native_success","safe_success","obstacle_contact","xy_forbidden_crossing")
    transitions={"R40_base_to_R44_control":{f:_transitions(rows["base_dp"],rows["constrain_dp"],f) for f in fields},
        "R43_control_to_R44_control":{f:_transitions(r43["constrain_dp"],rows["constrain_dp"],f) for f in fields},
        "R41_control_to_R44_control":{f:_transitions(r41["constrain_dp"],rows["constrain_dp"],f) for f in fields}}
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot
    metrics={p:summarize_pilot(rows[p],30) for p in rows}
    terminal={p:engine._terminal_phase_counts(rows[p]) for p in rows};contacts={p:engine._phase_counts(rows[p]) for p in rows}
    gate={"task":metrics["constrain_dp"]["task_success_count"]>=27,"safe":metrics["constrain_dp"]["safe_success_count"]>=26,
        "contact":metrics["constrain_dp"]["obstacle_contact_count"]<=3,"xy":metrics["constrain_dp"]["xy_crossing_count"]<=3,
        "complete":all(m["complete"] for m in metrics.values()),"base_exact":base_exact}
    row_hashes={f"{p}/episode_{i:03d}.json":sha256_file(PAIR/p/f"episode_{i:03d}.json") for p in rows for i in range(30)}
    inputs={"trial_registration_sha256":trial["trial_registration_sha256"],"training_registration_sha256":trial["training_registration_sha256"],
        "checkpoint_sha256":trial["checkpoint_sha256"],"native_row_sha256":row_hashes,"source_files":trial["source_files"],
        "dataset":{"h5_sha256":sha256_file(engine.DATA_H5),"manifest_sha256":sha256_file(engine.DATA_MANIFEST)},
        "native_evaluation_rerun":False}
    inputs["analysis_inputs_sha256"]=digest(inputs)
    if INPUTS.exists():
        if json.loads(INPUTS.read_text())!=inputs: raise ValueError("R44 immutable row-hash analysis inputs changed")
    else:
        INPUTS.write_text(json.dumps(inputs,indent=2,sort_keys=True)+"\n")
    audit=final_ema_gradient_audit("cuda")
    # Aggregate the full durable update log without selecting or filtering steps.
    keys=("loss","epsilon_mse","body_clearance","cube_clearance","lambda_eff","body_active_fraction","cube_active_fraction","x0_action_abs_gt_0_35_fraction","grad_norm")
    values={k:[] for k in keys};steps=0;all_finite=True;first=last=None
    with (EXP/"train_constrain"/"loss.jsonl").open() as stream:
        for line in stream:
            row=json.loads(line);steps+=1
            if first is None:first=row
            last=row
            all_finite &= all(np.isfinite(row[k]) for k in keys)
            for k in keys:values[k].append(float(row[k]))
    if steps!=150000 or not all_finite: raise ValueError("R44 retained train log count/finite audit failed")
    log_stats={k:{"first":v[0],"last":v[-1],"mean":float(np.mean(v)),"p50":float(np.percentile(v,50)),"p95":float(np.percentile(v,95)),"max":max(v)} for k,v in values.items()}
    protocol={p:[r.get("runtime_protocol") for r in rows[p]] for p in rows}
    if any(not x or x.get("model_kind")!=("SE3_horizon8_clearance" if p=="constrain_dp" else None)
           or x.get("runtime_clearance_cost_calls")!=0 for p in rows for x in protocol[p]):
        raise ValueError("R44 actual per-row protocol metadata missing or incorrect")
    report={"schema":"pickcube_horizon8_native_analysis_r44_v1","status":"complete","model_kind":"SE3_horizon8_clearance","horizon":8,
        "trial_registration_sha256":trial["trial_registration_sha256"],"training_registration_sha256":trial["training_registration_sha256"],
        "checkpoint_sha256":trial["checkpoint_sha256"],"response_fit_sha256":trial["response_fit_sha256"],"precheck_sha256":trial["precheck_sha256"],
        "data":DATA,"native_runtime_seconds":json.loads((EXP/"phase_runtime_native_evaluation.json").read_text())["elapsed_seconds"],
        "policy_metrics":metrics,"transition_tables":transitions,"base_action_trace_and_outcomes_exact_vs_r43":base_exact,
        "terminal_phase_counts":terminal,"first_contact_phase_counts":contacts,
        "first_contact_entity_pair_positions":"unknown; recorder does not expose contact-pair positions",
        "gate":{"passed":all(gate.values()),"criteria":gate},"native_evaluation_rerun":False,"formal_performance_claim":False,
        "inference_clearance_cost_calls":0,"per_episode_runtime_protocol":"retained in paired_native_r44/{base_dp,constrain_dp}/episode_*.json",
        "training":{"updates":steps,"elapsed_seconds":json.loads((EXP/"phase_runtime_control_training.json").read_text())["elapsed_seconds"],
            "checkpoint_sha256":trial["checkpoint_sha256"],"frozen_base_state_sha256":"5e5e7e9ebd007bd1b9319825e537aa922943e25d4844787e0e3ebd148b848b2d",
            "loss_log_all_finite":all_finite,"component_and_gradient_log_quantiles":log_stats,
            "pretraining_gradient_audit":first.get("gradient_audit"),"posttraining_gradient_audit":audit,
            "source_files_after_training":_source_files()},
        "analysis_inputs_sha256":inputs["analysis_inputs_sha256"]}
    if FINAL.exists():
        old=json.loads(FINAL.read_text())
        report["supersedes_analysis_sha256"]=sha256_file(FINAL)
    report["post_training_action_support_calibration"]=audit["action_support_calibration"]
    report["training"]["action_abs_gt_0_35_fraction_per_step"]=log_stats["x0_action_abs_gt_0_35_fraction"]
    if FINAL_V2.exists(): raise FileExistsError(FINAL_V2)
    FINAL_V2.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report


if __name__=="__main__":
    print(json.dumps(analyze(),indent=2,sort_keys=True))
