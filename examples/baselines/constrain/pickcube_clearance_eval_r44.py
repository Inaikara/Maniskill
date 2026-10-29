"""One paired R44 native trial on exactly the retained 30 development scenes."""

import hashlib
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.baselines.constrain import pickcube_body_hint_round41 as r41
from examples.baselines.constrain import pickcube_xy_gauge_round40 as r40
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_clearance_round44 import DATA, EXP, CKPT, digest
from examples.baselines.constrain.pickcube_clearance_train_r44 import (
    TRAIN_REG, TRAIN_FORMAT, PRECHECK, BASE, STATS_PATH, MODEL_KIND,
    Horizon8ClearanceActor, _audit_sources,
)
from examples.baselines.constrain.pickcube_clearance_eval_r43 import (
    R40_BASE, R41_CONTROL, R41_EXP, R41_PRIMARY, DEV_SCENES,
    EPISODE_TIMEOUT_SECONDS, OUTER_BUDGET_SECONDS,
    _validate_r41_primary, _tree_identity,
)

CONTROL=CKPT/"final.pt"
PAIR_DIR=EXP/"paired_native_r44"
TRIAL_REG=EXP/"native_trial_preregistration.json"
R43_EXP=Path("outputs/exp/20261029_052324_pickcube_clearance_r43")
R43_CONTROL=Path("outputs/ckpt/20261029_052324_pickcube_clearance_r43_constrain/final.pt")
R43_PRIMARY=R43_EXP/"paired_native_r43"
SOURCES=("examples/baselines/constrain/pickcube_clearance_eval_r44.py",
    "examples/baselines/constrain/pickcube_clearance_train_r44.py",
    "examples/baselines/constrain/pickcube_clearance_round44.py",
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/pickcube_learning_pilot.py",
    "examples/baselines/constrain/pickcube_transport_detour.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/pickcube_obstacle.py",
    "examples/baselines/constrain/evaluate_pickcube_paired.py",
    "mani_skill/agents/robots/panda/panda.py","mani_skill/assets/robots/panda/panda_v2.urdf")


def source_identity(): return {name:sha256_file(name) for name in SOURCES}


def _rows(root,policy):
    return [json.loads((root/policy/f"episode_{i:03d}.json").read_text()) for i in range(30)]


def _base_and_reference_rows():
    base={p:_rows(R41_PRIMARY,p) for p in ("base_dp","constrain_dp")}
    r43={p:_rows(R43_PRIMARY,p) for p in ("base_dp","constrain_dp")}
    return base,r43


def preflight_and_register():
    if TRIAL_REG.exists() or PAIR_DIR.exists(): raise FileExistsError("R44 sole native trial already registered/started")
    train=_audit_sources(json.loads(TRAIN_REG.read_text()))
    summary=json.loads((EXP/"train_constrain"/"train_summary.json").read_text())
    if summary.get("status")!="complete" or summary.get("updates")!=150000 or summary.get("checkpoint_sha256")!=sha256_file(CONTROL):
        raise ValueError("R44 final 150k checkpoint audit incomplete/mismatched")
    if sha256_file(R43_CONTROL)!="da818a9267a19a6ab8369a2d6b55b3e2116a28514c72662087607df635e53d48":
        raise ValueError("registered R43 primary checkpoint identity changed")
    r41reg=json.loads((R41_EXP/"preregistration.json").read_text());r41rows=_validate_r41_primary(r41reg);r41tree=_tree_identity(R41_PRIMARY)
    r43rows={p:_rows(R43_PRIMARY,p) for p in ("base_dp","constrain_dp")}
    for reference in (r41rows,r43rows):
        if any(row.get("status")!="evaluated" or row.get("scene_index")!=i for p in reference for i,row in enumerate(reference[p])):
            raise ValueError("R44 reference 30-scene cohort incomplete")
    r41._install_profile();r40._install_evaluation_loaders()
    base_equivalence=engine.preflight_action_equivalence(BASE,CONTROL,STATS_PATH,"cuda")
    if not base_equivalence.get("matches"): raise ValueError("R44 disabled Control/Base equivalence failed")
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
    scene=TransportDetourSceneSet(DEV_SCENES).load(0)
    if scene["metadata"]["scene_id"]!=r41rows["constrain_dp"][0].get("scene_id"):
        raise ValueError("R44 saved-row scene context mismatch")
    actor=Horizon8ClearanceActor(CONTROL,STATS_PATH,"cuda")
    obs=torch.as_tensor(np.asarray(scene["observation"],dtype=np.float32),device="cuda").reshape(1,45).repeat(2,1)
    torch.manual_seed(int(scene["metadata"]["seed"]));action=actor.get_action(obs)
    if action.shape!=(1,8,7) or not torch.isfinite(action).all(): raise ValueError("R44 actor saved-row preflight failed")
    context={"status":"passed","scene_index":0,"scene_id":scene["metadata"]["scene_id"],"seed":scene["metadata"]["seed"],
        "raw45_sha256":hashlib.sha256(np.ascontiguousarray(scene["observation"],dtype=np.float32).tobytes()).hexdigest(),
        "checkpoint_sha256":sha256_file(CONTROL),"model_kind":MODEL_KIND,"horizon":8,"action_shape":list(action.shape),
        "first_action_sha256":hashlib.sha256(action.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "inference_clearance_cost_calls":actor.metadata["inference_clearance_cost_calls"],"no_env_step":True}
    prereg={"schema":"pickcube_clearance_native_trial_r44_v1","status":"frozen_before_native_trial","attempt":1,
        "training_registration_sha256":train["registration_sha256"],"training_summary_sha256":sha256_file(EXP/"train_constrain"/"train_summary.json"),
        "precheck_sha256":train["precheck_sha256"],"response_fit_sha256":train["fit_sha256"],"checkpoint_sha256":sha256_file(CONTROL),
        "base_checkpoint_sha256":sha256_file(BASE),"r41_registration_sha256":r41reg["registration_sha256"],"r41_tree":r41tree,
        "r43_trial_registration_sha256":json.loads((R43_EXP/"native_trial_preregistration.json").read_text())["trial_registration_sha256"],
        "r43_tree":_tree_identity(R43_PRIMARY),"data":DATA,"normalizer_sha256":sha256_file(STATS_PATH),
        "scenes":{"path":str(DEV_SCENES),"manifest_sha256":sha256_file(DEV_SCENES/"manifest.json"),"h5_sha256":sha256_file(DEV_SCENES/"scenes.h5")},
        "source_files":source_identity(),"disabled_control_base_equivalence":base_equivalence,"saved_row_preflight":context,
        "protocol":{"policies":["R40 frozen Base","R44 horizon-8 Control"],"episodes_per_policy":30,"max_control_steps":500,
            "episode_timeout_seconds":300,"overall_timeout_seconds":36000,"sampler":"DDPM-100 epsilon squaredcos_cap_v2 clip_sample, slice [1:9], execute/replan 8",
            "actions":"native identity 7-D; no new clamp, codec, grip rule, guidance, or shield","seed":"same retained scene seed per policy","augmentation":"off"},
        "registered_comparators":{"r43_control":{"task":30,"safe":14,"contact":16,"xy":5},"r41_control":{"task":29,"safe":17,"contact":13,"xy":6},"r40_base":{"task":27,"safe":0,"contact":30,"xy":21}},
        "acceptance_gate":{"task_gte":27,"safe_gte":26,"contact_lte":3,"xy_lte":3},"formal_claim":False,"no_new_scenes_or_demos":True}
    prereg["trial_registration_sha256"]=digest(prereg);TRIAL_REG.write_text(json.dumps(prereg,indent=2,sort_keys=True)+"\n")
    return prereg


def run_trial():
    prereg=json.loads(TRIAL_REG.read_text());body=dict(prereg);ident=body.pop("trial_registration_sha256",None)
    if ident!=digest(body) or prereg.get("source_files")!=source_identity(): raise ValueError("R44 native preregistration/source mismatch")
    train=_audit_sources(json.loads(TRAIN_REG.read_text()))
    if train["registration_sha256"]!=prereg["training_registration_sha256"] or sha256_file(CONTROL)!=prereg["checkpoint_sha256"]:
        raise ValueError("R44 checkpoint/training registration mismatch before native trial")
    if PAIR_DIR.exists(): raise FileExistsError("R44 one native trial already started")
    r41._install_profile();r40._install_evaluation_loaders()
    class R44Adapter:
        def __init__(self,checkpoint,stats_path,base_checkpoint,device="cpu",control_enabled=True):
            self.actor=Horizon8ClearanceActor(checkpoint,stats_path,device);self.device=self.actor.device;self.stats=self.actor.stats
            self.control_enabled=bool(control_enabled);self.metadata=dict(self.actor.metadata,control_enabled=self.control_enabled,base_checkpoint_sha256=sha256_file(base_checkpoint))
        def base_cond(self,raw): return self.actor.base_cond(raw)
        def get_action(self,raw):
            if self.control_enabled:return self.actor.get_action(raw)
            raw=self.actor.base_cond(raw);g=engine.normalize_observation(raw[...,:42],self.stats).flatten(1)
            hints=__import__("examples.baselines.constrain.pickcube_body_hint",fromlist=["body_hint_from_raw"]).body_hint_from_raw(raw)[:,None].expand(-1,16,-1)
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            scheduler=DDPMScheduler(num_train_timesteps=100,beta_schedule="squaredcos_cap_v2",clip_sample=True,prediction_type="epsilon");scheduler.set_timesteps(100)
            sample=torch.randn((len(raw),16,7),device=self.device)
            with torch.no_grad():
                for t in scheduler.timesteps:sample=scheduler.step(self.actor.model(sample,t,global_cond=g,hint=hints,enabled=False),t,sample).prev_sample
            return sample[:,1:9]
        map_action_ee_delta_pose=staticmethod(Horizon8ClearanceActor.map_action_ee_delta_pose)
    engine.ZScoreConstrainAdapter=R44Adapter
    engine.EPISODE_TIMEOUT_SECONDS=300
    # The legacy evaluator's report code is now fixed to derive terminal counts
    # from recorded rows; all row/report failures remain in the 30-row denominator.
    from examples.baselines.constrain import pickcube_learning_pilot as pilot
    original=pilot.run_pilot_episode
    def record_protocol(adapter,scene_set,index,device,max_steps=500):
        row=original(adapter,scene_set,index,device,max_steps=max_steps)
        row["runtime_protocol"]={"model_kind":MODEL_KIND if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "horizon":8 if adapter.metadata.get("format")==TRAIN_FORMAT else None,"checkpoint_sha256":adapter.metadata.get("checkpoint_sha256"),
            "training_registration_sha256":train["registration_sha256"] if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "precheck_sha256":train["precheck_sha256"] if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "normalizer_sha256":sha256_file(STATS_PATH),"runtime_clearance_cost_calls":0,"native_actions":True,"augmentation":"off"}
        return row
    pilot.run_pilot_episode=record_protocol
    native=EXP/"phase_runtime_native_evaluation.json"
    runtime={"phase":"single_paired_30_scene_native_trial","status":"running","argv":sys.argv,"started_at_unix":time.time(),
        "budget_seconds":36000,"episode_timeout_seconds":300,"trial_registration_sha256":ident,"checkpoint_sha256":sha256_file(CONTROL),
        "base_sha256":sha256_file(BASE),"source_files":source_identity(),"data_h5_sha256":sha256_file(engine.DATA_H5),"manifest_sha256":sha256_file(engine.DATA_MANIFEST)}
    native.write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
    r41reg=dict(json.loads((R41_EXP/"preregistration.json").read_text()));r41reg["paths"]=dict(r41reg["paths"])
    r41reg["paths"].update({"registration":str(EXP/"native_engine_registration.json"),"base_checkpoint":str(BASE),"constrain_checkpoint":str(CONTROL),"paired":str(PAIR_DIR),"dev_scenes":str(DEV_SCENES)})
    r41reg["native_evaluation"]=dict(r41reg["native_evaluation"],episodes=30,max_control_steps=500,wall_timeout_seconds_per_episode=300,outer_timeout_seconds=36000)
    r41reg["matched_base_checkpoint"]={"path":str(BASE),"sha256":sha256_file(BASE)};r41reg["matched_constrain_checkpoint"]={"path":str(CONTROL),"sha256":sha256_file(CONTROL)}
    r41reg.pop("registration_sha256",None);r41reg["registration_sha256"]=digest(r41reg)
    (EXP/"native_engine_registration.json").write_text(json.dumps(r41reg,indent=2,sort_keys=True)+"\n")
    started=time.monotonic();prior=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit(143)))
    try:
        engine.evaluate_round37(EXP/"native_engine_registration.json")
        report=analyze_rows()
        runtime.update(status="complete",elapsed_seconds=time.monotonic()-started,analysis_sha256=sha256_file(EXP/"native_analysis_r44.json"),
            source_files_after=source_identity(),data_h5_sha256_after=sha256_file(engine.DATA_H5),manifest_sha256_after=sha256_file(engine.DATA_MANIFEST))
        return report
    except BaseException as exc:
        runtime.update(status="failed",error=f"{type(exc).__name__}: {exc}",elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        signal.signal(signal.SIGTERM,prior);pilot.run_pilot_episode=original;runtime["finished_at_unix"]=time.time();native.write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")


def analyze_rows():
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot
    rows={p:_rows(PAIR_DIR,p) for p in ("base_dp","constrain_dp")}
    if any(row.get("scene_index")!=i for p in rows for i,row in enumerate(rows[p])): raise ValueError("R44 retained rows incomplete/unordered")
    metrics={p:summarize_pilot(rows[p],30) for p in rows}
    if any(not metrics[p]["complete"] for p in metrics): raise ValueError("R44 runtime-error denominators stay 30; do not rerun")
    r41rows,r43rows=_base_and_reference_rows()
    fields=("native_success","safe_success","obstacle_contact","xy_forbidden_crossing")
    from examples.baselines.constrain.pickcube_zscore_round37 import _transitions,_terminal_phase_counts,_phase_counts
    transitions={"base_vs_control":{f:_transitions(rows["base_dp"],rows["constrain_dp"],f) for f in fields},
        "r43_control_to_r44_control":{f:_transitions(r43rows["constrain_dp"],rows["constrain_dp"],f) for f in fields},
        "r41_control_to_r44_control":{f:_transitions(r41rows["constrain_dp"],rows["constrain_dp"],f) for f in fields}}
    base_exact=all(a.get("step_trace")==b.get("step_trace") and a.get("result")==b.get("result") for a,b in zip(r43rows["base_dp"],rows["base_dp"]))
    if not base_exact: raise ValueError("R44 matched R40 Base traces do not exactly match registered R43/R41 Base")
    contacts={p:_phase_counts(rows[p]) for p in rows};terminal={p:_terminal_phase_counts(rows[p]) for p in rows}
    gate={"task":metrics["constrain_dp"]["task_success_count"]>=27,"safe":metrics["constrain_dp"]["safe_success_count"]>=26,
        "contact":metrics["constrain_dp"]["obstacle_contact_count"]<=3,"xy":metrics["constrain_dp"]["xy_crossing_count"]<=3,
        "complete":True,"base_exact":base_exact}
    report={"schema":"pickcube_clearance_native_analysis_r44_v1","model_kind":MODEL_KIND,"horizon":8,"status":"complete",
        "policy_metrics":metrics,"transition_tables":transitions,"terminal_phase_counts":terminal,"first_contact_phase_counts":contacts,
        "base_action_trace_and_outcomes_exact_vs_r43":base_exact,"gate":{"passed":all(gate.values()),"criteria":gate},
        "native_trial_registration_sha256":json.loads(TRIAL_REG.read_text())["trial_registration_sha256"],
        "training_registration_sha256":json.loads(TRAIN_REG.read_text())["registration_sha256"],"checkpoint_sha256":sha256_file(CONTROL),
        "runtime_clearance_cost_calls":0,"native_evaluation_rerun":False,"formal_performance_claim":False}
    path=EXP/"native_analysis_r44.json"
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    return report
