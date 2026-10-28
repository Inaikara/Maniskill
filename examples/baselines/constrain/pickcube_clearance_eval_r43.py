"""One preregistered paired native R43 clearance trial and immutable analysis."""

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
from examples.baselines.constrain.pickcube_clearance_round43 import DATA, EXP, CKPT, _digest
from examples.baselines.constrain.pickcube_clearance_train_r43 import (
    ClearanceActor, PRECHECK_REPORT, TRAIN_REGISTRATION, TRAIN_FORMAT, _verify_training_registration,
)

R41_EXP = Path("outputs/exp/20261029_031500_pickcube_bodyhint_r41")
R41_PRIMARY = R41_EXP / "paired"
R40_BASE = Path("outputs/ckpt/20261029_004500_pickcube_xy_gauge_r40_base/final.pt")
R41_CONTROL = Path("outputs/ckpt/20261029_031500_pickcube_bodyhint_r41_constrain/final.pt")
BASE = R40_BASE
CONTROL = CKPT / "final.pt"
STATS = R41_EXP / "position_encoding.json"
DEV_SCENES = Path("outputs/exp/20261023_235744_pickcube_learning_pilot_r2/dev_scenes")
PAIR_DIR = EXP / "paired_native_r43"
TRIAL_REG = EXP / "native_trial_preregistration.json"
EPISODE_TIMEOUT_SECONDS = 300
OUTER_BUDGET_SECONDS = 36000

SOURCES = (
    "examples/baselines/constrain/pickcube_clearance_eval_r43.py",
    "examples/baselines/constrain/pickcube_clearance_train_r43.py",
    "examples/baselines/constrain/pickcube_clearance_proxy_r43.py",
    "examples/baselines/constrain/pickcube_clearance_round43.py",
    "examples/baselines/constrain/pickcube_body_hint.py",
    "examples/baselines/constrain/pickcube_body_hint_round41.py",
    "examples/baselines/constrain/pickcube_xy_gauge.py",
    "examples/baselines/constrain/pickcube_xy_gauge_round40.py",
    "examples/baselines/constrain/pickcube_zscore_round37.py",
    "examples/baselines/constrain/pickcube_learning_pilot.py",
    "examples/baselines/constrain/pickcube_transport_detour.py",
    "examples/baselines/constrain/pickcube_expert.py",
    "examples/baselines/constrain/pickcube_obstacle.py",
    "examples/baselines/constrain/diagnose_pickcube_contact.py",
    "examples/baselines/constrain/evaluate_pickcube_paired.py",
    "mani_skill/agents/robots/panda/panda.py",
    "mani_skill/assets/robots/panda/panda_v2.urdf",
)


def _source_identity():
    return {name: sha256_file(name) for name in SOURCES}


def _tree_identity(root):
    root=Path(root)
    files={p.relative_to(root).as_posix():sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}
    return {"files":files,"sha256":_digest(files)}


def _r41_rows():
    return {policy:[json.loads((R41_PRIMARY/policy/f"episode_{i:03d}.json").read_text())
                     for i in range(30)] for policy in ("base_dp","constrain_dp")}


def _validate_r41_primary(reg):
    body=dict(reg);checksum=body.pop("registration_sha256",None)
    if checksum!=_digest(body): raise ValueError("R41 primary registration checksum mismatch")
    if reg["paths"]["dev_scenes"]!=str(DEV_SCENES): raise ValueError("R41 scene source mismatch")
    if sha256_file(R40_BASE)!="9cee2f8ce05461c8b3ab75a834db94c2b88cf9c864cb95d79b78c501cd961719":
        raise ValueError("R40 Base checkpoint hash mismatch")
    if sha256_file(R41_CONTROL)!="838745ceb9437fa9db9c5c289a88a032cdf8a12dca42145e896d3c7ac3529642":
        raise ValueError("R41 primary Control checkpoint hash mismatch")
    rows=_r41_rows()
    if any(row.get("status")!="evaluated" or row.get("scene_index")!=i
           for policy in rows for i,row in enumerate(rows[policy])):
        raise ValueError("R41 primary 30-scene rows incomplete")
    return rows


def _install_actor_adapters():
    r41._install_profile()
    # Preserve the proven matched-R40 Base adapter including gauge metadata checks.
    r40._install_evaluation_loaders()

    class R43ControlAdapter:
        def __init__(self, checkpoint, stats_path, base_checkpoint, device="cpu", control_enabled=True):
            self.actor=ClearanceActor(checkpoint,stats_path,device)
            self.device=self.actor.device
            self.stats=self.actor.stats
            self.control_enabled=bool(control_enabled)
            self.metadata=dict(self.actor.metadata,control_enabled=self.control_enabled,
                base_checkpoint_sha256=sha256_file(base_checkpoint))

        def base_cond(self,raw):
            return self.actor.base_cond(raw)

        def get_action(self,raw):
            raw=self.actor.base_cond(raw)
            global_cond=engine.normalize_observation(raw[...,:42],self.stats).flatten(1)
            from examples.baselines.constrain.pickcube_body_hint import body_hint_from_raw
            hint=body_hint_from_raw(raw)[:,None,:].expand(-1,16,-1)
            sample=torch.randn((len(raw),16,7),device=self.device)
            with torch.no_grad():
                for timestep in self.actor.scheduler.timesteps:
                    epsilon=self.actor.model(sample,timestep,global_cond=global_cond,
                        hint=hint,enabled=self.control_enabled)
                    sample=self.actor.scheduler.step(epsilon,timestep,sample).prev_sample
            return sample[:,1:9]

        @staticmethod
        def map_action_ee_delta_pose(action):
            return ClearanceActor.map_action_ee_delta_pose(action)

    engine.ZScoreConstrainAdapter=R43ControlAdapter
    return R43ControlAdapter


def preflight_and_register():
    if TRIAL_REG.exists() or PAIR_DIR.exists(): raise FileExistsError("R43 paired trial already registered/started")
    training_reg=_verify_training_registration()
    summary=json.loads((EXP/"train_constrain"/"train_summary.json").read_text())
    if summary.get("status")!="complete" or summary.get("updates")!=150000:
        raise ValueError("R43 final 150k training audit is incomplete")
    if summary.get("checkpoint_sha256")!=sha256_file(CONTROL):
        raise ValueError("R43 final EMA hash differs from training audit")
    r41_reg=json.loads((R41_EXP/"preregistration.json").read_text())
    primary_rows=_validate_r41_primary(r41_reg)
    primary_tree=_tree_identity(R41_PRIMARY)

    adapter_cls=_install_actor_adapters()
    equivalence=engine.preflight_action_equivalence(BASE,CONTROL,STATS,"cuda")
    if not equivalence.get("matches"):
        raise ValueError("R43 disabled-Control vs independent frozen Base preflight failed")

    # Revisit one already-retained R41 scene observation without env reset/step.
    from examples.baselines.constrain.pickcube_transport_detour import TransportDetourSceneSet
    scene_set=TransportDetourSceneSet(DEV_SCENES)
    scene=scene_set.load(0)
    r41_row=primary_rows["constrain_dp"][0]
    if scene["metadata"]["scene_id"]!=r41_row.get("scene_id") or int(scene["metadata"]["seed"])!=int(r41_row.get("seed")):
        raise ValueError("R41 retained row does not match the restored scene fixture")
    actor=ClearanceActor(CONTROL,STATS,"cuda")
    observation=torch.as_tensor(np.asarray(scene["observation"],dtype=np.float32),device="cuda")
    queued=observation.reshape(1,45).repeat(2,1)
    torch.manual_seed(int(scene["metadata"]["seed"]))
    first_action=actor.get_action(queued)
    if first_action.shape!=(1,8,7) or not bool(torch.isfinite(first_action).all()):
        raise ValueError("R43 actor failed the R41 saved-row metadata/action preflight")
    reporter={"status":"passed","scene_index":0,"scene_id":scene["metadata"]["scene_id"],
        "scene_seed":scene["metadata"]["seed"],"r41_episode_row_sha256":sha256_file(R41_PRIMARY/"constrain_dp"/"episode_000.json"),
        "raw45_observation_sha256":hashlib.sha256(np.ascontiguousarray(scene["observation"],dtype=np.float32).tobytes()).hexdigest(),
        "normalizer_sha256":sha256_file(STATS),"checkpoint_sha256":sha256_file(CONTROL),
        "model_format":TRAIN_FORMAT,"objective_type":"swept_body_and_grasped_cube_xy_training_only",
        "hint_dim":9,"condition_dim":84,"first_native_action_shape":list(first_action.shape),
        "first_native_action_sha256":hashlib.sha256(first_action.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        "geometry_cost_calls_at_inference":actor.metadata["inference_clearance_cost_calls"],
        "no_environment_constructor_or_step":True}
    preflight_dir=EXP/"preflight"
    reporter_path=preflight_dir/"r41_row_r43_actor_context.json"
    if reporter_path.exists(): raise FileExistsError(reporter_path)
    reporter_path.write_text(json.dumps(reporter,indent=2,sort_keys=True)+"\n")

    scenes={"manifest_sha256":sha256_file(DEV_SCENES/"manifest.json"),
            "scenes_h5_sha256":sha256_file(DEV_SCENES/"scenes.h5"),"path":str(DEV_SCENES)}
    prereg={"schema":"pickcube_clearance_native_trial_r43_v1","status":"frozen_before_native_trial",
        "attempt":1,"training_registration_sha256":training_reg["registration_sha256"],
        "precheck_sha256":training_reg["precheck_sha256"],"precheck_registration_sha256":training_reg["precheck_registration_sha256"],
        "training_summary_sha256":sha256_file(EXP/"train_constrain"/"train_summary.json"),
        "checkpoint_sha256":sha256_file(CONTROL),"base_checkpoint_sha256":sha256_file(BASE),
        "r41_primary_registration_sha256":r41_reg["registration_sha256"],
        "r41_primary_tree":primary_tree,"r41_control_checkpoint_sha256":sha256_file(R41_CONTROL),
        "data":DATA,"normalizer_sha256":sha256_file(STATS),"scenes":scenes,
        "source_files":_source_identity(),"disabled_control_base_equivalence":equivalence,
        "r41_saved_row_context_preflight":reporter,
        "protocol":{"policies":["R40 matched Base","R43 clearance Control"],"episodes_per_policy":30,
            "max_control_steps":500,"wall_timeout_seconds_per_episode":EPISODE_TIMEOUT_SECONDS,
            "outer_wall_budget_seconds":OUTER_BUDGET_SECONDS,"sampler":"DDPM-100 epsilon squaredcos_cap_v2 clip_sample; predicted slice [1:9]; execute 8",
            "native_action":"identity pd_ee_delta_pose 7-D; no codec/clamp/shield/guidance; scale 1",
            "inference_augmentation":"off","scene_seed":"same retained per-scene seed independently for each policy",
            "success":"native first success; record actual terminated/truncated flags"},
        "primary_expected":{"r40_base":{"task":27,"safe":0,"contact":30,"xy":21},
            "r41_control":{"task":29,"safe":17,"contact":13,"xy":6}},
        "acceptance_gate":{"task_gte":27,"safe_gte":26,"contact_lte":3,"xy_lte":3},
        "no_new_scenes_or_demos":True}
    prereg["trial_registration_sha256"]=_digest(prereg)
    TRIAL_REG.write_text(json.dumps(prereg,indent=2,sort_keys=True)+"\n")
    return prereg


def _distribution(values):
    values=[float(x) for x in values if x is not None]
    return None if not values else {"min":min(values),"median":float(np.median(values)),
        "p90":float(np.percentile(values,90)),"max":max(values)}


def analyze_rows():
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot
    primary=_validate_r41_primary(json.loads((R41_EXP/"preregistration.json").read_text()))
    rows={policy:[json.loads((PAIR_DIR/policy/f"episode_{i:03d}.json").read_text())
                  for i in range(30)] for policy in ("base_dp","constrain_dp")}
    if any(row.get("scene_index")!=i for policy in rows for i,row in enumerate(rows[policy])):
        raise ValueError("R43 retained native rows are unordered/incomplete")
    metrics={p:summarize_pilot(rows[p],30) for p in rows}
    if any(not metrics[p]["complete"] for p in metrics):
        raise ValueError("R43 trial has runtime errors; retain denominator 30 and do not rerun")
    same_pair=all(a.get(k)==b.get(k) for a,b in zip(rows["base_dp"],rows["constrain_dp"])
        for k in ("scene_id","seed","restored_state_sha256","restored_controller_state_sha256","restored_observation_sha256","normalizer_condition_sha256"))
    same_r41={p:all(a.get(k)==b.get(k) for a,b in zip(primary[p],rows[p])
        for k in ("scene_id","seed","restored_state_sha256","restored_controller_state_sha256","restored_observation_sha256","normalizer_condition_sha256")) for p in rows}
    if not same_pair or not all(same_r41.values()): raise ValueError("R43/R41 paired scene-state identity mismatch")
    fields=("native_success","safe_success","obstacle_contact","xy_forbidden_crossing")
    from examples.baselines.constrain.pickcube_zscore_round37 import _transitions
    transitions={"base_vs_control":{f:_transitions(rows["base_dp"],rows["constrain_dp"],f) for f in fields},
        "r41_control_to_r43_control":{f:_transitions(primary["constrain_dp"],rows["constrain_dp"],f) for f in fields},
        "r41_base_to_r43_base":{f:_transitions(primary["base_dp"],rows["base_dp"],f) for f in fields}}
    first_contacts={}
    for policy in rows:
        counter={}
        for row in rows[policy]:
            phase=(row.get("first_contact") or {}).get("phase")
            if phase: counter[phase]=counter.get(phase,0)+1
        first_contacts[policy]=counter
    terminal={}
    for policy in rows:
        counts={}
        for row in rows[policy]:
            phase=row.get("result",{}).get("phase","unknown")
            counts[phase]=counts.get(phase,0)+1
        terminal[policy]=counts
    gate={"task":metrics["constrain_dp"]["task_success_count"]>=27,
        "safe":metrics["constrain_dp"]["safe_success_count"]>=26,
        "contact":metrics["constrain_dp"]["obstacle_contact_count"]<=3,
        "xy":metrics["constrain_dp"]["xy_crossing_count"]<=3,
        "complete":True,"paired":same_pair}
    reg=json.loads(TRIAL_REG.read_text())
    row_hashes={f"{p}/episode_{i:03d}.json":sha256_file(PAIR_DIR/p/f"episode_{i:03d}.json")
                for p in rows for i in range(30)}
    inputs={"trial_registration_sha256":reg["trial_registration_sha256"],
        "r41_primary_tree_sha256":reg["r41_primary_tree"]["sha256"],"r41_primary_tree_files":reg["r41_primary_tree"]["files"],
        "r43_native_row_sha256":row_hashes,"scene_manifest_sha256":reg["scenes"]["manifest_sha256"],
        "scenes_h5_sha256":reg["scenes"]["scenes_h5_sha256"],"checkpoint_sha256":reg["checkpoint_sha256"],
        "source_files_at_trial":reg["source_files"],"native_evaluation_rerun":False}
    inputs["analysis_inputs_sha256"]=_digest(inputs)
    input_path=EXP/"native_analysis_inputs.json"
    if input_path.exists(): raise FileExistsError(input_path)
    input_path.write_text(json.dumps(inputs,indent=2,sort_keys=True)+"\n")
    report={"status":"complete","policy_metrics":metrics,"transition_tables":transitions,
        "pairing_identity_matches_scene_by_scene":same_pair,
        "r41_restored_state_identity_matches_scene_by_scene":same_r41,
        "first_contact_phase_counts":first_contacts,"terminal_phase_counts":terminal,
        "first_contact_entity_pair_positions":"unknown; native recorder does not expose contact entity-pair positions",
        "runtime_errors":{p:metrics[p]["runtime_errors"] for p in rows},
        "first_success_step_distribution":{p:_distribution([r.get("first_native_success_step") for r in rows[p]]) for p in rows},
        "wall_seconds_distribution":{p:_distribution([r.get("wall_seconds") for r in rows[p]]) for p in rows},
        "final_goal_distance_3d_distribution":{p:_distribution([r.get("final_goal_distance_3d") for r in rows[p]]) for p in rows},
        "termination_flags":{p:{"terminated":sum(bool(r["result"].get("terminated")) for r in rows[p]),
            "truncated":sum(bool(r["result"].get("truncated")) for r in rows[p])} for p in rows},
        "gate":{"passed":all(gate.values()),"criteria":gate},
        "primary_comparison":"R40 Base + exact R43 Control objective/checkpoint versus retained R41 Base/Control on same 30 restored R36 scenes",
        "analysis_inputs_sha256":inputs["analysis_inputs_sha256"],"native_evaluation_rerun":False,
        "formal_performance_claim":False}
    path=EXP/"native_analysis_offline.json"
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    status={"status":"complete","retained_rows":{"base_dp":30,"constrain_dp":30},
        "runtime_errors":report["runtime_errors"],"gate_passed":report["gate"]["passed"],
        "analysis_sha256":sha256_file(path),"native_evaluation_rerun":False}
    (EXP/"native_analysis_status.json").write_text(json.dumps(status,indent=2,sort_keys=True)+"\n")
    return report


def run_trial():
    prereg=json.loads(TRIAL_REG.read_text())
    body=dict(prereg);identity=body.pop("trial_registration_sha256",None)
    if identity!=_digest(body) or prereg["source_files"]!=_source_identity():
        raise ValueError("R43 native trial registration/source identity mismatch")
    training_reg=_verify_training_registration()
    if prereg["training_registration_sha256"]!=training_reg["registration_sha256"]:
        raise ValueError("R43 trial is not bound to the completed training registration")
    if sha256_file(CONTROL)!=prereg["checkpoint_sha256"] or sha256_file(BASE)!=prereg["base_checkpoint_sha256"]:
        raise ValueError("R43 frozen actor/Base checkpoint changed before native trial")
    if PAIR_DIR.exists(): raise FileExistsError("the one R43 native trial has already started")
    r41._install_profile();_install_actor_adapters()
    native_path=EXP/"phase_runtime_native_evaluation.json"
    runtime={"phase":"one_paired_30_scene_native_trial","status":"running",
        "argv":sys.argv,"started_at_unix":time.time(),"budget_seconds":OUTER_BUDGET_SECONDS,
        "episode_wall_timeout_seconds":EPISODE_TIMEOUT_SECONDS,
        "trial_registration_sha256":prereg["trial_registration_sha256"],
        "training_registration_sha256":training_reg["registration_sha256"],
        "checkpoint_sha256":sha256_file(CONTROL),"base_checkpoint_sha256":sha256_file(BASE),
        "source_files":_source_identity(),"data_h5_sha256":sha256_file(engine.DATA_H5),
        "manifest_sha256":sha256_file(engine.DATA_MANIFEST),"gpu_name":torch.cuda.get_device_name(0)}
    native_path.write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")
    reg=json.loads((R41_EXP/"preregistration.json").read_text())
    reg=dict(reg)
    reg["paths"]=dict(reg["paths"])
    reg["paths"].update({"registration":str(EXP/"native_engine_registration.json"),
        "base_checkpoint":str(BASE),"constrain_checkpoint":str(CONTROL),
        "paired":str(PAIR_DIR),"dev_scenes":str(DEV_SCENES)})
    reg["native_evaluation"]=dict(reg["native_evaluation"],episodes=30,max_control_steps=500,
        wall_timeout_seconds_per_episode=EPISODE_TIMEOUT_SECONDS,outer_timeout_seconds=OUTER_BUDGET_SECONDS)
    reg["matched_base_checkpoint"]={"path":str(BASE),"sha256":sha256_file(BASE)}
    reg["matched_constrain_checkpoint"]={"path":str(CONTROL),"sha256":sha256_file(CONTROL)}
    reg["source_identity"]=r41.source_identity()
    reg.pop("registration_sha256",None);reg["registration_sha256"]=_digest(reg)
    engine_reg=EXP/"native_engine_registration.json"
    engine_reg.write_text(json.dumps(reg,indent=2,sort_keys=True)+"\n")
    engine.EPISODE_TIMEOUT_SECONDS=EPISODE_TIMEOUT_SECONDS
    from examples.baselines.constrain import pickcube_learning_pilot as pilot
    original=pilot.run_pilot_episode
    def with_runtime_protocol(adapter,scene_set,index,device,max_steps=500):
        row=original(adapter,scene_set,index,device,max_steps=max_steps)
        row["runtime_protocol"]={"model_format":TRAIN_FORMAT if adapter.metadata.get("format")==TRAIN_FORMAT else adapter.metadata.get("format"),
            "checkpoint_sha256":adapter.metadata["checkpoint_sha256"],
            "objective_type":"swept_body_and_grasped_cube_xy_training_only" if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "training_registration_sha256":training_reg["registration_sha256"] if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "precheck_sha256":training_reg["precheck_sha256"] if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "normalizer_sha256":sha256_file(STATS),"hint_dim":9 if adapter.metadata.get("format")==TRAIN_FORMAT else None,
            "runtime_clearance_cost_calls":0,"guidance_scale":1.0,"inference_augmentation":"off"}
        return row
    pilot.run_pilot_episode=with_runtime_protocol
    started=time.monotonic();old_term=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(SystemExit(143)))
    try:
        try: engine.evaluate_round37(engine_reg)
        except KeyError as exc:
            # R37's known summary-only phase_counts access can fail after all rows are durable.
            if exc.args!=("phase_counts",):raise
        report=analyze_rows()
        runtime.update(status="complete",elapsed_seconds=time.monotonic()-started,
            analysis_sha256=sha256_file(EXP/"native_analysis_offline.json"),
            source_files_after=_source_identity(),data_h5_sha256_after=sha256_file(engine.DATA_H5),
            manifest_sha256_after=sha256_file(engine.DATA_MANIFEST),checkpoint_sha256_after=sha256_file(CONTROL))
        return report
    except BaseException as exc:
        runtime.update(status="failed",error=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        signal.signal(signal.SIGTERM,old_term)
        pilot.run_pilot_episode=original
        runtime.update(finished_at_unix=time.time())
        native_path.write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\n")


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=("preflight-register","run","analyze"))
    args=parser.parse_args()
    if args.command=="preflight-register":result=preflight_and_register()
    elif args.command=="run":result=run_trial()
    else:result=analyze_rows()
    print(json.dumps(result,indent=2,sort_keys=True,default=str))


if __name__=="__main__":main()
