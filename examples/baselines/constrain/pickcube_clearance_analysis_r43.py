"""Offline-only R43 report audit over immutable paired episode rows."""

import json
from pathlib import Path

from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_clearance_eval_r43 import (
    EXP, PAIR_DIR, R41_EXP, R41_PRIMARY, _r41_rows, _tree_identity,
)
from examples.baselines.constrain.pickcube_zscore_round37 import _transitions
from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot


def analyze():
    trial=json.loads((EXP/"native_trial_preregistration.json").read_text())
    trial_sha=trial.pop("trial_registration_sha256")
    from examples.baselines.constrain.pickcube_clearance_round43 import _digest
    if trial_sha!=_digest(trial): raise ValueError("R43 paired trial registration checksum mismatch")
    trial["trial_registration_sha256"]=trial_sha
    registration=json.loads((R41_EXP/"preregistration.json").read_text())
    primary=_r41_rows()
    tree=_tree_identity(R41_PRIMARY)
    if tree!=trial["r41_primary_tree"]: raise ValueError("R41 primary evidence tree changed after R43 trial preregistration")
    snapshot=json.loads((EXP/"native_analysis_inputs.json").read_text())
    snapshot_body=dict(snapshot);snapshot_sha=snapshot_body.pop("analysis_inputs_sha256")
    if snapshot_sha!=_digest(snapshot_body): raise ValueError("R43 immutable analysis input snapshot checksum mismatch")
    if snapshot["trial_registration_sha256"]!=trial_sha: raise ValueError("analysis inputs belong to another trial")
    policies=("base_dp","constrain_dp")
    rows={p:[json.loads((PAIR_DIR/p/f"episode_{i:03d}.json").read_text()) for i in range(30)] for p in policies}
    actual_hashes={f"{p}/episode_{i:03d}.json":sha256_file(PAIR_DIR/p/f"episode_{i:03d}.json")
                   for p in policies for i in range(30)}
    if actual_hashes!=snapshot["r43_native_row_sha256"]: raise ValueError("retained native rows changed after analysis input freeze")
    if any(r.get("status")!="evaluated" or r.get("scene_index")!=i for p in policies for i,r in enumerate(rows[p])):
        raise ValueError("retained native cohort incomplete; denominator remains 30")
    if any(not summarize_pilot(rows[p],30)["complete"] for p in policies):
        raise ValueError("runtime-error denominator prevents complete trial summary")
    base_trace_exact=all(a.get("step_trace")==b.get("step_trace") and a.get("result")==b.get("result")
        for a,b in zip(primary["base_dp"],rows["base_dp"]))
    if not base_trace_exact:
        raise ValueError("R43 matched R40 Base action/outcome traces differ from retained R41 Base")
    fields=("native_success","safe_success","obstacle_contact","xy_forbidden_crossing")
    transitions={"R40_base_to_R43_base":{f:_transitions(primary["base_dp"],rows["base_dp"],f) for f in fields},
        "R41_control_to_R43_control":{f:_transitions(primary["constrain_dp"],rows["constrain_dp"],f) for f in fields},
        "R43_base_to_control":{f:_transitions(rows["base_dp"],rows["constrain_dp"],f) for f in fields}}
    metrics={p:summarize_pilot(rows[p],30) for p in policies}
    stages={};contacts={}
    for policy in policies:
        stage={};contact={}
        for row in rows[policy]:
            phase=row["result"].get("phase","unknown");stage[phase]=stage.get(phase,0)+1
            phase=(row.get("first_contact") or {}).get("phase")
            if phase:contact[phase]=contact.get(phase,0)+1
        stages[policy]=stage;contacts[policy]=contact
    criteria={"task":metrics["constrain_dp"]["task_success_count"]>=27,
        "safe":metrics["constrain_dp"]["safe_success_count"]>=26,
        "contact":metrics["constrain_dp"]["obstacle_contact_count"]<=3,
        "xy":metrics["constrain_dp"]["xy_crossing_count"]<=3,
        "complete":all(metrics[p]["complete"] for p in policies),"paired":True}
    report={"status":"complete","trial_registration_sha256":trial_sha,
        "training_registration_sha256":trial["training_registration_sha256"],
        "precheck_sha256":trial["precheck_sha256"],"checkpoint_sha256":trial["checkpoint_sha256"],
        "primary_r41_registration_sha256":trial["r41_primary_registration_sha256"],
        "primary_r41_tree_sha256":tree["sha256"],"policy_metrics":metrics,
        "base_native_action_trace_and_outcomes_exact_vs_r41":base_trace_exact,
        "transition_tables":transitions,"terminal_phase_counts":stages,
        "first_contact_phase_counts":contacts,
        "first_contact_entity_pair_positions":"unknown; native recorder does not expose contact entity-pair positions",
        "gate":{"passed":all(criteria.values()),"criteria":criteria},
        "analysis_inputs_sha256":snapshot_sha,"native_evaluation_rerun":False,
        "formal_performance_claim":False}
    final=EXP/"native_analysis_final.json"
    if final.exists(): raise FileExistsError(final)
    final.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    correction={"event":"offline_analysis_enrichment_from_immutable_native_rows",
        "analysis_only_source":str(Path(__file__).relative_to(Path.cwd())),
        "analysis_only_source_sha256":sha256_file(__file__),
        "native_trial_registration_sha256":trial_sha,"analysis_inputs_sha256":snapshot_sha,
        "r43_row_hashes_verified":True,"r41_primary_tree_hash_verified":True,
        "base_action_traces_exact":base_trace_exact,"native_evaluation_rerun":False,
        "original_offline_analysis_preserved":str(EXP/"native_analysis_offline.json"),
        "final_analysis_sha256":sha256_file(final)}
    correction_path=EXP/"native_analysis_source_correction.json"
    correction_path.write_text(json.dumps(correction,indent=2,sort_keys=True)+"\n")
    return report


if __name__=="__main__":
    print(json.dumps(analyze(),indent=2,sort_keys=True))
