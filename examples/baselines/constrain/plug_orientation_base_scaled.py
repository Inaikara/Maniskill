"""Explicit v2 action-scaled corrective family for Plug Base DP."""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from examples.baselines.constrain import plug_orientation_base_dp as base
from examples.baselines.constrain.plug_orientation_data import sha256_file

FORMAT = "plug_orientation_fixed_base_dp_action_scaled_v2"
ADDENDUM = Path(__file__).with_name("plug_orientation_base_scaled_contract.json")
BASE_CONTRACT = Path(__file__).with_name("plug_orientation_contract.json")


def resolved_config():
    cfg = base.resolved_config()
    cfg.update(format=FORMAT, normalization="symmetric_train_maxabs",
               action_serialization="native_actions_divided_by_registered_scale",
               sampler_decode="normalized_sampler_output_times_registered_scale")
    return cfg


def _contract():
    c = json.loads(ADDENDUM.read_text())
    if c["base_contract_sha256"] != sha256_file(BASE_CONTRACT):
        raise ValueError("v2 addendum no longer binds the frozen v1 contract")
    return c


def fit_action_scale(data_path, manifest_path, output_path):
    """Fit and persist maxabs scales from all rows of the admitted TRAIN HDF5."""
    contract = _contract()
    manifest_path = Path(manifest_path)
    dataset_manifest = manifest_path.with_name("dataset_manifest.json")
    expected = contract["training_data"]
    paths = {"hdf5": Path(data_path), "json": manifest_path, "dataset_manifest": dataset_manifest}
    for name, path in paths.items():
        key = {"hdf5": "hdf5_sha256", "json": "json_sha256", "dataset_manifest": "dataset_manifest_sha256"}[name]
        if not path.is_file() or sha256_file(path) != expected[key]:
            raise ValueError(f"v2 scale fitting requires the admitted TRAIN {name} identity")
    before = {name: sha256_file(path) for name, path in paths.items()}
    manifest=json.loads(manifest_path.read_text())
    episodes=manifest.get("episodes",[])
    if len(episodes)!=1000 or [int(row.get("episode_id",-1)) for row in episodes]!=list(range(1000)):
        raise ValueError("scale fitting requires all 1000 ordered admitted TRAIN episodes")
    with h5py.File(data_path,"r") as h5:
        if len(h5)!=1000: raise ValueError("scale fitting requires 1000 TRAIN trajectories")
        rows=[]
        for episode in range(1000):
            actions=np.asarray(h5[f"traj_{episode}/actions"][:],dtype=np.float32)
            if actions.ndim!=2 or actions.shape[1]!=7: raise ValueError("TRAIN episode does not contain native 7-D actions")
            rows.append(actions)
    action_rows=np.concatenate(rows,axis=0)
    if action_rows.shape != (expected["action_rows"], base.ACT_DIM) or not np.isfinite(action_rows).all():
        raise ValueError("admitted TRAIN action rows have unexpected shape or non-finite values")
    maxima = np.max(np.abs(action_rows), axis=0).astype(np.float32)
    scale = np.maximum(maxima, np.float32(1e-6)).astype(np.float32)
    if scale.shape != (7,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("invalid fitted action scale")
    if (action_rows / scale).min() < -1.000001 or (action_rows / scale).max() > 1.000001:
        raise ValueError("TRAIN actions do not fit normalized [-1,1]")
    if before != {name: sha256_file(path) for name, path in paths.items()}:
        raise ValueError("TRAIN data changed while fitting action scales")
    full_row_sha = hashlib.sha256(np.ascontiguousarray(action_rows).tobytes()).hexdigest()
    payload = {"format": FORMAT, "method": "per_channel_full_train_maxabs", "scale": scale.tolist(),
        "episode_count": 1000, "action_row_count": len(action_rows), "full_action_rows_sha256": full_row_sha,
        "dataset_h5_path": str(paths["hdf5"].resolve()), "dataset_h5_sha256": before["hdf5"],
        "dataset_json_path": str(paths["json"].resolve()), "dataset_json_sha256": before["json"],
        "dataset_manifest_path": str(paths["dataset_manifest"].resolve()),
        "dataset_manifest_sha256": before["dataset_manifest"],
        "addendum_sha256": sha256_file(ADDENDUM)}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists(): raise FileExistsError(output_path)
    output_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


class ScaledAgent(nn.Module):
    """Keep the official diffusion Agent intact; scale targets and decode actions."""
    def __init__(self, agent, scale):
        super().__init__()
        self.agent = agent
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32).reshape(7))
        self.obs_horizon, self.act_horizon, self.pred_horizon = (
            agent.obs_horizon, agent.act_horizon, agent.pred_horizon)
        self.act_dim = agent.act_dim

    def compute_loss(self, observations, native_actions):
        normalized = native_actions / self.action_scale
        if not torch.isfinite(normalized).all() or normalized.abs().max() > 1.00001:
            raise ValueError("normalized training actions outside [-1,1]")
        return self.agent.compute_loss(observations, normalized)

    def get_action(self, observations):
        normalized = self.agent.get_action(observations)
        native = normalized * self.action_scale
        if not torch.isfinite(native).all(): raise ValueError("decoded native action is non-finite")
        if (native.abs() > self.action_scale + 1e-5).any():
            raise ValueError("decoded action exceeds TRAIN-derived per-channel bounds")
        return native


def _read_scale(path, data_path, manifest_path):
    payload = json.loads(Path(path).read_text())
    contract = _contract(); expected = contract["training_data"]
    if (payload.get("format") != FORMAT or payload.get("method") != "per_channel_full_train_maxabs"
            or payload.get("addendum_sha256") != sha256_file(ADDENDUM)
            or payload.get("dataset_h5_sha256") != sha256_file(data_path)
            or payload.get("dataset_json_sha256") != sha256_file(manifest_path)
            or payload.get("dataset_manifest_sha256") != sha256_file(Path(manifest_path).with_name("dataset_manifest.json"))
            or payload.get("dataset_h5_sha256") != expected["hdf5_sha256"]
            or payload.get("dataset_json_sha256") != expected["json_sha256"]
            or payload.get("dataset_manifest_sha256") != expected["dataset_manifest_sha256"]):
        raise ValueError("scale statistics do not bind the admitted TRAIN package and v2 addendum")
    scale = np.asarray(payload.get("scale"), dtype=np.float32)
    if (scale.shape != (7,) or not np.isfinite(scale).all() or (scale <= 0).any()
            or payload.get("episode_count") != 1000 or payload.get("action_row_count") != expected["action_rows"]):
        raise ValueError("scale statistics must contain seven positive finite values")
    with h5py.File(data_path,"r") as h5:
        if len(h5)!=1000: raise ValueError("scale statistics do not cover all TRAIN episodes")
        rows=np.concatenate([np.asarray(h5[f"traj_{i}/actions"][:],dtype=np.float32) for i in range(1000)],axis=0)
    actual_scale=np.maximum(np.max(np.abs(rows),axis=0).astype(np.float32),np.float32(1e-6))
    actual_sha=hashlib.sha256(np.ascontiguousarray(rows).tobytes()).hexdigest()
    if (rows.shape!=(expected["action_rows"],7) or not np.array_equal(scale,actual_scale)
            or actual_sha!=payload.get("full_action_rows_sha256")):
        raise ValueError("scale statistics are not the exact full-row TRAIN maxabs fit")
    return payload, scale


def _build_scaled_agent(scale, device):
    return ScaledAgent(base.build_agent(device), scale).to(device)


def train(data_path, manifest_path, catalog_dirs, scale_path, exp_dir, ckpt_dir,
          *, device="cuda", _updates=100000, _batch_size=1024, agent_factory=None):
    if (_updates, _batch_size) != (100000, 1024):
        raise ValueError("v2 production budget is fixed at 100000 updates and batch 1024")
    addendum = _contract()
    scale_payload, scale = _read_scale(scale_path, data_path, manifest_path)
    data_hashes = [sha256_file(data_path), sha256_file(manifest_path),
                   sha256_file(Path(manifest_path).with_name("dataset_manifest.json"))]
    dataset = base.PlugBaseDataset(data_path, manifest_path)
    if set(catalog_dirs) != set(base.SPLITS): raise ValueError("all four frozen catalog splits are required")
    catalog_root = Path(next(iter(catalog_dirs.values()))).resolve().parent
    if any(Path(path).resolve()!=(catalog_root/name).resolve() for name,path in catalog_dirs.items()):
        raise ValueError("all v2 catalogs must be the registered siblings in one frozen catalog set")
    catalog_set_path=catalog_root/"catalog_set.json"
    catalog_set=json.loads(catalog_set_path.read_text())
    if (catalog_set.get("format")!="plug-orientation-catalog-set-v1"
            or sha256_file(catalog_set_path)!=addendum["training_data"]["catalog_set_sha256"]):
        raise ValueError("v2 training requires the admitted frozen catalog set")
    for split, directory in catalog_dirs.items():
        catalog = base.PlugSceneCatalog(directory)
        digest=sha256_file(Path(directory)/"manifest.json")
        if (catalog.manifest["split"] != split or len(catalog) != len(base.SPLITS[split])
                or catalog_set.get("catalogs",{}).get(split,{}).get("manifest_sha256")!=digest):
            raise ValueError(f"invalid frozen catalog {split}")
    exp_dir, ckpt_dir = Path(exp_dir), Path(ckpt_dir)
    exp_dir.mkdir(parents=True, exist_ok=False); ckpt_dir.mkdir(parents=True, exist_ok=False)
    archive = base._archive_base_sources(exp_dir)
    for source in (Path(__file__), ADDENDUM):
        rel = source.relative_to(base.ROOT)
        target = exp_dir / "source_archive" / rel
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(source.read_bytes())
        archive["source_sha256"][str(rel)] = sha256_file(source)
        if sha256_file(target) != archive["source_sha256"][str(rel)]: raise RuntimeError("v2 archive mismatch")
    archived_scale=exp_dir/"source_archive"/"scale_statistics.json"
    archived_scale.write_bytes(Path(scale_path).read_bytes())
    if sha256_file(archived_scale)!=sha256_file(scale_path): raise RuntimeError("v2 scale archive mismatch")
    if data_hashes != [sha256_file(data_path), sha256_file(manifest_path),
                       sha256_file(Path(manifest_path).with_name("dataset_manifest.json"))]:
        raise ValueError("TRAIN data changed before training")
    cfg = resolved_config()
    provenance = {"format": FORMAT, "model_family": FORMAT, "normalization_method": "per_channel_full_train_maxabs",
        "scale_statistics_path": str(Path(scale_path).resolve()), "scale_statistics_sha256": sha256_file(scale_path),
        "action_scale": scale.tolist(), "addendum_sha256": sha256_file(ADDENDUM),
        "base_contract_sha256": sha256_file(BASE_CONTRACT),
        "contract_sha256": sha256_file(ADDENDUM),
        "dataset_h5_path": str(Path(data_path).resolve()), "dataset_h5_sha256": data_hashes[0],
        "dataset_json_path": str(Path(manifest_path).resolve()), "dataset_json_sha256": data_hashes[1],
        "dataset_manifest_path": str(Path(manifest_path).with_name("dataset_manifest.json").resolve()),
        "dataset_manifest_sha256": data_hashes[2], "dataset_episode_count": 1000,
        "full_action_rows_sha256": scale_payload["full_action_rows_sha256"], **archive}
    catalog_hashes = {name: sha256_file(Path(path) / "manifest.json") for name, path in catalog_dirs.items()}
    catalog_set = catalog_root / "catalog_set.json"
    provenance.update(frozen_scene_catalog_sha256=catalog_hashes, catalog_set_path=str(catalog_set),
                      catalog_set_sha256=sha256_file(catalog_set))
    prereg={"format":FORMAT,"config":cfg,"addendum_sha256":sha256_file(ADDENDUM),
        "base_contract_sha256":sha256_file(BASE_CONTRACT),"scale_statistics_sha256":sha256_file(scale_path),
        "training_data_sha256":data_hashes,"catalog_manifest_sha256":catalog_hashes,
        "created_before_first_update":True}
    (exp_dir/"preregistration.json").write_text(json.dumps(prereg,indent=2,sort_keys=True)+"\n")
    (exp_dir / "resolved_config.json").write_text(json.dumps({**cfg, **provenance}, indent=2, sort_keys=True)+"\n")
    seed_device = torch.device(device); base._seed_everything(1, seed_device)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=0)
    agent = (agent_factory(scale, device) if agent_factory else _build_scaled_agent(scale, device)).to(device)
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4, betas=(.95,.999), weight_decay=1e-6)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=500, num_training_steps=100000)
    ema = EMAModel(agent.parameters(), power=.75)
    start=time.monotonic(); update=0
    while update < 100000:
        seen=False
        for batch in loader:
            if update == 100000: break
            if time.monotonic()-start >= 21600: raise TimeoutError("v2 training exceeded registered wall cap")
            seen=True; obs=batch["observations"].to(device); actions=batch["actions"].to(device)
            loss=agent.compute_loss(obs, actions)
            if not torch.isfinite(loss): raise FloatingPointError("non-finite v2 loss")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in agent.parameters()):
                raise FloatingPointError("non-finite v2 gradient")
            optimizer.step(); scheduler.step(); ema.step(agent.parameters()); update+=1
            if update % 100 == 0:
                with (exp_dir/"training.jsonl").open("a") as f:
                    f.write(json.dumps({"update":update,"loss":float(loss.detach().cpu()),"lr":optimizer.param_groups[0]["lr"],"elapsed_seconds":time.monotonic()-start},allow_nan=False)+"\n")
            if update in base.CHECKPOINT_STEPS:
                base._save_checkpoint(ckpt_dir/f"checkpoint_{update:06d}.pt",agent,ema,optimizer,scheduler,
                                      update,cfg,provenance,format_id=FORMAT)
        if not seen: raise ValueError("v2 training loader empty")
    return {"updates":update,"elapsed_seconds":time.monotonic()-start,"exp_dir":str(exp_dir),"ckpt_dir":str(ckpt_dir)}


def _validate_checkpoint(path, scale_path=None):
    p=torch.load(path,map_location="cpu",weights_only=False)
    if (p.get("format") != FORMAT or p.get("model_family") != FORMAT or p.get("config") != resolved_config()
            or p.get("checkpoint_role") != "validation_candidate" or int(p.get("update",-1)) not in base.CHECKPOINT_STEPS):
        raise ValueError("v2 checkpoint family/config/update mismatch")
    prov=p.get("dataset_provenance",{})
    if (prov.get("contract_sha256") != sha256_file(ADDENDUM) or prov.get("addendum_sha256") != sha256_file(ADDENDUM)
            or prov.get("base_contract_sha256") != sha256_file(BASE_CONTRACT)):
        raise ValueError("v2 checkpoint contract identity mismatch")
    expected=_contract()["training_data"]
    if (prov.get("dataset_h5_sha256")!=expected["hdf5_sha256"]
            or prov.get("dataset_json_sha256")!=expected["json_sha256"]
            or prov.get("dataset_manifest_sha256")!=expected["dataset_manifest_sha256"]
            or prov.get("catalog_set_sha256")!=expected["catalog_set_sha256"]):
        raise ValueError("v2 checkpoint does not bind the admitted training/catalog identities")
    for pk,hk in (("dataset_h5_path","dataset_h5_sha256"),("dataset_json_path","dataset_json_sha256"),
                  ("dataset_manifest_path","dataset_manifest_sha256"),("catalog_set_path","catalog_set_sha256"),
                  ("scale_statistics_path","scale_statistics_sha256")):
        path0=prov.get(pk); digest=prov.get(hk)
        if not path0 or not digest or not Path(path0).is_file() or sha256_file(path0)!=digest:
            raise ValueError(f"v2 checkpoint {pk} provenance mismatch")
    if scale_path is not None and (Path(scale_path).resolve()!=Path(prov["scale_statistics_path"]).resolve()
                                   or sha256_file(scale_path)!=prov["scale_statistics_sha256"]):
        raise ValueError("v2 checkpoint scale statistics identity mismatch")
    scale=np.asarray(prov.get("action_scale"),dtype=np.float32)
    if scale.shape!=(7,) or not np.isfinite(scale).all() or (scale<=0).any(): raise ValueError("invalid v2 scale")
    stats,scale_from_stats=_read_scale(prov["scale_statistics_path"],prov["dataset_h5_path"],prov["dataset_json_path"])
    if (not np.array_equal(scale_from_stats,scale)
            or stats.get("full_action_rows_sha256")!=prov.get("full_action_rows_sha256")):
        raise ValueError("v2 checkpoint scale statistics do not match checkpoint provenance")
    # Validate both persistent buffers before strict load can replace a fresh
    # wrapper's buffer with untrusted serialized state.
    expected_scale = torch.as_tensor(scale, dtype=torch.float32)
    for state_name in ("agent", "ema_agent"):
        state = p.get(state_name)
        buffer = state.get("action_scale") if isinstance(state, dict) else None
        if (not isinstance(buffer, torch.Tensor) or buffer.dtype != torch.float32
                or tuple(buffer.shape) != (7,) or not torch.isfinite(buffer).all()
                or (buffer <= 0).any() or not torch.equal(buffer, expected_scale)):
            raise ValueError(f"v2 checkpoint {state_name} action_scale buffer mismatch")
    return p,scale


class ScaledAdapter:
    def __init__(self, checkpoint, device="cpu"):
        p,scale=_validate_checkpoint(checkpoint)
        self.device=torch.device(device); self.agent=_build_scaled_agent(scale,device)
        self.agent.load_state_dict(p["ema_agent"],strict=True); self.agent.eval()
        self.metadata={"format":FORMAT,"model_family":FORMAT,"normalization_method":"per_channel_full_train_maxabs",
                       "checkpoint":str(checkpoint),"checkpoint_sha256":sha256_file(checkpoint),"action_scale":scale.tolist()}
    def get_action(self, observations):
        obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        if obs.shape!=(1,2,46) or not torch.isfinite(obs).all(): raise ValueError("expected finite (1,2,46) observations")
        with torch.no_grad(): action=self.agent.get_action(obs)
        if action.shape!=(1,8,7) or not torch.isfinite(action).all(): raise ValueError("invalid decoded native action chunk")
        return action


def _validate_eval_catalog(checkpoint, catalog_dir, split):
    p,_=_validate_checkpoint(checkpoint)
    catalog=base.PlugSceneCatalog(catalog_dir)
    if catalog.manifest.get("split")!=split or len(catalog)!=(50 if split=="fixed_validation" else 100):
        raise ValueError("v2 evaluation requires the complete registered catalog")
    prov=p["dataset_provenance"]; expected=prov.get("frozen_scene_catalog_sha256",{}).get(split)
    actual=sha256_file(Path(catalog_dir)/"manifest.json")
    if expected!=actual: raise ValueError("v2 evaluation catalog identity mismatch")
    root=Path(prov["catalog_set_path"]).parent
    bundle=json.loads(Path(prov["catalog_set_path"]).read_text())
    if (Path(catalog_dir).resolve()!=(root/split).resolve()
            or bundle.get("catalogs",{}).get(split,{}).get("manifest_sha256")!=actual):
        raise ValueError("v2 evaluation catalog bundle identity mismatch")
    return p


def evaluate_validation(checkpoint_dir, catalog_dir, output_root, *, policy_factory=None, env_factory=None):
    paths=[Path(checkpoint_dir)/f"checkpoint_{s:06d}.pt" for s in base.CHECKPOINT_STEPS]
    if any(not p.is_file() for p in paths) or len({sha256_file(p) for p in paths})!=5:
        raise ValueError("v2 validation requires five distinct registered checkpoints")
    for path,step in zip(paths,base.CHECKPOINT_STEPS):
        p,_=_validate_checkpoint(path)
        if int(p["update"])!=step: raise ValueError("v2 checkpoint update mismatch")
        _validate_eval_catalog(path,catalog_dir,"fixed_validation")
    out=Path(output_root); out.mkdir(parents=True,exist_ok=False); rows=[]; start=time.monotonic()
    for step,path in zip(base.CHECKPOINT_STEPS,paths):
        if time.monotonic()-start>=18000: raise TimeoutError("v2 validation exceeded 18000 seconds")
        validated,_=_validate_checkpoint(path)
        summary=base.evaluate_checkpoint(path,catalog_dir,out/f"validation_{step:06d}",
            policy_factory=policy_factory or (lambda p:ScaledAdapter(p)),env_factory=env_factory,
            execution_stride=8,_validated_payload=validated)
        summary.update(model_family=FORMAT,normalization_method="per_channel_full_train_maxabs")
        sp=out/f"validation_{step:06d}"/"summary.json"; sp.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
        rows.append({"update":step,"successes":summary["successes"],"episodes":summary["episodes"],"checkpoint":str(path),
            "checkpoint_sha256":sha256_file(path),"catalog_split":"fixed_validation",
            "catalog_manifest_sha256":summary["catalog_manifest_sha256"],"evaluation_summary_path":str(sp),
            "evaluation_summary_sha256":sha256_file(sp),"model_family":FORMAT})
    chosen=max(rows,key=lambda r:(r["successes"],r["update"]))
    payload={"format":"plug-orientation-scaled-validation-selection-v2","model_family":FORMAT,
        "normalization_method":"per_channel_full_train_maxabs","validation_successes":chosen["successes"],
        "validation_episodes":50,"selected_update":chosen["update"],"selected_checkpoint":chosen["checkpoint"],
        "selected_checkpoint_sha256":chosen["checkpoint_sha256"],"validation_catalog_manifest_sha256":rows[0]["catalog_manifest_sha256"],
        "candidate_results":rows}
    payload["selection_sha256"]=base._canonical_sha256(payload)
    (out/"validation_selection.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return {"candidates":rows,"selection":payload,"test_authorized":chosen["successes"]>=40}


def _validate_v2_selection(checkpoint, selection):
    body={k:v for k,v in selection.items() if k!="selection_sha256"}
    candidates=selection.get("candidate_results",[])
    if (selection.get("format")!="plug-orientation-scaled-validation-selection-v2" or selection.get("model_family")!=FORMAT
            or selection.get("normalization_method")!="per_channel_full_train_maxabs"
            or selection.get("validation_episodes")!=50 or int(selection.get("validation_successes",-1))<40
            or selection.get("selection_sha256")!=base._canonical_sha256(body)
            or sorted(r.get("update") for r in candidates)!=list(base.CHECKPOINT_STEPS)
            or len({r.get("checkpoint_sha256") for r in candidates})!=5
            or selection.get("selected_checkpoint_sha256")!=sha256_file(checkpoint)):
        raise ValueError("fixed test requires a valid v2 selection with at least 40/50")
    if any(not 0<=int(row.get("successes",-1))<=50 or int(row.get("episodes",-1))!=50
           or row.get("model_family")!=FORMAT or row.get("catalog_split")!="fixed_validation"
           for row in candidates):
        raise ValueError("v2 selection candidates must be complete family-bound 50-scene results")
    chosen=max(candidates,key=lambda r:(int(r["successes"]),int(r["update"])))
    if (int(selection["validation_successes"])!=int(chosen["successes"])
            or int(selection["selected_update"])!=int(chosen["update"])
            or selection.get("selected_checkpoint")!=chosen.get("checkpoint")
            or selection.get("selected_checkpoint_sha256")!=chosen.get("checkpoint_sha256")):
        raise ValueError("v2 selection does not implement the frozen max-success/latest-tie rule")
    selected_payload,_=_validate_checkpoint(checkpoint)
    expected_validation=selected_payload["dataset_provenance"].get("frozen_scene_catalog_sha256",{}).get("fixed_validation")
    if selection.get("validation_catalog_manifest_sha256")!=expected_validation or len({r.get("checkpoint_sha256") for r in candidates})!=5:
        raise ValueError("v2 selection is not bound to five distinct candidates and frozen validation catalog")
    for row in candidates:
        path=Path(row.get("checkpoint",""))
        if (not path.is_file() or path.name!=f"checkpoint_{int(row['update']):06d}.pt"
                or sha256_file(path)!=row.get("checkpoint_sha256")
                or row.get("catalog_manifest_sha256")!=expected_validation):
            raise ValueError("v2 selection includes a changed checkpoint or foreign validation catalog")
        payload,_=_validate_checkpoint(path)
        if (int(payload["update"])!=int(row["update"])
                or payload["dataset_provenance"].get("frozen_scene_catalog_sha256",{}).get("fixed_validation")!=expected_validation):
            raise ValueError("v2 candidate checkpoint identity differs from selection")
        summary_path=Path(row.get("evaluation_summary_path",""))
        if not summary_path.is_file() or sha256_file(summary_path)!=row.get("evaluation_summary_sha256"):
            raise ValueError("v2 validation summary identity mismatch")
        summary=json.loads(summary_path.read_text())
        if (summary.get("model_family")!=FORMAT or summary.get("normalization_method")!="per_channel_full_train_maxabs"
                or summary.get("episodes")!=50 or summary.get("successes")!=int(row["successes"])
                or summary.get("execution_stride")!=8 or summary.get("checkpoint_sha256")!=row["checkpoint_sha256"]
                or summary.get("catalog_manifest_sha256")!=expected_validation):
            raise ValueError("v2 validation result is not a complete matching family summary")
    return selected_payload


def evaluate_fixed_test(checkpoint, catalog_dir, selection_path, output_dir, *, policy_factory=None, env_factory=None):
    selection=json.loads(Path(selection_path).read_text())
    p=_validate_v2_selection(checkpoint,selection)
    p=_validate_eval_catalog(checkpoint,catalog_dir,"fixed_base_test")
    # Reuse only the physical rollout loop; v2 gates and identities above remain authoritative.
    summary=base.evaluate_checkpoint(checkpoint,catalog_dir,output_dir,
        policy_factory=policy_factory or (lambda p:ScaledAdapter(p)),env_factory=env_factory,
        _authorized_test=True,execution_stride=8,_validated_payload=p,
        _prevalidated_selection=True)
    summary.update(model_family=FORMAT,normalization_method="per_channel_full_train_maxabs",
        base_qualification_gate={"required_successes":80,"denominator":100,"passed":summary["successes"]>=80})
    Path(output_dir,"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    fit=sub.add_parser("fit-scale")
    fit.add_argument("--data",required=True); fit.add_argument("--manifest",required=True); fit.add_argument("--output",required=True)
    tr=sub.add_parser("train-scaled")
    tr.add_argument("--data",required=True); tr.add_argument("--manifest",required=True); tr.add_argument("--scale",required=True)
    tr.add_argument("--catalog-root",required=True); tr.add_argument("--exp-dir",required=True); tr.add_argument("--ckpt-dir",required=True)
    tr.add_argument("--device",default="cuda")
    val=sub.add_parser("evaluate-validation-scaled")
    val.add_argument("--ckpt-dir",required=True); val.add_argument("--catalog",required=True); val.add_argument("--output-root",required=True)
    test=sub.add_parser("evaluate-fixed-test-scaled")
    test.add_argument("--checkpoint",required=True); test.add_argument("--catalog",required=True)
    test.add_argument("--selection",required=True); test.add_argument("--output-dir",required=True)
    args=parser.parse_args(argv)
    if args.command=="fit-scale": fit_action_scale(args.data,args.manifest,args.output)
    elif args.command=="train-scaled":
        root=Path(args.catalog_root); train(args.data,args.manifest,{name:root/name for name in base.SPLITS},args.scale,
            args.exp_dir,args.ckpt_dir,device=args.device)
    elif args.command=="evaluate-validation-scaled": evaluate_validation(args.ckpt_dir,args.catalog,args.output_root)
    else: evaluate_fixed_test(args.checkpoint,args.catalog,args.selection,args.output_dir)


if __name__=="__main__": main()
