"""Strict post-training Phase C loader for the typed R44 horizon-8 Control EMA."""

import json
from pathlib import Path

import torch

from examples.baselines.constrain import pickcube_body_hint_round41 as r41
from examples.baselines.constrain import pickcube_zscore_round37 as engine
from examples.baselines.constrain.pickcube_base_dp import sha256_file
from examples.baselines.constrain.pickcube_clearance_train_r44 import (
    TRAIN_REG, TRAIN_SCHEMA, TRAIN_FORMAT, MODEL_KIND, TOTAL, EMA_POWER,
    BASE, FIT, PRECHECK, STATS_PATH, digest,
)
from examples.baselines.constrain.pickcube_clearance_round43 import DATA, registered_assets


def load_actor(checkpoint, stats_path=STATS_PATH, device="cpu"):
    """Require exact persisted training/data/normalizer identities before loading."""
    registration=json.loads(TRAIN_REG.read_text())
    body={k:v for k,v in registration.items() if k!="registration_sha256"}
    if (registration.get("schema")!=TRAIN_SCHEMA or registration.get("model_kind")!=MODEL_KIND
            or registration.get("horizon")!=8 or registration.get("registration_sha256")!=digest(body)):
        raise ValueError("R44 training registration type/checksum mismatch")
    for path,expected in registration["source_files"].items():
        if sha256_file(path)!=expected:
            raise ValueError(f"R44 immutable training source changed: {path}")
    if registration.get("data")!=DATA or registration.get("asset_hashes")!=registered_assets():
        raise ValueError("R44 frozen data/geometry identity mismatch")
    if (registration.get("fit_sha256")!=sha256_file(FIT)
            or registration.get("precheck_sha256")!=sha256_file(PRECHECK)
            or registration.get("base_checkpoint",{}).get("sha256")!=sha256_file(BASE)):
        raise ValueError("R44 response/precheck/R40 Base source identity mismatch")
    reg_profile=registration.get("profile",{})
    if sha256_file(stats_path)!=reg_profile.get("normalizer_sha256"):
        raise ValueError("R44 normalizer file identity mismatch")
    stats=engine.load_stats(stats_path,expected_sha=reg_profile["normalizer_sha256"])
    ck=torch.load(checkpoint,map_location="cpu",weights_only=False);cfg=ck.get("config",{})
    expected=(ck.get("format")==TRAIN_FORMAT and cfg.get("format")==TRAIN_FORMAT
        and cfg.get("model_kind")==MODEL_KIND and cfg.get("horizon")==8
        and ck.get("step")==TOTAL and ck.get("ema") is not None
        and cfg.get("training_registration_sha256")==registration["registration_sha256"]
        and cfg.get("precheck_sha256")==sha256_file(PRECHECK)
        and cfg.get("response_fit_sha256")==sha256_file(FIT)
        and cfg.get("base_weights",{}).get("sha256")==sha256_file(BASE)
        and cfg.get("normalization",{}).get("observation_stats_sha256")==stats.get("sha256"))
    if not expected:
        raise ValueError("R44 EMA checkpoint rejected: horizon-8 typed metadata mismatch")
    if (cfg.get("normalization",{}).get("observation_stats")!=stats
            or cfg.get("typed_hint_metadata")!=r41.body_hint_metadata()
            or cfg.get("training_augmentation")!=r41.r40.METADATA
            or cfg.get("selected_weights")!="ema"
            or cfg.get("clearance_objective")!=registration.get("objective")):
        raise ValueError("R44 EMA profile/action/objective metadata mismatch")
    r41._install_profile()
    from examples.baselines.constrain.pickcube_body_hint_round41 import CONTROL_CONFIG
    from diffusers.training_utils import EMAModel
    model=engine.build_constrain_denoiser(CONTROL_CONFIG,device="cpu")
    model.load_state_dict(ck["model"],strict=True)
    ema=EMAModel(parameters=model.trainable_parameters(),power=EMA_POWER)
    ema.load_state_dict(ck["ema"]);ema.copy_to(model.trainable_parameters())
    model.to(device).eval()
    return model,ck,stats
