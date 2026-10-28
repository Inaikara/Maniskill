import copy

import numpy as np
import pytest

from examples.baselines.constrain.pickcube_observation_norm import (
    DIRECT_IDS, PROFILE, fit_direct_observations,
)
from examples.baselines.constrain.pickcube_zscore_round37 import (
    BASE_FORMAT, CONSTRAIN_FORMAT, CONSTRAIN_UPDATES, DATA_H5_SHA,
    DATA_MANIFEST_SHA, DIFFUSION, CONSTRAIN_CONFIG, TRAINING, _base_contract,
    POLICY, _validate_control, prepare_control_observation,
)
from examples.baselines.constrain.pickcube_base_dp import MODEL_CONFIG, contract


def stats_fixture():
    source = {"selected_class": "direct", "selected_episode_ids": DIRECT_IDS,
              "h5_sha256": DATA_H5_SHA, "manifest_sha256": DATA_MANIFEST_SHA}
    episodes = {i: np.zeros((1, 45), np.float32) for i in DIRECT_IDS}
    return fit_direct_observations(episodes, source_identity=source)


def valid_base(stats):
    return {
        "format": BASE_FORMAT, "profile": PROFILE, "contract": contract(),
        "model_config": dict(MODEL_CONFIG), "normalizer": stats,
        "normalizer_sha256": stats["sha256"],
        "normalization": {"observation_profile": PROFILE,
                          "observation_stats_sha256": stats["sha256"],
                          "action": "identity native seven-channel"},
        "action_encoding": None, "action_encoding_identity": True,
        "data_identity": {"selected_class": "direct", "selected_episode_ids": DIRECT_IDS,
                           "h5_sha256": DATA_H5_SHA, "manifest_sha256": DATA_MANIFEST_SHA,
                           "episodes": 1000, "windows": 83004},
        "selected_weights": "ema", "step": 30000,
        "training": dict(TRAINING, updates=30000),
    }


def test_typed_base_checkpoint_accepts_matching_profile_and_rejects_legacy_or_wrong_stats():
    stats = stats_fixture()
    assert _base_contract(valid_base(stats), stats)["profile"] == PROFILE
    wrong = valid_base(stats); wrong["normalizer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Base checkpoint contract"):
        _base_contract(wrong, stats)
    legacy = valid_base(stats); legacy["format"] = "pickcube_pilot_base_v1"
    with pytest.raises(ValueError, match="Base checkpoint contract"):
        _base_contract(legacy, stats)


def test_typed_control_checkpoint_rejects_mismatched_profile_hash_or_data(tmp_path):
    stats = stats_fixture()
    base_path = tmp_path / "base.pt"
    base_path.write_bytes(b"registered base bytes")
    base_hash = __import__("hashlib").sha256(base_path.read_bytes()).hexdigest()
    config = {
        "profile": PROFILE,
        "normalization": {"observation_profile": PROFILE,
                          "observation_stats_sha256": stats["sha256"],
                          "observation_stats": stats},
        "model": copy.deepcopy(CONSTRAIN_CONFIG), "diffusion": copy.deepcopy(DIFFUSION),
        "policy": POLICY,
        "model_action_space": "native identity coordinates", "action_encoding": None,
        "action_encoding_identity": True,
        "data_identity": {"selected_class": "obstacle",
                          "selected_episode_ids": list(range(1000, 1500)),
                          "h5_sha256": DATA_H5_SHA, "manifest_sha256": DATA_MANIFEST_SHA,
                          "episodes": 500, "windows": 45840},
        "base_weights": {"sha256": base_hash, "subkey": "weights.ema", "updates": 30000},
        "training": dict(TRAINING, updates=CONSTRAIN_UPDATES), "selected_weights": "ema",
    }
    checkpoint = {"format": CONSTRAIN_FORMAT, "config": config,
                  "step": CONSTRAIN_UPDATES, "selected_weights": "ema"}
    assert _validate_control(checkpoint, stats, base_path)["format"] == CONSTRAIN_FORMAT
    bad = copy.deepcopy(checkpoint); bad["config"]["normalization"]["observation_stats_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Constrain checkpoint contract"):
        _validate_control(bad, stats, base_path)
    bad = copy.deepcopy(checkpoint); bad["config"]["data_identity"]["selected_episode_ids"][0] = 999
    with pytest.raises(ValueError, match="Constrain checkpoint contract"):
        _validate_control(bad, stats, base_path)


def test_shared_control_input_seam_normalizes_42_once_and_builds_hint_from_raw_45d():
    stats = stats_fixture()
    raw = np.zeros((2, 45), np.float32)
    raw[:, 19:21] = (0.1, 0.2)
    raw[:, 42:45] = (0.4, 0.3, 0.03)
    encoded, hint = prepare_control_observation(raw, stats)
    assert encoded.shape == (2, 42)
    assert encoded[0, 19].item() == pytest.approx(10.0)
    assert hint.shape == (3,)
    np.testing.assert_allclose(hint.numpy(), [0.6, 0.2, 0.375], atol=1e-6)
