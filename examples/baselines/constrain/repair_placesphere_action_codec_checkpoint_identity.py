"""Create metadata-corrected copies without changing trained model tensors."""

import argparse
import copy
import json
from pathlib import Path

import torch

from examples.baselines.constrain.placesphere_action_codec import CONTROLLER_RANGE
from examples.baselines.constrain.placesphere_base_dp import sha256_file
from examples.baselines.constrain.placesphere_manifest_dataset import MATCHED_HOLD8_OBSTACLE500_VERSION


OLD_BASE_SHA256 = "08879f41d1524a30f5aba9bfaaf2f107a5417b5fdc4d7463c8e274c7edda41cc"
OLD_CONTROL_SHA256 = "476d582a5993d55a80382dffec24fe7059bf3621b0ed2bede9a2f74166c345ff"
DATA_H5_SHA256 = "796c7a65f4ac3d9b8f434f71764ab393eec4257c217a79d4351f338e78993961"
DATA_MANIFEST_SHA256 = "aebbb0d79eed376b8c6a79e10b297751ee51077d168e03ceb12db643358f9fc9"
DATA_IDENTITY_SHA256 = "648f59863a747d603ec8dc6ccf056995e60a93f35b1875bf0b230a261f28fe78"
OLD_DATASET_VERSION = "placesphere_matched_terminal_r23_hold8_v1"


def _state_sha(state):
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def correct_metadata(base_source, control_source, base_output, control_output, record_path):
    if sha256_file(base_source) != OLD_BASE_SHA256 or sha256_file(control_source) != OLD_CONTROL_SHA256:
        raise ValueError("source checkpoint identity differs from the retained R30 training outputs")
    base = torch.load(base_source, map_location="cpu", weights_only=False)
    control = torch.load(control_source, map_location="cpu", weights_only=False)
    old_data = base.get("data_identity", {})
    if (base.get("step") != 30000 or base.get("action_encoding") != CONTROLLER_RANGE
            or old_data.get("dataset_version") != OLD_DATASET_VERSION
            or old_data.get("selected_class") != "direct" or old_data.get("episodes") != 1000
            or old_data.get("h5_sha256") != DATA_H5_SHA256
            or old_data.get("manifest_sha256") != DATA_MANIFEST_SHA256
            or old_data.get("training_identity_sha256") != DATA_IDENTITY_SHA256
            or old_data.get("selected_episode_ids") != [*range(0, 600, 2), *range(600, 1300)]):
        raise ValueError("Base checkpoint's actual frozen-data identity does not match the correction")
    if (control.get("step") != 150000
            or control.get("config", {}).get("data_identity", {}).get("dataset_version") != MATCHED_HOLD8_OBSTACLE500_VERSION
            or control.get("config", {}).get("base_weights", {}).get("state_sha256") != _state_sha({
                key.removeprefix("noise_pred_net."): value
                for key, value in base["weights"]["ema"].items()
                if key.startswith("noise_pred_net.")})):
        raise ValueError("Constrain checkpoint is not the exact matched 150k model from this Base EMA")
    corrected_base = copy.deepcopy(base)
    corrected_base["data_identity"]["dataset_version"] = MATCHED_HOLD8_OBSTACLE500_VERSION
    corrected_base_path, corrected_control_path = Path(base_output), Path(control_output)
    if corrected_base_path.exists() or corrected_control_path.exists():
        raise FileExistsError("metadata-corrected checkpoint target already exists")
    corrected_base_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_control_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_base = corrected_base_path.with_name(corrected_base_path.name + ".tmp")
    tmp_control = corrected_control_path.with_name(corrected_control_path.name + ".tmp")
    torch.save(corrected_base, tmp_base)
    control["config"]["base_weights"]["source"] = str(corrected_base_path)
    control["config"]["base_weights"]["sha256"] = sha256_file(tmp_base)
    torch.save(control, tmp_control)
    tmp_base.replace(corrected_base_path)
    tmp_control.replace(corrected_control_path)
    # Reload and establish that only metadata changed; all tensor payloads remain exact.
    base_check = torch.load(corrected_base_path, map_location="cpu", weights_only=False)
    control_check = torch.load(corrected_control_path, map_location="cpu", weights_only=False)
    if (_state_sha(base["weights"]["ema"]) != _state_sha(base_check["weights"]["ema"])
            or _state_sha(control["model"]) != _state_sha(control_check["model"])
            or any(not torch.equal(a, b) for a, b in zip(
                control["ema"]["shadow_params"], control_check["ema"]["shadow_params"]))):
        raise AssertionError("checkpoint metadata correction changed model or EMA tensors")
    result = {
        "format": "placesphere_action_codec_checkpoint_metadata_correction_v1",
        "reason": "Base trainer serialized the prior 300-obstacle version constant although it selected the pinned 1000 direct IDs from the frozen 1000/500 HDF5; actual data hashes, identity hash, IDs and all policy tensors match.",
        "old_base": {"path": str(base_source), "sha256": OLD_BASE_SHA256,
                     "recorded_dataset_version": OLD_DATASET_VERSION},
        "corrected_base": {"path": str(corrected_base_path), "sha256": sha256_file(corrected_base_path),
                           "dataset_version": MATCHED_HOLD8_OBSTACLE500_VERSION},
        "old_constrain": {"path": str(control_source), "sha256": OLD_CONTROL_SHA256,
                          "base_ema_state_sha256": control["config"]["base_weights"]["state_sha256"]},
        "corrected_constrain": {"path": str(corrected_control_path), "sha256": sha256_file(corrected_control_path),
                                "base_source": str(corrected_base_path),
                                "base_sha256": control_check["config"]["base_weights"]["sha256"]},
        "tensor_identity": {"base_ema_state_sha256": _state_sha({
                                key.removeprefix("noise_pred_net."): value
                                for key, value in base["weights"]["ema"].items()
                                if key.startswith("noise_pred_net.")}),
                            "constrain_model_state_sha256": _state_sha(control["model"]),
                            "constrain_ema_shadow_params_unchanged": True},
        "data_identity": {"h5_sha256": DATA_H5_SHA256,
                          "manifest_sha256": DATA_MANIFEST_SHA256,
                          "training_identity_sha256": DATA_IDENTITY_SHA256,
                          "selected_direct_ids": 1000},
        "source_checkpoint_bytes_preserved": True,
    }
    Path(record_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--control-source", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--control-output", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(correct_metadata(args.base_source, args.control_source,
                                      args.base_output, args.control_output,
                                      args.record), indent=2), flush=True)


if __name__ == "__main__":
    main()
