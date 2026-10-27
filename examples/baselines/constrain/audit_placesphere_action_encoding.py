"""Read-only native-action range, distribution, and diffusion-SNR audit."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from examples.baselines.constrain.placesphere_action_codec import CONTROLLER_RANGE
from examples.baselines.constrain.placesphere_base_dp import sha256_file
from examples.baselines.constrain.placesphere_manifest_dataset import (
    MATCHED_HOLD8_OBSTACLE500_VERSION, load_matched_training_identity,
)


FIELDS = ["delta_x", "delta_y", "delta_z", "delta_euler_x", "delta_euler_y",
          "delta_euler_z", "gripper"]
TIMESTEPS = [0, 10, 25, 50, 75, 99]


def audit(data_path, manifest_path, identity_path):
    identity = load_matched_training_identity(
        identity_path, data_path, manifest_path,
        expected_version=MATCHED_HOLD8_OBSTACLE500_VERSION)
    manifest = json.loads(Path(manifest_path).read_text())
    rows = {name: [] for name in ("direct", "obstacle")}
    with h5py.File(data_path, "r") as h5:
        for row in manifest["episodes"]:
            rows[row["kind"]].append(h5[f"traj_{int(row['episode_id'])}"]["actions"][:])
    class_stats = {}
    bounds = np.asarray(CONTROLLER_RANGE["scale"], dtype=np.float64)
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    scheduler = DDPMScheduler(num_train_timesteps=100,
                              beta_schedule="squaredcos_cap_v2",
                              prediction_type="epsilon")
    for kind, episodes in rows.items():
        native = np.concatenate(episodes, axis=0).astype(np.float64, copy=False)
        if native.ndim != 2 or native.shape[1] != 7 or not np.isfinite(native).all():
            raise ValueError(f"{kind}: malformed native actions")
        encoded = native / bounds
        if ((np.abs(native[:, :6]) > 0.350001).any()
                or (np.abs(native[:, 6]) > 1.000001).any()):
            raise ValueError(f"{kind}: frozen actions exceed fixed controller-range bounds")
        variance = np.var(encoded, axis=0)
        channel = {}
        for index, name in enumerate(FIELDS):
            channel[name] = {
                "native_min": float(native[:, index].min()),
                "native_max": float(native[:, index].max()),
                "abs_quantiles_native_q00_q50_q90_q99_q100": np.quantile(
                    np.abs(native[:, index]), [0, .5, .9, .99, 1]).tolist(),
                "rms_native": float(np.sqrt(np.mean(native[:, index] ** 2))),
                "zero_fraction_native": float(np.mean(native[:, index] == 0)),
                "encoded_rms": float(np.sqrt(np.mean(encoded[:, index] ** 2))),
                "encoded_zero_fraction": float(np.mean(encoded[:, index] == 0)),
                "encoded_variance": float(variance[index]),
            }
        snr = {}
        for timestep in TIMESTEPS:
            alpha_bar = float(scheduler.alphas_cumprod[timestep])
            snr[str(timestep)] = {
                "alpha_bar": alpha_bar,
                "per_channel_signal_to_noise_ratio": (
                    alpha_bar * variance / max(1 - alpha_bar, 1e-15)).tolist(),
            }
        class_stats[kind] = {"episodes": len(episodes), "action_rows": len(native),
                             "channels": channel, "effective_snr_by_timestep": snr}
    return {
        "format": "placesphere_native_action_codec_audit_v1",
        "codec_profile": CONTROLLER_RANGE,
        "dataset": {"h5_path": str(data_path), "h5_sha256": sha256_file(data_path),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "training_identity_path": str(identity_path),
                    "training_identity_sha256": sha256_file(identity_path),
                    "dataset_version": identity["dataset_version"],
                    "class_counts": identity["class_counts"]},
        "audit_passed": True,
        "bounds": {"pose_channels_native_normalized_controller_coordinates": [-.35, .35],
                   "gripper_native": [-1., 1.],
                   "actual_actions_clipped_or_modified": False},
        "class_statistics": class_stats,
        "effective_snr_note": "Per-channel variance signal-to-noise ratio under standard forward diffusion; descriptive, not causal proof.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.data, args.manifest, args.identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
