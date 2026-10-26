"""Manifest-ID reader for the matched Round-21b PlaceSphere package.

The historical formal readers deliberately retain their fixed 1,000/100 IDs.
This opt-in reader selects classes from manifest IDs and independently checks
the HDF5 labels, hashes, dimensions, and native terminal evidence.
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.constrain.placesphere_base_dp import sha256_file

MATCHED_H5_SHA256 = "eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0"
MATCHED_MANIFEST_SHA256 = "ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86"
MATCHED_VERSION = "placesphere_matched_terminal_r21b_v1"


class PlaceSphereManifestDataset(Dataset):
    """Validated class-selected trajectories and official 2/16 action windows."""

    def __init__(self, data_path, manifest_path, selected_class, *,
                 expected_h5_sha256=MATCHED_H5_SHA256,
                 expected_manifest_sha256=MATCHED_MANIFEST_SHA256,
                 obs_horizon=2, pred_horizon=16, expected_count=None):
        if selected_class not in ("direct", "obstacle"):
            raise ValueError(f"unsupported PlaceSphere class {selected_class!r}")
        self.data_path, self.manifest_path = Path(data_path), Path(manifest_path)
        self.selected_class = selected_class
        self.h5_sha256 = sha256_file(self.data_path)
        self.manifest_sha256 = sha256_file(self.manifest_path)
        if self.h5_sha256 != expected_h5_sha256:
            raise ValueError("matched PlaceSphere HDF5 identity mismatch")
        if self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("matched PlaceSphere manifest identity mismatch")

        manifest = json.loads(self.manifest_path.read_text())
        records = manifest.get("episodes")
        if not isinstance(records, list):
            raise ValueError("matched manifest must contain episode records")
        by_id = {}
        for record in records:
            if "episode_id" not in record or "kind" not in record:
                raise ValueError("matched manifest episode is missing episode_id or kind")
            episode_id = int(record["episode_id"])
            if episode_id in by_id:
                raise ValueError(f"duplicate manifest episode_id {episode_id}")
            by_id[episode_id] = record
        if set(by_id) != set(range(1300)):
            raise ValueError("matched manifest must cover episode IDs 0..1299 exactly")
        selected = [eid for eid, record in by_id.items()
                    if record["kind"] == selected_class]
        other = "obstacle" if selected_class == "direct" else "direct"
        if any(record["kind"] not in ("direct", "obstacle") for record in records):
            raise ValueError("matched manifest contains an unknown class label")
        if len(selected) != (expected_count or (1000 if selected_class == "direct" else 300)):
            raise ValueError(f"matched manifest has wrong {selected_class} episode count")
        if len([r for r in records if r["kind"] == other]) != (300 if selected_class == "direct" else 1000):
            raise ValueError("matched manifest class counts do not cover 1,000/300")

        self.trajectory_ids = selected
        self.trajectories_obs, self.trajectories_raw, self.trajectories_act = [], [], []
        self.trajectory_lengths, self.slices = [], []
        with h5py.File(self.data_path, "r") as handle:
            ids = {int(key.removeprefix("traj_")) for key in handle.keys()}
            if ids != set(range(1300)) or len(handle) != 1300:
                raise ValueError("matched HDF5 must contain traj_0..traj_1299 exactly")
            for trajectory_index, episode_id in enumerate(selected):
                record, group = by_id[episode_id], handle[f"traj_{episode_id}"]
                if record["kind"] != selected_class or group.attrs.get("kind") != selected_class:
                    raise ValueError(f"traj_{episode_id}: manifest/HDF5 class mismatch")
                obs, raw, actions = group["obs"][:], group["raw_obs"][:], group["actions"][:]
                if (actions.ndim != 2 or actions.shape[1] != 7 or
                        obs.shape != (len(actions) + 1, 39) or
                        raw.shape != (len(actions) + 1, 42)):
                    raise ValueError(f"traj_{episode_id}: invalid 39/42 observation or 7-action shape")
                if (not np.isfinite(obs).all() or not np.isfinite(raw).all() or
                        not np.isfinite(actions).all() or np.max(np.abs(actions), initial=0) > 1.00001):
                    raise ValueError(f"traj_{episode_id}: non-finite or illegal native data")
                if not np.array_equal(obs, raw[:, :39]):
                    raise ValueError(f"traj_{episode_id}: policy state does not match raw-state selection")
                metadata = json.loads(group.attrs.get("metadata_json", "{}"))
                terminal = metadata.get("expert_record", {})
                if (not bool(group["success"][-1]) or not bool(group["terminated"][-1]) or
                        bool(group["truncated"][:].any()) or
                        not bool(terminal.get("native_success")) or
                        not bool(terminal.get("terminated")) or bool(terminal.get("truncated"))):
                    raise ValueError(f"traj_{episode_id}: missing native successful terminal evidence")
                infos = record.get("infos", [])
                if (not infos or not bool(infos[-1].get("native_success", [False])[-1]) or
                        any(bool(step.get(k, [False])[-1]) for step in infos
                            for k in ("obstacle_collision", "obstacle_crossing")) or
                        bool(terminal.get("obstacle_contact")) or bool(terminal.get("obstacle_crossing"))):
                    raise ValueError(f"traj_{episode_id}: missing or unsafe full-prefix audit")
                self.trajectories_obs.append(torch.as_tensor(obs, dtype=torch.float32))
                self.trajectories_raw.append(torch.as_tensor(raw, dtype=torch.float32))
                self.trajectories_act.append(torch.as_tensor(actions, dtype=torch.float32))
                length = len(actions)
                self.trajectory_lengths.append(length)
                before, after = obs_horizon - 1, pred_horizon - obs_horizon
                self.slices.extend((trajectory_index, start, start + pred_horizon)
                                   for start in range(-before, length - pred_horizon + after))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start, end = self.slices[index]
        obs, raw, actions = (self.trajectories_obs[trajectory],
                             self.trajectories_raw[trajectory],
                             self.trajectories_act[trajectory])
        length = self.trajectory_lengths[trajectory]
        obs_seq = obs[max(0, start):start + 2]
        raw_seq = raw[max(0, start):start + 2]
        act_seq = actions[max(0, start):min(end, length)]
        if start < 0:
            obs_seq = torch.cat((obs_seq[:1].repeat(-start, 1), obs_seq))
            raw_seq = torch.cat((raw_seq[:1].repeat(-start, 1), raw_seq))
            act_seq = torch.cat((act_seq[:1].repeat(-start, 1), act_seq))
        if end > length:
            pad = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=act_seq.dtype)
            act_seq = torch.cat((act_seq, pad.repeat(end - length, 1)))
        if obs_seq.shape != (2, 39) or raw_seq.shape != (2, 42) or act_seq.shape != (16, 7):
            raise AssertionError("matched PlaceSphere official window invariant failed")
        return {"observations": obs_seq, "raw_observations": raw_seq, "actions": act_seq}
