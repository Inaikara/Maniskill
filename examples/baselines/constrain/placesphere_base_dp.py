"""Independent native-contract Base DP for the PlaceSphere obstacle task."""

import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.constrain.base_dp import BasePolicyArgs, _PolicyEnvStub

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DP = ROOT / "examples/baselines/diffusion_policy"
FORMAL_DIR = ROOT / "demos/PlaceSphere-obstacle/20261022_184558_formal"
FORMAL_H5 = FORMAL_DIR / "trajectory.h5"
FORMAL_MANIFEST = FORMAL_DIR / "trajectory.json"
FORMAL_CONTRACT = FORMAL_DIR / "frozen_contract.json"
FORMAL_REPORT = FORMAL_DIR / "report.json"
TRUSTED_H5_SHA256 = "fec4620c496ae3c5c0dec7db469333f302f7170363e11f66cba961cbb5d56ec4"
TRUSTED_MANIFEST_SHA256 = "740379c6a575a470f930b249494cb603a1ff36e2c71c5235e93d6eb3ec4c11ba"
TRUSTED_CONTRACT_SHA256 = "ff11b940ede86e290928d26ccd4ac96e7ba480995af774d46a018dbda8b6e761"
TRUSTED_REPORT_SHA256 = "d861bf23e813e06e77f8791cd11e94844b000de8586f5169d33467f2c8df4b67"
MODEL_CONFIG = {"obs_dim": 39, "raw_obs_dim": 42, "obstacle_obs_dim": 42,
                "act_dim": 7, "obs_horizon": 2, "pred_horizon": 16,
                "act_horizon": 8, "diffusion_step_embed_dim": 64,
                "down_dims": [64, 128, 256], "n_groups": 8,
                "num_diffusion_iters": 100}
NAMED_FIELDS = [
    ["agent/qpos", 9], ["agent/qvel", 9], ["extra/is_grasped", 1],
    ["extra/tcp_pose", 7], ["extra/bin_pos", 3], ["extra/obj_pose", 7],
    ["extra/tcp_to_obj_pos", 3], ["extra/placesphere_obstacle", 3],
]
FORMAT = "placesphere_base_dp_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract():
    return {
        "policy": {"robot": "panda", **MODEL_CONFIG, "control_mode": "pd_ee_delta_pose",
                   "observation_mode": "state", "observation_fields": NAMED_FIELDS,
                   "policy_selection": "named fields excluding extra/placesphere_obstacle",
                   "action_fields": ["delta_x", "delta_y", "delta_z", "delta_euler_x",
                                     "delta_euler_y", "delta_euler_z", "gripper"],
                   "translation_scale_m": 0.1,
                   "rotation_scale": "signed controller XYZ Euler range [-0.35,0.35] rad",
                   "action_bounds": [-1.0, 1.0], "gripper_open": 1.0,
                   "gripper_closed": -1.0},
        "normalization": {"observation": "identity (native state)",
                          "action": "identity; native controller coordinates"},
        "diffusion": {"prediction_type": "epsilon", "scheduler": "DDPMScheduler",
                      "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
                      "num_train_timesteps": 100},
        "model": dict(MODEL_CONFIG),
        "window_padding": {"observation_history": "repeat first observation",
                            "action_tail_arm": "zero all six native pose coordinates",
                            "action_tail_gripper": "open +1 in native coordinate",
                            "terminal_actions": "retain every recorded action"},
    }


def verify_runtime_observation_contract():
    import gymnasium as gym
    import examples.baselines.constrain.placesphere_obstacle  # register task
    from mani_skill.utils.common import flatten_state_dict

    env = gym.make("PlaceSphereObstacle-v1", robot_uids="panda", control_mode="pd_ee_delta_pose",
                   obs_mode="state", sim_backend="physx_cpu", num_envs=1,
                   obstacle_enabled=False, reconfiguration_freq=1)
    try:
        env.reset(seed=20261022)
        named = env.unwrapped._get_obs_state_dict(env.unwrapped.get_info())
        flat = flatten_state_dict(named, use_torch=True)
        fields = []
        def describe(node, prefix=""):
            for key, value in node.items():
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    describe(value, name)
                else:
                    fields.append([name, int(np.prod(value.shape[1:])) or 1])
        describe(named)
        if fields != NAMED_FIELDS or tuple(flat.shape) != (1, 42):
            raise ValueError(f"runtime PlaceSphere named observation contract mismatch: {fields}, {tuple(flat.shape)}")
        policy = torch.cat((flat[:, :39],), dim=-1)
        if policy.shape[-1] != 39:
            raise ValueError("PlaceSphere policy state selection is not 39d")
        return {"flat_dim": 42, "base_dim": 39, "obstacle_field": "extra/placesphere_obstacle",
                "field_order": [field[0] for field in fields], "named_fields": fields}
    finally:
        env.close()


def build_agent(device="cpu"):
    if str(OFFICIAL_DP) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DP))
    from train import Agent
    args = BasePolicyArgs(obs_dim=39, act_dim=7, obs_horizon=2, act_horizon=8,
                          pred_horizon=16, diffusion_step_embed_dim=64, n_groups=8)
    args.unet_dims = [64, 128, 256]
    return Agent(_PolicyEnvStub(39, 7, 2), args).to(device)


class PlaceSphereBaseDataset(Dataset):
    """Official aligned windows from exactly the manifest's 1,000 direct episodes."""

    def __init__(self, data_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST,
                 obs_horizon=2, pred_horizon=16, expected_obs_dim=42):
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        manifest = json.loads(Path(manifest_path).read_text())
        episodes = manifest["episodes"]
        ids = [int(e["episode_id"]) for e in episodes]
        if sorted(ids) != list(range(1100)) or len(set(ids)) != 1100:
            raise ValueError("formal manifest episode IDs must cover 0..1099 exactly once")
        direct = [int(e["episode_id"]) for e in episodes if e["kind"] == "direct"]
        if direct != list(range(1000)) or sum(e["kind"] == "obstacle" for e in episodes) != 100:
            raise ValueError("formal manifest must select exactly direct IDs 0..999 and 100 obstacle episodes")
        by_id = {int(e["episode_id"]): e for e in episodes}
        self.trajectory_ids = direct
        self.trajectories_obs, self.trajectories_act, self.slices = [], [], []
        with h5py.File(data_path, "r") as handle:
            if len(handle) != 1100:
                raise ValueError(f"formal HDF5 must contain 1100 episodes, found {len(handle)}")
            for trajectory_index, episode_id in enumerate(direct):
                group = handle[f"traj_{episode_id}"]
                if group.attrs.get("kind") != "direct" or by_id[episode_id]["kind"] != "direct":
                    raise ValueError(f"traj_{episode_id}: selected episode is not direct")
                obs, raw, actions = group["obs"][:], group["raw_obs"][:], group["actions"][:]
                if obs.shape != (len(actions) + 1, 39) or raw.shape != (len(actions) + 1, expected_obs_dim):
                    raise ValueError(f"traj_{episode_id}: expected T+1 39d policy and 42d raw state")
                if not np.array_equal(obs, np.concatenate((raw[:, :39],), axis=1)):
                    raise ValueError(f"traj_{episode_id}: policy observation is not named raw selection")
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"traj_{episode_id}: expected native seven-channel actions")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all() or np.max(np.abs(actions)) > 1.00001:
                    raise ValueError(f"traj_{episode_id}: invalid non-finite or out-of-range data")
                if not bool(group["success"][-1]) or not bool(group["terminated"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"traj_{episode_id}: missing native successful terminal evidence")
                record = by_id[episode_id]
                if not record.get("terminated", [0])[-1] or record.get("truncated", [1])[-1]:
                    raise ValueError(f"traj_{episode_id}: manifest terminal evidence mismatch")
                self.trajectories_obs.append(torch.as_tensor(obs, dtype=torch.float32))
                self.trajectories_act.append(torch.as_tensor(actions, dtype=torch.float32))
                length = len(actions)
                before, after = obs_horizon - 1, pred_horizon - obs_horizon
                self.slices.extend((trajectory_index, start, start + pred_horizon)
                                   for start in range(-before, length - pred_horizon + after))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start, end = self.slices[index]
        obs, actions = self.trajectories_obs[trajectory], self.trajectories_act[trajectory]
        length = len(actions)
        obs_seq = obs[max(0, start):start + self.obs_horizon]
        act_seq = actions[max(0, start):min(end, length)]
        if start < 0:
            obs_seq = torch.cat((obs_seq[0].repeat(-start, 1), obs_seq))
            act_seq = torch.cat((act_seq[0].repeat(-start, 1), act_seq))
        if end > length:
            pad = torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype=act_seq.dtype)
            act_seq = torch.cat((act_seq, pad.repeat(end - length, 1)))
        if obs_seq.shape != (self.obs_horizon, 39) or act_seq.shape != (self.pred_horizon, 7):
            raise AssertionError("PlaceSphere official window/padding invariant failed")
        return {"observations": obs_seq, "actions": act_seq}


class PlaceSphereBaseDPAdapter:
    """Strict PlaceSphere checkpoint loader and base-condition/action interface."""

    @staticmethod
    def validate_checkpoint(checkpoint, expected_episode_ids=None, trusted_h5_sha256=None):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        data, schema = ck.get("data_identity", {}), ck.get("named_state_schema", {})
        matched = data.get("dataset_version") == "placesphere_matched_terminal_r21b_v1"
        expected_ids = data.get("selected_episode_ids") if matched else list(range(1000))
        if (ck.get("format") != FORMAT or ck.get("contract") != contract()
                or schema.get("base_dim") != 39 or schema.get("flat_dim") != 42
                or schema.get("obstacle_field") != "extra/placesphere_obstacle"
                or schema.get("field_order") != [field[0] for field in NAMED_FIELDS]
                or data.get("selected_class") != "direct" or data.get("episodes") != 1000
                or not isinstance(expected_ids, list) or len(expected_ids) != 1000
                or not data.get("h5_sha256") or not data.get("manifest_sha256")
                or (not matched and data.get("selected_episode_ids") != list(range(1000)))
                or (not matched and data.get("frozen_contract_sha256") != TRUSTED_CONTRACT_SHA256)
                or (matched and (data.get("h5_sha256") != "eecfcdac7e357334eb68921c1e784e6858c3ccdd11d69c17e4e79b5cbf4556d0"
                                 or data.get("manifest_sha256") != "ff971fb2b9278126c20d37a7b43a3f9ad74d0a86adc5ce926922b0ad9f796e86"))):
            raise ValueError("PlaceSphere checkpoint contract mismatch")
        if expected_episode_ids is not None and data["selected_episode_ids"] != list(expected_episode_ids):
            raise ValueError("PlaceSphere checkpoint selected episode IDs mismatch")
        if trusted_h5_sha256 is not None and data["h5_sha256"] != trusted_h5_sha256:
            raise ValueError("PlaceSphere checkpoint HDF5 identity mismatch")
        if trusted_h5_sha256 is None and not matched and data["h5_sha256"] != TRUSTED_H5_SHA256:
            raise ValueError("PlaceSphere checkpoint is not from the trusted formal HDF5")
        if not matched and data["manifest_sha256"] != TRUSTED_MANIFEST_SHA256:
            raise ValueError("PlaceSphere checkpoint formal manifest identity mismatch")
        if not matched and data.get("collection_report_sha256") != TRUSTED_REPORT_SHA256:
            raise ValueError("PlaceSphere checkpoint collection report identity mismatch")
        return ck

    def __init__(self, checkpoint, device="cpu", weights="ema"):
        ck = self.validate_checkpoint(checkpoint)
        self.agent = build_agent(device)
        self.agent.load_state_dict(ck["weights"][weights], strict=True)
        self.agent.eval()
        self.device, self.obs_horizon, self.obs_dim, self.raw_obs_dim = torch.device(device), 2, 39, 42
        self.metadata = {"checkpoint": str(checkpoint), "weights": weights,
                         "contract": ck["contract"], "checkpoint_sha256": sha256_file(checkpoint),
                         "native_execution_action_bounds": {"pose_channels": [-0.35, 0.35],
                                                             "gripper": [-1.0, 1.0]}}

    def base_cond(self, queued_obs):
        x = torch.as_tensor(queued_obs)
        if x.shape[-2:] != (2, 42):
            raise ValueError(f"expected queued (2,42) PlaceSphere observations, got {tuple(x.shape)}")
        return x[..., :39].flatten(start_dim=x.ndim - 2)

    def get_action(self, obs_seq):
        obs = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        if obs.shape[-2:] != (2, 39):
            raise ValueError(f"expected (batch,2,39) PlaceSphere conditions, got {tuple(obs.shape)}")
        with torch.no_grad():
            return self.agent.get_action(obs)

    def initialize_constrain_base(self, denoiser):
        target = getattr(denoiser, "base_net", None)
        if target is None:
            raise TypeError("task-matched denoiser must expose base_net")
        target.load_state_dict(self.agent.noise_pred_net.state_dict(), strict=True)
        return target

    @staticmethod
    def map_action_ee_delta_pose(action):
        action = torch.as_tensor(action)
        if action.shape[-1] != 7 or not torch.isfinite(action).all():
            raise ValueError("expected finite native seven-channel PlaceSphere actions")
        if bool((action[..., 6].abs() > 1.000001).any()):
            raise ValueError("native PlaceSphere gripper action must remain in [-1,1]")
        return torch.cat((action[..., :6].clamp(-0.35, 0.35), action[..., 6:7].clamp(-1, 1)), dim=-1)
