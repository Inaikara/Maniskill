"""Independent native-contract Base DP for PickCubeObstacle.

The network is the official ManiSkill ConditionalUnet1D architecture, freshly
initialized for this 42d/7d task. It never imports StackCube weights.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from examples.baselines.constrain.base_dp import BasePolicyArgs, _PolicyEnvStub

ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DP = ROOT / "examples" / "baselines" / "diffusion_policy"
FORMAL_DIR = ROOT / "demos/PickCube-obstacle/20261022_132753_formal"
FORMAL_H5 = FORMAL_DIR / "trajectory.h5"
FORMAL_MANIFEST = FORMAL_DIR / "trajectory.json"
FORMAL_CONTRACT = FORMAL_DIR / "frozen_contract.json"
FORMAL_REPORT = FORMAL_DIR / "report.json"
MODEL_CONFIG = {
    "obs_dim": 42, "obstacle_obs_dim": 45, "act_dim": 7,
    "obs_horizon": 2, "pred_horizon": 16, "act_horizon": 8,
    "diffusion_step_embed_dim": 64, "down_dims": [64, 128, 256],
    "n_groups": 8, "num_diffusion_iters": 100,
}
NAMED_FIELDS = [
    ["agent/qpos", 9], ["agent/qvel", 9], ["extra/is_grasped", 1],
    ["extra/tcp_pose", 7], ["extra/goal_pos", 3], ["extra/obj_pose", 7],
    ["extra/tcp_to_obj_pos", 3], ["extra/obj_to_goal_pos", 3],
    ["extra/pickcube_obstacle", 3],
]
FORMAT = "pickcube_base_dp_v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract():
    return {
        "policy": {"robot": "panda", "obs_dim": 42, "obstacle_obs_dim": 45,
                   "act_dim": 7, "obs_horizon": 2, "pred_horizon": 16,
                    "act_horizon": 8, "control_mode": "pd_ee_delta_pose",
                    "observation_mode": "state", "observation_fields": NAMED_FIELDS,
                    "obstacle_feature_semantics": "pickcube_obstacle world XY center and radius",
                    "action_fields": ["delta_x", "delta_y", "delta_z", "delta_euler_x",
                                      "delta_euler_y", "delta_euler_z", "gripper"],
                    "translation_scale_m": 0.1,
                    "rotation_scale": "signed controller XYZ Euler range [-0.1,0.1] rad",
                    "action_bounds": [-1.0, 1.0]},
        "normalization": {"observation": "identity (native state)", "action": "identity; validated native values in [-1,1]"},
        "diffusion": {"prediction_type": "epsilon", "scheduler": "DDPMScheduler",
                      "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
                      "num_train_timesteps": 100},
        "model": dict(MODEL_CONFIG),
    }


def verify_runtime_observation_contract():
    """Resolve the named flat observation layout from the live PickCube env."""
    from examples.baselines.constrain.pickcube_expert import make_pickcube_env
    from mani_skill.utils.common import flatten_state_dict

    env = make_pickcube_env()
    try:
        obs, _ = env.reset(seed=20261022)
        named = env.unwrapped._get_obs_state_dict(env.unwrapped.get_info())
        flat = flatten_state_dict(named, use_torch=True)
        extra = named["extra"]
        obstacle = extra["pickcube_obstacle"]
        fields = {}

        def describe(node, prefix=""):
            for key, value in node.items():
                name = f"{prefix}/{key}" if prefix else key
                if isinstance(value, dict):
                    describe(value, name)
                else:
                    fields[name] = list(value.shape[1:])

        describe(named)
        observed_fields = [[name, int(np.prod(shape)) if shape else 1]
                           for name, shape in fields.items()]
        if observed_fields != NAMED_FIELDS:
            raise ValueError(f"runtime named observation fields changed: {observed_fields}")
        obstacle_count = int(obstacle.shape[-1])
        if tuple(obs.shape) != (1, 45) or flat.shape[-1] != 45 or obstacle_count != 3:
            raise ValueError(f"runtime PickCube named observation contract mismatch: obs={tuple(obs.shape)} fields={fields}")
        if not torch.equal(flat[0, -obstacle_count:], obstacle[0]):
            raise ValueError("named pickcube_obstacle fields are not the final three flat channels")
        if flat.shape[-1] - obstacle_count != 42:
            raise ValueError("runtime PickCube base observation is not 42d")
        return {"flat_dim": int(flat.shape[-1]), "base_dim": 42,
                "obstacle_field": "extra/pickcube_obstacle", "obstacle_dim": obstacle_count,
                "named_field_shapes": fields, "field_order": list(fields),
                "runtime_observation_sha256": hashlib.sha256(flat.detach().cpu().numpy().tobytes()).hexdigest()}
    finally:
        env.close()


def build_agent(device="cpu"):
    if str(OFFICIAL_DP) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DP))
    from train import Agent

    args = BasePolicyArgs(obs_dim=42, act_dim=7, obs_horizon=2, act_horizon=8,
                          pred_horizon=16, diffusion_step_embed_dim=64, n_groups=8)
    args.unet_dims = [64, 128, 256]
    return Agent(_PolicyEnvStub(42, 7, 2), args).to(device)


class PickCubeBaseDataset(Dataset):
    """Official DP windows from exactly the manifest's 1,000 direct episodes."""

    def __init__(self, data_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST,
                 obs_horizon=2, pred_horizon=16, expected_obs_dim=45):
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        manifest = json.loads(Path(manifest_path).read_text())
        episodes = manifest["episodes"]
        episode_ids = [int(e["episode_id"]) for e in episodes]
        if sorted(episode_ids) != list(range(1100)) or len(set(episode_ids)) != 1100:
            raise ValueError("formal manifest episode IDs must cover 0..1099 exactly once")
        by_id = {int(e["episode_id"]): e for e in episodes}
        direct = [int(e["episode_id"]) for e in episodes if e["kind"] == "direct"]
        if len(direct) != 1000 or sum(e["kind"] == "obstacle" for e in episodes) != 100:
            raise ValueError("formal manifest must contain exactly 1000 direct and 100 obstacle episodes")
        self.trajectory_ids = direct
        self.trajectories_obs, self.trajectories_act, self.trajectory_lengths = [], [], []
        self.slices = []
        with h5py.File(data_path, "r") as handle:
            if len(handle) != 1100:
                raise ValueError(f"formal HDF5 must contain 1100 episodes, found {len(handle)}")
            for episode_id in direct:
                key = f"traj_{episode_id}"
                group = handle[key]
                if group.attrs.get("kind") != "direct":
                    raise ValueError(f"{key}: HDF5 class metadata does not match direct manifest selection")
                obs, actions = group["obs"][:], group["actions"][:]
                if obs.ndim != 2 or obs.shape[1] != expected_obs_dim:
                    raise ValueError(f"{key}: observation layout mismatch: {obs.shape}")
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError(f"{key}: native action schema mismatch: {actions.shape}")
                if obs.shape[0] != len(actions) + 1:
                    raise ValueError(f"{key}: expected T+1 observations")
                if not np.isfinite(obs).all() or not np.isfinite(actions).all():
                    raise ValueError(f"{key}: non-finite training data")
                if "success" not in group or not bool(group["success"][-1]):
                    raise ValueError(f"{key}: missing native successful terminal evidence")
                if "terminated" not in group or not bool(group["terminated"][-1]) or bool(group["truncated"][:].any()):
                    raise ValueError(f"{key}: direct demonstration did not terminate successfully")
                if np.max(np.abs(actions)) > 1.00001:
                    raise ValueError(f"{key}: action outside native normalized bounds")
                record = by_id[episode_id]
                if not record.get("terminated", [0])[-1] or record.get("truncated", [1])[-1]:
                    raise ValueError(f"{key}: manifest lacks native successful terminal")
                self.trajectories_obs.append(torch.as_tensor(obs, dtype=torch.float32))
                self.trajectories_act.append(torch.as_tensor(actions, dtype=torch.float32))
                length = len(actions)
                self.trajectory_lengths.append(length)
                before, after = obs_horizon - 1, pred_horizon - obs_horizon
                self.slices.extend((len(self.trajectory_lengths)-1, start, start+pred_horizon)
                                   for start in range(-before, length-pred_horizon+after))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        trajectory, start, end = self.slices[index]
        obs, actions = self.trajectories_obs[trajectory], self.trajectories_act[trajectory]
        length = len(actions)
        obs_seq = obs[max(0, start):start+self.obs_horizon]
        act_seq = actions[max(0, start):end]
        if start < 0:
            obs_seq = torch.cat((obs_seq[0].repeat(-start, 1), obs_seq))
            act_seq = torch.cat((act_seq[0].repeat(-start, 1), act_seq))
        if end > length:
            # Official delta-pose padding holds still in all six pose channels
            # and preserves the final gripper command.
            pad = torch.cat((torch.zeros(6), act_seq[-1, 6:7]))
            act_seq = torch.cat((act_seq, pad.repeat(end-length, 1)))
        if obs_seq.shape != (self.obs_horizon, 45) or act_seq.shape != (self.pred_horizon, 7):
            raise AssertionError("official PickCube window/padding invariant failed")
        # Training condition deliberately excludes the three named obstacle fields.
        return {"observations": obs_seq[..., :42], "actions": act_seq}


class PickCubeBaseDPAdapter:
    """Strict standalone inference and ConstrainNet base-network seam."""

    @staticmethod
    def validate_checkpoint(checkpoint):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        schema = ck.get("named_state_schema", {})
        data = ck.get("data_identity", {})
        if (ck.get("format") != FORMAT or ck.get("contract") != contract()
                or schema.get("base_dim") != 42 or schema.get("flat_dim") != 45
                or schema.get("obstacle_field") != "extra/pickcube_obstacle"
                or schema.get("field_order") != [field[0] for field in NAMED_FIELDS]
                or data.get("selected_class") != "direct" or data.get("episodes") != 1000
                or len(data.get("selected_episode_ids", [])) != 1000
                or not data.get("h5_sha256") or not data.get("manifest_sha256")):
            raise ValueError("PickCube checkpoint contract mismatch")
        return ck

    def __init__(self, checkpoint, device="cpu", weights="ema"):
        ck = self.validate_checkpoint(checkpoint)
        self.agent = build_agent(device)
        self.agent.load_state_dict(ck["weights"][weights], strict=True)
        self.agent.eval()
        self.device = torch.device(device)
        self.obs_horizon = 2
        self.obs_dim = 42
        self.obstacle_obs_dim = 45
        self.metadata = {"checkpoint": str(checkpoint), "weights": weights,
                         "contract": ck["contract"], "checkpoint_sha256": sha256_file(checkpoint)}

    def base_cond(self, queued_obs):
        x = torch.as_tensor(queued_obs)
        if x.shape[-2:] != (2, 45):
            raise ValueError(f"expected queued (2,45) obstacle observations, got {tuple(x.shape)}")
        x = x[..., :42]
        return x.flatten(start_dim=x.ndim - 2)

    def get_action(self, obs_seq):
        obs = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device)
        if obs.shape[-2:] != (2, 42):
            raise ValueError(f"expected (batch,2,42) base conditions, got {tuple(obs.shape)}")
        with torch.no_grad():
            return self.agent.get_action(obs)

    def initialize_constrain_base(self, denoiser):
        """Strictly initialize a task-matched ConstrainDenoiser base network."""
        target = getattr(denoiser, "base_net", None)
        if target is None:
            raise TypeError("ConstrainNet denoiser must expose its base_net")
        target.load_state_dict(self.agent.noise_pred_net.state_dict(), strict=True)
        return target

    def map_action_ee_delta_pose(self, action):
        action = torch.as_tensor(action)
        if action.shape[-1] != 7 or not torch.isfinite(action).all():
            raise ValueError("expected finite native seven-channel actions")
        if bool((action.abs() > 1.000001).any()):
            raise ValueError("native seven-channel actions must remain in [-1,1]")
        return action
