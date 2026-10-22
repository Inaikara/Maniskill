"""PlaceSphere-only ConstrainNet contract and shared-condition tests."""

import pytest
import torch

from examples.baselines.constrain.placesphere_constrain import (
    BASE_CKPT, FORMAT, MODEL, POLICY, PlaceSphereConstrainAdapter,
    NORM, TRUSTED_BASE_SHA256, TRUSTED_CONTRACT_SHA256, TRUSTED_H5_SHA256,
    TRUSTED_MANIFEST_SHA256, TRUSTED_REPORT_SHA256, PlaceSphereObstacleDataset, load_placesphere_constrain,
    placesphere_hint,
)


def state(tcp=(0.10, -0.05), obstacle=(0.20, 0.10, 0.03)):
    value = torch.zeros(42)
    value[19:21] = torch.tensor(tcp)  # named tcp_pose offset from 9+9+1
    value[39:42] = torch.tensor(obstacle)
    return value


def test_hint_uses_latest_named_placesphere_fields_and_fixed_normalization():
    queued = torch.stack((state(), state((0.05, 0.02), (0.25, 0.17, 0.04))))
    result = placesphere_hint(queued)
    torch.testing.assert_close(result, torch.tensor([0.4, 0.3, 0.5]))
    shifted = queued.clone()
    shifted[-1, 19:21] += torch.tensor([0.4, -0.2])
    shifted[-1, 39:41] += torch.tensor([0.4, -0.2])
    torch.testing.assert_close(placesphere_hint(shifted), result)


def test_hint_rejects_pickcube_schema_and_nonfinite_values():
    with pytest.raises(ValueError, match="42"):
        placesphere_hint(torch.zeros(2, 45))
    malformed = torch.zeros(2, 42)
    malformed[-1, 39] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        placesphere_hint(malformed)


def test_placesphere_contract_keeps_native_euler_and_action_channels():
    assert MODEL["input_dim"] == 7 and MODEL["global_cond_dim"] == 78
    assert POLICY["base_obs_dim"] == 39 and POLICY["raw_obs_dim"] == POLICY["obstacle_obs_dim"] == 42
    action = torch.tensor([[.1, -.2, .3, .2, -.3, .1, -1.]])
    torch.testing.assert_close(PlaceSphereConstrainAdapter.map_action(action), action)
    with pytest.raises(ValueError, match="seven"):
        PlaceSphereConstrainAdapter.map_action(action[:, :4])


def test_formal_training_data_identity_is_strict():
    data = PlaceSphereObstacleDataset()
    assert data.trajectory_ids == list(range(1000, 1100))
    assert len(data) > 100
    item = data[0]
    assert item["observations"].shape == (2, 42)
    assert item["actions"].shape == (16, 7)
    terminal = data[-1]["actions"]
    torch.testing.assert_close(terminal[-1], torch.tensor([0., 0., 0., 0., 0., 0., 1.]))


@pytest.mark.parametrize("mutation", ["format", "task", "schema", "data", "control"])
def test_checkpoint_rejects_task_schema_data_and_control_contract_mutations(tmp_path, mutation):
    config = {"model": dict(MODEL), "policy": dict(POLICY), "normalization": dict(NORM),
              "diffusion": {"scheduler": "DDPMScheduler", "beta_schedule": "squaredcos_cap_v2", "clip_sample": True,
                            "prediction_type": "epsilon", "num_train_timesteps": 100},
              "data_identity": {"h5_sha256": TRUSTED_H5_SHA256, "manifest_sha256": TRUSTED_MANIFEST_SHA256,
                                "frozen_contract_sha256": TRUSTED_CONTRACT_SHA256,
                                "collection_report_sha256": TRUSTED_REPORT_SHA256,
                                "selected_episode_ids": list(range(1000, 1100)), "selected_class": "obstacle", "episodes": 100},
              "base_weights": {"sha256": TRUSTED_BASE_SHA256, "subkey": "weights.ema", "updates": 30000},
              "training": {"seed": 1, "total_iters": 30000, "batch_size": 1024, "lr": 1e-4,
                           "warmup": 500, "ema_power": .75, "selected_weights": "final EMA fixed before training"}}
    ck = {"format": FORMAT, "step": 30000, "model": {}, "config": config}
    if mutation == "format": ck["format"] = "pickcube_constrain_v1"
    if mutation == "task": ck["config"]["policy"]["robot"] = "xarm6"
    if mutation == "schema": ck["config"]["policy"]["base_obs_dim"] = 42
    if mutation == "data": ck["config"]["data_identity"]["selected_episode_ids"] = list(range(999, 1099))
    if mutation == "control": ck["config"]["policy"]["control_mode"] = "pd_ee_delta_pos"
    path = tmp_path / "bad.pt"
    torch.save(ck, path)
    with pytest.raises(ValueError, match="contract"):
        load_placesphere_constrain(path)


def test_zero_residual_and_disabled_forward_match_independent_base():
    from examples.baselines.constrain.placesphere_constrain import _base_checkpoint, _state_sha
    from examples.baselines.constrain.placesphere_base_dp import build_agent
    from examples.baselines.constrain.constrain_denoiser import build_constrain_denoiser

    base_ck = _base_checkpoint(BASE_CKPT)
    independent = build_agent("cpu")
    independent.load_state_dict(base_ck["weights"]["ema"], strict=True)
    denoiser = build_constrain_denoiser(MODEL, device="cpu")
    denoiser.base_net.load_state_dict(independent.noise_pred_net.state_dict(), strict=True)
    assert _state_sha(denoiser.base_net.state_dict()) == _state_sha(independent.noise_pred_net.state_dict())
    torch.manual_seed(73)
    sample, cond, hint = torch.randn(1, 16, 7), torch.randn(1, 78), torch.randn(1, 16, 3)
    with torch.no_grad():
        expected = independent.noise_pred_net(sample, 41, global_cond=cond)
        initialized = denoiser(sample, 41, global_cond=cond, hint=hint)
        disabled = denoiser(sample, 41, global_cond=cond, enabled=False)
    torch.testing.assert_close(initialized, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(disabled, expected, atol=1e-6, rtol=1e-6)
