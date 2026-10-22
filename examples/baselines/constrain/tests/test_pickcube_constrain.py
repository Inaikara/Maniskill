"""PickCube ConstrainNet public contract tests (Ticket 04)."""

import pytest
import torch

from examples.baselines.constrain.pickcube_base_dp import (
    FORMAL_H5,
    FORMAL_MANIFEST,
    PickCubeBaseDPAdapter,
)
from examples.baselines.constrain.pickcube_constrain import (
    PickCubeConditionNorm,
    PickCubeConstrainAdapter,
    PickCubeConstrainTrainer,
    equivalence_evidence,
    load_pickcube_constrain,
    pickcube_relative_condition,
)


def state(tcp=(0.10, -0.05), obstacle=(0.20, 0.10, 0.03)):
    # Named-field order: qpos(9), qvel(9), grasp(1), tcp_pose(7), goal(3),
    # object_pose(7), tcp->object(3), object->goal(3), obstacle(3).
    obs = torch.zeros(45)
    obs[19:21] = torch.tensor(tcp)
    obs[42:45] = torch.tensor(obstacle)
    return obs


def test_condition_is_translation_invariant_and_uses_latest_native_frame():
    old, latest = state(), state((0.05, 0.02), (0.25, 0.17, 0.04))
    shifted = latest.clone()
    shifted[19:21] += torch.tensor([0.4, -0.2])
    shifted[42:44] += torch.tensor([0.4, -0.2])
    cfg = PickCubeConditionNorm()
    condition = pickcube_relative_condition(torch.stack([old, latest]), cfg)
    shifted_condition = pickcube_relative_condition(torch.stack([old, shifted]), cfg)
    # rel XY scale 0.5m -> normalized delta/radius scale 0.08m.
    torch.testing.assert_close(condition, torch.tensor([0.4, 0.3, 0.5]))
    torch.testing.assert_close(shifted_condition, condition)


def test_native_action_mapping_retains_all_seven_channels():
    action = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7]])
    torch.testing.assert_close(PickCubeConstrainAdapter.map_action(action), action)
    with pytest.raises(ValueError, match="seven"):
        PickCubeConstrainAdapter.map_action(action[:, :4])


def test_pickcube_trainer_contract_and_strict_data_provenance():
    trainer = PickCubeConstrainTrainer(
        demo_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST,
        base_checkpoint="outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt",
    )
    assert trainer.model_config["input_dim"] == 7
    assert trainer.policy_contract["act_dim"] == 7
    assert trainer.policy_contract["control_mode"] == "pd_ee_delta_pose"
    assert trainer.load_dataset().num_trajectories == 100


def test_checkpoint_contract_mutation_is_rejected(tmp_path):
    checkpoint = tmp_path / "valid.pt"
    trainer = PickCubeConstrainTrainer(
        demo_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST,
        base_checkpoint="outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt", device="cpu",
    )
    trainer.setup()
    trainer.step = 30000
    trainer.save_checkpoint(checkpoint)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ck["config"]["policy"]["control_mode"] = "pd_ee_delta_pos"
    bad = tmp_path / "bad.pt"
    torch.save(ck, bad)
    with pytest.raises(ValueError, match="contract"):
        load_pickcube_constrain(bad)


def test_base_checkpoint_requires_exact_ids_and_trusted_hdf5_identity():
    base = "outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt"
    assert PickCubeBaseDPAdapter.validate_checkpoint(
        base, expected_episode_ids=list(range(1000)),
        trusted_h5_sha256="fa846f52b5ae08de0243cfa52b40a4b0e7ce160881aa9db59ebaf4952c02904e",
    )["step"] == 30000
    with pytest.raises(ValueError, match="episode IDs"):
        PickCubeBaseDPAdapter.validate_checkpoint(base, expected_episode_ids=list(range(1, 1001)))
    with pytest.raises(ValueError, match="HDF5 identity"):
        PickCubeBaseDPAdapter.validate_checkpoint(base, trusted_h5_sha256="0" * 64)


def test_zero_residual_and_training_update_preserve_base(tmp_path):
    trainer = PickCubeConstrainTrainer(
        demo_path=FORMAL_H5, manifest_path=FORMAL_MANIFEST,
        base_checkpoint="outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt",
        device="cpu", batch_size=2, total_iters=2, run_dir=tmp_path,
    )
    trainer.setup()
    equivalence = equivalence_evidence(
        trainer.denoiser,
        "outputs/ckpt/20261022_144000_pickcube_base_dp/final.pt",
        device="cpu",
    )
    assert equivalence["epsilon_matches"] and equivalence["sampling_matches"]
    before = {k: v.clone() for k, v in trainer.denoiser.base_net.state_dict().items()}
    sample = torch.randn(2, 16, 7)
    cond = torch.randn(2, 84)
    hint = torch.randn(2, 16, 3)
    with torch.no_grad():
        base = trainer.denoiser.base_net(sample, 13, global_cond=cond)
        initial = trainer.denoiser(sample, 13, global_cond=cond, hint=hint)
    torch.testing.assert_close(initial, base)
    trainer.train_steps(2)
    assert any(not torch.equal(a, b) for a, b in zip(
        trainer.denoiser.trainable_parameters(), trainer.initial_control_parameters
    ))
    assert all(torch.equal(before[k], v) for k, v in trainer.denoiser.base_net.state_dict().items())
