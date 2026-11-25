import numpy as np
import pytest

from examples.baselines.constrain.peg_orientation_near_hole_hold import MinimalConstantHoldAdapter
from examples.baselines.constrain.peg_orientation_near_hole_hold import prepare_output_directory


def test_constant_hold_returns_batched_zero_pose_and_closed_gripper():
    actions = MinimalConstantHoldAdapter().get_action(np.zeros((1, 2, 39), dtype=np.float32))
    assert actions.shape == (1, 8, 7)
    np.testing.assert_array_equal(actions[0, :, :6], np.zeros((8, 6), dtype=np.float32))
    np.testing.assert_array_equal(actions[0, :, 6], -np.ones(8, dtype=np.float32))


def test_constant_hold_rejects_wrong_actual_history_shape():
    with pytest.raises(ValueError, match="history shape"):
        MinimalConstantHoldAdapter().get_action(np.zeros((2, 39), dtype=np.float32))


def test_runner_accepts_precreated_attempt_workspace(tmp_path):
    (tmp_path / "attempt_001.sh").write_text("frozen wrapper\n")
    (tmp_path / "regression_pre.stdout.log").write_text("test evidence\n")
    assert prepare_output_directory(tmp_path) == tmp_path


def test_runner_refuses_to_overwrite_prior_attempt_artifacts(tmp_path):
    (tmp_path / "attempt_001.sh").write_text("frozen wrapper\n")
    (tmp_path / "preregistration.json").write_text("prior registration\n")
    with pytest.raises(FileExistsError, match="prior run artifact"):
        prepare_output_directory(tmp_path)
