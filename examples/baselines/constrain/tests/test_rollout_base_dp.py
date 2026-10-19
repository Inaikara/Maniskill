"""Tests for the shared final-step info seam used by the rollout and
evaluation runners (Ticket 02 corrective round).

Seam under test: :func:`unwrap_final_info` — gymnasium 0.29
``SyncVectorEnv`` auto-resets on the step where the episode ends: the
top-level info becomes the *reset* info of the fresh episode and the real
final step info moves to ``info['final_info'][0]``, guarded by the
``_final_info`` mask. Flags occurring exactly on the last step must be read
from there; normal non-terminal info must pass through unchanged.
"""

import numpy as np

from examples.baselines.constrain.rollout_base_dp import unwrap_final_info


def test_unwrap_final_info_passes_normal_nonterminal_info_through():
    info = {"success": np.array([False]), "obstacle_collision": np.array([True])}
    assert unwrap_final_info(info) is info


def test_unwrap_final_info_returns_the_masked_final_step_info():
    final = {
        # raw unbatched (0-dimensional) single-env values, as produced by
        # the CPUGymWrapper step info that SyncVectorEnv moves into final_info
        "success": np.asarray(True),
        "stack_success": np.asarray(True),
        "obstacle_collision": np.asarray(True),
        "obstacle_crossing": np.asarray(False),
    }
    info = {
        # reset info of the fresh episode reported at the top level
        "success": np.array([False]),
        "stack_success": np.array([False]),
        "obstacle_collision": np.array([False]),
        "obstacle_crossing": np.array([False]),
        "final_info": np.array([final], dtype=object),
        "_final_info": np.array([True]),
    }
    unwrapped = unwrap_final_info(info)
    assert unwrapped is not info
    assert bool(unwrapped["success"][0]) is True
    assert bool(unwrapped["stack_success"][0]) is True
    assert bool(unwrapped["obstacle_collision"][0]) is True
    assert bool(unwrapped["obstacle_crossing"][0]) is False


def test_unwrap_final_info_respects_empty_final_info_mask():
    info = {
        "final_info": np.array([None], dtype=object),
        "_final_info": np.array([False]),
    }
    assert unwrap_final_info(info) is info
