import numpy as np
import sapien.core as sapien
import inspect
from transforms3d.euler import euler2quat

from examples.baselines.constrain import plug_orientation_expert_candidate as candidate
from examples.baselines.constrain import plug_orientation_preflight as preflight


def test_grasp_tilt_is_applied_about_grasp_local_y():
    base = sapien.Pose([0.2, -0.1, 0.4], euler2quat(0.3, -0.2, 0.7))
    unchanged = candidate.apply_grasp_tilt(base, 0)
    tilted = candidate.apply_grasp_tilt(base, 15)
    expected = base * sapien.Pose(q=euler2quat(0, np.deg2rad(15), 0))
    assert np.allclose(unchanged.p, base.p)
    assert np.allclose(unchanged.q, base.q)
    assert np.allclose(tilted.p, expected.p)
    assert np.allclose(tilted.q, expected.q)


def test_candidate_keeps_official_default_tilt():
    assert candidate.DEFAULT_GRASP_TILT_DEG == 15


def test_collection_can_preregister_matched_seed_and_tilt_without_changing_default():
    assert preflight.scene_plan("fixed")[0] == {"seed": 2026113000, "target_yaw": 0.0}
    assert inspect.signature(preflight.collect).parameters["expert_grasp_tilt_deg"].default is None
