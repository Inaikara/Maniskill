import h5py
import numpy as np

from examples.baselines.constrain.materialize_pickcube_fullscene500 import write_tree


def test_native_state_series_writer_preserves_nested_t_plus_one_arrays(tmp_path):
    path = tmp_path / "states.h5"
    rows = [
        {"actors": {"cube": np.asarray([1, 2], dtype=np.float32)},
         "task": np.asarray([0], dtype=np.int32)},
        {"actors": {"cube": np.asarray([3, 4], dtype=np.float32)},
         "task": np.asarray([1], dtype=np.int32)},
    ]
    with h5py.File(path, "w") as handle:
        write_tree(handle, "env_states", rows)
    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["env_states/actors/cube"][:], [[1, 2], [3, 4]])
        np.testing.assert_array_equal(handle["env_states/task"][:], [[0], [1]])
