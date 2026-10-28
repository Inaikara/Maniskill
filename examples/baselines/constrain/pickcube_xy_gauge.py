"""Training-only translation of the PickCube world-coordinate origin."""

import torch

from examples.baselines.constrain.pickcube_base_dp import NAMED_FIELDS

KIND = "global_xy_gauge_v1"
HALF_WIDTH_M = 0.02
SEED_OFFSET = 424242
METADATA = {
    "kind": KIND,
    "half_width_m": HALF_WIDTH_M,
    "seed_offset": SEED_OFFSET,
    "rng_algorithm": "torch.Generator(device=cpu).manual_seed(training_seed+424242); once per batch call torch.rand((B,2), dtype=float32, generator=cpu_generator); consume row-major [window0-x, window0-y, window1-x, window1-y, ...], map each value (2u-1)*0.02 meters",
    "application_order": "after device transfer and before position-unit encoding; same draw broadcast across both history frames",
    "inference": "off",
    "source": "same_frozen_physical_episodes",
}


def _field_slices(fields):
    result = {}
    start = 0
    for name, width in fields:
        if name in result or int(width) <= 0:
            raise ValueError("PickCube named observation schema is invalid")
        result[name] = (start, start + int(width))
        start += int(width)
    if start != 45:
        raise ValueError("PickCube named observation schema must total 45 features")
    return result


def _world_xy_indices(fields=NAMED_FIELDS, include_obstacle=False):
    slices = _field_slices(fields)
    selected = [("extra/tcp_pose", 7), ("extra/goal_pos", 3), ("extra/obj_pose", 7)]
    if include_obstacle:
        selected.append(("extra/pickcube_obstacle", 3))
    indices = []
    for name, expected_width in selected:
        if name not in slices:
            raise ValueError(f"PickCube observation schema is missing {name}")
        start, end = slices[name]
        if end - start != expected_width:
            raise ValueError(f"PickCube observation field width mismatch for {name}")
        indices.extend((start, start + 1))
    return tuple(indices)


WORLD_XY_42 = _world_xy_indices()
WORLD_XY_45 = _world_xy_indices(include_obstacle=True)


def make_generator(training_seed=1):
    """Create the dedicated CPU generator, independent of all training RNGs."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(training_seed) + SEED_OFFSET)
    return generator


def augment_window_observations(observations, generator):
    """Apply one shared XY-origin shift per window, preserving relative fields.

    `observations` is BxHx42 (Base) or BxHx45 (Control). The returned tensor is
    a clone; cached dataset tensors and action labels are never modified.
    """
    raw = torch.as_tensor(observations)
    if raw.ndim != 3 or raw.shape[-1] not in (42, 45):
        raise ValueError("gauge augmentation expects (batch, history, 42|45) observations")
    if generator.device.type != "cpu":
        raise ValueError("gauge augmentation requires its dedicated CPU generator")
    batch_size = raw.shape[0]
    delta = (torch.rand((batch_size, 2), generator=generator, dtype=torch.float32) * 2 - 1)
    delta = (delta * HALF_WIDTH_M).to(device=raw.device, dtype=raw.dtype)
    shifted, _ = apply_xy_shift(raw, delta)
    return shifted, delta


def apply_xy_shift(observations, delta):
    """Apply a specified per-window translation; also defines the zero identity."""
    raw = torch.as_tensor(observations)
    if raw.ndim != 3 or raw.shape[-1] not in (42, 45):
        raise ValueError("gauge augmentation expects (batch, history, 42|45) observations")
    delta = torch.as_tensor(delta, dtype=raw.dtype, device=raw.device)
    if delta.shape != (raw.shape[0], 2) or not bool(torch.isfinite(delta).all()):
        raise ValueError("XY gauge translation must be finite with shape (batch, 2)")
    result = raw.clone()
    indices = WORLD_XY_45 if raw.shape[-1] == 45 else WORLD_XY_42
    for x_index, y_index in zip(indices[::2], indices[1::2]):
        result[..., x_index] += delta[:, None, 0]
        result[..., y_index] += delta[:, None, 1]
    return result, delta


def validate_metadata(value):
    if value != METADATA:
        raise ValueError("PickCube training augmentation metadata mismatch")
    return value


def validate_checkpoint_metadata(value, expected):
    """Require checkpoint augmentation identity to match the selected profile."""
    if expected is not None:
        validate_metadata(expected)
    if value != expected:
        raise ValueError("PickCube checkpoint training augmentation identity mismatch")
    return value


def empty_delta_stats():
    return {"windows": 0, "minimum_m": [float("inf"), float("inf")],
            "maximum_m": [float("-inf"), float("-inf")], "sum_m": [0.0, 0.0]}


def update_delta_stats(stats, delta):
    values = torch.as_tensor(delta, dtype=torch.float64).reshape(-1, 2)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("sampled PickCube gauge deltas must be finite")
    if not len(values):
        return stats
    result = {"windows": int(stats["windows"]) + len(values),
              "minimum_m": [min(stats["minimum_m"][i], float(values[:, i].min())) for i in range(2)],
              "maximum_m": [max(stats["maximum_m"][i], float(values[:, i].max())) for i in range(2)],
              "sum_m": [stats["sum_m"][i] + float(values[:, i].sum()) for i in range(2)]}
    return result


def finalize_delta_stats(stats):
    count = int(stats["windows"])
    if count == 0:
        raise ValueError("no PickCube gauge shifts were sampled")
    return {"windows": count, "minimum_m": list(stats["minimum_m"]),
            "maximum_m": list(stats["maximum_m"]),
            "mean_m": [value / count for value in stats["sum_m"]]}
