from examples.baselines.constrain.placesphere_obstacle1000_expansion import task_layout_conflict
from examples.baselines.constrain.placesphere_r26_replacements import _layout_identity


def test_replacement_audit_normalizes_nested_episode_layouts():
    candidate = {"seed": 1, "start_xy": [0.0, 0.0], "container_xy": [0.2, 0.1]}
    old_episode = {"seed": 2, "kind": "obstacle", "layout": {
        "start_xy": [0.01, 0.0], "container_xy": [0.2, 0.11]}}
    assert task_layout_conflict(candidate, _layout_identity(old_episode))
