from examples.baselines.constrain.audit_pickcube_existing_pool import layout_key, task_duplicate


def test_obstacle_pool_dedup_uses_start_and_goal_componentwise_tolerance():
    base = {
        "kind": "obstacle",
        "layout": {"start_xy": [0.011, 0.021], "goal_xy": [0.201, 0.101], "center_xy": [0.101, 0.051]},
    }
    same = {
        "kind": "obstacle",
        "layout": {"start_xy": [0.0111, 0.0211], "goal_xy": [0.2011, 0.1011],
                   "center_xy": [0.1011, 0.0511]},
    }
    changed_cylinder = {
        "kind": "obstacle",
        "layout": {"start_xy": [0.011, 0.021], "goal_xy": [0.201, 0.101], "center_xy": [0.141, 0.051]},
    }
    assert task_duplicate(base, same)
    assert task_duplicate(base, changed_cylinder)


def test_seed_identity_is_duplicate_even_when_layout_coordinates_differ():
    left = {"kind": "obstacle", "layout": {"seed": 10, "start_xy": [0.0, 0.0], "goal_xy": [0.1, 0.1]}}
    right = {"kind": "obstacle", "layout": {"seed": 10, "start_xy": [0.1, 0.0], "goal_xy": [0.2, 0.1]}}
    assert task_duplicate(left, right)


def test_direct_layout_identity_omits_obstacle():
    direct = {"kind": "direct", "layout": {"start_xy": [0.011, 0.021], "goal_xy": [0.201, 0.101]}}
    assert layout_key(direct) == (1, 1, 10, 5)


def test_incomplete_obstacle_identity_is_not_counted_as_unique():
    incomplete = {"kind": "obstacle", "layout": {"start_xy": [0.011, 0.021], "goal_xy": [0.201, 0.101]}}
    assert layout_key(incomplete) is None
