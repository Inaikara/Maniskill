import json

from examples.baselines.constrain.placesphere_matched_collection import (
    _greedy_task_identity,
    _task_layout_conflict,
    layout_rows,
    select_role_candidates,
)
from examples.baselines.constrain.validate_placesphere_matched_collection import _flag


def test_existing_catalog_rows_keep_layout_and_source_identity(tmp_path):
    scene = {
        "scene_id": "placesphere-042", "seed": 731042,
        "start_xy": [-0.14, 0.03], "container_xy": [0.11, -0.02],
        "cylinder": {"center_xy": [0.0, 0.1], "radius": 0.03, "height": 0.127},
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"scenes": [scene]}))

    rows = layout_rows([str(tmp_path)])

    assert rows == [{
        "source": str(tmp_path), "index": 0, "seed": 731042,
        "scene_id": "placesphere-042", "catalog": str(tmp_path),
        "start_xy": [-0.14, 0.03], "container_xy": [0.11, -0.02],
        "cylinder": {"center_xy": [0.0, 0.1], "radius": 0.03, "height": 0.127},
    }]


def test_task_identity_matches_frozen_rule_and_ignores_cylinder_proximity():
    first = {"seed": 1, "start_xy": [0.0, 0.0], "container_xy": [0.2, 0.0],
             "cylinder": {"center_xy": [0.0, 0.0]}}
    same_task = {"seed": 2, "start_xy": [0.019, 0.0], "container_xy": [0.2, 0.019],
                 "cylinder": {"center_xy": [0.4, 0.4]}}
    nearby_cylinder_only = {"seed": 3, "start_xy": [0.1, 0.1], "container_xy": [0.3, 0.1],
                            "cylinder": {"center_xy": [0.001, 0.001]}}

    assert _task_layout_conflict(first, same_task)
    assert not _task_layout_conflict(first, nearby_cylinder_only)
    assert _greedy_task_identity([first, same_task, nearby_cylinder_only]) == {
        "count": 3, "greedy_unique": 2, "conflicts": 1,
    }


def test_obstacle_candidates_are_the_unique_paired_and_obstacle_origins_with_fixed_backups():
    def row(catalog, seed, start, goal):
        return {"catalog": catalog, "seed": seed, "start_xy": start, "container_xy": goal}

    pair = "outputs/pair_scenes/merged"
    direct = "outputs/direct_scenes/assembled"
    obstacle = "outputs/obstacle_scenes/catalog_full"
    rows = [
        row(pair, 1, [0.0, 0.0], [0.2, 0.0]),
        row(obstacle, 2, [-0.1, 0.0], [0.2, 0.1]),
        row(direct, 3, [0.0, 0.0], [0.2, 0.0]),
        row(direct, 4, [-0.2, 0.1], [0.2, -0.1]),
    ]

    selected = select_role_candidates(rows, obstacle_target=2)

    assert [item["seed"] for item in selected["paired_primary"]] == [1, 2]
    assert [item["seed"] for item in selected["direct_only"]] == [3, 4]
    assert [item["seed"] for item in selected["obstacle_replacements"]] == [4]


def test_saved_native_info_flags_are_read_from_the_single_environment_value():
    assert _flag({"native_success": [True]}, "native_success")
    assert not _flag({"is_obj_grasped": [False]}, "is_obj_grasped")
