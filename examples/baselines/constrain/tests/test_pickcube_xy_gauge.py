import torch

from examples.baselines.constrain.pickcube_base_dp import NAMED_FIELDS
from examples.baselines.constrain.pickcube_constrain import (
    PickCubeConditionNorm,
    pickcube_relative_condition,
)
from examples.baselines.constrain.pickcube_xy_gauge import (
    HALF_WIDTH_M,
    METADATA,
    SEED_OFFSET,
    WORLD_XY_42,
    WORLD_XY_45,
    augment_window_observations,
    apply_xy_shift,
    empty_delta_stats,
    finalize_delta_stats,
    make_generator,
    validate_checkpoint_metadata,
    validate_metadata,
    update_delta_stats,
)


def _sample_raw(batch=4, history=2, dim=45):
    values = torch.arange(batch * history * dim, dtype=torch.float32).reshape(batch, history, dim) / 100
    if dim == 45:
        values[..., 42] = 0.31
        values[..., 43] = -0.27
        values[..., 44] = 0.04
    return values


def test_gauge_shift_is_one_scene_translation_and_preserves_physical_invariants():
    raw = _sample_raw()
    original = raw.clone()
    actions = torch.arange(4 * 16 * 7, dtype=torch.float32).reshape(4, 16, 7)
    actions_before = actions.clone()
    generator = make_generator()

    augmented, delta = augment_window_observations(raw, generator)

    assert augmented.data_ptr() != raw.data_ptr()
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
    assert torch.equal(actions, actions_before)
    assert delta.shape == (4, 2)
    assert bool((delta.abs() <= HALF_WIDTH_M).all())
    for x_index, y_index in zip(WORLD_XY_45[::2], WORLD_XY_45[1::2]):
        torch.testing.assert_close(augmented[..., x_index] - raw[..., x_index], delta[:, None, 0].expand(-1, 2))
        torch.testing.assert_close(augmented[..., y_index] - raw[..., y_index], delta[:, None, 1].expand(-1, 2))
    changed = set(WORLD_XY_45)
    for index in set(range(45)) - changed:
        assert torch.equal(augmented[..., index], raw[..., index]), index

    # Independently compare raw relative vectors and cylinder clearance margins.
    for field_start in (36, 39):
        assert torch.equal(augmented[..., field_start:field_start + 3], raw[..., field_start:field_start + 3])
    raw_tcp_obj = raw[..., 19:21] - raw[..., 29:31]
    aug_tcp_obj = augmented[..., 19:21] - augmented[..., 29:31]
    raw_obj_goal = raw[..., 29:31] - raw[..., 26:28]
    aug_obj_goal = augmented[..., 29:31] - augmented[..., 26:28]
    torch.testing.assert_close(aug_tcp_obj, raw_tcp_obj, rtol=0, atol=1e-6)
    torch.testing.assert_close(aug_obj_goal, raw_obj_goal, rtol=0, atol=1e-6)
    radius = raw[..., 44]
    raw_margin = torch.linalg.vector_norm(raw[..., 29:31] - raw[..., 42:44], dim=-1) - radius
    aug_margin = torch.linalg.vector_norm(augmented[..., 29:31] - augmented[..., 42:44], dim=-1) - radius
    torch.testing.assert_close(aug_margin, raw_margin, rtol=0, atol=1e-6)

    hint_norm = PickCubeConditionNorm()
    raw_hint = pickcube_relative_condition(raw, hint_norm)
    aug_hint = pickcube_relative_condition(augmented, hint_norm)
    torch.testing.assert_close(aug_hint, raw_hint, rtol=0, atol=1e-6)


def test_gauge_zero_identity_base_shape_and_fixed_dedicated_random_stream():
    raw = _sample_raw(dim=42)
    before = raw.clone()
    generator = torch.Generator(device="cpu").manual_seed(4)
    state_before = generator.get_state().clone()
    actual, delta = augment_window_observations(raw, generator)
    assert torch.equal(raw, before)
    zero_shift, zero = apply_xy_shift(raw, torch.zeros(4, 2))
    assert torch.equal(zero, torch.zeros(4, 2))
    assert torch.equal(zero_shift, before)
    assert WORLD_XY_42 == tuple(i for pair in ((19, 20), (26, 27), (29, 30)) for i in pair)
    assert WORLD_XY_45 == WORLD_XY_42 + (42, 43)
    assert generator.get_state().ne(state_before).any()
    torch.testing.assert_close(actual[..., 36:42], raw[..., 36:42], rtol=0, atol=0)

    left, _ = augment_window_observations(raw, make_generator(1))
    right, _ = augment_window_observations(raw, make_generator(1))
    assert torch.equal(left, right)
    assert make_generator(1).initial_seed() == 1 + SEED_OFFSET

    torch.manual_seed(831)
    global_before = torch.get_rng_state().clone()
    state = make_generator(7).get_state()
    local = make_generator(7)
    first, _ = augment_window_observations(raw, local)
    saved_state = local.get_state()
    second, _ = augment_window_observations(raw, local)
    local.set_state(saved_state)
    replayed, _ = augment_window_observations(raw, local)
    assert torch.equal(second, replayed)
    assert torch.equal(torch.get_rng_state(), global_before)
    assert torch.equal(state, make_generator(7).get_state())
    assert not torch.equal(first, second)

    stats = update_delta_stats(empty_delta_stats(), torch.tensor([[-0.01, 0.02], [0.01, -0.02]]))
    stats = update_delta_stats(stats, torch.tensor([[0.0, 0.01]]))
    final_stats = finalize_delta_stats(stats)
    assert final_stats["windows"] == 3
    assert abs(final_stats["minimum_m"][0] + 0.01) < 1e-8
    assert abs(final_stats["minimum_m"][1] + 0.02) < 1e-8
    assert abs(final_stats["maximum_m"][0] - 0.01) < 1e-8
    assert abs(final_stats["maximum_m"][1] - 0.02) < 1e-8
    assert abs(final_stats["mean_m"][0]) < 1e-8
    assert abs(final_stats["mean_m"][1] - 0.01 / 3) < 1e-8


def test_augmentation_metadata_is_exact_and_named_indices_are_schema_derived():
    validate_metadata(dict(METADATA))
    validate_checkpoint_metadata(dict(METADATA), METADATA)
    validate_checkpoint_metadata(None, None)
    for value, expected in ((None, METADATA), (METADATA, None),
                            ({**METADATA, "seed_offset": 7}, METADATA)):
        try:
            validate_checkpoint_metadata(value, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("checkpoint accepted a missing or mismatched augmentation profile")
    try:
        validate_metadata({**METADATA, "half_width_m": 0.03})
    except ValueError:
        pass
    else:
        raise AssertionError("augmentation metadata accepted an unregistered width")
    assert [name for name, _ in NAMED_FIELDS][-1] == "extra/pickcube_obstacle"


def test_paired_report_aggregator_accepts_actual_shaped_rows_before_native_trial():
    from examples.baselines.constrain.pickcube_xy_gauge_round40 import validate_report_contract

    evidence = validate_report_contract()
    assert evidence == {
        "actual_shaped_row_count": 30,
        "summary_complete": True,
        "transition_fields": ["native_success", "safe_success", "obstacle_contact",
                               "xy_forbidden_crossing"],
        "aggregated_fields": ["policy_metrics", "terminal_phase_counts",
                              "contact_first_phase_counts", "paired_base_vs_control",
                              "vs_r38", "gate", "first_contact_link_counts",
                              "final_cube_xyz_and_goal_xyz_present"],
        "reconciled": True,
    }


def test_r38_primary_comparator_is_complete_and_content_pinned():
    from examples.baselines.constrain.pickcube_xy_gauge_round40 import _load_r38_primary

    registration, tree = _load_r38_primary()
    assert registration["registration_sha256"] == "65963d4c4c01f95e497e4387051b17153a63384584af0b824461131cce4b1b38"
    assert len(tree["files"]) >= 60
    assert len([name for name in tree["files"] if name.startswith("base_dp/episode_")]) == 30
    assert len([name for name in tree["files"] if name.startswith("constrain_dp/episode_")]) == 30


def test_round37_terminal_phase_report_uses_real_retained_rows_and_keeps_contact_phases_distinct():
    import json
    from pathlib import Path

    from examples.baselines.constrain import pickcube_zscore_round37 as engine
    from examples.baselines.constrain.pickcube_learning_pilot import summarize_pilot

    root = Path("outputs/exp/20261029_052324_pickcube_clearance_r43/paired_native_r43")
    policies = ("base_dp", "constrain_dp")
    rows = {
        policy: [json.loads((root / policy / f"episode_{index:03d}.json").read_text())
                 for index in range(30)]
        for policy in policies
    }
    for policy in policies:
        summary = summarize_pilot(rows[policy], 30)
        terminal = engine._terminal_phase_counts(rows[policy])
        contacts = engine._phase_counts(rows[policy])
        assert summary["complete"]
        assert sum(terminal.values()) == 30
        assert terminal == {
            phase: sum(row["result"].get("phase", "unknown") == phase for row in rows[policy])
            for phase in {row["result"].get("phase", "unknown") for row in rows[policy]}
        }
        assert contacts == {
            phase: sum((row.get("first_contact") or {}).get("phase") == phase
                       for row in rows[policy])
            for phase in {(row.get("first_contact") or {}).get("phase")
                          for row in rows[policy] if (row.get("first_contact") or {}).get("phase")}
        }

    # The new objective context is report metadata only and cannot supply phases.
    objective_context = {"model_kind": "SE3_horizon8_clearance", "horizon": 8}
    aggregate = {
        "objective_context": objective_context,
        "policy_metrics": {policy: summarize_pilot(rows[policy], 30)
                           for policy in policies},
        "terminal_phase_counts": {policy: engine._terminal_phase_counts(rows[policy])
                                   for policy in policies},
        "first_contact_phase_counts": {policy: engine._phase_counts(rows[policy])
                                       for policy in policies},
    }
    assert aggregate["objective_context"]["model_kind"] == "SE3_horizon8_clearance"
    assert aggregate["terminal_phase_counts"]["constrain_dp"]["task_success"] == 30
    assert sum(aggregate["first_contact_phase_counts"]["constrain_dp"].values()) == 16
