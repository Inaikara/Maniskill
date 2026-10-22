import json
import hashlib

import numpy as np
import pytest

from examples.baselines.constrain.placesphere_pilot import (
    PlaceSpherePilotConfig, _obs_vector, _policy_observation, make_env,
)
from examples.baselines.constrain.placesphere_scenes import (
    PlaceSphereSceneSet, _next_attempt_number, _tree_hash, generate,
)


def _successful_fixture_runner(kind, seed, rng, config):
    env = make_env(config=config)
    try:
        base = env.unwrapped
        start, container, center = [-.14, .02], [.10, .02], [-.02, .13]
        _, info = env.reset(seed=seed, options={"sphere_xy": start, "container_xy": container,
            "obstacle_enabled": True, "obstacle_center": center})
        trace = {"obstacle_center": center, "actual_cylinder_radius_m": base.obstacle_collision_radius,
            "actual_cylinder_height_m": base.obstacle_collision_height,
            "base_xy": base.agent.robot.pose.p[0, :2].detach().cpu().numpy().tolist(),
            "waypoint_xy": [-.02, .18], "phases": [{"phase": "release_open_and_retreat", "reached": True}],
            "post_release_control_steps": 2,
            "post_release_safety": {key: True for key in ("checked_through_native_success", "all_ungrasped",
                "no_obstacle_contact", "no_forbidden_crossing", "terminal_on_bin", "terminal_static", "terminal_ungrasped")}}
        return {"states": [base.get_state_dict()], "controller_states": [base.agent.controller.get_state()],
            "raw_obs": np.stack([_obs_vector(base.get_obs(info=info, unflattened=True))]),
            "obs": np.stack([_policy_observation(base, info)]), "actions": np.zeros((1, 7), np.float32),
            "trace": trace, "layout": {"start_xy": start, "container_xy": container},
            "native_success": True, "safe_success": True, "counters": {}, "phase_timings": {}}, None
    finally:
        env.close()


def test_attempt_counter_never_invokes_past_exact_cap_or_after_target():
    assert _next_attempt_number(attempts=8, accepted=0, target=3, max_attempts=9,
                               elapsed=1, max_wall_seconds=10) == 9
    assert _next_attempt_number(attempts=9, accepted=0, target=3, max_attempts=9,
                               elapsed=1, max_wall_seconds=10) is None
    assert _next_attempt_number(attempts=8, accepted=3, target=3, max_attempts=9,
                               elapsed=1, max_wall_seconds=10) is None


def test_scene_load_and_fresh_physical_restore_verify_geometry_state_and_controller(tmp_path):
    output = tmp_path / "scenes"
    report = generate(output, target=1, max_attempts=1, attempt_runner=_successful_fixture_runner)
    assert report["accepted"] == report["actual_runner_invocations"] == 1
    catalog = PlaceSphereSceneSet(output)
    scene = catalog.load(0)
    env, restored = catalog.restore(0)
    try:
        assert restored["norms"]["full_native_state_max_abs"] <= catalog.tolerances["full_state_atol"]
        assert restored["norms"]["controller_state_max_abs"] <= catalog.tolerances["controller_atol"]
        assert _tree_hash(restored["state"]) == scene["metadata"]["state_sha256"]
    finally:
        env.close()
    wrong = make_env(config=PlaceSpherePilotConfig(obstacle_radius=.04))
    try:
        with pytest.raises(ValueError, match="geometry"):
            catalog.restore(0, wrong)
    finally:
        wrong.close()


def test_loader_rejects_changed_controller_after_valid_manifest_container_hashes(tmp_path):
    output = tmp_path / "scenes"
    generate(output, target=1, max_attempts=1, attempt_runner=_successful_fixture_runner)
    catalog = PlaceSphereSceneSet(output)
    import h5py
    with h5py.File(catalog.path, "r+") as h5:
        # The native controller snapshot is an empty state dict; adding a key
        # is a schema/hash mutation even though there are no tensor leaves.
        h5["placesphere-000/controller"].create_dataset("tampered", data=np.asarray([1]))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["scenes_file_sha256"] = hashlib.sha256(catalog.path.read_bytes()).hexdigest()
    payload = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256"}
    manifest["manifest_identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, indent=2).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="controller checksum"):
        PlaceSphereSceneSet(output).load(0)


def test_report_marks_failure_at_real_runner_count(tmp_path):
    calls = []

    def fail(kind, seed, rng, config):
        calls.append(seed)
        return {"failure": "fixture_rejected", "counters": {"geometry_tries": 1}}, "fixture_rejected"

    with pytest.raises(RuntimeError, match="incomplete"):
        generate(tmp_path / "incomplete", target=2, max_attempts=2,
                 max_wall_seconds=60, attempt_runner=fail)
    report = json.loads((tmp_path / "incomplete/report.json").read_text())
    journal = [json.loads(line) for line in (tmp_path / "incomplete/attempts.jsonl").read_text().splitlines()]
    assert len(calls) == len(journal) == report["attempts"] == report["actual_runner_invocations"] == 2
    assert report["accepted"] == 0 and report["complete"] is False


def test_scene_generation_rejects_native_success_without_post_release_proof(tmp_path):
    def unsafe(*args):
        outcome, reason = _successful_fixture_runner(*args)
        outcome["trace"]["post_release_safety"]["no_obstacle_contact"] = False
        return outcome, reason

    with pytest.raises(ValueError, match="post-release safety proof"):
        generate(tmp_path / "unsafe", target=1, max_attempts=1, attempt_runner=unsafe)
