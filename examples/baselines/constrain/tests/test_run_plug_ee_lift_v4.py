import hashlib
import json

import pytest

from examples.baselines.constrain import run_plug_ee_lift_v4 as runner


def _fake_result(*args, **kwargs):
    return {"expert_controls": 1, "replay_controls": 1,
            "expert_elapsed_seconds": 0.01, "replay_elapsed_seconds": 0.01,
            "expert": {"success": True, "official_success": True},
            "replay": {"success": True}}


def test_full_seal_exists_and_matches_before_first_case(tmp_path, monkeypatch):
    exp, demos = tmp_path / "exp", tmp_path / "demos"
    seen = []
    prereg_seal = []

    def checked_case(seed, case_dir, target_yaw):
        prereg_path = exp / "preregistration.json"
        assert prereg_path.is_file()
        prereg = json.loads(prereg_path.read_text())
        seal = hashlib.sha256(prereg_path.read_bytes()).hexdigest()
        prereg_seal.append(seal)
        assert len(set(prereg_seal)) == 1
        assert prereg["source_sha256"]
        assert len(prereg["planned_cases"]) == 20
        assert prereg["source_file_count"] == len(prereg["source_sha256"])
        assert prereg["historical_numeric_results_viewed"] is True
        assert "nonblind" in prereg["blinding"]
        assert prereg["control_caps"]["per_episode"] == 600
        assert prereg["control_caps"]["align"] == 180
        assert prereg["control_caps"]["full_expert"] == 12000
        assert prereg["wall_seconds"]["expert"] == 3600
        assert prereg["wall_seconds"]["replay"] == 3600
        assert prereg["official_gate"] == {"position_m": 0.005,
                                              "quaternion_angle_rad": 0.2}
        assert prereg["acceptance"]["replay_fraction_each_profile"] == 0.95
        assert "timeout --signal=TERM --kill-after=20s 9000s" in prereg["outer_command"]
        assert prereg["random_target_yaws"][0] == -runner.math.pi / 8
        assert prereg["random_target_yaws"][-1] == runner.math.pi / 8
        for relative, digest in prereg["source_sha256"].items():
            archive = exp / "source_archive" / relative
            current = runner.Path(__file__).resolve().parents[4] / relative
            assert archive.is_file()
            assert runner._sha(archive) == digest
            assert runner._sha(current) == digest
        seen.append((seed, target_yaw))
        return _fake_result()

    monkeypatch.setattr(runner, "run_case", checked_case)
    runner.run("full", exp, demos)
    assert len(seen) == 20
    report = json.loads((exp / "native_attempt_001.json").read_text())
    assert len(report["cases"]) == 20
    assert report["random_yaw_bin_counts_planned"] == [2, 2, 2, 2, 2]
    assert report["random_accepted_yaw_bin_counts"] == [2, 2, 2, 2, 2]
    assert report["random_accepted_endpoint_covered"] is True
    assert len(set(prereg_seal)) == 1


def test_existing_registration_refuses_run_without_overwrite(tmp_path, monkeypatch):
    exp, demos = tmp_path / "exp", tmp_path / "demos"
    exp.mkdir()
    demos.mkdir()
    prereg = exp / "preregistration.json"
    prereg.write_text('{"historical": true}\n')
    before = prereg.read_bytes()
    called = []
    monkeypatch.setattr(runner, "run_case", lambda *a, **k: called.append(True))
    with pytest.raises(FileExistsError):
        runner.run("pair", exp, demos)
    assert not called
    assert prereg.read_bytes() == before
