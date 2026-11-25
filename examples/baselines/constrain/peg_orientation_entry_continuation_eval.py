"""Fresh paired Base / random-500 update-30k / update-50k evaluation."""
from pathlib import Path

from examples.baselines.constrain import peg_orientation_entry_scaling_eval as scaling

BASE = scaling.BASE
CN500_30K = "cn500_30k"
CN500_50K = "cn500_50k"
ARMS = {
    BASE: scaling.ARMS[BASE],
    CN500_30K: scaling.ARMS["cn500_30k"],
    CN500_50K: "outputs/ckpt/20261119_135500_peg_orientation_random500_extend50k/update_50000.pt",
}
CURVE = (BASE, CN500_30K, CN500_50K)
MAX_NATIVE_CONTROLS, STAGE_SECONDS, WHOLE_SECONDS = 334800, 16200, 21600


def create_adapters(device="cuda"):
    from examples.baselines.constrain.peg_orientation_base_dp import PegBaseAdapter
    from examples.baselines.constrain.peg_orientation_constrain import PegConstrainAdapter

    adapters = {BASE: PegBaseAdapter(scaling.ROOT / ARMS[BASE], device=device)}
    for name in (CN500_30K, CN500_50K):
        adapters[name] = PegConstrainAdapter(
            scaling.ROOT / ARMS[name], device=device, task=None, record_all_hints=True
        )
    return adapters


def run_comparison(output, **kwargs):
    if kwargs.get("adapters") is None:
        kwargs["adapters"] = create_adapters()
    return scaling.run_comparison(
        output,
        arms=ARMS,
        primary=CURVE,
        secondary=CURVE,
        max_native_controls=MAX_NATIVE_CONTROLS,
        stage_seconds=STAGE_SECONDS,
        whole_seconds=WHOLE_SECONDS,
        source_archive_files=(Path(__file__), Path(__file__).parent / "tests" / "test_peg_orientation_entry_scaling_eval.py"),
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_comparison(args.output, adapters=create_adapters(args.device))
