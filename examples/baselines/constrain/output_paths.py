"""Output locations shared by experiment entry points."""

from datetime import datetime
from pathlib import Path

OUTPUT_ROOT = Path(__file__).resolve().parents[3] / "outputs"
DEMO_ROOT = OUTPUT_ROOT.parent / "demos"
CKPT_ROOT = OUTPUT_ROOT / "ckpt"
EXP_ROOT = OUTPUT_ROOT / "exp"


def experiment_dir(name):
    return EXP_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{name}"
