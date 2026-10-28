"""Fit the preregistered PickCube z-score profile on frozen direct demonstrations."""

import argparse
import json
from pathlib import Path

from examples.baselines.constrain.pickcube_observation_norm import (
    fit_frozen_direct_dataset, sha256_path,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--h5-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    stats = fit_frozen_direct_dataset(args.data, args.manifest,
                                      args.h5_sha256, args.manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_path(args.output),
                      "fit_frame_count": stats["fit_frame_count"],
                      "floored_fields": stats["floored_fields"]}, indent=2))


if __name__ == "__main__":
    main()
