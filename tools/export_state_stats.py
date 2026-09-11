#!/usr/bin/env python


from __future__ import annotations

import argparse
import struct
from pathlib import Path

from safetensors import safe_open

DEFAULT_STATS = Path(
    "/home/ubuntu/AI/deploy_vla/020000/pretrained_model/"
    "policy_preprocessor_step_5_normalizer_processor.safetensors"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "artifacts" / "state_norm_stats.bin"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with safe_open(str(args.stats_file), framework="np") as f:
        mean = f.get_tensor("observation.state.mean").flatten().tolist()
        std = f.get_tensor("observation.state.std").flatten().tolist()
    assert len(mean) == len(std), "mean/std 维度不一致"
    dim = len(mean)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        for v in mean + std:
            f.write(struct.pack("<f", float(v)))

    print(f"已写出 {args.output}  (dim={dim})")
    print("mean:", [round(v, 5) for v in mean])
    print("std :", [round(v, 5) for v in std])


if __name__ == "__main__":
    main()
