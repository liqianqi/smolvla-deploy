#!/usr/bin/env python
"""导出动作反归一化统计量(mean/std)为一个小二进制文件, 供 C++ 运行时加载.

动作用 MEAN_STD 归一化, 真实(绝对关节角度)动作 = norm * std + mean.
输出文件 action_norm_stats.bin 布局(全部 float32, little-endian):
    [ mean[0..D-1], std[0..D-1] ]   其中 D = real_action_dim
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

from safetensors.torch import load_file

DEFAULT_STATS = Path(
    "/home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/"
    "pretrained_model/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
)
DEFAULT_OUTPUT = Path("/home/ubuntu/smolvla/smolvla-deploy/artifacts/action_norm_stats.bin")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-file", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    d = load_file(str(args.stats_file))
    mean = d["action.mean"].flatten().tolist()
    std = d["action.std"].flatten().tolist()
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
