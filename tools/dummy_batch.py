"""导出用的一帧观测:有本地数据集就读第 0 帧,否则按 checkpoint 合成 dummy.

LeRobotDataset 在 root 里没有 meta/ 时会去 HuggingFace 拉 repo_id.
local/* 不是 Hub 上的仓库,会变成 401 RepositoryNotFoundError.
导出只需要正确形状的 dummy 张量,权重来自 checkpoint,不依赖真实图像内容.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open


def detect_image_keys(policy_path: Path) -> tuple[str, str]:
    pre_path = policy_path / "policy_preprocessor.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text())
        for step in pre.get("steps", []):
            rename_map = step.get("config", {}).get("rename_map") or {}
            inv = {v: k for k, v in rename_map.items()}
            primary = inv.get("observation.images.camera1")
            wrist = inv.get("observation.images.camera2")
            if primary and wrist:
                return primary, wrist
    return "observation.images.image", "observation.images.wrist_image"


def detect_state_dim(policy_path: Path) -> int:
    candidates = sorted(policy_path.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))
    if not candidates:
        return 7
    with safe_open(str(candidates[0]), framework="pt") as f:
        return int(f.get_tensor("observation.state.mean").shape[0])


def add_batch_dim(value):
    return value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]


def make_dummy_raw_batch(
    policy_path: Path,
    instruction: str = "pick up the blue block",
) -> dict:
    image_key, wrist_key = detect_image_keys(policy_path)
    state_dim = detect_state_dim(policy_path)
    raw = {
        "task": instruction,
        "observation.state": torch.zeros(state_dim, dtype=torch.float32),
        image_key: torch.rand(3, 256, 256),
        wrist_key: torch.rand(3, 256, 256),
    }
    print(f"      dummy 观测: state_dim={state_dim}  images=({image_key}, {wrist_key})")
    return {k: add_batch_dim(v) for k, v in raw.items()}


def load_export_raw_batch(
    policy_path: Path,
    dataset_root: Path | None = None,
    repo_id: str | None = None,
    instruction: str = "pick up the blue block",
) -> dict:
    """优先本地 LeRobot 数据集;目录不存在或缺少 meta/ 则回退 dummy,不去 Hub."""
    if dataset_root is not None and repo_id:
        candidates = [Path(dataset_root), Path(dataset_root) / repo_id]
        if any((p / "meta" / "info.json").is_file() for p in candidates):
            from lerobot.datasets import LeRobotDataset

            image_key, wrist_key = detect_image_keys(policy_path)
            dataset = LeRobotDataset(repo_id, root=dataset_root)
            sample = dataset[0]
            raw = {
                "task": sample["task"],
                "observation.state": sample["observation.state"],
                image_key: sample[image_key],
                wrist_key: sample[wrist_key],
            }
            print(f"      数据集第 0 帧: {dataset_root}  {image_key}/{wrist_key}")
            return {k: add_batch_dim(v) for k, v in raw.items()}
        print(f"      本地数据集不存在({dataset_root}),改用 dummy 观测导出")
    return make_dummy_raw_batch(policy_path, instruction=instruction)
