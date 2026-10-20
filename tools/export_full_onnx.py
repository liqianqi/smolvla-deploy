#!/usr/bin/env python
"""把 SmolVLA 整条 sample_actions 导出为单个 ONNX(视觉 + prefix + prefill + 10 步去噪全部在图内).

与 4 引擎方案的区别: 去噪步数 / 相机数 / chunk 均固定在图里, 改动需重新导出.
输入: images[N,1,3,H,W](已 resize_with_pad 并归一化到 [-1,1]) img_masks[N,1] lang_tokens[1,L]
      lang_masks[1,L] state[1,max_state_dim](已归一化) noise[1,chunk,max_action_dim]
输出: actions[1,chunk,max_action_dim](归一化空间, 取前 real_action_dim 维再反归一化)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import lerobot.policies.smolvla.modeling_smolvla as msv
from dummy_batch import load_export_raw_batch
from export_vision_encoder import VisionEncoderWrapper
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# 与 export_denoise 一致:时间编码用 float32(onnxruntime 的 Cos 不支持 float64)
_orig_get_safe_dtype = msv.get_safe_dtype
msv.get_safe_dtype = lambda requested, device_type: (
    torch.float32 if requested == torch.float64 else _orig_get_safe_dtype(requested, device_type)
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POLICY_PATH = REPO_ROOT / "ela3_smolvla_new_dataset_v3/checkpoints/015000/pretrained_model"
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/smolvla/lerobot_dataset")
DEFAULT_REPO_ID = "local/ela3_blue_block_new"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts" / "smolvla_full.onnx"


class FullPolicy(nn.Module):
    """images/img_masks 以堆叠张量传入, 内部拆回 list 后调 sample_actions."""

    def __init__(self, model: nn.Module, num_images: int):
        super().__init__()
        self.model = model  # VLAFlowMatching
        self.num_images = num_images

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        imgs = [images[i] for i in range(self.num_images)]
        masks = [img_masks[i] for i in range(self.num_images)]
        return self.model.sample_actions(imgs, masks, lang_tokens, lang_masks, state, noise=noise)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 加载 policy(fp32): {args.policy_path}")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.eval()
    policy.model.float()
    model = policy.model
    # 视觉编码换成与 vision_encoder.onnx 相同的无 mask 实现(eager 下与原实现等价)
    vision = VisionEncoderWrapper(model.vlm_with_expert)
    model.vlm_with_expert.embed_image = lambda img: vision(img)
    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=str(args.policy_path))

    print("[2/5] 取一帧观测作为示例输入")
    raw = load_export_raw_batch(args.policy_path, args.dataset_root, args.repo_id)
    batch = preprocessor(raw)
    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
    images_t = torch.stack(images, dim=0).float().contiguous()
    masks_t = torch.stack(img_masks, dim=0).contiguous()
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    noise = torch.randn(
        (1, cfg.chunk_size, cfg.max_action_dim), generator=gen, device=args.device, dtype=torch.float32
    )
    print(f"      images: {tuple(images_t.shape)}  img_masks: {tuple(masks_t.shape)}  "
          f"lang_tokens: {tuple(lang_tokens.shape)}  state: {tuple(state.shape)}  noise: {tuple(noise.shape)}")

    wrapper = FullPolicy(model, images_t.shape[0]).eval()
    inputs = (images_t, masks_t, lang_tokens, lang_masks, state, noise)

    print("[3/5] PyTorch 参考前向")
    with torch.inference_mode():
        ref = wrapper(*inputs).cpu().numpy()
    print(f"      actions: {ref.shape}")

    print(f"[4/5] 导出 ONNX -> {args.output}(10 步去噪展开在图内, 约 4-5 分钟)")
    t0 = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(args.output),
            input_names=["images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"],
            output_names=["actions"],
            opset_version=args.opset,
            dynamo=True,
        )
    data = args.output.with_name(args.output.name + ".data")
    size_mb = (args.output.stat().st_size + (data.stat().st_size if data.exists() else 0)) / 1e6
    print(f"      导出耗时 {time.perf_counter() - t0:.0f}s, 文件大小(含 .data): {size_mb:.0f} MB")

    print("[5/5] onnxruntime 验证数值一致性")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    feed = {
        "images": images_t.cpu().numpy(),
        "img_masks": masks_t.cpu().numpy(),
        "lang_tokens": lang_tokens.cpu().numpy().astype(np.int64),
        "lang_masks": lang_masks.cpu().numpy(),
        "state": state.cpu().numpy().astype(np.float32),
        "noise": noise.cpu().numpy(),
    }
    out = sess.run(["actions"], feed)[0]
    max_abs = float(np.max(np.abs(out - ref)))
    print(f"      单图 ONNX vs 原 policy  max|diff|={max_abs:.3e}  "
          f"{'通过 ✓' if max_abs < 5e-3 else '偏差偏大,需排查 ✗'}")


if __name__ == "__main__":
    main()
