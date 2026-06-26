#!/usr/bin/env python
"""把 SmolVLA 的 prefix 组装(embed_prefix 的非视觉部分)导出为 ONNX(路线②第①b步).

作用:接收"已编码的图像 embeds"(来自 vision 引擎)+ 语言 token + state,
完成 语言查表 / state_proj / 图像缩放 / 拼接 / 掩码,输出 prefill 引擎直接可用的:
    prefix_embs      (B, prefix_len, dim)
    prefix_pad_masks (B, prefix_len) bool
    attn_2d_mask     (B, prefix_len, prefix_len) bool
    position_ids     (B, prefix_len) int64

这样 C++ 端就只需串联 4 个 ONNX 引擎 + tokenizer + 去噪循环,无需手写任何模型算子.
仅适配当前 checkpoint 配置:add_image_special_tokens=False, prefix_length=0.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

DEFAULT_POLICY_PATH = Path(
    "/home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/smolvla/data/ela3_lerobot_v2_joint")
DEFAULT_REPO_ID = "local/ela3_blue_cube_v2_joint"
DEFAULT_OUTPUT = Path("/home/ubuntu/smolvla/smolvla-deploy/artifacts/prefix_assembler.onnx")


class PrefixAssembler(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model  # VLAFlowMatching

    def forward(self, image_embeds, img_masks, lang_tokens, lang_masks, state):
        model = self.model
        embs, pad, att = [], [], []
        num_img = image_embeds.shape[0]
        for i in range(num_img):
            ie = image_embeds[i]
            d = ie.shape[-1]
            ie = ie * math.sqrt(d)
            b, n = ie.shape[0], ie.shape[1]
            embs.append(ie)
            pad.append(img_masks[i][:, None].expand(b, n))
            att += [0] * n
        lang_emb = model.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad.append(lang_masks)
        att += [0] * lang_emb.shape[1]

        state_emb = model.state_proj(state)[:, None, :]
        embs.append(state_emb)
        b = state_emb.shape[0]
        pad.append(torch.ones(b, 1, dtype=torch.bool, device=state_emb.device))
        att += [1]

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad = torch.cat(pad, dim=1)
        att_masks = torch.tensor(att, dtype=torch.bool, device=prefix_embs.device)[None, :].expand(b, -1)

        attn_2d = make_att_2d_masks(prefix_pad, att_masks)
        position_ids = torch.cumsum(prefix_pad, dim=1) - 1
        return prefix_embs, prefix_pad, attn_2d, position_ids


def add_batch_dim(v):
    return v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=18)
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
    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=str(args.policy_path))

    print("[2/5] 取一帧观测,编码图像 embeds 作为示例输入")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    sample = dataset[0]
    raw = {
        "task": sample["task"],
        "observation.state": sample["observation.state"],
        "observation.images.image": sample["observation.images.image"],
        "observation.images.wrist_image": sample["observation.images.wrist_image"],
    }
    batch = preprocessor({k: add_batch_dim(v) for k, v in raw.items()})

    with torch.inference_mode():
        images, img_masks_list = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        # 用 PyTorch embed_image 生成图像 embeds(部署时来自 vision ONNX)
        image_embeds = torch.stack(
            [model.vlm_with_expert.embed_image(img).float() for img in images], dim=0
        ).contiguous()
        img_masks = torch.stack(img_masks_list, dim=0).contiguous()
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

        # 参考:原 embed_prefix(注意它内部自己 embed_image,与上面同源)
        ref_embs, ref_pad, ref_att = model.embed_prefix(images, img_masks_list, lang_tokens, lang_masks, state)
        ref_attn2d = make_att_2d_masks(ref_pad, ref_att)
        ref_pos = torch.cumsum(ref_pad, dim=1) - 1

    print(f"      image_embeds: {tuple(image_embeds.shape)}  img_masks: {tuple(img_masks.shape)}  "
          f"lang_tokens: {tuple(lang_tokens.shape)}")

    wrapper = PrefixAssembler(model).eval()
    with torch.inference_mode():
        w_embs, w_pad, w_attn2d, w_pos = wrapper(image_embeds, img_masks, lang_tokens, lang_masks, state)
    eq = float(torch.max(torch.abs(w_embs - ref_embs)))
    print(f"[3/5] 组装器 vs 原 embed_prefix  max|diff|={eq:.3e}  ({'等价 ✓' if eq < 1e-4 else '✗'})")

    print(f"[4/5] 导出 ONNX -> {args.output}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (image_embeds, img_masks, lang_tokens, lang_masks, state),
            str(args.output),
            input_names=["image_embeds", "img_masks", "lang_tokens", "lang_masks", "state"],
            output_names=["prefix_embs", "prefix_pad_masks", "attn_2d_mask", "position_ids"],
            opset_version=args.opset,
            dynamo=True,
        )
    print(f"      文件大小: {args.output.stat().st_size / 1e6:.1f} MB")

    print("[5/5] onnxruntime 验证")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    outs = sess.run(
        ["prefix_embs", "prefix_pad_masks", "attn_2d_mask", "position_ids"],
        {
            "image_embeds": image_embeds.cpu().numpy().astype(np.float32),
            "img_masks": img_masks.cpu().numpy(),
            "lang_tokens": lang_tokens.cpu().numpy().astype(np.int64),
            "lang_masks": lang_masks.cpu().numpy(),
            "state": state.cpu().numpy().astype(np.float32),
        },
    )
    d = float(np.max(np.abs(outs[0] - ref_embs.cpu().numpy())))
    print(f"      prefix_embs: max|diff|={d:.3e}  {'通过 ✓' if d < 1e-3 else '✗'}")

    print("\n完成.4 个 ONNX 引擎齐全:vision / prefix_assembler / prefill / denoise.")


if __name__ == "__main__":
    main()
