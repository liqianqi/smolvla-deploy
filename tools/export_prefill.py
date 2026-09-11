#!/usr/bin/env python


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dummy_batch import load_export_raw_batch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

DEFAULT_POLICY_PATH = Path(
    "/home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/smolvla/data/ela3_lerobot_v2_joint")
DEFAULT_REPO_ID = "local/ela3_blue_cube_v2_joint"
DEFAULT_OUTPUT = Path("/home/ubuntu/smolvla/smolvla-deploy/artifacts/vlm_prefill.onnx")


class PrefillWrapper(nn.Module):
    """VLM prefill:输入 prefix 嵌入与 mask,输出堆叠后的 per-layer KV cache."""

    def __init__(self, vlm_with_expert: nn.Module, num_layers: int):
        super().__init__()
        self.vlm = vlm_with_expert
        self.num_layers = num_layers

    def forward(self, prefix_embs, attn_2d_mask, position_ids):
        _, past_key_values = self.vlm.forward(
            attention_mask=attn_2d_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        keys = torch.stack([past_key_values[i]["key_states"] for i in range(self.num_layers)], dim=0)
        values = torch.stack([past_key_values[i]["value_states"] for i in range(self.num_layers)], dim=0)
        return keys, values


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

    print(f"[1/5] 加载 policy: {args.policy_path}")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.eval()
    # 转 fp32 干净基线(VLM 默认 bf16,不利于 ONNX 导出/校验)
    policy.model.float()

    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=str(args.policy_path))

    print("[2/5] 取一帧观测并构造 prefix 嵌入")
    raw_batch = load_export_raw_batch(args.policy_path, args.dataset_root, args.repo_id)
    batch = preprocessor(raw_batch)

    model = policy.model
    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        attn_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    prefix_embs = prefix_embs.float().contiguous()
    print(f"      prefix_embs: {tuple(prefix_embs.shape)}  attn_2d: {tuple(attn_2d.shape)}  "
          f"pos_ids: {tuple(position_ids.shape)}")

    num_layers = model.vlm_with_expert.num_vlm_layers
    wrapper = PrefillWrapper(model.vlm_with_expert, num_layers).eval()

    print("[3/5] PyTorch 参考前向")
    with torch.inference_mode():
        ref_keys, ref_values = wrapper(prefix_embs, attn_2d, position_ids)
    print(f"      keys: {tuple(ref_keys.shape)}  values: {tuple(ref_values.shape)}  ({num_layers} 层)")

    print(f"[4/5] 导出 ONNX -> {args.output}")
    with torch.inference_mode():
        seq = torch.export.Dim("seq")
        torch.onnx.export(
            wrapper,
            (prefix_embs, attn_2d, position_ids),
            str(args.output),
            input_names=["prefix_embs", "attn_2d_mask", "position_ids"],
            output_names=["kv_keys", "kv_values"],
            dynamic_shapes={
                "prefix_embs": {1: seq},
                "attn_2d_mask": {1: seq, 2: seq},
                "position_ids": {1: seq},
            },
            opset_version=args.opset,
            dynamo=True,
        )
    print(f"      文件大小: {args.output.stat().st_size / 1e6:.1f} MB")

    print("[5/5] onnxruntime 验证数值一致性")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_out = sess.run(
        ["kv_keys", "kv_values"],
        {
            "prefix_embs": prefix_embs.cpu().numpy().astype(np.float32),
            "attn_2d_mask": attn_2d.cpu().numpy(),
            "position_ids": position_ids.cpu().numpy().astype(np.int64),
        },
    )
    for name, ref, got in zip(["kv_keys", "kv_values"], [ref_keys, ref_values], ort_out):
        d = float(np.max(np.abs(ref.detach().cpu().numpy() - got)))
        print(f"      {name}: max|diff|={d:.3e}  {'✓' if d < 1e-3 else '✗'}")

    print("\n完成.下一步:导出单步 denoise(action expert),KV cache 作为输入.")


if __name__ == "__main__":
    main()
