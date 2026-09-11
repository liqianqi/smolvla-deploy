#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

import lerobot.policies.smolvla.modeling_smolvla as msv
from dummy_batch import load_export_raw_batch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# 时间正弦编码原本用 float64 算 sin/cos,onnxruntime/TensorRT 的 Cos 不支持 float64.
# 降到 float32(对时间编码精度无实质影响),让导出的 ONNX 可被 ORT/TRT 加载.
_orig_get_safe_dtype = msv.get_safe_dtype
msv.get_safe_dtype = lambda requested, device_type: (
    torch.float32 if requested == torch.float64 else _orig_get_safe_dtype(requested, device_type)
)

DEFAULT_POLICY_PATH = Path(
    "/home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/smolvla/data/ela3_lerobot_v2_joint")
DEFAULT_REPO_ID = "local/ela3_blue_cube_v2_joint"
DEFAULT_OUTPUT = Path("/home/ubuntu/smolvla/smolvla-deploy/artifacts/action_denoise_step.onnx")


class DenoiseWrapper(nn.Module):
    """单步 denoise:KV cache 以显式张量传入,内部重组为 dict 后调 denoise_step."""

    def __init__(self, model: nn.Module, num_layers: int):
        super().__init__()
        self.model = model  # VLAFlowMatching
        self.num_layers = num_layers

    def forward(self, x_t, timestep, kv_keys, kv_values, prefix_pad_masks):
        past_key_values = {
            i: {"key_states": kv_keys[i], "value_states": kv_values[i]}
            for i in range(self.num_layers)
        }
        return self.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=timestep,
        )


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
    policy.model.float()
    model = policy.model

    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=str(args.policy_path))

    print("[2/5] 取一帧观测并 prefill 得到 KV cache(作为示例输入)")
    raw_batch = load_export_raw_batch(args.policy_path, args.dataset_root, args.repo_id)
    batch = preprocessor(raw_batch)

    num_layers = model.vlm_with_expert.num_vlm_layers
    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
        )
        attn_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, pkv = model.vlm_with_expert.forward(
            attention_mask=attn_2d, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs.float(), None], use_cache=True, fill_kv_cache=True,
        )
        kv_keys = torch.stack([pkv[i]["key_states"] for i in range(num_layers)], dim=0).float().contiguous()
        kv_values = torch.stack([pkv[i]["value_states"] for i in range(num_layers)], dim=0).float().contiguous()

    bsize = state.shape[0]
    x_t = torch.randn(
        (bsize, cfg.chunk_size, cfg.max_action_dim), device=args.device, dtype=torch.float32
    )
    timestep = torch.tensor([1.0], device=args.device, dtype=torch.float32).expand(bsize)
    print(f"      x_t: {tuple(x_t.shape)}  kv_keys: {tuple(kv_keys.shape)}  "
          f"prefix_pad_masks: {tuple(prefix_pad_masks.shape)}")

    wrapper = DenoiseWrapper(model, num_layers).eval()

    print("[3/5] PyTorch 参考前向")
    with torch.inference_mode():
        ref_v = wrapper(x_t, timestep, kv_keys, kv_values, prefix_pad_masks)
    print(f"      v_t: {tuple(ref_v.shape)}")

    print(f"[4/5] 导出 ONNX -> {args.output}")
    with torch.inference_mode():
        seq = torch.export.Dim("seq")
        torch.onnx.export(
            wrapper,
            (x_t, timestep, kv_keys, kv_values, prefix_pad_masks),
            str(args.output),
            input_names=["x_t", "timestep", "kv_keys", "kv_values", "prefix_pad_masks"],
            output_names=["v_t"],
            dynamic_shapes={
                "x_t": None,
                "timestep": None,
                "kv_keys": {2: seq},
                "kv_values": {2: seq},
                "prefix_pad_masks": {1: seq},
            },
            opset_version=args.opset,
            dynamo=True,
        )
    print(f"      文件大小: {args.output.stat().st_size / 1e6:.1f} MB")

    print("[5/5] onnxruntime 验证数值一致性")
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_v = sess.run(
        ["v_t"],
        {
            "x_t": x_t.cpu().numpy().astype(np.float32),
            "timestep": timestep.cpu().numpy().astype(np.float32),
            "kv_keys": kv_keys.cpu().numpy().astype(np.float32),
            "kv_values": kv_values.cpu().numpy().astype(np.float32),
            "prefix_pad_masks": prefix_pad_masks.cpu().numpy(),
        },
    )[0]
    d = float(np.max(np.abs(ref_v.detach().cpu().numpy() - ort_v)))
    print(f"      v_t: max|diff|={d:.3e}  {'通过 ✓' if d < 1e-3 else '偏差偏大 ✗'}")

    print("\n完成.三块(vision/prefill/denoise)均已导出,下一步:C++ 编排骨架.")


if __name__ == "__main__":
    main()
