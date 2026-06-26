#!/usr/bin/env python
"""把 SmolVLA 的视觉编码器(SigLIP vision_model + connector)导出为 ONNX.

为什么只导视觉编码器?
  SmolVLA 整体结构里 VLM 与 action expert 是逐层交织的,并且推理含 10 步
  flow-matching 去噪循环(动态控制流),无法整体一次性导出 ONNX.
  而视觉编码器是纯前向,输入形状固定(resize 到 512x512),是最干净,
  收益最大的可导出子模块,所以作为 PyTorch -> ONNX -> TensorRT 链路的第一步.

流程:
  1. 加载训练好的 SmolVLA policy(safetensors / PyTorch 权重).
  2. 抽出 vision_model + connector 组成一个薄包装模块.
  3. torch.onnx.export 导出为 .onnx(fp32,作为干净基线).
  4. 用 onnxruntime 重新加载,和 PyTorch 输出逐元素对比,验证数值一致.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

DEFAULT_POLICY_PATH = Path(
    "/home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/pretrained_model"
)
DEFAULT_OUTPUT = Path("/home/ubuntu/smolvla/smolvla-deploy/artifacts/vision_encoder.onnx")


class VisionEncoderWrapper(nn.Module):
    """只包含 SmolVLA 推理时 `embed_image` 用到的两步:vision_model -> connector.

    输入:pixel_values,形状 (B, 3, H, W),数值范围 [-1, 1](SigLIP 约定,
          对应 SmolVLAPolicy.prepare_images 里 `img * 2 - 1` 之后的结果).
    输出:图像 token 的隐藏状态 (B, num_tokens, hidden).
    """

    def __init__(self, vlm_with_expert: nn.Module):
        super().__init__()
        vlm = vlm_with_expert.get_vlm_model()
        self.vision_model = vlm.vision_model
        self.connector = vlm.connector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # 手动展开 vision_model.forward,针对"方形,无 padding,全 patch 有效"的部署场景做两处简化,
        # 以获得对 ONNX 友好的静态图:
        #   1) 跳过 create_bidirectional_mask(全注意力 == 不传 mask).
        #   2) embeddings 的位置编码:全 mask 下 NaViT 的桶化位置 id 退化为标准光栅顺序
        #      arange(num_patches),从而绕开原实现里基于 mask 的 index_put(ONNX 里会变成
        #      int64/float 混用的 Where,无法被 onnxruntime/TensorRT 加载).
        vm = self.vision_model
        emb = vm.embeddings

        patch_embeds = emb.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        num_patches = embeddings.shape[1]
        position_ids = torch.arange(num_patches, device=pixel_values.device).unsqueeze(0)
        embeddings = embeddings + emb.position_embedding(position_ids)

        encoder_out = vm.encoder(inputs_embeds=embeddings, attention_mask=None)
        last_hidden = vm.post_layernorm(encoder_out.last_hidden_state)
        return self.connector(last_hidden)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=512, help="resize_imgs_with_padding 的边长")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--exporter",
        type=str,
        default="dynamo",
        choices=["dynamo", "legacy"],
        help="dynamo=新版 torch.export 导出器(推荐);legacy=旧版 TorchScript 导出器",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="导出建议用 cpu(最稳);验证也在 cpu 上,避免 CUDA 算子差异.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 加载 policy: {args.policy_path}")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.eval()

    # 抽出视觉编码器并转成 fp32(干净基线;FP16/INT8 量化是后续 TensorRT 阶段的事)
    encoder = VisionEncoderWrapper(policy.model.vlm_with_expert).to(args.device).float().eval()

    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"      视觉编码器参数量: {num_params / 1e6:.1f}M")

    # 构造一个符合约定的样例输入:(1, 3, H, W),范围 [-1, 1]
    dummy = (torch.rand(1, 3, args.image_size, args.image_size, device=args.device) * 2 - 1).float()

    print("[2/4] PyTorch 前向(作为参考输出)")
    with torch.inference_mode():
        ref_out = encoder(dummy)
        # 校验:简化版(无 mask) 与 原始 vision_model(带 mask) 在 eager 模式下是否等价
        vm = encoder.vision_model
        orig_hidden = vm(pixel_values=dummy, patch_attention_mask=None).last_hidden_state
        orig_out = encoder.connector(orig_hidden)
    simplify_diff = float(torch.max(torch.abs(ref_out - orig_out)))
    print(f"      输出形状: {tuple(ref_out.shape)}  dtype={ref_out.dtype}")
    print(f"      简化(无mask) vs 原始(带mask)  max|diff|={simplify_diff:.3e}  "
          f"({'等价 ✓' if simplify_diff < 1e-4 else '不等价 ✗'})")

    print(f"[3/4] 导出 ONNX({args.exporter} 导出器)-> {args.output}")
    with torch.inference_mode():
        if args.exporter == "dynamo":
            # 新版 torch.export-based 导出器,对 transformers 的动态算子更友好
            batch = torch.export.Dim("batch")
            torch.onnx.export(
                encoder,
                (dummy,),
                str(args.output),
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                dynamic_shapes={"pixel_values": {0: batch}},
                opset_version=args.opset,
                dynamo=True,
            )
        else:
            torch.onnx.export(
                encoder,
                (dummy,),
                str(args.output),
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "image_embeds": {0: "batch"},
                },
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,
            )

    size_mb = args.output.stat().st_size / 1e6
    print(f"      导出完成,文件大小: {size_mb:.1f} MB")

    print("[4/4] 用 onnxruntime 验证数值一致性")
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(str(args.output))

    sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_out = sess.run(["image_embeds"], {"pixel_values": dummy.cpu().numpy().astype(np.float32)})[0]

    ref_np = ref_out.detach().cpu().numpy()
    max_abs = float(np.max(np.abs(ref_np - ort_out)))
    mean_abs = float(np.mean(np.abs(ref_np - ort_out)))
    print(f"      PyTorch vs ONNX  max|diff|={max_abs:.3e}  mean|diff|={mean_abs:.3e}")
    ok = max_abs < 1e-3
    print(f"      数值一致性: {'通过 ✓' if ok else '偏差偏大,需排查 ✗'}")

    # 简单计时对比(CPU 上,仅作直观参考;真正加速看 TensorRT 阶段)
    def bench(fn, n=10):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000

    with torch.inference_mode():
        pt_ms = bench(lambda: encoder(dummy))
    ort_in = dummy.cpu().numpy().astype(np.float32)
    ort_ms = bench(lambda: sess.run(["image_embeds"], {"pixel_values": ort_in}))
    print(f"      单帧延迟(CPU): PyTorch {pt_ms:.1f} ms | onnxruntime {ort_ms:.1f} ms")

    print("\n完成.下一步:用 trtexec/TensorRT 把该 ONNX 编译成 FP16/INT8 engine.")


if __name__ == "__main__":
    main()
