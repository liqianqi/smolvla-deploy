#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import lerobot.policies.smolvla.modeling_smolvla as msv
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# 与导出口径一致: 时间编码 float32
_orig = msv.get_safe_dtype
msv.get_safe_dtype = lambda req, dt: torch.float32 if req == torch.float64 else _orig(req, dt)

POLICY = Path("/home/ubuntu/AI/deploy_vla/020000/pretrained_model")
ART = Path(__file__).resolve().parent.parent / "artifacts"


def load_img(p: str) -> torch.Tensor:
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0  # CHW [0,1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="/tmp/obs/primary_camera1.jpg")
    ap.add_argument("--wrist", default="/tmp/obs/wrist_camera2.jpg")
    ap.add_argument("--state", default="0.0864,1.6489,-1.4201,1.0095,-0.0449,-0.0288,1.0")
    ap.add_argument("--instruction", default="pick up the blue block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-noise", type=Path, default=None,
                    help="把本次使用的 noise 存为 float32 bin, 供 C++ --test-noise 加载")
    ap.add_argument("--cpp-actions", type=Path, default=None,
                    help="C++ --dump-actions 输出的 bin, 与 PyTorch 参考对比")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state7 = np.array([float(x) for x in args.state.split(",")], dtype=np.float64)
    assert state7.shape[0] == 7

    print("[1/4] 加载 PyTorch policy")
    cfg = PreTrainedConfig.from_pretrained(POLICY)
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(POLICY, config=cfg)
    policy.eval()
    policy.model.float()
    policy.reset()
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(POLICY))

    raw = {
        "task": args.instruction,
        "observation.state": torch.as_tensor(state7, dtype=torch.float32).unsqueeze(0),
        "observation.images.image": load_img(args.primary).unsqueeze(0),
        "observation.images.wrist_image": load_img(args.wrist).unsqueeze(0),
    }
    raw["task"] = [raw["task"]]
    batch = pre(raw)

    chunk, adim = cfg.chunk_size, cfg.max_action_dim
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((1, chunk, adim), generator=gen, device=device, dtype=torch.float32)
    if args.dump_noise is not None:
        noise.cpu().numpy().astype("<f4").tofile(args.dump_noise)
        print(f"noise 已写出 {args.dump_noise}")

    print("[2/4] PyTorch predict_action_chunk")
    with torch.inference_mode():
        ref = policy.predict_action_chunk(batch, noise=noise.clone())
        ref = post(ref).squeeze(0).cpu().numpy()

    print("[3/4] ONNX 4 引擎编排(与 C++ runtime 相同)")
    import onnxruntime as ort

    prov = ["CPUExecutionProvider"]
    vision = ort.InferenceSession(str(ART / "vision_encoder.onnx"), providers=prov)
    assembler = ort.InferenceSession(str(ART / "prefix_assembler.onnx"), providers=prov)
    prefill = ort.InferenceSession(str(ART / "vlm_prefill.onnx"), providers=prov)
    denoise = ort.InferenceSession(str(ART / "action_denoise_step.onnx"), providers=prov)

    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)  # 用 policy 的预处理保证输入一致
        state32 = policy.prepare_state(batch)

    ie = np.stack([vision.run(["image_embeds"], {"pixel_values": im.cpu().numpy().astype(np.float32)})[0]
                   for im in images])  # [ni,1,nt,h]
    outs = assembler.run(
        ["prefix_embs", "prefix_pad_masks", "attn_2d_mask", "position_ids"],
        {
            "image_embeds": ie,
            "img_masks": np.stack([m.cpu().numpy() for m in img_masks]).astype(bool),
            "lang_tokens": batch[OBS_LANGUAGE_TOKENS].cpu().numpy().astype(np.int64),
            "lang_masks": batch[OBS_LANGUAGE_ATTENTION_MASK].cpu().numpy().astype(bool),
            "state": state32.cpu().numpy().astype(np.float32),
        },
    )
    prefix_embs, prefix_pad, attn2d, pos = outs
    kv_keys, kv_values = prefill.run(
        ["kv_keys", "kv_values"],
        {"prefix_embs": prefix_embs, "attn_2d_mask": attn2d, "position_ids": pos.astype(np.int64)},
    )
    num_steps = cfg.num_steps
    dt = -1.0 / num_steps
    x_t = noise.cpu().numpy()
    for step in range(num_steps):
        t = np.float32(1.0 + step * dt)
        v_t = denoise.run(["v_t"], {
            "x_t": x_t.astype(np.float32), "timestep": np.array([t], dtype=np.float32),
            "kv_keys": kv_keys, "kv_values": kv_values, "prefix_pad_masks": prefix_pad,
        })[0]
        x_t = x_t + dt * v_t
    onnx_norm = x_t[0, :, :7]

    # 反归一化(C++ 用 action_norm_stats.bin)
    stats = np.fromfile(ART / "action_norm_stats.bin", dtype="<f4")
    a_mean, a_std = stats[:7], stats[7:14]
    onnx_act = onnx_norm * a_std + a_mean

    print("[4/4] 对比")
    diff = np.abs(ref - onnx_act)
    print(f"  max|diff|={diff.max():.4e}  mean|diff|={diff.mean():.4e}")
    print(f"  state7 = {np.round(state7, 4).tolist()}")
    print("  前 5 步动作 (PyTorch | ONNX):")
    for i in range(5):
        print(f"   [{i}] {np.round(ref[i], 4).tolist()}")
        print(f"       {np.round(onnx_act[i], 4).tolist()}")
    print("  chunk 末 3 步 (PyTorch):")
    for i in range(cfg.chunk_size - 3, cfg.chunk_size):
        print(f"   [{i}] {np.round(ref[i], 4).tolist()}")

    if args.cpp_actions is not None:
        cpp = np.fromfile(args.cpp_actions, dtype="<f4").reshape(cfg.chunk_size, 7)
        d2 = np.abs(ref - cpp)
        print(f"\n  C++ 实机引擎 vs PyTorch:  max|diff|={d2.max():.4e}  mean|diff|={d2.mean():.4e}")
        worst = np.unravel_index(d2.argmax(), d2.shape)
        print(f"  最大偏差位置: step={worst[0]} dim={worst[1]}  "
              f"PyTorch={ref[worst]:.4f}  C++={cpp[worst]:.4f}")
        print("  C++ 前 5 步动作:")
        for i in range(5):
            print(f"   [{i}] {np.round(cpp[i], 4).tolist()}")


if __name__ == "__main__":
    main()
