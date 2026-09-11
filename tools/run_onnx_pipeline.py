#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import lerobot.policies.smolvla.modeling_smolvla as msv
from dummy_batch import load_export_raw_batch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

# 与 export_denoise 一致:时间编码用 float32(onnxruntime 的 Cos 不支持 float64)
_orig = msv.get_safe_dtype
msv.get_safe_dtype = lambda req, dt: torch.float32 if req == torch.float64 else _orig(req, dt)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = Path(
    "/home/ubuntu/AI/deploy_vla/ela3_smolvla_new_dataset_v3/checkpoints/015000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/smolvla/lerobot_dataset")
DEFAULT_REPO_ID = "local/ela3_blue_block_new"
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    art = args.artifacts

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[1/5] 加载 policy(fp32)+ 预处理器")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.eval()
    policy.model.float()
    policy.reset()
    preprocessor, _ = make_pre_post_processors(cfg, pretrained_path=str(args.policy_path))

    print(f"[2/5] 加载 prefill / denoise ONNX 引擎(onnxruntime CPU) <- {art}")
    prefill = ort.InferenceSession(str(art / "vlm_prefill.onnx"), providers=["CPUExecutionProvider"])
    denoise = ort.InferenceSession(str(art / "action_denoise_step.onnx"), providers=["CPUExecutionProvider"])

    print("[3/5] 取一帧观测,组装 prefix(轻量 glue)")
    raw = load_export_raw_batch(args.policy_path, args.dataset_root, args.repo_id)
    batch = preprocessor(raw)

    model = policy.model
    num_layers = model.vlm_with_expert.num_vlm_layers
    chunk, adim = cfg.chunk_size, cfg.max_action_dim

    with torch.inference_mode():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state=state
        )
        attn_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # 固定 noise,供参考侧与 ONNX 侧共用
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn((1, chunk, adim), generator=gen, device=device, dtype=torch.float32)

    print("[4/5] ONNX 编排:prefill -> 10 步 denoise 去噪循环")
    np_prefix = prefix_embs.float().cpu().numpy()
    np_attn = attn_2d.cpu().numpy()
    np_pos = position_ids.cpu().numpy().astype(np.int64)
    np_pad = prefix_pad_masks.cpu().numpy()

    kv_keys, kv_values = prefill.run(
        ["kv_keys", "kv_values"],
        {"prefix_embs": np_prefix, "attn_2d_mask": np_attn, "position_ids": np_pos},
    )

    num_steps = cfg.num_steps
    dt = -1.0 / num_steps
    x_t = noise.cpu().numpy()
    for step in range(num_steps):
        t = np.float32(1.0 + step * dt)
        v_t = denoise.run(
            ["v_t"],
            {
                "x_t": x_t.astype(np.float32),
                "timestep": np.array([t], dtype=np.float32),
                "kv_keys": kv_keys,
                "kv_values": kv_values,
                "prefix_pad_masks": np_pad,
            },
        )[0]
        x_t = x_t + dt * v_t
    onnx_actions = x_t[:, :, : policy.config.action_feature.shape[0]]

    print("[5/5] 对比原 policy.predict_action_chunk")
    with torch.inference_mode():
        ref = policy.predict_action_chunk(batch, noise=noise.clone()).cpu().numpy()

    max_abs = float(np.max(np.abs(ref - onnx_actions)))
    mean_abs = float(np.mean(np.abs(ref - onnx_actions)))
    print(f"      ONNX 端到端 vs 原 policy  max|diff|={max_abs:.3e}  mean|diff|={mean_abs:.3e}")
    print(f"      一致性: {'通过 ✓' if max_abs < 5e-3 else '偏差偏大,需排查 ✗'}")
    print("\n      第 1 个动作(前 7 维):")
    print("      ONNX     :", onnx_actions[0, 0])
    print("      参考      :", ref[0, 0])


if __name__ == "__main__":
    main()
