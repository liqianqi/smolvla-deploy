#!/usr/bin/env python
"""SmolVLA 推理的"分块"参考实现(脱离 LeRobot 的 sample_actions 封装).

目的(路线②的第①步):
  把一次推理显式拆成 4 个"可独立导出/可被 C++ 编排"的块,自己驱动它们跑出动作,
  并验证结果与原 policy.predict_action_chunk 完全一致(用同一份固定 noise).
  这是后续逐块导出 ONNX/TensorRT,以及用 C++ 写去噪循环的"黄金参考".

四个块(对应未来的引擎 / C++ 职责):
  Block 1  视觉编码器 embed_image         -> 已导出 ONNX,未来 TensorRT engine
  Block 2  prefix 组装 embed_prefix       -> 视觉emb + 语言emb(查表) + state_proj,拼成序列
  Block 3  VLM prefill (fill_kv_cache)     -> 16 层文本 transformer,产出 per-layer KV cache
  Block 4  flow-matching 去噪循环 x10      -> 每步 denoise_step(action expert) + 欧拉更新
                                              (循环本身将由 C++ 实现)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

DEFAULT_POLICY_PATH = Path(
    "/home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/pretrained_model"
)
DEFAULT_DATASET_ROOT = Path("/home/ubuntu/smolvla/data/ela3_lerobot_v2_joint")
DEFAULT_REPO_ID = "local/ela3_blue_cube_v2_joint"


def add_batch_dim(value):
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    return [value]


@torch.inference_mode()
def decomposed_sample_actions(policy: SmolVLAPolicy, batch: dict, noise: torch.Tensor) -> torch.Tensor:
    """显式分块复现 VLAFlowMatching.sample_actions,返回 (B, chunk_size, max_action_dim)."""
    model = policy.model
    cfg = policy.config

    # ---- 输入预备:这些张量在 C++ 端由 tokenizer + 预处理产生 ----
    images, img_masks = policy.prepare_images(batch)  # 内部对每路图像做 resize/归一化
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    device = state.device
    bsize = state.shape[0]

    # ===== Block 1+2: prefix 嵌入 =====
    # embed_prefix 内部依次做:
    #   - 对每路 image 调 embed_image(Block1,视觉编码器 -> ONNX/TRT)
    #   - 语言 token 查表 embed_language_tokens
    #   - state_proj 投影
    #   - 拼接成 prefix 序列,并生成 pad/attention mask
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    # ===== Block 3: VLM prefill,产出 per-layer KV cache =====
    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=cfg.use_cache,
        fill_kv_cache=True,
    )

    # ===== Block 4: flow-matching 去噪循环(这部分将由 C++ 控制流实现)=====
    num_steps = cfg.num_steps
    dt = -1.0 / num_steps
    x_t = noise
    for step in range(num_steps):
        t = 1.0 + step * dt
        t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
        # denoise_step 内部:embed_suffix(噪声动作+时间) -> action expert cross-attn 读 KV cache -> v_t
        v_t = model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=t_tensor,
        )
        x_t = x_t + dt * v_t  # 欧拉积分一步

    return x_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[1/4] 加载 policy: {args.policy_path}")
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.device = args.device
    policy = SmolVLAPolicy.from_pretrained(args.policy_path, config=cfg)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=str(args.policy_path)
    )

    print(f"[2/4] 取一帧观测: {args.repo_id} frame={args.frame_index}")
    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    sample = dataset[args.frame_index]
    raw_batch = {
        "task": sample["task"],
        "observation.state": sample["observation.state"],
        "observation.images.image": sample["observation.images.image"],
        "observation.images.wrist_image": sample["observation.images.wrist_image"],
    }
    raw_batch = {k: add_batch_dim(v) for k, v in raw_batch.items()}
    model_batch = preprocessor(raw_batch)

    # 固定 noise,保证两条路径可逐元素对比
    chunk_size = policy.config.chunk_size
    max_action_dim = policy.config.max_action_dim
    gen = torch.Generator(device=args.device).manual_seed(args.seed)
    noise = torch.randn(
        (1, chunk_size, max_action_dim), generator=gen, device=args.device, dtype=torch.float32
    )

    print("[3/4] 原 policy.predict_action_chunk(参考)")
    with torch.inference_mode():
        ref = policy.predict_action_chunk(model_batch, noise=noise.clone())
    print(f"      参考输出形状: {tuple(ref.shape)}")

    print("[4/4] 分块推理 decomposed_sample_actions")
    with torch.inference_mode():
        out = decomposed_sample_actions(policy, model_batch, noise.clone())
        out = out[:, :, : policy.config.action_feature.shape[0]]  # unpad 到真实动作维度

    max_abs = float(torch.max(torch.abs(ref - out)))
    mean_abs = float(torch.mean(torch.abs(ref - out)))
    print(f"      分块 vs 原 policy  max|diff|={max_abs:.3e}  mean|diff|={mean_abs:.3e}")
    print(f"      一致性: {'通过 ✓' if max_abs < 1e-4 else '偏差偏大,需排查 ✗'}")

    print("\n      第 1 个动作(前 7 维):")
    print("      ", out[0, 0].detach().cpu().numpy())


if __name__ == "__main__":
    main()
