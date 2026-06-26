# SmolVLA 部署 / 推理优化

把训练好的 SmolVLA(PyTorch / safetensors)一步步做成可在边缘设备低延迟运行的推理方案.
本仓库记录 `PyTorch -> ONNX -> TensorRT` 的部署链路,从最干净,收益最大的**视觉编码器**起步.

## 为什么不能整体导出

SmolVLA = SmolVLM(VLM)+ action expert,二者**逐层交织**共享 KV cache,推理还含
**10 步 flow-matching 去噪循环**(动态控制流).整体无法一次性导出 ONNX,必须按模块拆:

| 模块 | 是否易导出 | 部署策略 |
|---|---|---|
| 视觉编码器(SigLIP + connector) | ✅ 纯前向,静态形状 | **ONNX → TensorRT**(本仓库已完成第一步) |
| 交织 Transformer(prefill + cross-attn) | ⚠️ 自定义 KV cache / 手写 attention | torch.compile / torch-tensorrt 子图 |
| flow-matching 去噪循环 | ⚠️ 动态 for 循环 | 自己用 Python/C++ 实现,循环内调上面子模块 |

## 阶段 1(已完成):视觉编码器导出 ONNX

```bash
/home/ubuntu/smolvla/.venv/bin/python export_vision_encoder.py \
  --policy-path /home/ubuntu/smolvla/outputs/train/ela3_smolvla_v5_joint_sd/checkpoints/015000/pretrained_model \
  --output artifacts/vision_encoder.onnx
```

脚本做了 4 件事:加载 policy → 抽出 `vision_model + connector` → `torch.onnx.export`(dynamo 导出器)
→ 用 onnxruntime 验证数值一致并计时.

### 导出时踩的两个坑(已在脚本里解决)

新版 transformers 的 SmolVLM 视觉前向有两处对 ONNX 不友好的算子,脚本通过**手动展开 forward** 绕过,
且在 eager 模式下校验过与原实现 `max|diff| = 0`(数学等价,仅在"方形,无 padding,全 patch 有效"
的部署场景成立--而我们总是 resize 到 512×512):

1. `create_bidirectional_mask`:tracing 时报 `IndexError`.全 patch 有效 ⇒ 全注意力 ⇒ 直接不传 mask.
2. embeddings 里基于 mask 的 `index_put` 位置编码:ONNX 里变成 int64/float 混用的 `Where`,
   onnxruntime/TensorRT 无法加载.全 mask 下 NaViT 桶化位置 id 退化为 `arange(num_patches)`,直接替换.

### 结果

- 输入 `pixel_values` (B, 3, 512, 512),范围 [-1, 1];输出 `image_embeds` (B, 64, 960).
- PyTorch vs ONNX:`max|diff| ≈ 2e-4`(fp32,通过).
- CPU 单帧:PyTorch ~404ms → onnxruntime ~206ms(已约 2×).
- 产物:`artifacts/vision_encoder.onnx`(图)+ `vision_encoder.onnx.data`(fp32 权重,~393MB).

## 分块导出进度(路线②:自己拆引擎 + C++ 编排)

| 块 | 脚本 | 产物 | 验证 (vs PyTorch) |
|---|---|---|---|
| ① 视觉编码器 | `export_vision_encoder.py` | `vision_encoder.onnx` | `max|diff|≈2e-4` ✓ |
| ①b prefix 组装 | `export_prefix_assembler.py` | `prefix_assembler.onnx` | `max|diff|=0` ✓ |
| ② VLM prefill | `export_prefill.py` | `vlm_prefill.onnx` | `max|diff|≈1e-5` ✓ |
| ③ 单步 denoise | `export_denoise.py` | `action_denoise_step.onnx` | `max|diff|≈1.7e-6` ✓ |
| 分块参考实现 | `decomposed_inference.py` | - | 与原 policy `max|diff|=0` ✓ |
| **端到端集成** | `run_onnx_pipeline.py` | - | **ONNX 串联跑 10 步循环 vs 原 policy `max|diff|=3.6e-7`** ✓ |
| **C++ 编排骨架** | `cpp/` | `smolvla_demo` | ONNX Runtime + TRT EP(待 Orin 编译,见 `cpp/README.md`) |

### 导出中解决的关键问题(部署工程师的核心价值)

- **视觉**:`create_bidirectional_mask` tracing 报错,embeddings 的 mask `index_put` 变成 int64/float 混用 `Where`.
  → 在"方形全 patch 有效"场景下手动展开 forward,用全注意力 + `arange` 位置编码替代(eager 下 `max|diff|=0` 验证等价).
- **prefill**:KV cache 是 Python dict,ONNX 无法表达.→ 包一层,堆叠成 `kv_keys/kv_values` 两个显式张量输出.
- **denoise**:KV cache 作为显式张量输入,内部重组 dict;时间正弦编码原用 float64(ORT 的 `Cos` 不支持)→ 降 float32.

### KV cache 接口(C++ 端要持有的中间张量)

```
kv_keys / kv_values: (num_layers=16, B=1, L=prefix_len, n_kv_heads=5, head_dim=64)
prefix_pad_masks:    (B, L) bool
```

## 阶段 2(下一步):编译 TensorRT engine

```bash
# FP16(最常用,精度损失极小)
trtexec --onnx=artifacts/vision_encoder.onnx \
        --saveEngine=artifacts/vision_encoder_fp16.plan \
        --fp16 --shapes=pixel_values:1x3x512x512

# INT8(需准备校准数据,进一步压缩 + 提速)
# trtexec --onnx=... --int8 --calib=<calibration cache> ...
```

之后用 TensorRT runtime(Python 或 C++)加载 `.plan`,替换推理时的 `embed_image`.

## 环境

复用 `/home/ubuntu/smolvla/.venv`(torch 2.11 + cu128,RTX 4090).额外依赖见 `requirements.txt`
(`onnx`,`onnxruntime`,`onnxscript`).
