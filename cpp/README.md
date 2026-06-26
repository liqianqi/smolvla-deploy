# SmolVLA C++ 推理运行时(骨架)

用 **ONNX Runtime C++ API + TensorRT Execution Provider** 编排 4 个 ONNX 引擎,
复刻已验证的 Python 管线(`../run_onnx_pipeline.py`,与原 policy `max|diff|=3.6e-7`).

## 编排流程

```
相机图像 ──[vision_encoder.onnx]──► image_embeds ─┐
语言指令 ──[tokenizer]──► lang_tokens ────────────┤
机器人状态 ──────────────────────────────────────┴─[prefix_assembler.onnx]─► prefix_embs/masks
                                                       │
                                          [vlm_prefill.onnx] ─► KV cache (kv_keys/kv_values)
                                                       │
        x_t = noise; 重复 10 步: v_t=[action_denoise_step.onnx](x_t,t,KV); x_t += dt·v_t
                                                       │
                                                  取前 7 维 ─► 动作
```

## 文件

- `include/smolvla_runtime.hpp` - `SmolVLARuntime` 类 + 维度常量(取自 checkpoint config)
- `src/smolvla_runtime.cpp` - 4 引擎加载 + 编排 + 欧拉去噪循环
- `src/main.cpp` - demo 入口(含 `[TODO-CAMERA]` / `[TODO-TOKENIZER]` 占位)

## 构建

需要带 TensorRT EP 的 ONNX Runtime(本机用官方 `onnxruntime-linux-x64-gpu` 包;
Jetson Orin 用 NVIDIA 提供的 onnxruntime-gpu,匹配 JetPack 的 CUDA/TensorRT).

```bash
cd cpp && mkdir build && cd build
cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
cmake --build . -j
./smolvla_demo ../../artifacts
```

首次运行 TensorRT EP 会 build engine 并缓存到 `artifacts/.trt_cache`(~30-90s),二次启动秒级.

## 仍需补齐的两块(真实部署)

1. **图像预处理**:相机帧 → resize+pad 到 512×512 → 归一化到 [-1,1](对应 Python `resize_with_pad` + `*2-1`).
2. **Tokenizer**:SmolVLM2 tokenizer 的 C++ 实现.选项:HuggingFace `tokenizers` 的 Rust/C++ 绑定,或离线 tokenize 后传入.

## 验证建议

把 `main.cpp` 的占位换成 `../run_onnx_pipeline.py` 用的同一帧观测(可让 Python 侧把预处理后的
`images/lang_tokens/state` 存成二进制),对比 C++ 与 Python 的动作输出应当一致.
