# SmolVLA 部署

把训练好的 SmolVLA(PyTorch)拆成 4 个 ONNX 引擎,用 C++(ONNX Runtime + TensorRT EP)编排,
在实机上跑通「相机采集 → 推理 → EL-A3 机械臂控制」的完整闭环。

## 1. 导出(Python,一次性)

```bash
PY=/home/ubuntu/AI/deploy_vla/.smolvla_venv/bin/python

# 4 个推理引擎 → artifacts/*.onnx(默认读训练 checkpoint,可用 --policy-path 覆盖)
$PY tools/export_vision_encoder.py     # ① 视觉编码器(SigLIP + connector)
$PY tools/export_prefix_assembler.py   # ② prefix 组装(图像/语言/状态 → prefix embs)
$PY tools/export_prefill.py            # ③ VLM prefill(输出显式 KV cache)
$PY tools/export_denoise.py            # ④ 单步 flow-matching 去噪

# 部署所需的二进制资产 → artifacts/*.bin
$PY tools/export_lang_tokens.py --instruction "pick up the blue block"  # 换指令时重跑
$PY tools/export_state_stats.py        # state 归一化统计
$PY tools/export_action_stats.py       # action 反归一化统计

# 端到端校验:4 引擎 ONNX 串联 vs 原 policy
$PY tools/run_onnx_pipeline.py         # max|diff| = 3.6e-7 ✓
```

## 2. 编译部署程序(C++)

第三方依赖准备(ONNX Runtime、cuDNN,一次性)见 [cpp/README.md](cpp/README.md)。

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build . -j        # 产出 smolvla_demo + smolvla_deploy
```

## 3. 实机运行

```bash
# 配置 CAN(机械臂 + 夹爪都在 can0)
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

# 在 cpp/build 下运行(默认 can0、USB 相机设备号 10、夹爪 CAN id=7)
./smolvla_deploy --artifacts ../../artifacts
#   -> 零力矩+重力补偿,手动摆位后按 Enter,加载完成后自动开始推理抓取

# 调试模式
./smolvla_deploy --artifacts ../../artifacts --no-arm                 # 无机械臂,只打印动作
./smolvla_deploy --artifacts ../../artifacts --no-arm --fake-cameras  # 纯推理冒烟测试
```

运行参数与依赖细节见 [cpp/README.md](cpp/README.md)。

## 验证结果

- 4 引擎 ONNX 串联 vs 原 policy:`max|diff| = 3.6e-7`
- C++ 图像预处理 vs lerobot `resize_with_pad`:`max|diff| ≈ 0.006`(uint8 量化误差)
- 稳态单次推理 ~40ms(RTX 5060 Laptop,TRT FP16);首次运行会 build engine 并缓存到
  `artifacts/.trt_cache`(约 3-4 分钟),二次启动秒级

## 为什么拆成 4 块

SmolVLA = SmolVLM + action expert 逐层交织共享 KV cache,推理含 10 步 flow-matching
去噪循环(动态控制流),无法整体导出 ONNX。因此拆成 4 个静态图,去噪循环由 C++ 编排;
KV cache 以显式张量(`kv_keys/kv_values`,16×1×L×5×64)在引擎间传递。
导出中绕过的算子问题(mask tracing、`index_put` 位置编码、float64 正弦编码等)见各
`tools/export_*.py` 脚本内注释。
