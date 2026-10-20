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

# (可选)单图 ONNX:整条 sample_actions 一个图 → artifacts/smolvla_full.onnx,自带校验
$PY tools/export_full_onnx.py          # 约 4-5 分钟, max|diff| = 2.6e-6 ✓
```

## 2. 编译部署程序(C++)

第三方依赖准备(ONNX Runtime、cuDNN,一次性)见 [cpp/README.md](cpp/README.md)。

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build . -j        # 产出 smolvla_deploy(+ _raw / _full)+ smolvla_bench
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

# 原样执行版(无偏置/滤波): 4 引擎 / 单图 ONNX 两种推理后端, 参数完全相同
./smolvla_deploy_raw  --artifacts ../../artifacts
./smolvla_deploy_raw  --artifacts ../../artifacts --fast   # 优化版运行时(显存常驻 + 视觉合批 + 去噪 CUDA Graph, FP32)
./smolvla_deploy_full --artifacts ../../artifacts   # 需先运行 tools/export_full_onnx.py

# 运行时基准: 原版 vs 优化版(同输入, 输出分阶段耗时与数值差异)
./smolvla_bench --artifacts ../../artifacts
```

运行参数与依赖细节见 [cpp/README.md](cpp/README.md)。

## 验证结果

- 4 引擎 ONNX 串联 vs 原 policy:`max|diff| = 3.6e-7`
- 单图 ONNX vs 4 引擎(C++, TRT FP32, 同观测同噪声, 反归一化后):`max|diff| ≈ 4e-4 rad`;
  单次推理 4 引擎 ~117ms / 单图 ~146ms(RTX 5060 Laptop)
- 4 引擎优化版(`--fast`, TRT FP32)vs 原版:同输入输出一致(`max|diff| = 0`),单帧 ~156ms → ~118ms;
  分阶段 视觉 ~67 / 组装 ~0.6 / prefill ~13 / 去噪×10 ~37ms。去噪开 CUDA Graph 无可测收益(GPU 计算受限)
- C++ 图像预处理 vs lerobot `resize_with_pad`:`max|diff| ≈ 0.006`(uint8 量化误差)
- 首次运行会 build TensorRT engine 并缓存到 `artifacts/.trt_cache`(约 3-4 分钟),二次启动秒级。
  (早期记录的 ~40ms 来自已删除的 smolvla_demo 全局 FP16 配置,该配置会打崩输出,不可作为可用速度参考)

## 为什么拆成 4 块

SmolVLA = SmolVLM + action expert 逐层交织共享 KV cache,推理含 10 步 flow-matching
去噪循环。整体导出时 torch.export 会把循环展开(视觉 ×3、去噪 ×10, 约 1.2 万节点),
步数/相机数被写死在图里, 也无法单独调优或复用某一段。因此默认拆成 4 个静态图,去噪循环由 C++ 编排;
KV cache 以显式张量(`kv_keys/kv_values`,16×1×L×5×64)在引擎间传递。
单图方案(`smolvla_full.onnx` + `smolvla_deploy_full`)作为对照保留, 输出与 4 引擎一致。
导出中绕过的算子问题(mask tracing、`index_put` 位置编码、float64 正弦编码等)见各
`tools/export_*.py` 脚本内注释。
