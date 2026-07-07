# SmolVLA C++ 推理运行时 + 实机部署

用 **ONNX Runtime C++ API + TensorRT Execution Provider** 编排 4 个 ONNX 引擎,
复刻已验证的 Python 管线(`../run_onnx_pipeline.py`,与原 policy `max|diff|=3.6e-7`).

包含两个可执行文件:

- `smolvla_demo`:推理 demo(占位输入,验证引擎链路)
- `smolvla_deploy`:**实机部署**——RealSense + USB 相机采集 + RobStride 电机控制的完整闭环,
  复刻 `deploy_vla` 的 `run_ela3_smolvla_inference.py --mode deploy`

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
- `src/main.cpp` - demo 入口(占位输入)
- `src/deploy_main.cpp` - **实机部署入口**(完整控制回路)
- `include/deploy_camera.hpp` / `src/deploy_camera.cpp` - RealSense(彩色流 RGB8)+ USB UVC(V4L2+MJPG,BGR→RGB)
- `include/image_preproc.hpp` / `src/image_preproc.cpp` - lerobot `resize_with_pad`(512,左/上 pad)+ `[0,1]→[-1,1]`(已对 PyTorch 验证 `max|diff|≈0.006`,来自 uint8 量化)
- `include/arm_ela3.hpp` / `src/arm_ela3.cpp` - EL-A3 机械臂(自实现 RobStride SocketCAN MIT 帧协议:上电清 fault、零力矩+重力补偿手动摆位、启动保持、kp=80/kd=3 伺服、标定/限位/限速)
- `../tools/export_lang_tokens.py` - 离线 tokenize 指令 → `artifacts/lang_tokens.bin`(换指令重跑即可)
- `../tools/export_state_stats.py` - 导出 state 归一化统计 → `artifacts/state_norm_stats.bin`

## 实机部署控制回路(与 Python 部署对齐)

```
RealSense 640x480 RGB(主视角)┐
USB      640x480 RGB(腕部)  ├─ resize_with_pad 512 + [-1,1] ─┐
第 3 路 empty camera(-1)     ┘                                ├─► SmolVLARuntime ─► 50 步 chunk
[J1..J6, gripper_open] ─ MEAN_STD 归一化 ─ pad 32 ────────────┤        │
lang_tokens.bin(48 tokens)────────────────────────────────────┘        ▼
                    执行前 exec_horizon=8 步:每步 0.1s @30Hz 线性插值下发
                    (URDF 限位 clip、单步跳变 ≤0.25 rad、每 tick ≤0.08 rad)
```

### 启动流程(与 Python `--manual_init_before_model` 一致)

1. 上电清 fault(disable×2 → enable)
2. **零力矩 + 重力补偿**(kp=0,J1..J6 前馈力矩默认 `0,1.2,-1.2,0.1,0,0` Nm),手动把机械臂摆到起始姿态
3. **按 Enter** → 捕获当前关节角,加载模型/相机期间以 kp=80/kd=5 保持(防下垂)
4. 就绪后自动释放保持,进入模型推理与抓取控制回路

关键默认参数:CAN=`can0`、USB 设备号 10、电机 ID 1-6、标定方向 `(-1,1,-1,1,-1,1)`、
`exec_horizon=8`、`action_dt=0.1s`(与 Python 实机成功配置 `--exec_horizon 8 --action_dt 0.1` 一致;
只执行前 2 步会一直在 chunk 轨迹起点打转、原地晃动)、`control_hz=30`、伺服 kp=80/kd=3(MIT 力位混合帧)。
夹爪暂不控制(第 7 维状态跟随模型输出,与 Python `--gripper_id None` 一致)。

**动作限幅**(避免运动剧烈,均可调):
- `--max-dq-per-action`(默认 0.25 rad):每个模型动作相对当前关节角的最大跳变
- `--max-dq-per-tick`(默认 0.08 rad):30Hz 插值下发时每 tick 的最大步长
- 另有 URDF 关节限位硬 clip

## 依赖

| 依赖 | 是否必需 | CMake 开关 | 用途 | 获取方式 |
|---|---|---|---|---|
| **ONNX Runtime**(带 TensorRT/CUDA EP) | 必需 | `-DONNXRUNTIME_ROOT=...` | 4 引擎推理后端 | x64:官方 `onnxruntime-linux-x64-gpu_cuda12`;Orin:NVIDIA 的 onnxruntime-gpu(匹配 JetPack) |
| **cuDNN 9** | GPU 必需 | `-DCUDNN_LIB_DIR=...` | CUDA/TensorRT EP 运行时依赖(x64 官方 ORT 包不自带) | `pip download nvidia-cudnn-cu12` 解压,或系统安装;放到 `third_party/cudnn/lib` |
| **CUDA / TensorRT**(运行时) | GPU 必需 | 走系统环境 | EP 底层库 | 系统已装(本机 `/usr/local/cuda-12.8`、`/usr/local/TensorRT-10.10.0.31`,已在 `.zshrc` 导出) |
| **OpenCV** | deploy 必需 | `-DSMOLVLA_WITH_DEPLOY=ON`(默认开) | 图像预处理 + USB 相机 | 系统已装(apt 4.5.4);Orin 用 JetPack 自带 |
| **librealsense2** | deploy 必需 | 同上 | RealSense 主视角相机 | 本机由 ROS Humble 提供(`ros-humble-librealsense2` 2.57.7);或 Intel 官方 apt 源 |
| **SocketCAN**(内核) | deploy 必需 | 无 | RobStride 电机 MIT 帧协议(自实现于 `arm_ela3.cpp`,无外部驱动库依赖) | Linux 内核自带,`ip link set can0 up` 即可 |
| **tokenizers-cpp** | 可选 | `-DSMOLVLA_WITH_TOKENIZERS=ON` | C++ 内 tokenize 任意指令(当前用离线 `lang_tokens.bin` 替代,无需打开) | 克隆 [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp)(需 Rust/cargo) |
| **TensorRT / CUDA**(直连) | 可选 | `-DSMOLVLA_WITH_TENSORRT=ON` | 阶段2 直接用 TRT C++ runtime 加载 `.plan`;走 ORT 的 TRT EP 时**无需**打开 | 随 CUDA/TensorRT 安装或 JetPack 自带 |

平台由 CMake 自动探测(`x86_64` vs `aarch64`),分别对应本机 GPU 与 Jetson Orin。

### 准备 third_party

```bash
cd cpp/..                       # 仓库根目录
mkdir -p third_party

# 1) ONNX Runtime(必需)—— x64 + CUDA 12 包(本机 CUDA 12.8,已安装此版本)
#    CUDA 13 换成 ...gpu_cuda13...;Orin 换成 NVIDIA 提供的 onnxruntime-gpu 包
curl -L -o ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-gpu_cuda12-1.27.0.tgz
tar xf ort.tgz && mv onnxruntime-linux-x64-gpu_cuda12-1.27.0 third_party/onnxruntime

# 2) cuDNN 9(GPU 必需)—— x64 官方 ORT 包不含,需另放到 third_party/cudnn/lib
#    最省事:从 pip wheel 取(免登录),解压其中 nvidia/cudnn/lib 下的 libcudnn*.so.9
pip download nvidia-cudnn-cu12 -d /tmp/cudnn && \
  (cd /tmp/cudnn && unzip -o nvidia_cudnn_cu12*.whl -d x) && \
  mkdir -p third_party/cudnn/lib && \
  cp -av /tmp/cudnn/x/nvidia/cudnn/lib/libcudnn*.so.9 third_party/cudnn/lib/

# 3) tokenizers-cpp(仅当 -DSMOLVLA_WITH_TOKENIZERS=ON 时)
git submodule add https://github.com/mlc-ai/tokenizers-cpp third_party/tokenizers-cpp
git submodule update --init --recursive
```

> CUDA runtime 与 TensorRT 由系统提供(本机已装且 `.zshrc` 已导出到 `LD_LIBRARY_PATH`)。
> 可执行文件用 `DT_RPATH` 把 `third_party/onnxruntime/lib` 与 `third_party/cudnn/lib` 写入,
> 故运行时无需再手动设置 cuDNN 路径。

> ONNX Runtime 版本号按实际下载的填;`third_party/` 已在 `.gitignore` 中忽略,预编译包不入库。

## 构建

```bash
cd cpp && mkdir -p build && cd build
cmake ..            # 默认构建 smolvla_demo + smolvla_deploy
cmake --build . -j
# 不构建部署程序(如缺相机/电机依赖的机器):
#   cmake -DSMOLVLA_WITH_DEPLOY=OFF ..
```

首次运行 TensorRT EP 会 build engine 并缓存到 `artifacts/.trt_cache`(~3-4 分钟),二次启动秒级.

## 运行实机部署

```bash
# 0) 一次性准备:导出语言指令与 state 统计(换指令时重跑第一条)
/home/ubuntu/AI/deploy_vla/.smolvla_venv/bin/python tools/export_lang_tokens.py \
    --instruction "pick up the blue block"
/home/ubuntu/AI/deploy_vla/.smolvla_venv/bin/python tools/export_state_stats.py

# 1) 配置 CAN
sudo ip link set can0 type can bitrate 1000000 && sudo ip link set can0 up

# 2) 运行(在 cpp/build 下, 默认 can0 / USB 10)
./smolvla_deploy --artifacts ../../artifacts
#   -> 零力矩+重力补偿, 手动摆位后按 Enter, 加载完成后自动开始推理抓取

# 调试模式:
./smolvla_deploy --artifacts ../../artifacts --no-arm                 # 有相机无机械臂, 只打印动作
./smolvla_deploy --artifacts ../../artifacts --no-arm --fake-cameras  # 纯推理冒烟测试
./smolvla_deploy --artifacts ../../artifacts --no-manual-init         # 跳过手动摆位直接开始
```

常用参数:`--rs-serial SN`(多台 RealSense 时指定)、`--swap-cameras`、
`--gravity-torques "0,1.2,-1.2,0.1,0,0"`(重力补偿力矩)、`--exec-horizon N`、
`--action-dt S`、`--max-dq-per-action RAD`、`--max-dq-per-tick RAD`、`--max-iters N`、
`--save-obs DIR`(把模型每次推理实际看到的两路相机帧存成 jpg,诊断视觉输入用)。
Ctrl+C 停止时会命令电机保持当前位置。

## 已验证

- 4 引擎 ONNX 串联 vs 原 policy:`max|diff|=3.6e-7`(Python 侧,`../run_onnx_pipeline.py`)
- C++ 图像预处理 vs lerobot PyTorch `resize_with_pad`:`max|diff|≈0.006`(uint8 量化误差)
- `smolvla_deploy --no-arm --fake-cameras`:完整推理回路跑通,稳态单次推理 ~40ms(RTX 5060 Laptop, TRT FP16)
- 实机全链路(相机在线 + CAN 电机)待硬件接好后实测
