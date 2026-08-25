# VLA 发展史与技术路线笔记

> 整理自面试准备讨论，覆盖 RT-2 → OpenVLA → π0 → SmolVLA / GR00T 的演进脉络、
> flow matching vs 自回归的路线之争，以及分层论文阅读清单。

---

## 1. 发展史：一条主线串起来

### 前史：VLA 之前的困境（~2022）

传统模仿学习（behavior cloning）是「一个任务、一个机器人、一个模型」：
采几百条遥操作演示，训一个小网络（CNN+MLP，后来是 Diffusion Policy），
换任务、换光照就废。同期 NLP/CV 已证明「大模型 + 海量数据 = 泛化」。
核心问题：**怎么把互联网级的视觉-语言知识灌进机器人控制？**

### RT-1 → RT-2：VLA 概念诞生（2022–2023，Google DeepMind）

- **RT-1**（2022）：Transformer + 13 万条真机演示 + 700+ 任务，动作**离散化成 token**
  （每维切 256 桶）当分类问题做。证明 scaling 在机器人上成立。
- **RT-2**（2023）：提出 "Vision-Language-Action" 一词。做法：拿预训练 VLM（PaLI-X, 55B），
  把动作 token **当成文本词表里的词**，与网络图文数据 co-fine-tuning。
  互联网知识第一次迁移到动作上（能理解「放到 3 减 1 的数字上」这类指令）。
  缺点：55B、闭源、~3Hz，学术界只能围观。

### Octo → OpenVLA：开源化与数据聚合（2024）

背景：**Open X-Embodiment（OXE）** 数据集——21 家机构、100 万+ 条轨迹、22 种机器人，
「机器人界的 ImageNet」。

- **Octo**（UC Berkeley, 2024.5）：93M 小 Transformer，开源，跨形态。
  用 **diffusion head 输出连续动作**——离散 vs 连续路线分叉的早期信号。
- **OpenVLA**（Stanford, 2024.6）：7B，SigLIP + DINOv2 双视觉编码器 → Llama-2，
  动作仍走**离散 token 自回归**。全开源（权重+训练代码），打赢 55B 的 RT-2-X，
  成为学术界标准 baseline。
  暴露离散自回归的硬伤：逐 token 串行生成 → 推理仅 ~6Hz；离散化损失动作精度。

### π0：flow matching 路线确立（Physical Intelligence, 2024.10）

技术路线第二个分水岭，三个关键设计：

1. **架构**：PaliGemma-3B VLM + ~300M **action expert**（独立小 Transformer，
   与 VLM **逐层交织、共享 KV cache**）。
2. **动作生成**：**flow matching** 输出连续动作，从纯噪声积分 10 步得到动作。
3. **action chunking**：一次推理输出 50 步动作序列，执行一段再重新推理——
   推理 ~14Hz 支撑 50Hz 控制，第一次做出折衣服级别的灵巧长程任务。

「**VLM backbone + flow-matching action expert + action chunk**」从此成为主流配方。

后续：**π0-FAST**（2025 初）用 DCT 频域压缩离散化动作，证明自回归提速后仍有竞争力；
**π0.5**（2025.4）加分层推理（先输出语义子任务再输出动作），主打开放世界泛化。

### 2025：分化——巨头做大，社区做小

**做大（人形/双系统）**：

- **GR00T N1**（NVIDIA, 2025.3）：System 2 = VLM 负责理解（~10Hz），
  System 1 = diffusion transformer 出动作（~120Hz）。数据混真机轨迹、人类视频、仿真合成。
- Figure **Helix**、Google **Gemini Robotics** 同属双系统思路，目标人形机器人。

**做小（平民化）**：

- **SmolVLA**（Hugging Face, 2025.6）：π0 配方缩到 **450M**（约 π0 的 1/7）。
  backbone 换 SmolVLM2；推理效率设计：跳过 VLM 一半层取中间特征、
  self-attn 与 cross-attn 交替、**异步推理**（执行与下次推理并行）。
  训练数据全部来自 **LeRobot 社区数据**（~3 万条），LIBERO 上打平甚至超过 OpenVLA 和 π0。

### 演进对照表

| 维度 | RT-2 (2023) | OpenVLA (2024) | π0 (2024) | SmolVLA (2025) |
|---|---|---|---|---|
| 动作表示 | 离散 token | 离散 token | flow matching 连续 | flow matching 连续 |
| 架构 | VLM 直接吐 token | VLM 直接吐 token | VLM + action expert | VLM + action expert（更小） |
| 规模 | 55B | 7B | ~3.3B | 0.45B |
| 数据 | 网络图文 + 自有真机 | OXE 97 万条 | 自采跨形态 + 网络 | 社区 3 万条 |
| 开源 | 闭 | 全开 | 部分（openpi） | 全开 |

一句话总结：**RT-2 证明 VLM 知识能迁移到动作上；OpenVLA 把它开源平民化；
π0 用 action expert + flow matching 解决连续控制和推理频率；
SmolVLA 证明这套配方能缩小 15 倍跑在消费级硬件上。**

---

## 2. GR00T N1 细看（NVIDIA, 2025.3）

- 2.2B 参数（VLM 占 1.34B），权重/数据/微调脚本全开源（HF + `NVIDIA/Isaac-GR00T`）。
- **双系统架构**：System 2 = Eagle-2 VLM（~10Hz）；System 1 = DiT，flow matching 训练，
  cross-attend VLM 输出 token，~120Hz 出动作。每种形态有自己的 state/action 编解码器。
- 剥掉营销包装，配方与 π0/SmolVLA 同族（VLM + flow matching + chunk）。
  真正区别在**耦合方式**：π0/SmolVLA 逐层交织共享 KV cache；
  GR00T 的 DiT 是独立模块，仅输入端 cross-attention 松耦合 → 两系统可异频运行，对部署更友好。
- **数据金字塔**（真正的差异化贡献）：塔基 = 互联网人类视频（伪标注动作）；
  塔中 = Omniverse/Cosmos 合成数据（混入后性能 +40%）；塔尖 = 少量真机遥操作。
- 版本演进：N1.5（Eagle 2.5 冻结 + FLARE + DreamGen）→ N1.6（Cosmos-Reason-2B，DiT 加深）
  → N1.7（Qwen3-VL 系 backbone）。策略是把自家 Cosmos 模型逐步塞进 backbone。
- 商业逻辑：模型免费，但 Omniverse 仿真、Cosmos 合成数据、Isaac Lab 训练、
  Jetson Thor 部署每一环都卖 NVIDIA 平台——「机器人界的 Android」打法。

---

## 3. 路线之争：flow matching vs 自回归

**结论：低层连续控制上 flow matching 已是主流，但自回归没有被淘汰——
它退到了语义推理层，并有一条靠新型 tokenizer 翻身的支线。**

### 为什么低层控制 flow matching 赢了

1. **快**：一次并行出整个 chunk；10 步去噪循环内每步也全并行。
   自回归要逐 token 串行几百次（OpenVLA ~6Hz）。
2. **精度**：连续输出，无离散化损失（每维 256 桶对灵巧操作太粗）。
3. **多模态分布**：同一场景「从左绕/从右绕」，离散分类易学出均值；
   diffusion/flow 天然表达多峰分布。

### 自回归的三条活路

1. **tokenizer 革命（π0-FAST）**：DCT 把动作序列变换到频域再压缩离散化（类似 JPEG），
   token 数降一个数量级。自回归版训练比 flow matching 版快 5 倍、性能打平。
   证明「慢不是路线问题，是 tokenization 问题」。
2. **占据语义层（分层架构）**：π0.5 / Hi Robot——高层自回归输出语义子任务
   （本来就是离散语言），低层 flow matching 出连续动作。分工而非替代。
3. **LLM 基础设施红利**：自回归动作模型与 LLM 同计算模式，
   vLLM、投机解码、KV cache 优化、INT4 量化整套生态直接可用。
   flow matching 的迭代去噪则要自己拆引擎、编排循环。

### flow matching 自己的软肋

10 步去噪 = 一次动作生成跑 10 遍 expert。演化方向：
**步数蒸馏**（consistency policy、shortcut model、MeanFlow，目标 1–2 步）；
以及 discrete diffusion VLA 这类杂交方向。

### 与本项目的联系

flow matching 在模型侧赢了，**代价转嫁给了部署侧**：
10 步去噪是动态控制流导不出 ONNX、VLM 与 expert 交织共享 KV cache 无法整体导出——
本项目（拆 4 个 ONNX 引擎、KV cache 显式张量化、C++ 编排去噪循环）
本质上就是在替这个架构选择买单。若是纯自回归模型，部署路径反而更标准化。

---

## 4. 论文阅读清单

### 第一层：VLA 主线必读（按时间顺序，串史线）

| 论文 | 链接 | 读法 / 重点 |
|---|---|---|
| RT-2 | [arxiv.org/abs/2307.15818](https://arxiv.org/abs/2307.15818) | 速读。动作 token 化 + co-fine-tuning、涌现能力实验 |
| OpenVLA | [arxiv.org/abs/2406.09246](https://arxiv.org/abs/2406.09246) | 架构、OXE 数据配方、推理延迟分析（理解转向 flow matching 的原因） |
| **π0** | [arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164) | **精读**。action expert 注意力交织、flow matching 目标、10 步欧拉积分、chunking。对照 `tools/export_prefill.py` / `tools/export_denoise.py` 读 |
| **SmolVLA** | [arxiv.org/abs/2506.01844](https://arxiv.org/abs/2506.01844) | **精读**。跳层取特征、self/cross-attn 交替、异步推理（对应 C++ 的 exec_horizon）、与 π0 的差异表 |
| π0-FAST | [arxiv.org/abs/2501.09747](https://arxiv.org/abs/2501.09747) | 速读。DCT 频域 tokenization，自回归翻身的关键 |
| GR00T N1 | [arxiv.org/abs/2503.14734](https://arxiv.org/abs/2503.14734) | 速读。双系统架构图、数据金字塔、与 π0 交织式的耦合差异 |

### 第二层：理论补课

| 论文 | 链接 | 读法 / 重点 |
|---|---|---|
| **Diffusion Policy** | [arxiv.org/abs/2303.04137](https://arxiv.org/abs/2303.04137) | **精读**。「动作分布多峰、MSE 回归不行」的论证——整条 diffusion/flow 路线的动机，面试必考 |
| Flow Matching | [arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747) | 不必全啃。搞懂「学速度场 v_t，从噪声沿 ODE 积分到数据」及与 DDPM（学去噪、随机采样）的区别 |
| **ACT / ALOHA** | [arxiv.org/abs/2304.13705](https://arxiv.org/abs/2304.13705) | **精读**。action chunking 出处：为什么一次预测一段动作能对抗误差累积（解释 chunk=50、执行 8 步再推理） |

### 第三层：部署方向相关

| 论文 | 链接 | 读法 / 重点 |
|---|---|---|
| Open X-Embodiment | [arxiv.org/abs/2310.08864](https://arxiv.org/abs/2310.08864) | 扫一遍，了解数据侧格局 |
| OpenVLA-OFT | [arxiv.org/abs/2502.19645](https://arxiv.org/abs/2502.19645) | 并行解码、连续动作头、吞吐 26×。「模型侧提速 vs 部署侧提速」的对比素材。项目页 [openvla-oft.github.io](https://openvla-oft.github.io/) 比论文好读 |

### 第四层：选读

| 论文 | 链接 | 读法 / 重点 |
|---|---|---|
| π0.5 | [arxiv.org/abs/2504.16054](https://arxiv.org/abs/2504.16054) | 分层推理、开放世界泛化 |
| RDT-1B | [arxiv.org/abs/2410.07864](https://arxiv.org/abs/2410.07864) | 纯 DiT 路线双臂大模型，对比用 |
| RT-1 | [arxiv.org/abs/2212.06817](https://arxiv.org/abs/2212.06817) | 补前史 |

### 配套代码

- π0 系列开源实现：[github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
- SmolVLA 实现（本项目所用）：[github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
  ——论文对照 `modeling_smolvla.py` 读效率最高（导出脚本 patch 的 `get_safe_dtype`、
  包装的 `denoise_step` 都在这个文件里）

### 建议读法

时间紧就按 **π0 → SmolVLA → Diffusion Policy → ACT** 四篇精读，其余速读摘要 + 架构图。
读 π0 和 SmolVLA 时开着本仓库代码对照，始终带着一个问题读：
**「这个架构设计给部署带来了什么代价？」**

---

## 5. 补充：VLA 转 ONNX/TensorRT 与推理加速

这个方向论文不多（更多是工程实践），分「工程实践文章/代码」和「推理加速论文」两类。

### 工程实践（与本项目路线直接可比）

| 资源 | 链接 | 要点 |
|---|---|---|
| **Deploying VLA models on Jetson**（博客，强烈推荐） | [jared-hpc.com/posts/vla-physical-ai-tensorrt-tether](https://jared-hpc.com/posts/vla-physical-ai-tensorrt-tether/) | 与本项目路线几乎一致：ONNX → ORT + TRT EP，FP16 kernel fusion 报 5.55× 加速。给出 **SmolVLA 在 Orin Nano 上 FP16 ~25ms / INT8 ~20ms** 的参考数字（可对照本项目 RTX 5060 ~40ms 和未来 Orin 实测）。还讨论了 SnapFlow 一步蒸馏、统一内存预算、engine 缓存 |
| tensorrt-openvla（Berkeley RAIL） | [github.com/rail-berkeley/tensorrt-openvla](https://github.com/rail-berkeley/tensorrt-openvla) | 用 **TensorRT-LLM** 编译 OpenVLA 的 Llama backbone（自回归路线的部署样板），支持 Hopper FP8。与本项目「ORT+TRT EP 拆引擎」形成路线对比：自回归 VLA 可直接吃 LLM 推理生态 |
| TensorRT Edge-LLM（Jetson AI Lab 教程） | [jetson-ai-lab.com/tutorials/tensorrt-edge-llm](https://www.jetson-ai-lab.com/tutorials/tensorrt-edge-llm/) | NVIDIA 官方 Jetson 上 LLM/VLM 的纯 C++ 部署链路：x86 量化导出 ONNX → **目标设备上构建 engine**（engine 与硬件绑定）→ C++ runtime。量化支持按架构分级：Orin(sm_87) 只有 FP16/INT4-AWQ，FP8/NVFP4 要 Thor。做 Orin INT8/INT4 前必读 |

### 推理加速论文（模型侧优化，与部署侧互补）

| 论文 | 链接 | 要点 |
|---|---|---|
| EfficientVLA | [arxiv.org/abs/2506.10100](https://arxiv.org/abs/2506.10100) | training-free 三合一：剪语言层 + 任务感知视觉 token 筛选 + **diffusion action head 中间特征跨步缓存**，CogACT 上 1.93×。第三条对本项目 10 步去噪循环直接相关 |
| VLA-Cache | [arxiv.org/abs/2502.02175](https://arxiv.org/abs/2502.02175) | 利用闭环控制中**相邻帧图像大部分不变**，跨帧复用静态视觉 token，只重算变化区域。training-free、即插即用 |
| TinyVLA | [arxiv.org/abs/2409.12514](https://arxiv.org/abs/2409.12514) | 架构侧做小：小 VLM backbone + diffusion 头，绕过自回归解码。SmolVLA 的同路先行者 |
| QVLA | [OpenReview PDF](https://openreview.net/attachment?id=TpL2nXanru&name=pdf) | VLA 专用量化（指出 SmoothQuant 等 LLM 量化方法直接搬到 VLA 效果差）：OpenVLA-OFT 上 29.2% 显存、1.49× 提速、保 98.9% 性能。做 INT8 校准前值得看 |

顺藤摸瓜的关键词（这些论文的相关工作里反复出现）：DeeR-VLA（动态深度）、
MoLe-VLA（层路由）、RoboMamba（换架构）、SnapFlow / consistency policy（去噪步数蒸馏）。

### 两类优化的关系（面试可用的框架）

- **模型侧**（上面这些论文）：剪层、缓存、量化感知、蒸馏——改变计算量本身；
- **部署侧**（本项目做的事）：ONNX 导出、TRT kernel fusion、FP16/INT8、C++ 编排——
  把既定计算量跑得更快。
- 两者正交可叠加。例如「10 步去噪」这个瓶颈：模型侧答案是步数蒸馏（SnapFlow 10→1 步），
  部署侧答案是 TRT FP16 fusion + engine 缓存；EfficientVLA 的特征缓存则介于两者之间。
