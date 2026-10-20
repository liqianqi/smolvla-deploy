// SmolVLA C++ 推理运行时(单图版: 整条 sample_actions 导出为一个 smolvla_full.onnx)
//
// 图内已包含: 视觉编码 -> prefix 组装 -> VLM prefill -> 10 步 flow-matching 去噪.
// C++ 侧只负责: 拼接输入张量 -> 一次 Run -> 取前 real_action_dim 维并反归一化.
// 去噪步数/相机数/chunk 固定在图里(由 tools/export_full_onnx.py 决定), dims.num_steps 不起作用.
//
// 与 SmolVLARuntime 共用 RuntimeConfig / Observation, 两者可互换.
#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "smolvla_runtime.hpp"

namespace smolvla
{

class SmolVLAFullRuntime
{
 public:
    // 读 cfg.artifacts_dir/smolvla_full.onnx(+ .onnx.data); fp16_engines 中写 "full" 即开 FP16
    explicit SmolVLAFullRuntime(const RuntimeConfig& cfg);

    // 返回 [chunk, real_action_dim] 的动作序列(行优先展开).
    // noise 可选:传入固定噪声以复现/对比;为空则内部随机采样.
    std::vector<float> SelectActionChunk(const Observation& obs,
                                         const std::vector<float>* noise = nullptr);

 private:
    RuntimeConfig cfg_;
    ModelDims d_;

    Ort::Env env_;
    Ort::MemoryInfo mem_info_;
    std::unique_ptr<Ort::Session> session_;

    // 输入缓冲(跨帧复用, 避免每帧重新分配约 9MB 的图像缓冲)
    // bool 掩码用 uint8_t 连续缓冲(ONNX Runtime 的 BOOL 张量按 1 字节存储)
    std::vector<float> images_;
    std::vector<uint8_t> img_masks_;
    std::vector<int64_t> lang_tokens_;
    std::vector<uint8_t> lang_masks_;
    std::vector<float> state_;
    std::vector<float> noise_;

    std::vector<float> action_mean_;
    std::vector<float> action_std_;
};

}  // namespace smolvla
