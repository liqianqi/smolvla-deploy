// SmolVLA C++ 推理运行时(4 引擎优化版, 与 SmolVLARuntime 接口相同, 数值一致)
//
// 相对 SmolVLARuntime 的改动:
//   1. 中间结果常驻 GPU: 4 个引擎全部用 IO Binding 绑定到预分配的固定显存,
//      image_embeds / prefix / KV cache 不再在 CPU 与 GPU 之间来回拷贝
//      (原实现每步 denoise 都要把约 10MB 的 KV cache 重新上传一次).
//   2. 视觉编码合批: 有效相机一次 batch 前向; empty camera 的输入恒为 -1,
//      其 embedding 在构造时算一次并缓存.
//   3. denoise 引擎开 CUDA Graph(cfg.denoise_cuda_graph): 首次运行录制, 之后每步只重放.
// 帧内其余阶段(组装 -> prefill -> 10 步去噪)存在严格数据依赖, 只能串行.
// 精度固定 FP32(忽略 fp16 / fp16_engines).
#pragma once

#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "smolvla_runtime.hpp"

namespace smolvla
{

// 最近一次 SelectActionChunk 的分阶段耗时(毫秒, 含该阶段的数据上传)
struct StageTimes
{
    double vision = 0, assembler = 0, prefill = 0, denoise = 0;
    double total() const { return vision + assembler + prefill + denoise; }
};

class SmolVLAFastRuntime
{
 public:
    explicit SmolVLAFastRuntime(const RuntimeConfig& cfg);
    ~SmolVLAFastRuntime();
    SmolVLAFastRuntime(const SmolVLAFastRuntime&) = delete;
    SmolVLAFastRuntime& operator=(const SmolVLAFastRuntime&) = delete;

    // 返回 [chunk, real_action_dim] 的动作序列(行优先展开).
    // 约定: image_valid=false 的相机为 empty camera(输入全 -1), 直接使用缓存的 embedding.
    std::vector<float> SelectActionChunk(const Observation& obs,
                                         const std::vector<float>* noise = nullptr);

    const StageTimes& LastStageTimes() const { return times_; }

 private:
    std::unique_ptr<Ort::Session> MakeSession(const char* file, bool cuda_graph);
    template <typename T>
    T* Alloc(size_t n);  // cudaMalloc n 个 T, 析构时统一释放
    void RunVision(int batch);

    RuntimeConfig cfg_;
    ModelDims d_;
    int prefix_len_ = 0;
    std::string trt_cache_dir_;

    Ort::Env env_;
    Ort::MemoryInfo gpu_mem_;
    std::unique_ptr<Ort::Session> vision_, assembler_, prefill_, denoise_;
    std::unique_ptr<Ort::IoBinding> vision_bind_, assembler_bind_, prefill_bind_, denoise_bind_;
    int vision_batch_ = -1;  // vision_bind_ 当前绑定的 batch

    // 固定显存(地址在整个生命周期内不变, CUDA Graph 依赖这一点)
    std::vector<void*> allocs_;
    float* d_pix_ = nullptr;          // [num_images, 3, img, img] 有效相机按顺序紧排
    float* d_vis_out_ = nullptr;      // [num_images, image_tokens, hidden]
    float* d_empty_emb_ = nullptr;    // [image_tokens, hidden] empty camera 缓存
    float* d_img_embeds_ = nullptr;   // [num_images, 1, image_tokens, hidden]
    uint8_t* d_img_masks_ = nullptr;  // [num_images, 1]
    int64_t* d_lang_ = nullptr;       // [1, lang_len]
    uint8_t* d_lang_mask_ = nullptr;  // [1, lang_len]
    float* d_state_ = nullptr;        // [1, state_dim]
    float* d_prefix_ = nullptr;       // [1, L, hidden]
    uint8_t* d_pad_ = nullptr;        // [1, L]
    uint8_t* d_attn_ = nullptr;       // [1, L, L]
    int64_t* d_pos_ = nullptr;        // [1, L]
    float* d_kv_keys_ = nullptr;      // [num_layers, 1, L, kv_heads, head_dim]
    float* d_kv_values_ = nullptr;
    float* d_x_ = nullptr;            // [1, chunk, action_dim]
    float* d_t_ = nullptr;            // [1]
    float* d_v_ = nullptr;            // [1, chunk, action_dim]

    std::vector<float> action_mean_, action_std_;
    StageTimes times_;
};

}  // namespace smolvla
