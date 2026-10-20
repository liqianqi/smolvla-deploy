#include "smolvla_full_runtime.hpp"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace smolvla
{
namespace
{

template <typename T>
Ort::Value MakeTensor(const Ort::MemoryInfo& mi, T* data, const std::vector<int64_t>& shape)
{
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return Ort::Value::CreateTensor<T>(mi, data, n, shape.data(), shape.size());
}

// BOOL 张量:用 typeless 接口 + uint8_t 缓冲(sizeof(bool)==1)
Ort::Value MakeBool(const Ort::MemoryInfo& mi, uint8_t* data, const std::vector<int64_t>& shape)
{
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return Ort::Value::CreateTensor(mi, data, n * sizeof(bool), shape.data(), shape.size(),
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL);
}

}  // namespace

SmolVLAFullRuntime::SmolVLAFullRuntime(const RuntimeConfig& cfg)
    : cfg_(cfg),
      d_(cfg.dims),
      env_(ORT_LOGGING_LEVEL_WARNING, "smolvla_full"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    const bool fp16 = cfg_.fp16_engines.empty()
                          ? cfg_.fp16
                          : ("," + cfg_.fp16_engines + ",").find(",full,") != std::string::npos;

    // SessionOptions 与其中的 cache path 字符串必须活到 Session 构造完成
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // 展开的 10 步 KV 写入产生约 500 个 ScatterND, CUDA EP 每次 Run 都会刷
    // "reduction=='none' only guarantees to be correct if indices are not duplicated" 警告
    // (此处索引不重复, 已与 4 引擎方案对齐验证), 只保留 ERROR 级日志.
    so.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
    std::string cache_dir;
    if (cfg_.use_tensorrt)
    {
        // TRT 引擎缓存不区分精度标志, 必须按精度拆缓存目录, 否则会加载错的引擎.
        if (!cfg_.trt_cache_dir.empty())
        {
            cache_dir = (std::filesystem::absolute(cfg_.trt_cache_dir) / (fp16 ? "fp16" : "fp32"))
                            .string();
            std::filesystem::create_directories(cache_dir);
        }
        OrtTensorRTProviderOptions trt{};
        trt.device_id = 0;
        trt.trt_fp16_enable = fp16 ? 1 : 0;
        trt.trt_engine_cache_enable = cache_dir.empty() ? 0 : 1;
        trt.trt_engine_cache_path = cache_dir.c_str();
        so.AppendExecutionProvider_TensorRT(trt);
    }
    // CUDA 作为 TensorRT 不支持算子时的回退
    OrtCUDAProviderOptions cuda{};
    cuda.device_id = 0;
    so.AppendExecutionProvider_CUDA(cuda);
    const std::string path = cfg_.artifacts_dir + "/smolvla_full.onnx";
    session_ = std::make_unique<Ort::Session>(env_, path.c_str(), so);

    images_.resize(static_cast<size_t>(d_.num_images) * 3 * d_.img_size * d_.img_size);
    img_masks_.resize(d_.num_images);
    lang_tokens_.resize(d_.lang_len);
    lang_masks_.resize(d_.lang_len);
    state_.resize(d_.state_dim);
    noise_.resize(static_cast<size_t>(d_.chunk) * d_.action_dim);

    // 加载动作反归一化统计量: [mean(D), std(D)] 的 float32, D=real_action_dim
    if (cfg_.unnormalize_action)
    {
        std::string stats = cfg_.action_stats_path.empty()
                                ? cfg_.artifacts_dir + "/action_norm_stats.bin"
                                : cfg_.action_stats_path;
        std::ifstream f(stats, std::ios::binary);
        if (!f) throw std::runtime_error("找不到动作统计量文件: " + stats);
        const int D = d_.real_action_dim;
        action_mean_.resize(D);
        action_std_.resize(D);
        f.read(reinterpret_cast<char*>(action_mean_.data()), sizeof(float) * D);
        f.read(reinterpret_cast<char*>(action_std_.data()), sizeof(float) * D);
        if (!f) throw std::runtime_error("动作统计量文件大小不足: " + stats);
    }
}

std::vector<float> SmolVLAFullRuntime::SelectActionChunk(const Observation& obs,
                                                         const std::vector<float>* noise)
{
    const int ni = d_.num_images;
    const size_t img_n = static_cast<size_t>(3) * d_.img_size * d_.img_size;
    if (static_cast<int>(obs.images.size()) != ni || static_cast<int>(obs.image_valid.size()) != ni)
        throw std::runtime_error("images/image_valid 数量与 num_images 不一致");
    for (int i = 0; i < ni; ++i)
    {
        if (obs.images[i].size() != img_n) throw std::runtime_error("image size mismatch");
        std::memcpy(images_.data() + i * img_n, obs.images[i].data(), sizeof(float) * img_n);
        img_masks_[i] = obs.image_valid[i] ? 1 : 0;
    }
    for (int i = 0; i < d_.lang_len; ++i)
    {
        lang_tokens_[i] = obs.lang_tokens[i];
        lang_masks_[i] = obs.lang_mask[i] ? 1 : 0;
    }
    std::copy(obs.state.begin(), obs.state.end(), state_.begin());

    if (noise != nullptr)
    {
        if (noise->size() != noise_.size()) throw std::runtime_error("noise size mismatch");
        std::copy(noise->begin(), noise->end(), noise_.begin());
    }
    else
    {
        std::mt19937 gen(0);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : noise_) v = nd(gen);
    }

    std::array<Ort::Value, 6> ins{
        MakeTensor(mem_info_, images_.data(), {ni, 1, 3, d_.img_size, d_.img_size}),
        MakeBool(mem_info_, img_masks_.data(), {ni, 1}),
        MakeTensor(mem_info_, lang_tokens_.data(), {1, d_.lang_len}),
        MakeBool(mem_info_, lang_masks_.data(), {1, d_.lang_len}),
        MakeTensor(mem_info_, state_.data(), {1, d_.state_dim}),
        MakeTensor(mem_info_, noise_.data(), {1, d_.chunk, d_.action_dim}),
    };
    const char* in_names[] = {"images", "img_masks", "lang_tokens", "lang_masks", "state", "noise"};
    const char* out_names[] = {"actions"};
    auto outs =
        session_->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), ins.size(), out_names, 1);
    const float* x = outs[0].GetTensorData<float>();  // [1, chunk, action_dim]

    // unpad:取每步前 real_action_dim 维 -> [chunk, real_action_dim]
    // 并做反归一化: 真实(绝对关节角度)动作 = norm*std + mean
    const int D = d_.real_action_dim;
    const bool denorm = !action_mean_.empty();
    std::vector<float> actions(static_cast<size_t>(d_.chunk) * D);
    for (int s = 0; s < d_.chunk; ++s)
        for (int a = 0; a < D; ++a)
        {
            float v = x[s * d_.action_dim + a];
            if (denorm) v = v * action_std_[a] + action_mean_[a];
            actions[s * D + a] = v;
        }
    return actions;
}

}  // namespace smolvla
