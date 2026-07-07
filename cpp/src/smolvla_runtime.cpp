#include "smolvla_runtime.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace smolvla
{
namespace
{

// 用连续缓冲创建 ONNX Runtime 张量的小工具
Ort::Value MakeFloat(const Ort::MemoryInfo& mi, float* data, const std::vector<int64_t>& shape)
{
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return Ort::Value::CreateTensor<float>(mi, data, n, shape.data(), shape.size());
}

Ort::Value MakeInt64(const Ort::MemoryInfo& mi, int64_t* data, const std::vector<int64_t>& shape)
{
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return Ort::Value::CreateTensor<int64_t>(mi, data, n, shape.data(), shape.size());
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

SmolVLARuntime::SmolVLARuntime(const RuntimeConfig& cfg)
    : cfg_(cfg),
      d_(cfg.dims),
      env_(ORT_LOGGING_LEVEL_WARNING, "smolvla"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    prefix_len_ = d_.prefix_len();

    // 每个引擎可独立选 FP16/FP32(定位精度敏感引擎、以及混合精度部署用).
    // 注意 TRT 引擎缓存不区分精度标志, 必须按精度拆缓存目录, 否则会加载错的引擎.
    auto engine_fp16 = [&](const char* name)
    {
        if (cfg_.fp16_engines.empty()) return cfg_.fp16;
        return ("," + cfg_.fp16_engines + ",").find("," + std::string(name) + ",") !=
               std::string::npos;
    };
    // SessionOptions 与其中的 cache path 字符串必须活到 Session 构造完成
    auto make_session = [&](const char* file, const char* name,
                            std::string& cache_dir) -> std::unique_ptr<Ort::Session>
    {
        const bool fp16 = engine_fp16(name);
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (cfg_.use_tensorrt)
        {
            OrtTensorRTProviderOptions trt{};
            trt.device_id = 0;
            trt.trt_fp16_enable = fp16 ? 1 : 0;
            cache_dir = cfg_.trt_cache_dir.empty()
                            ? std::string()
                            : cfg_.trt_cache_dir + (fp16 ? "/fp16" : "/fp32");
            trt.trt_engine_cache_enable = cache_dir.empty() ? 0 : 1;
            trt.trt_engine_cache_path = cache_dir.c_str();
            so.AppendExecutionProvider_TensorRT(trt);
        }
        // CUDA 作为 TensorRT 不支持算子时的回退
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        so.AppendExecutionProvider_CUDA(cuda);
        const std::string p = cfg_.artifacts_dir + "/" + file;
        return std::make_unique<Ort::Session>(env_, p.c_str(), so);
    };

    std::string cache_dirs[4];
    vision_ = make_session("vision_encoder.onnx", "vision", cache_dirs[0]);
    assembler_ = make_session("prefix_assembler.onnx", "assembler", cache_dirs[1]);
    prefill_ = make_session("vlm_prefill.onnx", "prefill", cache_dirs[2]);
    denoise_ = make_session("action_denoise_step.onnx", "denoise", cache_dirs[3]);

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

// ---- Block 1: vision,单图 -> image_embeds [1, image_tokens, hidden] ----
Ort::Value SmolVLARuntime::RunVision(const float* image_data)
{
    std::vector<int64_t> in_shape{1, 3, d_.img_size, d_.img_size};
    std::vector<float> buf(image_data, image_data + 3 * d_.img_size * d_.img_size);
    Ort::Value in = MakeFloat(mem_info_, buf.data(), in_shape);

    const char* in_names[] = {"pixel_values"};
    const char* out_names[] = {"image_embeds"};
    auto out = vision_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);
    return std::move(out[0]);
}

// ---- Block 2: prefix 组装 ----
void SmolVLARuntime::RunAssembler(const std::vector<Ort::Value>& image_embeds,
                                  const Observation& obs)
{
    // image_embeds 堆叠成 [num_images, 1, image_tokens, hidden]
    const int ni = d_.num_images, nt = d_.image_tokens, h = d_.hidden;
    std::vector<float> ie(static_cast<size_t>(ni) * nt * h);
    for (int i = 0; i < ni; ++i)
    {
        const float* src = image_embeds[i].GetTensorData<float>();
        std::memcpy(ie.data() + static_cast<size_t>(i) * nt * h, src, sizeof(float) * nt * h);
    }
    std::vector<uint8_t> img_masks(ni);
    for (int i = 0; i < ni; ++i) img_masks[i] = obs.image_valid[i] ? 1 : 0;

    std::vector<int64_t> lang(obs.lang_tokens.begin(), obs.lang_tokens.end());
    std::vector<uint8_t> lmask(d_.lang_len);
    for (int i = 0; i < d_.lang_len; ++i) lmask[i] = obs.lang_mask[i] ? 1 : 0;
    std::vector<float> state(obs.state);

    std::array<Ort::Value, 5> ins{
        MakeFloat(mem_info_, ie.data(), {ni, 1, nt, h}),
        MakeBool(mem_info_, img_masks.data(), {ni, 1}),
        MakeInt64(mem_info_, lang.data(), {1, d_.lang_len}),
        MakeBool(mem_info_, lmask.data(), {1, d_.lang_len}),
        MakeFloat(mem_info_, state.data(), {1, d_.state_dim}),
    };
    const char* in_names[] = {"image_embeds", "img_masks", "lang_tokens", "lang_masks", "state"};
    const char* out_names[] = {"prefix_embs", "prefix_pad_masks", "attn_2d_mask", "position_ids"};
    auto outs =
        assembler_->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), ins.size(), out_names, 4);

    const int L = prefix_len_;
    prefix_embs_.assign(outs[0].GetTensorData<float>(),
                        outs[0].GetTensorData<float>() + static_cast<size_t>(L) * h);
    const bool* pad = outs[1].GetTensorData<bool>();
    prefix_pad_.assign(reinterpret_cast<const uint8_t*>(pad),
                       reinterpret_cast<const uint8_t*>(pad) + L);
    const bool* a2 = outs[2].GetTensorData<bool>();
    attn_2d_.assign(reinterpret_cast<const uint8_t*>(a2),
                    reinterpret_cast<const uint8_t*>(a2) + static_cast<size_t>(L) * L);
    position_ids_.assign(outs[3].GetTensorData<int64_t>(), outs[3].GetTensorData<int64_t>() + L);
}

// ---- Block 3: prefill -> KV cache ----
void SmolVLARuntime::RunPrefill()
{
    const int L = prefix_len_, h = d_.hidden;
    std::array<Ort::Value, 3> ins{
        MakeFloat(mem_info_, prefix_embs_.data(), {1, L, h}),
        MakeBool(mem_info_, attn_2d_.data(), {1, L, L}),
        MakeInt64(mem_info_, position_ids_.data(), {1, L}),
    };
    const char* in_names[] = {"prefix_embs", "attn_2d_mask", "position_ids"};
    const char* out_names[] = {"kv_keys", "kv_values"};
    auto outs =
        prefill_->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), ins.size(), out_names, 2);

    const size_t kv_n = static_cast<size_t>(d_.num_layers) * L * d_.kv_heads * d_.head_dim;  // B=1
    kv_keys_.assign(outs[0].GetTensorData<float>(), outs[0].GetTensorData<float>() + kv_n);
    kv_values_.assign(outs[1].GetTensorData<float>(), outs[1].GetTensorData<float>() + kv_n);
}

// ---- Block 4: 单步 denoise ----
std::vector<float> SmolVLARuntime::RunDenoiseStep(const std::vector<float>& x_t, float t)
{
    const int L = prefix_len_;
    std::vector<float> x = x_t;  // 可写缓冲
    std::vector<float> ts{t};
    std::array<Ort::Value, 5> ins{
        MakeFloat(mem_info_, x.data(), {1, d_.chunk, d_.action_dim}),
        MakeFloat(mem_info_, ts.data(), {1}),
        MakeFloat(mem_info_, kv_keys_.data(), {d_.num_layers, 1, L, d_.kv_heads, d_.head_dim}),
        MakeFloat(mem_info_, kv_values_.data(), {d_.num_layers, 1, L, d_.kv_heads, d_.head_dim}),
        MakeBool(mem_info_, prefix_pad_.data(), {1, L}),
    };
    const char* in_names[] = {"x_t", "timestep", "kv_keys", "kv_values", "prefix_pad_masks"};
    const char* out_names[] = {"v_t"};
    auto outs =
        denoise_->Run(Ort::RunOptions{nullptr}, in_names, ins.data(), ins.size(), out_names, 1);
    const float* v = outs[0].GetTensorData<float>();
    return std::vector<float>(v, v + static_cast<size_t>(d_.chunk) * d_.action_dim);
}

std::vector<float> SmolVLARuntime::SelectActionChunk(const Observation& obs,
                                                     const std::vector<float>* noise)
{
    // 1) 视觉编码
    std::vector<Ort::Value> image_embeds;
    image_embeds.reserve(d_.num_images);
    for (int i = 0; i < d_.num_images; ++i)
    {
        image_embeds.push_back(RunVision(obs.images[i].data()));
    }
    // 2) prefix 组装  3) prefill
    RunAssembler(image_embeds, obs);
    RunPrefill();

    // 4) flow-matching 去噪循环(欧拉积分,与 PyTorch sample_actions 一致)
    const size_t act_n = static_cast<size_t>(d_.chunk) * d_.action_dim;
    std::vector<float> x_t(act_n);
    if (noise != nullptr)
    {
        if (noise->size() != act_n) throw std::runtime_error("noise size mismatch");
        x_t = *noise;
    }
    else
    {
        std::mt19937 gen(0);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x_t) v = nd(gen);
    }
    const float dt = -1.0f / static_cast<float>(d_.num_steps);
    for (int step = 0; step < d_.num_steps; ++step)
    {
        float t = 1.0f + static_cast<float>(step) * dt;
        std::vector<float> v_t = RunDenoiseStep(x_t, t);
        for (size_t i = 0; i < act_n; ++i) x_t[i] += dt * v_t[i];
    }

    // 5) unpad:取每步前 real_action_dim 维 -> [chunk, real_action_dim]
    //    并做反归一化: 真实(绝对关节角度)动作 = norm*std + mean
    const int D = d_.real_action_dim;
    const bool denorm = !action_mean_.empty();
    std::vector<float> actions(static_cast<size_t>(d_.chunk) * D);
    for (int s = 0; s < d_.chunk; ++s)
        for (int a = 0; a < D; ++a)
        {
            float v = x_t[s * d_.action_dim + a];
            if (denorm) v = v * action_std_[a] + action_mean_[a];
            actions[s * D + a] = v;
        }
    return actions;
}

}  // namespace smolvla
