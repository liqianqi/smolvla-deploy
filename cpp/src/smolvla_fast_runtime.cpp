#include "smolvla_fast_runtime.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

namespace smolvla
{
namespace
{

using Clock = std::chrono::steady_clock;

constexpr auto kF32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto kI64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
constexpr auto kBool = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;

double MsSince(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void Check(cudaError_t e, const char* what)
{
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void Upload(void* dst, const void* src, size_t bytes)
{
    Check(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

// ORT 在自己的 non-blocking stream 上执行, 与 cudaMemcpy 所在的默认流互不同步,
// 且 pageable 内存的 H2D / D2D 拷贝返回时不保证已完成, 所以每次 Run 前都要整卡同步.
void Sync()
{
    Check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

Ort::Value GpuTensor(const Ort::MemoryInfo& mi, void* p, size_t elem_bytes,
                     const std::vector<int64_t>& shape, ONNXTensorElementDataType type)
{
    size_t n = 1;
    for (auto s : shape) n *= static_cast<size_t>(s);
    return Ort::Value::CreateTensor(mi, p, n * elem_bytes, shape.data(), shape.size(), type);
}

}  // namespace

SmolVLAFastRuntime::SmolVLAFastRuntime(const RuntimeConfig& cfg)
    : cfg_(cfg),
      d_(cfg.dims),
      env_(ORT_LOGGING_LEVEL_WARNING, "smolvla_fast"),
      gpu_mem_("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault)
{
    const int ni = d_.num_images, nt = d_.image_tokens, h = d_.hidden;
    const int L = prefix_len_ = d_.prefix_len();
    const size_t img_n = static_cast<size_t>(3) * d_.img_size * d_.img_size;
    const size_t emb_n = static_cast<size_t>(nt) * h;
    const size_t kv_n = static_cast<size_t>(d_.num_layers) * L * d_.kv_heads * d_.head_dim;
    const size_t act_n = static_cast<size_t>(d_.chunk) * d_.action_dim;

    // ---- session ----
    if (!cfg_.trt_cache_dir.empty())
    {
        trt_cache_dir_ = (std::filesystem::absolute(cfg_.trt_cache_dir) / "fp32").string();
        std::filesystem::create_directories(trt_cache_dir_);
    }
    vision_ = MakeSession("vision_encoder.onnx", false);
    assembler_ = MakeSession("prefix_assembler.onnx", false);
    prefill_ = MakeSession("vlm_prefill.onnx", false);
    denoise_ = MakeSession("action_denoise_step.onnx", cfg_.denoise_cuda_graph);

    // ---- 固定显存 ----
    d_pix_ = Alloc<float>(ni * img_n);
    d_vis_out_ = Alloc<float>(ni * emb_n);
    d_empty_emb_ = Alloc<float>(emb_n);
    d_img_embeds_ = Alloc<float>(ni * emb_n);
    d_img_masks_ = Alloc<uint8_t>(ni);
    d_lang_ = Alloc<int64_t>(d_.lang_len);
    d_lang_mask_ = Alloc<uint8_t>(d_.lang_len);
    d_state_ = Alloc<float>(d_.state_dim);
    d_prefix_ = Alloc<float>(static_cast<size_t>(L) * h);
    d_pad_ = Alloc<uint8_t>(L);
    d_attn_ = Alloc<uint8_t>(static_cast<size_t>(L) * L);
    d_pos_ = Alloc<int64_t>(L);
    d_kv_keys_ = Alloc<float>(kv_n);
    d_kv_values_ = Alloc<float>(kv_n);
    d_x_ = Alloc<float>(act_n);
    d_t_ = Alloc<float>(1);
    d_v_ = Alloc<float>(act_n);

    // ---- 绑定 assembler / prefill / denoise(地址固定, 只绑一次)----
    const std::vector<int64_t> kv_shape{d_.num_layers, 1, L, d_.kv_heads, d_.head_dim};
    const std::vector<int64_t> act_shape{1, d_.chunk, d_.action_dim};

    assembler_bind_ = std::make_unique<Ort::IoBinding>(*assembler_);
    assembler_bind_->BindInput("image_embeds", GpuTensor(gpu_mem_, d_img_embeds_, 4, {ni, 1, nt, h}, kF32));
    assembler_bind_->BindInput("img_masks", GpuTensor(gpu_mem_, d_img_masks_, 1, {ni, 1}, kBool));
    assembler_bind_->BindInput("lang_tokens", GpuTensor(gpu_mem_, d_lang_, 8, {1, d_.lang_len}, kI64));
    assembler_bind_->BindInput("lang_masks", GpuTensor(gpu_mem_, d_lang_mask_, 1, {1, d_.lang_len}, kBool));
    assembler_bind_->BindInput("state", GpuTensor(gpu_mem_, d_state_, 4, {1, d_.state_dim}, kF32));
    assembler_bind_->BindOutput("prefix_embs", GpuTensor(gpu_mem_, d_prefix_, 4, {1, L, h}, kF32));
    assembler_bind_->BindOutput("prefix_pad_masks", GpuTensor(gpu_mem_, d_pad_, 1, {1, L}, kBool));
    assembler_bind_->BindOutput("attn_2d_mask", GpuTensor(gpu_mem_, d_attn_, 1, {1, L, L}, kBool));
    assembler_bind_->BindOutput("position_ids", GpuTensor(gpu_mem_, d_pos_, 8, {1, L}, kI64));

    prefill_bind_ = std::make_unique<Ort::IoBinding>(*prefill_);
    prefill_bind_->BindInput("prefix_embs", GpuTensor(gpu_mem_, d_prefix_, 4, {1, L, h}, kF32));
    prefill_bind_->BindInput("attn_2d_mask", GpuTensor(gpu_mem_, d_attn_, 1, {1, L, L}, kBool));
    prefill_bind_->BindInput("position_ids", GpuTensor(gpu_mem_, d_pos_, 8, {1, L}, kI64));
    prefill_bind_->BindOutput("kv_keys", GpuTensor(gpu_mem_, d_kv_keys_, 4, kv_shape, kF32));
    prefill_bind_->BindOutput("kv_values", GpuTensor(gpu_mem_, d_kv_values_, 4, kv_shape, kF32));

    denoise_bind_ = std::make_unique<Ort::IoBinding>(*denoise_);
    denoise_bind_->BindInput("x_t", GpuTensor(gpu_mem_, d_x_, 4, act_shape, kF32));
    denoise_bind_->BindInput("timestep", GpuTensor(gpu_mem_, d_t_, 4, {1}, kF32));
    denoise_bind_->BindInput("kv_keys", GpuTensor(gpu_mem_, d_kv_keys_, 4, kv_shape, kF32));
    denoise_bind_->BindInput("kv_values", GpuTensor(gpu_mem_, d_kv_values_, 4, kv_shape, kF32));
    denoise_bind_->BindInput("prefix_pad_masks", GpuTensor(gpu_mem_, d_pad_, 1, {1, L}, kBool));
    denoise_bind_->BindOutput("v_t", GpuTensor(gpu_mem_, d_v_, 4, act_shape, kF32));

    // ---- empty camera 的 embedding: 输入恒为 -1, 只算一次 ----
    // batch 取部署时的有效相机数, 避免 TensorRT 为不同 batch 重建 engine
    const int real_cams = std::max(1, ni - 1);
    const std::vector<float> neg(real_cams * img_n, -1.0f);
    Upload(d_pix_, neg.data(), neg.size() * sizeof(float));
    RunVision(real_cams);
    Check(cudaMemcpy(d_empty_emb_, d_vis_out_, emb_n * sizeof(float), cudaMemcpyDeviceToDevice),
          "cudaMemcpy D2D");

    // ---- 动作反归一化统计量: [mean(D), std(D)] 的 float32, D=real_action_dim ----
    if (cfg_.unnormalize_action)
    {
        const std::string stats = cfg_.action_stats_path.empty()
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

    // ---- 预热: 首次 Run 的显存分配与 CUDA Graph 录制(约 0.5s)放在构造阶段, 而不是第一帧 ----
    Observation warm;
    warm.images.assign(ni, std::vector<float>(img_n, -1.0f));
    warm.image_valid.assign(ni, true);
    warm.image_valid[ni - 1] = false;
    warm.lang_tokens.assign(d_.lang_len, 0);
    warm.lang_mask.assign(d_.lang_len, false);
    warm.lang_mask[0] = true;
    warm.state.assign(d_.state_dim, 0.0f);
    for (int i = 0; i < 2; ++i) SelectActionChunk(warm);
}

SmolVLAFastRuntime::~SmolVLAFastRuntime()
{
    // 先释放引用这些显存的 binding / session, 再 cudaFree
    vision_bind_.reset();
    assembler_bind_.reset();
    prefill_bind_.reset();
    denoise_bind_.reset();
    vision_.reset();
    assembler_.reset();
    prefill_.reset();
    denoise_.reset();
    for (void* p : allocs_) cudaFree(p);
}

template <typename T>
T* SmolVLAFastRuntime::Alloc(size_t n)
{
    void* p = nullptr;
    Check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    allocs_.push_back(p);
    return static_cast<T*>(p);
}

std::unique_ptr<Ort::Session> SmolVLAFastRuntime::MakeSession(const char* file, bool cuda_graph)
{
    const auto& api = Ort::GetApi();
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (cfg_.use_tensorrt)
    {
        OrtTensorRTProviderOptionsV2* trt = nullptr;
        Ort::ThrowOnError(api.CreateTensorRTProviderOptions(&trt));
        std::vector<const char*> keys{"device_id", "trt_cuda_graph_enable"};
        std::vector<const char*> vals{"0", cuda_graph ? "1" : "0"};
        if (!trt_cache_dir_.empty())
        {
            keys.insert(keys.end(), {"trt_engine_cache_enable", "trt_engine_cache_path"});
            vals.insert(vals.end(), {"1", trt_cache_dir_.c_str()});
        }
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(trt, keys.data(), vals.data(), keys.size()));
        so.AppendExecutionProvider_TensorRT_V2(*trt);
        api.ReleaseTensorRTProviderOptions(trt);
    }
    // CUDA EP: TensorRT 不支持算子时的回退; 不用 TensorRT 时由它负责 CUDA Graph
    OrtCUDAProviderOptionsV2* cuda = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
    const char* ckeys[] = {"device_id", "enable_cuda_graph"};
    const char* cvals[] = {"0", (cuda_graph && !cfg_.use_tensorrt) ? "1" : "0"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, ckeys, cvals, 2));
    so.AppendExecutionProvider_CUDA_V2(*cuda);
    api.ReleaseCUDAProviderOptions(cuda);

    const std::string path = cfg_.artifacts_dir + "/" + file;
    return std::make_unique<Ort::Session>(env_, path.c_str(), so);
}

void SmolVLAFastRuntime::RunVision(int batch)
{
    if (vision_batch_ != batch)  // 有效相机数变化时才重新绑定
    {
        const int nt = d_.image_tokens, h = d_.hidden, s = d_.img_size;
        vision_bind_ = std::make_unique<Ort::IoBinding>(*vision_);
        vision_bind_->BindInput("pixel_values", GpuTensor(gpu_mem_, d_pix_, 4, {batch, 3, s, s}, kF32));
        vision_bind_->BindOutput("image_embeds", GpuTensor(gpu_mem_, d_vis_out_, 4, {batch, nt, h}, kF32));
        vision_batch_ = batch;
    }
    Sync();
    vision_->Run(Ort::RunOptions{nullptr}, *vision_bind_);
    vision_bind_->SynchronizeOutputs();
}

std::vector<float> SmolVLAFastRuntime::SelectActionChunk(const Observation& obs,
                                                         const std::vector<float>* noise)
{
    const int ni = d_.num_images;
    const size_t img_n = static_cast<size_t>(3) * d_.img_size * d_.img_size;
    const size_t emb_n = static_cast<size_t>(d_.image_tokens) * d_.hidden;
    const size_t act_n = static_cast<size_t>(d_.chunk) * d_.action_dim;
    if (static_cast<int>(obs.images.size()) != ni || static_cast<int>(obs.image_valid.size()) != ni)
        throw std::runtime_error("images/image_valid 数量与 num_images 不一致");

    // 1) 视觉编码: 有效相机合成一个 batch, empty camera 用缓存
    auto t0 = Clock::now();
    int nv = 0;
    for (int i = 0; i < ni; ++i)
    {
        if (!obs.image_valid[i]) continue;
        if (obs.images[i].size() != img_n) throw std::runtime_error("image size mismatch");
        Upload(d_pix_ + nv * img_n, obs.images[i].data(), img_n * sizeof(float));
        ++nv;
    }
    if (nv > 0) RunVision(nv);
    for (int i = 0, k = 0; i < ni; ++i)
    {
        const float* src = obs.image_valid[i] ? d_vis_out_ + (k++) * emb_n : d_empty_emb_;
        Check(cudaMemcpy(d_img_embeds_ + i * emb_n, src, emb_n * sizeof(float), cudaMemcpyDeviceToDevice),
              "cudaMemcpy D2D");
    }
    times_.vision = MsSince(t0);

    // 2) prefix 组装
    t0 = Clock::now();
    std::vector<uint8_t> img_masks(ni), lang_mask(d_.lang_len);
    for (int i = 0; i < ni; ++i) img_masks[i] = obs.image_valid[i] ? 1 : 0;
    for (int i = 0; i < d_.lang_len; ++i) lang_mask[i] = obs.lang_mask[i] ? 1 : 0;
    Upload(d_img_masks_, img_masks.data(), ni);
    Upload(d_lang_, obs.lang_tokens.data(), d_.lang_len * sizeof(int64_t));
    Upload(d_lang_mask_, lang_mask.data(), d_.lang_len);
    Upload(d_state_, obs.state.data(), d_.state_dim * sizeof(float));
    Sync();
    assembler_->Run(Ort::RunOptions{nullptr}, *assembler_bind_);
    assembler_bind_->SynchronizeOutputs();
    times_.assembler = MsSince(t0);

    // 3) prefill -> KV cache(留在显存, 直接被 denoise 读取)
    t0 = Clock::now();
    prefill_->Run(Ort::RunOptions{nullptr}, *prefill_bind_);
    prefill_bind_->SynchronizeOutputs();
    times_.prefill = MsSince(t0);

    // 4) flow-matching 去噪循环(欧拉积分, 与 PyTorch sample_actions 一致)
    //    每步只上传 x_t 与 t、取回 v_t(各约 6KB), 欧拉更新在 CPU 上做
    t0 = Clock::now();
    std::vector<float> x_t(act_n), v_t(act_n);
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
        const float t = 1.0f + static_cast<float>(step) * dt;
        Upload(d_x_, x_t.data(), act_n * sizeof(float));
        Upload(d_t_, &t, sizeof(float));
        Sync();
        denoise_->Run(Ort::RunOptions{nullptr}, *denoise_bind_);
        denoise_bind_->SynchronizeOutputs();
        Check(cudaMemcpy(v_t.data(), d_v_, act_n * sizeof(float), cudaMemcpyDeviceToHost),
              "cudaMemcpy D2H");
        for (size_t i = 0; i < act_n; ++i) x_t[i] += dt * v_t[i];
    }
    times_.denoise = MsSince(t0);

    // 5) unpad: 取每步前 real_action_dim 维 -> [chunk, real_action_dim]
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
