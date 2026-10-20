// 4 引擎运行时基准: SmolVLARuntime(原版) vs SmolVLAFastRuntime(不开 / 开 denoise CUDA Graph)
// 同一份观测与噪声, 比较单帧耗时、分阶段耗时与输出差异(反归一化后).
// 用法: ./smolvla_bench [--artifacts DIR] [--iters N] [--no-tensorrt]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include "smolvla_fast_runtime.hpp"
#include "smolvla_runtime.hpp"

namespace
{

using Clock = std::chrono::steady_clock;

struct Result
{
    std::vector<float> actions;
    double ms = 0;
    smolvla::StageTimes stages;
};

template <typename Runtime>
Result Bench(Runtime& rt, const smolvla::Observation& obs, const std::vector<float>& noise, int iters)
{
    Result r;
    for (int i = 0; i < 3; ++i) r.actions = rt.SelectActionChunk(obs, &noise);  // 预热(含 CUDA Graph 录制)
    for (int i = 0; i < iters; ++i)
    {
        const auto t0 = Clock::now();
        r.actions = rt.SelectActionChunk(obs, &noise);
        r.ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / iters;
        if constexpr (std::is_same_v<Runtime, smolvla::SmolVLAFastRuntime>)
        {
            const auto& s = rt.LastStageTimes();
            r.stages.vision += s.vision / iters;
            r.stages.assembler += s.assembler / iters;
            r.stages.prefill += s.prefill / iters;
            r.stages.denoise += s.denoise / iters;
        }
    }
    return r;
}

double MaxDiff(const std::vector<float>& a, const std::vector<float>& b)
{
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    return m;
}

}  // namespace

int main(int argc, char** argv)
{
    smolvla::RuntimeConfig cfg;
    cfg.artifacts_dir = "../../artifacts";
    int iters = 20;
    for (int i = 1; i < argc; ++i)
    {
        const std::string k = argv[i];
        if (k == "--artifacts" && i + 1 < argc)
            cfg.artifacts_dir = argv[++i];
        else if (k == "--iters" && i + 1 < argc)
            iters = std::stoi(argv[++i]);
        else if (k == "--no-tensorrt")
            cfg.use_tensorrt = false;
        else
        {
            std::printf("用法: %s [--artifacts DIR] [--iters N] [--no-tensorrt]\n", argv[0]);
            return k == "--help" || k == "-h" ? 0 : 1;
        }
    }
    cfg.artifacts_dir = std::filesystem::absolute(cfg.artifacts_dir).lexically_normal().string();
    cfg.trt_cache_dir = cfg.artifacts_dir + "/.trt_cache";
    const auto& d = cfg.dims;

    // 固定种子的观测: 2 路随机图 + 1 路 empty camera, 指令 "pick up the blue block"
    std::mt19937 g(42);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::normal_distribution<float> nd(0.f, 1.f);
    smolvla::Observation obs;
    obs.images.resize(d.num_images);
    obs.image_valid.assign(d.num_images, true);
    obs.image_valid[d.num_images - 1] = false;
    for (int i = 0; i < d.num_images; ++i)
    {
        obs.images[i].resize(static_cast<size_t>(3) * d.img_size * d.img_size);
        for (auto& v : obs.images[i]) v = obs.image_valid[i] ? u(g) : -1.f;
    }
    const int64_t toks[] = {18188, 614, 260, 4461, 3608, 198};
    obs.lang_tokens.assign(d.lang_len, 0);
    obs.lang_mask.assign(d.lang_len, false);
    for (int i = 0; i < 6; ++i) obs.lang_tokens[i] = toks[i], obs.lang_mask[i] = true;
    obs.state.assign(d.state_dim, 0.f);
    for (int i = 0; i < 7; ++i) obs.state[i] = nd(g) * 0.5f;
    std::vector<float> noise(static_cast<size_t>(d.chunk) * d.action_dim);
    for (auto& v : noise) v = nd(g);

    std::printf("后端: %s FP32, 每组 %d 次取平均\n", cfg.use_tensorrt ? "TensorRT" : "CUDA", iters);
    Result base, fast, graph;
    {
        smolvla::SmolVLARuntime rt(cfg);
        base = Bench(rt, obs, noise, iters);
    }
    {
        cfg.denoise_cuda_graph = false;
        smolvla::SmolVLAFastRuntime rt(cfg);
        fast = Bench(rt, obs, noise, iters);
    }
    {
        cfg.denoise_cuda_graph = true;
        smolvla::SmolVLAFastRuntime rt(cfg);
        graph = Bench(rt, obs, noise, iters);
    }

    std::printf("\n%-28s %8s   %7s %7s %7s %7s   %s\n", "", "单帧ms", "视觉", "组装", "prefill",
                "去噪x10", "与原版 max|diff|(rad)");
    std::printf("%-26s %8.1f   %31s   -\n", "原版 SmolVLARuntime", base.ms, "");
    auto row = [&](const char* name, const Result& r)
    {
        std::printf("%-26s %8.1f   %7.1f %7.1f %7.1f %7.1f   %.2e\n", name, r.ms, r.stages.vision,
                    r.stages.assembler, r.stages.prefill, r.stages.denoise, MaxDiff(base.actions, r.actions));
    };
    row("优化版", fast);
    row("优化版 + 去噪 CUDA Graph", graph);
    return 0;
}
