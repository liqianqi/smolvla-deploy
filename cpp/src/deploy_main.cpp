// SmolVLA C++ 实机部署入口:复刻 run_ela3_smolvla_inference.py --mode deploy 的闭环
//
//   RealSense(主视角) + USB(腕部) -> resize_with_pad 512 + [-1,1]
//   6 关节角 + 夹爪标志 -> MEAN_STD 归一化 -> pad 32
//   lang_tokens.bin(tools/export_lang_tokens.py 离线导出)
//        -> SmolVLARuntime(4 引擎 ONNX + TensorRT EP)
//        -> 50 步 chunk -> 执行前 exec_horizon 步(0.2s/步 @30Hz 插值 + 限速)
//
// 流程: 上电 -> 零力矩+重力补偿手动摆位 -> Enter -> 保持 + 加载模型/相机 -> 模型控制
// 用法示例:
//   ./smolvla_deploy --artifacts ../../artifacts              # 默认 can0 / USB 10
//   ./smolvla_deploy --artifacts ../../artifacts --no-arm     # 无机械臂,只推理打印
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "arm_ela3.hpp"
#include "deploy_camera.hpp"
#include "image_preproc.hpp"
#include "smolvla_runtime.hpp"

namespace
{

std::atomic<bool> g_stop{false};
void OnSigInt(int)
{
    g_stop.store(true);
}

struct DeployArgs
{
    std::string artifacts = "../../artifacts";
    std::string can_iface = "can0";
    std::string rs_serial;
    int usb_device = 10;
    bool swap_cameras = false;
    bool no_arm = false;
    bool fake_cameras = false;  // 无相机硬件时用全黑帧冒烟测试
    bool manual_init = true;    // 启动时零力矩+重力补偿手动摆位, Enter 后开始
    // 与 Python 实机成功配置一致 (--exec_horizon 8 --action_dt 0.1):
    // chunk 前几步几乎贴着当前位置, 只执行前 2 步会一直在轨迹起点打转
    int exec_horizon = 8;
    double action_dt = 0.1;
    double control_hz = 30.0;
    double max_dq_per_action = 0.25;  // 动作限幅: 单 action 相对当前最大跳变(rad)
    double max_dq_per_tick = 0.08;    // 动作限幅: 每 30Hz tick 最大步长(rad)
    std::array<double, 6> gravity_torques = {0, 1.2, -1.2, 0.1, 0, 0};
    int max_iters = -1;        // <0 表示不限
    std::string save_obs_dir;  // 非空: 每次推理把两路相机帧存到该目录(诊断用)

    // 自检模式: 从文件读固定观测跑一次推理, 输出与 Python 参考逐位对比
    std::string test_obs_dir;  // 含 primary_camera1.jpg / wrist_camera2.jpg
    std::string test_noise;  // noise.bin: float32 x (chunk*32), tools/compare_with_pytorch.py 导出
    std::string test_state = "0.0864,1.6489,-1.4201,1.0095,-0.0449,-0.0288,1.0";
    std::string dump_actions;  // 输出 cpp_actions.bin: float32 x (chunk*7)
    bool no_tensorrt = false;  // 关 TensorRT EP(退回 CUDA EP)
    // 默认 FP32: 实测 TensorRT FP16 会把该模型输出打崩(动作塌向训练均值,
    // 机械臂原地晃动不朝目标走), FP32/CUDA 与 PyTorch 逐位一致
    bool fp16 = false;
    // 按引擎开 FP16(定位精度问题用), 如 "vision" 或 "vision,denoise";
    // 可选名字: vision / assembler / prefill / denoise. 非空时覆盖 --fp16
    std::string fp16_engines;
};

std::array<double, 6> ParseSix(const std::string& s)
{
    std::array<double, 6> out{};
    std::stringstream ss(s);
    std::string item;
    int i = 0;
    while (std::getline(ss, item, ',') && i < 6) out[i++] = std::stod(item);
    if (i != 6) throw std::runtime_error("需要 6 个逗号分隔的数值: " + s);
    return out;
}

DeployArgs ParseArgs(int argc, char** argv)
{
    DeployArgs a;
    for (int i = 1; i < argc; ++i)
    {
        std::string k = argv[i];
        auto next = [&]() -> std::string
        {
            if (i + 1 >= argc) throw std::runtime_error("参数缺少值: " + k);
            return argv[++i];
        };
        if (k == "--artifacts")
            a.artifacts = next();
        else if (k == "--can")
            a.can_iface = next();
        else if (k == "--rs-serial")
            a.rs_serial = next();
        else if (k == "--usb-device")
            a.usb_device = std::stoi(next());
        else if (k == "--swap-cameras")
            a.swap_cameras = true;
        else if (k == "--no-arm")
            a.no_arm = true;
        else if (k == "--fake-cameras")
            a.fake_cameras = true;
        else if (k == "--no-manual-init")
            a.manual_init = false;
        else if (k == "--gravity-torques")
            a.gravity_torques = ParseSix(next());
        else if (k == "--exec-horizon")
            a.exec_horizon = std::stoi(next());
        else if (k == "--action-dt")
            a.action_dt = std::stod(next());
        else if (k == "--control-hz")
            a.control_hz = std::stod(next());
        else if (k == "--max-dq-per-action")
            a.max_dq_per_action = std::stod(next());
        else if (k == "--max-dq-per-tick")
            a.max_dq_per_tick = std::stod(next());
        else if (k == "--max-iters")
            a.max_iters = std::stoi(next());
        else if (k == "--save-obs")
            a.save_obs_dir = next();
        else if (k == "--test-obs")
            a.test_obs_dir = next();
        else if (k == "--test-noise")
            a.test_noise = next();
        else if (k == "--test-state")
            a.test_state = next();
        else if (k == "--dump-actions")
            a.dump_actions = next();
        else if (k == "--no-tensorrt")
            a.no_tensorrt = true;
        else if (k == "--fp16")
            a.fp16 = true;
        else if (k == "--fp16-engines")
            a.fp16_engines = next();
        else
        {
            std::printf(
                "用法: %s [--artifacts DIR] [--can IFACE] [--rs-serial SN] [--usb-device N]\n"
                "          [--swap-cameras] [--no-arm] [--fake-cameras] [--no-manual-init]\n"
                "          [--gravity-torques \"0,1.2,-1.2,0.1,0,0\"] [--exec-horizon N]\n"
                "          [--action-dt S] [--control-hz HZ] [--max-dq-per-action RAD]\n"
                "          [--max-dq-per-tick RAD] [--max-iters N] [--save-obs DIR]\n"
                "          [--no-tensorrt] [--fp16] [--fp16-engines vision,prefill,...]\n"
                "          [--test-obs DIR] [--test-noise BIN]\n"
                "          [--test-state \"q1,..,q6,grip\"] [--dump-actions BIN]\n",
                argv[0]);
            std::exit(k == "--help" || k == "-h" ? 0 : 1);
        }
    }
    return a;
}

// lang_tokens.bin: [int64 x 48 ids][uint8 x 48 mask](tools/export_lang_tokens.py)
void LoadLangTokens(const std::string& path, int lang_len, std::vector<int64_t>& ids,
                    std::vector<bool>& mask)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error("找不到 " + path +
                                 ", 请先运行 tools/export_lang_tokens.py 导出语言指令");
    }
    ids.resize(lang_len);
    f.read(reinterpret_cast<char*>(ids.data()), sizeof(int64_t) * lang_len);
    std::vector<uint8_t> m(lang_len);
    f.read(reinterpret_cast<char*>(m.data()), lang_len);
    if (!f) throw std::runtime_error("lang_tokens.bin 大小不足: " + path);
    mask.resize(lang_len);
    for (int i = 0; i < lang_len; ++i) mask[i] = m[i] != 0;
}

// state_norm_stats.bin: [float32 mean x7][float32 std x7](tools/export_state_stats.py)
void LoadStateStats(const std::string& path, int dim, std::vector<float>& mean,
                    std::vector<float>& stdv)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        throw std::runtime_error("找不到 " + path +
                                 ", 请先运行 tools/export_state_stats.py 导出状态统计量");
    }
    mean.resize(dim);
    stdv.resize(dim);
    f.read(reinterpret_cast<char*>(mean.data()), sizeof(float) * dim);
    f.read(reinterpret_cast<char*>(stdv.data()), sizeof(float) * dim);
    if (!f) throw std::runtime_error("state_norm_stats.bin 大小不足: " + path);
}

}  // namespace

int main(int argc, char** argv)
{
    using Clock = std::chrono::steady_clock;
    DeployArgs args;
    try
    {
        args = ParseArgs(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::signal(SIGINT, OnSigInt);

    try
    {
        // ---- 推理运行时(4 引擎 ONNX, 输出已反归一化的 7 维动作)----
        smolvla::RuntimeConfig cfg;
        cfg.artifacts_dir = args.artifacts;
        cfg.use_tensorrt = !args.no_tensorrt;
        cfg.fp16 = args.fp16;
        cfg.fp16_engines = args.fp16_engines;
        // 运行时内部会按每个引擎的精度追加 /fp32 或 /fp16 子目录
        cfg.trt_cache_dir = cfg.artifacts_dir + "/.trt_cache";
        const auto& d = cfg.dims;

        constexpr int kStateReal = 7;  // [J1..J6, gripper_open]
        std::vector<int64_t> lang_ids;
        std::vector<bool> lang_mask;
        LoadLangTokens(args.artifacts + "/lang_tokens.bin", d.lang_len, lang_ids, lang_mask);
        std::vector<float> st_mean, st_std;
        LoadStateStats(args.artifacts + "/state_norm_stats.bin", kStateReal, st_mean, st_std);

        // ---- 自检模式: 固定观测 + 固定噪声跑一次推理, 供与 Python 参考对比 ----
        if (!args.test_obs_dir.empty())
        {
            std::printf("[TEST] 自检模式: obs=%s tensorrt=%d fp16=%d fp16_engines=%s\n",
                        args.test_obs_dir.c_str(), cfg.use_tensorrt ? 1 : 0, cfg.fp16 ? 1 : 0,
                        cfg.fp16_engines.empty() ? "(无)" : cfg.fp16_engines.c_str());
            cv::Mat bgr_p = cv::imread(args.test_obs_dir + "/primary_camera1.jpg");
            cv::Mat bgr_w = cv::imread(args.test_obs_dir + "/wrist_camera2.jpg");
            if (bgr_p.empty() || bgr_w.empty())
            {
                throw std::runtime_error(
                    "--test-obs 目录需要 primary_camera1.jpg / wrist_camera2.jpg");
            }
            cv::Mat rgb_p, rgb_w;
            cv::cvtColor(bgr_p, rgb_p, cv::COLOR_BGR2RGB);
            cv::cvtColor(bgr_w, rgb_w, cv::COLOR_BGR2RGB);

            // state: 7 个逗号分隔值
            std::vector<double> state7;
            {
                std::stringstream ss(args.test_state);
                std::string item;
                while (std::getline(ss, item, ',')) state7.push_back(std::stod(item));
                if (state7.size() != static_cast<size_t>(kStateReal))
                {
                    throw std::runtime_error("--test-state 需要 7 个值");
                }
            }

            const size_t noise_n = static_cast<size_t>(d.chunk) * d.action_dim;
            std::vector<float> noise(noise_n);
            if (!args.test_noise.empty())
            {
                std::ifstream f(args.test_noise, std::ios::binary);
                f.read(reinterpret_cast<char*>(noise.data()), sizeof(float) * noise_n);
                if (!f) throw std::runtime_error("noise 文件大小不足: " + args.test_noise);
            }
            else
            {
                std::mt19937 gen(0);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (auto& v : noise) v = nd(gen);
            }

            smolvla::SmolVLARuntime rt(cfg);
            smolvla::Observation obs;
            obs.images.resize(d.num_images);
            obs.image_valid.assign(d.num_images, true);
            obs.image_valid[d.num_images - 1] = false;
            obs.images[d.num_images - 1].assign(static_cast<size_t>(3) * d.img_size * d.img_size,
                                                -1.0f);
            obs.images[0] = smolvla::PreprocessImage(rgb_p, d.img_size);
            obs.images[1] = smolvla::PreprocessImage(rgb_w, d.img_size);
            obs.lang_tokens = lang_ids;
            obs.lang_mask = lang_mask;
            obs.state.assign(d.state_dim, 0.0f);
            for (int i = 0; i < kStateReal; ++i)
            {
                obs.state[i] = static_cast<float>((state7[i] - st_mean[i]) / (st_std[i] + 1e-8));
            }

            std::vector<float> actions = rt.SelectActionChunk(obs, &noise);  // [chunk, 7]
            const int D = d.real_action_dim;
            std::printf("[TEST] 前 5 步动作:\n");
            for (int s = 0; s < 5; ++s)
            {
                std::printf("  [%d]", s);
                for (int a = 0; a < D; ++a) std::printf(" %+.4f", actions[s * D + a]);
                std::printf("\n");
            }
            if (!args.dump_actions.empty())
            {
                std::ofstream f(args.dump_actions, std::ios::binary);
                f.write(reinterpret_cast<const char*>(actions.data()),
                        sizeof(float) * actions.size());
                std::printf("[TEST] 动作 chunk 已写出 %s (float32 x %zu)\n",
                            args.dump_actions.c_str(), actions.size());
            }
            return 0;
        }

        // ---- 机械臂(先上电: 零力矩+重力补偿手动摆位要在模型加载前)----
        std::unique_ptr<smolvla::ElA3Arm> arm;
        if (!args.no_arm)
        {
            smolvla::ArmConfig acfg;
            acfg.can_iface = args.can_iface;
            acfg.max_dq_per_action = args.max_dq_per_action;
            acfg.max_dq_per_tick = args.max_dq_per_tick;
            acfg.control_hz = args.control_hz;
            acfg.gravity_torques = args.gravity_torques;
            arm = std::make_unique<smolvla::ElA3Arm>(acfg);
            if (args.manual_init)
            {
                // 手动摆位 -> Enter -> 捕获并保持(加载模型/相机期间不下垂)
                arm->ManualInitAndHold();
            }
        }
        else
        {
            std::printf("[INIT] --no-arm: 跳过机械臂, state 用训练均值占位, 只打印动作\n");
        }

        std::printf("[INIT] 加载 ONNX 引擎(首次运行会构建 TensorRT engine, 约 3-4 分钟)...\n");
        smolvla::SmolVLARuntime rt(cfg);
        std::printf("[INIT] 推理引擎就绪\n");

        // ---- 相机 ----
        std::unique_ptr<smolvla::RealSenseCamera> rs_cam;
        std::unique_ptr<smolvla::UsbCamera> usb_cam;
        if (!args.fake_cameras)
        {
            std::printf("[INIT] 打开 RealSense(主视角)+ USB %d(腕部)...\n", args.usb_device);
            rs_cam = std::make_unique<smolvla::RealSenseCamera>(640, 480, 30, args.rs_serial);
            usb_cam = std::make_unique<smolvla::UsbCamera>(args.usb_device, 640, 480, 30);
            std::printf("[INIT] 相机就绪\n");
        }
        else
        {
            std::printf("[INIT] --fake-cameras: 用全黑帧代替真实相机(仅冒烟测试)\n");
        }

        // 一切就绪, 释放启动保持, 交给模型控制
        if (arm) arm->StopHold();

        // ---- 观测缓冲(lang/empty camera 固定不变)----
        smolvla::Observation obs;
        obs.images.resize(d.num_images);
        obs.image_valid.assign(d.num_images, true);
        obs.image_valid[d.num_images - 1] = false;  // 第 3 路 empty camera
        obs.images[d.num_images - 1].assign(static_cast<size_t>(3) * d.img_size * d.img_size,
                                            -1.0f);  // 与 Python 一致: 填 -1
        obs.lang_tokens = lang_ids;
        obs.lang_mask = lang_mask;
        obs.state.assign(d.state_dim, 0.0f);

        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> noise(static_cast<size_t>(d.chunk) * d.action_dim);

        double tracked_gripper_open = 1.0;  // 无夹爪电机时, 跟随模型输出
        std::printf(
            "[RUN] 控制回路启动 (exec_horizon=%d action_dt=%.2fs). Ctrl+C 停止\n"
            "[SAFE] 动作限幅: 单 action ≤%.3f rad, 每 tick(%.0fHz) ≤%.3f rad\n",
            args.exec_horizon, args.action_dt, args.max_dq_per_action, args.control_hz,
            args.max_dq_per_tick);

        int iter = 0;
        while (!g_stop.load() && (args.max_iters < 0 || iter < args.max_iters))
        {
            ++iter;
            // 1) 相机
            cv::Mat rgb_primary, rgb_wrist;
            if (!args.fake_cameras)
            {
                rgb_primary = rs_cam->Read();
                rgb_wrist = usb_cam->Read();
            }
            else
            {
                rgb_primary = cv::Mat::zeros(480, 640, CV_8UC3);
                rgb_wrist = cv::Mat::zeros(480, 640, CV_8UC3);
            }
            if (args.swap_cameras) std::swap(rgb_primary, rgb_wrist);
            if (!args.save_obs_dir.empty())
            {
                // 存模型实际看到的两路画面(RGB->BGR 供 imwrite), 每次覆盖
                std::filesystem::create_directories(args.save_obs_dir);
                cv::Mat bgr;
                cv::cvtColor(rgb_primary, bgr, cv::COLOR_RGB2BGR);
                cv::imwrite(args.save_obs_dir + "/primary_camera1.jpg", bgr);
                cv::cvtColor(rgb_wrist, bgr, cv::COLOR_RGB2BGR);
                cv::imwrite(args.save_obs_dir + "/wrist_camera2.jpg", bgr);
            }
            obs.images[0] = smolvla::PreprocessImage(rgb_primary, d.img_size);
            obs.images[1] = smolvla::PreprocessImage(rgb_wrist, d.img_size);

            // 2) 状态: [J1..J6, gripper_open] -> MEAN_STD -> pad 到 32
            std::vector<double> state7(kStateReal);
            if (arm)
            {
                std::vector<double> q = arm->GetJointPositions();
                for (int i = 0; i < 6; ++i) state7[i] = q[i];
                state7[6] = tracked_gripper_open;
            }
            else
            {
                for (int i = 0; i < kStateReal; ++i) state7[i] = st_mean[i];  // 归一化后为 0
            }
            std::fill(obs.state.begin(), obs.state.end(), 0.0f);
            for (int i = 0; i < kStateReal; ++i)
            {
                obs.state[i] = static_cast<float>((state7[i] - st_mean[i]) / (st_std[i] + 1e-8));
            }

            // 3) 推理(每次采新噪声, 与 Python 部署一致)
            for (auto& v : noise) v = nd(rng);
            const auto t0 = Clock::now();
            std::vector<float> actions = rt.SelectActionChunk(obs, &noise);  // [chunk, 7]
            const double inf_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            const int D = d.real_action_dim;
            const int n_exec = std::min(args.exec_horizon, d.chunk);
            std::printf("[INF %d] %.0fms, 执行前 %d 步\n", iter, inf_ms, n_exec);

            // 4) 执行(或打印)
            for (int s = 0; s < n_exec && !g_stop.load(); ++s)
            {
                std::vector<float> action(actions.begin() + static_cast<size_t>(s) * D,
                                          actions.begin() + static_cast<size_t>(s + 1) * D);
                tracked_gripper_open = action[6] > 0.5f ? 1.0 : 0.0;
                std::printf("[ACT %d]", s);
                for (int a = 0; a < D; ++a) std::printf(" %+.4f", action[a]);
                std::printf("\n");
                if (arm)
                {
                    arm->ExecuteJointAction(action, args.action_dt, g_stop);
                }
            }
        }

        if (arm)
        {
            std::printf("[EXIT] 停住当前位置并退出\n");
            arm->HoldCurrentPosition();
        }
        std::printf("[EXIT] 完成\n");
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "[FATAL] %s\n", e.what());
        return 1;
    }
    return 0;
}
