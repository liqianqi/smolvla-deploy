// 流程: 上电 -> 零力矩+重力补偿手动摆位 -> Enter -> 保持 + 加载模型/相机 -> 模型控制
// 用法示例:
// ./smolvla_deploy --artifacts ../../artifacts    # 默认 can0 / USB 10
// usbcamera的序列号需要自己查询
#include <algorithm>
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
    std::string can_iface = "can1";
    std::string rs_serial;
    int usb_device = 10;
    bool swap_cameras = false;
    bool no_arm = false;
    bool fake_cameras = false;  // 无相机硬件时用全黑帧冒烟测试
    bool manual_init = true;
    // 展示用: 任务指令. 实际 token 来自 artifacts/lang_tokens.bin(tools/export_lang_tokens.py 离线编码),
    // 此参数只打印, 换指令需重新导出 lang_tokens.bin
    std::string instruction = "pick up the blue block";
    int exec_horizon = 8;
    double action_dt = 0.11;
    double control_hz = 200.0;
    double max_dq_per_action = 0.20;
    double max_dq_per_tick = 0.08;  
    int skip_chunk_head = 0;
    double align_max_jump = 0.35;
    int align_search = 12;
    double action_filter = 0.5;
    double j1_filter = -1.0;
    double j1_bias = 0.0;
    // 全程关节偏置(rad). 负=更低/更前: 失败抓取比成功的 J3 高 0.07、J4 高 0.16, 各补一部分
    double j4_bias = -0.04;
    double j3_filter = -1.0;
    double j3_bias = -0.02;
    double gripper_filter = 0.6;
    // 夹爪动作相对关节错开 k 步(k*action_dt 秒): 正=夹爪晚动, 负=夹爪早动(先夹紧再抬), 0=关
    int gripper_delay = 0;
    double kp = 120.0;
    double kd = 5.0;
    int gripper_id = 7;
    double gripper_kp = 8.0;
    double gripper_kd = 0.4;
    double gripper_init_angle = -0.55;
    double gripper_open_extra = -0.18;  
    double grasp_j2_bias = 0.06;
    double grasp_j3_bias = -0.04;
    double grasp_j4_bias = -0.04;
    double grasp_grip_bias = 0.18;
    double lift_start = 0.50;
    double lift_j2 = 0.0;
    double lift_j3 = 0.0;
    std::array<double, 6> gravity_torques = {0, 1.2, -1.2, 0.1, 0, 0};
    int max_iters = -1;        // <0 表示不限
    std::string save_obs_dir;  // 非空: 每次推理把两路相机帧存到该目录(诊断用)

    // 自检模式: 从文件读固定观测跑一次推理, 输出与 Python 参考逐位对比
    std::string test_obs_dir;  // 含 primary_camera1.jpg / wrist_camera2.jpg
    std::string test_noise;  // noise.bin: float32 x (chunk*32), tools/compare_with_pytorch.py 导出
    std::string test_state = "0.0864,1.6489,-1.4201,1.0095,-0.0449,-0.0288,1.0";
    std::string dump_actions;  // 输出 cpp_actions.bin: float32 x (chunk*7)
    bool no_tensorrt = false;  // 关 TensorRT EP(退回 CUDA EP)
    bool fp16 = false;
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
        else if (k == "--instruction")
            a.instruction = next();
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
        else if (k == "--skip-chunk-head")
            a.skip_chunk_head = std::stoi(next());
        else if (k == "--align-max-jump")
            a.align_max_jump = std::stod(next());
        else if (k == "--align-search")
            a.align_search = std::stoi(next());
        else if (k == "--grasp-j2-bias")
            a.grasp_j2_bias = std::stod(next());
        else if (k == "--grasp-j3-bias")
            a.grasp_j3_bias = std::stod(next());
        else if (k == "--grasp-j4-bias")
            a.grasp_j4_bias = std::stod(next());
        else if (k == "--grasp-grip-bias")
            a.grasp_grip_bias = std::stod(next());
        else if (k == "--lift-start")
            a.lift_start = std::stod(next());
        else if (k == "--lift-j2")
            a.lift_j2 = std::stod(next());
        else if (k == "--lift-j3")
            a.lift_j3 = std::stod(next());
        else if (k == "--action-filter")
            a.action_filter = std::stod(next());
        else if (k == "--j1-filter")
            a.j1_filter = std::stod(next());
        else if (k == "--j1-bias")
            a.j1_bias = std::stod(next());
        else if (k == "--j4-bias")
            a.j4_bias = std::stod(next());
        else if (k == "--j3-filter")
            a.j3_filter = std::stod(next());
        else if (k == "--j3-bias")
            a.j3_bias = std::stod(next());
        else if (k == "--kp")
            a.kp = std::stod(next());
        else if (k == "--kd")
            a.kd = std::stod(next());
        else if (k == "--gripper-filter")
            a.gripper_filter = std::stod(next());
        else if (k == "--gripper-delay")
            a.gripper_delay = std::stoi(next());
        else if (k == "--gripper-id")
            a.gripper_id = std::stoi(next());
        else if (k == "--no-gripper")
            a.gripper_id = 0;
        else if (k == "--gripper-kp")
            a.gripper_kp = std::stod(next());
        else if (k == "--gripper-kd")
            a.gripper_kd = std::stod(next());
        else if (k == "--gripper-open-extra")
            a.gripper_open_extra = std::stod(next());
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
                "用法: %s [--artifacts DIR] [--instruction TEXT] [--can IFACE] [--rs-serial SN] [--usb-device N]\n"
                "          [--swap-cameras] [--no-arm] [--fake-cameras] [--no-manual-init]\n"
                "          [--gravity-torques \"0,1.2,-1.2,0.1,0,0\"] [--exec-horizon N]\n"
                "          [--action-dt S] [--control-hz HZ] [--max-dq-per-action RAD]\n"
                "          [--max-dq-per-tick RAD] [--skip-chunk-head N]\n"
                "          [--align-max-jump RAD] [--align-search N] [--action-filter A]\n"
                "          [--j1-filter A] [--j1-bias RAD] [--j3-filter A] [--j3-bias RAD] [--j4-bias RAD]\n"
                "          [--lift-j2 RAD] [--lift-j3 RAD] [--lift-start A]\n"
                "          [--kp KP] [--kd KD]\n"
                "          [--gripper-id N] [--no-gripper] [--gripper-filter A] [--gripper-delay K]\n"
                "          [--gripper-kp KP] [--gripper-kd KD] [--gripper-open-extra RAD]\n"
                "          [--max-iters N] [--save-obs DIR]\n"
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

// 丢掉新 chunk 开头的"收臂重开"前缀, 再向后找离参考角最近的一步.
// 参考必须是上次命令角: 用实测角会对齐到滞后姿态, 每段都先缩回再伸出.
int AlignActionChunk(const std::vector<float>& actions, int chunk, int dim,
                     const std::vector<double>* q_ref, int skip_head, double align_max_jump,
                     int search_n)
{
    const int start = std::clamp(skip_head, 0, std::max(0, chunk - 1));
    if (q_ref == nullptr || q_ref->size() < 6) return start;
    auto dist = [&](int i) -> double
    {
        double s = 0;
        for (int j = 0; j < 6; ++j)
        {
            const double dlt = static_cast<double>(actions[static_cast<size_t>(i) * dim + j]) -
                               (*q_ref)[j];
            s += dlt * dlt;
        }
        return std::sqrt(s);
    };
    const double d0 = dist(start);
    if (d0 <= align_max_jump) return start;
    const int n = std::min(std::max(1, search_n), chunk - start);
    int best = 0;
    double best_d = d0;
    for (int j = 0; j < n; ++j)
    {
        const double dj = dist(start + j);
        if (dj < best_d)
        {
            best_d = dj;
            best = j;
        }
    }
    if (best_d + 1e-6 < d0)
    {
        std::printf("[ALIGN] skip extra %d after head=%d, jump %.3f->%.3f rad (vs meas, like Python)\n",
                    best, start, d0, best_d);
        return start + best;
    }
    std::printf("[ALIGN] chunk head jump=%.3f rad after skip_head=%d (vs meas)\n", d0, start);
    return start;
}

// 一阶低通: y = a*x + (1-a)*y. a 越小越稳.
struct JointActionFilter
{
    double alpha = 0.25;
    double j1_alpha = 0.50;
    double j3_alpha = 0.50;
    double grip_alpha = 0.50;
    bool has_q = false;
    bool has_g = false;
    double q[6]{};
    double grip = 0;

    void Reset(const std::vector<double>& q0, double g)
    {
        for (int i = 0; i < 6; ++i) q[i] = q0[i];
        has_q = true;
        grip = g;
        has_g = true;
    }

    void Step(std::vector<float>& action)
    {
        const double a = std::clamp(alpha, 0.0, 1.0);
        const double a1 = std::clamp(j1_alpha, 0.0, 1.0);
        const double a3 = std::clamp(j3_alpha, 0.0, 1.0);
        if (!has_q)
        {
            for (int i = 0; i < 6; ++i) q[i] = action[i];
            has_q = true;
        }
        else
        {
            q[0] = a1 * action[0] + (1.0 - a1) * q[0];
            q[2] = a3 * action[2] + (1.0 - a3) * q[2];
            for (int i = 1; i < 6; ++i)
            {
                if (i == 2) continue;
                q[i] = a * action[i] + (1.0 - a) * q[i];
            }
        }
        for (int i = 0; i < 6; ++i) action[i] = static_cast<float>(q[i]);
        if (action.size() < 7) return;
        const double ag = std::clamp(grip_alpha, 0.0, 1.0);
        if (!has_g || ag >= 1.0)
        {
            grip = action[6];
            has_g = true;
        }
        else
        {
            grip = ag * action[6] + (1.0 - ag) * grip;
        }
        action[6] = static_cast<float>(grip);
    }
};

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
        cfg.artifacts_dir = std::filesystem::absolute(args.artifacts).lexically_normal().string();
        cfg.use_tensorrt = !args.no_tensorrt;
        cfg.fp16 = args.fp16;
        cfg.fp16_engines = args.fp16_engines;
        // 运行时内部会按每个引擎的精度追加 /fp32 或 /fp16 子目录
        cfg.trt_cache_dir = cfg.artifacts_dir + "/.trt_cache";
        args.artifacts = cfg.artifacts_dir;
        const auto& d = cfg.dims;

        constexpr int kStateReal = 7;  // [J1..J6, gripper_angle]
        std::vector<int64_t> lang_ids;
        std::vector<bool> lang_mask;
        LoadLangTokens(args.artifacts + "/lang_tokens.bin", d.lang_len, lang_ids, lang_mask);
        std::printf("[INIT] 任务指令: \"%s\"\n", args.instruction.c_str());
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
            acfg.kp = args.kp;
            acfg.kd = args.kd;
            acfg.gripper_id = static_cast<uint8_t>(std::max(0, args.gripper_id));
            acfg.gripper_kp = args.gripper_kp;
            acfg.gripper_kd = args.gripper_kd;
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

        JointActionFilter filt;
        const auto inherit = [&](double v) { return v >= 0.0 ? v : args.action_filter; };
        filt.alpha = args.action_filter;
        filt.j1_alpha = inherit(args.j1_filter);
        filt.j3_alpha = inherit(args.j3_filter);
        filt.grip_alpha = inherit(args.gripper_filter);
        if (arm)
        {
            filt.Reset(arm->GetJointPositions(),
                       args.gripper_id > 0 ? arm->GetGripperAngle() : args.gripper_init_angle);
        }

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

        // 新数据集第 7 维是夹爪角(rad, 越负越开, 接近 0 闭合), 不是 0/1 flag.
        double tracked_gripper_angle = args.gripper_init_angle;
        float grip_prev = static_cast<float>(filt.has_g ? filt.grip : args.gripper_init_angle);
        constexpr double kGraspOpenRad = -1.10;
        constexpr double kGraspCloseRad = -0.05;
        constexpr double kGripMin = -1.75;
        constexpr double kGripMax = 0.10;

        std::printf(
            "[RUN] 控制回路启动. Ctrl+C 停止\n"
            "[PARITY] exec_horizon=%d action_dt=%.2f control_hz=%.0f "
            "action_filter=%.2f max_dq_per_action=%.3f max_dq_per_tick=%.3f\n"
            "[PARITY] skip_chunk_head=%d align_max_jump=%.2f align_search=%d "
            "align_ref=meas kp=%.0f kd=%.1f gravity=%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"
            "[PARITY] grasp j2=%+.3f j3=%+.3f j4=%+.3f grip=%+.3f | "
            "lift j2=%+.3f j3=%+.3f (0=Python无抬升)\n"
            "[PARITY] gripper id=%d kp=%.1f kd=%.1f filter=%.2f open_extra=%+.2f "
            "j1_filter=%.2f j3_filter=%.2f\n",
            args.exec_horizon, args.action_dt, args.control_hz, args.action_filter,
            args.max_dq_per_action, args.max_dq_per_tick, args.skip_chunk_head,
            args.align_max_jump, args.align_search, args.kp, args.kd, args.gravity_torques[0],
            args.gravity_torques[1], args.gravity_torques[2], args.gravity_torques[3],
            args.gravity_torques[4], args.gravity_torques[5], args.grasp_j2_bias,
            args.grasp_j3_bias, args.grasp_j4_bias, args.grasp_grip_bias, args.lift_j2,
            args.lift_j3, args.gripper_id, args.gripper_kp, args.gripper_kd, filt.grip_alpha,
            args.gripper_open_extra, filt.j1_alpha, filt.j3_alpha);

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

            // 2) 状态: [J1..J6, gripper_angle] -> MEAN_STD -> pad 到 32
            std::vector<double> state7(kStateReal);
            if (arm)
            {
                std::vector<double> q = arm->GetJointPositions();
                for (int i = 0; i < 6; ++i) state7[i] = q[i];
                state7[6] = arm->GetGripperAngle();
                tracked_gripper_angle = state7[6];
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
            // 不做对齐搜索(AlignActionChunk 已停用), 只固定丢掉 chunk 开头 skip_chunk_head 步
            const int chunk_off = std::clamp(args.skip_chunk_head, 0, d.chunk - 1);
            const int n_exec = std::min(args.exec_horizon, d.chunk - chunk_off);
            if (arm && !arm->LastCommand().empty())
            {
                const auto& cmd = arm->LastCommand();
                std::printf("[TRACK] J1 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "J2 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "J3 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "G cmd=%+.3f meas=%+.3f err=%+.3f\n",
                            cmd[0], state7[0], state7[0] - cmd[0], cmd[1], state7[1],
                            state7[1] - cmd[1], cmd[2], state7[2], state7[2] - cmd[2],
                            arm->HasGripperCommand() ? arm->LastGripperCommand() : state7[6],
                            state7[6],
                            state7[6] - (arm->HasGripperCommand() ? arm->LastGripperCommand()
                                                                  : state7[6]));
            }
            std::printf("[INF %d] %.0fms, 执行前 %d 步 skip_head=%d\n", iter, inf_ms, n_exec,
                        chunk_off);

            // 4) 执行(或打印)
            for (int s = 0; s < n_exec && !g_stop.load(); ++s)
            {
                std::vector<float> action(
                    actions.begin() + static_cast<size_t>(chunk_off + s) * D,
                    actions.begin() + static_cast<size_t>(chunk_off + s + 1) * D);
                // 改动A: 夹爪取错开 k 步的动作(负 k 即提前); 越界时沿用上一步/钉在 chunk 末尾
                if (args.gripper_delay != 0)
                {
                    const int gs = std::min(chunk_off + s - args.gripper_delay, d.chunk - 1);
                    action[6] = gs >= 0 ? actions[static_cast<size_t>(gs) * D + 6] : grip_prev;
                    grip_prev = action[6];
                }
                filt.Step(action);
                if (args.j1_bias != 0.0) action[0] += static_cast<float>(args.j1_bias);
                if (args.j3_bias != 0.0) action[2] += static_cast<float>(args.j3_bias);
                if (args.j4_bias != 0.0) action[3] += static_cast<float>(args.j4_bias);
                // 抓取相位偏置(同 Python _apply_grasp_bias): 只在夹爪趋于闭合时按闭合程度
                // amt∈[0,1] 线性加一点下探/前俯并夹紧, 空中接近阶段(amt=0)不受影响
                {
                    const double amt = std::clamp(
                        (action[6] - kGraspOpenRad) / (kGraspCloseRad - kGraspOpenRad), 0.0, 1.0);
                    if (amt > 1e-3)
                    {
                        action[1] += static_cast<float>(amt * args.grasp_j2_bias);
                        action[2] += static_cast<float>(amt * args.grasp_j3_bias);
                        action[3] += static_cast<float>(amt * args.grasp_j4_bias);
                        action[6] = static_cast<float>(std::clamp(
                            action[6] + amt * args.grasp_grip_bias, kGripMin, kGripMax));
                    }
                }
                tracked_gripper_angle = action[6];
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
