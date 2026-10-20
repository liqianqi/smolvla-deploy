// 原始效果版部署: 模型输出多少就执行多少, 不做任何策略侧修正.
//
// 与 deploy_main.cpp 的区别(刻意去掉的东西):
//   - 无关节偏置(j1/j3/j4_bias)、无抓取相位偏置(grasp_*_bias)、无抬升偏置(lift_*)
//   - 无低通滤波(action_filter/j1_filter/j3_filter/gripper_filter)
//   - 无夹爪时序错开(gripper_delay)、无 chunk 头部丢弃(skip_chunk_head)、无对齐搜索
//   - 无自检模式(--test-obs 等), 需要时用 smolvla_deploy
// 保留的只有"怎么执行"这一层: exec_horizon / action_dt / control_hz, 以及硬件与相机参数.
// 机械臂侧的限位 clip、单步限幅、每 tick 限速是硬件安全措施(ArmConfig 默认值), 不作为调参项暴露.
//
// 流程与 deploy_main 一致: 上电 -> 零力矩+重力补偿手动摆位 -> Enter -> 保持 + 加载模型/相机 -> 模型控制
// 用法示例:
// ./smolvla_deploy_raw --artifacts ../../artifacts
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "arm_ela3.hpp"
#include "deploy_camera.hpp"
#include "image_preproc.hpp"
#include "smolvla_fast_runtime.hpp"
#include "smolvla_runtime.hpp"

namespace
{

std::atomic<bool> g_stop{false};
void OnSigInt(int)
{
    g_stop.store(true);
}

struct RawArgs
{
    std::string artifacts = "../../artifacts";
    std::string can_iface = "can0";
    std::string rs_serial;
    int usb_device = 4;
    bool swap_cameras = false;
    bool no_arm = false;
    bool fake_cameras = false;  // 无相机硬件时用全黑帧冒烟测试
    bool show = false;          // 弹窗实时显示两路相机画面
    bool manual_init = true;
    std::string instruction = "pick up the blue block";  // 仅用于打印, 实际 token 来自 lang_tokens.bin
    int gripper_id = 7;

    // 执行层(非策略修正): 每次推理执行前 N 步, 每步 action_dt 秒, 内部按 control_hz 插值
    int exec_horizon = 8;
    double action_dt = 0.11;
    double control_hz = 200.0;

    int max_iters = -1;        // <0 表示不限
    std::string save_obs_dir;  // 非空: 每次推理把两路相机帧存到该目录(诊断用)
    bool no_tensorrt = false;
    bool fp16 = false;
    std::string fp16_engines;
    bool fast = false;              // SmolVLAFastRuntime: 显存常驻 + 视觉合批(仅 FP32)
    bool denoise_cuda_graph = true; // --fast 时 denoise 是否开 CUDA Graph
};

RawArgs ParseArgs(int argc, char** argv)
{
    RawArgs a;
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
        else if (k == "--show")
            a.show = true;
        else if (k == "--no-manual-init")
            a.manual_init = false;
        else if (k == "--gripper-id")
            a.gripper_id = std::stoi(next());
        else if (k == "--no-gripper")
            a.gripper_id = 0;
        else if (k == "--exec-horizon")
            a.exec_horizon = std::stoi(next());
        else if (k == "--action-dt")
            a.action_dt = std::stod(next());
        else if (k == "--control-hz")
            a.control_hz = std::stod(next());
        else if (k == "--max-iters")
            a.max_iters = std::stoi(next());
        else if (k == "--save-obs")
            a.save_obs_dir = next();
        else if (k == "--no-tensorrt")
            a.no_tensorrt = true;
        else if (k == "--fp16")
            a.fp16 = true;
        else if (k == "--fp16-engines")
            a.fp16_engines = next();
        else if (k == "--fast")
            a.fast = true;
        else if (k == "--no-cuda-graph")
            a.denoise_cuda_graph = false;
        else
        {
            std::printf(
                "用法: %s [--artifacts DIR] [--instruction TEXT] [--can IFACE] [--rs-serial SN] [--usb-device N]\n"
                "          [--swap-cameras] [--no-arm] [--fake-cameras] [--show] [--no-manual-init]\n"
                "          [--gripper-id N] [--no-gripper]\n"
                "          [--exec-horizon N] [--action-dt S] [--control-hz HZ]\n"
                "          [--max-iters N] [--save-obs DIR]\n"
                "          [--no-tensorrt] [--fp16] [--fp16-engines vision,prefill,...]\n"
                "          [--fast [--no-cuda-graph]]  优化版运行时(FP32, 忽略 fp16 选项)\n"
                "说明: 本程序不含任何动作修正/滤波/偏置, 模型输出原样执行.\n",
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
    RawArgs args;
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
        cfg.trt_cache_dir = cfg.artifacts_dir + "/.trt_cache";
        cfg.denoise_cuda_graph = args.denoise_cuda_graph;
        args.artifacts = cfg.artifacts_dir;
        const auto& d = cfg.dims;

        constexpr int kStateReal = 7;  // [J1..J6, gripper_angle]
        std::vector<int64_t> lang_ids;
        std::vector<bool> lang_mask;
        LoadLangTokens(args.artifacts + "/lang_tokens.bin", d.lang_len, lang_ids, lang_mask);
        std::printf("[INIT] 任务指令: \"%s\"\n", args.instruction.c_str());
        std::vector<float> st_mean, st_std;
        LoadStateStats(args.artifacts + "/state_norm_stats.bin", kStateReal, st_mean, st_std);

        // ---- 机械臂(先上电: 手动摆位要在模型加载前)----
        // 控制增益与 deploy_main 相同(kp=120/kd=5), 其余均为 ArmConfig 默认值.
        std::unique_ptr<smolvla::ElA3Arm> arm;
        smolvla::ArmConfig acfg;
        acfg.can_iface = args.can_iface;
        acfg.control_hz = args.control_hz;
        acfg.kp = 120.0;
        acfg.kd = 5.0;
        acfg.gripper_id = static_cast<uint8_t>(std::max(0, args.gripper_id));
        if (!args.no_arm)
        {
            arm = std::make_unique<smolvla::ElA3Arm>(acfg);
            if (args.manual_init) arm->ManualInitAndHold();
        }
        else
        {
            std::printf("[INIT] --no-arm: 跳过机械臂, state 用训练均值占位, 只打印动作\n");
        }

        std::printf("[INIT] 加载 ONNX 引擎(首次运行会构建 TensorRT engine, 约 3-4 分钟)...\n");
        std::unique_ptr<smolvla::SmolVLARuntime> rt_base;
        std::unique_ptr<smolvla::SmolVLAFastRuntime> rt_fast;
        if (args.fast)
            rt_fast = std::make_unique<smolvla::SmolVLAFastRuntime>(cfg);
        else
            rt_base = std::make_unique<smolvla::SmolVLARuntime>(cfg);
        auto select_action_chunk = [&](const smolvla::Observation& o, const std::vector<float>* n)
        { return rt_fast ? rt_fast->SelectActionChunk(o, n) : rt_base->SelectActionChunk(o, n); };
        std::printf("[INIT] 推理引擎就绪(%s)\n",
                    !args.fast                  ? "原版运行时"
                    : args.denoise_cuda_graph ? "优化版运行时, 去噪 CUDA Graph 开"
                                              : "优化版运行时, 去噪 CUDA Graph 关");

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

        // ---- 相机采集线程: 持续保存最新帧, 推理永远拿当前画面 ----
        std::mutex cam_mu;
        cv::Mat latest_primary, latest_wrist;
        std::atomic<bool> cam_stop{false};
        std::thread cam_thread;
        struct CamThreadGuard  // 任何路径退出(含异常)都先停线程, 再析构相机
        {
            std::atomic<bool>& stop;
            std::thread& th;
            ~CamThreadGuard()
            {
                stop.store(true);
                if (th.joinable()) th.join();
            }
        } cam_guard{cam_stop, cam_thread};
        if (!args.fake_cameras)
        {
            cam_thread = std::thread(
                [&]
                {
                    if (args.show)
                    {
                        cv::namedWindow("smolvla_deploy_raw",
                                        cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
                        cv::resizeWindow("smolvla_deploy_raw", 1280, 480);
                    }
                    while (!cam_stop.load())
                    {
                        cv::Mat p, w;
                        try
                        {
                            p = rs_cam->Read();
                            w = usb_cam->Read();
                        }
                        catch (const std::exception& e)
                        {
                            std::printf("[CAM] WARN: %s\n", e.what());
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                            continue;
                        }
                        {
                            std::lock_guard<std::mutex> lk(cam_mu);
                            latest_primary = p;
                            latest_wrist = w;
                        }
                        if (args.show)
                        {
                            cv::Mat canvas;
                            cv::hconcat(p, w, canvas);
                            cv::cvtColor(canvas, canvas, cv::COLOR_RGB2BGR);
                            const cv::Scalar green(0, 255, 0);
                            cv::putText(canvas, "primary (RealSense)", {10, 30},
                                        cv::FONT_HERSHEY_SIMPLEX, 0.8, green, 2);
                            cv::putText(canvas, "wrist (USB)", {p.cols + 10, 30},
                                        cv::FONT_HERSHEY_SIMPLEX, 0.8, green, 2);
                            cv::putText(canvas, "RAW | task: " + args.instruction,
                                        {10, canvas.rows - 15}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                        green, 2);
                            cv::imshow("smolvla_deploy_raw", canvas);
                            cv::waitKey(1);
                        }
                    }
                });
            for (;;)  // 等首帧到位
            {
                {
                    std::lock_guard<std::mutex> lk(cam_mu);
                    if (!latest_primary.empty() && !latest_wrist.empty()) break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
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

        const int D = d.real_action_dim;
        const int n_exec = std::clamp(args.exec_horizon, 1, d.chunk);
        std::printf(
            "[RUN] RAW 模式: 模型输出原样执行, 无偏置/滤波/错开. Ctrl+C 停止\n"
            "[RUN] exec_horizon=%d/%d action_dt=%.2f control_hz=%.0f kp=%.0f kd=%.1f "
            "arm_safety: max_dq_per_action=%.2f max_dq_per_tick=%.2f\n",
            n_exec, d.chunk, args.action_dt, args.control_hz, acfg.kp, acfg.kd,
            acfg.max_dq_per_action, acfg.max_dq_per_tick);

        int iter = 0;
        while (!g_stop.load() && (args.max_iters < 0 || iter < args.max_iters))
        {
            ++iter;
            // 1) 相机
            cv::Mat rgb_primary, rgb_wrist;
            if (!args.fake_cameras)
            {
                std::lock_guard<std::mutex> lk(cam_mu);
                rgb_primary = latest_primary;
                rgb_wrist = latest_wrist;
            }
            else
            {
                rgb_primary = cv::Mat::zeros(480, 640, CV_8UC3);
                rgb_wrist = cv::Mat::zeros(480, 640, CV_8UC3);
            }
            if (args.swap_cameras) std::swap(rgb_primary, rgb_wrist);
            if (!args.save_obs_dir.empty())
            {
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
            std::vector<float> actions = select_action_chunk(obs, &noise);  // [chunk, 7]
            const double inf_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            if (arm && !arm->LastCommand().empty())
            {
                const auto& cmd = arm->LastCommand();
                const double gcmd =
                    arm->HasGripperCommand() ? arm->LastGripperCommand() : state7[6];
                std::printf("[TRACK] J1 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "J2 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "J3 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "J4 cmd=%+.3f meas=%+.3f err=%+.3f | "
                            "G cmd=%+.3f meas=%+.3f err=%+.3f\n",
                            cmd[0], state7[0], state7[0] - cmd[0], cmd[1], state7[1],
                            state7[1] - cmd[1], cmd[2], state7[2], state7[2] - cmd[2], cmd[3],
                            state7[3], state7[3] - cmd[3], gcmd, state7[6], state7[6] - gcmd);
            }
            if (rt_fast)
            {
                const auto& st = rt_fast->LastStageTimes();
                std::printf("[INF %d] %.0fms(视觉 %.1f / 组装 %.1f / prefill %.1f / 去噪 %.1f), "
                            "执行 chunk 前 %d 步\n",
                            iter, inf_ms, st.vision, st.assembler, st.prefill, st.denoise, n_exec);
            }
            else
            {
                std::printf("[INF %d] %.0fms, 执行 chunk 前 %d 步\n", iter, inf_ms, n_exec);
            }

            // 4) 原样执行(或打印): 从 chunk 第 0 步起, 不改动任何一维
            for (int s = 0; s < n_exec && !g_stop.load(); ++s)
            {
                std::vector<float> action(actions.begin() + static_cast<size_t>(s) * D,
                                          actions.begin() + static_cast<size_t>(s + 1) * D);
                std::printf("[ACT %d]", s);
                for (int a = 0; a < D; ++a) std::printf(" %+.4f", action[a]);
                std::printf("\n");
                if (arm) arm->ExecuteJointAction(action, args.action_dt, g_stop);
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
