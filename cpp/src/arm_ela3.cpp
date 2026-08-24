#include "arm_ela3.hpp"

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace smolvla
{
namespace
{

// ---- RobStride 私有协议常量(与 Python robstride_dynamics 一致)----
constexpr uint8_t kHostId = 0xFF;
constexpr uint8_t kTypeMit = 1;     // OPERATION_CONTROL
constexpr uint8_t kTypeStatus = 2;  // OPERATION_STATUS
constexpr uint8_t kTypeEnable = 3;
constexpr uint8_t kTypeDisable = 4;
constexpr uint8_t kTypeReadParam = 17;
constexpr uint8_t kTypeWriteParam = 18;
constexpr uint8_t kTypeFault = 21;
constexpr uint16_t kParamMeasuredPos = 0x3016;  // mechPos, float32
constexpr uint16_t kParamTorqueLimit = 0x700B;  // limit_torque, float32

// rs-00 的 MIT 帧量程
constexpr double kMitPosRange = 4.0 * M_PI;  // rad
constexpr double kMitVelRange = 50.0;        // rad/s
constexpr double kMitTorqueRange = 17.0;     // Nm
constexpr double kMitKpRange = 500.0;
constexpr double kMitKdRange = 5.0;

// URDF 关节限位(rad), 与 Python _JOINT_LIMITS 一致
constexpr double kJointLimits[6][2] = {
    {-2.79253, 2.79253}, {0.0, 3.66519},    {-4.01426, 0.0},
    {-1.5708, 1.5708},   {-1.5708, 1.5708}, {-1.5708, 1.5708},
};

uint16_t ScaleU16(double v, double range)
{
    v = std::clamp(v, -range, range);
    return static_cast<uint16_t>(
        std::clamp(static_cast<int>((v / range + 1.0) * 0x7FFF), 0, 0xFFFF));
}

}  // namespace

// ============================================================================
// RobstrideBus: SocketCAN 上的 RobStride 私有协议(复刻 Python bus.py)
//   扩展帧 29-bit ID = (type << 24) | (extra << 8) | dev_id, 数据区大端
// ============================================================================
class RobstrideBus
{
 public:
    explicit RobstrideBus(const std::string& iface)
    {
        fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
        if (fd_ < 0) throw std::runtime_error("打不开 CAN socket: " + iface);
        ifreq ifr{};
        std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
        if (ioctl(fd_, SIOCGIFINDEX, &ifr) < 0)
        {
            close(fd_);
            throw std::runtime_error("CAN 接口不存在: " + iface + "(sudo ip link set " + iface +
                                     " up?)");
        }
        sockaddr_can addr{};
        addr.can_family = AF_CAN;
        addr.can_ifindex = ifr.ifr_ifindex;
        if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
        {
            close(fd_);
            throw std::runtime_error("CAN bind 失败: " + iface);
        }
    }
    ~RobstrideBus()
    {
        if (fd_ >= 0) close(fd_);
    }

    void Transmit(uint8_t type, uint16_t extra, uint8_t dev_id, const uint8_t* data, uint8_t len)
    {
        std::lock_guard<std::mutex> lk(mu_);
        can_frame fr{};
        fr.can_id = (static_cast<uint32_t>(type) << 24) | (static_cast<uint32_t>(extra) << 8) |
                    dev_id | CAN_EFF_FLAG;
        fr.can_dlc = len;
        if (len) std::memcpy(fr.data, data, len);
        if (write(fd_, &fr, sizeof(fr)) != sizeof(fr))
        {
            throw std::runtime_error("CAN 发送失败(总线 down?)");
        }
    }

    // MIT 力位混合帧. 所有量为电机原始坐标(标定换算由上层完成)
    void WriteMitFrame(uint8_t dev_id, double pos, double vel, double kp, double kd, double torque)
    {
        const uint16_t pos_u = ScaleU16(pos, kMitPosRange);
        const uint16_t vel_u = ScaleU16(vel, kMitVelRange);
        const uint16_t tor_u = ScaleU16(torque, kMitTorqueRange);
        const uint16_t kp_u =
            static_cast<uint16_t>(std::clamp(kp, 0.0, kMitKpRange) / kMitKpRange * 0xFFFF);
        const uint16_t kd_u =
            static_cast<uint16_t>(std::clamp(kd, 0.0, kMitKdRange) / kMitKdRange * 0xFFFF);
        uint8_t d[8] = {static_cast<uint8_t>(pos_u >> 8), static_cast<uint8_t>(pos_u),
                        static_cast<uint8_t>(vel_u >> 8), static_cast<uint8_t>(vel_u),
                        static_cast<uint8_t>(kp_u >> 8),  static_cast<uint8_t>(kp_u),
                        static_cast<uint8_t>(kd_u >> 8),  static_cast<uint8_t>(kd_u)};
        Transmit(kTypeMit, tor_u, dev_id, d, 8);
    }

    void Enable(uint8_t dev_id)
    {
        uint8_t z[8] = {};
        Transmit(kTypeEnable, kHostId, dev_id, z, 8);
    }
    void Disable(uint8_t dev_id)
    {
        uint8_t z[8] = {};
        Transmit(kTypeDisable, kHostId, dev_id, z, 8);
    }

    // 等 dev_id 的状态帧, 返回电机原始位置(rad); 超时抛异常
    double ReadStatusPosition(uint8_t dev_id, double timeout_sec = 0.75)
    {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
        while (std::chrono::steady_clock::now() < deadline)
        {
            can_frame fr{};
            if (!RecvFrame(fr, 0.05)) continue;
            const uint8_t type = (fr.can_id >> 24) & 0x1F;
            const uint16_t extra = (fr.can_id >> 8) & 0xFFFF;
            if (type == kTypeFault)
            {
                const uint8_t fault_dev = extra & 0xFF;
                uint32_t fault = 0;
                std::memcpy(&fault, fr.data, 4);
                if (fault != last_fault_[fault_dev & 0x3F])
                {
                    last_fault_[fault_dev & 0x3F] = fault;
                    std::printf("[ARM] WARN: 电机 %d fault=0x%08X\n", fault_dev, fault);
                }
                continue;
            }
            if (type != kTypeStatus || (extra & 0xFF) != dev_id) continue;
            if (extra & (1u << 12)) std::printf("[ARM] WARN: 电机 %d 堵转\n", dev_id);
            if (extra & (1u << 10)) std::printf("[ARM] WARN: 电机 %d 过温\n", dev_id);
            const uint16_t pos_u = (static_cast<uint16_t>(fr.data[0]) << 8) | fr.data[1];
            return (static_cast<double>(pos_u) / 0x7FFF - 1.0) * kMitPosRange;
        }
        throw std::runtime_error("电机 " + std::to_string(dev_id) + " 状态帧超时");
    }

    void WriteParamF32(uint8_t dev_id, uint16_t index, float value)
    {
        uint8_t d[8] = {static_cast<uint8_t>(index & 0xFF), static_cast<uint8_t>(index >> 8), 0, 0};
        std::memcpy(d + 4, &value, 4);  // float32 LE
        Transmit(kTypeWriteParam, kHostId, dev_id, d, 8);
    }

    // 参数读(READ_PARAMETER 0x11): 返回 float32 值(电机原始坐标)
    double ReadParamF32(uint8_t dev_id, uint16_t index, double timeout_sec = 0.75)
    {
        uint8_t d[8] = {
            static_cast<uint8_t>(index & 0xFF), static_cast<uint8_t>(index >> 8), 0, 0, 0, 0, 0, 0};
        Transmit(kTypeReadParam, kHostId, dev_id, d, 8);
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_sec);
        while (std::chrono::steady_clock::now() < deadline)
        {
            can_frame fr{};
            if (!RecvFrame(fr, 0.05)) continue;
            const uint8_t type = (fr.can_id >> 24) & 0x1F;
            const uint16_t extra = (fr.can_id >> 8) & 0xFFFF;
            if (type != kTypeReadParam || (extra & 0xFF) != dev_id) continue;
            float v = 0;
            std::memcpy(&v, fr.data + 4, 4);  // data[4:8] = float32 LE
            return v;
        }
        throw std::runtime_error("电机 " + std::to_string(dev_id) + " 参数读超时");
    }

    void FlushRx()
    {
        can_frame fr{};
        while (RecvFrame(fr, 0.0))
        {
        }
    }

 private:
    bool RecvFrame(can_frame& fr, double timeout_sec)
    {
        pollfd p{fd_, POLLIN, 0};
        if (poll(&p, 1, static_cast<int>(timeout_sec * 1000)) <= 0) return false;
        if (read(fd_, &fr, sizeof(fr)) != static_cast<ssize_t>(sizeof(fr))) return false;
        if (!(fr.can_id & CAN_EFF_FLAG)) return false;  // 只认扩展帧
        fr.can_id &= CAN_EFF_MASK;
        return true;
    }

    int fd_ = -1;
    std::mutex mu_;
    uint32_t last_fault_[64] = {};
};

// ============================================================================
// ElA3Arm
// ============================================================================
ElA3Arm::ElA3Arm(const ArmConfig& cfg) : cfg_(cfg)
{
    using namespace std::chrono_literals;
    bus_ = std::make_unique<RobstrideBus>(cfg_.can_iface);
    last_q_.assign(6, 0.0);
    cmd_q_.assign(6, 0.0);

    // 上电清 fault latch: disable x2 -> enable(复刻 Python ElA3RobstrideRobot)
    std::vector<uint8_t> all_ids = cfg_.motor_ids;
    if (cfg_.gripper_id > 0) all_ids.push_back(cfg_.gripper_id);

    std::this_thread::sleep_for(300ms);
    for (int round = 0; round < 2; ++round)
    {
        for (uint8_t id : all_ids)
        {
            bus_->Disable(id);
            std::this_thread::sleep_for(20ms);
        }
        std::this_thread::sleep_for(300ms);
    }
    bus_->FlushRx();
    for (uint8_t id : all_ids)
    {
        bus_->Enable(id);
        std::this_thread::sleep_for(20ms);
    }
    std::this_thread::sleep_for(100ms);
    bus_->FlushRx();

    if (cfg_.gripper_id > 0)
    {
        try
        {
            bus_->WriteParamF32(cfg_.gripper_id, kParamTorqueLimit,
                                static_cast<float>(cfg_.gripper_torque_limit));
            std::printf("[ARM] 夹爪力矩上限 = %.2f N·m (id=%u)\n", cfg_.gripper_torque_limit,
                        cfg_.gripper_id);
        }
        catch (const std::exception& e)
        {
            std::printf("[ARM] WARN: 写夹爪力矩上限失败: %s\n", e.what());
        }
    }

    // 读初始关节角(参数读, 不发 MIT 帧避免扰动)
    for (int i = 0; i < 6; ++i) last_q_[i] = ReadJoint(i);
    if (cfg_.gripper_id > 0) last_grip_ = GetGripperAngle();
    std::printf("[ARM] 初始化完成 (%s), 当前关节角:", cfg_.can_iface.c_str());
    for (double v : last_q_) std::printf(" %+.3f", v);
    std::printf("  grip=%+.3f rad\n", last_grip_);
}

ElA3Arm::~ElA3Arm()
{
    StopHold();
}

double ElA3Arm::ReadJoint(int i)
{
    const uint8_t id = cfg_.motor_ids[i];
    double raw;
    if (has_command_)
    {
        // 有命令目标后: 重发带增益的 MIT 控制帧并读状态帧(避免零力矩探测导致释放)
        const double raw_cmd = cmd_q_[i] * cfg_.direction[i] + cfg_.homing_offset[i];
        const double ff = cfg_.gravity_torques[i] * cfg_.direction[i];
        bus_->WriteMitFrame(id, raw_cmd, 0, cfg_.kp, cfg_.kd, ff);
        raw = bus_->ReadStatusPosition(id);
    }
    else
    {
        raw = bus_->ReadParamF32(id, kParamMeasuredPos);
    }
    return (raw - cfg_.homing_offset[i]) * cfg_.direction[i];
}

std::vector<double> ElA3Arm::GetJointPositions()
{
    // 先清掉积压的旧状态帧, 保证下面读到的是本次探测触发的新鲜回帧
    bus_->FlushRx();
    for (int i = 0; i < 6; ++i)
    {
        try
        {
            last_q_[i] = ReadJoint(i);
        }
        catch (const std::exception& e)
        {
            std::printf("[ARM] WARN: 读 J%d 失败, 保留上次值 %+.4f (%s)\n", i + 1, last_q_[i],
                        e.what());
        }
    }
    return last_q_;
}

double ElA3Arm::GetGripperAngle()
{
    if (cfg_.gripper_id == 0) return last_grip_;
    try
    {
        bus_->FlushRx();  // 同 GetJointPositions: 只认新鲜回帧
        double raw;
        if (has_grip_cmd_)
        {
            // 与手臂 ReadJoint 相同: 读的时候重发 MIT, 否则推理间隙夹爪会卸力张开.
            const double raw_cmd = cmd_grip_ * cfg_.gripper_direction + cfg_.gripper_homing_offset;
            bus_->WriteMitFrame(cfg_.gripper_id, raw_cmd, 0, cfg_.gripper_kp, cfg_.gripper_kd, 0);
            raw = bus_->ReadStatusPosition(cfg_.gripper_id);
        }
        else
        {
            raw = bus_->ReadParamF32(cfg_.gripper_id, kParamMeasuredPos);
        }
        last_grip_ = (raw - cfg_.gripper_homing_offset) * cfg_.gripper_direction;
    }
    catch (const std::exception& e)
    {
        std::printf("[ARM] WARN: 读夹爪失败, 保留上次值 %+.4f (%s)\n", last_grip_, e.what());
    }
    return last_grip_;
}

void ElA3Arm::CommandGripper(double angle_urdf)
{
    if (cfg_.gripper_id == 0) return;
    const double clipped =
        std::clamp(angle_urdf, cfg_.gripper_angle_min, cfg_.gripper_angle_max);
    const double raw = clipped * cfg_.gripper_direction + cfg_.gripper_homing_offset;
    bus_->WriteMitFrame(cfg_.gripper_id, raw, 0, cfg_.gripper_kp, cfg_.gripper_kd, 0);
    cmd_grip_ = clipped;
    has_grip_cmd_ = true;
    bus_->FlushRx();  // 消费回帧, 防止积压(见 ServoJoint)
}

std::vector<double> ElA3Arm::ClipToLimits(const std::vector<double>& q) const
{
    std::vector<double> out = q;
    for (int i = 0; i < 6; ++i)
    {
        out[i] = std::clamp(out[i], kJointLimits[i][0], kJointLimits[i][1]);
    }
    return out;
}

void ElA3Arm::ServoJoint(const std::vector<double>& q_urdf, double kp, double kd)
{
    for (int i = 0; i < 6; ++i)
    {
        const double raw = q_urdf[i] * cfg_.direction[i] + cfg_.homing_offset[i];
        const double ff = cfg_.gravity_torques[i] * cfg_.direction[i];
        bus_->WriteMitFrame(cfg_.motor_ids[i], raw, 0, kp, kd, ff);
    }
    cmd_q_ = q_urdf;
    has_command_ = true;
    // 每条 MIT 都触发一条状态回帧. 不消费会在内核 rcvbuf 积压成秒级旧数据,
    // 之后 ReadStatusPosition 从队头读到的"实测角"全是过期回声(模型吃旧观测).
    bus_->FlushRx();
}

std::vector<double> ElA3Arm::ManualInitAndHold()
{
    using namespace std::chrono_literals;
    std::printf("[MANUAL_INIT] 零力矩+重力补偿模式, gravity_torques =");
    for (double g : cfg_.gravity_torques) std::printf(" %.2f", g);
    std::printf(" Nm\n[MANUAL_INIT] 手动摆到初始姿态后按 Enter...\n");

    std::atomic<bool> stop{false};
    std::array<bool, 6> seen{};
    std::vector<double> latest = last_q_;
    std::mutex latest_mu;

    std::thread compliance(
        [&]
        {
            const auto dt = std::chrono::duration<double>(1.0 / std::max(1.0, cfg_.manual_init_hz));
            while (!stop.load())
            {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 6; ++i)
                {
                    const uint8_t id = cfg_.motor_ids[i];
                    try
                    {
                        // kp=0(零刚度) + kd 阻尼 + 重力矩前馈(力矩按标定方向换算)
                        bus_->WriteMitFrame(id, 0, 0, 0.0, cfg_.manual_init_kd,
                                            cfg_.gravity_torques[i] * cfg_.direction[i]);
                        const double raw = bus_->ReadStatusPosition(id, 0.2);
                        std::lock_guard<std::mutex> lk(latest_mu);
                        latest[i] = (raw - cfg_.homing_offset[i]) * cfg_.direction[i];
                        seen[i] = true;
                    }
                    catch (const std::exception& e)
                    {
                        std::printf("[MANUAL_INIT] WARN: J%d 柔顺帧失败: %s\n", i + 1, e.what());
                    }
                    std::this_thread::sleep_for(2ms);
                }
                std::this_thread::sleep_until(t0 + dt);
            }
        });

    std::string line;
    std::getline(std::cin, line);  // 等 Enter
    stop.store(true);
    compliance.join();

    std::vector<double> hold_q;
    {
        std::lock_guard<std::mutex> lk(latest_mu);
        hold_q = latest;
    }
    for (int i = 0; i < 6; ++i)
    {
        if (!seen[i])
        {
            throw std::runtime_error("[MANUAL_INIT] J" + std::to_string(i + 1) +
                                     " 未读到位置, 拒绝用默认零位保持");
        }
    }
    last_q_ = hold_q;
    cmd_q_ = hold_q;
    has_command_ = true;

    std::printf("[MANUAL_INIT] 捕获保持位:");
    for (double v : hold_q) std::printf(" %+.4f", v);
    std::printf(" rad\n[MANUAL_INIT] 加载模型/相机期间保持 (kp=%.0f kd=%.0f)\n", cfg_.hold_kp,
                cfg_.hold_kd);

    // 启动 hold 线程(模型/相机加载期间防下垂)
    hold_stop_.store(false);
    hold_thread_ = std::thread(
        [this, hold_q]
        {
            using namespace std::chrono_literals;
            const auto dt = std::chrono::duration<double>(1.0 / std::max(1.0, cfg_.manual_init_hz));
            while (!hold_stop_.load())
            {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 6; ++i)
                {
                    const double raw = hold_q[i] * cfg_.direction[i] + cfg_.homing_offset[i];
                    const double ff = cfg_.gravity_torques[i] * cfg_.direction[i];
                    bus_->WriteMitFrame(cfg_.motor_ids[i], raw, 0, cfg_.hold_kp, cfg_.hold_kd, ff);
                    std::this_thread::sleep_for(2ms);
                }
                if (cfg_.gripper_id > 0)
                {
                    const double raw =
                        last_grip_ * cfg_.gripper_direction + cfg_.gripper_homing_offset;
                    bus_->WriteMitFrame(cfg_.gripper_id, raw, 0, cfg_.gripper_kp, cfg_.gripper_kd,
                                        0);
                }
                // 加载模型/相机可能持续几十秒, hold 帧的回帧必须及时消费,
                // 否则控制回路一开始读到的就是加载期间的陈旧位置.
                bus_->FlushRx();
                std::this_thread::sleep_until(t0 + dt);
            }
        });
    return hold_q;
}

void ElA3Arm::StopHold()
{
    if (!hold_stop_.exchange(true))
    {
        if (hold_thread_.joinable()) hold_thread_.join();
        std::printf("[MANUAL_INIT] 释放启动保持, 切换到模型控制\n");
    }
    else if (hold_thread_.joinable())
    {
        hold_thread_.join();
    }
}

void ElA3Arm::HoldCurrentPosition()
{
    try
    {
        ServoJoint(GetJointPositions(), cfg_.kp, cfg_.kd);
    }
    catch (...)
    {
    }
}

void ElA3Arm::ExecuteJointAction(const std::vector<float>& action, double duration,
                                 const std::atomic<bool>& stop)
{
    const double ctrl_dt = 1.0 / cfg_.control_hz;
    // 与 Python 一致: 从上次命令角插值.用实测角当 q0 会每步拽回当前姿态,到不了目标.
    std::vector<double> q0 = has_command_ ? cmd_q_ : GetJointPositions();

    std::vector<double> q_target(6);
    for (int i = 0; i < 6; ++i) q_target[i] = action[i];
    q_target = ClipToLimits(q_target);
    bool clipped = false;
    for (int i = 0; i < 6; ++i)
    {
        double dq = q_target[i] - q0[i];
        if (std::abs(dq) > cfg_.max_dq_per_action)
        {
            dq = std::clamp(dq, -cfg_.max_dq_per_action, cfg_.max_dq_per_action);
            q_target[i] = q0[i] + dq;
            clipped = true;
        }
    }
    if (clipped)
    {
        std::printf("[SAFE] 动作跳变被限幅 max_dq_per_action=%.3f rad\n", cfg_.max_dq_per_action);
    }

    const double g0 = has_grip_cmd_ ? cmd_grip_ : last_grip_;
    const double g_tgt =
        action.size() >= 7
            ? std::clamp(static_cast<double>(action[6]), cfg_.gripper_angle_min,
                         cfg_.gripper_angle_max)
            : g0;

    const int n = std::max(1, static_cast<int>(std::ceil(duration * cfg_.control_hz)));
    for (int i = 1; i <= n && !stop.load(); ++i)
    {
        const auto t0 = std::chrono::steady_clock::now();
        const double alpha = static_cast<double>(i) / n;
        std::vector<double> next(6);
        for (int j = 0; j < 6; ++j)
        {
            double wp = (1.0 - alpha) * q0[j] + alpha * q_target[j];
            if (has_command_)
            {
                wp = cmd_q_[j] +
                     std::clamp(wp - cmd_q_[j], -cfg_.max_dq_per_tick, cfg_.max_dq_per_tick);
            }
            next[j] = wp;
        }
        ServoJoint(ClipToLimits(next), cfg_.kp, cfg_.kd);
        if (action.size() >= 7)
        {
            // 每个 tick 都发夹爪 MIT. 只在最后一拍发的话, 插值/推理间隙会卸力张开.
            CommandGripper((1.0 - alpha) * g0 + alpha * g_tgt);
        }
        std::this_thread::sleep_until(t0 + std::chrono::duration<double>(ctrl_dt));
    }
}

}  // namespace smolvla
