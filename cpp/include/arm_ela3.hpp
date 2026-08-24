// EL-A3 机械臂:自实现 RobStride CAN 协议(SocketCAN + MIT 力位混合帧),
// 完整复刻 deploy_vla 的 Python 部署行为(bus.py + deploy_runtime.py):
//   - 上电: disable x2 -> enable(清 fault latch)
//   - 手动初始化: 零力矩(kp=0)+ 各关节重力补偿力矩前馈, 手动摆位后按 Enter
//   - 保持: 捕获摆位关节角, 模型/相机加载期间用 kp=80/kd=5 hold 住
//   - 控制: MIT 帧 kp=80/kd=3(与 Python servo_joint 一致)
//   - 标定: URDF 关节角 = (电机原始角 - homing_offset) * direction
//   - 安全: URDF 限位 clip、单 action 跳变限幅、每 tick 限速
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace smolvla
{

struct ArmConfig
{
    std::string can_iface = "can0";
    std::vector<uint8_t> motor_ids = {1, 2, 3, 4, 5, 6};
    // DEFAULT_CALIBRATION: J1..J6 的方向与零点偏移
    std::array<double, 6> direction = {-1, 1, -1, 1, -1, 1};
    std::array<double, 6> homing_offset = {0, 0, 0, 0, 0, 0};

    // 夹爪(CAN id 7, 与 Python --gripper_id 7 一致). 0 关闭夹爪控制.
    uint8_t gripper_id = 7;
    double gripper_direction = 1.0;
    double gripper_homing_offset = 0.0;
    double gripper_kp = 8.0;   // Python DEFAULT_GRIPPER_KP
    double gripper_kd = 0.4;   // Python DEFAULT_GRIPPER_KD
    double gripper_torque_limit = 0.6;
    double gripper_angle_min = -1.75;
    double gripper_angle_max = 0.10;

    double kp = 100.0, kd = 3.0;           // 控制期 MIT 增益(与 Python --kp/--kd 默认一致)
    double hold_kp = 80.0, hold_kd = 5.0;  // Enter 后加载期间的保持增益

    // 手动初始化(零力矩 + 重力补偿)
    double manual_init_hz = 30.0;
    double manual_init_kd = 0.0;
    std::array<double, 6> gravity_torques = {0, 1.2, -1.2, 0.1, 0, 0};  // Nm, 与 Python DEFAULT_MANUAL_INIT_GRAVITY_TORQUES 一致

    // 安全限幅
    double max_dq_per_action = 0.25;  // rad, 单 action 相对当前的最大跳变
    double max_dq_per_tick = 0.08;    // rad, 每个控制 tick 的最大步长
    double control_hz = 30.0;
};

class RobstrideBus;  // 内部 CAN 协议实现

class ElA3Arm
{
 public:
    explicit ElA3Arm(const ArmConfig& cfg);
    ~ElA3Arm();

    // 零力矩+重力补偿, 等用户手动摆位并按 Enter; 返回捕获的关节角(URDF, rad).
    // 随后自动进入 hold(hold_kp/hold_kd), 直到调用 StopHold().
    std::vector<double> ManualInitAndHold();
    void StopHold();

    // 当前 6 关节角(URDF 坐标, rad)
    std::vector<double> GetJointPositions();
    // 最近一次下发的命令角(URDF). 尚无命令时为空.
    const std::vector<double>& LastCommand() const { return cmd_q_; }

    // 夹爪角(URDF, rad). 读失败返回上次缓存. gripper_id==0 时返回缓存初值.
    // 已有命令时会重发 MIT 保持, 避免推理/插值间隙卸力张开.
    double GetGripperAngle();
    void CommandGripper(double angle_urdf);
    bool HasGripperCommand() const { return has_grip_cmd_; }
    double LastGripperCommand() const { return cmd_grip_; }

    // 执行一个绝对关节角 action(前 6 维手臂 + 第 7 维夹爪角):
    // 从上次命令角插值(与 Python 一致, 不用实测角当 q0),
    // 限位 clip + 单步限幅 + duration 内 control_hz 线性插值;
    // 每个 tick 都下发夹爪(保持 MIT, 不能只在最后一拍发).
    void ExecuteJointAction(const std::vector<float>& action, double duration,
                            const std::atomic<bool>& stop);

    // 命令电机保持当前位置(退出前停住)
    void HoldCurrentPosition();

 private:
    std::vector<double> ClipToLimits(const std::vector<double>& q) const;
    void ServoJoint(const std::vector<double>& q_urdf, double kp, double kd);
    double ReadJoint(int i);  // 读单关节(URDF), 优先状态帧, 无命令时参数读

    ArmConfig cfg_;
    std::unique_ptr<RobstrideBus> bus_;
    std::vector<double> last_q_;  // 最近关节角缓存(URDF)
    std::vector<double> cmd_q_;   // 最近命令目标(URDF)
    bool has_command_ = false;    // 有命令后读状态改用带增益的 MIT 探测帧
    double last_grip_ = -0.55;    // 夹爪实测角缓存(URDF), 数据集均值
    double cmd_grip_ = -0.55;     // 最近夹爪命令角(URDF)
    bool has_grip_cmd_ = false;

    std::atomic<bool> hold_stop_{true};
    std::thread hold_thread_;
};

}  // namespace smolvla
