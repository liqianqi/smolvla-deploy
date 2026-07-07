// 相机采集:RealSense(主视角)+ USB UVC(腕部),行为对齐 Python 部署
// (deploy_runtime.py 的 RealSenseD430Camera / USBCamera):
//   - 两路均输出 640x480 RGB uint8(HWC),不做 crop/resize
//   - RealSense 只开彩色流(rs2 RGB8);USB 走 V4L2+MJPG,BGR->RGB,buffer=1
//   - 打开后各丢 5 帧等自动曝光稳定
#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>

namespace smolvla
{

class RealSenseCamera
{
 public:
    // serial 为空则接第一台设备
    RealSenseCamera(int width = 640, int height = 480, int fps = 30,
                    const std::string& serial = "");
    ~RealSenseCamera();

    // 返回 RGB uint8 HWC(CV_8UC3)
    cv::Mat Read();

 private:
    struct Impl;  // 隐藏 librealsense 头依赖
    std::unique_ptr<Impl> impl_;
};

class UsbCamera
{
 public:
    UsbCamera(int device = 6, int width = 640, int height = 480, int fps = 30);
    ~UsbCamera();

    // 返回 RGB uint8 HWC(CV_8UC3)
    cv::Mat Read();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace smolvla
