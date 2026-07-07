#include "deploy_camera.hpp"

#include <librealsense2/rs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <stdexcept>

namespace smolvla
{

// ============================ RealSense ============================
struct RealSenseCamera::Impl
{
    rs2::pipeline pipeline;
};

RealSenseCamera::RealSenseCamera(int width, int height, int fps, const std::string& serial)
    : impl_(new Impl)
{
    rs2::config config;
    if (!serial.empty()) config.enable_device(serial);
    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGB8, fps);
    impl_->pipeline.start(config);
    // 丢掉前几帧自动曝光稳定期(与 Python 一致)
    for (int i = 0; i < 5; ++i) impl_->pipeline.wait_for_frames();
}

RealSenseCamera::~RealSenseCamera()
{
    try
    {
        impl_->pipeline.stop();
    }
    catch (...)
    {
    }
}

cv::Mat RealSenseCamera::Read()
{
    rs2::frameset frames = impl_->pipeline.wait_for_frames();
    rs2::video_frame color = frames.get_color_frame();
    if (!color) throw std::runtime_error("RealSense 丢帧: 未取到 color frame");
    // rs2 RGB8 -> 拷贝为独立的 cv::Mat(frameset 生命周期结束后数据失效)
    cv::Mat rgb(color.get_height(), color.get_width(), CV_8UC3, const_cast<void*>(color.get_data()),
                color.get_stride_in_bytes());
    return rgb.clone();
}

// ============================ USB UVC ============================
struct UsbCamera::Impl
{
    cv::VideoCapture cap;
};

UsbCamera::UsbCamera(int device, int width, int height, int fps) : impl_(new Impl)
{
    // 先用 V4L2 后端,不行回落默认(与 Python 一致)
    impl_->cap.open(device, cv::CAP_V4L2);
    if (!impl_->cap.isOpened()) impl_->cap.open(device);
    if (!impl_->cap.isOpened())
    {
        throw std::runtime_error("USB 相机打不开: device=" + std::to_string(device) +
                                 ". 用 `v4l2-ctl --list-devices` 确认设备号");
    }
    // FOURCC 必须在设 size/fps 之前(V4L2 quirk);MJPG 避免 USB2.0 带宽瓶颈
    impl_->cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    impl_->cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    impl_->cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    impl_->cap.set(cv::CAP_PROP_FPS, fps);
    impl_->cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // 压到 1 帧,避免拿到老帧

    cv::Mat tmp;
    for (int i = 0; i < 5; ++i) impl_->cap.read(tmp);  // 曝光预热
}

UsbCamera::~UsbCamera()
{
    impl_->cap.release();
}

cv::Mat UsbCamera::Read()
{
    cv::Mat bgr;
    if (!impl_->cap.read(bgr) || bgr.empty()) throw std::runtime_error("USB 相机丢帧");
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

}  // namespace smolvla
