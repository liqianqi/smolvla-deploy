#include "image_preproc.hpp"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace smolvla
{

std::vector<float> PreprocessImage(const cv::Mat& rgb_hwc_u8, int out_size)
{
    if (rgb_hwc_u8.type() != CV_8UC3) throw std::runtime_error("PreprocessImage 需要 CV_8UC3 RGB");
    const int cur_h = rgb_hwc_u8.rows, cur_w = rgb_hwc_u8.cols;

    // 与 lerobot resize_with_pad 一致:ratio 取 max,新尺寸用 int() 截断
    const double ratio =
        std::max(static_cast<double>(cur_w) / out_size, static_cast<double>(cur_h) / out_size);
    const int resized_h = static_cast<int>(cur_h / ratio);
    const int resized_w = static_cast<int>(cur_w / ratio);

    // torch F.interpolate(bilinear, align_corners=False) 与 cv::INTER_LINEAR 同为半像素对齐
    cv::Mat resized;
    cv::resize(rgb_hwc_u8, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR);

    const int pad_h = std::max(0, out_size - resized_h);  // pad 在上侧
    const int pad_w = std::max(0, out_size - resized_w);  // pad 在左侧

    // CHW float:pad 区域 = 0*2-1 = -1,图像区域 = px/255*2-1
    std::vector<float> chw(static_cast<size_t>(3) * out_size * out_size, -1.0f);
    const float scale = 2.0f / 255.0f;
    for (int y = 0; y < resized_h; ++y)
    {
        const uint8_t* row = resized.ptr<uint8_t>(y);
        const int oy = y + pad_h;
        for (int x = 0; x < resized_w; ++x)
        {
            const int ox = x + pad_w;
            const size_t base = static_cast<size_t>(oy) * out_size + ox;
            chw[0 * static_cast<size_t>(out_size) * out_size + base] =
                row[3 * x + 0] * scale - 1.0f;
            chw[1 * static_cast<size_t>(out_size) * out_size + base] =
                row[3 * x + 1] * scale - 1.0f;
            chw[2 * static_cast<size_t>(out_size) * out_size + base] =
                row[3 * x + 2] * scale - 1.0f;
        }
    }
    return chw;
}

}  // namespace smolvla
