// 图像预处理:精确复刻 lerobot modeling_smolvla.py 的
//   resize_with_pad(pad_value=0, 双线性, pad 在左侧和上侧) + [0,1] -> [-1,1]
#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace smolvla
{

// RGB uint8 HWC -> CHW float32 [-1,1],输出尺寸 out_size x out_size.
// 流程: /255 -> 保比例 resize(双线性) -> 左/上 pad 0 -> *2-1(pad 区域变为 -1)
std::vector<float> PreprocessImage(const cv::Mat& rgb_hwc_u8, int out_size);

}  // namespace smolvla
