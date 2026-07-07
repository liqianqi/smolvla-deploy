// SmolVLA C++ 推理 demo 入口
//
// 演示如何用 SmolVLARuntime 跑一次动作推理.真实部署时需替换两处占位:
//   [TODO-CAMERA]    用真实相机帧填充 obs.images(已 resize 到 512x512 且归一化到 [-1,1])
//   [TODO-TOKENIZER] 用语言指令的 tokenize 结果填充 obs.lang_tokens / lang_mask
//                    (SmolVLM2 tokenizer,可用 HF tokenizers 的 C++/Rust 绑定,或离线预 tokenize)
#include <cstdio>
#include <vector>

#include "smolvla_runtime.hpp"

int main(int argc, char** argv)
{
    smolvla::RuntimeConfig cfg;
    cfg.artifacts_dir = argc > 1 ? argv[1] : "../artifacts";
    cfg.use_tensorrt = true;
    cfg.fp16 = true;
    cfg.trt_cache_dir = cfg.artifacts_dir + "/.trt_cache";

    smolvla::SmolVLARuntime rt(cfg);
    const auto& d = cfg.dims;

    // ---- 构造一帧观测(此处用占位数据;真实部署见上方 TODO)----
    smolvla::Observation obs;
    obs.images.resize(d.num_images);
    obs.image_valid.assign(d.num_images, true);
    obs.image_valid[d.num_images - 1] = false;  // 最后一路为 empty_camera
    for (int i = 0; i < d.num_images; ++i)
    {
        obs.images[i].assign(static_cast<size_t>(3) * d.img_size * d.img_size, 0.0f);
        // [TODO-CAMERA] 填入真实图像像素
    }
    obs.lang_tokens.assign(d.lang_len, 0);  // [TODO-TOKENIZER]
    obs.lang_mask.assign(d.lang_len, false);
    obs.lang_mask[0] = true;  // 占位:至少一个有效 token
    obs.state.assign(d.state_dim, 0.0f);

    std::vector<float> actions = rt.SelectActionChunk(obs);

    std::printf("action_chunk: [%d steps x %d dims]\n", d.chunk, d.real_action_dim);
    std::printf("first action:");
    for (int a = 0; a < d.real_action_dim; ++a) std::printf(" %.4f", actions[a]);
    std::printf("\n");
    return 0;
}
