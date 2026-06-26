// SmolVLA C++ 推理运行时(路线②:分块 ONNX 引擎 + C++ 编排)
//
// 编排逻辑(与 Python 的 run_onnx_pipeline.py 完全一致,已验证 max|diff|=3.6e-7):
//   1. vision 引擎     : 每路相机图像 -> image_embeds
//   2. prefix_assembler: image_embeds + lang_tokens + state -> prefix_embs/masks/pos
//   3. prefill 引擎     : prefix_embs -> per-layer KV cache (kv_keys/kv_values)
//   4. denoise 循环     : x_t=noise; 重复 10 步 { v_t = denoise(...); x_t += dt*v_t }
//   5. 取前 7 维 -> 动作
//
// 运行时后端:ONNX Runtime C++ API + TensorRT Execution Provider(FP16).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace smolvla {

// ===== 模型结构常量(取自 checkpoint config,见 README 的"接口"表)=====
struct ModelDims {
  int num_images = 3;      // 2 路真实相机 + 1 路 empty_camera
  int img_size = 512;      // resize_imgs_with_padding
  int image_tokens = 64;   // connector 输出的每图 token 数
  int hidden = 960;        // text/vlm hidden size
  int lang_len = 48;       // tokenizer_max_length
  int state_dim = 32;      // max_state_dim
  int num_layers = 16;     // num_vlm_layers
  int kv_heads = 5;        // num_key_value_heads
  int head_dim = 64;       // head_dim
  int chunk = 50;          // chunk_size
  int action_dim = 32;     // max_action_dim(含 padding)
  int real_action_dim = 7; // 真实输出动作维度
  int num_steps = 10;      // flow-matching 去噪步数

  int prefix_len() const { return num_images * image_tokens + lang_len + 1; }
};

struct RuntimeConfig {
  std::string artifacts_dir;   // 含 4 个 .onnx 的目录
  bool use_tensorrt = true;    // true: TensorRT EP(FP16); false: CUDA EP
  bool fp16 = true;
  std::string trt_cache_dir;   // TensorRT engine 缓存目录(加速二次启动)
  // 动作反归一化: 真实动作 = norm*std+mean. 留空则不反归一化(输出归一化值).
  // 默认读 artifacts_dir/action_norm_stats.bin(由 export_action_stats.py 生成).
  bool unnormalize_action = true;
  std::string action_stats_path;
  ModelDims dims;
};

// 一次观测的输入(已完成图像预处理与 tokenize)
struct Observation {
  // 每路图像:num_images 个,单图 [3, img_size, img_size],已归一化到 [-1,1]
  std::vector<std::vector<float>> images;
  std::vector<bool> image_valid;     // num_images,empty 相机为 false
  std::vector<int64_t> lang_tokens;  // [lang_len]
  std::vector<bool> lang_mask;       // [lang_len]
  std::vector<float> state;          // [state_dim](已按训练统计归一化)
};

class SmolVLARuntime {
 public:
  explicit SmolVLARuntime(const RuntimeConfig& cfg);

  // 返回 [chunk, real_action_dim] 的动作序列(行优先展开).
  // noise 可选:传入固定噪声以复现/对比;为空则内部随机采样.
  std::vector<float> SelectActionChunk(const Observation& obs,
                                       const std::vector<float>* noise = nullptr);

 private:
  Ort::Value RunVision(const float* image_data);
  void RunAssembler(const std::vector<Ort::Value>& image_embeds, const Observation& obs);
  void RunPrefill();
  std::vector<float> RunDenoiseStep(const std::vector<float>& x_t, float t);

  RuntimeConfig cfg_;
  ModelDims d_;

  Ort::Env env_;
  Ort::MemoryInfo mem_info_;
  std::unique_ptr<Ort::Session> vision_;
  std::unique_ptr<Ort::Session> assembler_;
  std::unique_ptr<Ort::Session> prefill_;
  std::unique_ptr<Ort::Session> denoise_;

  // prefill / assembler 的中间产物(在一次 SelectActionChunk 内复用)
  // 注意:bool 掩码用 uint8_t 连续缓冲(ONNX Runtime 的 BOOL 张量按 1 字节存储,
  //       不能用位压缩的 std::vector<bool>).
  std::vector<float> prefix_embs_;
  std::vector<uint8_t> prefix_pad_;
  std::vector<uint8_t> attn_2d_;
  std::vector<int64_t> position_ids_;
  std::vector<float> kv_keys_;
  std::vector<float> kv_values_;
  int prefix_len_ = 0;

  // 动作反归一化统计量(各 real_action_dim 维), 为空表示不反归一化
  std::vector<float> action_mean_;
  std::vector<float> action_std_;
};

}  // namespace smolvla
