cmake_minimum_required(VERSION 3.18)
project(smolvla_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json 供 clangd/IDE 解析头文件

# ONNX Runtime 根目录(含 include/ 与 lib/).
# 默认指向本仓库 third_party/ 下解压好的预编译包(onnxruntime-linux-x64-gpu_cuda12).
# Jetson Orin:改用 NVIDIA 提供的,带 TensorRT EP 的 onnxruntime-gpu.
#   cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime)
endif()
if(NOT EXISTS ${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h)
  message(FATAL_ERROR "找不到 ONNX Runtime 头文件, 请用 -DONNXRUNTIME_ROOT=/path/to/onnxruntime 指定")
endif()

add_executable(smolvla_demo
  src/main.cpp
  src/smolvla_runtime.cpp
)

target_include_directories(smolvla_demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXRUNTIME_ROOT}/include
)

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
target_link_libraries(smolvla_demo PRIVATE ${ONNXRUNTIME_LIB})

# 运行时让加载器找到 onnxruntime 动态库
set_target_properties(smolvla_demo PROPERTIES
  BUILD_RPATH ${ONNXRUNTIME_ROOT}/lib
  INSTALL_RPATH ${ONNXRUNTIME_ROOT}/lib
)
