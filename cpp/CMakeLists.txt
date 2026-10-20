cmake_minimum_required(VERSION 3.18)
project(smolvla_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json 供 clangd/IDE 解析头文件

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "构建类型" FORCE)
endif()

option(SMOLVLA_WITH_DEPLOY "构建实机部署程序 smolvla_deploy*(OpenCV+librealsense2)" ON)

# 目标平台探测:Jetson Orin 等 aarch64 用 NVIDIA 提供的 onnxruntime-gpu
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(SMOLVLA_TARGET_JETSON ON)
  message(STATUS "检测到 aarch64:按 Jetson Orin(JetPack)配置")
else()
  set(SMOLVLA_TARGET_JETSON OFF)
  message(STATUS "检测到 x86_64:按本机 GPU(onnxruntime-linux-x64-gpu)配置")
endif()

# ============================================================================
# 1) ONNX Runtime(必需)—— C++ API + TensorRT/CUDA Execution Provider
#    x64  : 官方预编译 onnxruntime-linux-x64-gpu_cuda12(自带 TRT/CUDA EP)
#    Orin : NVIDIA 提供的 onnxruntime-gpu(匹配 JetPack 的 CUDA/TensorRT)
#    默认指向本仓库 third_party/onnxruntime;也可用 -DONNXRUNTIME_ROOT=... 覆盖.
# ============================================================================
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime)
endif()
if(NOT EXISTS ${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h)
  message(FATAL_ERROR
    "找不到 ONNX Runtime 头文件(${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h).\n"
    "请下载对应平台的预编译包解压到 third_party/onnxruntime,"
    "或用 -DONNXRUNTIME_ROOT=/path/to/onnxruntime 指定.详见 README.")
endif()

find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)

add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION ${ONNXRUNTIME_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include)

# ONNX Runtime 的 CUDA/TensorRT EP 运行时还依赖 cuDNN 9(x64 官方包不自带).
# 默认指向本仓库 third_party/cudnn/lib(见 README 的 cuDNN 获取说明);
# 也可用 -DCUDNN_LIB_DIR=... 指向系统已装的 cuDNN(如 /usr/local/cudnn/lib).
# CUDA runtime 与 TensorRT 库沿用系统环境(本机 .zshrc 已导出 /usr/local/cuda 与 TensorRT 到 LD_LIBRARY_PATH).
if(NOT DEFINED CUDNN_LIB_DIR)
  set(CUDNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cudnn/lib)
endif()
set(SMOLVLA_RPATH_DIRS ${ONNXRUNTIME_ROOT}/lib)
if(EXISTS ${CUDNN_LIB_DIR}/libcudnn.so.9)
  list(APPEND SMOLVLA_RPATH_DIRS ${CUDNN_LIB_DIR})
  message(STATUS "cuDNN: ${CUDNN_LIB_DIR}")
else()
  message(WARNING
    "未在 ${CUDNN_LIB_DIR} 找到 libcudnn.so.9;GPU(CUDA/TensorRT EP)运行时会报"
    " 'libcudnn.so.9: cannot open shared object file'.\n"
    "  请按 README 准备 cuDNN 9,或用 -DCUDNN_LIB_DIR=/path/to/cudnn/lib 指定,"
    "或把其目录加入 LD_LIBRARY_PATH.")
endif()

# 4 引擎优化版运行时(IO Binding 常驻显存 + 视觉合批 + 去噪 CUDA Graph)需要 CUDA runtime.
find_package(CUDAToolkit REQUIRED)

# 下面所有可执行文件都要让加载器找到 onnxruntime(及其 TRT/CUDA EP)与 cuDNN 动态库.
# 用旧式 DT_RPATH(--disable-new-dtags)而非默认的 DT_RUNPATH:
# EP(如 libonnxruntime_providers_tensorrt.so)是被 ORT 用 dlopen 加载的,
# 其依赖(libcudnn.so.9)的解析只走可传递的 DT_RPATH,不走 DT_RUNPATH.

# ============================================================================
# 2) 基准程序 smolvla_bench: 原版 vs 优化版运行时的耗时与数值对比
# ============================================================================
add_executable(smolvla_bench
  src/bench_main.cpp
  src/smolvla_runtime.cpp
  src/smolvla_fast_runtime.cpp
)
target_include_directories(smolvla_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(smolvla_bench PRIVATE onnxruntime::onnxruntime CUDA::cudart)
set_target_properties(smolvla_bench PROPERTIES
  BUILD_RPATH "${SMOLVLA_RPATH_DIRS}"
  INSTALL_RPATH "${SMOLVLA_RPATH_DIRS}")
target_link_options(smolvla_bench PRIVATE "-Wl,--disable-new-dtags")

# ============================================================================
# 3) 实机部署 smolvla_deploy —— RealSense + USB 相机 + RobStride 电机 + ONNX 推理
#    依赖:
#      - OpenCV(系统 apt 版即可, 图像预处理 + USB 相机)
#      - librealsense2(本机由 ROS Humble 提供: ros-humble-librealsense2)
#    电机控制为自实现的 RobStride SocketCAN 协议(arm_ela3.cpp), 无外部驱动依赖.
# ============================================================================
if(SMOLVLA_WITH_DEPLOY)
  find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

  # librealsense2: 优先系统安装, 其次 ROS Humble 自带的
  find_package(realsense2 REQUIRED
    PATHS /opt/ros/humble/lib/x86_64-linux-gnu/cmake/realsense2
          /opt/ros/humble/lib/aarch64-linux-gnu/cmake/realsense2)
  get_target_property(REALSENSE2_SO realsense2::realsense2 LOCATION)
  get_filename_component(REALSENSE2_LIB_DIR ${REALSENSE2_SO} DIRECTORY)

  find_package(Threads REQUIRED)

  add_executable(smolvla_deploy
    src/deploy_main.cpp
    src/smolvla_runtime.cpp
    src/deploy_camera.cpp
    src/image_preproc.cpp
    src/arm_ela3.cpp
  )
  target_include_directories(smolvla_deploy PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(smolvla_deploy PRIVATE
    onnxruntime::onnxruntime
    realsense2::realsense2
    ${OpenCV_LIBS}
    Threads::Threads
  )
  set(SMOLVLA_DEPLOY_RPATH ${SMOLVLA_RPATH_DIRS} ${REALSENSE2_LIB_DIR})
  set_target_properties(smolvla_deploy PROPERTIES
    BUILD_RPATH "${SMOLVLA_DEPLOY_RPATH}"
    INSTALL_RPATH "${SMOLVLA_DEPLOY_RPATH}")
  target_link_options(smolvla_deploy PRIVATE "-Wl,--disable-new-dtags")
  message(STATUS "smolvla_deploy: OpenCV=${OpenCV_VERSION} realsense2=${realsense2_VERSION}")

  # 原始效果版: 模型输出原样执行, 不含任何偏置/滤波/错开(对照 smolvla_deploy 用)
  add_executable(smolvla_deploy_raw
    src/deploy_raw_main.cpp
    src/smolvla_runtime.cpp
    src/smolvla_fast_runtime.cpp
    src/deploy_camera.cpp
    src/image_preproc.cpp
    src/arm_ela3.cpp
  )
  target_include_directories(smolvla_deploy_raw PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(smolvla_deploy_raw PRIVATE
    onnxruntime::onnxruntime
    realsense2::realsense2
    ${OpenCV_LIBS}
    Threads::Threads
    CUDA::cudart
  )
  set_target_properties(smolvla_deploy_raw PROPERTIES
    BUILD_RPATH "${SMOLVLA_DEPLOY_RPATH}"
    INSTALL_RPATH "${SMOLVLA_DEPLOY_RPATH}")
  target_link_options(smolvla_deploy_raw PRIVATE "-Wl,--disable-new-dtags")

  # 单图 ONNX 版: 与 smolvla_deploy_raw 相同的执行逻辑, 推理走 smolvla_full.onnx(tools/export_full_onnx.py)
  add_executable(smolvla_deploy_full
    src/deploy_full_main.cpp
    src/smolvla_full_runtime.cpp
    src/deploy_camera.cpp
    src/image_preproc.cpp
    src/arm_ela3.cpp
  )
  target_include_directories(smolvla_deploy_full PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
  )
  target_link_libraries(smolvla_deploy_full PRIVATE
    onnxruntime::onnxruntime
    realsense2::realsense2
    ${OpenCV_LIBS}
    Threads::Threads
  )
  set_target_properties(smolvla_deploy_full PROPERTIES
    BUILD_RPATH "${SMOLVLA_DEPLOY_RPATH}"
    INSTALL_RPATH "${SMOLVLA_DEPLOY_RPATH}")
  target_link_options(smolvla_deploy_full PRIVATE "-Wl,--disable-new-dtags")
endif()

# ============================================================================
# 配置摘要
# ============================================================================
message(STATUS "================ smolvla_cpp 配置摘要 ================")
message(STATUS "  目标平台        : ${CMAKE_SYSTEM_PROCESSOR} (Jetson=${SMOLVLA_TARGET_JETSON})")
message(STATUS "  ONNX Runtime    : ${ONNXRUNTIME_ROOT}")
message(STATUS "  cuDNN lib       : ${CUDNN_LIB_DIR}")
message(STATUS "  CUDA toolkit    : ${CUDAToolkit_VERSION}")
message(STATUS "  实机部署 deploy : ${SMOLVLA_WITH_DEPLOY}")
message(STATUS "======================================================")
