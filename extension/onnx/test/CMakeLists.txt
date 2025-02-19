# 最低要求的 CMake 版本
cmake_minimum_required(VERSION 3.10)

# 项目名称
project(MNIST_Inference)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)

# 查找 OpenCV 库
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(OpenCV REQUIRED)

include_directories(../onnxruntime/include)
find_library(ONNXRUNTIME_LIB ../onnxruntime PATHS ../onnxruntime/lib)
if (NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "onnxruntime library not found!")
endif()
# 添加源文件
add_executable(mnist_inference mnist_inference.cpp)

# 链接 OpenCV 和 ONNX Runtime 库
target_link_libraries(mnist_inference PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIB})

