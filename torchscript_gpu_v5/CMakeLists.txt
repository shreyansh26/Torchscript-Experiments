cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(torchscript_gpu_v5 torchscript_gpu_v5.cpp)
target_link_libraries(torchscript_gpu_v5 "${TORCH_LIBRARIES}")
set_property(TARGET torchscript_gpu_v5 PROPERTY CXX_STANDARD 14)