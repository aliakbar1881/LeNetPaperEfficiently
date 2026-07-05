cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(lenet5)

set(Torch_DIR "/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    lenet5
    src/main.cpp
    src/loadpt.cpp
)

target_link_libraries(lenet5 "${TORCH_LIBRARIES}")
set_property(TARGET lenet5 PROPERTY CXX_STANDARD 20)
