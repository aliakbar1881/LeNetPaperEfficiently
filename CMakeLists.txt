cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(lenet5)

set(Torch_DIR "/home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)
find_package(simdjson REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(
    lenet5
    src/main.cpp
    src/loadpt.cpp
    src/jsonModel.cpp
    src/loadjson.cpp
)

target_include_directories(lenet5 PRIVATE internal /home/aliakbar/Desktop/Projects/LeNetPaperEfficiently/internal)

target_link_libraries(lenet5 "${TORCH_LIBRARIES}" simdjson::simdjson )
set_property(TARGET lenet5 PROPERTY CXX_STANDARD 20)
