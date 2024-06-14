cmake_minimum_required(VERSION 3.10)
project(Kitokarosu VERSION 1.0 DESCRIPTION "Kitokarosu header-only C++ library")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含目录
include_directories(include)

# o3 native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# 示例程序
add_executable(example examples/example.cpp)

# 安装规则
install(DIRECTORY include/Kitokarosu DESTINATION include)