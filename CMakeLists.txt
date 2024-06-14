cmake_minimum_required(VERSION 3.10)
project(Kitokarosu VERSION 1.0 DESCRIPTION "Kitokarosu header-only C++ library")

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 包含目录
include_directories(include)

# o3 native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(example examples/example.cpp)
endif()

add_library(Kitokarosu INTERFACE)
target_include_directories(Kitokarosu INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)