cmake_minimum_required(VERSION 3.15)
project(Kitokarosu VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加子模块（Eigen3）
add_subdirectory(extern/eigen)

# 创建接口库目标
add_library(Kitokarosu INTERFACE)

# 包含目录配置
target_include_directories(Kitokarosu
    INTERFACE
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# 链接依赖项（Eigen3）
target_link_libraries(Kitokarosu
    INTERFACE
        Eigen3::Eigen
)

# 启用示例（可选）
option(BUILD_EXAMPLES "Build examples" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()