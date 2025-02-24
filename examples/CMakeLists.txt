cmake_minimum_required(VERSION 3.15)
project(KitokarosuExamples)

# 添加示例可执行文件
add_executable(basic_usage basic_usage.cpp)

# 链接主库
target_link_libraries(basic_usage
    PRIVATE
        Kitokarosu
)

# 可选：设置输出目录
set_target_properties(basic_usage PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)