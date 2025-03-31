cmake_minimum_required(VERSION 3.15)
project(KitokarosuExamples)

# 自动查找所有.cpp源文件（排除CMake自动生成的文件）
file(GLOB SOURCE_FILES CONFIGURE_DEPENDS *.cpp)

# 遍历每个源文件创建独立可执行文件
foreach(source IN LISTS SOURCE_FILES)
    # 获取不带扩展名的文件名作为目标名称
    get_filename_component(target_name ${source} NAME_WE)
    
    # 添加可执行目标
    add_executable(${target_name} ${source})
    
    # 链接主库
    target_link_libraries(${target_name}
        PRIVATE
            Kitokarosu
    )
    
    # 设置统一输出目录
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endforeach()