cmake_minimum_required(VERSION 3.0)
project(my_slam_v2)

set(CMAKE_CXX_STANDARD 17)

# 1. 寻找依赖库
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(DBoW3 REQUIRED)

set(OpenGL_GL_PREFERENCE "GLVND") # 显式告诉 CMake 使用新版 OpenGL 查找策略，消除警告
find_package(g2o REQUIRED) # 确保你安装了 g2o

# 2. 包含头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${g2o_INCLUDE_DIRS}
    ${DBoW3_INCLUDE_DIRS}
)

# 3. 编译库文件 (把核心逻辑编译成一个库 my_slam_lib)
add_library(my_slam_lib
    src/Frame.cpp
    src/Frontend.cpp
    src/Backend.cpp
    src/Visualizer.cpp
    src/LoopClosing.cpp
)

# 链接库的依赖
target_link_libraries(my_slam_lib
    ${OpenCV_LIBS}
    Eigen3::Eigen
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_types_slam3d
    ${DBoW3_LIBS}
)

# 4. 编译可执行文件 (main)
add_executable(run_slam src/main.cpp)
target_link_libraries(run_slam my_slam_lib)
