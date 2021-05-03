cmake_minimum_required(VERSION 3.5)
project(gpu-slam)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# Compiler flag
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

set(CUDA_NVCC_FLAGS "-arch=sm_61" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)

#### Include
include_directories(include
  ${std_msgs_INCLUDE_DIRS}
  ${gazebo_msgs_INCLUDE_DIRS}
  ${range_msgs_INCLUDE_DIRS}
)

CUDA_ADD_LIBRARY(particleFilter 
                ${LIB_TYPE} 
                src/ParticleFilter_init.cu 
                src/ParticleFilter_filter.cu 
                src/ParticleFilter_kernel.cu
                src/ParticleFilter_if.cu)

add_library(slam
  src/Slam.cpp
)

add_executable(${PROJECT_NAME} src/main.cpp)
ament_target_dependencies(${PROJECT_NAME} rclcpp std_msgs)

target_link_libraries(${PROJECT_NAME}
  slam
  particleFilter
  ${GSL_LIBRARIES}
  ${catkin_LIBRARIES}
)

install(TARGETS
  ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()