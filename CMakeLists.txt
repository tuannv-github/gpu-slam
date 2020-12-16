cmake_minimum_required(VERSION 2.8.3)
project(gpu-slam)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  tf
  message_filters
)

find_package(CUDA REQUIRED)
find_package(Boost REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_61" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
set(LIB_TYPE STATIC)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
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

catkin_package()

add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME}
  slam
  particleFilter
  ${GSL_LIBRARIES}
  ${catkin_LIBRARIES}
)

install(PROGRAMS
  scripts/listener.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
