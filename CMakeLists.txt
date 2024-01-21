cmake_minimum_required(VERSION 3.6)
project(telekinesis)

cmake_policy(SET CMP0146 OLD)
cmake_policy(SET CMP0148 OLD)

find_package(CUDA REQUIRED)
find_package(PythonInterp 3.10 REQUIRED)
find_package(PythonLibs 3.10 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    ./include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

cuda_add_library(telekinesis SHARED
  src/telekinesis.cu)

target_link_libraries(telekinesis
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(telekinesis PROPERTIES PREFIX "")
