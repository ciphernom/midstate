cmake_minimum_required(VERSION 3.18)

# Build for the GPUs in this machine unless told otherwise, for example
# -DCMAKE_CUDA_ARCHITECTURES=89 for an RTX 40-series. nvidia-smi says which.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  execute_process(
    COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
    OUTPUT_VARIABLE _walk_caps
    RESULT_VARIABLE _walk_rc
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_walk_rc EQUAL 0 AND _walk_caps MATCHES "^[0-9]+\\.[0-9]+")
    string(REPLACE "." "" _walk_caps "${_walk_caps}")
    string(REGEX REPLACE "[ \t\r\n]+" ";" _walk_caps "${_walk_caps}")
    list(REMOVE_DUPLICATES _walk_caps)
  else()
    set(_walk_caps 52)
    message(STATUS "nvidia-smi didn't say which GPU this is: building for compute 5.2 "
                   "(pass -DCMAKE_CUDA_ARCHITECTURES=<yours> to choose)")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_walk_caps}" CACHE STRING "CUDA architectures to build for")
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

project(midstate_cuda_walk LANGUAGES CXX CUDA)
message(STATUS "Building for CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# The CUDA source stays at C++14, which every toolkit that still builds for
# Maxwell accepts.
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

add_executable(walk-cpu walk_cpu.cpp)
target_link_libraries(walk-cpu PRIVATE Threads::Threads)

add_executable(walk-cuda bench.cpp nvml_power.cpp walk_cuda.cu)
target_link_libraries(walk-cuda PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
