cmake_minimum_required(VERSION 3.20)
project(titan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TITAN_ENABLE_CUDA "Build CUDA backend (requires CUDA toolkit)" OFF)
option(TITAN_BUILD_TESTS "Build test suite" ON)

if(TITAN_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

add_subdirectory(runtime/tensor)

if(TITAN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
