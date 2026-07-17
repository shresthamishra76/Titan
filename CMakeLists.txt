cmake_minimum_required(VERSION 3.20)
project(titan LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(TITAN_BUILD_TESTS "Build test suite" ON)

add_subdirectory(runtime/tensor)
add_subdirectory(runtime/tokenizer)
add_subdirectory(runtime/models)

if(TITAN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
