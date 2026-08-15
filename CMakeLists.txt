cmake_minimum_required(VERSION 3.20)
project(titan LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(TITAN_BUILD_TESTS "Build test suite" ON)

add_subdirectory(runtime/tensor)
add_subdirectory(runtime/tokenizer)
add_subdirectory(runtime/models)
add_subdirectory(runtime/transformer)
add_subdirectory(runtime/sampling)

add_subdirectory(examples)

if(TITAN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
