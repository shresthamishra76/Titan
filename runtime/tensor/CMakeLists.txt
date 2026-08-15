find_package(Threads REQUIRED)

add_library(titan_tensor
    Tensor.cpp
    ThreadPool.cpp
)

target_include_directories(titan_tensor
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(titan_tensor
    PUBLIC Threads::Threads
)
