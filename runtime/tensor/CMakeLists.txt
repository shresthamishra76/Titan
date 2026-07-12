add_library(titan_tensor
    Tensor.cpp
)

target_include_directories(titan_tensor
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
