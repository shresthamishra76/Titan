add_library(titan_sampling
    Sampler.cpp
)

target_include_directories(titan_sampling
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(titan_sampling
    PUBLIC titan_tensor
)
