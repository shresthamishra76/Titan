add_library(titan_models
    GGUF.cpp
)

target_include_directories(titan_models
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(titan_models
    PUBLIC titan_tensor
)
