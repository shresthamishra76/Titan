add_library(titan_transformer
    Model.cpp
)

target_include_directories(titan_transformer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(titan_transformer
    PUBLIC titan_tensor titan_models
)
