add_library(titan_tokenizer
    Tokenizer.cpp
)

target_include_directories(titan_tokenizer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)
