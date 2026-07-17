add_executable(titan_generate
    generate.cpp
)

target_link_libraries(titan_generate
    PRIVATE
        titan_tensor
        titan_tokenizer
        titan_models
        titan_transformer
        titan_sampling
)
