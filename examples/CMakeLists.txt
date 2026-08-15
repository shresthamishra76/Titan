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

add_executable(titan_benchmark
    benchmark.cpp
)

target_link_libraries(titan_benchmark
    PRIVATE titan_tensor
)
