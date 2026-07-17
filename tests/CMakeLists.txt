include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(titan_tests
    tensor_test.cpp
    tokenizer_test.cpp
    gguf_test.cpp
    transformer_test.cpp
)

target_link_libraries(titan_tests
    PRIVATE titan_tensor titan_tokenizer titan_models titan_transformer GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(titan_tests)
