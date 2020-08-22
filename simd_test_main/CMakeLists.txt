project(simd_test_main VERSION 0.1 LANGUAGES CXX)

file(GLOB_RECURSE INC "include/*.hh")
file(GLOB_RECURSE SRC "src/*.cc")

add_executable(simd_test_main)
target_sources(simd_test_main PRIVATE ${INC} ${SRC})
target_link_libraries(simd_test_main PRIVATE simd_test_copts_common)

configure_lto(simd_test_main)
configure_tidy(simd_test_main)

