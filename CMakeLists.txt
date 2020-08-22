cmake_minimum_required(VERSION 3.14)

project(simd_test VERSION 0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

include(FetchContent)
include(lto.helper)
include(test.helper)
include(with.helper)
include(tidy.helper)

option(SIMD_TEST_CLANG_TIDY "Build with clang-tidy" ON)
option(SIMD_TEST_TESTS "Build and add tests" ON)
option(SIMD_TEST_COVERAGE "Build with coverage support" OFF)
option(LTO "Build with Link-Time Optimization" OFF)

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Debug")
else()
    set(SIMD_TEST_CLANG_TIDY OFF)
    # Disable DeathTests
    set(ENV{EXTRA_TEST_ARGS} --gtest_filter=-*DeathTest*)
endif()

WithMsg(tests ${SIMD_TEST_TESTS})
WithMsg(clang-tidy ${SIMD_TEST_CLANG_TIDY})
WithMsg(coverage ${SIMD_TEST_COVERAGE})

if (SIMD_TEST_CLANG_TIDY)
    include(clang-tidy)
endif()

set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --child-silent-after-fork=yes --leak-check=full --error-exitcode=5")

if (SIMD_TEST_TESTS)
    include(CTest)
    enable_testing()
endif()

set(SIMD_TEST_EXTRA_CFLAGS -march=native -O3 -ffast-math -fvisibility=hidden)

set_property(GLOBAL PROPERTY SIMD_TEST_RELEASE_EXTRA_CFLAGS ${SIMD_TEST_EXTRA_CFLAGS})

add_library(simd_test_copts_common INTERFACE)
target_compile_options(simd_test_copts_common INTERFACE
    -pedantic
    -W
    -Wall
    -Wextra
    -Wcast-align
    -Wcast-qual
    -Wmissing-declarations
    -Wwrite-strings
    -Wundef
    -Wswitch-enum
    -Wshadow
    -Werror
    -Wstrict-aliasing=2
    $<$<C_COMPILER_ID:Clang>:-Wshadow-all>
    $<$<CXX_COMPILER_ID:Clang>:-Wshadow-all>
    $<$<STREQUAL:$<CONFIG>,Debug>:-ggdb>
    $<$<STREQUAL:$<CONFIG>,Release>:${SIMD_TEST_EXTRA_CFLAGS}>
    $<$<BOOL:${SIMD_TEST_COVERAGE}>:-g -fprofile-arcs -ftest-coverage>)
target_link_libraries(simd_test_copts_common
    INTERFACE
    $<$<BOOL:${SIMD_TEST_COVERAGE}>:
        $<IF:$<STREQUAL:${CMAKE_CXX_COMPILER_ID},GNU>,
            gcov, # GCC
            --coverage # Clang
        >
    >)


add_subdirectory(3rdparty)

add_subdirectory(simd_test_main)
