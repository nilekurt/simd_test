function(configure_tidy TARGET_NAME)
    if (SIMD_TEST_CLANG_TIDY)
        clang_tidy_check(${TARGET_NAME})
    endif()
endfunction()
