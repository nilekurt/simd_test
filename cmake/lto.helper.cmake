function(configure_lto TARGET_NAME)
    if (SIMD_TEST_LTO)
        cmake_policy(SET CMP0069 NEW)
        set_property(TARGET ${TARGET_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()