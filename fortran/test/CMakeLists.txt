# Define test files
set(TEST_FILES
    test_integration
    test_ext
    test_core
    test_sampling
    test_dlr
    test_multidim
    test_basis_functions
    test_error_handling
    test_memory
    test_analytic_gf
)

# Define benchmark files (not run during ctest)
set(BENCHMARK_FILES
    test_timing
)

# Add test executables
foreach(TEST ${TEST_FILES})
    add_executable(${TEST} ${TEST}.f90)
endforeach()

# Add benchmark executables
foreach(BENCH ${BENCHMARK_FILES})
    add_executable(${BENCH} ${BENCH}.f90)
endforeach()

get_target_property(SPARSEIR_FORTRAN_MODDIR sparseir_fortran Fortran_MODULE_DIRECTORY)

# Process each test
foreach(TEST ${TEST_FILES})
    target_include_directories(${TEST} PRIVATE
        ${SPARSEIR_FORTRAN_MODDIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )

    target_link_libraries(${TEST} PRIVATE sparseir_fortran)

    set_target_properties(${TEST} PROPERTIES
        Fortran_MODULE_DIRECTORY ${SPARSEIR_FORTRAN_MODDIR}
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH_USE_LINK_PATH ON
        INSTALL_RPATH "$<TARGET_FILE_DIR:sparseir_fortran>"
    )

    add_test(NAME fortran_${TEST}
        COMMAND ${TEST}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endforeach()

# Process each benchmark (similar setup but no add_test)
foreach(BENCH ${BENCHMARK_FILES})
    target_include_directories(${BENCH} PRIVATE
        ${SPARSEIR_FORTRAN_MODDIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
    )

    target_link_libraries(${BENCH} PRIVATE sparseir_fortran)

    set_target_properties(${BENCH} PROPERTIES
        Fortran_MODULE_DIRECTORY ${SPARSEIR_FORTRAN_MODDIR}
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH_USE_LINK_PATH ON
        INSTALL_RPATH "$<TARGET_FILE_DIR:sparseir_fortran>"
    )
endforeach()