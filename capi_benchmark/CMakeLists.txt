cmake_minimum_required(VERSION 3.15)
project(SparseIR_CAPI_Benchmark LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Find Rust C API library and header
# Expected layout: _install/include/sparseir/sparseir.h, _install/lib/libsparseir_capi.*
set(SPARSEIR_CAPI_PREFIX "${CMAKE_CURRENT_LIST_DIR}/_install")
set(SPARSEIR_CAPI_INCLUDE_DIR "${SPARSEIR_CAPI_PREFIX}/include")
set(SPARSEIR_CAPI_LIB_DIR "${SPARSEIR_CAPI_PREFIX}/lib")

if(NOT EXISTS "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
    message(FATAL_ERROR "sparseir/sparseir.h not found at ${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
endif()

include_directories("${SPARSEIR_CAPI_INCLUDE_DIR}")

# Find library (supports .so, .dylib, .dll)
find_library(SPARSEIR_CAPI_LIB
    NAMES sparseir_capi
    PATHS "${SPARSEIR_CAPI_LIB_DIR}"
    REQUIRED
    NO_DEFAULT_PATH
)

if(NOT SPARSEIR_CAPI_LIB)
    message(FATAL_ERROR "libsparseir_capi not found in ${SPARSEIR_CAPI_LIB_DIR}")
endif()

message(STATUS "Found sparseir/sparseir.h at: ${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
message(STATUS "Found libsparseir_capi at: ${SPARSEIR_CAPI_LIB}")

# Find Threads
find_package(Threads REQUIRED)

# Get all .c files in current directory and sort them
file(GLOB BENCHMARK_SOURCES "*.c")
list(SORT BENCHMARK_SOURCES)

# Create executable for each benchmark file
foreach(BENCHMARK_SOURCE ${BENCHMARK_SOURCES})
    get_filename_component(BENCHMARK_NAME ${BENCHMARK_SOURCE} NAME_WE)
    add_executable(${BENCHMARK_NAME} ${BENCHMARK_SOURCE})
    target_link_libraries(${BENCHMARK_NAME} PRIVATE 
        ${SPARSEIR_CAPI_LIB}
        Threads::Threads
        m
    )
    list(APPEND BENCHMARK_NAMES ${BENCHMARK_NAME})
endforeach()

# Optional BLAS linking
# Check if ILP64 BLAS is requested (via environment variable or CMake option)
if(DEFINED ENV{SPARSEIR_USE_BLAS_ILP64} AND "$ENV{SPARSEIR_USE_BLAS_ILP64}")
    set(BLA_SIZEOF_INTEGER 8)
    message(STATUS "capi_benchmark: Using ILP64 BLAS (64-bit integers)")
else()
    set(BLA_SIZEOF_INTEGER 4)
    message(STATUS "capi_benchmark: Using LP64 BLAS (32-bit integers, default)")
endif()

find_package(BLAS QUIET)
if(BLAS_FOUND)
    foreach(BENCHMARK_NAME ${BENCHMARK_NAMES})
        target_link_libraries(${BENCHMARK_NAME} PRIVATE ${BLAS_LIBRARIES})
    endforeach()
endif()

# Create a custom target to run all benchmarks
add_custom_target(benchmark
    DEPENDS ${BENCHMARK_NAMES}
    COMMENT "Running all benchmarks"
)

# Add initial message
add_custom_command(TARGET benchmark
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Running all benchmarks..."
    COMMAND ${CMAKE_COMMAND} -E echo "----------------------------------------"
)

# Add commands for each benchmark in sorted order
foreach(BENCHMARK_NAME ${BENCHMARK_NAMES})
    add_custom_command(TARGET benchmark
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Running ${BENCHMARK_NAME}..."
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${BENCHMARK_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo "----------------------------------------"
    )
endforeach()

# Add final message
add_custom_command(TARGET benchmark
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "All benchmarks completed!"
)

