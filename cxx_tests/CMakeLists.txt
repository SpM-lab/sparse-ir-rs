cmake_minimum_required(VERSION 3.15)
project(SparseIR_CAPI_Tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Find Eigen3
find_package(Eigen3 3.4.1 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found in system, fetching from GitLab...")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.1
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(Eigen3)
    message(STATUS "Eigen3 fetch completed")
endif()

# Find Threads
find_package(Threads REQUIRED)

message(STATUS "SPARSEIR_USE_BLAS: ${SPARSEIR_USE_BLAS}")

# Find Catch2 in system first
find_package(Catch2 3.4.0 QUIET)
if(NOT Catch2_FOUND)
    message(STATUS "Catch2 not found in system, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    # Set verbose output for FetchContent to see progress
    set(FETCHCONTENT_QUIET OFF)
    # Check if Catch2 was already populated (to avoid re-downloading)
    FetchContent_GetProperties(Catch2)
    if(NOT Catch2_POPULATED)
        message(STATUS "Catch2 not yet populated, fetching from GitHub...")
        FetchContent_MakeAvailable(Catch2)
        message(STATUS "Catch2 fetch completed")
    else()
        message(STATUS "Catch2 already populated at: ${catch2_SOURCE_DIR}")
    endif()
else()
    message(STATUS "Catch2 found in system: ${Catch2_VERSION}")
endif()

# C-API integration test
add_executable(cinterface_integration cinterface_integration.cxx)

target_include_directories(cinterface_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(cinterface_integration PRIVATE
    Catch2::Catch2WithMain
    ${SPARSEIR_CAPI_LIB}
    Eigen3::Eigen
    Threads::Threads
)


# C-API core test
add_executable(cinterface_core cinterface_core.cxx)

target_include_directories(cinterface_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(cinterface_core PRIVATE
    Catch2::Catch2WithMain
    ${SPARSEIR_CAPI_LIB}
    Eigen3::Eigen
    Threads::Threads
)

# Optional BLAS linking
# Check environment variable first, then CMake option (default: OFF)
if(DEFINED ENV{SPARSEIR_USE_BLAS})
    if("$ENV{SPARSEIR_USE_BLAS}" MATCHES "^(ON|1|YES|TRUE)$")
        set(SPARSEIR_USE_BLAS ON)
    else()
        set(SPARSEIR_USE_BLAS OFF)
    endif()
else()
    option(SPARSEIR_USE_BLAS "Use BLAS" OFF)
endif()

# CMake option for ILP64 BLAS (can be set via -DSPARSEIR_USE_BLAS_ILP64=ON or environment variable)
# Check environment variable first, then CMake option (default: OFF)
if(DEFINED ENV{SPARSEIR_USE_BLAS_ILP64})
    if("$ENV{SPARSEIR_USE_BLAS_ILP64}" MATCHES "^(ON|1|YES|TRUE)$")
        set(SPARSEIR_USE_BLAS_ILP64 ON)
    else()
        set(SPARSEIR_USE_BLAS_ILP64 OFF)
    endif()
else()
    option(SPARSEIR_USE_BLAS_ILP64 "Use ILP64 BLAS (64-bit integers) instead of LP64 (32-bit integers)" OFF)
endif()

if (SPARSEIR_USE_BLAS)
    if(SPARSEIR_USE_BLAS_ILP64)
        set(BLA_SIZEOF_INTEGER 8)
        message(STATUS "capi_test: Using ILP64 BLAS (64-bit integers)")
    else()
        set(BLA_SIZEOF_INTEGER 4)
        message(STATUS "capi_test: Using LP64 BLAS (32-bit integers, default)")
    endif()

    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        target_link_libraries(cinterface_integration PRIVATE ${BLAS_LIBRARIES})
        target_link_libraries(cinterface_core PRIVATE ${BLAS_LIBRARIES})
        # Add compile definition to enable BLAS backend usage
        target_compile_definitions(cinterface_integration PRIVATE SPARSEIR_USE_BLAS=1)
        target_compile_definitions(cinterface_core PRIVATE SPARSEIR_USE_BLAS=1)
        # Add ILP64 definition if requested
        if(SPARSEIR_USE_BLAS_ILP64)
            target_compile_definitions(cinterface_integration PRIVATE SPARSEIR_USE_BLAS_ILP64=1)
            target_compile_definitions(cinterface_core PRIVATE SPARSEIR_USE_BLAS_ILP64=1)
        endif()
    endif()
endif()

# Enable testing
enable_testing()
add_test(NAME cinterface_integration COMMAND cinterface_integration -d yes)
add_test(NAME cinterface_core COMMAND cinterface_core -d yes)

