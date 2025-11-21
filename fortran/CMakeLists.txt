cmake_minimum_required(VERSION 3.15)

# Include GNUInstallDirs for standard install paths
include(GNUInstallDirs)

# Set project (version will be set later if needed)
project(SparseIR_Fortran 
    VERSION 1.0.0
    LANGUAGES Fortran
)

# Find Rust C API library and header
# Expected layout: _install/include/sparseir/sparseir.h, _install/lib/libsparse_ir_capi.*
set(SPARSEIR_CAPI_PREFIX "${CMAKE_CURRENT_LIST_DIR}/_install" CACHE PATH "Path to Rust C API installation")
set(SPARSEIR_CAPI_INCLUDE_DIR "${SPARSEIR_CAPI_PREFIX}/include")
set(SPARSEIR_CAPI_LIB_DIR "${SPARSEIR_CAPI_PREFIX}/lib")

if(NOT EXISTS "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
    message(FATAL_ERROR "sparseir/sparseir.h not found at ${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
endif()

# Find library (supports .so, .dylib, .dll)
find_library(SPARSEIR_CAPI_LIB
    NAMES sparse_ir_capi
    PATHS "${SPARSEIR_CAPI_LIB_DIR}"
    REQUIRED
    NO_DEFAULT_PATH
)

if(NOT SPARSEIR_CAPI_LIB)
    message(FATAL_ERROR "libsparse_ir_capi not found in ${SPARSEIR_CAPI_LIB_DIR}")
endif()

message(STATUS "Found sparseir/sparseir.h at: ${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
message(STATUS "Found libsparse_ir_capi at: ${SPARSEIR_CAPI_LIB}")

# Enable Fortran preprocessor
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cpp")

# Set Fortran module directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)

add_library(sparseir_fortran SHARED
    src/sparseir.f90
    src/sparseir_ext.F90
    src/_cbinding.inc
    src/_cbinding_public.inc
    ${TYPE_FILES}
)

# Add compiler-specific flags for heap arrays and fastmath
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    # Intel Fortran: Put all arrays on heap to avoid stack overflow
    #target_compile_options(sparseir_fortran PRIVATE -heap-arrays 0)
    # Intel C++ uses -fp-model=precise to disable fastmath for xprec compatibility
    # This is handled in the C++ backend CMakeLists.txt
endif()

# Add include directory
target_include_directories(sparseir_fortran PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Add interface include directory for installed module files
target_include_directories(sparseir_fortran INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/fortran>
    $<INSTALL_INTERFACE:include/sparseir>
)

set_target_properties(sparseir_fortran PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON
)

# Find BLAS library
if(APPLE)
    # On macOS, use Accelerate framework
    find_library(ACCELERATE_LIB Accelerate)
    if(ACCELERATE_LIB)
        target_link_libraries(sparseir_fortran PUBLIC ${ACCELERATE_LIB})
    else()
        # Fallback to OpenBLAS if Accelerate not found
        find_package(BLAS REQUIRED)
        target_link_libraries(sparseir_fortran PUBLIC ${BLAS_LIBRARIES})
    endif()
else()
    # On Linux, use standard BLAS
    find_package(BLAS REQUIRED)
    target_link_libraries(sparseir_fortran PUBLIC ${BLAS_LIBRARIES})
endif()

# Link Rust C API library
target_link_libraries(sparseir_fortran PUBLIC ${SPARSEIR_CAPI_LIB})

# Add include directory for sparseir.h
target_include_directories(sparseir_fortran PRIVATE
    ${SPARSEIR_CAPI_INCLUDE_DIR}
)

# Add alias (for compatibility)
add_library(SparseIR::sparseir_fortran ALIAS sparseir_fortran)

# Testing (if enabled via option)
option(SPARSEIR_BUILD_TESTING "Enable creation of SparseIR tests" OFF)
if(SPARSEIR_BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()

# Install Fortran library
install(TARGETS sparseir_fortran
    EXPORT sparseirFortranTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT sparseir_fortran
)

# Install Fortran module files
install(
    DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sparseir"
    FILES_MATCHING PATTERN "*.mod"
)

# CMake package config for Fortran bindings
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SparseIRFortranConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(EXPORT sparseirFortranTargets
    FILE SparseIRFortranTargets.cmake
    NAMESPACE SparseIR::
    DESTINATION "${CMAKE_INSTALL_DATADIR}/cmake/SparseIRFortran"
)
