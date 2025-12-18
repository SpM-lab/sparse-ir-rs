cmake_minimum_required(VERSION 3.15)

# Include GNUInstallDirs for standard install paths
include(GNUInstallDirs)

# Set project (version will be set later if needed)
project(SparseIR_Fortran 
    VERSION 1.0.0
    LANGUAGES Fortran
)

# Option to automatically build Rust C API with cargo
option(SPARSEIR_BUILD_RUST_CAPI "Automatically build Rust C API with cargo" ON)

# Find Rust C API library and header
# Expected layout: _install/include/sparseir/sparseir.h, _install/lib/libsparse_ir_capi.*
# Note: If SPARSEIR_CAPI_PREFIX is set from parent (QE), use that; otherwise use default
if(NOT DEFINED SPARSEIR_CAPI_PREFIX)
    if(SPARSEIR_BUILD_RUST_CAPI)
        # Use build directory for auto-built C API
        set(SPARSEIR_CAPI_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/_rust_capi_install" CACHE PATH "Path to Rust C API installation")
    else()
        # Use default install directory
        set(SPARSEIR_CAPI_PREFIX "${CMAKE_CURRENT_LIST_DIR}/_install" CACHE PATH "Path to Rust C API installation")
    endif()
endif()
set(SPARSEIR_CAPI_INCLUDE_DIR "${SPARSEIR_CAPI_PREFIX}/include")
set(SPARSEIR_CAPI_LIB_DIR "${SPARSEIR_CAPI_PREFIX}/lib")

# Determine sparseir-rust source directory (parent directory of fortran/)
set(SPARSEIR_RUST_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "Path to sparseir-rust source directory")
if(NOT EXISTS "${SPARSEIR_RUST_SOURCE_DIR}/Cargo.toml")
    message(FATAL_ERROR "sparseir-rust source directory not found at ${SPARSEIR_RUST_SOURCE_DIR}. "
            "Set SPARSEIR_RUST_SOURCE_DIR to the sparseir-rust source directory.")
endif()

# Build Rust C API automatically if requested
if(SPARSEIR_BUILD_RUST_CAPI)
    # Find cargo
    find_program(CARGO_EXECUTABLE cargo)
    if(NOT CARGO_EXECUTABLE)
        message(FATAL_ERROR "cargo not found. Please install Rust toolchain (https://rustup.rs/) or set SPARSEIR_BUILD_RUST_CAPI=OFF")
    endif()
    message(STATUS "Found cargo: ${CARGO_EXECUTABLE}")
    message(STATUS "Will automatically build Rust C API from ${SPARSEIR_RUST_SOURCE_DIR}")
    
    # Determine library extension based on platform
    if(APPLE)
        set(SPARSEIR_LIB_EXT "dylib")
    elseif(UNIX)
        set(SPARSEIR_LIB_EXT "so")
    elseif(WIN32)
        set(SPARSEIR_LIB_EXT "dll")
    else()
        set(SPARSEIR_LIB_EXT "so")
    endif()
    
    set(SPARSEIR_CAPI_LIB_NAME "libsparse_ir_capi.${SPARSEIR_LIB_EXT}")
    set(SPARSEIR_CAPI_BUILD_LIB "${SPARSEIR_RUST_SOURCE_DIR}/target/release/${SPARSEIR_CAPI_LIB_NAME}")
    set(SPARSEIR_CAPI_BUILD_HEADER "${SPARSEIR_RUST_SOURCE_DIR}/sparse-ir-capi/include/sparseir/sparseir.h")
    
    # Custom target to build sparse-ir-capi with cargo
    add_custom_target(sparseir_capi_build
        COMMAND ${CARGO_EXECUTABLE} build --release -p sparse-ir-capi
        WORKING_DIRECTORY ${SPARSEIR_RUST_SOURCE_DIR}
        COMMENT "Building sparse-ir-capi with cargo"
        BYPRODUCTS
            "${SPARSEIR_CAPI_BUILD_LIB}"
            "${SPARSEIR_CAPI_BUILD_HEADER}"
    )
    
    # Custom target to install C-API library and header to build directory
    add_custom_target(sparseir_capi_install
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPARSEIR_CAPI_LIB_DIR}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${SPARSEIR_CAPI_BUILD_LIB}"
            "${SPARSEIR_CAPI_LIB_DIR}/${SPARSEIR_CAPI_LIB_NAME}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${SPARSEIR_CAPI_BUILD_HEADER}"
            "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h"
        DEPENDS sparseir_capi_build
        COMMENT "Installing sparse-ir-capi library and header to build directory"
    )
    
    # Create directory structure so find_library doesn't fail
    file(MAKE_DIRECTORY "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir")
    file(MAKE_DIRECTORY "${SPARSEIR_CAPI_LIB_DIR}")
    message(STATUS "sparseir/sparseir.h will be created during build by sparseir_capi_install target")
else()
    # Check if header exists (it will be created by sparseir_capi_install during build)
    if(NOT EXISTS "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
        # Create directory structure so find_library doesn't fail
        file(MAKE_DIRECTORY "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir")
        file(MAKE_DIRECTORY "${SPARSEIR_CAPI_LIB_DIR}")
        message(STATUS "sparseir/sparseir.h will be created during build by sparseir_capi_install target")
    else()
        message(STATUS "Found sparseir/sparseir.h at: ${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h")
    endif()
endif()

# Find library (supports .so, .dylib, .dll)
# Use NO_DEFAULT_PATH but don't require it - it will be created during build
find_library(SPARSEIR_CAPI_LIB
    NAMES sparse_ir_capi
    PATHS "${SPARSEIR_CAPI_LIB_DIR}"
    NO_DEFAULT_PATH
)

if(NOT SPARSEIR_CAPI_LIB)
    # Set a placeholder that will be resolved at build time
    # Determine library extension based on platform
    if(APPLE)
        set(SPARSEIR_LIB_EXT "dylib")
    elseif(UNIX)
        set(SPARSEIR_LIB_EXT "so")
    elseif(WIN32)
        set(SPARSEIR_LIB_EXT "dll")
    else()
        set(SPARSEIR_LIB_EXT "so")
    endif()
    set(SPARSEIR_CAPI_LIB "${SPARSEIR_CAPI_LIB_DIR}/libsparse_ir_capi.${SPARSEIR_LIB_EXT}")
    message(STATUS "libsparse_ir_capi will be created during build by sparseir_capi_install target")
else()
    message(STATUS "Found libsparse_ir_capi at: ${SPARSEIR_CAPI_LIB}")
endif()

# Set Fortran module directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)

add_library(sparseir_fortran SHARED
    src/sparse_ir_c.f90
    src/sparse_ir_extension.f90
    src/_cbinding.inc
    src/_cbinding_public.inc
    ${TYPE_FILES}
)

# Add compiler-specific flags for heap arrays and fastmath
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    # Intel Fortran: Put all arrays on heap to avoid stack overflow
    #target_compile_options(sparseir_fortran PRIVATE -heap-arrays 0)
    # Intel Fortran: Use precise floating-point model (same as Intel C++ for xprec compatibility)
    target_compile_options(sparseir_fortran PRIVATE -fp-model precise)
    # Also apply to test executables
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fp-model precise")
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
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    # On Linux with Intel compilers, try MKL CONFIG mode first
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "Found MKL via CONFIG mode")
        target_link_libraries(sparseir_fortran PUBLIC MKL::MKL)
    else()
        # Fallback to BLA_VENDOR method
        # Intel10_64lp: Intel MKL v10, 64-bit, LP64 model (32-bit integers), LP thread model
        message(STATUS "MKL CONFIG mode not found, trying BLA_VENDOR method")
        set(BLA_VENDOR "Intel10_64lp" CACHE STRING "BLAS vendor: Intel MKL v10 64-bit LP64")
        set(BLA_SIZEOF_INTEGER 4 CACHE STRING "BLAS integer size (4 for LP64, 8 for ILP64)")
        find_package(BLAS REQUIRED)
        if(BLAS_FOUND)
            message(STATUS "BLAS found: ${BLAS_LIBRARIES}")
            target_link_libraries(sparseir_fortran PUBLIC ${BLAS_LIBRARIES})
        else()
            message(FATAL_ERROR "BLAS not found - required for Intel compiler")
        endif()
    endif()
else()
    # On Linux with other compilers (GNU, etc.), try multiple methods
    # First, try MKL CONFIG mode (works with GNU compilers too if MKL is available)
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "Found MKL via CONFIG mode")
        target_link_libraries(sparseir_fortran PUBLIC MKL::MKL)
    else()
        # Try OpenBLAS (common on Linux)
        find_package(OpenBLAS QUIET)
        if(OpenBLAS_FOUND)
            message(STATUS "Found OpenBLAS: ${OpenBLAS_LIBRARIES}")
            target_link_libraries(sparseir_fortran PUBLIC ${OpenBLAS_LIBRARIES})
        else()
            # Try standard BLAS with BLA_VENDOR hint
            # Try Generic first (works with most BLAS implementations)
            set(BLA_VENDOR "Generic" CACHE STRING "BLAS vendor")
            find_package(BLAS QUIET)
            if(BLAS_FOUND)
                message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
                target_link_libraries(sparseir_fortran PUBLIC ${BLAS_LIBRARIES})
            else()
                # Try to find BLAS libraries manually
                find_library(BLAS_LIB
                    NAMES blas openblas
                    PATHS
                        /usr/lib
                        /usr/lib64
                        /usr/local/lib
                        /usr/local/lib64
                        /opt/OpenBLAS/lib
                    NO_DEFAULT_PATH
                )
                if(BLAS_LIB)
                    message(STATUS "Found BLAS library: ${BLAS_LIB}")
                    target_link_libraries(sparseir_fortran PUBLIC ${BLAS_LIB})
                else()
                    # Try one more time with default paths
                    find_library(BLAS_LIB
                        NAMES blas openblas
                    )
                    if(BLAS_LIB)
                        message(STATUS "Found BLAS library: ${BLAS_LIB}")
                        target_link_libraries(sparseir_fortran PUBLIC ${BLAS_LIB})
                    else()
                        message(FATAL_ERROR
                            "BLAS library not found.\n"
                            "Please install BLAS library:\n"
                            "  Ubuntu/Debian: sudo apt-get install libopenblas-dev\n"
                            "  Fedora/RHEL: sudo dnf install openblas-devel\n"
                            "  Arch Linux: sudo pacman -S openblas\n"
                            "  Or set BLAS_LIBRARIES manually via -DBLAS_LIBRARIES=/path/to/libblas.so\n"
                            "  Or set MKL_DIR if using Intel MKL: -DMKL_DIR=/path/to/mkl"
                        )
                    endif()
                endif()
            endif()
        endif()
    endif()
endif()

# Link Rust C API library
# SPARSEIR_LIB_EXT is already set above if SPARSEIR_BUILD_RUST_CAPI is ON
# Otherwise, determine library extension based on platform
if(NOT DEFINED SPARSEIR_LIB_EXT)
    if(APPLE)
        set(SPARSEIR_LIB_EXT "dylib")
    elseif(UNIX)
        set(SPARSEIR_LIB_EXT "so")
    elseif(WIN32)
        set(SPARSEIR_LIB_EXT "dll")
    else()
        set(SPARSEIR_LIB_EXT "so")
    endif()
endif()

# Use the library path - it will be created by sparseir_capi_install during build
if(EXISTS "${SPARSEIR_CAPI_LIB}")
    target_link_libraries(sparseir_fortran PUBLIC ${SPARSEIR_CAPI_LIB})
else()
    # Library doesn't exist yet, but will be created during build
    # Use the expected path - CMake will resolve it at link time
    set(SPARSEIR_CAPI_LIB_PATH "${SPARSEIR_CAPI_LIB_DIR}/libsparse_ir_capi.${SPARSEIR_LIB_EXT}")
    target_link_libraries(sparseir_fortran PUBLIC "${SPARSEIR_CAPI_LIB_PATH}")
endif()

# Make Fortran library depend on Rust C API build if auto-build is enabled
if(SPARSEIR_BUILD_RUST_CAPI)
    add_dependencies(sparseir_fortran sparseir_capi_install)
endif()

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

# Install Rust C API library and header
if(SPARSEIR_BUILD_RUST_CAPI)
    # Install Rust C API library
    install(
        FILES "${SPARSEIR_CAPI_LIB_DIR}/${SPARSEIR_CAPI_LIB_NAME}"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT sparseir_capi
    )
    
    # Install Rust C API header
    install(
        FILES "${SPARSEIR_CAPI_INCLUDE_DIR}/sparseir/sparseir.h"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/sparseir"
        COMPONENT sparseir_capi
    )
endif()

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

# Copy Fortran source files and module files to QE build directory if requested
# This is used when sparseir-rust/fortran is built as a subdirectory from QE
if(DEFINED QE_BUILD_DIR)
    set(SPARSEIR_FORTRAN_QE_DIR "${QE_BUILD_DIR}/include/sparseir_fortran")
    
    # Custom target to copy Fortran sources to QE build directory
    add_custom_target(sparseir_fortran_copy_to_qe
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPARSEIR_FORTRAN_QE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_ir_c.f90"
            "${SPARSEIR_FORTRAN_QE_DIR}/sparse_ir_c.f90"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/src/sparse_ir_extension.f90"
            "${SPARSEIR_FORTRAN_QE_DIR}/sparse_ir_extension.f90"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/src/_cbinding.inc"
            "${SPARSEIR_FORTRAN_QE_DIR}/_cbinding.inc"
        COMMAND ${CMAKE_COMMAND} -E copy
            "${CMAKE_CURRENT_SOURCE_DIR}/src/_cbinding_public.inc"
            "${SPARSEIR_FORTRAN_QE_DIR}/_cbinding_public.inc"
        COMMENT "Copying Fortran binding sources to QE build directory: ${SPARSEIR_FORTRAN_QE_DIR}"
    )
    
    # Copy module files after build (only .mod files)
    add_custom_command(TARGET sparseir_fortran POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SPARSEIR_FORTRAN_QE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_Fortran_MODULE_DIRECTORY}/sparse_ir_c.mod"
            "${SPARSEIR_FORTRAN_QE_DIR}/sparse_ir_c.mod"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_Fortran_MODULE_DIRECTORY}/sparse_ir_extension.mod"
            "${SPARSEIR_FORTRAN_QE_DIR}/sparse_ir_extension.mod"
        COMMENT "Copying Fortran module files to QE build directory"
    )
    
    # Ensure copy happens after build
    add_dependencies(sparseir_fortran_copy_to_qe sparseir_fortran)
    
    message(STATUS "Will copy Fortran bindings to QE build directory: ${SPARSEIR_FORTRAN_QE_DIR}")
endif()
