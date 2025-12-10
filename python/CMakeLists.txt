cmake_minimum_required(VERSION 3.15)

# This is the Python package CMakeLists.txt
# It builds the shared library and installs it for Python use

message(STATUS "CMakeLists.txt invoked")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type specified, defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add parent directory to module path for version extraction
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules")

# Provides CMAKE_INSTALL_* directories
include(GNUInstallDirs)

# Configure project for Python packaging
project(
    pylibsparseir
    LANGUAGES CXX
    DESCRIPTION "Python bindings for SparseIR library"
    )

# Set installation directories
set(SPARSEIR_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}"
    CACHE PATH "directory into which to install sparseir include files" FORCE)
set(SPARSEIR_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
    CACHE PATH "directory into which to install sparseir library files" FORCE)

# Add Fortran language support if building Fortran bindings
option(SPARSEIR_BUILD_FORTRAN "Build Fortran bindings" OFF)
if(SPARSEIR_BUILD_FORTRAN)
    enable_language(Fortran)
    message(STATUS "Fortran support enabled for building Fortran bindings")
endif()

# Do not resolve BLAS library symbols (assume sharing with NumPy)
# Do not call find_package(BLAS) or, if called, do not pass to target_link_libraries

# External FBLAS pointer option
if(SPARSEIR_USE_EXTERN_FBLAS_PTR)
    add_compile_definitions(SPARSEIR_USE_EXTERN_FBLAS_PTR)
    message(STATUS "External FBLAS pointer support enabled")
endif()

set(EIGEN3_REQUIRED_VERSION "3.4.1")
find_package (Eigen3 ${EIGEN3_REQUIRED_VERSION} QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found in system, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG ${EIGEN3_REQUIRED_VERSION}
    )
    FetchContent_MakeAvailable(Eigen3)
    message(STATUS "Eigen3 fetched from GitHub")
else()
    message(STATUS "Eigen3 found in system: ${Eigen3_VERSION}")
endif()


find_package(Threads REQUIRED)

# libxprec - built only as a static dependency, not for installation
include(FetchContent)
FetchContent_Declare(XPrec
    GIT_REPOSITORY https://github.com/tuwien-cms/libxprec
    GIT_TAG v0.7.0
)

# Extract the content but don't process it through add_subdirectory
FetchContent_GetProperties(XPrec)
if(NOT xprec_POPULATED)
    FetchContent_Populate(XPrec)
    message(STATUS "XPrec source directory: ${xprec_SOURCE_DIR}")
endif()

# ---------------------------------
# Building

# Enable debug logging
option(SPARSEIR_DEBUG "Enable debug logging output" OFF)
if(SPARSEIR_DEBUG)
    add_definitions(-DDEBUG_SPIR)
    message(STATUS "Debug logging enabled")
endif()

# Set the default component name for installations
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME sparseir)

# Check if source files exist and provide helpful error message
#if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../src/utils.cpp")
    #message(FATAL_ERROR "Source files not found. Please run: python3 prepare_build.py")
#endif()

# Explicitly list source files from copied directories
file(GLOB SPARSEIR_SOURCES "src/*.cpp")
add_library(sparseir SHARED ${SPARSEIR_SOURCES})
if(NOT MSVC)
    target_compile_options(sparseir PRIVATE -Wall -Wextra -pedantic)
endif()

if(APPLE)
    # macOS: Resolve undefined symbols at runtime (BLAS/OpenBLAS)
    set_target_properties(sparseir PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
elseif(UNIX)
    # Linux: Allow unresolved symbols
    set_target_properties(sparseir PROPERTIES
        LINK_FLAGS "-Wl,--unresolved-symbols=ignore-all"
    )
endif()

target_include_directories(sparseir
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        $<BUILD_INTERFACE:${xprec_SOURCE_DIR}/include>
)
set_target_properties(sparseir PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    )

target_link_libraries(sparseir PRIVATE Eigen3::Eigen Threads::Threads)

# Use library convention.
add_library(SparseIR::sparseir ALIAS sparseir)

# Note: ctypes_autogen.py should be generated before building
# Run: uv run python tools/gen_ctypes.py (or use update_wrapper.sh)
# CMake does not generate it to avoid uv run freeze during configuration

# Installation

# Install the shared library to the Python package directory
install(TARGETS sparseir
    LIBRARY DESTINATION pylibsparseir
    ARCHIVE DESTINATION pylibsparseir
    RUNTIME DESTINATION pylibsparseir
    COMPONENT sparseir
)

# Install Python source files (including ctypes_autogen.py if generated)
install(FILES
    pylibsparseir/__init__.py
    pylibsparseir/core.py
    pylibsparseir/constants.py
    pylibsparseir/ctypes_wrapper.py
    pylibsparseir/clean_build_artifacts.py
    pylibsparseir/ctypes_autogen.py
    DESTINATION pylibsparseir
    COMPONENT sparseir
)
