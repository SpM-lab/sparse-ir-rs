cmake_minimum_required(VERSION 3.15)

# This is the Python package CMakeLists.txt
# It builds the Rust sparse-ir-capi library and installs it for Python use

project(pylibsparseir LANGUAGES NONE)

message(STATUS "CMakeLists.txt invoked for Python package")

# Find cargo
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR "cargo not found. Please install Rust toolchain (https://rustup.rs/)")
endif()

message(STATUS "Found cargo: ${CARGO_EXECUTABLE}")

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
set(SPARSEIR_RUST_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../sparse-ir-capi")
set(SPARSEIR_WORKSPACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
# Workspace uses shared target directory at workspace root
set(SPARSEIR_CAPI_BUILD_LIB "${SPARSEIR_WORKSPACE_ROOT}/target/release/${SPARSEIR_CAPI_LIB_NAME}")
set(SPARSEIR_CAPI_BUILD_HEADER "${SPARSEIR_RUST_SOURCE_DIR}/include/sparseir/sparseir.h")

# Set the default component name for installations
set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME sparseir)

# Custom target to build sparse-ir-capi with cargo
add_custom_target(sparseir_capi_build
    COMMAND ${CARGO_EXECUTABLE} build --release -p sparse-ir-capi
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..
    COMMENT "Building sparse-ir-capi with cargo"
    BYPRODUCTS
        "${SPARSEIR_CAPI_BUILD_LIB}"
        "${SPARSEIR_CAPI_BUILD_HEADER}"
)

# Ensure the pylibsparseir directory exists
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pylibsparseir")

# Copy library to pylibsparseir directory
add_custom_command(
    TARGET sparseir_capi_build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SPARSEIR_CAPI_BUILD_LIB}"
        "${CMAKE_CURRENT_SOURCE_DIR}/pylibsparseir/${SPARSEIR_CAPI_LIB_NAME}"
    COMMENT "Copying Rust library to pylibsparseir"
)

# Copy header file to pylibsparseir directory (optional, for reference)
add_custom_command(
    TARGET sparseir_capi_build POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SPARSEIR_CAPI_BUILD_HEADER}"
        "${CMAKE_CURRENT_SOURCE_DIR}/pylibsparseir/sparseir.h"
    COMMENT "Copying header file to pylibsparseir"
)

# Installation
# Note: scikit-build-core will automatically build sparseir_capi_build before installation
# The POST_BUILD commands ensure files are copied to pylibsparseir before install

# Install Python source files
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

# Install the Rust library (depends on build target)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pylibsparseir/${SPARSEIR_CAPI_LIB_NAME}"
    DESTINATION pylibsparseir
    COMPONENT sparseir
    OPTIONAL
)

# Install header file (optional, for reference)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pylibsparseir/sparseir.h"
    DESTINATION pylibsparseir
    COMPONENT sparseir
    OPTIONAL
)
