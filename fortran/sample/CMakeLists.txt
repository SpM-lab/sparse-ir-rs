cmake_minimum_required(VERSION 3.10)
project(second_order_perturbation_fort LANGUAGES Fortran C)

# Set Fortran standard
set(CMAKE_Fortran_STANDARD 90)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Rust C-API library
# SPARSEIR_CAPI_PREFIX should be set via CMAKE_PREFIX_PATH
find_library(SPARSEIR_CAPI_LIB
    NAMES sparse_ir_capi
    PATHS ${CMAKE_PREFIX_PATH}/lib
          ${CMAKE_PREFIX_PATH}/lib64
    NO_DEFAULT_PATH
)

if(NOT SPARSEIR_CAPI_LIB)
    # Fallback: try standard system paths
    find_library(SPARSEIR_CAPI_LIB NAMES sparse_ir_capi)
endif()

if(NOT SPARSEIR_CAPI_LIB)
    message(FATAL_ERROR "Could not find libsparse_ir_capi. Set CMAKE_PREFIX_PATH to point to the Rust C-API installation directory.")
endif()

# Find include directory
find_path(SPARSEIR_CAPI_INCLUDE_DIR
    NAMES sparseir/sparseir.h
    PATHS ${CMAKE_PREFIX_PATH}/include
    NO_DEFAULT_PATH
)

if(NOT SPARSEIR_CAPI_INCLUDE_DIR)
    find_path(SPARSEIR_CAPI_INCLUDE_DIR NAMES sparseir/sparseir.h)
endif()

if(NOT SPARSEIR_CAPI_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find sparseir/sparseir.h. Set CMAKE_PREFIX_PATH to point to the Rust C-API installation directory.")
endif()

# Add parent directory to build sparseir_fortran
add_subdirectory(.. ${CMAKE_CURRENT_BINARY_DIR}/sparseir_fortran)

find_library(FFTW3_LIB fftw3)
find_path(FFTW3_INCLUDE_DIR fftw3.f)

# Create executable
add_executable(second_order_perturbation_fort second_order_perturbation_fort.f90)

# Get module directory from sparseir_fortran
get_target_property(SPARSEIR_FORTRAN_MODDIR sparseir_fortran Fortran_MODULE_DIRECTORY)

target_include_directories(second_order_perturbation_fort PRIVATE
    ${SPARSEIR_FORTRAN_MODDIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${FFTW3_INCLUDE_DIR}
)

# Link with sparseir_fortran (which links to Rust C-API) and FFTW3
target_link_libraries(second_order_perturbation_fort PRIVATE
    sparseir_fortran
    ${FFTW3_LIB}
)

# Set RPATH for both Linux and macOS
set_target_properties(second_order_perturbation_fort PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH_USE_LINK_PATH ON
    INSTALL_RPATH "$<TARGET_FILE_DIR:sparseir_fortran>"
    MACOSX_RPATH TRUE
)