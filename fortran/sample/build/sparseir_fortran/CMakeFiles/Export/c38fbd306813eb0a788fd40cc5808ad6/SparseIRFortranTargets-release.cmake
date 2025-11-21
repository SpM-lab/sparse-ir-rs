#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SparseIR::sparseir_fortran" for configuration "Release"
set_property(TARGET SparseIR::sparseir_fortran APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SparseIR::sparseir_fortran PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsparseir_fortran.1.0.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libsparseir_fortran.1.dylib"
  )

list(APPEND _cmake_import_check_targets SparseIR::sparseir_fortran )
list(APPEND _cmake_import_check_files_for_SparseIR::sparseir_fortran "${_IMPORT_PREFIX}/lib/libsparseir_fortran.1.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
