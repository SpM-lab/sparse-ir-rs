! Fortran interface for the SparseIR library
! This module provides Fortran bindings to the SparseIR C API using iso_c_binding
module sparseir
   use, intrinsic :: iso_c_binding
   implicit none
   private

   ! Export public interfaces
   !include '_fortran_types_public.inc'
   include '_cbinding_public.inc'
   public :: SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC, SPIR_ORDER_COLUMN_MAJOR
   public :: SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2, SPIR_TWORK_AUTO

   ! Constants for statistics types
   integer(c_int32_t), parameter :: SPIR_STATISTICS_FERMIONIC = 1_c_int32_t
   integer(c_int32_t), parameter :: SPIR_STATISTICS_BOSONIC = 0_c_int32_t
   integer(c_int32_t), parameter :: SPIR_ORDER_ROW_MAJOR = 0_c_int32_t
   integer(c_int32_t), parameter :: SPIR_ORDER_COLUMN_MAJOR = 1_c_int32_t

   ! Constants for Twork types
   integer(c_int32_t), parameter :: SPIR_TWORK_FLOAT64 = 0_c_int32_t
   integer(c_int32_t), parameter :: SPIR_TWORK_FLOAT64X2 = 1_c_int32_t
   integer(c_int32_t), parameter :: SPIR_TWORK_AUTO = -1_c_int32_t

   ! C bindings
   interface
      include '_cbinding.inc'
   end interface
end module sparseir
