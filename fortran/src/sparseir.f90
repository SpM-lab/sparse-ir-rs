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

   ! Fortran-friendly wrapper for BLAS backend creation
   public :: spir_gemm_backend_new_from_fblas_lp64, spir_gemm_backend_release

contains

   !-----------------------------------------------------------------------
   function spir_gemm_backend_new_from_fblas_lp64(dgemm, zgemm) result(backend_ptr)
      !-----------------------------------------------------------------------
      !! Create a GEMM backend from Fortran BLAS function pointers (LP64)
      !!
      !! This function creates a backend handle from Fortran BLAS function pointers.
      !! The function pointers are converted to C pointers internally.
      !!
      !! @param dgemm Fortran BLAS dgemm function (double precision)
      !! @param zgemm Fortran BLAS zgemm function (complex double precision)
      !! @return Pointer to spir_gemm_backend, or c_null_ptr on failure
      !-----------------------------------------------------------------------
      ! Declare BLAS function interfaces
      interface
         subroutine dgemm_blas(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import :: c_char, c_int, c_double
            character(c_char), intent(in) :: transa, transb
            integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
            real(c_double), intent(in) :: alpha, beta
            real(c_double), intent(in) :: a(lda, *), b(ldb, *)
            real(c_double), intent(inout) :: c(ldc, *)
         end subroutine dgemm_blas
         subroutine zgemm_blas(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import :: c_char, c_int, c_double_complex
            character(c_char), intent(in) :: transa, transb
            integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
            complex(c_double_complex), intent(in) :: alpha, beta
            complex(c_double_complex), intent(in) :: a(lda, *), b(ldb, *)
            complex(c_double_complex), intent(inout) :: c(ldc, *)
         end subroutine zgemm_blas
      end interface

      procedure(dgemm_blas) :: dgemm
      procedure(zgemm_blas) :: zgemm
      type(c_ptr) :: backend_ptr

      type(c_funptr) :: dgemm_ptr, zgemm_ptr
      type(c_ptr) :: dgemm_cptr, zgemm_cptr

      ! Get Fortran BLAS function pointers
      dgemm_ptr = c_funloc(dgemm)
      zgemm_ptr = c_funloc(zgemm)

      ! Convert c_funptr to c_ptr using transfer
      dgemm_cptr = transfer(dgemm_ptr, dgemm_cptr)
      zgemm_cptr = transfer(zgemm_ptr, zgemm_cptr)

      ! Create backend from Fortran BLAS pointers
      backend_ptr = c_spir_gemm_backend_new_from_fblas_lp64(dgemm_cptr, zgemm_cptr)
   end function spir_gemm_backend_new_from_fblas_lp64

   !-----------------------------------------------------------------------
   subroutine spir_gemm_backend_release(backend_ptr)
      !-----------------------------------------------------------------------
      !! Release a GEMM backend handle
      !!
      !! @param backend_ptr Pointer to spir_gemm_backend to release
      !-----------------------------------------------------------------------
      type(c_ptr), intent(in) :: backend_ptr
      call c_spir_gemm_backend_release(backend_ptr)
   end subroutine spir_gemm_backend_release

end module sparseir
