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
   ! C-compatible wrapper functions for BLAS (bind(c) to allow c_funloc)
   ! These wrappers call the standard Fortran BLAS functions internally
   !-----------------------------------------------------------------------
   subroutine spir_dgemm_wrapper(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(c)
      !! C-compatible wrapper for dgemm that calls standard Fortran BLAS
      use, intrinsic :: iso_c_binding
      implicit none
      character(c_char), intent(in) :: transa, transb
      integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
      real(c_double), intent(in) :: alpha, beta
      real(c_double), intent(in), target :: a(*), b(*)
      real(c_double), intent(inout), target :: c(*)
      
      ! Declare standard Fortran BLAS as EXTERNAL
      external :: dgemm
      
      ! Convert C types to Fortran types
      character(1) :: transa_f, transb_f
      integer :: m_f, n_f, k_f, lda_f, ldb_f, ldc_f
      real(8) :: alpha_f, beta_f
      real(8), pointer :: a_f(:), b_f(:), c_f(:)
      
      ! Convert character (c_char to character(1))
      transa_f = achar(iachar(transa))
      transb_f = achar(iachar(transb))
      
      ! Convert integers
      m_f = int(m)
      n_f = int(n)
      k_f = int(k)
      lda_f = int(lda)
      ldb_f = int(ldb)
      ldc_f = int(ldc)
      
      ! Convert reals (c_double and real(8) are usually the same)
      alpha_f = real(alpha, kind=8)
      beta_f = real(beta, kind=8)
      
      ! Associate C arrays with Fortran arrays using c_f_pointer
      ! Since c_double and real(8) have the same memory layout, this is safe
      ! Use large enough size to cover the arrays (BLAS uses leading dimensions)
      call c_f_pointer(c_loc(a), a_f, [lda_f * max(m_f, k_f)])
      call c_f_pointer(c_loc(b), b_f, [ldb_f * max(n_f, k_f)])
      call c_f_pointer(c_loc(c), c_f, [ldc_f * n_f])
      
      ! Call standard Fortran BLAS
      call dgemm(transa_f, transb_f, m_f, n_f, k_f, alpha_f, &
                 a_f, lda_f, b_f, ldb_f, beta_f, c_f, ldc_f)
   end subroutine spir_dgemm_wrapper

   subroutine spir_zgemm_wrapper(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(c)
      !! C-compatible wrapper for zgemm that calls standard Fortran BLAS
      use, intrinsic :: iso_c_binding
      implicit none
      character(c_char), intent(in) :: transa, transb
      integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
      complex(c_double_complex), intent(in) :: alpha, beta
      complex(c_double_complex), intent(in), target :: a(*), b(*)
      complex(c_double_complex), intent(inout), target :: c(*)
      
      ! Declare standard Fortran BLAS as EXTERNAL
      external :: zgemm
      
      ! Convert C types to Fortran types
      character(1) :: transa_f, transb_f
      integer :: m_f, n_f, k_f, lda_f, ldb_f, ldc_f
      complex(8) :: alpha_f, beta_f
      complex(8), pointer :: a_f(:), b_f(:), c_f(:)
      
      ! Convert character (c_char to character(1))
      transa_f = achar(iachar(transa))
      transb_f = achar(iachar(transb))
      
      ! Convert integers
      m_f = int(m)
      n_f = int(n)
      k_f = int(k)
      lda_f = int(lda)
      ldb_f = int(ldb)
      ldc_f = int(ldc)
      
      ! Convert complex (c_double_complex and complex(8) are usually the same)
      alpha_f = cmplx(real(alpha, kind=8), aimag(alpha), kind=8)
      beta_f = cmplx(real(beta, kind=8), aimag(beta), kind=8)
      
      ! Associate C arrays with Fortran arrays using c_f_pointer
      call c_f_pointer(c_loc(a), a_f, [lda_f * max(m_f, k_f)])
      call c_f_pointer(c_loc(b), b_f, [ldb_f * max(n_f, k_f)])
      call c_f_pointer(c_loc(c), c_f, [ldc_f * n_f])
      
      ! Call standard Fortran BLAS
      call zgemm(transa_f, transb_f, m_f, n_f, k_f, alpha_f, &
                 a_f, lda_f, b_f, ldb_f, beta_f, c_f, ldc_f)
   end subroutine spir_zgemm_wrapper

   !-----------------------------------------------------------------------
   function spir_gemm_backend_new_from_fblas_lp64() result(backend_ptr)
      !-----------------------------------------------------------------------
      !! Create a GEMM backend using C-compatible wrapper functions
      !!
      !! This function creates a backend handle using C-compatible wrapper functions.
      !! The wrappers internally call the standard Fortran BLAS functions via EXTERNAL declarations.
      !!
      !! @return Pointer to spir_gemm_backend, or c_null_ptr on failure
      !-----------------------------------------------------------------------
      type(c_ptr) :: backend_ptr
      type(c_funptr) :: dgemm_ptr, zgemm_ptr
      type(c_ptr) :: dgemm_cptr, zgemm_cptr

      ! Get C-compatible wrapper function pointers (bind(c) allows c_funloc)
      dgemm_ptr = c_funloc(spir_dgemm_wrapper)
      zgemm_ptr = c_funloc(spir_zgemm_wrapper)

      ! Convert c_funptr to c_ptr using transfer
      dgemm_cptr = transfer(dgemm_ptr, dgemm_cptr)
      zgemm_cptr = transfer(zgemm_ptr, zgemm_cptr)

      ! Create backend from C-compatible wrapper pointers
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
