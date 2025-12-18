! Test basis function evaluation in detail
program test_basis_functions
   use sparse_ir_c
   use sparse_ir_extension
   use, intrinsic :: iso_c_binding
   implicit none

   integer, parameter :: dp = KIND(1.0D0)

   print *, "======================================"
   print *, "Testing basis function evaluation"
   print *, "======================================"

   call test_eval_u_tau()
   call test_basis_functions_batch()

   ! Test with different positive_only values
   call test_basis_functions_with_positive_only(.false., "all frequencies")
   call test_basis_functions_with_positive_only(.true., "positive only")

   print *, "======================================"
   print *, "All basis function tests passed!"
   print *, "======================================"

contains

   subroutine test_eval_u_tau()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 5.0_DP
      real(kind=dp), parameter :: omega_max = 2.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-10_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      real(kind=dp), allocatable :: u_tau(:)
      real(kind=dp) :: tau
      integer :: i

      print *, "Testing eval_u_tau..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      ! Test evaluation at multiple tau points
      ! Allocate output array before calling eval_u_tau
      allocate(u_tau(irobj%size))
      do i = 1, min(5, irobj%ntau)
         tau = irobj%tau(i)
         u_tau = eval_u_tau(irobj, SPIR_STATISTICS_FERMIONIC, tau)

         ! Verify result
         if (size(u_tau) /= irobj%size) then
            print *, "Error: u_tau size mismatch at tau point", i
            stop 1
         end if

         ! Verify values are finite
         if (any(.not. (abs(u_tau) < huge(1.0_DP)))) then
            print *, "Error: Non-finite values in u_tau at tau point", i
            stop 1
         end if
      end do
      deallocate(u_tau)
      call finalize_ir(irobj)

      print *, "  eval_u_tau: PASSED"
   end subroutine test_eval_u_tau

   subroutine test_basis_functions_batch()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, u_ptr, v_ptr, uhat_ptr
      integer(c_int), target :: status, basis_size, funcs_size
      real(c_double), allocatable, target :: u_eval(:), v_eval(:)
      real(c_double), allocatable, target :: batch_u_eval(:, :), batch_v_eval(:, :)
      integer(c_int64_t), allocatable, target :: matsu_indices(:)
      real(c_double), parameter :: beta = 2.0_c_double
      real(c_double), parameter :: omega_max = 5.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-6_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: test_tau = 0.5_c_double
      real(c_double), parameter :: test_omega = 0.5_c_double * omega_max
      integer(c_int), target :: max_size
      integer :: i, j, n_points
      real(c_double) :: tau_point, omega_point
      integer(c_int), target :: nmatsu
      integer(c_int), parameter :: positive_only = 0_c_int

      print *, "Testing basis functions batch evaluation..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))
      allocate(u_eval(basis_size))
      allocate(v_eval(basis_size))

      ! Test u functions
      u_ptr = c_spir_basis_get_u(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(u_ptr)) then
         print *, "Error: Failed to get u functions"
         stop 1
      end if

      status = c_spir_funcs_get_size(u_ptr, c_loc(funcs_size))
      if (funcs_size /= basis_size) then
         print *, "Error: u functions size mismatch"
         stop 1
      end if

      ! Single point evaluation
      status = c_spir_funcs_eval(u_ptr, test_tau, c_loc(u_eval))
      if (status /= 0) then
         print *, "Error: Failed to evaluate u functions"
         stop 1
      end if

      ! Verify values are finite
      do i = 1, basis_size
         if (.not. (abs(u_eval(i)) < huge(1.0_c_double))) then
            print *, "Error: Non-finite value in u_eval at index", i
            stop 1
         end if
      end do

      ! Batch evaluation for u functions
      n_points = 5
      allocate(batch_u_eval(n_points, basis_size))
      do i = 1, n_points
         tau_point = real(i, kind=c_double) * test_tau / real(n_points, kind=c_double)
         status = c_spir_funcs_eval(u_ptr, tau_point, c_loc(batch_u_eval(i, 1)))
         if (status /= 0) then
            print *, "Error: Failed to evaluate u functions in batch at point", i
            stop 1
         end if
      end do

      ! Test v functions
      v_ptr = c_spir_basis_get_v(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(v_ptr)) then
         print *, "Error: Failed to get v functions"
         stop 1
      end if

      ! Single point evaluation for v functions
      status = c_spir_funcs_eval(v_ptr, test_omega, c_loc(v_eval))
      if (status /= 0) then
         print *, "Error: Failed to evaluate v functions"
         stop 1
      end if

      ! Verify values are finite
      do i = 1, basis_size
         if (.not. (abs(v_eval(i)) < huge(1.0_c_double))) then
            print *, "Error: Non-finite value in v_eval at index", i
            stop 1
         end if
      end do

      ! Batch evaluation for v functions
      allocate(batch_v_eval(n_points, basis_size))
      do i = 1, n_points
         omega_point = real(i, c_double) * test_omega / real(n_points, c_double)
         status = c_spir_funcs_eval(v_ptr, omega_point, c_loc(batch_v_eval(i, 1)))
         if (status /= 0) then
            print *, "Error: Failed to evaluate v functions in batch at point", i
            stop 1
         end if
      end do

      ! Test uhat functions (Matsubara)
      uhat_ptr = c_spir_basis_get_uhat(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(uhat_ptr)) then
         print *, "Error: Failed to get uhat functions"
         stop 1
      end if

      ! Cleanup
      ! Release in reverse order of acquisition (following test_object_release_order pattern)
      deallocate(u_eval, v_eval, batch_u_eval, batch_v_eval)
      call c_spir_funcs_release(uhat_ptr)  ! Release uhat_ptr first (reverse order)
      call c_spir_funcs_release(v_ptr)
      call c_spir_funcs_release(u_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Basis functions batch evaluation: PASSED"
      flush(6)
   end subroutine test_basis_functions_batch

   subroutine test_basis_functions_with_positive_only(positive_only_val, case_name)
      logical, intent(in) :: positive_only_val
      character(len=*), intent(in) :: case_name
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr
      integer(c_int), target :: status, basis_size, nmatsu
      integer(c_int), target :: max_size
      integer(c_int) :: positive_only_c
      real(c_double), parameter :: beta = 5.0_c_double
      real(c_double), parameter :: omega_max = 2.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max

      print *, "Testing basis functions with ", case_name, "..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      if (status /= 0 .or. .not. c_associated(k_ptr)) then
         print *, "Error: Failed to create kernel"
         stop 1
      end if

      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      if (status /= 0 .or. .not. c_associated(sve_ptr)) then
         print *, "Error: Failed to create SVE result"
         stop 1
      end if

      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))
      if (status /= 0 .or. .not. c_associated(basis_ptr)) then
         print *, "Error: Failed to create basis"
         stop 1
      end if

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Convert logical to integer(c_int) explicitly, following create_matsu_smpl pattern
      positive_only_c = MERGE(1, 0, positive_only_val)

      ! Verify that we can get the number of Matsubara frequencies
      ! This is a basic test to verify the basis setup works correctly
      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only_c, c_loc(nmatsu))
      if (status /= 0) then
         print *, "Error: Failed to get number of Matsubara frequencies"
         stop 1
      end if

      ! Cleanup
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Basis functions (", case_name, "): PASSED"
   end subroutine test_basis_functions_with_positive_only

end program test_basis_functions

