! Test core functionality: kernels, SVE, and basis functions
program test_core
   use sparse_ir_c
   use, intrinsic :: iso_c_binding
   implicit none

   print *, "======================================"
   print *, "Testing core functionality"
   print *, "======================================"

   call test_kernel_creation()
   call test_sve_computation()
   call test_basis_creation()
   call test_basis_functions()
   call test_kernel_clone()

   print *, "======================================"
   print *, "All core tests passed!"
   print *, "======================================"

contains

   subroutine test_kernel_creation()
      type(c_ptr) :: k_log_ptr, k_bose_ptr
      integer(c_int), target :: status
      real(c_double), target :: xmin, xmax, ymin, ymax
      real(c_double), parameter :: lambda = 10.0_c_double

      print *, "Testing kernel creation..."

      ! Test logistic kernel
      k_log_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      if (status /= 0 .or. .not. c_associated(k_log_ptr)) then
         print *, "Error: Failed to create logistic kernel"
         stop 1
      end if

      ! Test kernel domain
      status = c_spir_kernel_get_domain(k_log_ptr, c_loc(xmin), c_loc(xmax), c_loc(ymin), c_loc(ymax))
      if (status /= 0) then
         print *, "Error: Failed to get kernel domain"
         stop 1
      end if
      if (abs(xmin - (-1.0_c_double)) > 1.0e-10 .or. abs(xmax - 1.0_c_double) > 1.0e-10) then
         print *, "Error: Kernel domain mismatch"
         stop 1
      end if

      ! Test regularized boson kernel
      k_bose_ptr = c_spir_reg_bose_kernel_new(lambda, c_loc(status))
      if (status /= 0 .or. .not. c_associated(k_bose_ptr)) then
         print *, "Error: Failed to create regularized boson kernel"
         stop 1
      end if

      ! Cleanup
      call c_spir_kernel_release(k_log_ptr)
      call c_spir_kernel_release(k_bose_ptr)

      print *, "  Kernel creation: PASSED"
   end subroutine test_kernel_creation

   subroutine test_sve_computation()
      type(c_ptr) :: k_ptr, sve_ptr
      integer(c_int), target :: status, sve_size
      real(c_double), allocatable, target :: svals(:)
      real(c_double), parameter :: lambda = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-6_c_double
      integer :: i

      print *, "Testing SVE computation..."

      ! Create kernel
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      if (status /= 0) then
         print *, "Error: Failed to create kernel"
         stop 1
      end if

      ! Create SVE result
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      if (status /= 0 .or. .not. c_associated(sve_ptr)) then
         print *, "Error: Failed to create SVE result"
         stop 1
      end if

      ! Get SVE size
      status = c_spir_sve_result_get_size(sve_ptr, c_loc(sve_size))
      if (status /= 0 .or. sve_size <= 0) then
         print *, "Error: Invalid SVE size"
         stop 1
      end if

      ! Get singular values
      allocate(svals(sve_size))
      status = c_spir_sve_result_get_svals(sve_ptr, c_loc(svals))
      if (status /= 0) then
         print *, "Error: Failed to get singular values"
         stop 1
      end if

      ! Verify singular values are positive and in descending order
      do i = 1, sve_size
         if (svals(i) <= 0.0_c_double) then
            print *, "Error: Singular value is not positive:", svals(i)
            stop 1
         end if
         if (i > 1 .and. svals(i) > svals(i-1)) then
            print *, "Error: Singular values are not in descending order"
            stop 1
         end if
      end do

      ! Cleanup
      deallocate(svals)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  SVE computation: PASSED"
   end subroutine test_sve_computation

   subroutine test_basis_creation()
      type(c_ptr) :: k_ptr, sve_ptr, basis_f_ptr, basis_b_ptr
      integer(c_int), target :: status, basis_size_f, basis_size_b, stats_f, stats_b
      real(c_double), allocatable, target :: svals_f(:), svals_b(:)
      real(c_double), parameter :: beta = 10.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-6_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size

      print *, "Testing basis creation..."

      ! Create kernel and SVE
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))

      ! Test fermion basis
      max_size = -1
      basis_f_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                     k_ptr, sve_ptr, max_size, c_loc(status))
      if (status /= 0 .or. .not. c_associated(basis_f_ptr)) then
         print *, "Error: Failed to create fermion basis"
         stop 1
      end if

      status = c_spir_basis_get_size(basis_f_ptr, c_loc(basis_size_f))
      if (status /= 0 .or. basis_size_f <= 0) then
         print *, "Error: Invalid fermion basis size"
         stop 1
      end if

      status = c_spir_basis_get_stats(basis_f_ptr, c_loc(stats_f))
      if (status /= 0 .or. stats_f /= SPIR_STATISTICS_FERMIONIC) then
         print *, "Error: Fermion basis statistics mismatch"
         stop 1
      end if

      allocate(svals_f(basis_size_f))
      status = c_spir_basis_get_svals(basis_f_ptr, c_loc(svals_f))
      if (status /= 0) then
         print *, "Error: Failed to get fermion basis singular values"
         stop 1
      end if

      ! Test boson basis
      basis_b_ptr = c_spir_basis_new(SPIR_STATISTICS_BOSONIC, beta, omega_max, epsilon, &
                                     k_ptr, sve_ptr, max_size, c_loc(status))
      if (status /= 0 .or. .not. c_associated(basis_b_ptr)) then
         print *, "Error: Failed to create boson basis"
         stop 1
      end if

      status = c_spir_basis_get_size(basis_b_ptr, c_loc(basis_size_b))
      if (status /= 0 .or. basis_size_b <= 0) then
         print *, "Error: Invalid boson basis size"
         stop 1
      end if

      status = c_spir_basis_get_stats(basis_b_ptr, c_loc(stats_b))
      if (status /= 0 .or. stats_b /= SPIR_STATISTICS_BOSONIC) then
         print *, "Error: Boson basis statistics mismatch"
         stop 1
      end if

      allocate(svals_b(basis_size_b))
      status = c_spir_basis_get_svals(basis_b_ptr, c_loc(svals_b))
      if (status /= 0) then
         print *, "Error: Failed to get boson basis singular values"
         stop 1
      end if

      ! Cleanup
      deallocate(svals_f, svals_b)
      call c_spir_basis_release(basis_f_ptr)
      call c_spir_basis_release(basis_b_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Basis creation: PASSED"
   end subroutine test_basis_creation

   subroutine test_basis_functions()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, u_ptr, v_ptr, uhat_ptr
      integer(c_int), target :: status, basis_size, funcs_size
      real(c_double), allocatable, target :: u_eval(:)
      real(c_double), parameter :: beta = 5.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: test_tau = 0.5_c_double
      integer(c_int), target :: max_size

      print *, "Testing basis functions..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))
      allocate(u_eval(basis_size))

      ! Test u functions
      u_ptr = c_spir_basis_get_u(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(u_ptr)) then
         print *, "Error: Failed to get u functions"
         stop 1
      end if

      status = c_spir_funcs_get_size(u_ptr, c_loc(funcs_size))
      if (status /= 0 .or. funcs_size /= basis_size) then
         print *, "Error: u functions size mismatch"
         stop 1
      end if

      status = c_spir_funcs_eval(u_ptr, test_tau, c_loc(u_eval))
      if (status /= 0) then
         print *, "Error: Failed to evaluate u functions"
         stop 1
      end if

      ! Test v functions
      v_ptr = c_spir_basis_get_v(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(v_ptr)) then
         print *, "Error: Failed to get v functions"
         stop 1
      end if

      ! Test uhat functions
      uhat_ptr = c_spir_basis_get_uhat(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(uhat_ptr)) then
         print *, "Error: Failed to get uhat functions"
         stop 1
      end if

      ! Cleanup
      deallocate(u_eval)
      call c_spir_funcs_release(u_ptr)
      call c_spir_funcs_release(v_ptr)
      call c_spir_funcs_release(uhat_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Basis functions: PASSED"
   end subroutine test_basis_functions

   subroutine test_kernel_clone()
      type(c_ptr) :: k_ptr, k_copy_ptr
      integer(c_int), target :: status
      real(c_double), target :: xmin1, xmax1, ymin1, ymax1
      real(c_double), target :: xmin2, xmax2, ymin2, ymax2
      real(c_double), parameter :: lambda = 10.0_c_double

      print *, "Testing kernel clone..."

      ! Create original kernel
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      if (status /= 0) then
         print *, "Error: Failed to create kernel"
         stop 1
      end if

      ! Get domain of original
      status = c_spir_kernel_get_domain(k_ptr, c_loc(xmin1), c_loc(xmax1), c_loc(ymin1), c_loc(ymax1))
      if (status /= 0) then
         print *, "Error: Failed to get original kernel domain"
         stop 1
      end if

      ! Clone kernel
      k_copy_ptr = c_spir_kernel_clone(k_ptr)
      if (.not. c_associated(k_copy_ptr)) then
         print *, "Error: Failed to clone kernel"
         stop 1
      end if

      ! Get domain of clone
      status = c_spir_kernel_get_domain(k_copy_ptr, c_loc(xmin2), c_loc(xmax2), c_loc(ymin2), c_loc(ymax2))
      if (status /= 0) then
         print *, "Error: Failed to get cloned kernel domain"
         stop 1
      end if

      ! Verify domains match
      if (abs(xmin1 - xmin2) > 1.0e-10 .or. abs(xmax1 - xmax2) > 1.0e-10 .or. &
          abs(ymin1 - ymin2) > 1.0e-10 .or. abs(ymax1 - ymax2) > 1.0e-10) then
         print *, "Error: Cloned kernel domain mismatch"
         stop 1
      end if

      ! Cleanup
      call c_spir_kernel_release(k_ptr)
      call c_spir_kernel_release(k_copy_ptr)

      print *, "  Kernel clone: PASSED"
   end subroutine test_kernel_clone

end program test_core










