! Test memory management: object release order, memory leaks
program test_memory
   use sparse_ir_c
   use sparse_ir_extension
   use, intrinsic :: iso_c_binding
   implicit none

   print *, "======================================"
   print *, "Testing memory management"
   print *, "======================================"

   call test_object_release_order()
   call test_multiple_allocations()
   call test_set_beta_and_deallocate()

   print *, "======================================"
   print *, "All memory management tests passed!"
   print *, "======================================"

contains

   subroutine test_object_release_order()
      type(c_ptr) :: k_ptr, k_copy_ptr, sve_ptr
      type(c_ptr) :: basis_ptr, dlr_ptr
      type(c_ptr) :: tau_sampling_ptr, matsu_sampling_ptr
      type(c_ptr) :: u_ptr, v_ptr, uhat_ptr
      integer(c_int), target :: status, ntau, nmatsu
      integer(c_int), parameter :: positive_only = 0_c_int
      real(c_double), allocatable, target :: taus(:)
      integer(c_int64_t), allocatable, target :: matsus(:)
      real(c_double), parameter :: beta = 10.0_c_double
      real(c_double), parameter :: omega_max = 2.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size

      print *, "Testing object release order..."

      ! Create all objects
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      k_copy_ptr = c_spir_kernel_clone(k_ptr)
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get sampling points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only, c_loc(nmatsu))
      allocate(matsus(nmatsu))
      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only, c_loc(matsus))
      matsu_sampling_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only, nmatsu, c_loc(matsus), c_loc(status))

      ! Create DLR
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))

      ! Get basis functions
      u_ptr = c_spir_basis_get_u(basis_ptr, c_loc(status))
      v_ptr = c_spir_basis_get_v(basis_ptr, c_loc(status))
      uhat_ptr = c_spir_basis_get_uhat(basis_ptr, c_loc(status))

      ! Release in correct order (reverse of creation)
      call c_spir_funcs_release(uhat_ptr)
      call c_spir_funcs_release(v_ptr)
      call c_spir_funcs_release(u_ptr)
      call c_spir_basis_release(dlr_ptr)
      call c_spir_sampling_release(matsu_sampling_ptr)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_copy_ptr)
      call c_spir_kernel_release(k_ptr)

      ! Cleanup arrays
      deallocate(taus, matsus)

      print *, "  Object release order: PASSED (no segfault)"
   end subroutine test_object_release_order

   subroutine test_multiple_allocations()
      type(c_ptr) :: k_ptr1, k_ptr2, sve_ptr1, sve_ptr2
      type(c_ptr) :: basis_ptr1, basis_ptr2
      integer(c_int), target :: status
      real(c_double), parameter :: beta = 10.0_c_double
      real(c_double), parameter :: omega_max1 = 1.0_c_double
      real(c_double), parameter :: omega_max2 = 2.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda1 = beta*omega_max1
      real(c_double), parameter :: lambda2 = beta*omega_max2
      integer(c_int), target :: max_size

      print *, "Testing multiple allocations..."

      ! Create first set of objects
      k_ptr1 = c_spir_logistic_kernel_new(lambda1, c_loc(status))
      sve_ptr1 = c_spir_sve_result_new(k_ptr1, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr1 = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max1, epsilon, &
                                     k_ptr1, sve_ptr1, max_size, c_loc(status))

      ! Create second set of objects
      k_ptr2 = c_spir_logistic_kernel_new(lambda2, c_loc(status))
      sve_ptr2 = c_spir_sve_result_new(k_ptr2, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      basis_ptr2 = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max2, epsilon, &
                                     k_ptr2, sve_ptr2, max_size, c_loc(status))

      ! Verify both are valid
      if (.not. c_associated(basis_ptr1) .or. .not. c_associated(basis_ptr2)) then
         print *, "Error: Failed to create multiple basis objects"
         stop 1
      end if

      ! Release all objects
      call c_spir_basis_release(basis_ptr2)
      call c_spir_sve_result_release(sve_ptr2)
      call c_spir_kernel_release(k_ptr2)
      call c_spir_basis_release(basis_ptr1)
      call c_spir_sve_result_release(sve_ptr1)
      call c_spir_kernel_release(k_ptr1)

      print *, "  Multiple allocations: PASSED (no memory conflicts)"
   end subroutine test_multiple_allocations

   subroutine test_set_beta_and_deallocate()
      integer, parameter :: dp = KIND(1.0D0)
      type(IR) :: irobj
      real(kind=dp), parameter :: beta1 = 10.0_dp
      real(kind=dp), parameter :: beta2 = 20.0_dp
      real(kind=dp), parameter :: omega_max = 2.0_dp
      real(kind=dp), parameter :: epsilon = 1.0e-10_dp
      real(kind=dp), parameter :: lambda = beta1*omega_max
      logical, parameter :: positive_only = .false.

      print *, "Testing set_beta and deallocate_ir..."

      ! Initialize IR object
      call init_ir(irobj, beta1, lambda, epsilon, positive_only)

      ! Change beta
      call set_beta(irobj, beta2)

      ! Verify beta was updated
      if (abs(irobj%beta - beta2) > 1.0e-10_dp) then
         print *, "Error: set_beta did not update beta correctly"
         stop 1
      end if

      ! Test deallocate_ir (should deallocate beta-dependent arrays)
      call deallocate_ir(irobj)

      ! Note: After deallocate_ir, we should not call finalize_ir
      ! because arrays are already deallocated. finalize_ir expects
      ! arrays to be allocated. For normal cleanup, use finalize_ir only.

      print *, "  set_beta and deallocate_ir: PASSED"
   end subroutine test_set_beta_and_deallocate

end program test_memory

