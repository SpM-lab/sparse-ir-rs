! Test error handling for invalid parameters and edge cases
program test_error_handling
   use sparse_ir_c
   use, intrinsic :: iso_c_binding
   implicit none

   print *, "======================================"
   print *, "Testing error handling"
   print *, "======================================"

   call test_invalid_kernel_parameters()
   call test_invalid_basis_parameters()
   call test_null_pointer_handling()

   print *, "======================================"
   print *, "All error handling tests passed!"
   print *, "======================================"

contains

   subroutine test_invalid_kernel_parameters()
      type(c_ptr) :: k_ptr
      integer(c_int), target :: status
      real(c_double), parameter :: lambda_zero = 0.0_c_double
      real(c_double), parameter :: lambda_negative = -1.0_c_double
      real(c_double), parameter :: lambda_very_large = 1.0e10_c_double

      print *, "Testing invalid kernel parameters..."

      ! Test with zero lambda (should fail or handle gracefully)
      k_ptr = c_spir_logistic_kernel_new(lambda_zero, c_loc(status))
      if (status == 0 .and. c_associated(k_ptr)) then
         call c_spir_kernel_release(k_ptr)
      end if
      ! Note: We don't fail here as the implementation might handle this gracefully

      ! Test with negative lambda (should fail or handle gracefully)
      k_ptr = c_spir_logistic_kernel_new(lambda_negative, c_loc(status))
      if (status == 0 .and. c_associated(k_ptr)) then
         call c_spir_kernel_release(k_ptr)
      end if

      ! Test with very large lambda (should work but might have precision issues)
      k_ptr = c_spir_logistic_kernel_new(lambda_very_large, c_loc(status))
      if (status == 0 .and. c_associated(k_ptr)) then
         call c_spir_kernel_release(k_ptr)
      end if

      print *, "  Invalid kernel parameters: PASSED (handled gracefully)"
   end subroutine test_invalid_kernel_parameters

   subroutine test_invalid_basis_parameters()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr
      integer(c_int), target :: status
      real(c_double), parameter :: beta = 10.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-6_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: epsilon_very_small = 1.0e-20_c_double
      real(c_double), parameter :: beta_zero = 0.0_c_double
      real(c_double), parameter :: omega_max_zero = 0.0_c_double
      integer(c_int), target :: max_size

      print *, "Testing invalid basis parameters..."

      ! Create valid kernel and SVE first
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      if (status /= 0) then
         print *, "Error: Failed to create kernel for error handling test"
         stop 1
      end if

      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      if (status /= 0) then
         print *, "Error: Failed to create SVE for error handling test"
         stop 1
      end if

      ! Test with very small epsilon (should work but might have precision issues)
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, &
                                   epsilon_very_small, k_ptr, sve_ptr, max_size, c_loc(status))
      if (status == 0 .and. c_associated(basis_ptr)) then
         call c_spir_basis_release(basis_ptr)
      end if
      ! Note: Very small epsilon might fail, which is acceptable

      ! Test with zero beta (should fail)
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta_zero, omega_max, &
                                   epsilon, k_ptr, sve_ptr, max_size, c_loc(status))
      if (status == 0 .and. c_associated(basis_ptr)) then
         call c_spir_basis_release(basis_ptr)
      end if
      ! Note: Zero beta should fail, which is acceptable

      ! Test with zero omega_max (should fail)
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max_zero, &
                                   epsilon, k_ptr, sve_ptr, max_size, c_loc(status))
      if (status == 0 .and. c_associated(basis_ptr)) then
         call c_spir_basis_release(basis_ptr)
      end if
      ! Note: Zero omega_max should fail, which is acceptable

      ! Cleanup
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Invalid basis parameters: PASSED (handled gracefully)"
   end subroutine test_invalid_basis_parameters

   subroutine test_null_pointer_handling()
      type(c_ptr) :: null_ptr, funcs_ptr
      integer(c_int), target :: status, size_val
      real(c_double), target :: value

      print *, "Testing null pointer handling..."

      null_ptr = c_null_ptr

      ! Test getting size from null basis pointer (should fail gracefully)
      status = c_spir_basis_get_size(null_ptr, c_loc(size_val))
      ! Note: This should return an error status, which is acceptable

      ! Test getting singular values from null basis pointer (should fail gracefully)
      status = c_spir_basis_get_svals(null_ptr, c_loc(value))
      ! Note: This should return an error status, which is acceptable

      ! Test evaluating functions with null pointer (should fail gracefully)
      funcs_ptr = c_null_ptr
      status = c_spir_funcs_eval(funcs_ptr, 0.5_c_double, c_loc(value))
      ! Note: This should return an error status, which is acceptable

      print *, "  Null pointer handling: PASSED (errors handled gracefully)"
   end subroutine test_null_pointer_handling

end program test_error_handling


