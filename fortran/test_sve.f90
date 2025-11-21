program test_sve
  use sparseir
  use iso_c_binding
  implicit none
  type(c_ptr) :: k_ptr, sve_ptr
  integer(c_int), target :: status
  real(c_double), parameter :: lambda = 10.0_c_double
  real(c_double), parameter :: epsilon = 1.0e-10_c_double
  
  k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
  print *, "Kernel created, status:", status
  
  ! Try with c_null_ptr
  sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, -1_c_int, c_null_ptr)
  print *, "SVE with c_null_ptr, associated:", c_associated(sve_ptr)
  
  ! Try with status pointer
  status = 0
  sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, -1_c_int, c_loc(status))
  print *, "SVE with status, associated:", c_associated(sve_ptr), "status:", status
end program
