module test_snippet
  implicit none
  interface
function c_spir_basis_new_from_sve_and_inv_weight( &
    & statistics, beta, omega_max, epsilon, lambda, _ypower, _conv_radius, sve, &
    & inv_weight_funcs, max_size, status) &
    bind(c, name="spir_basis_new_from_sve_and_inv_weight")
  import :: c_double, c_int, c_ptr
  integer(c_int), value :: statistics
  real(c_double), value :: beta
  real(c_double), value :: omega_max
  real(c_double), value :: epsilon
  real(c_double), value :: lambda
  integer(c_int), value :: _ypower
  real(c_double), value :: _conv_radius
  type(c_ptr), value :: sve
  type(c_ptr), value :: inv_weight_funcs
  integer(c_int), value :: max_size
  type(c_ptr), value :: status
  type(c_ptr) :: c_spir_basis_new_from_sve_and_inv_weight

end function
  end interface
end module
