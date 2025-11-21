module test_snippet2
  use iso_c_binding
  implicit none
  interface
function func( &
    & statistics, beta, omega_max, epsilon, lambda, _ypower, _conv_radius, sve, &
    & inv_weight_funcs, max_size, status) &
    bind(c, name="func")
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
  type(c_ptr) :: func

end function
  end interface
end module
