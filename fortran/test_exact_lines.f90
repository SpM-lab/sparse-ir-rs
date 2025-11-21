module test_exact_lines
  use iso_c_binding
  interface
    function func( &
        & statistics, beta, omega_max, epsilon, lambda, _ypower, _conv_radius, sve, &
        & inv_weight_funcs, max_size, status) &
        bind(c, name="func")
      import :: c_double, c_int, c_ptr
      integer(c_int), value :: statistics
      real(c_double), value :: beta
      type(c_ptr) :: func
    end function
  end interface
end module
