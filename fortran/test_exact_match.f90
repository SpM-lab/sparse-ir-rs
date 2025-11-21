module test
  use iso_c_binding
  interface
    function c_spir_basis_new_from_sve_and_inv_weight(statistics, beta, omega_max, epsilon, &
        & lambda, _ypower, _conv_radius, sve, inv_weight_funcs, max_size, status) &
        bind(c, name="spir_basis_new_from_sve_and_inv_weight")
      use iso_c_binding
      integer(c_int), value :: statistics
      real(c_double), value :: beta
      type(c_ptr) :: c_spir_basis_new_from_sve_and_inv_weight
    end function
  end interface
end module
