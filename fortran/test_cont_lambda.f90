module test_lambda
  use iso_c_binding
  interface
    function test(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, &
        lambda, arg10) &
        bind(c, name="test")
      use iso_c_binding
      integer(c_int), value :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
      integer(c_int) :: test
    end function
  end interface
end module
