module test
  use iso_c_binding
  interface
    function long_function_name(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11) &
        bind(c, name="long_function_name")
      use iso_c_binding
      integer(c_int), value :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11
      integer(c_int) :: long_function_name
    end function
  end interface
end module
