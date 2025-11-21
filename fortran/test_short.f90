module test
  use iso_c_binding
  interface
    function short_func(arg1, arg2, arg3) &
        bind(c, name="short_func")
      use iso_c_binding
      integer(c_int), value :: arg1, arg2, arg3
      integer(c_int) :: short_func
    end function
  end interface
end module
