module test
  use iso_c_binding
  interface
    ! Test 1: Simple continuation
    function test1(arg1, arg2, arg3) &
        bind(c, name="test1")
      use iso_c_binding
      integer(c_int), value :: arg1, arg2, arg3
      integer(c_int) :: test1
    end function
    
    ! Test 2: Continuation with & at start of next line
    function test2(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, &
        &arg9, arg10) &
        bind(c, name="test2")
      use iso_c_binding
      integer(c_int), value :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
      integer(c_int) :: test2
    end function
    
    ! Test 3: Continuation with space after &
    function test3(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, &
        & arg9, arg10) &
        bind(c, name="test3")
      use iso_c_binding
      integer(c_int), value :: arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10
      integer(c_int) :: test3
    end function
  end interface
end module
