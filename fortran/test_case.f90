module test_case
  use iso_c_binding
  implicit none
  interface
    function foo( &
        & arg1, arg2, arg3, &
        & arg4) &
        bind(c, name="foo")
      import :: c_int
      integer(c_int), value :: arg1, arg2, arg3, arg4
      integer(c_int) :: foo
    end function
  end interface
end module
