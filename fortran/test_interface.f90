module test
  use iso_c_binding
  implicit none
  interface
    function test_func(x) bind(c, name="test_func")
      import :: c_int
      integer(c_int), value :: x
      integer(c_int) :: test_func
    end function
  end interface
end module
