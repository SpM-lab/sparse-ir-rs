module test
  use iso_c_binding
  implicit none
  interface
    subroutine test_sub(x) bind(c, name="test_sub")
      import :: c_int
      integer(c_int), value :: x
    end subroutine
  end interface
end module
