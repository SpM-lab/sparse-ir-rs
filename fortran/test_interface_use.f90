module test
  use iso_c_binding
  interface
    subroutine test_sub(x) bind(c, name="test_sub")
      use iso_c_binding
      integer(c_int), value :: x
    end subroutine
  end interface
end module
