module test_problem
  use iso_c_binding
  implicit none
  interface
    function foo( &
        & a, b, c, d, e, f, g, &
        & h, i) &
        bind(c, name="foo")
      import :: c_int
      integer(c_int), value :: a, b, c, d, e, f, g, h, i
      integer(c_int) :: foo
    end function
  end interface
end module
