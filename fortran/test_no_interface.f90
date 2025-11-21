module test
  use iso_c_binding
  implicit none
  ! No interface block - direct include
  include 'test_cbinding.inc'
end module
