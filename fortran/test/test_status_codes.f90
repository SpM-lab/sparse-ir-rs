! Status codes of invalid input: one check per case
program test_status_codes
   use sparse_ir_c
   use, intrinsic :: iso_c_binding
   use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
   implicit none

   integer(c_int), parameter :: SUCCESS = 0_c_int
   integer(c_int), parameter :: INPUT_DIMENSION_MISMATCH = -3_c_int
   integer(c_int), parameter :: NOT_SUPPORTED = -5_c_int
   integer(c_int), parameter :: INVALID_ARGUMENT = -6_c_int

   print *, "Testing status codes"
   call test_from_matrix_nan()              ! S1
   call test_from_matrix_sizes()            ! S2
   call test_basis_size_and_accuracy()      ! S3
   call test_truncate_accuracy_and_size()   ! S4
   call test_basis_needs_a_unit_square()    ! S5
   call test_dlr_poles()                    ! S6
   call test_gauss_segments()               ! S7
   call test_sampling_matrix_nan()          ! S8
   call test_default_matsus_need_parity()   ! S9
   call test_dlr2ir_target_extent()         ! S10
   call test_regularizer_domain()           ! S11
   call test_dlr_default_points_are_nodes()    ! S12
   print *, "All status code tests passed!"

contains

   subroutine check(got, expected, what)
      integer(c_int), intent(in) :: got, expected
      character(len=*), intent(in) :: what
      if (got /= expected) then
         print *, "FAILED: ", what, " status ", got, " expected ", expected
         stop 1
      end if
   end subroutine check

   subroutine check_null(ptr, what)
      type(c_ptr), intent(in) :: ptr
      character(len=*), intent(in) :: what
      if (c_associated(ptr)) then
         print *, "FAILED: ", what, " returned a non-null pointer"
         stop 1
      end if
   end subroutine check_null

   ! A stand-in kernel matrix of nx * ny finite entries, in row-major order
   subroutine stand_in_kernel_matrix(nx, ny, k)
      integer, intent(in) :: nx, ny
      real(c_double), intent(out) :: k(nx*ny)
      integer :: i
      do i = 1, nx*ny
         k(i) = 0.1_c_double*(1.0_c_double + sin(real(i - 1, c_double)))
      end do
   end subroutine stand_in_kernel_matrix

   ! A fermionic basis with beta = 10, omega_max = 1 and the given max_size
   function fermionic_basis(max_size) result(basis)
      integer(c_int), intent(in) :: max_size
      type(c_ptr) :: basis, kernel
      integer(c_int), target :: status
      kernel = c_spir_logistic_kernel_new(10.0_c_double, c_loc(status))
      call check(status, SUCCESS, "logistic kernel of the stand-in basis")
      basis = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 10.0_c_double, &
         1.0_c_double, 1.0e-6_c_double, kernel, c_null_ptr, max_size, c_loc(status))
      call check(status, SUCCESS, "stand-in basis")
      call c_spir_kernel_release(kernel)
   end function fermionic_basis

   subroutine test_from_matrix_nan()
      real(c_double), target :: segs(3) = [-1.0_c_double, 0.0_c_double, 1.0_c_double]
      real(c_double), target :: k(16)
      integer(c_int), target :: status
      type(c_ptr) :: sve

      call stand_in_kernel_matrix(4, 4, k)
      k(1) = ieee_value(0.0_c_double, ieee_quiet_nan)
      sve = c_spir_sve_result_from_matrix(c_loc(k), c_null_ptr, 4_c_int, 4_c_int, &
         SPIR_ORDER_ROW_MAJOR, c_loc(segs), 2_c_int, c_loc(segs), 2_c_int, 2_c_int, &
         1.0e-8_c_double, c_loc(status))
      call check(status, INVALID_ARGUMENT, "from_matrix with a NaN entry")
      call check_null(sve, "from_matrix with a NaN entry")
   end subroutine test_from_matrix_nan

   subroutine test_from_matrix_sizes()
      real(c_double), target :: segs(3) = [-1.0_c_double, 0.0_c_double, 1.0_c_double]
      real(c_double), target, allocatable :: k(:)
      integer(c_int), target :: status
      type(c_ptr) :: sve
      integer :: nx

      do nx = 3, 5, 2
         allocate (k(nx*4))
         call stand_in_kernel_matrix(nx, 4, k)
         sve = c_spir_sve_result_from_matrix(c_loc(k), c_null_ptr, int(nx, c_int), 4_c_int, &
            SPIR_ORDER_ROW_MAJOR, c_loc(segs), 2_c_int, c_loc(segs), 2_c_int, 2_c_int, &
            1.0e-8_c_double, c_loc(status))
         call check(status, INVALID_ARGUMENT, "from_matrix with a row count other than the Gauss points")
         call check_null(sve, "from_matrix with a row count other than the Gauss points")
         deallocate (k)
      end do
   end subroutine test_from_matrix_sizes

   subroutine test_basis_size_and_accuracy()
      type(c_ptr) :: kernel, basis
      integer(c_int), target :: status, basis_size

      kernel = c_spir_logistic_kernel_new(10.0_c_double, c_loc(status))
      call check(status, SUCCESS, "logistic kernel")

      basis = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 10.0_c_double, 1.0_c_double, &
         1.0e-6_c_double, kernel, c_null_ptr, 0_c_int, c_loc(status))
      call check(status, INVALID_ARGUMENT, "a basis of max_size 0")
      call check_null(basis, "a basis of max_size 0")

      basis = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 10.0_c_double, 1.0_c_double, &
         1.0_c_double, kernel, c_null_ptr, -1_c_int, c_loc(status))
      call check(status, INVALID_ARGUMENT, "a basis of accuracy 1")
      call check_null(basis, "a basis of accuracy 1")

      basis = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 10.0_c_double, 1.0_c_double, &
         1.0e-6_c_double, kernel, c_null_ptr, 1_c_int, c_loc(status))
      call check(status, SUCCESS, "a basis of max_size 1")
      call check(c_spir_basis_get_size(basis, c_loc(basis_size)), SUCCESS, "the size of that basis")
      if (basis_size /= 1) then
         print *, "FAILED: a basis of max_size 1 has size ", basis_size
         stop 1
      end if

      call c_spir_basis_release(basis)
      call c_spir_kernel_release(kernel)
   end subroutine test_basis_size_and_accuracy

   subroutine test_truncate_accuracy_and_size()
      type(c_ptr) :: kernel, sve, truncated
      integer(c_int), target :: status

      kernel = c_spir_logistic_kernel_new(10.0_c_double, c_loc(status))
      call check(status, SUCCESS, "logistic kernel")
      sve = c_spir_sve_result_new(kernel, 1.0e-6_c_double, -1_c_int, -1_c_int, &
         SPIR_TWORK_AUTO, c_loc(status))
      call check(status, SUCCESS, "SVE of the logistic kernel")

      truncated = c_spir_sve_result_truncate(sve, 2.0_c_double, -1_c_int, c_loc(status))
      call check(status, INVALID_ARGUMENT, "truncate to an accuracy of 2")
      call check_null(truncated, "truncate to an accuracy of 2")

      truncated = c_spir_sve_result_truncate(sve, 1.0e-6_c_double, 0_c_int, c_loc(status))
      call check(status, INVALID_ARGUMENT, "truncate to a size of 0")
      call check_null(truncated, "truncate to a size of 0")

      call c_spir_sve_result_release(sve)
      call c_spir_kernel_release(kernel)
   end subroutine test_truncate_accuracy_and_size

   subroutine test_basis_needs_a_unit_square()
      real(c_double), target :: wide(3) = [-2.0_c_double, 0.0_c_double, 2.0_c_double]
      real(c_double), target :: k(16)
      integer(c_int), target :: status
      type(c_ptr) :: sve, kernel, basis

      call stand_in_kernel_matrix(4, 4, k)
      sve = c_spir_sve_result_from_matrix(c_loc(k), c_null_ptr, 4_c_int, 4_c_int, &
         SPIR_ORDER_ROW_MAJOR, c_loc(wide), 2_c_int, c_loc(wide), 2_c_int, 2_c_int, &
         1.0e-8_c_double, c_loc(status))
      call check(status, SUCCESS, "an SVE on [-2, 2]")

      kernel = c_spir_logistic_kernel_new(10.0_c_double, c_loc(status))
      call check(status, SUCCESS, "logistic kernel")
      basis = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 10.0_c_double, 1.0_c_double, &
         1.0e-6_c_double, kernel, sve, -1_c_int, c_loc(status))
      call check(status, INVALID_ARGUMENT, "a basis on an SVE that is not on the unit square")
      call check_null(basis, "a basis on an SVE that is not on the unit square")

      call c_spir_kernel_release(kernel)
      call c_spir_sve_result_release(sve)
   end subroutine test_basis_needs_a_unit_square

   subroutine test_dlr_poles()
      type(c_ptr) :: basis, dlr
      integer(c_int), target :: status
      real(c_double), target :: outside(2), nan_pole(2)

      basis = fermionic_basis(1_c_int)
      outside = [0.1_c_double, 2.0_c_double]
      nan_pole = [0.1_c_double, ieee_value(0.0_c_double, ieee_quiet_nan)]

      dlr = c_spir_dlr_new_with_poles(basis, 2_c_int, c_loc(outside), c_loc(status))
      call check(status, INVALID_ARGUMENT, "a pole outside [-omega_max, omega_max]")
      call check_null(dlr, "a pole outside [-omega_max, omega_max]")

      dlr = c_spir_dlr_new_with_poles(basis, 2_c_int, c_loc(nan_pole), c_loc(status))
      call check(status, INVALID_ARGUMENT, "a pole that is not a number")
      call check_null(dlr, "a pole that is not a number")

      call c_spir_basis_release(basis)
   end subroutine test_dlr_poles

   subroutine test_gauss_segments()
      real(c_double), target :: segments(2)
      real(c_double), target :: x(4), w(4)
      integer(c_int), target :: status
      integer(c_int) :: ret

      segments = [0.0_c_double, ieee_value(0.0_c_double, ieee_quiet_nan)]
      x = 0.0_c_double
      w = 0.0_c_double
      ret = c_spir_gauss_legendre_rule_piecewise_double(4_c_int, c_loc(segments), &
         1_c_int, c_loc(x), c_loc(w), c_loc(status))
      call check(ret, INVALID_ARGUMENT, "a Gauss rule on a NaN boundary (return value)")
      call check(status, INVALID_ARGUMENT, "a Gauss rule on a NaN boundary")
   end subroutine test_gauss_segments

   subroutine test_sampling_matrix_nan()
      real(c_double), target :: taus(2), tau_matrix(4)
      integer(c_int64_t), target :: ns(2)
      complex(c_double_complex), target :: matsu_matrix(4)
      integer(c_int), target :: status
      type(c_ptr) :: sampling

      taus = [0.1_c_double, 0.2_c_double]
      tau_matrix = [1.0_c_double, ieee_value(0.0_c_double, ieee_quiet_nan), &
                    1.0_c_double, 1.0_c_double]
      sampling = c_spir_tau_sampling_new_with_matrix(SPIR_ORDER_ROW_MAJOR, &
         SPIR_STATISTICS_FERMIONIC, 2_c_int, 2_c_int, c_loc(taus), c_loc(tau_matrix), &
         c_loc(status))
      call check(status, INVALID_ARGUMENT, "a tau sampling matrix with a NaN entry")
      call check_null(sampling, "a tau sampling matrix with a NaN entry")

      ns = [1_c_int64_t, 3_c_int64_t]
      matsu_matrix = [(1.0_c_double, 0.0_c_double), &
                      cmplx(ieee_value(0.0_c_double, ieee_quiet_nan), 0.0_c_double, c_double_complex), &
                      (1.0_c_double, 0.0_c_double), (1.0_c_double, 0.0_c_double)]
      sampling = c_spir_matsu_sampling_new_with_matrix(SPIR_ORDER_ROW_MAJOR, &
         SPIR_STATISTICS_FERMIONIC, 2_c_int, .false._c_bool, 2_c_int, c_loc(ns), &
         c_loc(matsu_matrix), c_loc(status))
      call check(status, INVALID_ARGUMENT, "a Matsubara sampling matrix with a NaN entry")
      call check_null(sampling, "a Matsubara sampling matrix with a NaN entry")
   end subroutine test_sampling_matrix_nan

   subroutine test_default_matsus_need_parity()
      real(c_double), target :: segs(3) = [-1.0_c_double, 0.0_c_double, 1.0_c_double]
      real(c_double), target :: k(16)
      integer(c_int), target :: status, n_points
      type(c_ptr) :: sve, kernel, basis

      call stand_in_kernel_matrix(4, 4, k)
      sve = c_spir_sve_result_from_matrix(c_loc(k), c_null_ptr, 4_c_int, 4_c_int, &
         SPIR_ORDER_ROW_MAJOR, c_loc(segs), 2_c_int, c_loc(segs), 2_c_int, 2_c_int, &
         1.0e-8_c_double, c_loc(status))
      call check(status, SUCCESS, "an SVE from a matrix")

      kernel = c_spir_logistic_kernel_new(10.0_c_double, c_loc(status))
      call check(status, SUCCESS, "logistic kernel")
      basis = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 10.0_c_double, 1.0_c_double, &
         1.0e-6_c_double, kernel, sve, -1_c_int, c_loc(status))
      call check(status, SUCCESS, "a basis on an SVE from a matrix")

      n_points = -12345_c_int
      call check(c_spir_basis_get_n_default_matsus(basis, .false._c_bool, c_loc(n_points)), &
         NOT_SUPPORTED, "the default Matsubara points of a basis without parity")
      if (n_points /= -12345_c_int) then
         print *, "FAILED: the number of points was written: ", n_points
         stop 1
      end if

      call c_spir_basis_release(basis)
      call c_spir_kernel_release(kernel)
      call c_spir_sve_result_release(sve)
   end subroutine test_default_matsus_need_parity

   subroutine test_dlr2ir_target_extent()
      type(c_ptr) :: basis, dlr
      integer(c_int), target :: status, n_poles
      integer(c_int), target :: input_dims(1)
      real(c_double), target, allocatable :: input(:), out(:)
      real(c_double), parameter :: sentinel = -12345.5_c_double
      integer :: i

      basis = fermionic_basis(1_c_int)
      dlr = c_spir_dlr_new(basis, c_loc(status))
      call check(status, SUCCESS, "a DLR of the stand-in basis")
      call check(c_spir_dlr_get_npoles(dlr, c_loc(n_poles)), SUCCESS, "the number of poles")

      ! One coefficient too many along the transformed axis
      input_dims(1) = n_poles + 1_c_int
      allocate (input(n_poles + 1), out(n_poles + 1))
      input = 1.0_c_double
      out = sentinel
      call check(c_spir_dlr2ir_dd(dlr, c_null_ptr, SPIR_ORDER_ROW_MAJOR, 1_c_int, &
         c_loc(input_dims), 0_c_int, c_loc(input), c_loc(out)), &
         INPUT_DIMENSION_MISMATCH, "a DLR transform of a wrong target extent")
      do i = 1, size(out)
         if (out(i) /= sentinel) then
            print *, "FAILED: the output was written at ", i
            stop 1
         end if
      end do

      deallocate (input, out)
      call c_spir_basis_release(dlr)
      call c_spir_basis_release(basis)
   end subroutine test_dlr2ir_target_extent

   subroutine test_regularizer_domain()
      type(c_ptr) :: k1, b1, u, k10, sve, basis
      integer(c_int), target :: status

      ! u of a basis with omega_max = 1 is defined on [-1, 1], not at
      ! omega_max / 2 = 5 of the basis built below
      k1 = c_spir_logistic_kernel_new(1.0_c_double, c_loc(status))
      call check(status, SUCCESS, "a kernel of lambda 1")
      b1 = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, 1.0_c_double, 1.0_c_double, &
         1.0e-6_c_double, k1, c_null_ptr, -1_c_int, c_loc(status))
      call check(status, SUCCESS, "a basis of omega_max 1")
      u = c_spir_basis_get_u(b1, c_loc(status))
      call check(status, SUCCESS, "u of that basis")

      k10 = c_spir_logistic_kernel_new(10.0_c_double, c_loc(status))
      call check(status, SUCCESS, "a kernel of lambda 10")
      sve = c_spir_sve_result_new(k10, 1.0e-6_c_double, -1_c_int, -1_c_int, &
         SPIR_TWORK_AUTO, c_loc(status))
      call check(status, SUCCESS, "an SVE of lambda 10")
      basis = c_spir_basis_new_from_sve_and_regularizer(SPIR_STATISTICS_FERMIONIC, &
         1.0_c_double, 10.0_c_double, 1.0e-6_c_double, 10.0_c_double, 0_c_int, &
         0.0_c_double, sve, u, -1_c_int, c_loc(status))
      call check(status, INVALID_ARGUMENT, "a regularizer undefined at omega_max / 2")
      call check_null(basis, "a regularizer undefined at omega_max / 2")

      call c_spir_sve_result_release(sve)
      call c_spir_kernel_release(k10)
      call c_spir_funcs_release(u)
      call c_spir_basis_release(b1)
      call c_spir_kernel_release(k1)
   end subroutine test_regularizer_domain

   subroutine test_dlr_default_points_are_nodes()
      type(c_ptr) :: basis, dlr
      integer(c_int), target :: status, n_poles, n_taus, n_matsus

      basis = fermionic_basis(1_c_int)
      dlr = c_spir_dlr_new(basis, c_loc(status))
      call check(status, SUCCESS, "a DLR of the stand-in basis")
      call check(c_spir_dlr_get_npoles(dlr, c_loc(n_poles)), SUCCESS, &
         "the number of poles of a DLR")

      ! One interpolation node per pole (there were none up to 0.10)
      n_taus = -1_c_int
      call check(c_spir_basis_get_n_default_taus(dlr, c_loc(n_taus)), SUCCESS, &
         "the number of default tau points of a DLR")
      if (n_taus /= n_poles) then
         print *, "FAILED: a DLR reported ", n_taus, " default tau points for ", n_poles, " poles"
         stop 1
      end if

      n_matsus = -1_c_int
      call check(c_spir_basis_get_n_default_matsus(dlr, .false._c_bool, c_loc(n_matsus)), &
         SUCCESS, "the number of default Matsubara points of a DLR")
      if (n_matsus /= n_poles) then
         print *, "FAILED: a DLR reported ", n_matsus, " default Matsubara points for ", &
            n_poles, " poles"
         stop 1
      end if

      call c_spir_basis_release(dlr)
      call c_spir_basis_release(basis)
   end subroutine test_dlr_default_points_are_nodes

end program test_status_codes
