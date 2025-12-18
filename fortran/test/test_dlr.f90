! Test DLR functionality: custom poles, multi-dimensional conversions
program test_dlr
   use sparse_ir_c
   use, intrinsic :: iso_c_binding
   implicit none

   print *, "======================================"
   print *, "Testing DLR functionality"
   print *, "======================================"

   call test_dlr_construction()
   call test_dlr_with_custom_poles()
   call test_dlr_ir_conversion_1d()
   call test_dlr_ir_conversion_2d()

   ! Test with different statistics
   call test_dlr_with_statistics(SPIR_STATISTICS_FERMIONIC, "Fermionic")
   call test_dlr_with_statistics(SPIR_STATISTICS_BOSONIC, "Bosonic")

   ! Test multi-dimensional conversions (3D, 4D)
   call test_dlr_ir_conversion_3d()
   call test_dlr_ir_conversion_4d()

   ! Test with different target_dim values
   call test_dlr_ir_conversion_3d_all_target_dims()
   call test_dlr_ir_conversion_4d_all_target_dims()

   ! Test complex conversions
   call test_dlr_ir_conversion_complex_1d()
   call test_dlr_ir_conversion_complex_2d()

   print *, "======================================"
   print *, "All DLR tests passed!"
   print *, "======================================"

contains

   subroutine test_dlr_construction()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr, dlr_with_poles_ptr
      integer(c_int), target :: status, npoles, npoles_with_poles
      real(c_double), allocatable, target :: default_poles(:), poles(:), poles_with_poles(:)
      real(c_double), parameter :: beta = 10000.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-12_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size
      integer :: i

      print *, "Testing DLR construction..."

      ! Create IR basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get default poles
      status = c_spir_basis_get_n_default_ws(basis_ptr, c_loc(npoles))
      if (status /= 0 .or. npoles <= 0) then
         print *, "Error: Failed to get number of default poles"
         stop 1
      end if

      allocate(default_poles(npoles))
      status = c_spir_basis_get_default_ws(basis_ptr, c_loc(default_poles))
      if (status /= 0) then
         print *, "Error: Failed to get default poles"
         stop 1
      end if

      ! Create DLR using default poles
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      if (status /= 0 .or. .not. c_associated(dlr_ptr)) then
         print *, "Error: Failed to create DLR"
         stop 1
      end if

      ! Get number of poles
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))
      if (status /= 0) then
         print *, "Error: Failed to get number of poles"
         stop 1
      end if

      ! Get poles
      allocate(poles(npoles))
      status = c_spir_dlr_get_poles(dlr_ptr, c_loc(poles))
      if (status /= 0) then
         print *, "Error: Failed to get poles"
         stop 1
      end if

      ! Create DLR with custom poles (same as default)
      dlr_with_poles_ptr = c_spir_dlr_new_with_poles(basis_ptr, npoles, c_loc(default_poles), c_loc(status))
      if (status /= 0 .or. .not. c_associated(dlr_with_poles_ptr)) then
         print *, "Error: Failed to create DLR with custom poles"
         stop 1
      end if

      ! Get number of poles from DLR with custom poles
      status = c_spir_dlr_get_npoles(dlr_with_poles_ptr, c_loc(npoles_with_poles))
      if (status /= 0) then
         print *, "Error: Failed to get number of poles from DLR with custom poles"
         stop 1
      end if

      if (npoles_with_poles /= npoles) then
         print *, "Error: Number of poles mismatch"
         stop 1
      end if

      ! Get poles from DLR with custom poles
      allocate(poles_with_poles(npoles_with_poles))
      status = c_spir_dlr_get_poles(dlr_with_poles_ptr, c_loc(poles_with_poles))
      if (status /= 0) then
         print *, "Error: Failed to get poles from DLR with custom poles"
         stop 1
      end if

      ! Verify poles match (within tolerance)
      do i = 1, min(npoles, npoles_with_poles)
         if (abs(poles(i) - poles_with_poles(i)) > 1.0e-14_c_double) then
            print *, "Error: Poles mismatch at index", i
            stop 1
         end if
      end do

      ! Cleanup
      deallocate(default_poles, poles, poles_with_poles)
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(dlr_with_poles_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  DLR construction: PASSED"
   end subroutine test_dlr_construction

   subroutine test_dlr_with_custom_poles()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_custom_ptr
      integer(c_int), target :: status, npoles_default, npoles_custom
      real(c_double), allocatable, target :: default_poles(:), custom_poles(:), poles_custom(:)
      real(c_double), parameter :: beta = 1000.0_c_double
      real(c_double), parameter :: omega_max = 2.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size
      integer :: i

      print *, "Testing DLR with custom poles..."

      ! Create IR basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get default poles
      status = c_spir_basis_get_n_default_ws(basis_ptr, c_loc(npoles_default))
      if (status /= 0 .or. npoles_default <= 0) then
         print *, "Error: Failed to get number of default poles"
         stop 1
      end if

      allocate(default_poles(npoles_default))
      status = c_spir_basis_get_default_ws(basis_ptr, c_loc(default_poles))
      if (status /= 0) then
         print *, "Error: Failed to get default poles"
         stop 1
      end if

      ! Use subset of poles (every other pole)
      if (npoles_default >= 2) then
         npoles_custom = (npoles_default + 1) / 2
         allocate(custom_poles(npoles_custom))
         do i = 1, npoles_custom
            custom_poles(i) = default_poles(2*i - 1)
         end do

         ! Create DLR with custom poles
         dlr_custom_ptr = c_spir_dlr_new_with_poles(basis_ptr, npoles_custom, c_loc(custom_poles), c_loc(status))
         if (status /= 0 .or. .not. c_associated(dlr_custom_ptr)) then
            print *, "Error: Failed to create DLR with custom poles"
            stop 1
         end if

         ! Get number of poles
         status = c_spir_dlr_get_npoles(dlr_custom_ptr, c_loc(npoles_custom))
         if (status /= 0) then
            print *, "Error: Failed to get number of custom poles"
            stop 1
         end if

         ! Get poles and verify they are reasonable
         allocate(poles_custom(npoles_custom))
         status = c_spir_dlr_get_poles(dlr_custom_ptr, c_loc(poles_custom))
         if (status /= 0) then
            print *, "Error: Failed to get custom poles"
            stop 1
         end if

         ! Verify poles are finite
         do i = 1, npoles_custom
            if (.not. (abs(poles_custom(i)) < huge(1.0_c_double))) then
               print *, "Error: Invalid pole at index", i
               stop 1
            end if
         end do

         ! Cleanup
         deallocate(custom_poles, poles_custom)
         call c_spir_basis_release(dlr_custom_ptr)
      end if

      ! Cleanup
      deallocate(default_poles)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  DLR with custom poles: PASSED"
   end subroutine test_dlr_with_custom_poles

   subroutine test_dlr_ir_conversion_1d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      real(c_double), allocatable, target :: dlr_coeffs(:), ir_coeffs(:)
      real(c_double), parameter :: beta = 100.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size
      integer :: i
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 1D DLR-IR conversion..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Create test DLR coefficients
         allocate(dlr_coeffs(npoles))
         do i = 1, npoles
            call random_number(dlr_coeffs(i))
            dlr_coeffs(i) = (dlr_coeffs(i) - 0.5_c_double) * 2.0_c_double
         end do

         ! Convert DLR to IR
         allocate(ir_coeffs(ir_size))
         status = c_spir_dlr2ir_dd(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                   1_c_int, c_loc(npoles), 0_c_int, c_loc(dlr_coeffs), c_loc(ir_coeffs))
         if (status /= 0) then
            print *, "Error: Failed to convert DLR to IR"
            stop 1
         end if

         ! Verify that we got some non-zero IR coefficients
         if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
            print *, "Error: All IR coefficients are zero"
            stop 1
         end if

         ! Cleanup
         deallocate(dlr_coeffs, ir_coeffs)
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  1D DLR-IR conversion: PASSED"
   end subroutine test_dlr_ir_conversion_1d

   subroutine test_dlr_ir_conversion_2d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      integer(c_int), target :: input_dims(2)
      real(c_double), allocatable, target :: dlr_coeffs(:, :), ir_coeffs(:, :)
      real(c_double), parameter :: beta = 50.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: extra_size = 3
      integer(c_int), target :: max_size
      integer :: i, j
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 2D DLR-IR conversion..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Create test 2D DLR coefficients (column-major order)
         allocate(dlr_coeffs(npoles, extra_size))
         do j = 1, extra_size
            do i = 1, npoles
               call random_number(dlr_coeffs(i, j))
               dlr_coeffs(i, j) = (dlr_coeffs(i, j) - 0.5_c_double) * 2.0_c_double
            end do
         end do

         ! Convert DLR to IR (target_dim = 0)
         allocate(ir_coeffs(ir_size, extra_size))
         input_dims = [npoles, extra_size]
         status = c_spir_dlr2ir_dd(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                   2_c_int, c_loc(input_dims), 0_c_int, c_loc(dlr_coeffs), c_loc(ir_coeffs))
         if (status /= 0) then
            print *, "Error: Failed to convert 2D DLR to IR"
            stop 1
         end if

         ! Verify that we got some non-zero IR coefficients
         if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
            print *, "Error: All IR coefficients are zero"
            stop 1
         end if

         ! Cleanup
         deallocate(dlr_coeffs, ir_coeffs)
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  2D DLR-IR conversion: PASSED"
   end subroutine test_dlr_ir_conversion_2d

   subroutine test_dlr_with_statistics(statistics, case_name)
      integer(c_int32_t), intent(in) :: statistics
      character(len=*), intent(in) :: case_name
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      real(c_double), parameter :: beta = 100.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size

      print *, "Testing DLR construction with ", case_name, " statistics..."

      ! Create IR basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(statistics, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! DLR creation succeeded
      else
         print *, "Warning: DLR has zero poles or IR size is zero"
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  DLR construction (", case_name, "): PASSED"
   end subroutine test_dlr_with_statistics

   subroutine test_dlr_ir_conversion_3d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      integer(c_int), target :: input_dims(3)
      real(c_double), allocatable, target :: dlr_coeffs(:, :, :), ir_coeffs(:, :, :)
      real(c_double), parameter :: beta = 50.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: d1 = 2, d2 = 3
      integer(c_int), target :: max_size
      integer :: i, j, k
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 3D DLR-IR conversion..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Create test 3D DLR coefficients (column-major order)
         allocate(dlr_coeffs(npoles, d1, d2))
         do k = 1, d2
            do j = 1, d1
               do i = 1, npoles
                  call random_number(dlr_coeffs(i, j, k))
                  dlr_coeffs(i, j, k) = (dlr_coeffs(i, j, k) - 0.5_c_double) * 2.0_c_double
               end do
            end do
         end do

         ! Convert DLR to IR (target_dim = 0)
         allocate(ir_coeffs(ir_size, d1, d2))
         input_dims = [npoles, d1, d2]
         status = c_spir_dlr2ir_dd(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                   3_c_int, c_loc(input_dims), 0_c_int, c_loc(dlr_coeffs), c_loc(ir_coeffs))
         if (status /= 0) then
            print *, "Error: Failed to convert 3D DLR to IR"
            stop 1
         end if

         ! Verify that we got some non-zero IR coefficients
         if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
            print *, "Error: All IR coefficients are zero"
            stop 1
         end if

         ! Cleanup
         deallocate(dlr_coeffs, ir_coeffs)
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  3D DLR-IR conversion: PASSED"
   end subroutine test_dlr_ir_conversion_3d

   subroutine test_dlr_ir_conversion_4d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      integer(c_int), target :: input_dims(4)
      real(c_double), allocatable, target :: dlr_coeffs(:, :, :, :), ir_coeffs(:, :, :, :)
      real(c_double), parameter :: beta = 50.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: d1 = 2, d2 = 2, d3 = 2
      integer(c_int), target :: max_size
      integer :: i, j, k, l
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 4D DLR-IR conversion..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Create test 4D DLR coefficients (column-major order)
         allocate(dlr_coeffs(npoles, d1, d2, d3))
         do l = 1, d3
            do k = 1, d2
               do j = 1, d1
                  do i = 1, npoles
                     call random_number(dlr_coeffs(i, j, k, l))
                     dlr_coeffs(i, j, k, l) = (dlr_coeffs(i, j, k, l) - 0.5_c_double) * 2.0_c_double
                  end do
               end do
            end do
         end do

         ! Convert DLR to IR (target_dim = 0)
         allocate(ir_coeffs(ir_size, d1, d2, d3))
         input_dims = [npoles, d1, d2, d3]
         status = c_spir_dlr2ir_dd(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                   4_c_int, c_loc(input_dims), 0_c_int, c_loc(dlr_coeffs), c_loc(ir_coeffs))
         if (status /= 0) then
            print *, "Error: Failed to convert 4D DLR to IR"
            stop 1
         end if

         ! Verify that we got some non-zero IR coefficients
         if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
            print *, "Error: All IR coefficients are zero"
            stop 1
         end if

         ! Cleanup
         deallocate(dlr_coeffs, ir_coeffs)
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  4D DLR-IR conversion: PASSED"
   end subroutine test_dlr_ir_conversion_4d

   subroutine test_dlr_ir_conversion_3d_all_target_dims()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      integer(c_int), target :: input_dims(3)
      real(c_double), allocatable, target :: dlr_coeffs(:, :, :), ir_coeffs(:, :, :)
      real(c_double), parameter :: beta = 50.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: d1 = 2, d2 = 3
      integer(c_int), target :: max_size
      integer :: i, j, k
      integer(c_int) :: target_dim
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 3D DLR-IR conversion with all target_dim values..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Test each target_dim (0, 1, 2)
         do target_dim = 0, 2
            ! Allocate arrays with different layouts based on target_dim
            if (target_dim == 0) then
               allocate(dlr_coeffs(npoles, d1, d2))
               allocate(ir_coeffs(ir_size, d1, d2))
               input_dims = [npoles, d1, d2]
            else if (target_dim == 1) then
               allocate(dlr_coeffs(d1, npoles, d2))
               allocate(ir_coeffs(d1, ir_size, d2))
               input_dims = [d1, npoles, d2]
            else ! target_dim == 2
               allocate(dlr_coeffs(d1, d2, npoles))
               allocate(ir_coeffs(d1, d2, ir_size))
               input_dims = [d1, d2, npoles]
            end if

            ! Generate random coefficients
            do k = 1, size(dlr_coeffs, 3)
               do j = 1, size(dlr_coeffs, 2)
                  do i = 1, size(dlr_coeffs, 1)
                     call random_number(dlr_coeffs(i, j, k))
                     dlr_coeffs(i, j, k) = (dlr_coeffs(i, j, k) - 0.5_c_double) * 2.0_c_double
                  end do
               end do
            end do

            ! Convert DLR to IR
            status = c_spir_dlr2ir_dd(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      3_c_int, c_loc(input_dims), target_dim, c_loc(dlr_coeffs), c_loc(ir_coeffs))
            if (status /= 0) then
               print *, "Error: Failed to convert 3D DLR to IR (target_dim=", target_dim, ")"
               stop 1
            end if

            ! Verify that we got some non-zero IR coefficients
            if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
               print *, "Error: All IR coefficients are zero (target_dim=", target_dim, ")"
               stop 1
            end if

            ! Cleanup
            deallocate(dlr_coeffs, ir_coeffs)
         end do
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  3D DLR-IR conversion (all target_dim): PASSED"
   end subroutine test_dlr_ir_conversion_3d_all_target_dims

   subroutine test_dlr_ir_conversion_4d_all_target_dims()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      integer(c_int), target :: input_dims(4)
      real(c_double), allocatable, target :: dlr_coeffs(:, :, :, :), ir_coeffs(:, :, :, :)
      real(c_double), parameter :: beta = 50.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: d1 = 2, d2 = 2, d3 = 2
      integer(c_int), target :: max_size
      integer :: i, j, k, l
      integer(c_int) :: target_dim
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 4D DLR-IR conversion with all target_dim values..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Test each target_dim (0, 1, 2, 3)
         do target_dim = 0, 3
            ! Allocate arrays with different layouts based on target_dim
            if (target_dim == 0) then
               allocate(dlr_coeffs(npoles, d1, d2, d3))
               allocate(ir_coeffs(ir_size, d1, d2, d3))
               input_dims = [npoles, d1, d2, d3]
            else if (target_dim == 1) then
               allocate(dlr_coeffs(d1, npoles, d2, d3))
               allocate(ir_coeffs(d1, ir_size, d2, d3))
               input_dims = [d1, npoles, d2, d3]
            else if (target_dim == 2) then
               allocate(dlr_coeffs(d1, d2, npoles, d3))
               allocate(ir_coeffs(d1, d2, ir_size, d3))
               input_dims = [d1, d2, npoles, d3]
            else ! target_dim == 3
               allocate(dlr_coeffs(d1, d2, d3, npoles))
               allocate(ir_coeffs(d1, d2, d3, ir_size))
               input_dims = [d1, d2, d3, npoles]
            end if

            ! Generate random coefficients
            do l = 1, size(dlr_coeffs, 4)
               do k = 1, size(dlr_coeffs, 3)
                  do j = 1, size(dlr_coeffs, 2)
                     do i = 1, size(dlr_coeffs, 1)
                        call random_number(dlr_coeffs(i, j, k, l))
                        dlr_coeffs(i, j, k, l) = (dlr_coeffs(i, j, k, l) - 0.5_c_double) * 2.0_c_double
                     end do
                  end do
               end do
            end do

            ! Convert DLR to IR
            status = c_spir_dlr2ir_dd(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      4_c_int, c_loc(input_dims), target_dim, c_loc(dlr_coeffs), c_loc(ir_coeffs))
            if (status /= 0) then
               print *, "Error: Failed to convert 4D DLR to IR (target_dim=", target_dim, ")"
               stop 1
            end if

            ! Verify that we got some non-zero IR coefficients
            if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
               print *, "Error: All IR coefficients are zero (target_dim=", target_dim, ")"
               stop 1
            end if

            ! Cleanup
            deallocate(dlr_coeffs, ir_coeffs)
         end do
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  4D DLR-IR conversion (all target_dim): PASSED"
   end subroutine test_dlr_ir_conversion_4d_all_target_dims

   subroutine test_dlr_ir_conversion_complex_1d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      complex(c_double_complex), allocatable, target :: dlr_coeffs(:), ir_coeffs(:)
      real(c_double), parameter :: beta = 100.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size
      integer :: i
      integer :: seed_size
      integer, allocatable :: seed(:)
      real(c_double) :: r_real, r_imag

      print *, "Testing complex 1D DLR-IR conversion..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Create complex test DLR coefficients
         allocate(dlr_coeffs(npoles))
         do i = 1, npoles
            call random_number(r_real)
            call random_number(r_imag)
            dlr_coeffs(i) = cmplx((r_real - 0.5_c_double) * 2.0_c_double, &
                                 (r_imag - 0.5_c_double) * 2.0_c_double, kind=c_double_complex)
         end do

         ! Convert DLR to IR
         allocate(ir_coeffs(ir_size))
         status = c_spir_dlr2ir_zz(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                   1_c_int, c_loc(npoles), 0_c_int, c_loc(dlr_coeffs), c_loc(ir_coeffs))
         if (status /= 0) then
            print *, "Error: Failed to convert complex 1D DLR to IR"
            stop 1
         end if

         ! Verify that we got some non-zero IR coefficients
         if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
            print *, "Error: All IR coefficients are zero"
            stop 1
         end if

         ! Cleanup
         deallocate(dlr_coeffs, ir_coeffs)
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Complex 1D DLR-IR conversion: PASSED"
   end subroutine test_dlr_ir_conversion_complex_1d

   subroutine test_dlr_ir_conversion_complex_2d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, dlr_ptr
      integer(c_int), target :: status, npoles, ir_size
      integer(c_int), target :: input_dims(2)
      complex(c_double_complex), allocatable, target :: dlr_coeffs(:, :), ir_coeffs(:, :)
      real(c_double), parameter :: beta = 50.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-8_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: extra_size = 3
      integer(c_int), target :: max_size
      integer :: i, j
      integer :: seed_size
      integer, allocatable :: seed(:)
      real(c_double) :: r_real, r_imag

      print *, "Testing complex 2D DLR-IR conversion..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create IR basis and DLR
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(ir_size))
      dlr_ptr = c_spir_dlr_new(basis_ptr, c_loc(status))
      status = c_spir_dlr_get_npoles(dlr_ptr, c_loc(npoles))

      if (npoles > 0 .and. ir_size > 0) then
         ! Create complex test 2D DLR coefficients (column-major order)
         allocate(dlr_coeffs(npoles, extra_size))
         do j = 1, extra_size
            do i = 1, npoles
               call random_number(r_real)
               call random_number(r_imag)
               dlr_coeffs(i, j) = cmplx((r_real - 0.5_c_double) * 2.0_c_double, &
                                       (r_imag - 0.5_c_double) * 2.0_c_double, kind=c_double_complex)
            end do
         end do

         ! Convert DLR to IR (target_dim = 0)
         allocate(ir_coeffs(ir_size, extra_size))
         input_dims = [npoles, extra_size]
         status = c_spir_dlr2ir_zz(dlr_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                   2_c_int, c_loc(input_dims), 0_c_int, c_loc(dlr_coeffs), c_loc(ir_coeffs))
         if (status /= 0) then
            print *, "Error: Failed to convert complex 2D DLR to IR"
            stop 1
         end if

         ! Verify that we got some non-zero IR coefficients
         if (maxval(abs(ir_coeffs)) < 1.0e-15_c_double) then
            print *, "Error: All IR coefficients are zero"
            stop 1
         end if

         ! Cleanup
         deallocate(dlr_coeffs, ir_coeffs)
      end if

      ! Cleanup
      call c_spir_basis_release(dlr_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Complex 2D DLR-IR conversion: PASSED"
   end subroutine test_dlr_ir_conversion_complex_2d

end program test_dlr

