! Test sampling functionality: condition number, 1D/multi-dimensional, complex
program test_sampling
   use sparse_ir_c
   use, intrinsic :: iso_c_binding
   implicit none

   print *, "======================================"
   print *, "Testing sampling functionality"
   print *, "======================================"

   call test_tau_sampling_basic()
   call test_matsubara_sampling_basic()
   call test_sampling_condition_number()
   call test_sampling_1d_roundtrip()
   call test_sampling_2d_roundtrip()

   ! Test with different statistics and positive_only combinations
   call test_sampling_with_statistics(SPIR_STATISTICS_FERMIONIC, .FALSE., "Fermionic, all frequencies")
   call test_sampling_with_statistics(SPIR_STATISTICS_FERMIONIC, .TRUE., "Fermionic, positive only")
   call test_sampling_with_statistics(SPIR_STATISTICS_BOSONIC, .FALSE., "Bosonic, all frequencies")
   call test_sampling_with_statistics(SPIR_STATISTICS_BOSONIC, .TRUE., "Bosonic, positive only")

   ! Test multi-dimensional sampling (3D, 4D)
   call test_sampling_3d_roundtrip()
   call test_sampling_4d_roundtrip()

   ! Test with different target_dim values
   call test_sampling_3d_all_target_dims()
   call test_sampling_4d_all_target_dims()

   ! Test complex sampling
   call test_sampling_complex_1d()
   call test_sampling_complex_2d()

   print *, "======================================"
   print *, "All sampling tests passed!"
   print *, "======================================"

contains

   subroutine test_tau_sampling_basic()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, npoints
      real(c_double), allocatable, target :: taus(:), retrieved_taus(:)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-15_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size

      print *, "Testing tau sampling basics..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      if (status /= 0 .or. ntau <= 0) then
         print *, "Error: Failed to get number of tau points"
         stop 1
      end if

      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))
      if (status /= 0) then
         print *, "Error: Failed to get tau points"
         stop 1
      end if

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))
      if (status /= 0 .or. .not. c_associated(tau_sampling_ptr)) then
         print *, "Error: Failed to create tau sampling"
         stop 1
      end if

      ! Get number of points
      status = c_spir_sampling_get_npoints(tau_sampling_ptr, c_loc(npoints))
      if (status /= 0 .or. npoints <= 0) then
         print *, "Error: Failed to get number of sampling points"
         stop 1
      end if

      ! Get tau points
      allocate(retrieved_taus(npoints))
      status = c_spir_sampling_get_taus(tau_sampling_ptr, c_loc(retrieved_taus))
      if (status /= 0) then
         print *, "Error: Failed to get tau points from sampling"
         stop 1
      end if

      ! Verify tau points match
      if (npoints /= ntau) then
         print *, "Error: Number of points mismatch"
         stop 1
      end if

      ! Cleanup
      deallocate(taus, retrieved_taus)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Tau sampling basics: PASSED"
   end subroutine test_tau_sampling_basic

   subroutine test_matsubara_sampling_basic()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, matsu_sampling_ptr
      integer(c_int), target :: status, nmatsu, npoints
      integer(c_int64_t), allocatable, target :: matsus(:), retrieved_matsus(:)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-15_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), parameter :: positive_only = 0
      integer(c_int), target :: max_size

      print *, "Testing Matsubara sampling basics..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get default Matsubara points
      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only, c_loc(nmatsu))
      if (status /= 0 .or. nmatsu <= 0) then
         print *, "Error: Failed to get number of Matsubara points"
         stop 1
      end if

      allocate(matsus(nmatsu))
      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only, c_loc(matsus))
      if (status /= 0) then
         print *, "Error: Failed to get Matsubara points"
         stop 1
      end if

      ! Create Matsubara sampling
      matsu_sampling_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only, nmatsu, &
                                                     c_loc(matsus), c_loc(status))
      if (status /= 0 .or. .not. c_associated(matsu_sampling_ptr)) then
         print *, "Error: Failed to create Matsubara sampling"
         stop 1
      end if

      ! Get number of points
      status = c_spir_sampling_get_npoints(matsu_sampling_ptr, c_loc(npoints))
      if (status /= 0 .or. npoints <= 0) then
         print *, "Error: Failed to get number of sampling points"
         stop 1
      end if

      ! Cleanup
      deallocate(matsus)
      call c_spir_sampling_release(matsu_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Matsubara sampling basics: PASSED"
   end subroutine test_matsubara_sampling_basic

   subroutine test_sampling_condition_number()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau
      real(c_double), allocatable, target :: taus(:)
      real(c_double), target :: cond_num
      real(c_double), parameter :: beta = 100.0_c_double
      real(c_double), parameter :: omega_max = 1.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size

      print *, "Testing sampling condition number..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Get condition number
      status = c_spir_sampling_get_cond_num(tau_sampling_ptr, c_loc(cond_num))
      if (status /= 0) then
         print *, "Error: Failed to get condition number"
         stop 1
      end if

      ! Condition number should be > 1.0 for realistic parameters
      if (cond_num <= 1.0_c_double) then
         print *, "Error: Condition number should be > 1.0, got:", cond_num
         stop 1
      end if

      ! Cleanup
      deallocate(taus)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Sampling condition number: PASSED (cond_num = ", cond_num, ")"
   end subroutine test_sampling_condition_number

   subroutine test_sampling_1d_roundtrip()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, basis_size
      real(c_double), allocatable, target :: taus(:)
      real(c_double), allocatable, target :: coeffs(:), eval_output(:), fit_output(:)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-12_c_double
      integer(c_int), target :: max_size
      integer :: i
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 1D sampling roundtrip..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Allocate arrays
      allocate(coeffs(basis_size))
      allocate(eval_output(ntau))
      allocate(fit_output(basis_size))

      ! Generate random coefficients
      do i = 1, basis_size
         call random_number(coeffs(i))
         coeffs(i) = (coeffs(i) - 0.5_c_double) * 2.0_c_double
      end do

      ! Evaluate: coefficients -> tau values
      status = c_spir_sampling_eval_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                        1_c_int, c_loc(basis_size), 0_c_int, c_loc(coeffs), c_loc(eval_output))
      if (status /= 0) then
         print *, "Error: Failed to evaluate sampling"
         stop 1
      end if

      ! Fit: tau values -> coefficients
      status = c_spir_sampling_fit_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      1_c_int, c_loc(ntau), 0_c_int, c_loc(eval_output), c_loc(fit_output))
      if (status /= 0) then
         print *, "Error: Failed to fit sampling"
         stop 1
      end if

      ! Verify roundtrip accuracy
      do i = 1, basis_size
         if (abs(coeffs(i) - fit_output(i)) > tol * max(abs(coeffs(i)), 1.0_c_double)) then
            print *, "Error: Roundtrip accuracy failed at index", i
            print *, "  Original:", coeffs(i), " Reconstructed:", fit_output(i)
            stop 1
         end if
      end do

      ! Cleanup
      deallocate(taus, coeffs, eval_output, fit_output)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  1D sampling roundtrip: PASSED"
   end subroutine test_sampling_1d_roundtrip

   subroutine test_sampling_2d_roundtrip()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, basis_size
      integer(c_int), target :: input_dims(2)
      real(c_double), allocatable, target :: taus(:)
      real(c_double), allocatable, target :: coeffs(:, :), eval_output(:, :), fit_output(:, :)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-10_c_double
      integer(c_int), parameter :: extra_size = 2
      integer(c_int), target :: max_size
      integer :: i, j
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 2D sampling roundtrip..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Allocate arrays (column-major order)
      allocate(coeffs(basis_size, extra_size))
      allocate(eval_output(ntau, extra_size))
      allocate(fit_output(basis_size, extra_size))

      ! Generate random coefficients
      do j = 1, extra_size
         do i = 1, basis_size
            call random_number(coeffs(i, j))
            coeffs(i, j) = (coeffs(i, j) - 0.5_c_double) * 2.0_c_double
         end do
      end do

      ! Evaluate: coefficients -> tau values (target_dim = 0)
      input_dims = [basis_size, extra_size]
      status = c_spir_sampling_eval_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                       2_c_int, c_loc(input_dims), 0_c_int, c_loc(coeffs), c_loc(eval_output))
      if (status /= 0) then
         print *, "Error: Failed to evaluate 2D sampling"
         stop 1
      end if

      ! Fit: tau values -> coefficients (target_dim = 0)
      input_dims = [ntau, extra_size]
      status = c_spir_sampling_fit_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      2_c_int, c_loc(input_dims), 0_c_int, c_loc(eval_output), c_loc(fit_output))
      if (status /= 0) then
         print *, "Error: Failed to fit 2D sampling"
         stop 1
      end if

      ! Verify roundtrip accuracy
      do j = 1, extra_size
         do i = 1, basis_size
            if (abs(coeffs(i, j) - fit_output(i, j)) > tol * max(abs(coeffs(i, j)), 1.0_c_double)) then
               print *, "Error: 2D roundtrip accuracy failed at (", i, ",", j, ")"
               print *, "  Original:", coeffs(i, j), " Reconstructed:", fit_output(i, j)
               stop 1
            end if
         end do
      end do

      ! Cleanup
      deallocate(taus, coeffs, eval_output, fit_output)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  2D sampling roundtrip: PASSED"
   end subroutine test_sampling_2d_roundtrip

   subroutine test_sampling_with_statistics(statistics, positive_only, case_name)
      integer(c_int32_t), intent(in) :: statistics
      logical, intent(in) :: positive_only
      character(len=*), intent(in) :: case_name
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, matsu_sampling_ptr
      integer(c_int), target :: status, nmatsu, npoints
      integer(c_int64_t), allocatable, target :: matsus(:)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-15_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      integer(c_int), target :: max_size
      integer(c_int) :: positive_only_c
      !
      positive_only_c = MERGE(1_c_int, 0_c_int, positive_only)

      print *, "Testing Matsubara sampling with ", case_name, "..."

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(statistics, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      ! Get default Matsubara points
      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only_c, c_loc(nmatsu))
      if (status /= 0 .or. nmatsu <= 0) then
         print *, "Error: Failed to get number of Matsubara points"
         stop 1
      end if

      allocate(matsus(nmatsu))
      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only_c, c_loc(matsus))
      if (status /= 0) then
         print *, "Error: Failed to get Matsubara points"
         stop 1
      end if

      ! Create Matsubara sampling
      matsu_sampling_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only_c, nmatsu, &
                                                     c_loc(matsus), c_loc(status))
      if (status /= 0 .or. .not. c_associated(matsu_sampling_ptr)) then
         print *, "Error: Failed to create Matsubara sampling"
         stop 1
      end if

      ! Get number of points
      status = c_spir_sampling_get_npoints(matsu_sampling_ptr, c_loc(npoints))
      if (status /= 0 .or. npoints <= 0) then
         print *, "Error: Failed to get number of sampling points"
         stop 1
      end if

      ! Cleanup
      deallocate(matsus)
      call c_spir_sampling_release(matsu_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Matsubara sampling (", case_name, "): PASSED"
   end subroutine test_sampling_with_statistics

   subroutine test_sampling_3d_roundtrip()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, basis_size
      integer(c_int), target :: input_dims(3)
      real(c_double), allocatable, target :: taus(:)
      real(c_double), allocatable, target :: coeffs(:, :, :), eval_output(:, :, :), fit_output(:, :, :)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-10_c_double
      integer(c_int), parameter :: d1 = 2, d2 = 3
      integer(c_int), target :: max_size
      integer :: i, j, k
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 3D sampling roundtrip..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Allocate arrays (column-major order)
      allocate(coeffs(basis_size, d1, d2))
      allocate(eval_output(ntau, d1, d2))
      allocate(fit_output(basis_size, d1, d2))

      ! Generate random coefficients
      do k = 1, d2
         do j = 1, d1
            do i = 1, basis_size
               call random_number(coeffs(i, j, k))
               coeffs(i, j, k) = (coeffs(i, j, k) - 0.5_c_double) * 2.0_c_double
            end do
         end do
      end do

      ! Evaluate: coefficients -> tau values (target_dim = 0)
      input_dims = [basis_size, d1, d2]
      status = c_spir_sampling_eval_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                       3_c_int, c_loc(input_dims), 0_c_int, c_loc(coeffs), c_loc(eval_output))
      if (status /= 0) then
         print *, "Error: Failed to evaluate 3D sampling"
         stop 1
      end if

      ! Fit: tau values -> coefficients (target_dim = 0)
      input_dims = [ntau, d1, d2]
      status = c_spir_sampling_fit_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      3_c_int, c_loc(input_dims), 0_c_int, c_loc(eval_output), c_loc(fit_output))
      if (status /= 0) then
         print *, "Error: Failed to fit 3D sampling"
         stop 1
      end if

      ! Verify roundtrip accuracy
      do k = 1, d2
         do j = 1, d1
            do i = 1, basis_size
               if (abs(coeffs(i, j, k) - fit_output(i, j, k)) > tol * max(abs(coeffs(i, j, k)), 1.0_c_double)) then
                  print *, "Error: 3D roundtrip accuracy failed at (", i, ",", j, ",", k, ")"
                  print *, "  Original:", coeffs(i, j, k), " Reconstructed:", fit_output(i, j, k)
                  stop 1
               end if
            end do
         end do
      end do

      ! Cleanup
      deallocate(taus, coeffs, eval_output, fit_output)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  3D sampling roundtrip: PASSED"
   end subroutine test_sampling_3d_roundtrip

   subroutine test_sampling_4d_roundtrip()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, basis_size
      integer(c_int), target :: input_dims(4)
      real(c_double), allocatable, target :: taus(:)
      real(c_double), allocatable, target :: coeffs(:, :, :, :), eval_output(:, :, :, :), fit_output(:, :, :, :)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-10_c_double
      integer(c_int), parameter :: d1 = 2, d2 = 2, d3 = 2
      integer(c_int), target :: max_size
      integer :: i, j, k, l
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 4D sampling roundtrip..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Allocate arrays (column-major order)
      allocate(coeffs(basis_size, d1, d2, d3))
      allocate(eval_output(ntau, d1, d2, d3))
      allocate(fit_output(basis_size, d1, d2, d3))

      ! Generate random coefficients
      do l = 1, d3
         do k = 1, d2
            do j = 1, d1
               do i = 1, basis_size
                  call random_number(coeffs(i, j, k, l))
                  coeffs(i, j, k, l) = (coeffs(i, j, k, l) - 0.5_c_double) * 2.0_c_double
               end do
            end do
         end do
      end do

      ! Evaluate: coefficients -> tau values (target_dim = 0)
      input_dims = [basis_size, d1, d2, d3]
      status = c_spir_sampling_eval_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                       4_c_int, c_loc(input_dims), 0_c_int, c_loc(coeffs), c_loc(eval_output))
      if (status /= 0) then
         print *, "Error: Failed to evaluate 4D sampling"
         stop 1
      end if

      ! Fit: tau values -> coefficients (target_dim = 0)
      input_dims = [ntau, d1, d2, d3]
      status = c_spir_sampling_fit_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      4_c_int, c_loc(input_dims), 0_c_int, c_loc(eval_output), c_loc(fit_output))
      if (status /= 0) then
         print *, "Error: Failed to fit 4D sampling"
         stop 1
      end if

      ! Verify roundtrip accuracy
      do l = 1, d3
         do k = 1, d2
            do j = 1, d1
               do i = 1, basis_size
                  if (abs(coeffs(i, j, k, l) - fit_output(i, j, k, l)) > tol * max(abs(coeffs(i, j, k, l)), 1.0_c_double)) then
                     print *, "Error: 4D roundtrip accuracy failed at (", i, ",", j, ",", k, ",", l, ")"
                     print *, "  Original:", coeffs(i, j, k, l), " Reconstructed:", fit_output(i, j, k, l)
                     stop 1
                  end if
               end do
            end do
         end do
      end do

      ! Cleanup
      deallocate(taus, coeffs, eval_output, fit_output)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  4D sampling roundtrip: PASSED"
   end subroutine test_sampling_4d_roundtrip

   subroutine test_sampling_3d_all_target_dims()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, basis_size
      integer(c_int), target :: input_dims(3)
      real(c_double), allocatable, target :: taus(:)
      real(c_double), allocatable, target :: coeffs(:, :, :), eval_output(:, :, :), fit_output(:, :, :)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-10_c_double
      integer(c_int), parameter :: d1 = 2, d2 = 3
      integer(c_int), target :: max_size
      integer :: i, j, k
      integer(c_int) :: target_dim
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 3D sampling roundtrip with all target_dim values..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Test each target_dim (0, 1, 2)
      do target_dim = 0, 2
         ! Allocate arrays with different layouts based on target_dim
         if (target_dim == 0) then
            allocate(coeffs(basis_size, d1, d2))
            allocate(eval_output(ntau, d1, d2))
            allocate(fit_output(basis_size, d1, d2))
            input_dims = [basis_size, d1, d2]
         else if (target_dim == 1) then
            allocate(coeffs(d1, basis_size, d2))
            allocate(eval_output(d1, ntau, d2))
            allocate(fit_output(d1, basis_size, d2))
            input_dims = [d1, basis_size, d2]
         else ! target_dim == 2
            allocate(coeffs(d1, d2, basis_size))
            allocate(eval_output(d1, d2, ntau))
            allocate(fit_output(d1, d2, basis_size))
            input_dims = [d1, d2, basis_size]
         end if

         ! Generate random coefficients
         do k = 1, size(coeffs, 3)
            do j = 1, size(coeffs, 2)
               do i = 1, size(coeffs, 1)
                  call random_number(coeffs(i, j, k))
                  coeffs(i, j, k) = (coeffs(i, j, k) - 0.5_c_double) * 2.0_c_double
               end do
            end do
         end do

         ! Evaluate: coefficients -> tau values
         status = c_spir_sampling_eval_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                           3_c_int, c_loc(input_dims), target_dim, c_loc(coeffs), c_loc(eval_output))
         if (status /= 0) then
            print *, "Error: Failed to evaluate 3D sampling (target_dim=", target_dim, ")"
            stop 1
         end if

         ! Fit: tau values -> coefficients
         if (target_dim == 0) then
            input_dims = [ntau, d1, d2]
         else if (target_dim == 1) then
            input_dims = [d1, ntau, d2]
         else
            input_dims = [d1, d2, ntau]
         end if
         status = c_spir_sampling_fit_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                         3_c_int, c_loc(input_dims), target_dim, c_loc(eval_output), c_loc(fit_output))
         if (status /= 0) then
            print *, "Error: Failed to fit 3D sampling (target_dim=", target_dim, ")"
            stop 1
         end if

         ! Verify roundtrip accuracy
         do k = 1, size(coeffs, 3)
            do j = 1, size(coeffs, 2)
               do i = 1, size(coeffs, 1)
                  if (abs(coeffs(i, j, k) - fit_output(i, j, k)) > tol * max(abs(coeffs(i, j, k)), 1.0_c_double)) then
                     print *, "Error: 3D roundtrip accuracy failed at (", i, ",", j, ",", k, ", target_dim=", target_dim, ")"
                     print *, "  Original:", coeffs(i, j, k), " Reconstructed:", fit_output(i, j, k)
                     stop 1
                  end if
               end do
            end do
         end do

         ! Cleanup
         deallocate(coeffs, eval_output, fit_output)
      end do

      ! Cleanup
      deallocate(taus)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  3D sampling roundtrip (all target_dim): PASSED"
   end subroutine test_sampling_3d_all_target_dims

   subroutine test_sampling_4d_all_target_dims()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, tau_sampling_ptr
      integer(c_int), target :: status, ntau, basis_size
      integer(c_int), target :: input_dims(4)
      real(c_double), allocatable, target :: taus(:)
      real(c_double), allocatable, target :: coeffs(:, :, :, :), eval_output(:, :, :, :), fit_output(:, :, :, :)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-10_c_double
      integer(c_int), parameter :: d1 = 2, d2 = 2, d3 = 2
      integer(c_int), target :: max_size
      integer :: i, j, k, l
      integer(c_int) :: target_dim
      integer :: seed_size
      integer, allocatable :: seed(:)

      print *, "Testing 4D sampling roundtrip with all target_dim values..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default tau points
      status = c_spir_basis_get_n_default_taus(basis_ptr, c_loc(ntau))
      allocate(taus(ntau))
      status = c_spir_basis_get_default_taus(basis_ptr, c_loc(taus))

      ! Create tau sampling
      tau_sampling_ptr = c_spir_tau_sampling_new(basis_ptr, ntau, c_loc(taus), c_loc(status))

      ! Test each target_dim (0, 1, 2, 3)
      do target_dim = 0, 3
         ! Allocate arrays with different layouts based on target_dim
         if (target_dim == 0) then
            allocate(coeffs(basis_size, d1, d2, d3))
            allocate(eval_output(ntau, d1, d2, d3))
            allocate(fit_output(basis_size, d1, d2, d3))
            input_dims = [basis_size, d1, d2, d3]
         else if (target_dim == 1) then
            allocate(coeffs(d1, basis_size, d2, d3))
            allocate(eval_output(d1, ntau, d2, d3))
            allocate(fit_output(d1, basis_size, d2, d3))
            input_dims = [d1, basis_size, d2, d3]
         else if (target_dim == 2) then
            allocate(coeffs(d1, d2, basis_size, d3))
            allocate(eval_output(d1, d2, ntau, d3))
            allocate(fit_output(d1, d2, basis_size, d3))
            input_dims = [d1, d2, basis_size, d3]
         else ! target_dim == 3
            allocate(coeffs(d1, d2, d3, basis_size))
            allocate(eval_output(d1, d2, d3, ntau))
            allocate(fit_output(d1, d2, d3, basis_size))
            input_dims = [d1, d2, d3, basis_size]
         end if

         ! Generate random coefficients
         do l = 1, size(coeffs, 4)
            do k = 1, size(coeffs, 3)
               do j = 1, size(coeffs, 2)
                  do i = 1, size(coeffs, 1)
                     call random_number(coeffs(i, j, k, l))
                     coeffs(i, j, k, l) = (coeffs(i, j, k, l) - 0.5_c_double) * 2.0_c_double
                  end do
               end do
            end do
         end do

         ! Evaluate: coefficients -> tau values
         status = c_spir_sampling_eval_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                           4_c_int, c_loc(input_dims), target_dim, c_loc(coeffs), c_loc(eval_output))
         if (status /= 0) then
            print *, "Error: Failed to evaluate 4D sampling (target_dim=", target_dim, ")"
            stop 1
         end if

         ! Fit: tau values -> coefficients
         if (target_dim == 0) then
            input_dims = [ntau, d1, d2, d3]
         else if (target_dim == 1) then
            input_dims = [d1, ntau, d2, d3]
         else if (target_dim == 2) then
            input_dims = [d1, d2, ntau, d3]
         else
            input_dims = [d1, d2, d3, ntau]
         end if
         status = c_spir_sampling_fit_dd(tau_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                         4_c_int, c_loc(input_dims), target_dim, c_loc(eval_output), c_loc(fit_output))
         if (status /= 0) then
            print *, "Error: Failed to fit 4D sampling (target_dim=", target_dim, ")"
            stop 1
         end if

         ! Verify roundtrip accuracy
         do l = 1, size(coeffs, 4)
            do k = 1, size(coeffs, 3)
               do j = 1, size(coeffs, 2)
                  do i = 1, size(coeffs, 1)
                     if (abs(coeffs(i, j, k, l) - fit_output(i, j, k, l)) > tol * max(abs(coeffs(i, j, k, l)), 1.0_c_double)) then
                        print *, "Error: 4D roundtrip accuracy failed at (", i, ",", j, ",", k, ",", l, &
                                 ", target_dim=", target_dim, ")"
                        print *, "  Original:", coeffs(i, j, k, l), " Reconstructed:", fit_output(i, j, k, l)
                        stop 1
                     end if
                  end do
               end do
            end do
         end do

         ! Cleanup
         deallocate(coeffs, eval_output, fit_output)
      end do

      ! Cleanup
      deallocate(taus)
      call c_spir_sampling_release(tau_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  4D sampling roundtrip (all target_dim): PASSED"
   end subroutine test_sampling_4d_all_target_dims

   subroutine test_sampling_complex_1d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, matsu_sampling_ptr
      integer(c_int), target :: status, nmatsu, basis_size, npoints
      integer(c_int64_t), allocatable, target :: matsus(:)
      complex(c_double_complex), allocatable, target :: coeffs(:), eval_output(:), fit_output(:)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-12_c_double
      integer(c_int), parameter :: positive_only = 0
      integer(c_int), target :: max_size
      integer :: i
      integer :: seed_size
      integer, allocatable :: seed(:)
      real(c_double) :: r_real, r_imag

      print *, "Testing complex 1D sampling roundtrip..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default Matsubara points
      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only, c_loc(nmatsu))
      allocate(matsus(nmatsu))
      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only, c_loc(matsus))

      ! Create Matsubara sampling
      matsu_sampling_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only, nmatsu, &
                                                     c_loc(matsus), c_loc(status))
      status = c_spir_sampling_get_npoints(matsu_sampling_ptr, c_loc(npoints))

      ! Allocate arrays
      allocate(coeffs(basis_size))
      allocate(eval_output(npoints))
      allocate(fit_output(basis_size))

      ! Generate random complex coefficients
      do i = 1, basis_size
         call random_number(r_real)
         call random_number(r_imag)
         coeffs(i) = cmplx((r_real - 0.5_c_double) * 2.0_c_double, &
                          (r_imag - 0.5_c_double) * 2.0_c_double, kind=c_double_complex)
      end do

      ! Evaluate: coefficients -> Matsubara values
      status = c_spir_sampling_eval_zz(matsu_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      1_c_int, c_loc(basis_size), 0_c_int, c_loc(coeffs), c_loc(eval_output))
      if (status /= 0) then
         print *, "Error: Failed to evaluate complex 1D sampling"
         stop 1
      end if

      ! Fit: Matsubara values -> coefficients
      status = c_spir_sampling_fit_zz(matsu_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      1_c_int, c_loc(npoints), 0_c_int, c_loc(eval_output), c_loc(fit_output))
      if (status /= 0) then
         print *, "Error: Failed to fit complex 1D sampling"
         stop 1
      end if

      ! Verify roundtrip accuracy
      do i = 1, basis_size
         if (abs(coeffs(i) - fit_output(i)) > tol * max(abs(coeffs(i)), 1.0_c_double)) then
            print *, "Error: Complex 1D roundtrip accuracy failed at index", i
            print *, "  Original:", coeffs(i), " Reconstructed:", fit_output(i)
            stop 1
         end if
      end do

      ! Cleanup
      deallocate(matsus, coeffs, eval_output, fit_output)
      call c_spir_sampling_release(matsu_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Complex 1D sampling roundtrip: PASSED"
   end subroutine test_sampling_complex_1d

   subroutine test_sampling_complex_2d()
      type(c_ptr) :: k_ptr, sve_ptr, basis_ptr, matsu_sampling_ptr
      integer(c_int), target :: status, nmatsu, basis_size, npoints
      integer(c_int), target :: input_dims(2)
      integer(c_int64_t), allocatable, target :: matsus(:)
      complex(c_double_complex), allocatable, target :: coeffs(:, :), eval_output(:, :), fit_output(:, :)
      real(c_double), parameter :: beta = 1.0_c_double
      real(c_double), parameter :: omega_max = 10.0_c_double
      real(c_double), parameter :: epsilon = 1.0e-10_c_double
      real(c_double), parameter :: lambda = beta*omega_max
      real(c_double), parameter :: tol = 1.0e-12_c_double
      integer(c_int), parameter :: positive_only = 0
      integer(c_int), parameter :: extra_size = 2
      integer(c_int), target :: max_size
      integer :: i, j
      integer :: seed_size
      integer, allocatable :: seed(:)
      real(c_double) :: r_real, r_imag

      print *, "Testing complex 2D sampling roundtrip..."

      ! Initialize random seed
      call random_seed(size=seed_size)
      if (seed_size > 0) then
         allocate(seed(seed_size))
         seed = 42
         call random_seed(put=seed)
         deallocate(seed)
      end if

      ! Create basis
      k_ptr = c_spir_logistic_kernel_new(lambda, c_loc(status))
      sve_ptr = c_spir_sve_result_new(k_ptr, epsilon, -1_c_int, -1_c_int, SPIR_TWORK_AUTO, c_loc(status))
      max_size = -1
      basis_ptr = c_spir_basis_new(SPIR_STATISTICS_FERMIONIC, beta, omega_max, epsilon, &
                                   k_ptr, sve_ptr, max_size, c_loc(status))

      status = c_spir_basis_get_size(basis_ptr, c_loc(basis_size))

      ! Get default Matsubara points
      status = c_spir_basis_get_n_default_matsus(basis_ptr, positive_only, c_loc(nmatsu))
      allocate(matsus(nmatsu))
      status = c_spir_basis_get_default_matsus(basis_ptr, positive_only, c_loc(matsus))

      ! Create Matsubara sampling
      matsu_sampling_ptr = c_spir_matsu_sampling_new(basis_ptr, positive_only, nmatsu, &
                                                     c_loc(matsus), c_loc(status))
      status = c_spir_sampling_get_npoints(matsu_sampling_ptr, c_loc(npoints))

      ! Allocate arrays (column-major order)
      allocate(coeffs(basis_size, extra_size))
      allocate(eval_output(npoints, extra_size))
      allocate(fit_output(basis_size, extra_size))

      ! Generate random complex coefficients
      do j = 1, extra_size
         do i = 1, basis_size
            call random_number(r_real)
            call random_number(r_imag)
            coeffs(i, j) = cmplx((r_real - 0.5_c_double) * 2.0_c_double, &
                                (r_imag - 0.5_c_double) * 2.0_c_double, kind=c_double_complex)
         end do
      end do

      ! Evaluate: coefficients -> Matsubara values (target_dim = 0)
      input_dims = [basis_size, extra_size]
      status = c_spir_sampling_eval_zz(matsu_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      2_c_int, c_loc(input_dims), 0_c_int, c_loc(coeffs), c_loc(eval_output))
      if (status /= 0) then
         print *, "Error: Failed to evaluate complex 2D sampling"
         stop 1
      end if

      ! Fit: Matsubara values -> coefficients (target_dim = 0)
      input_dims = [npoints, extra_size]
      status = c_spir_sampling_fit_zz(matsu_sampling_ptr, c_null_ptr, SPIR_ORDER_COLUMN_MAJOR, &
                                      2_c_int, c_loc(input_dims), 0_c_int, c_loc(eval_output), c_loc(fit_output))
      if (status /= 0) then
         print *, "Error: Failed to fit complex 2D sampling"
         stop 1
      end if

      ! Verify roundtrip accuracy
      do j = 1, extra_size
         do i = 1, basis_size
            if (abs(coeffs(i, j) - fit_output(i, j)) > tol * max(abs(coeffs(i, j)), 1.0_c_double)) then
               print *, "Error: Complex 2D roundtrip accuracy failed at (", i, ",", j, ")"
               print *, "  Original:", coeffs(i, j), " Reconstructed:", fit_output(i, j)
               stop 1
            end if
         end do
      end do

      ! Cleanup
      deallocate(matsus, coeffs, eval_output, fit_output)
      call c_spir_sampling_release(matsu_sampling_ptr)
      call c_spir_basis_release(basis_ptr)
      call c_spir_sve_result_release(sve_ptr)
      call c_spir_kernel_release(k_ptr)

      print *, "  Complex 2D sampling roundtrip: PASSED"
   end subroutine test_sampling_complex_2d

end program test_sampling


