! Test multi-dimensional arrays (1D through 7D)
program test_multidim
   use sparse_ir_c
   use sparse_ir_extension
   use, intrinsic :: iso_c_binding
   implicit none

   integer, parameter :: dp = KIND(1.0D0)

   print *, "======================================"
   print *, "Testing multi-dimensional arrays"
   print *, "======================================"

   ! Test all dimensions from 1D to 7D
   call test_1d_operations()
   call test_2d_operations()
   call test_3d_operations()
   call test_4d_operations()
   call test_5d_operations()
   call test_6d_operations()
   call test_7d_operations()

   ! Test with different statistics
   call test_multidim_with_statistics(SPIR_STATISTICS_FERMIONIC, "Fermionic")
   call test_multidim_with_statistics(SPIR_STATISTICS_BOSONIC, "Bosonic")

   ! Test with different positive_only values
   call test_multidim_with_positive_only(SPIR_STATISTICS_FERMIONIC, .false., "Fermionic, all frequencies")
   call test_multidim_with_positive_only(SPIR_STATISTICS_FERMIONIC, .true., "Fermionic, positive only")
   call test_multidim_with_positive_only(SPIR_STATISTICS_BOSONIC, .false., "Bosonic, all frequencies")
   call test_multidim_with_positive_only(SPIR_STATISTICS_BOSONIC, .true., "Bosonic, positive only")

   ! Test with different target_dim values
   call test_3d_all_target_dims()
   call test_4d_all_target_dims()

   print *, "======================================"
   print *, "All multi-dimensional tests passed!"
   print *, "======================================"

contains

   subroutine test_1d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 2.0_DP
      real(kind=dp), parameter :: omega_max = 5.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-8_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:), giw(:), giw_reconst(:)
      complex(kind=dp), allocatable :: g_ir2_z(:), gtau_z(:)
      integer :: i
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 1D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 1D arrays
      allocate(g_ir(irobj%size))
      allocate(giw(nfreq))
      allocate(giw_reconst(nfreq))
      allocate(g_ir2_z(irobj%size))
      allocate(gtau_z(irobj%ntau))

      ! Generate random coefficients
      do i = 1, irobj%size
         call random_number(r)
         call random_number(r_imag)
         g_ir(i) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
      end do

      ! Evaluate Green's function at Matsubara frequencies
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_1d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 1D transformation cycle"
         stop 1
      end if

      ! Evaluate Green's function at tau points
      call evaluate_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, gtau_z)

      ! Convert tau points back to IR
      call fit_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z_1d(giw, giw_reconst, 10.0_DP*irobj%eps)) then
         print *, "Error: Matsubara frequencies do not match after 1D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  1D operations: PASSED"
   end subroutine test_1d_operations

   subroutine test_2d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 2.0_DP
      real(kind=dp), parameter :: omega_max = 5.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-8_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :), giw(:, :), giw_reconst(:, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :), gtau_z(:, :)
      integer :: i, j
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 2D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 2D arrays (column-major order: size x d1)
      allocate(g_ir(irobj%size, 3))
      allocate(giw(nfreq, 3))
      allocate(giw_reconst(nfreq, 3))
      allocate(g_ir2_z(irobj%size, 3))
      allocate(gtau_z(irobj%ntau, 3))

      ! Generate random coefficients
      do j = 1, 3
         do i = 1, irobj%size
            call random_number(r)
            call random_number(r_imag)
            g_ir(i, j) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_2d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 2D transformation cycle"
         stop 1
      end if

      ! Evaluate Green's function at tau points
      call evaluate_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, gtau_z)

      ! Convert tau points back to IR
      call fit_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z_2d(giw, giw_reconst, 10.0_DP*irobj%eps)) then
         print *, "Error: Matsubara frequencies do not match after 2D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  2D operations: PASSED"
   end subroutine test_2d_operations

   subroutine test_3d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 2.0_DP
      real(kind=dp), parameter :: omega_max = 5.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-8_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :), giw(:, :, :), giw_reconst(:, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :), gtau_z(:, :, :)
      integer :: i, j, k
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 3D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 3D arrays (column-major order: size x d1 x d2)
      allocate(g_ir(irobj%size, 3, 5))
      allocate(giw(nfreq, 3, 5))
      allocate(giw_reconst(nfreq, 3, 5))
      allocate(g_ir2_z(irobj%size, 3, 5))
      allocate(gtau_z(irobj%ntau, 3, 5))

      ! Generate random coefficients
      do k = 1, 5
         do j = 1, 3
            do i = 1, irobj%size
               call random_number(r)
               call random_number(r_imag)
               g_ir(i, j, k) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
            end do
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies (target_dim = 1)
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR (target_dim = 1)
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_3d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 3D transformation cycle"
         stop 1
      end if

      ! Evaluate Green's function at tau points (target_dim = 1)
      call evaluate_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, gtau_z)

      ! Convert tau points back to IR (target_dim = 1)
      call fit_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again (target_dim = 1)
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z_3d(giw, giw_reconst, 10.0_DP*irobj%eps)) then
         print *, "Error: Matsubara frequencies do not match after 3D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  3D operations: PASSED"
   end subroutine test_3d_operations

   subroutine test_4d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 1.0_DP
      real(kind=dp), parameter :: omega_max = 10.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-10_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :, :), giw(:, :, :, :), giw_reconst(:, :, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :, :), gtau_z(:, :, :, :)
      integer :: i, j, k, l
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 4D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 4D arrays (column-major order: size x d1 x d2 x d3)
      allocate(g_ir(irobj%size, 2, 3, 4))
      allocate(giw(nfreq, 2, 3, 4))
      allocate(giw_reconst(nfreq, 2, 3, 4))
      allocate(g_ir2_z(irobj%size, 2, 3, 4))
      allocate(gtau_z(irobj%ntau, 2, 3, 4))

      ! Generate random coefficients
      do l = 1, 4
         do k = 1, 3
            do j = 1, 2
               do i = 1, irobj%size
                  call random_number(r)
                  call random_number(r_imag)
                  g_ir(i, j, k, l) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
               end do
            end do
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies (target_dim = 1)
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR (target_dim = 1)
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_4d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 4D transformation cycle"
         stop 1
      end if

      ! Evaluate Green's function at tau points (target_dim = 1)
      call evaluate_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, gtau_z)

      ! Convert tau points back to IR (target_dim = 1)
      call fit_tau(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, gtau_z, g_ir2_z)

      ! Evaluate Green's function at Matsubara frequencies again (target_dim = 1)
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir2_z, giw_reconst)

      ! Compare the original and reconstructed Matsubara frequencies
      if (.not. compare_with_relative_error_z_4d(giw, giw_reconst, 10.0_DP*irobj%eps)) then
         print *, "Error: Matsubara frequencies do not match after 4D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  4D operations: PASSED"
   end subroutine test_4d_operations

   subroutine test_5d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 1.0_DP
      real(kind=dp), parameter :: omega_max = 10.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-10_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :, :, :), giw(:, :, :, :, :), giw_reconst(:, :, :, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :, :, :), gtau_z(:, :, :, :, :)
      integer :: i, j, k, l, m
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 5D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 5D arrays (column-major order: size x d1 x d2 x d3 x d4)
      allocate(g_ir(irobj%size, 2, 2, 2, 2))
      allocate(giw(nfreq, 2, 2, 2, 2))
      allocate(giw_reconst(nfreq, 2, 2, 2, 2))
      allocate(g_ir2_z(irobj%size, 2, 2, 2, 2))
      allocate(gtau_z(irobj%ntau, 2, 2, 2, 2))

      ! Generate random coefficients
      do m = 1, 2
         do l = 1, 2
            do k = 1, 2
               do j = 1, 2
                  do i = 1, irobj%size
                     call random_number(r)
                     call random_number(r_imag)
                     g_ir(i, j, k, l, m) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
                  end do
               end do
            end do
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_5d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 5D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  5D operations: PASSED"
   end subroutine test_5d_operations

   subroutine test_6d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 1.0_DP
      real(kind=dp), parameter :: omega_max = 10.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-10_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :, :, :, :), giw(:, :, :, :, :, :), giw_reconst(:, :, :, :, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :, :, :, :), gtau_z(:, :, :, :, :, :)
      integer :: i, j, k, l, m, n
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 6D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 6D arrays (column-major order: size x d1 x d2 x d3 x d4 x d5)
      allocate(g_ir(irobj%size, 2, 2, 2, 2, 2))
      allocate(giw(nfreq, 2, 2, 2, 2, 2))
      allocate(giw_reconst(nfreq, 2, 2, 2, 2, 2))
      allocate(g_ir2_z(irobj%size, 2, 2, 2, 2, 2))
      allocate(gtau_z(irobj%ntau, 2, 2, 2, 2, 2))

      ! Generate random coefficients
      do n = 1, 2
         do m = 1, 2
            do l = 1, 2
               do k = 1, 2
                  do j = 1, 2
                     do i = 1, irobj%size
                        call random_number(r)
                        call random_number(r_imag)
                        g_ir(i, j, k, l, m, n) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
                     end do
                  end do
               end do
            end do
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_6d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 6D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  6D operations: PASSED"
   end subroutine test_6d_operations

   subroutine test_7d_operations()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 1.0_DP
      real(kind=dp), parameter :: omega_max = 10.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-10_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :, :, :, :, :), giw(:, :, :, :, :, :, :), giw_reconst(:, :, :, :, :, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :, :, :, :, :), gtau_z(:, :, :, :, :, :, :)
      integer :: i, j, k, l, m, n, o
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 7D operations..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Allocate 7D arrays (column-major order: size x d1 x d2 x d3 x d4 x d5 x d6)
      allocate(g_ir(irobj%size, 2, 2, 2, 2, 2, 2))
      allocate(giw(nfreq, 2, 2, 2, 2, 2, 2))
      allocate(giw_reconst(nfreq, 2, 2, 2, 2, 2, 2))
      allocate(g_ir2_z(irobj%size, 2, 2, 2, 2, 2, 2))
      allocate(gtau_z(irobj%ntau, 2, 2, 2, 2, 2, 2))

      ! Generate random coefficients
      do o = 1, 2
         do n = 1, 2
            do m = 1, 2
               do l = 1, 2
                  do k = 1, 2
                     do j = 1, 2
                        do i = 1, irobj%size
                           call random_number(r)
                           call random_number(r_imag)
                           g_ir(i, j, k, l, m, n, o) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies
      call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_7d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 7D transformation cycle"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  7D operations: PASSED"
   end subroutine test_7d_operations

   subroutine test_multidim_with_statistics(statistics, case_name)
      integer(c_int32_t), intent(in) :: statistics
      character(len=*), intent(in) :: case_name
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 2.0_DP
      real(kind=dp), parameter :: omega_max = 5.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-8_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :), giw(:, :, :), giw_reconst(:, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :), gtau_z(:, :, :)
      integer :: i, j, k
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 3D operations with ", case_name, " statistics..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         nfreq = irobj%nfreq_f
      else if (statistics == SPIR_STATISTICS_BOSONIC) then
         nfreq = irobj%nfreq_b
      else
         print *, "Error: Invalid statistics"
         stop 1
      end if

      ! Allocate 3D arrays
      allocate(g_ir(irobj%size, 3, 5))
      allocate(giw(nfreq, 3, 5))
      allocate(giw_reconst(nfreq, 3, 5))
      allocate(g_ir2_z(irobj%size, 3, 5))
      allocate(gtau_z(irobj%ntau, 3, 5))

      ! Generate random coefficients
      do k = 1, 5
         do j = 1, 3
            do i = 1, irobj%size
               call random_number(r)
               call random_number(r_imag)
               g_ir(i, j, k) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
            end do
         end do
      end do

      ! Evaluate Green's function at Matsubara frequencies
      call evaluate_matsubara(irobj, statistics, target_dim, g_ir, giw)

      ! Convert Matsubara frequencies back to IR
      call fit_matsubara(irobj, statistics, target_dim, giw, g_ir2_z)

      ! Compare IR coefficients
      if (.not. compare_with_relative_error_z_3d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
         print *, "Error: IR coefficients do not match after 3D transformation cycle (", case_name, ")"
         stop 1
      end if

      ! Cleanup
      deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      call finalize_ir(irobj)

      print *, "  3D operations (", case_name, "): PASSED"
   end subroutine test_multidim_with_statistics

   subroutine test_multidim_with_positive_only(statistics, positive_only_val, case_name)
      integer(c_int32_t), intent(in) :: statistics
      logical, intent(in) :: positive_only_val
      character(len=*), intent(in) :: case_name
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 2.0_DP
      real(kind=dp), parameter :: omega_max = 5.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-8_DP
      real(kind=dp), parameter :: lambda = beta*omega_max

      complex(kind=dp), allocatable :: g_ir(:, :, :), giw(:, :, :), giw_reconst(:, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :), gtau_z(:, :, :)
      ! For positive_only=true, use real arrays
      real(kind=dp), allocatable :: g_ir_d(:, :, :), g_ir2_d(:, :, :)
      integer :: i, j, k
      integer, parameter :: target_dim = 1
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 3D operations with ", case_name, "..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only_val)

      if (statistics == SPIR_STATISTICS_FERMIONIC) then
         nfreq = irobj%nfreq_f
      else if (statistics == SPIR_STATISTICS_BOSONIC) then
         nfreq = irobj%nfreq_b
      else
         print *, "Error: Invalid statistics"
         stop 1
      end if

      print *, "  IR size:", irobj%size, " nfreq:", nfreq, " positive_only:", positive_only_val

      ! Allocate arrays
      allocate(giw(nfreq, 3, 5))
      allocate(giw_reconst(nfreq, 3, 5))
      allocate(gtau_z(irobj%ntau, 3, 5))

      if (positive_only_val) then
         ! For positive_only=true, use real IR coefficients with dz/zd conversions
         allocate(g_ir_d(irobj%size, 3, 5))
         allocate(g_ir2_d(irobj%size, 3, 5))

         ! Generate random real coefficients
         do k = 1, 5
            do j = 1, 3
               do i = 1, irobj%size
                  call random_number(r)
                  g_ir_d(i, j, k) = 2.0_DP*r - 1.0_DP
               end do
            end do
         end do

         ! Evaluate Green's function at Matsubara frequencies (real -> complex)
         call evaluate_matsubara(irobj, statistics, target_dim, g_ir_d, giw)

         ! Convert Matsubara frequencies back to IR (complex -> real)
         call fit_matsubara(irobj, statistics, target_dim, giw, g_ir2_d)

         ! Compare IR coefficients
         if (.not. compare_with_relative_error_d_3d(g_ir_d, g_ir2_d, 10.0_DP*irobj%eps)) then
            print *, "Error: IR coefficients do not match after 3D transformation cycle (", case_name, ")"
            stop 1
         end if

         deallocate(g_ir_d, g_ir2_d)
      else
         ! For positive_only=false, use complex IR coefficients
         allocate(g_ir(irobj%size, 3, 5))
         allocate(g_ir2_z(irobj%size, 3, 5))

         ! Generate random complex coefficients
         do k = 1, 5
            do j = 1, 3
               do i = 1, irobj%size
                  call random_number(r)
                  call random_number(r_imag)
                  g_ir(i, j, k) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
               end do
            end do
         end do

         ! Evaluate Green's function at Matsubara frequencies
         call evaluate_matsubara(irobj, statistics, target_dim, g_ir, giw)

         ! Convert Matsubara frequencies back to IR
         call fit_matsubara(irobj, statistics, target_dim, giw, g_ir2_z)

         ! Compare IR coefficients
         if (.not. compare_with_relative_error_z_3d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
            print *, "Error: IR coefficients do not match after 3D transformation cycle (", case_name, ")"
            stop 1
         end if

         deallocate(g_ir, g_ir2_z)
      end if

      ! Cleanup
      deallocate(giw, giw_reconst, gtau_z)
      call finalize_ir(irobj)

      print *, "  3D operations (", case_name, "): PASSED"
   end subroutine test_multidim_with_positive_only

   subroutine test_3d_all_target_dims()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 2.0_DP
      real(kind=dp), parameter :: omega_max = 5.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-8_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :), giw(:, :, :), giw_reconst(:, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :), gtau_z(:, :, :)
      integer :: i, j, k, target_dim
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 3D operations with all target_dim values..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Test each target_dim (1, 2, 3)
      do target_dim = 1, 3
         ! Allocate 3D arrays with different layouts based on target_dim
         if (target_dim == 1) then
            allocate(g_ir(irobj%size, 3, 5))
            allocate(giw(nfreq, 3, 5))
            allocate(giw_reconst(nfreq, 3, 5))
            allocate(g_ir2_z(irobj%size, 3, 5))
            allocate(gtau_z(irobj%ntau, 3, 5))
         else if (target_dim == 2) then
            allocate(g_ir(3, irobj%size, 5))
            allocate(giw(3, nfreq, 5))
            allocate(giw_reconst(3, nfreq, 5))
            allocate(g_ir2_z(3, irobj%size, 5))
            allocate(gtau_z(3, irobj%ntau, 5))
         else ! target_dim == 3
            allocate(g_ir(3, 5, irobj%size))
            allocate(giw(3, 5, nfreq))
            allocate(giw_reconst(3, 5, nfreq))
            allocate(g_ir2_z(3, 5, irobj%size))
            allocate(gtau_z(3, 5, irobj%ntau))
         end if

         ! Generate random coefficients
         do k = 1, size(g_ir, 3)
            do j = 1, size(g_ir, 2)
               do i = 1, size(g_ir, 1)
                  call random_number(r)
                  call random_number(r_imag)
                  g_ir(i, j, k) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
               end do
            end do
         end do

         ! Evaluate Green's function at Matsubara frequencies
         call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

         ! Convert Matsubara frequencies back to IR
         call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

         ! Compare IR coefficients
         if (.not. compare_with_relative_error_z_3d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
            print *, "Error: IR coefficients do not match after 3D transformation cycle (target_dim=", target_dim, ")"
            stop 1
         end if

         ! Cleanup
         deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      end do

      call finalize_ir(irobj)

      print *, "  3D operations (all target_dim): PASSED"
   end subroutine test_3d_all_target_dims

   subroutine test_4d_all_target_dims()
      type(IR) :: irobj
      real(kind=dp), parameter :: beta = 1.0_DP
      real(kind=dp), parameter :: omega_max = 10.0_DP
      real(kind=dp), parameter :: epsilon = 1.0e-10_DP
      real(kind=dp), parameter :: lambda = beta*omega_max
      logical, parameter :: positive_only = .false.

      complex(kind=dp), allocatable :: g_ir(:, :, :, :), giw(:, :, :, :), giw_reconst(:, :, :, :)
      complex(kind=dp), allocatable :: g_ir2_z(:, :, :, :), gtau_z(:, :, :, :)
      integer :: i, j, k, l, target_dim
      real(kind=dp) :: r, r_imag
      integer :: nfreq

      print *, "Testing 4D operations with all target_dim values..."

      call init_ir(irobj, beta, lambda, epsilon, positive_only)

      nfreq = irobj%nfreq_f

      ! Test each target_dim (0, 1, 2, 3)
      do target_dim = 1, 4
         ! Allocate 4D arrays with different layouts based on target_dim
         if (target_dim == 1) then
            allocate(g_ir(irobj%size, 2, 3, 4))
            allocate(giw(nfreq, 2, 3, 4))
            allocate(giw_reconst(nfreq, 2, 3, 4))
            allocate(g_ir2_z(irobj%size, 2, 3, 4))
            allocate(gtau_z(irobj%ntau, 2, 3, 4))
         else if (target_dim == 2) then
            allocate(g_ir(2, irobj%size, 3, 4))
            allocate(giw(2, nfreq, 3, 4))
            allocate(giw_reconst(2, nfreq, 3, 4))
            allocate(g_ir2_z(2, irobj%size, 3, 4))
            allocate(gtau_z(2, irobj%ntau, 3, 4))
         else if (target_dim == 3) then
            allocate(g_ir(2, 3, irobj%size, 4))
            allocate(giw(2, 3, nfreq, 4))
            allocate(giw_reconst(2, 3, nfreq, 4))
            allocate(g_ir2_z(2, 3, irobj%size, 4))
            allocate(gtau_z(2, 3, irobj%ntau, 4))
         else ! target_dim == 4
            allocate(g_ir(2, 3, 4, irobj%size))
            allocate(giw(2, 3, 4, nfreq))
            allocate(giw_reconst(2, 3, 4, nfreq))
            allocate(g_ir2_z(2, 3, 4, irobj%size))
            allocate(gtau_z(2, 3, 4, irobj%ntau))
         end if

         ! Generate random coefficients
         do l = 1, size(g_ir, 4)
            do k = 1, size(g_ir, 3)
               do j = 1, size(g_ir, 2)
                  do i = 1, size(g_ir, 1)
                     call random_number(r)
                     call random_number(r_imag)
                     g_ir(i, j, k, l) = cmplx(2.0_DP*r - 1.0_DP, r_imag, kind=DP)
                  end do
               end do
            end do
         end do

         ! Evaluate Green's function at Matsubara frequencies
         call evaluate_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, g_ir, giw)

         ! Convert Matsubara frequencies back to IR
         call fit_matsubara(irobj, SPIR_STATISTICS_FERMIONIC, target_dim, giw, g_ir2_z)

         ! Compare IR coefficients
         if (.not. compare_with_relative_error_z_4d(g_ir, g_ir2_z, 10.0_DP*irobj%eps)) then
            print *, "Error: IR coefficients do not match after 4D transformation cycle (target_dim=", target_dim, ")"
            stop 1
         end if

         ! Cleanup
         deallocate(g_ir, giw, giw_reconst, g_ir2_z, gtau_z)
      end do

      call finalize_ir(irobj)

      print *, "  4D operations (all target_dim): PASSED"
   end subroutine test_4d_all_target_dims

   function compare_with_relative_error_z_1d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:), b(:)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do i = 1, size(a)
         max_diff = max(max_diff, abs(a(i) - b(i)))
         max_ref = max(max_ref, abs(a(i)))
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_1d

   function compare_with_relative_error_z_2d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:, :), b(:, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do j = 1, size(a, 2)
         do i = 1, size(a, 1)
            max_diff = max(max_diff, abs(a(i, j) - b(i, j)))
            max_ref = max(max_ref, abs(a(i, j)))
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_2d

   function compare_with_relative_error_z_3d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:, :, :), b(:, :, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j, k

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do k = 1, size(a, 3)
         do j = 1, size(a, 2)
            do i = 1, size(a, 1)
               max_diff = max(max_diff, abs(a(i, j, k) - b(i, j, k)))
               max_ref = max(max_ref, abs(a(i, j, k)))
            end do
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_3d

   function compare_with_relative_error_z_4d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:, :, :, :), b(:, :, :, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j, k, l

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do l = 1, size(a, 4)
         do k = 1, size(a, 3)
            do j = 1, size(a, 2)
               do i = 1, size(a, 1)
                  max_diff = max(max_diff, abs(a(i, j, k, l) - b(i, j, k, l)))
                  max_ref = max(max_ref, abs(a(i, j, k, l)))
               end do
            end do
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_4d

   function compare_with_relative_error_z_5d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:, :, :, :, :), b(:, :, :, :, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j, k, l, m

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do m = 1, size(a, 5)
         do l = 1, size(a, 4)
            do k = 1, size(a, 3)
               do j = 1, size(a, 2)
                  do i = 1, size(a, 1)
                     max_diff = max(max_diff, abs(a(i, j, k, l, m) - b(i, j, k, l, m)))
                     max_ref = max(max_ref, abs(a(i, j, k, l, m)))
                  end do
               end do
            end do
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_5d

   function compare_with_relative_error_z_6d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:, :, :, :, :, :), b(:, :, :, :, :, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j, k, l, m, n

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do n = 1, size(a, 6)
         do m = 1, size(a, 5)
            do l = 1, size(a, 4)
               do k = 1, size(a, 3)
                  do j = 1, size(a, 2)
                     do i = 1, size(a, 1)
                        max_diff = max(max_diff, abs(a(i, j, k, l, m, n) - b(i, j, k, l, m, n)))
                        max_ref = max(max_ref, abs(a(i, j, k, l, m, n)))
                     end do
                  end do
               end do
            end do
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_6d

   function compare_with_relative_error_z_7d(a, b, tol) result(is_close)
      complex(kind=dp), intent(in) :: a(:, :, :, :, :, :, :), b(:, :, :, :, :, :, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j, k, l, m, n, o

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do o = 1, size(a, 7)
         do n = 1, size(a, 6)
            do m = 1, size(a, 5)
               do l = 1, size(a, 4)
                  do k = 1, size(a, 3)
                     do j = 1, size(a, 2)
                        do i = 1, size(a, 1)
                           max_diff = max(max_diff, abs(a(i, j, k, l, m, n, o) - b(i, j, k, l, m, n, o)))
                           max_ref = max(max_ref, abs(a(i, j, k, l, m, n, o)))
                        end do
                     end do
                  end do
               end do
            end do
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_z_7d

   function compare_with_relative_error_d_3d(a, b, tol) result(is_close)
      real(kind=dp), intent(in) :: a(:, :, :), b(:, :, :)
      real(kind=dp), intent(in) :: tol
      logical :: is_close
      real(kind=dp) :: max_diff, max_ref
      integer :: i, j, k

      max_diff = 0.0_DP
      max_ref = 0.0_DP

      do k = 1, size(a, 3)
         do j = 1, size(a, 2)
            do i = 1, size(a, 1)
               max_diff = max(max_diff, abs(a(i, j, k) - b(i, j, k)))
               max_ref = max(max_ref, abs(a(i, j, k)))
            end do
         end do
      end do

      is_close = max_diff <= tol*max_ref

      if (.not. is_close) then
         print *, "max_diff:", max_diff
         print *, "max_ref:", max_ref
         print *, "tol:", tol
         print *, "relative error:", max_diff/max_ref
      end if
   end function compare_with_relative_error_d_3d

end program test_multidim

