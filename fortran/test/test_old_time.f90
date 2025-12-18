program main
  use sparse_ir
  use sparse_ir_io
  use sparse_ir_preset
  implicit none
  integer, parameter :: num = 5000000

  call test_efficiency_f(.false., .false., num, 1)
  call test_efficiency_f(.false., .false., num, 10)
  call test_efficiency_f(.false., .false., num, 20)
  call test_efficiency_f(.false., .false., num, 50)
  call test_efficiency_f(.false., .false., num, 100)
  call test_efficiency_f(.false., .false., num, 200)
  call test_efficiency_f(.false., .false., num, 12)
  call test_efficiency_f(.false., .false., num, 24)
  call test_efficiency_f(.false., .false., num, 60)
  call test_efficiency_f(.false., .false., num, 120)
  !
  call test_efficiency_f(.false., .true., num, 1)
  call test_efficiency_f(.false., .true., num, 10)
  call test_efficiency_f(.false., .true., num, 20)
  call test_efficiency_f(.false., .true., num, 50)
  call test_efficiency_f(.false., .true., num, 100)
  call test_efficiency_f(.false., .true., num, 200)
  call test_efficiency_f(.false., .true., num, 12)
  call test_efficiency_f(.false., .true., num, 24)
  call test_efficiency_f(.false., .true., num, 60)
  call test_efficiency_f(.false., .true., num, 120)

  contains

  ! fermion
  subroutine test_efficiency_f(preset, positive_only, num, lsize_ir)
      logical, intent(in) :: preset
      logical, intent(in) :: positive_only
      integer, intent(in) :: num
      integer, intent(in) :: lsize_ir
      type(IR) :: ir_obj
      integer, parameter :: ndigit = 10, nlambda = 4
      double precision, parameter :: lambda = 1.d1 ** nlambda
      double precision, parameter :: wmax = 1.d0
      double precision :: PI

      double precision, parameter :: beta = lambda/wmax, omega0 = 1.d0/beta
      double precision, parameter :: eps = 1.d-1**ndigit

      complex(kind(0d0)),allocatable :: giv(:,:), gl_matsu(:, :), gl_tau(:, :), &
          giv_reconst(:, :), gtau_reconst(:, :)
      double precision,allocatable :: gl_matsu_d(:, :), gl_tau_d(:, :), gtau_reconst_d(:, :)
      integer :: n, t, l, i, ix, n_lowest
      integer :: time_begin_c, time_end_c, CountPerSec, CountMax
      double precision :: sum_of_giv
      complex(kind(0d0)), PARAMETER :: cone  = (1.0d0, 0.0d0)
      complex(kind(0d0)), PARAMETER :: ci  = (0.0d0, 1.0d0)
      complex(kind(0d0)), PARAMETER :: czero  = (0.0d0, 0.0d0)
      double precision, PARAMETER :: one = 1.0d0
      double precision, PARAMETER :: zero = 0.0d0

      PI =4.D0*DATAN(1.D0)

      if (preset) then
          ir_obj = mk_ir_preset(nlambda, ndigit, beta, positive_only)
      else
          open(99, file='ir_nlambda4_ndigit10.dat', status='old')
          ir_obj = read_ir(99, beta, positive_only)
          close(99)
      end if

      if (abs(ir_obj%beta - beta) > 1d-10) then
          write(*,*) "beta does not match"
          stop 1
      end if
      if (abs(ir_obj%wmax - wmax) > 1d-10) then
          write(*,*) "wmax does not match"
          stop 1
      end if

      allocate(giv(lsize_ir, ir_obj%nfreq_f))
      allocate(giv_reconst(lsize_ir, ir_obj%nfreq_f))
      allocate(gl_matsu(lsize_ir, ir_obj%size))
      allocate(gl_tau(lsize_ir, ir_obj%size))
      allocate(gtau_reconst(lsize_ir, ir_obj%ntau))
      allocate(gl_matsu_d(lsize_ir, ir_obj%size))
      allocate(gl_tau_d(lsize_ir, ir_obj%size))
      allocate(gtau_reconst_d(lsize_ir, ir_obj%ntau))
      sum_of_giv = zero

      write(*,*) "test_efficiency_f"
      write(*,*) "preset = ", preset
      write(*,*) "positive_only = ", positive_only
      write(*,*) "num =", num
      write(*,*) "lsize_ir =", lsize_ir

      if (positive_only) then
          n_lowest = 1
      else
          n_lowest = ir_obj%nfreq_f / 2 + 1 
      end if
      giv(1, 1) = 1.d0/(cmplx(0d0, PI*ir_obj%freq_f(n_lowest)/beta, kind(0d0)) - omega0)
      sum_of_giv = sum_of_giv + REAL(giv(1, 1), kind(0d0))
      sum_of_giv = sum_of_giv + AIMAG(giv(1, 1))
      sum_of_giv = REAL(num, kind(0d0)) * sum_of_giv
      write(*,*) "estimated sum_of_giv      =", sum_of_giv

      if (positive_only) then
          giv(:, :) = czero
          gl_matsu_d(:, :) = zero
          gtau_reconst_d(:, :) = zero
          gl_tau_d(:, :) = zero
          giv_reconst(:, :) = czero
      else
          giv(:, :) = czero
          gl_matsu(:, :) = czero
          gtau_reconst(:, :) = czero
          gl_tau(:, :) = czero
          giv_reconst(:, :) = czero
      end if
      sum_of_giv = zero

      call system_clock(time_begin_c, CountPerSec, CountMax)
      call sleep(1)

      call system_clock(time_begin_c)
      do i = 1, num
          ix = MOD(i - 1, lsize_ir) + 1
          do n = 1, ir_obj%nfreq_f
              giv(ix, n) = 1.d0/(cmplx(0d0, PI*ir_obj%freq_f(n)/beta, kind(0d0)) - omega0)
          end do
          if ((ix == lsize_ir) .OR. (i == num)) THEN
              if (positive_only) then
                  call fit_matsubara_f(ir_obj, giv, gl_matsu_d)
                  call evaluate_tau(ir_obj, gl_matsu_d, gtau_reconst_d)
                  call fit_tau(ir_obj, gtau_reconst_d, gl_tau_d)
                  call evaluate_matsubara_f(ir_obj, gl_tau_d, giv_reconst)
                  sum_of_giv = sum_of_giv + SUM(REAL(giv_reconst(1:ix, n_lowest), kind(0d0)))
                  sum_of_giv = sum_of_giv + SUM(AIMAG(giv_reconst(1:ix, n_lowest)))
                  giv(:, :) = czero
                  gl_matsu_d(:, :) = zero
                  gtau_reconst_d(:, :) = zero
                  gl_tau_d(:, :) = zero
                  giv_reconst(:, :) = czero
              else
                  call fit_matsubara_f(ir_obj, giv, gl_matsu)
                  call evaluate_tau(ir_obj, gl_matsu, gtau_reconst)
                  call fit_tau(ir_obj, gtau_reconst, gl_tau)
                  call evaluate_matsubara_f(ir_obj, gl_tau, giv_reconst)
                  sum_of_giv = sum_of_giv + SUM(REAL(giv_reconst(1:ix, n_lowest), kind(0d0)))
                  sum_of_giv = sum_of_giv + SUM(AIMAG(giv_reconst(1:ix, n_lowest)))
                  giv(:, :) = czero
                  gl_matsu(:, :) = czero
                  gtau_reconst(:, :) = czero
                  gl_tau(:, :) = czero
                  giv_reconst(:, :) = czero
              end if
          end if
      end do
      call system_clock(time_end_c)
      write(*,*) "test result of sum_of_giv =", sum_of_giv
      write(*,*) real(time_end_c - time_begin_c)/CountPerSec," sec"
      call sleep(1)

      deallocate(giv, gl_matsu, gl_tau, gtau_reconst, giv_reconst)
      deallocate(gl_matsu_d, gl_tau_d, gtau_reconst_d)

      call finalize_ir(ir_obj)
  end subroutine

end program