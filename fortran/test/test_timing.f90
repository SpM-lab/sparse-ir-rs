!-----------------------------------------------------------------------
PROGRAM test_timing
  !-----------------------------------------------------------------------
  !!
  !! This program measures timing performance of sparse-ir-rs Fortran bindings.
  !!
  !! Usage: ./test_timing [nlambda] [ndigit] [positive_only] [statistics] 
  !!                      [lreal_ir] [lreal_tau] [num] [lsize_ir]
  !!
  !! Default: nlambda=6, ndigit=8, positive_only=T, statistics=F(ermion),
  !!          lreal_ir=T, lreal_tau=T, num=185640, lsize_ir=1
  !!
  !! Statistics: F or Fermion for Fermionic, B or Boson for Bosonic
  !!
  !! Output format: Tab-separated values for easy Excel import
  !!
  !! Note: This version uses constant omega0 for all j to enable efficient
  !!       timing measurement with minimal SYSTEM_CLOCK calls.
  !!
  USE, INTRINSIC :: iso_c_binding, ONLY : c_int32_t
  USE sparse_ir_extension
  USE sparse_ir_c, ONLY : SPIR_STATISTICS_FERMIONIC, SPIR_STATISTICS_BOSONIC
  !
  IMPLICIT NONE
  !
  INTEGER, PARAMETER :: DP = KIND(1.0D0)
  REAL(KIND = DP), PARAMETER :: one = 1.0D0
  REAL(KIND = DP), PARAMETER :: zero = 0.0D0
  REAL(KIND = DP), PARAMETER :: pi = 4.D0 * ATAN(1.D0)
  COMPLEX(KIND = DP), PARAMETER :: cone = (1.0D0, 0.0D0)
  COMPLEX(KIND = DP), PARAMETER :: ci = (0.0D0, 1.0D0)
  COMPLEX(KIND = DP), PARAMETER :: czero = (0.0D0, 0.0D0)
  !
  INTEGER, PARAMETER :: TARGET_DIM = 2
  !
  ! Default values
  INTEGER :: nlambda = 6
  INTEGER :: ndigit = 8
  LOGICAL :: positive_only = .TRUE.
  INTEGER(c_int32_t) :: statistics = SPIR_STATISTICS_FERMIONIC
  LOGICAL :: lreal_ir = .TRUE.
  LOGICAL :: lreal_tau = .TRUE.
  INTEGER :: num = 185640
  INTEGER :: lsize_ir = 1
  !
  CHARACTER(LEN=256) :: arg
  INTEGER :: nargs
  !
  ! Parse command line arguments
  nargs = COMMAND_ARGUMENT_COUNT()
  !
  IF (nargs >= 1) THEN
    CALL GET_COMMAND_ARGUMENT(1, arg)
    READ(arg, *) nlambda
  END IF
  !
  IF (nargs >= 2) THEN
    CALL GET_COMMAND_ARGUMENT(2, arg)
    READ(arg, *) ndigit
  END IF
  !
  IF (nargs >= 3) THEN
    CALL GET_COMMAND_ARGUMENT(3, arg)
    IF (arg == 'T' .OR. arg == 't' .OR. arg == 'true' .OR. arg == 'TRUE' .OR. arg == '.TRUE.') THEN
      positive_only = .TRUE.
    ELSE
      positive_only = .FALSE.
    END IF
  END IF
  !
  IF (nargs >= 4) THEN
    CALL GET_COMMAND_ARGUMENT(4, arg)
    IF (arg == 'F' .OR. arg == 'f' .OR. arg == 'Fermion' .OR. arg == 'fermion' .OR. arg == 'FERMION') THEN
      statistics = SPIR_STATISTICS_FERMIONIC
    ELSE IF (arg == 'B' .OR. arg == 'b' .OR. arg == 'Boson' .OR. arg == 'boson' .OR. arg == 'BOSON') THEN
      statistics = SPIR_STATISTICS_BOSONIC
    ELSE
      WRITE(*,*) 'ERROR: Invalid statistics. Use F/Fermion or B/Boson'
      STOP 1
    END IF
  END IF
  !
  IF (nargs >= 5) THEN
    CALL GET_COMMAND_ARGUMENT(5, arg)
    IF (arg == 'T' .OR. arg == 't' .OR. arg == 'true' .OR. arg == 'TRUE' .OR. arg == '.TRUE.') THEN
      lreal_ir = .TRUE.
    ELSE
      lreal_ir = .FALSE.
    END IF
  END IF
  !
  IF (nargs >= 6) THEN
    CALL GET_COMMAND_ARGUMENT(6, arg)
    IF (arg == 'T' .OR. arg == 't' .OR. arg == 'true' .OR. arg == 'TRUE' .OR. arg == '.TRUE.') THEN
      lreal_tau = .TRUE.
    ELSE
      lreal_tau = .FALSE.
    END IF
  END IF
  !
  IF (nargs >= 7) THEN
    CALL GET_COMMAND_ARGUMENT(7, arg)
    READ(arg, *) num
  END IF
  !
  IF (nargs >= 8) THEN
    CALL GET_COMMAND_ARGUMENT(8, arg)
    READ(arg, *) lsize_ir
  END IF
  !
  ! Error check: if positive_only = false, both arrays must be complex
  IF (.NOT. positive_only .AND. (lreal_ir .OR. lreal_tau)) THEN
    WRITE(*,*) 'ERROR: When positive_only = false, both lreal_ir and lreal_tau must be false'
    STOP 1
  END IF
  !
  ! Error check: num must be divisible by lsize_ir
  IF (MOD(num, lsize_ir) /= 0) THEN
    WRITE(*,*) 'ERROR: num must be divisible by lsize_ir'
    WRITE(*,'(A,I10,A,I10,A,I10)') '  num = ', num, ', lsize_ir = ', lsize_ir, &
                                   ', remainder = ', MOD(num, lsize_ir)
    STOP 1
  END IF
  !
  ! Run timing test
  CALL run_timing_test(nlambda, ndigit, positive_only, statistics, &
                       lreal_ir, lreal_tau, num, lsize_ir)
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  SUBROUTINE run_timing_test(nlambda, ndigit, positive_only, statistics, &
                             lreal_ir, lreal_tau, num, lsize_ir)
  !-----------------------------------------------------------------------
    IMPLICIT NONE
    !
    INTEGER, INTENT(IN) :: nlambda
    INTEGER, INTENT(IN) :: ndigit
    LOGICAL, INTENT(IN) :: positive_only
    INTEGER(c_int32_t), INTENT(IN) :: statistics
    LOGICAL, INTENT(IN) :: lreal_ir
    LOGICAL, INTENT(IN) :: lreal_tau
    INTEGER, INTENT(IN) :: num
    INTEGER, INTENT(IN) :: lsize_ir
    !
    TYPE(IR) :: ir_obj
    REAL(KIND = DP) :: lambda, wmax, beta, omega0, eps
    !
    ! Arrays for transformations (small size: lsize_ir)
    COMPLEX(KIND = DP), ALLOCATABLE :: giv(:,:)
    COMPLEX(KIND = DP), ALLOCATABLE :: giv_reconst(:,:)
    COMPLEX(KIND = DP), ALLOCATABLE :: gl(:,:)
    REAL(KIND = DP), ALLOCATABLE :: gl_d(:,:)
    COMPLEX(KIND = DP), ALLOCATABLE :: gtau(:,:)
    REAL(KIND = DP), ALLOCATABLE :: gtau_d(:,:)
    !
    INTEGER :: nfreq, n, iter, num_iters
    !
    ! Timing variables
    INTEGER :: time_begin, time_end, count_rate, count_max
    REAL(KIND = DP) :: time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu
    REAL(KIND = DP) :: time_total, time_per_vector
    REAL(KIND = DP) :: sum_of_giv_expected, sum_of_giv_computed, relative_error, tol
    CHARACTER(LEN=16) :: stat_name
    INTEGER(KIND=8) :: freq_1
    !
    lambda = 1.d1 ** nlambda
    beta = 1.d2  ! Fixed inverse temperature
    wmax = lambda / beta
    omega0 = 1.d0 / beta  ! Constant omega0 for all j
    eps = 1.d-1 ** ndigit
    !
    IF (statistics == SPIR_STATISTICS_FERMIONIC) THEN
      stat_name = 'Fermion'
    ELSE
      stat_name = 'Boson'
    END IF
    !
    ! Print header information
    WRITE(*,*) '================================================'
    WRITE(*,*) 'Timing Test for sparse-ir-rs Fortran Bindings'
    WRITE(*,*) '(Simplified version with constant omega0)'
    WRITE(*,*) '================================================'
    WRITE(*,'(A,I3,A,ES10.2)') '  nlambda = ', nlambda, ', lambda = ', lambda
    WRITE(*,'(A,I3,A,ES10.2)') '  ndigit  = ', ndigit, ', eps = ', eps
    WRITE(*,'(A,L1)') '  positive_only = ', positive_only
    WRITE(*,'(A,A)') '  statistics = ', TRIM(stat_name)
    WRITE(*,'(A,L1)') '  lreal_ir = ', lreal_ir
    WRITE(*,'(A,L1)') '  lreal_tau = ', lreal_tau
    WRITE(*,'(A,I10)') '  num = ', num
    WRITE(*,'(A,I10)') '  lsize_ir = ', lsize_ir
    WRITE(*,'(A,ES12.4)') '  beta = ', beta
    WRITE(*,'(A,ES12.4)') '  omega0 = ', omega0
    WRITE(*,*) '================================================'
    !
    ! Initialize IR object
    CALL init_ir(ir_obj, beta, lambda, eps, positive_only)
    !
    ! Determine nfreq based on statistics
    IF (statistics == SPIR_STATISTICS_FERMIONIC) THEN
      nfreq = ir_obj%nfreq_f
      freq_1 = ir_obj%freq_f(1)
    ELSE
      nfreq = ir_obj%nfreq_b
      freq_1 = ir_obj%freq_b(1)
    END IF
    !
    num_iters = num / lsize_ir
    !
    WRITE(*,'(A,I4)') '  IR basis size = ', ir_obj%size
    WRITE(*,'(A,I4)') '  nfreq = ', nfreq
    WRITE(*,'(A,I4)') '  ntau = ', ir_obj%ntau
    WRITE(*,'(A,I10)') '  num_iterations = ', num_iters
    WRITE(*,*) '================================================'
    !
    ! Pre-calculate expected sum_of_giv (just real + imag of giv(1,1))
    sum_of_giv_expected = &
      REAL(one / (CMPLX(zero, pi * freq_1 / beta, KIND = DP) - omega0), KIND = DP) + &
      AIMAG(one / (CMPLX(zero, pi * freq_1 / beta, KIND = DP) - omega0))
    !
    ! Allocate arrays (small size: lsize_ir)
    ALLOCATE(giv(lsize_ir, nfreq))
    ALLOCATE(giv_reconst(lsize_ir, nfreq))
    ALLOCATE(gl(lsize_ir, ir_obj%size))
    ALLOCATE(gl_d(lsize_ir, ir_obj%size))
    ALLOCATE(gtau(lsize_ir, ir_obj%ntau))
    ALLOCATE(gtau_d(lsize_ir, ir_obj%ntau))
    !
    ! Initialize giv with constant omega0 (same for all batch elements)
    IF (statistics == SPIR_STATISTICS_FERMIONIC) THEN
      DO n = 1, nfreq
        giv(:, n) = one / (CMPLX(zero, pi * ir_obj%freq_f(n) / beta, KIND = DP) - omega0)
      ENDDO
    ELSE
      DO n = 1, nfreq
        giv(:, n) = one / (CMPLX(zero, pi * ir_obj%freq_b(n) / beta, KIND = DP) - omega0)
      ENDDO
    END IF
    !
    ! Initialize other arrays
    giv_reconst(:, :) = czero
    gl(:, :) = czero
    gl_d(:, :) = zero
    gtau(:, :) = czero
    gtau_d(:, :) = zero
    !
    ! Get clock rate
    CALL SYSTEM_CLOCK(count_rate=count_rate, count_max=count_max)
    !
    ! Branch based on array types
    IF (positive_only .AND. lreal_ir .AND. lreal_tau) THEN
      CALL timing_loop_dd(ir_obj, statistics, num_iters, count_rate, &
                          giv, gl_d, gtau_d, giv_reconst, &
                          time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
    ELSE IF (positive_only .AND. lreal_ir .AND. .NOT. lreal_tau) THEN
      CALL timing_loop_dc(ir_obj, statistics, num_iters, count_rate, &
                          giv, gl_d, gtau_d, gtau, giv_reconst, &
                          time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
    ELSE IF (positive_only .AND. .NOT. lreal_ir .AND. lreal_tau) THEN
      CALL timing_loop_cd(ir_obj, statistics, num_iters, count_rate, &
                          giv, gl, gtau, gtau_d, giv_reconst, &
                          time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
    ELSE
      CALL timing_loop_cc(ir_obj, statistics, num_iters, count_rate, &
                          giv, gl, gtau, giv_reconst, &
                          time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
    END IF
    !
    ! Compute sum_of_giv for verification (just real + imag of giv_reconst(1,1))
    sum_of_giv_computed = REAL(giv_reconst(1, 1), KIND = DP) + AIMAG(giv_reconst(1, 1))
    !
    ! Calculate relative error
    relative_error = ABS(sum_of_giv_computed - sum_of_giv_expected) / ABS(sum_of_giv_expected)
    !
    ! Check relative error against tolerance (same as test_analytic_gf)
    ! tol = 100 * eps, which is 1.d-6 for default ndigit=8
    tol = 1.d2 * eps
    IF (relative_error > tol) THEN
      WRITE(*,*) 'FAILED: Relative error exceeds tolerance!'
      WRITE(*,'(A,ES16.6)') '  Relative error = ', relative_error
      WRITE(*,'(A,ES16.6)') '  Tolerance     = ', tol
      STOP 1
    ENDIF
    !
    ! Calculate total time and per-vector time
    time_total = time_fit_matsu + time_eval_tau + time_fit_tau + time_eval_matsu
    time_per_vector = time_total / REAL(num, KIND = DP)
    !
    ! Print detailed results
    WRITE(*,*) ''
    WRITE(*,*) '==================== Results ===================='
    WRITE(*,'(A,ES16.6)') '  sum_of_giv (expected) = ', sum_of_giv_expected
    WRITE(*,'(A,ES16.6)') '  sum_of_giv (computed) = ', sum_of_giv_computed
    WRITE(*,'(A,ES16.6)') '  Relative error        = ', relative_error
    WRITE(*,*) ''
    WRITE(*,*) '  --- Timing Results ---'
    WRITE(*,'(A,F12.6,A)') '  fit_matsubara time      = ', time_fit_matsu, ' sec'
    WRITE(*,'(A,F12.6,A)') '  evaluate_tau time       = ', time_eval_tau, ' sec'
    WRITE(*,'(A,F12.6,A)') '  fit_tau time            = ', time_fit_tau, ' sec'
    WRITE(*,'(A,F12.6,A)') '  evaluate_matsubara time = ', time_eval_matsu, ' sec'
    WRITE(*,'(A,F12.6,A)') '  Total time              = ', time_total, ' sec'
    WRITE(*,'(A,ES16.6,A)') '  Time per vector         = ', time_per_vector, ' sec'
    WRITE(*,*) '================================================'
    !
    ! Print Excel-friendly output (tab-separated)
    WRITE(*,*) ''
    WRITE(*,*) '==================== Excel Output (Tab-Separated) ===================='
    WRITE(*,*) 'Copy the following lines to Excel:'
    WRITE(*,*) ''
    WRITE(*,*) '--- Header ---'
    WRITE(*,'(A)') 'nlambda' // CHAR(9) // 'ndigit' // CHAR(9) // 'positive_only' // CHAR(9) // &
                  'statistics' // CHAR(9) // 'lreal_ir' // CHAR(9) // 'lreal_tau' // CHAR(9) // &
                  'num' // CHAR(9) // 'lsize_ir' // CHAR(9) // 'IR_size' // CHAR(9) // &
                  'nfreq' // CHAR(9) // 'ntau' // CHAR(9) // &
                  'fit_matsu(s)' // CHAR(9) // 'eval_tau(s)' // CHAR(9) // &
                  'fit_tau(s)' // CHAR(9) // 'eval_matsu(s)' // CHAR(9) // &
                  'total(s)' // CHAR(9) // 'per_vector(s)' // CHAR(9) // 'rel_error'
    WRITE(*,*) ''
    WRITE(*,*) '--- Data ---'
    WRITE(*,'(I3,A,I3,A,L1,A,A,A,L1,A,L1,A,I10,A,I10,A,I4,A,I4,A,I4,A)', ADVANCE='NO') &
      nlambda, CHAR(9), ndigit, CHAR(9), positive_only, CHAR(9), &
      TRIM(stat_name), CHAR(9), lreal_ir, CHAR(9), lreal_tau, CHAR(9), &
      num, CHAR(9), lsize_ir, CHAR(9), ir_obj%size, CHAR(9), &
      nfreq, CHAR(9), ir_obj%ntau, CHAR(9)
    WRITE(*,'(ES12.6,A,ES12.6,A,ES12.6,A,ES12.6,A,ES12.6,A,ES12.6,A,ES12.6)') &
      time_fit_matsu, CHAR(9), time_eval_tau, CHAR(9), &
      time_fit_tau, CHAR(9), time_eval_matsu, CHAR(9), &
      time_total, CHAR(9), time_per_vector, CHAR(9), relative_error
    WRITE(*,*) ''
    WRITE(*,*) '======================================================================'
    !
    ! Deallocate arrays
    DEALLOCATE(giv, giv_reconst)
    DEALLOCATE(gl, gl_d)
    DEALLOCATE(gtau, gtau_d)
    !
    CALL finalize_ir(ir_obj)
    !
  END SUBROUTINE run_timing_test
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  SUBROUTINE timing_loop_dd(ir_obj, statistics, num_iters, count_rate, &
                            giv, gl_d, gtau_d, giv_reconst, &
                            time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
  !-----------------------------------------------------------------------
    !! Timing loop for Real IR (gl_d) and Real tau (gtau_d)
    !! SYSTEM_CLOCK is called only twice per subroutine (before and after loop)
    !
    TYPE(IR), INTENT(IN) :: ir_obj
    INTEGER(c_int32_t), INTENT(IN) :: statistics
    INTEGER, INTENT(IN) :: num_iters, count_rate
    COMPLEX(KIND = DP), INTENT(IN) :: giv(:,:)
    REAL(KIND = DP), INTENT(INOUT) :: gl_d(:,:), gtau_d(:,:)
    COMPLEX(KIND = DP), INTENT(OUT) :: giv_reconst(:,:)
    REAL(KIND = DP), INTENT(OUT) :: time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu
    !
    INTEGER :: iter, time_begin, time_end
    !
    ! fit_matsubara: G(iω) -> G(l)
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_matsubara(ir_obj, statistics, TARGET_DIM, giv, gl_d)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    ! evaluate_tau: G(l) -> G(τ)
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_tau(ir_obj, statistics, TARGET_DIM, gl_d, gtau_d)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    ! fit_tau: G(τ) -> G(l)
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_tau(ir_obj, statistics, TARGET_DIM, gtau_d, gl_d)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    ! evaluate_matsubara: G(l) -> G(iω)
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_matsubara(ir_obj, statistics, TARGET_DIM, gl_d, giv_reconst)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
  END SUBROUTINE timing_loop_dd
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  SUBROUTINE timing_loop_dc(ir_obj, statistics, num_iters, count_rate, &
                            giv, gl_d, gtau_d, gtau, giv_reconst, &
                            time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
  !-----------------------------------------------------------------------
    !! Timing loop for Real IR (gl_d) and Complex tau (gtau)
    !
    TYPE(IR), INTENT(IN) :: ir_obj
    INTEGER(c_int32_t), INTENT(IN) :: statistics
    INTEGER, INTENT(IN) :: num_iters, count_rate
    COMPLEX(KIND = DP), INTENT(IN) :: giv(:,:)
    REAL(KIND = DP), INTENT(INOUT) :: gl_d(:,:), gtau_d(:,:)
    COMPLEX(KIND = DP), INTENT(INOUT) :: gtau(:,:)
    COMPLEX(KIND = DP), INTENT(OUT) :: giv_reconst(:,:)
    REAL(KIND = DP), INTENT(OUT) :: time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu
    !
    INTEGER :: iter, time_begin, time_end
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_matsubara(ir_obj, statistics, TARGET_DIM, giv, gl_d)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_tau(ir_obj, statistics, TARGET_DIM, gl_d, gtau_d)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    gtau(:, :) = CMPLX(gtau_d, zero, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_tau(ir_obj, statistics, TARGET_DIM, gtau_d, gl_d)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_matsubara(ir_obj, statistics, TARGET_DIM, gl_d, giv_reconst)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
  END SUBROUTINE timing_loop_dc
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  SUBROUTINE timing_loop_cd(ir_obj, statistics, num_iters, count_rate, &
                            giv, gl, gtau, gtau_d, giv_reconst, &
                            time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
  !-----------------------------------------------------------------------
    !! Timing loop for Complex IR (gl) and Real tau (gtau_d)
    !
    TYPE(IR), INTENT(IN) :: ir_obj
    INTEGER(c_int32_t), INTENT(IN) :: statistics
    INTEGER, INTENT(IN) :: num_iters, count_rate
    COMPLEX(KIND = DP), INTENT(IN) :: giv(:,:)
    COMPLEX(KIND = DP), INTENT(INOUT) :: gl(:,:), gtau(:,:)
    REAL(KIND = DP), INTENT(INOUT) :: gtau_d(:,:)
    COMPLEX(KIND = DP), INTENT(OUT) :: giv_reconst(:,:)
    REAL(KIND = DP), INTENT(OUT) :: time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu
    !
    INTEGER :: iter, time_begin, time_end
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_matsubara(ir_obj, statistics, TARGET_DIM, giv, gl)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_tau(ir_obj, statistics, TARGET_DIM, gl, gtau)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    gtau_d(:, :) = REAL(gtau, KIND = DP)
    !
    gtau(:, :) = CMPLX(gtau_d, zero, KIND = DP)
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_tau(ir_obj, statistics, TARGET_DIM, gtau, gl)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_matsubara(ir_obj, statistics, TARGET_DIM, gl, giv_reconst)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
  END SUBROUTINE timing_loop_cd
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  SUBROUTINE timing_loop_cc(ir_obj, statistics, num_iters, count_rate, &
                            giv, gl, gtau, giv_reconst, &
                            time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu)
  !-----------------------------------------------------------------------
    !! Timing loop for Complex IR (gl) and Complex tau (gtau)
    !
    TYPE(IR), INTENT(IN) :: ir_obj
    INTEGER(c_int32_t), INTENT(IN) :: statistics
    INTEGER, INTENT(IN) :: num_iters, count_rate
    COMPLEX(KIND = DP), INTENT(IN) :: giv(:,:)
    COMPLEX(KIND = DP), INTENT(INOUT) :: gl(:,:), gtau(:,:)
    COMPLEX(KIND = DP), INTENT(OUT) :: giv_reconst(:,:)
    REAL(KIND = DP), INTENT(OUT) :: time_fit_matsu, time_eval_tau, time_fit_tau, time_eval_matsu
    !
    INTEGER :: iter, time_begin, time_end
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_matsubara(ir_obj, statistics, TARGET_DIM, giv, gl)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_tau(ir_obj, statistics, TARGET_DIM, gl, gtau)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL fit_tau(ir_obj, statistics, TARGET_DIM, gtau, gl)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_fit_tau = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
    CALL SYSTEM_CLOCK(time_begin)
    DO iter = 1, num_iters
      CALL evaluate_matsubara(ir_obj, statistics, TARGET_DIM, gl, giv_reconst)
    ENDDO
    CALL SYSTEM_CLOCK(time_end)
    time_eval_matsu = REAL(time_end - time_begin, KIND = DP) / REAL(count_rate, KIND = DP)
    !
  END SUBROUTINE timing_loop_cc
  !-----------------------------------------------------------------------
  !
!-----------------------------------------------------------------------
END PROGRAM test_timing
!-----------------------------------------------------------------------
