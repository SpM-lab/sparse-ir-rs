!-----------------------------------------------------------------------
PROGRAM test_analytic_gf
  !-----------------------------------------------------------------------
  !!
  !! This program tests the sparse-ir-rs Fortran bindings (sparse_ir_extension)
  !! using analytically known Green's functions.
  !!
  !! Test scenarios:
  !!   Test 1: G(iω) -> G(l) -> G(iω)
  !!   Test 2: G(iω) -> G(l) -> G(τ) -> G(l) -> G(iω)
  !!   Test 3: Compare G1 (from iω) and G2 (from τ):
  !!           G1(iω) -> G1(l) -> G1(τ)
  !!           G2(τ)  -> G2(l) -> G2(iω)
  !!           Compare: G1(iω) vs G2(iω), G1(l) vs G2(l), G1(τ) vs G2(τ)
  !!
  !! Analytic Green's function:
  !!   G(iν) = 1/(iν - ω0)
  !!   G(τ)  = -exp(-τ ω0)/(1 ± exp(-β ω0))  (+ for Fermion, - for Boson)
  !!
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
  INTEGER, PARAMETER :: TARGET_DIM = 2  ! Second dimension (index 2, 1-indexed) is the basis/tau/freq dimension
  !
  WRITE(*,*) '================================================'
  WRITE(*,*) 'Testing analytic Green function transformations'
  WRITE(*,*) '================================================'
  !
  ! Test with smaller lambda (should be easier)
  CALL test_fermion(2, 8, .FALSE., .FALSE., .FALSE., 1)  ! lambda=10^2, eps=10^-8, complex arrays
  CALL test_boson  (2, 8, .FALSE., .FALSE., .FALSE., 1)
  CALL test_fermion(2, 8, .TRUE., .TRUE., .TRUE., 1)  ! positive_only, real arrays
  CALL test_boson  (2, 8, .TRUE., .TRUE., .TRUE., 1)
  !
  ! Test with larger batch size (like EPW with siz_ir ~ 100)
  WRITE(*,*) ''
  WRITE(*,*) '================================================'
  WRITE(*,*) 'Testing with larger batch size (like EPW)'
  WRITE(*,*) '================================================'
  CALL test_fermion(5, 6, .FALSE., .FALSE., .FALSE., 100)  ! lambda=10^5, eps=10^-6, batch_size=100
  CALL test_fermion(5, 6, .TRUE., .TRUE., .TRUE., 100)
  !
  WRITE(*,*) ''
  WRITE(*,*) '================================================'
  WRITE(*,*) 'All tests passed successfully!'
  WRITE(*,*) '================================================'
  !
CONTAINS
  !
  !-----------------------------------------------------------------------
  SUBROUTINE test_fermion(nlambda, ndigit, positive_only, lreal_ir, lreal_tau, batch_size)
  !-----------------------------------------------------------------------
    !!
    !! This routine performs tests for fermionic Green's function.
    !!
    IMPLICIT NONE
    !
    INTEGER, INTENT(IN) :: nlambda
    !! Power of lambda: lambda = 10^nlambda
    INTEGER, INTENT(IN) :: ndigit
    !! Number of digits for epsilon: eps = 10^-ndigit
    LOGICAL, INTENT(IN) :: positive_only
    !! If true, use only positive Matsubara frequencies
    LOGICAL, INTENT(IN) :: lreal_ir
    !! If true, use real arrays for IR coefficients (g_ir)
    LOGICAL, INTENT(IN) :: lreal_tau
    !! If true, use real arrays for tau values (gtau)
    INTEGER, INTENT(IN) :: batch_size
    !! Batch size (number of states/bands/k-points)
    !
    TYPE(IR) :: ir_obj
    REAL(KIND = DP) :: lambda, wmax, beta, omega0, eps, tol
    !
    ! Arrays for Test 1 & 2
    COMPLEX(KIND = DP), ALLOCATABLE :: giv(:,:)
    !! Green's function at Matsubara frequencies (always complex)
    COMPLEX(KIND = DP), ALLOCATABLE :: giv_reconst1(:,:)
    !! Reconstructed G(iω) after Test 1
    COMPLEX(KIND = DP), ALLOCATABLE :: giv_reconst2(:,:)
    !! Reconstructed G(iω) after Test 2
    COMPLEX(KIND = DP), ALLOCATABLE :: gl(:,:)
    !! IR coefficients (complex)
    REAL(KIND = DP), ALLOCATABLE :: gl_d(:,:)
    !! IR coefficients (real, for positive_only)
    COMPLEX(KIND = DP), ALLOCATABLE :: gl_tmp(:,:)
    !! Temporary IR coefficients for type conversion
    COMPLEX(KIND = DP), ALLOCATABLE :: gtau(:,:)
    !! G(τ) (complex)
    REAL(KIND = DP), ALLOCATABLE :: gtau_d(:,:)
    !! G(τ) (real, for positive_only)
    !
    ! Arrays for Test 3 (comparison between Matsubara and tau starting points)
    COMPLEX(KIND = DP), ALLOCATABLE :: g1_iw(:,:), g1_l(:,:), g1_tau(:,:)
    COMPLEX(KIND = DP), ALLOCATABLE :: g2_iw(:,:), g2_l(:,:), g2_tau(:,:)
    REAL(KIND = DP), ALLOCATABLE :: g1_l_d(:,:), g1_tau_d(:,:)
    REAL(KIND = DP), ALLOCATABLE :: g2_l_d(:,:), g2_tau_d(:,:)
    !
    INTEGER :: n, t, j
    REAL(KIND = DP) :: max_diff1, max_diff2, max_val
    REAL(KIND = DP) :: max_diff_iw, max_diff_l, max_diff_tau
    REAL(KIND = DP) :: max_val_iw, max_val_l, max_val_tau
    REAL(KIND = DP) :: omega0_j
    !
    lambda = 1.d1 ** nlambda
    wmax = 1.d0
    beta = lambda / wmax
    omega0 = 1.d0 / beta
    eps = 1.d-1 ** ndigit
    tol = 1.d2 * eps  ! Tolerance for comparison
    !
    WRITE(*,*) ''
    WRITE(*,*) '================================================'
    WRITE(*,*) 'Testing Fermion with:'
    WRITE(*,'(A,I3,A,ES10.2)') '  nlambda = ', nlambda, ', lambda = ', lambda
    WRITE(*,'(A,I3,A,ES10.2)') '  ndigit  = ', ndigit, ', eps = ', eps
    WRITE(*,'(A,L1)') '  positive_only = ', positive_only
    WRITE(*,'(A,L1)') '  lreal_ir = ', lreal_ir
    WRITE(*,'(A,L1)') '  lreal_tau = ', lreal_tau
    WRITE(*,'(A,I6)') '  batch_size = ', batch_size
    WRITE(*,'(A,ES12.4)') '  beta = ', beta
    WRITE(*,'(A,ES12.4)') '  omega0 = ', omega0
    WRITE(*,*) '================================================'
    !
    ! Initialize IR object
    CALL init_ir(ir_obj, beta, lambda, eps, positive_only)
    !
    WRITE(*,'(A,I4)') '  IR basis size = ', ir_obj%size
    WRITE(*,'(A,I4)') '  nfreq_f = ', ir_obj%nfreq_f
    WRITE(*,'(A,I4)') '  ntau = ', ir_obj%ntau
    !
    ! Allocate arrays for Test 1 & 2
    ALLOCATE(giv(batch_size, ir_obj%nfreq_f))
    ALLOCATE(giv_reconst1(batch_size, ir_obj%nfreq_f))
    ALLOCATE(giv_reconst2(batch_size, ir_obj%nfreq_f))
    !
    ! Build analytic G(iν) = 1/(iν - ω0_j) for each batch element
    DO j = 1, batch_size
      omega0_j = omega0 * (1.0D0 + 0.1D0 * (j - 1) / batch_size)  ! Slightly different omega0 for each state
      DO n = 1, ir_obj%nfreq_f
        giv(j, n) = one / (CMPLX(zero, pi * ir_obj%freq_f(n) / beta, KIND = DP) - omega0_j)
      ENDDO
    ENDDO
    !
    IF (positive_only) THEN
      !
      IF (.NOT. lreal_ir) THEN
        ALLOCATE(gl(batch_size, ir_obj%size))
      ELSE
        ALLOCATE(gl_d(batch_size, ir_obj%size))
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
      ELSE
        ALLOCATE(gtau_d(batch_size, ir_obj%ntau))
      END IF
      !
      ! Test 1: G(iω) -> G(l) -> G(iω)
      IF (.NOT. lreal_ir) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, giv, gl)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, giv_reconst1)
      ELSE
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, giv, gl_d)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl_d, giv_reconst1)
      END IF
      !
      ! Test 2: G(iω) -> G(l) -> G(τ) -> G(l) -> G(iω)
      IF (.NOT. lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, gtau)
        gl(:, :) = czero
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gtau, gl)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, giv_reconst2)
      ELSE IF (.NOT. lreal_ir .AND. lreal_tau) THEN
        ! Use intermediate complex array for tau, then convert to real
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, gtau)
        gtau_d(:, :) = REAL(gtau, KIND = DP)
        DEALLOCATE(gtau)
        gl(:, :) = czero
        ! Use intermediate complex array for fit_tau (fit_tau needs matching types)
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        gtau(:, :) = CMPLX(gtau_d, zero, KIND = DP)
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gtau, gl_tmp)
        gl(:, :) = gl_tmp(:, :)
        DEALLOCATE(gl_tmp, gtau)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, giv_reconst2)
      ELSE IF (lreal_ir .AND. .NOT. lreal_tau) THEN
        ! Use intermediate real array for tau, then convert to complex
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl_d, gtau_d)
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        gtau(:, :) = CMPLX(gtau_d, zero, KIND = DP)
        gl_d(:, :) = zero
        ! Use intermediate complex array for fit_tau, then convert to real
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gtau, gl_tmp)
        gl_d(:, :) = REAL(gl_tmp, KIND = DP)
        DEALLOCATE(gl_tmp, gtau)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl_d, giv_reconst2)
      ELSE
        ! Both real
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl_d, gtau_d)
        gl_d(:, :) = zero
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gtau_d, gl_d)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl_d, giv_reconst2)
      END IF
      !
      IF (.NOT. lreal_ir) THEN
        DEALLOCATE(gl)
      ELSE
        DEALLOCATE(gl_d)
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        IF (ALLOCATED(gtau)) DEALLOCATE(gtau)
      ELSE
        IF (ALLOCATED(gtau_d)) DEALLOCATE(gtau_d)
      END IF
      !
    ELSE
      !
      ! positive_only = false: must use complex arrays
      ALLOCATE(gl(batch_size, ir_obj%size))
      ALLOCATE(gtau(batch_size, ir_obj%ntau))
      !
      ! Test 1: G(iω) -> G(l) -> G(iω)
      CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, giv, gl)
      CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, giv_reconst1)
      !
      ! Test 2: G(iω) -> G(l) -> G(τ) -> G(l) -> G(iω)
      CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, gtau)
      gl(:, :) = czero
      CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gtau, gl)
      CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gl, giv_reconst2)
      !
      DEALLOCATE(gl, gtau)
      !
    ENDIF
    !
    ! Compute errors for Test 1 & 2
    max_diff1 = MAXVAL(ABS(giv - giv_reconst1))
    max_diff2 = MAXVAL(ABS(giv - giv_reconst2))
    max_val = MAXVAL(ABS(giv))
    !
    WRITE(*,'(A,ES16.6)') '  Test 1 (G(iw)->G(l)->G(iw)) max_diff = ', max_diff1
    WRITE(*,'(A,ES16.6)') '  Test 1 relative error = ', max_diff1 / max_val
    WRITE(*,'(A,ES16.6)') '  Test 2 (G(iw)->G(l)->G(tau)->G(l)->G(iw)) max_diff = ', max_diff2
    WRITE(*,'(A,ES16.6)') '  Test 2 relative error = ', max_diff2 / max_val
    WRITE(*,'(A,ES16.6)') '  Tolerance = ', tol
    !
    IF (max_diff1 > tol) THEN
      WRITE(*,*) 'FAILED: Test 1 error exceeds tolerance!'
      STOP 1
    ENDIF
      !
    IF (max_diff2 > tol) THEN
      WRITE(*,*) 'FAILED: Test 2 error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    WRITE(*,*) '  Test 1 & 2: PASSED'
    !
    DEALLOCATE(giv_reconst1, giv_reconst2)
    !
    !-----------------------------------------------------------------------
    ! Test 3: Compare G1 (starting from iω) and G2 (starting from τ)
    !-----------------------------------------------------------------------
    WRITE(*,*) ''
    WRITE(*,*) '  --- Test 3: Comparison between Matsubara and tau ---'
    !
    ! Allocate arrays for Test 3
    ALLOCATE(g1_iw(batch_size, ir_obj%nfreq_f))
    ALLOCATE(g2_iw(batch_size, ir_obj%nfreq_f))
    ALLOCATE(g1_tau(batch_size, ir_obj%ntau))
    ALLOCATE(g2_tau(batch_size, ir_obj%ntau))
    !
    ! G1: Start from analytic G1(iω) = 1/(iω - ω0)
    ! (giv is already computed above, use first batch element)
    DO j = 1, batch_size
      omega0_j = omega0 * (1.0D0 + 0.1D0 * (j - 1) / batch_size)
      DO n = 1, ir_obj%nfreq_f
        g1_iw(j, n) = one / (CMPLX(zero, pi * ir_obj%freq_f(n) / beta, KIND = DP) - omega0_j)
      ENDDO
    ENDDO
    !
    ! G2: Start from analytic G2(τ) for Fermion
    ! sparse-ir-rs uses τ ∈ [-β/2, β/2]
    ! Fermionic Green's function has anti-periodicity: G(τ - β) = -G(τ)
    ! For τ ∈ [0, β]:   G(τ) = -exp(-τ ω0)/(1 + exp(-β ω0))
    ! For τ ∈ [-β, 0]:  G(τ) = -G(τ + β) = +exp(-(τ+β) ω0)/(1 + exp(-β ω0))
    DO j = 1, batch_size
      omega0_j = omega0 * (1.0D0 + 0.1D0 * (j - 1) / batch_size)
      DO t = 1, ir_obj%ntau
        IF (ir_obj%tau(t) >= zero) THEN
          ! τ >= 0: G(τ) = -exp(-τ ω0)/(1 + exp(-β ω0))
          g2_tau(j, t) = - EXP(-ir_obj%tau(t) * omega0_j) / (one + EXP(-beta * omega0_j))
        ELSE
          ! τ < 0: G(τ) = -G(τ + β) = +exp(-(τ+β) ω0)/(1 + exp(-β ω0))
          g2_tau(j, t) = + EXP(-(ir_obj%tau(t) + beta) * omega0_j) / (one + EXP(-beta * omega0_j))
        ENDIF
      ENDDO
    ENDDO
    !
    IF (positive_only) THEN
      !
      IF (.NOT. lreal_ir) THEN
        ALLOCATE(g1_l(batch_size, ir_obj%size))
        ALLOCATE(g2_l(batch_size, ir_obj%size))
      ELSE
        ALLOCATE(g1_l_d(batch_size, ir_obj%size))
        ALLOCATE(g2_l_d(batch_size, ir_obj%size))
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        ALLOCATE(g1_tau(batch_size, ir_obj%ntau))
        ALLOCATE(g2_tau(batch_size, ir_obj%ntau))
      ELSE
        ALLOCATE(g1_tau_d(batch_size, ir_obj%ntau))
        ALLOCATE(g2_tau_d(batch_size, ir_obj%ntau))
      END IF
      !
      ! G1: iω -> l -> τ
      IF (.NOT. lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_iw, g1_l)
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_l, g1_tau)
      ELSE IF (.NOT. lreal_ir .AND. lreal_tau) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_iw, g1_l)
        ! Use intermediate complex array for evaluate_tau, then convert to real
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_l, gtau)
        g1_tau_d(:, :) = REAL(gtau, KIND = DP)
        g1_tau(:, :) = CMPLX(g1_tau_d, zero, KIND = DP)
        DEALLOCATE(gtau)
      ELSE IF (lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_iw, g1_l_d)
        ! Use intermediate real array for evaluate_tau, then convert to complex
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_l_d, g1_tau_d)
        g1_tau(:, :) = CMPLX(g1_tau_d, zero, KIND = DP)
      ELSE
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_iw, g1_l_d)
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_l_d, g1_tau_d)
        g1_tau(:, :) = CMPLX(g1_tau_d, zero, KIND = DP)
      END IF
      !
      ! G2: τ -> l -> iω
      IF (.NOT. lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_tau, g2_l)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_l, g2_iw)
      ELSE IF (.NOT. lreal_ir .AND. lreal_tau) THEN
        ! Use intermediate complex array for fit_tau
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        gtau(:, :) = CMPLX(REAL(g2_tau, KIND = DP), zero, KIND = DP)
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, gtau, gl_tmp)
        g2_l(:, :) = gl_tmp(:, :)
        DEALLOCATE(gl_tmp, gtau)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_l, g2_iw)
      ELSE IF (lreal_ir .AND. .NOT. lreal_tau) THEN
        ! Use intermediate complex array for fit_tau, then convert to real
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_tau, gl_tmp)
        g2_l_d(:, :) = REAL(gl_tmp, KIND = DP)
        DEALLOCATE(gl_tmp)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_l_d, g2_iw)
      ELSE
        g2_tau_d(:, :) = REAL(g2_tau, KIND = DP)
        CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_tau_d, g2_l_d)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_l_d, g2_iw)
      END IF
      !
      ! Compute differences
      max_diff_iw = MAXVAL(ABS(g1_iw - g2_iw))
      max_val_iw = MAXVAL(ABS(g1_iw))
      !
      IF (.NOT. lreal_ir) THEN
        max_diff_l = MAXVAL(ABS(g1_l - g2_l))
        max_val_l = MAXVAL(ABS(g1_l))
      ELSE
        max_diff_l = MAXVAL(ABS(g1_l_d - g2_l_d))
        max_val_l = MAXVAL(ABS(g1_l_d))
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        max_diff_tau = MAXVAL(ABS(g1_tau - g2_tau))
        max_val_tau = MAXVAL(ABS(g1_tau))
      ELSE
        max_diff_tau = MAXVAL(ABS(g1_tau_d - g2_tau_d))
        max_val_tau = MAXVAL(ABS(g1_tau_d))
      END IF
      !
      IF (.NOT. lreal_ir) THEN
        DEALLOCATE(g1_l, g2_l)
      ELSE
        DEALLOCATE(g1_l_d, g2_l_d)
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        DEALLOCATE(g1_tau, g2_tau)
      ELSE
        DEALLOCATE(g1_tau_d, g2_tau_d)
      END IF
      !
    ELSE
      !
      ! positive_only = false: must use complex arrays
      ALLOCATE(g1_l(batch_size, ir_obj%size))
      ALLOCATE(g2_l(batch_size, ir_obj%size))
      !
      ! G1: iω -> l -> τ
      CALL fit_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_iw, g1_l)
      CALL evaluate_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g1_l, g1_tau)
      !
      ! G2: τ -> l -> iω
      CALL fit_tau(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_tau, g2_l)
      CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_FERMIONIC, TARGET_DIM, g2_l, g2_iw)
      !
      ! Compute differences
      max_diff_iw = MAXVAL(ABS(g1_iw - g2_iw))
      max_val_iw = MAXVAL(ABS(g1_iw))
      max_diff_l = MAXVAL(ABS(g1_l - g2_l))
      max_val_l = MAXVAL(ABS(g1_l))
      max_diff_tau = MAXVAL(ABS(g1_tau - g2_tau))
      max_val_tau = MAXVAL(ABS(g1_tau))
      !
      DEALLOCATE(g1_l, g2_l)
      !
    ENDIF
    !
    WRITE(*,'(A,ES16.6)') '  G1(iw) vs G2(iw) max_diff = ', max_diff_iw
    WRITE(*,'(A,ES16.6)') '  G1(iw) vs G2(iw) relative error = ', max_diff_iw / max_val_iw
    WRITE(*,'(A,ES16.6)') '  G1(l) vs G2(l) max_diff = ', max_diff_l
    WRITE(*,'(A,ES16.6)') '  G1(l) vs G2(l) relative error = ', max_diff_l / max_val_l
    WRITE(*,'(A,ES16.6)') '  G1(tau) vs G2(tau) max_diff = ', max_diff_tau
    WRITE(*,'(A,ES16.6)') '  G1(tau) vs G2(tau) relative error = ', max_diff_tau / max_val_tau
    !
    ! Use relative error for comparison (more appropriate for large values)
    IF (max_diff_iw / max_val_iw > tol) THEN
      WRITE(*,*) 'FAILED: G(iw) comparison relative error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    IF (max_diff_l / max_val_l > tol) THEN
      WRITE(*,*) 'FAILED: G(l) comparison relative error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    IF (max_diff_tau / max_val_tau > tol) THEN
      WRITE(*,*) 'FAILED: G(tau) comparison relative error exceeds tolerance!'
      STOP 1
    ENDIF 
    !
    WRITE(*,*) '  Test 3: PASSED'
    !
    DEALLOCATE(giv, g1_iw, g2_iw, g1_tau, g2_tau)
    CALL finalize_ir(ir_obj)
    !
  !-----------------------------------------------------------------------
  END SUBROUTINE test_fermion
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
  SUBROUTINE test_boson(nlambda, ndigit, positive_only, lreal_ir, lreal_tau, batch_size)
  !-----------------------------------------------------------------------
    !!
    !! This routine performs tests for bosonic Green's function.
    !!
    IMPLICIT NONE
    !
    INTEGER, INTENT(IN) :: nlambda
    !! Power of lambda: lambda = 10^nlambda
    INTEGER, INTENT(IN) :: ndigit
    !! Number of digits for epsilon: eps = 10^-ndigit
    LOGICAL, INTENT(IN) :: positive_only
    !! If true, use only positive Matsubara frequencies
    LOGICAL, INTENT(IN) :: lreal_ir
    !! If true, use real arrays for IR coefficients (g_ir)
    LOGICAL, INTENT(IN) :: lreal_tau
    !! If true, use real arrays for tau values (gtau)
    INTEGER, INTENT(IN) :: batch_size
    !! Batch size (number of states/bands/k-points)
    !
    TYPE(IR) :: ir_obj
    REAL(KIND = DP) :: lambda, wmax, beta, omega0, eps, tol
    !
    ! Arrays for Test 1 & 2
    COMPLEX(KIND = DP), ALLOCATABLE :: giv(:,:)
    !! Green's function at Matsubara frequencies (always complex)
    COMPLEX(KIND = DP), ALLOCATABLE :: giv_reconst1(:,:)
    !! Reconstructed G(iω) after Test 1
    COMPLEX(KIND = DP), ALLOCATABLE :: giv_reconst2(:,:)
    !! Reconstructed G(iω) after Test 2
    COMPLEX(KIND = DP), ALLOCATABLE :: gl(:,:)
    !! IR coefficients (complex)
    REAL(KIND = DP), ALLOCATABLE :: gl_d(:,:)
    !! IR coefficients (real, for positive_only)
    COMPLEX(KIND = DP), ALLOCATABLE :: gl_tmp(:,:)
    !! Temporary IR coefficients for type conversion
    COMPLEX(KIND = DP), ALLOCATABLE :: gtau(:,:)
    !! G(τ) (complex)
    REAL(KIND = DP), ALLOCATABLE :: gtau_d(:,:)
    !! G(τ) (real, for positive_only)
    !
    ! Arrays for Test 3 (comparison between Matsubara and tau starting points)
    COMPLEX(KIND = DP), ALLOCATABLE :: g1_iw(:,:), g1_l(:,:), g1_tau(:,:)
    COMPLEX(KIND = DP), ALLOCATABLE :: g2_iw(:,:), g2_l(:,:), g2_tau(:,:)
    REAL(KIND = DP), ALLOCATABLE :: g1_l_d(:,:), g1_tau_d(:,:)
    REAL(KIND = DP), ALLOCATABLE :: g2_l_d(:,:), g2_tau_d(:,:)
    !
    INTEGER :: n, t, j
    REAL(KIND = DP) :: max_diff1, max_diff2, max_val
    REAL(KIND = DP) :: max_diff_iw, max_diff_l, max_diff_tau
    REAL(KIND = DP) :: max_val_iw, max_val_l, max_val_tau
    REAL(KIND = DP) :: omega0_j
    !
    ! Error check: if positive_only = false, both arrays must be complex
    IF (.NOT. positive_only .AND. (lreal_ir .OR. lreal_tau)) THEN
      WRITE(*,*) 'ERROR: When positive_only = false, both lreal_ir and lreal_tau must be false'
      STOP 1
    END IF
    !
    lambda = 1.d1 ** nlambda
    wmax = 1.d0
    beta = lambda / wmax
    omega0 = 1.d0 / beta
    eps = 1.d-1 ** ndigit
    tol = 1.d2 * eps  ! Tolerance for comparison
    !
    WRITE(*,*) ''
    WRITE(*,*) '================================================'
    WRITE(*,*) 'Testing Boson with:'
    WRITE(*,'(A,I3,A,ES10.2)') '  nlambda = ', nlambda, ', lambda = ', lambda
    WRITE(*,'(A,I3,A,ES10.2)') '  ndigit  = ', ndigit, ', eps = ', eps
    WRITE(*,'(A,L1)') '  positive_only = ', positive_only
    WRITE(*,'(A,L1)') '  lreal_ir = ', lreal_ir
    WRITE(*,'(A,L1)') '  lreal_tau = ', lreal_tau
    WRITE(*,'(A,I6)') '  batch_size = ', batch_size
    WRITE(*,'(A,ES12.4)') '  beta = ', beta
    WRITE(*,'(A,ES12.4)') '  omega0 = ', omega0
    WRITE(*,*) '================================================'
    !
    ! Initialize IR object
    CALL init_ir(ir_obj, beta, lambda, eps, positive_only)
    !
    WRITE(*,'(A,I4)') '  IR basis size = ', ir_obj%size
    WRITE(*,'(A,I4)') '  nfreq_b = ', ir_obj%nfreq_b
    WRITE(*,'(A,I4)') '  ntau = ', ir_obj%ntau
    !
    ! Allocate arrays for Test 1 & 2
      ALLOCATE(giv(batch_size, ir_obj%nfreq_b))
    ALLOCATE(giv_reconst1(batch_size, ir_obj%nfreq_b))
    ALLOCATE(giv_reconst2(batch_size, ir_obj%nfreq_b))
    !
    ! Build analytic G(iν) = 1/(iν - ω0_j) for each batch element
    DO j = 1, batch_size
      omega0_j = omega0 * (1.0D0 + 0.1D0 * (j - 1) / batch_size)  ! Slightly different omega0 for each state
      DO n = 1, ir_obj%nfreq_b
        giv(j, n) = one / (CMPLX(zero, pi * ir_obj%freq_b(n) / beta, KIND = DP) - omega0_j)
      ENDDO
    ENDDO
    !
    IF (positive_only) THEN
      !
      IF (.NOT. lreal_ir) THEN
        ALLOCATE(gl(batch_size, ir_obj%size))
      ELSE
        ALLOCATE(gl_d(batch_size, ir_obj%size))
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
      ELSE
        ALLOCATE(gtau_d(batch_size, ir_obj%ntau))
      END IF
      !
      ! Test 1: G(iω) -> G(l) -> G(iω)
      IF (.NOT. lreal_ir) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, giv, gl)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, giv_reconst1)
      ELSE
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, giv, gl_d)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl_d, giv_reconst1)
      END IF
      !
      ! Test 2: G(iω) -> G(l) -> G(τ) -> G(l) -> G(iω)
      IF (.NOT. lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, gtau)
        gl(:, :) = czero
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gtau, gl)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, giv_reconst2)
      ELSE IF (.NOT. lreal_ir .AND. lreal_tau) THEN
        ! Use intermediate complex array for tau, then convert to real
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, gtau)
        gtau_d(:, :) = REAL(gtau, KIND = DP)
        DEALLOCATE(gtau)
        gl(:, :) = czero
        ! Use intermediate complex array for fit_tau (fit_tau needs matching types)
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        gtau(:, :) = CMPLX(gtau_d, zero, KIND = DP)
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gtau, gl_tmp)
        gl(:, :) = gl_tmp(:, :)
        DEALLOCATE(gl_tmp, gtau)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, giv_reconst2)
      ELSE IF (lreal_ir .AND. .NOT. lreal_tau) THEN
        ! Use intermediate real array for tau, then convert to complex
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl_d, gtau_d)
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        gtau(:, :) = CMPLX(gtau_d, zero, KIND = DP)
        gl_d(:, :) = zero
        ! Use intermediate complex array for fit_tau, then convert to real
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gtau, gl_tmp)
        gl_d(:, :) = REAL(gl_tmp, KIND = DP)
        DEALLOCATE(gl_tmp, gtau)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl_d, giv_reconst2)
      ELSE
        ! Both real
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl_d, gtau_d)
        gl_d(:, :) = zero
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gtau_d, gl_d)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl_d, giv_reconst2)
      END IF
      !
      IF (.NOT. lreal_ir) THEN
        DEALLOCATE(gl)
      ELSE
        DEALLOCATE(gl_d)
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        IF (ALLOCATED(gtau)) DEALLOCATE(gtau)
      ELSE
        IF (ALLOCATED(gtau_d)) DEALLOCATE(gtau_d)
      END IF
      !
    ELSE
      !
      ! positive_only = false: must use complex arrays
      ALLOCATE(gl(batch_size, ir_obj%size))
      ALLOCATE(gtau(batch_size, ir_obj%ntau))
      !
      ! Test 1: G(iω) -> G(l) -> G(iω)
      CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, giv, gl)
      CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, giv_reconst1)
      !
      ! Test 2: G(iω) -> G(l) -> G(τ) -> G(l) -> G(iω)
      CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, gtau)
      gl(:, :) = czero
      CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gtau, gl)
      CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gl, giv_reconst2)
      !
      DEALLOCATE(gl, gtau)
      !
    ENDIF
    !
    ! Compute errors for Test 1 & 2
    max_diff1 = MAXVAL(ABS(giv - giv_reconst1))
    max_diff2 = MAXVAL(ABS(giv - giv_reconst2))
    max_val = MAXVAL(ABS(giv))
    !
    WRITE(*,'(A,ES16.6)') '  Test 1 (G(iw)->G(l)->G(iw)) max_diff = ', max_diff1
    WRITE(*,'(A,ES16.6)') '  Test 1 relative error = ', max_diff1 / max_val
    WRITE(*,'(A,ES16.6)') '  Test 2 (G(iw)->G(l)->G(tau)->G(l)->G(iw)) max_diff = ', max_diff2
    WRITE(*,'(A,ES16.6)') '  Test 2 relative error = ', max_diff2 / max_val
    WRITE(*,'(A,ES16.6)') '  Tolerance = ', tol
    !
    IF (max_diff1 > tol) THEN
      WRITE(*,*) 'FAILED: Test 1 error exceeds tolerance!'
      STOP 1
    ENDIF 
    !
    IF (max_diff2 > tol) THEN
      WRITE(*,*) 'FAILED: Test 2 error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    WRITE(*,*) '  Test 1 & 2: PASSED'
    !
    DEALLOCATE(giv_reconst1, giv_reconst2)
    !
    !-----------------------------------------------------------------------
    ! Test 3: Compare G1 (starting from iω) and G2 (starting from τ)
    !-----------------------------------------------------------------------
    WRITE(*,*) ''
    WRITE(*,*) '  --- Test 3: Comparison between Matsubara and tau ---'
    !
    ! Allocate arrays for Test 3
    ALLOCATE(g1_iw(batch_size, ir_obj%nfreq_b))
    ALLOCATE(g2_iw(batch_size, ir_obj%nfreq_b))
    ALLOCATE(g1_tau(batch_size, ir_obj%ntau))
    ALLOCATE(g2_tau(batch_size, ir_obj%ntau))
    !
    ! G1: Start from analytic G1(iω) = 1/(iω - ω0)
    ! (giv is already computed above, use first batch element)
    DO j = 1, batch_size
      omega0_j = omega0 * (1.0D0 + 0.1D0 * (j - 1) / batch_size)
      DO n = 1, ir_obj%nfreq_b
        g1_iw(j, n) = one / (CMPLX(zero, pi * ir_obj%freq_b(n) / beta, KIND = DP) - omega0_j)
      ENDDO
    ENDDO
    !
    ! G2: Start from analytic G2(τ) for Boson
    ! sparse-ir-rs uses τ ∈ [-β/2, β/2]
    ! Bosonic Green's function has periodicity: G(τ - β) = G(τ)
    ! For τ ∈ [0, β]:   G(τ) = -exp(-τ ω0)/(1 - exp(-β ω0))
    ! For τ ∈ [-β, 0]:  G(τ) = G(τ + β) = -exp(-(τ+β) ω0)/(1 - exp(-β ω0))
    DO j = 1, batch_size
      omega0_j = omega0 * (1.0D0 + 0.1D0 * (j - 1) / batch_size)
      DO t = 1, ir_obj%ntau
        IF (ir_obj%tau(t) >= zero) THEN
          ! τ >= 0: G(τ) = -exp(-τ ω0)/(1 - exp(-β ω0))
          g2_tau(j, t) = - EXP(-ir_obj%tau(t) * omega0_j) / (one - EXP(-beta * omega0_j))
        ELSE
          ! τ < 0: G(τ) = G(τ + β) = -exp(-(τ+β) ω0)/(1 - exp(-β ω0))
          g2_tau(j, t) = - EXP(-(ir_obj%tau(t) + beta) * omega0_j) / (one - EXP(-beta * omega0_j))
        ENDIF
      ENDDO
    ENDDO
    !
    IF (positive_only) THEN
      !
      IF (.NOT. lreal_ir) THEN
        ALLOCATE(g1_l(batch_size, ir_obj%size))
        ALLOCATE(g2_l(batch_size, ir_obj%size))
      ELSE
        ALLOCATE(g1_l_d(batch_size, ir_obj%size))
        ALLOCATE(g2_l_d(batch_size, ir_obj%size))
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        ALLOCATE(g1_tau(batch_size, ir_obj%ntau))
        ALLOCATE(g2_tau(batch_size, ir_obj%ntau))
      ELSE
        ALLOCATE(g1_tau_d(batch_size, ir_obj%ntau))
        ALLOCATE(g2_tau_d(batch_size, ir_obj%ntau))
      END IF
      !
      ! G1: iω -> l -> τ
      IF (.NOT. lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_iw, g1_l)
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_l, g1_tau)
      ELSE IF (.NOT. lreal_ir .AND. lreal_tau) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_iw, g1_l)
        ! Use intermediate complex array for evaluate_tau, then convert to real
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_l, gtau)
        g1_tau_d(:, :) = REAL(gtau, KIND = DP)
        g1_tau(:, :) = CMPLX(g1_tau_d, zero, KIND = DP)
        DEALLOCATE(gtau)
      ELSE IF (lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_iw, g1_l_d)
        ! Use intermediate real array for evaluate_tau, then convert to complex
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_l_d, g1_tau_d)
        g1_tau(:, :) = CMPLX(g1_tau_d, zero, KIND = DP)
      ELSE
        CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_iw, g1_l_d)
        CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_l_d, g1_tau_d)
        g1_tau(:, :) = CMPLX(g1_tau_d, zero, KIND = DP)
      END IF
      !
      ! G2: τ -> l -> iω
      IF (.NOT. lreal_ir .AND. .NOT. lreal_tau) THEN
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_tau, g2_l)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_l, g2_iw)
      ELSE IF (.NOT. lreal_ir .AND. lreal_tau) THEN
        ! Use intermediate complex array for fit_tau
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        ALLOCATE(gtau(batch_size, ir_obj%ntau))
        gtau(:, :) = CMPLX(REAL(g2_tau, KIND = DP), zero, KIND = DP)
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, gtau, gl_tmp)
        g2_l(:, :) = gl_tmp(:, :)
        DEALLOCATE(gl_tmp, gtau)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_l, g2_iw)
      ELSE IF (lreal_ir .AND. .NOT. lreal_tau) THEN
        ! Use intermediate complex array for fit_tau, then convert to real
        ALLOCATE(gl_tmp(batch_size, ir_obj%size))
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_tau, gl_tmp)
        g2_l_d(:, :) = REAL(gl_tmp, KIND = DP)
        DEALLOCATE(gl_tmp)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_l_d, g2_iw)
      ELSE
        g2_tau_d(:, :) = REAL(g2_tau, KIND = DP)
        CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_tau_d, g2_l_d)
        CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_l_d, g2_iw)
      END IF
      !
      ! Compute differences
      max_diff_iw = MAXVAL(ABS(g1_iw - g2_iw))
      max_val_iw = MAXVAL(ABS(g1_iw))
      !
      IF (.NOT. lreal_ir) THEN
        max_diff_l = MAXVAL(ABS(g1_l - g2_l))
        max_val_l = MAXVAL(ABS(g1_l))
      ELSE
        max_diff_l = MAXVAL(ABS(g1_l_d - g2_l_d))
        max_val_l = MAXVAL(ABS(g1_l_d))
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        max_diff_tau = MAXVAL(ABS(g1_tau - g2_tau))
        max_val_tau = MAXVAL(ABS(g1_tau))
      ELSE
        max_diff_tau = MAXVAL(ABS(g1_tau_d - g2_tau_d))
        max_val_tau = MAXVAL(ABS(g1_tau_d))
      END IF
      !
      IF (.NOT. lreal_ir) THEN
        DEALLOCATE(g1_l, g2_l)
      ELSE
        DEALLOCATE(g1_l_d, g2_l_d)
      END IF
      !
      IF (.NOT. lreal_tau) THEN
        DEALLOCATE(g1_tau, g2_tau)
      ELSE
        DEALLOCATE(g1_tau_d, g2_tau_d)
      END IF
      !
    ELSE
      !
      ! positive_only = false: must use complex arrays
      ALLOCATE(g1_l(batch_size, ir_obj%size))
      ALLOCATE(g2_l(batch_size, ir_obj%size))
      !
      ! G1: iω -> l -> τ
      CALL fit_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_iw, g1_l)
      CALL evaluate_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g1_l, g1_tau)
      !
      ! G2: τ -> l -> iω
      CALL fit_tau(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_tau, g2_l)
      CALL evaluate_matsubara(ir_obj, SPIR_STATISTICS_BOSONIC, TARGET_DIM, g2_l, g2_iw)
      !
      ! Compute differences
      max_diff_iw = MAXVAL(ABS(g1_iw - g2_iw))
      max_val_iw = MAXVAL(ABS(g1_iw))
      max_diff_l = MAXVAL(ABS(g1_l - g2_l))
      max_val_l = MAXVAL(ABS(g1_l))
      max_diff_tau = MAXVAL(ABS(g1_tau - g2_tau))
      max_val_tau = MAXVAL(ABS(g1_tau))
      !
      DEALLOCATE(g1_l, g2_l)
      !
    ENDIF
    !
    WRITE(*,'(A,ES16.6)') '  G1(iw) vs G2(iw) max_diff = ', max_diff_iw
    WRITE(*,'(A,ES16.6)') '  G1(iw) vs G2(iw) relative error = ', max_diff_iw / max_val_iw
    WRITE(*,'(A,ES16.6)') '  G1(l) vs G2(l) max_diff = ', max_diff_l
    WRITE(*,'(A,ES16.6)') '  G1(l) vs G2(l) relative error = ', max_diff_l / max_val_l
    WRITE(*,'(A,ES16.6)') '  G1(tau) vs G2(tau) max_diff = ', max_diff_tau
    WRITE(*,'(A,ES16.6)') '  G1(tau) vs G2(tau) relative error = ', max_diff_tau / max_val_tau
    !
    ! Use relative error for comparison (more appropriate for large values)
    IF (max_diff_iw / max_val_iw > tol) THEN
      WRITE(*,*) 'FAILED: G(iw) comparison relative error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    IF (max_diff_l / max_val_l > tol) THEN
      WRITE(*,*) 'FAILED: G(l) comparison relative error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    IF (max_diff_tau / max_val_tau > tol) THEN
      WRITE(*,*) 'FAILED: G(tau) comparison relative error exceeds tolerance!'
      STOP 1
    ENDIF
    !
    WRITE(*,*) '  Test 3: PASSED'
    !
    DEALLOCATE(giv, g1_iw, g2_iw, g1_tau, g2_tau)
    CALL finalize_ir(ir_obj)
    !
  !-----------------------------------------------------------------------
  END SUBROUTINE test_boson
  !-----------------------------------------------------------------------
  !
  !-----------------------------------------------------------------------
END PROGRAM test_analytic_gf
!-----------------------------------------------------------------------

