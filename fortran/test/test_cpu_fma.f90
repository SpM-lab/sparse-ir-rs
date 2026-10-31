! The CPU check of #332 in the Fortran binding (sparse-ir-rs#353)
!
! A library built with FMA cannot run on a CPU without it: the first numerical
! call dies with SIGILL. The binding must say so instead. The rule is exposed
! as a function of the two flags so that all four combinations can be tested
! anywhere; `init_ir` calls it with the real ones before it touches a kernel.
PROGRAM test_cpu_fma
   USE, INTRINSIC :: iso_c_binding
   USE sparse_ir_c
   USE sparse_ir_extension
   IMPLICIT NONE

   INTEGER :: nfail

   nfail = 0

   ! Both queries answer, and this process is running, so the loaded library
   ! agrees with this CPU
   CALL expect_in(c_spir_build_uses_fma(), 0, 1, 'build_uses_fma is 0 or 1')
   CALL expect_in(c_spir_cpu_has_fma(), 0, 1, 'cpu_has_fma is 0 or 1')

   ! The rule, for every combination (the Python binding tests the same table)
   CALL expect(.NOT. fma_combination_is_supported(1_c_int, 0_c_int), &
               'FMA build on a CPU without FMA is refused')
   CALL expect(fma_combination_is_supported(1_c_int, 1_c_int), &
               'FMA build on a CPU with FMA is accepted')
   CALL expect(fma_combination_is_supported(0_c_int, 0_c_int), &
               'non-FMA build on any CPU is accepted')
   CALL expect(fma_combination_is_supported(0_c_int, 1_c_int), &
               'non-FMA build on a CPU with FMA is accepted')

   ! The real check passes here (this process is running the library)
   CALL check_cpu_fma('test_cpu_fma')

   WRITE (*, '(A)') 'test_cpu_fma: PASSED'
   IF (nfail /= 0) THEN
      WRITE (*, '(A,I0)') 'test_cpu_fma: FAILED, checks failed: ', nfail
      ERROR STOP 1
   END IF

CONTAINS

   SUBROUTINE expect(condition, what)
      LOGICAL, INTENT(IN) :: condition
      CHARACTER(LEN=*), INTENT(IN) :: what
      IF (condition) THEN
         WRITE (*, '(A,A)') 'ok   ', what
      ELSE
         WRITE (*, '(A,A)') 'FAIL ', what
         nfail = nfail + 1
      END IF
   END SUBROUTINE expect

   SUBROUTINE expect_in(value, lo, hi, what)
      INTEGER(KIND=c_int), INTENT(IN) :: value
      INTEGER, INTENT(IN) :: lo, hi
      CHARACTER(LEN=*), INTENT(IN) :: what
      CALL expect(value >= lo .AND. value <= hi, what)
   END SUBROUTINE expect_in

END PROGRAM test_cpu_fma
