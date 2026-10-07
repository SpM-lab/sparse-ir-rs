! Test the independent DLR constructor and the MiniPole C bindings
program test_minipole
   use sparse_ir_c
   use, intrinsic :: iso_c_binding
   implicit none

   print *, "======================================"
   print *, "Testing independent DLR and MiniPole"
   print *, "======================================"

   call test_dlr_independent()
   call test_minipole_from_matsubara()

   print *, "======================================"
   print *, "All MiniPole tests passed!"
   print *, "======================================"

contains

   subroutine test_dlr_independent()
      type(c_ptr) :: dlr
      integer(c_int), target :: status, npoles, ntaus, nmatsus

      dlr = c_spir_dlr_new_independent(SPIR_STATISTICS_FERMIONIC, 40.0_c_double, &
                                       1.0_c_double, 1.0e-12_c_double, c_loc(status))
      if (status /= 0 .or. .not. c_associated(dlr)) then
         print *, "Error: spir_dlr_new_independent failed", status
         stop 1
      end if
      status = c_spir_dlr_get_npoles(dlr, c_loc(npoles))
      status = c_spir_basis_get_n_default_taus(dlr, c_loc(ntaus))
      status = c_spir_basis_get_n_default_matsus(dlr, .false._c_bool, c_loc(nmatsus))
      if (npoles <= 0 .or. ntaus /= npoles .or. nmatsus /= npoles) then
         print *, "Error: DLR default points", npoles, ntaus, nmatsus
         stop 1
      end if
      call c_spir_basis_release(dlr)
      print *, "  Independent DLR default points: PASSED"
   end subroutine test_dlr_independent

   subroutine test_minipole_from_matsubara()
      integer, parameter :: nf = 120
      real(c_double), parameter :: beta = 40.0_c_double
      real(c_double), parameter :: pi = 3.14159265358979323846_c_double
      real(c_double), parameter :: xs(2) = [-0.5_c_double, 0.3_c_double]
      real(c_double), parameter :: amps(2) = [0.4_c_double, 0.6_c_double]
      integer(c_int64_t), target :: ns(nf)
      complex(c_double_complex), target :: values(nf), poles(2), residues(2), cst(1)
      integer(c_int), target :: dims(1), status, npoles, n0
      real(c_double), target :: err_max
      type(c_ptr) :: rep
      integer :: i, j
      complex(c_double_complex) :: z

      ! Uniform non-negative fermionic grid 1, 3, 5, ...
      do i = 1, nf
         ns(i) = 2_c_int64_t*(i - 1) + 1_c_int64_t
         z = cmplx(0.0_c_double, pi*real(ns(i), c_double)/beta, kind=c_double_complex)
         values(i) = sum(amps/(z - xs))
      end do
      dims(1) = nf

      ! n0 chosen from the data (-1), err = 1e-8 (absolute), defaults otherwise
      rep = c_spir_minipole_from_matsubara(beta, int(nf, c_int), c_loc(ns), &
                                           SPIR_ORDER_COLUMN_MAJOR, 1_c_int, c_loc(dims), &
                                           0_c_int, c_loc(values), -1_c_int, 0_c_int, &
                                           1.0e-8_c_double, 0_c_int, 0_c_int, .false._c_bool, &
                                           .false._c_bool, .false._c_bool, -1_c_int, &
                                           .false._c_bool, 0_c_int, 0.0_c_double, c_loc(status))
      if (status /= 0 .or. .not. c_associated(rep)) then
         print *, "Error: spir_minipole_from_matsubara failed", status
         stop 1
      end if
      status = c_spir_pole_repr_get_npoles(rep, c_loc(npoles))
      if (npoles /= 2) then
         print *, "Error: expected 2 poles, got", npoles
         stop 1
      end if
      status = c_spir_pole_repr_get_poles(rep, c_loc(poles))
      status = c_spir_pole_repr_get_residues(rep, c_loc(residues))
      do j = 1, 2
         if (abs(poles(j) - xs(j)) > 1.0e-6_c_double .or. &
             abs(residues(j) - amps(j)) > 1.0e-6_c_double) then
            print *, "Error: pole", j, poles(j), residues(j)
            stop 1
         end if
      end do
      status = c_spir_pole_repr_get_n0(rep, c_loc(n0))
      if (status /= 0 .or. n0 <= 0) then
         print *, "Error: n0", status, n0
         stop 1
      end if
      status = c_spir_pole_repr_get_err_max(rep, c_loc(err_max))
      if (status /= 0 .or. err_max <= 0.0_c_double .or. err_max > 1.0e-8_c_double) then
         print *, "Error: err_max", status, err_max
         stop 1
      end if
      status = c_spir_pole_repr_get_constant(rep, c_loc(cst))
      if (status /= 0 .or. abs(cst(1)) /= 0.0_c_double) then
         print *, "Error: constant", status, cst(1)
         stop 1
      end if
      call c_spir_pole_repr_release(rep)
      print *, "  MiniPole from Matsubara data: PASSED"
   end subroutine test_minipole_from_matsubara

end program test_minipole
