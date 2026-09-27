//! The two-particle self-consistent approximation, shared by `tpsc` and
//! `tpsc_scan`.
//!
//! TPSC fixes the two effective interactions \\(U_\mathrm{sp}\\) and
//! \\(U_\mathrm{ch}\\) not by summing a class of diagrams but by demanding
//! that the resulting susceptibilities obey the local sum rules exactly. Each
//! demand is one scalar equation in one unknown, so the solver is a pair of
//! root searches wrapped around the same RPA-like expression — which is why
//! [`crate::roots`] exists.

use num_complex::Complex64;
use sparse_ir::{
    Basis, Bosonic, Error, Fermionic, FiniteTempBasis, LogisticKernel, TworkType, compute_sve,
};

use crate::mesh::{IrMesh, MomentumGrid, evaluate_rows};
use crate::roots::{RootError, brent};

/// Absolute tolerance of every root search here.
///
/// The same number SciPy's `brentq` uses by default, so the two
/// implementations stop in the same place.
const XTOL: f64 = 2e-12;
/// Iteration cap for the root searches. Brent's method halves the bracket at
/// worst every other step, so a hundred is far more than any of these need.
const MAX_ITER: usize = 100;

/// Everything about the lattice and the basis that does not depend on `U`.
///
/// The scan builds one of these and reuses it for every interaction strength;
/// the single-point example builds one and uses it once.
pub struct Lattice {
    grid: MomentumGrid,
    ek: Vec<f64>,
    basis_f: FiniteTempBasis<LogisticKernel, Fermionic>,
    basis_b: FiniteTempBasis<LogisticKernel, Bosonic>,
    mesh_f: IrMesh<Fermionic>,
    mesh_b: IrMesh<Bosonic>,
    /// `u^F_l(0)` and `u^B_l(0)`, the rows that turn a Matsubara sum into an
    /// evaluation.
    uf_at_zero: mdarray::DTensor<f64, 2>,
    ub_at_zero: mdarray::DTensor<f64, 2>,
    /// The fermionic frequencies `ν = nπ/β` as plain numbers.
    nu: Vec<f64>,
    iw0_f: usize,
    iw0_b: usize,
}

impl Lattice {
    /// `nk1 × nk2` square lattice with nearest-neighbour hopping `t`, at
    /// inverse temperature `beta`.
    pub fn new(
        nk1: usize,
        nk2: usize,
        t: f64,
        beta: f64,
        wmax: f64,
        eps: f64,
    ) -> Result<Self, Error> {
        let kernel = LogisticKernel::new(beta * wmax)?;
        // One SVE for both statistics, as `FiniteTempBasisSet` does in Python.
        // It also halves the setup cost, and it is what makes the two τ grids
        // identical, which the solver relies on below.
        let sve = compute_sve(kernel, Some(eps), None, None, TworkType::Auto)?;
        let basis_f = FiniteTempBasis::<LogisticKernel, Fermionic>::from_sve_result(
            kernel,
            beta,
            sve.clone(),
            Some(eps),
            None,
        )?;
        let basis_b = FiniteTempBasis::<LogisticKernel, Bosonic>::from_sve_result(
            kernel,
            beta,
            sve,
            Some(eps),
            None,
        )?;
        let mesh_f = IrMesh::<Fermionic>::new(&basis_f)?;
        let mesh_b = IrMesh::<Bosonic>::new(&basis_b)?;
        // χ⁰ is built from `G` at the fermionic times and then fitted as a
        // bosonic function, which is only meaningful because the two grids
        // agree. They do, because the logistic kernel gives both statistics
        // the same imaginary-time basis functions.
        assert_eq!(
            mesh_f.tau_points(),
            mesh_b.tau_points(),
            "the fermionic and bosonic sampling times must coincide"
        );

        let uf_at_zero = basis_f.evaluate_tau(&[0.0])?;
        let ub_at_zero = basis_b.evaluate_tau(&[0.0])?;
        let nu: Vec<f64> = mesh_f
            .wn()
            .iter()
            .map(|w| w.n() as f64 * std::f64::consts::PI / beta)
            .collect();
        let iw0_f = mesh_f
            .wn()
            .iter()
            .position(|w| w.n() == 1)
            .expect("the fermionic grid contains n = 1");
        let iw0_b = mesh_b
            .wn()
            .iter()
            .position(|w| w.n() == 0)
            .expect("the bosonic grid contains n = 0");

        let grid = MomentumGrid::new(nk1, nk2);
        let ek = grid.square_lattice_dispersion(t);
        Ok(Self {
            grid,
            ek,
            basis_f,
            basis_b,
            mesh_f,
            mesh_b,
            uf_at_zero,
            ub_at_zero,
            nu,
            iw0_f,
            iw0_b,
        })
    }

    pub fn grid(&self) -> &MomentumGrid {
        &self.grid
    }

    pub fn dispersion(&self) -> &[f64] {
        &self.ek
    }

    pub fn basis_f(&self) -> &FiniteTempBasis<LogisticKernel, Fermionic> {
        &self.basis_f
    }

    pub fn basis_b(&self) -> &FiniteTempBasis<LogisticKernel, Bosonic> {
        &self.basis_b
    }

    pub fn mesh_f(&self) -> &IrMesh<Fermionic> {
        &self.mesh_f
    }

    pub fn mesh_b(&self) -> &IrMesh<Bosonic> {
        &self.mesh_b
    }

    /// The fermionic frequencies as plain numbers, in grid order.
    pub fn nu(&self) -> &[f64] {
        &self.nu
    }

    /// Index of `ν = πT`, the lowest fermionic frequency.
    pub fn iw0_f(&self) -> usize {
        self.iw0_f
    }

    /// Index of `ν = 0`, the lowest bosonic frequency.
    pub fn iw0_b(&self) -> usize {
        self.iw0_b
    }

    fn nk(&self) -> usize {
        self.grid.len()
    }
}

/// Why a TPSC calculation could not finish.
#[derive(Debug)]
pub enum TpscError {
    /// `U_sp` would have to exceed `U_crit = 1/max χ⁰`, where the RPA-like
    /// spin susceptibility diverges. The system has ordered: either `U` is too
    /// large or `T` too low for this filling.
    Ordered { u_crit: f64 },
    /// A root search failed.
    Root(RootError),
    /// The library refused something.
    Basis(Error),
}

impl std::fmt::Display for TpscError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ordered { u_crit } => write!(
                f,
                "the spin sum rule has no solution below U_crit = {u_crit}; \
                 U is too large or T too low for this filling"
            ),
            Self::Root(e) => write!(f, "root finding failed: {e}"),
            Self::Basis(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TpscError {}

impl From<RootError> for TpscError {
    fn from(value: RootError) -> Self {
        Self::Root(value)
    }
}

impl From<Error> for TpscError {
    fn from(value: Error) -> Self {
        Self::Basis(value)
    }
}

/// One converged TPSC calculation at a given `U` and filling.
pub struct Solution {
    /// Chemical potential of the interacting problem.
    pub mu: f64,
    /// Chemical potential of the non-interacting problem, found before the
    /// self-energy was known.
    pub mu_0: f64,
    /// `1/max χ⁰`, the largest interaction the spin sum rule could possibly
    /// admit.
    pub u_crit: f64,
    pub u_sp: f64,
    pub u_ch: f64,
    /// Double occupancy from Kanamori–Brueckner screening.
    pub docc: f64,
    /// `G(iν, k)`, row-major as `(frequency, momentum)`.
    pub green: Vec<Complex64>,
    /// `Σ(iν, k)`, same layout.
    pub self_energy: Vec<Complex64>,
    /// `χ_sp(iν^B, q)` and `χ_ch(iν^B, q)`, row-major as
    /// `(bosonic frequency, momentum)`.
    pub chi_spin: Vec<Complex64>,
    pub chi_charge: Vec<Complex64>,
    /// The irreducible susceptibility the two are built from.
    pub chi_0: Vec<Complex64>,
}

/// Solves TPSC for one interaction strength and filling.
///
/// The calculation is the notebook's, in order: fix `μ` so that the
/// non-interacting `G` has the requested filling, build `χ⁰` from it, find
/// `U_sp` and then `U_ch` from the local sum rules, assemble the interaction
/// and the self-energy from them, and finally refix `μ` for the interacting
/// `G`.
pub fn solve(lattice: &Lattice, u: f64, filling: f64) -> Result<Solution, TpscError> {
    let nk = lattice.nk();
    let zero_sigma = vec![Complex64::default(); lattice.mesh_f.n_wn() * nk];

    let mu_0 = find_chemical_potential(lattice, &zero_sigma, filling)?;
    let g_0 = green(lattice, &zero_sigma, mu_0);
    let chi_0 = irreducible_susceptibility(lattice, &g_0)?;

    // Above `U_crit = 1/max χ⁰` the RPA-like spin susceptibility has a pole,
    // so no `U_sp` below it can satisfy the sum rule.
    let max_chi = chi_0.iter().map(|c| c.re).fold(f64::NEG_INFINITY, f64::max);
    let u_crit = 1.0 / max_chi;

    let trace = |vertex: f64| -> Result<f64, Error> { rpa_trace(lattice, &chi_0, vertex) };

    // The spin sum rule: 2 Σ_q,m χ_sp = n − ½ (U_sp/U) n², with the double
    // occupancy already eliminated by Kanamori–Brueckner screening.
    let spin_equation = |u_sp: f64| -> Result<f64, Error> {
        Ok(2.0 * trace(u_sp)? - filling + 0.5 * (u_sp / u) * filling * filling)
    };
    // The bracket stops just short of `U_crit`, where the trace diverges.
    let upper = (u_crit * 100.0).floor() / 100.0;
    if spin_equation(upper)? <= 0.0 {
        return Err(TpscError::Ordered { u_crit });
    }
    let u_sp = root(&spin_equation, 0.0, upper)?;

    let docc = 0.25 * u_sp / u * filling * filling;

    // The charge sum rule, with the double occupancy now a number. `U_ch` is
    // not bounded from above by anything, so the bracket is simply generous.
    let charge_equation = |u_ch: f64| -> Result<f64, Error> {
        Ok(2.0 * trace(-u_ch)? - filling - 2.0 * docc + filling * filling)
    };
    let u_ch = root(&charge_equation, 0.0, 100.0)?;

    let chi_spin = rpa(&chi_0, u_sp);
    let chi_charge = rpa(&chi_0, -u_ch);

    let self_energy = self_energy(lattice, &g_0, &chi_spin, &chi_charge, u, u_sp, u_ch)?;
    let mu = find_chemical_potential(lattice, &self_energy, filling)?;
    let green = green(lattice, &self_energy, mu);

    Ok(Solution {
        mu,
        mu_0,
        u_crit,
        u_sp,
        u_ch,
        docc,
        green,
        self_energy,
        chi_spin,
        chi_charge,
        chi_0,
    })
}

/// `G(iν, k) = 1/(iν − (ε(k) − μ) − Σ(iν, k))`.
fn green(lattice: &Lattice, sigma: &[Complex64], mu: f64) -> Vec<Complex64> {
    let nk = lattice.nk();
    let mut out = Vec::with_capacity(lattice.nu.len() * nk);
    for (i, &nu) in lattice.nu.iter().enumerate() {
        for (k, &e) in lattice.ek.iter().enumerate() {
            out.push((Complex64::new(-(e - mu), nu) - sigma[i * nk + k]).inv());
        }
    }
    out
}

/// The filling `n = 2(1 + G(τ = 0))` of the zone-averaged Green's function.
///
/// `G(τ = 0)` is the Matsubara sum, which is one evaluation of the fitted
/// coefficients — the same trick the exchange-interaction example leans on.
fn filling_of(lattice: &Lattice, sigma: &[Complex64], mu: f64) -> Result<f64, Error> {
    let nk = lattice.nk();
    let gkio = green(lattice, sigma, mu);
    let averaged: Vec<Complex64> = (0..lattice.mesh_f.n_wn())
        .map(|i| gkio[i * nk..(i + 1) * nk].iter().sum::<Complex64>() / nk as f64)
        .collect();
    let coefficients = lattice.mesh_f.wn_to_l(&averaged, 1)?;
    let g_at_zero = evaluate_rows(&lattice.uf_at_zero, &coefficients, 1)[0];
    Ok(2.0 * (1.0 + g_at_zero.re))
}

fn find_chemical_potential(
    lattice: &Lattice,
    sigma: &[Complex64],
    filling: f64,
) -> Result<f64, TpscError> {
    let lower = 3.0 * lattice.ek.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = 3.0 * lattice.ek.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    root(
        &|mu: f64| Ok(filling_of(lattice, sigma, mu)? - filling),
        lower,
        upper,
    )
}

/// `χ⁰(iν^B, q)` from `G(τ, r) G(β − τ, r)`.
fn irreducible_susceptibility(
    lattice: &Lattice,
    gkio: &[Complex64],
) -> Result<Vec<Complex64>, Error> {
    let nk = lattice.nk();
    let grt = {
        let gkt = lattice.mesh_f.wn_to_tau(gkio, nk)?;
        lattice.grid.k_to_r(&gkt)
    };
    let reversed = lattice.mesh_f.reverse_tau(&grt, nk);
    let product: Vec<Complex64> = grt.iter().zip(&reversed).map(|(a, b)| a * b).collect();
    let in_momentum = lattice.grid.r_to_k(&product);
    lattice.mesh_b.tau_to_wn(&in_momentum, nk)
}

/// The RPA-like susceptibility `χ⁰/(1 − U χ⁰)`.
fn rpa(chi_0: &[Complex64], vertex: f64) -> Vec<Complex64> {
    chi_0.iter().map(|c| c / (1.0 - vertex * c)).collect()
}

/// `Σ_q,m χ(iν^B_m, q) / nk`, which is the RPA-like susceptibility at
/// `τ = 0`: one fit and one evaluation instead of a truncated Matsubara sum.
fn rpa_trace(lattice: &Lattice, chi_0: &[Complex64], vertex: f64) -> Result<f64, Error> {
    let nk = lattice.nk();
    let chi = rpa(chi_0, vertex);
    let averaged: Vec<Complex64> = (0..lattice.mesh_b.n_wn())
        .map(|i| chi[i * nk..(i + 1) * nk].iter().sum::<Complex64>() / nk as f64)
        .collect();
    let coefficients = lattice.mesh_b.wn_to_l(&averaged, 1)?;
    Ok(evaluate_rows(&lattice.ub_at_zero, &coefficients, 1)[0].re)
}

/// `Σ(iν, k)` from `V(τ, r) G(τ, r)`, where
/// `V = U/4 (3 U_sp χ_sp + U_ch χ_ch)`.
///
/// The constant Hartree term is left out: it is a frequency-independent shift
/// that the basis cannot represent compactly, and in a single band it is
/// absorbed into the chemical potential.
fn self_energy(
    lattice: &Lattice,
    gkio: &[Complex64],
    chi_spin: &[Complex64],
    chi_charge: &[Complex64],
    u: f64,
    u_sp: f64,
    u_ch: f64,
) -> Result<Vec<Complex64>, Error> {
    let nk = lattice.nk();
    let interaction: Vec<Complex64> = chi_spin
        .iter()
        .zip(chi_charge)
        .map(|(sp, ch)| (u / 4.0) * (3.0 * u_sp * sp + u_ch * ch))
        .collect();
    let v_rt = {
        let in_real_space = lattice.grid.k_to_r(&interaction);
        lattice.mesh_b.wn_to_tau(&in_real_space, nk)?
    };
    let grt = {
        let gkt = lattice.mesh_f.wn_to_tau(gkio, nk)?;
        lattice.grid.k_to_r(&gkt)
    };
    let product: Vec<Complex64> = v_rt.iter().zip(&grt).map(|(v, g)| v * g).collect();
    let in_momentum = lattice.grid.r_to_k(&product);
    lattice.mesh_f.tau_to_wn(&in_momentum, nk)
}

/// Brent's method on a function that can fail, with the tutorial's tolerance.
fn root<F>(f: &F, a: f64, b: f64) -> Result<f64, TpscError>
where
    F: Fn(f64) -> Result<f64, Error>,
{
    let mut failure = None;
    let root = brent(
        |x| match f(x) {
            Ok(value) => value,
            Err(e) => {
                // Brent's method has no way to carry an error out, so remember
                // the first one and let the search finish on nonsense; the
                // caller sees the error rather than the number.
                failure.get_or_insert(e);
                f64::NAN
            }
        },
        a,
        b,
        XTOL,
        MAX_ITER,
    );
    match failure {
        Some(e) => Err(TpscError::Basis(e)),
        None => Ok(root?),
    }
}

/// The indices of Γ → X → M → Γ on an `nk × nk` grid, in fractional
/// coordinates `(0,0) → (½,0) → (½,½) → (0,0)`.
pub fn high_symmetry_path(nk_lin: usize) -> Vec<usize> {
    let half = nk_lin / 2;
    let mut path = Vec::new();
    for i in 0..=half {
        path.push(i * nk_lin);
    }
    for j in 1..=half {
        path.push(half * nk_lin + j);
    }
    for step in 1..=half {
        let i = half - step;
        path.push(i * nk_lin + i);
    }
    path
}
