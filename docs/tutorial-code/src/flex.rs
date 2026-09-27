//! The fluctuation-exchange approximation, shared by `flex` and `flex_scan`.
//!
//! Ported from the Python notebook `FLEX_py.ipynb` of sparse-ir-tutorial,
//! whose author is Niklas Witt.
//!
//! FLEX is the self-consistent sum of the bubble and ladder series. Unlike
//! [`crate::tpsc`] it has no sum rules to satisfy and no root to find: the
//! interaction is the bare `U`, and the work is in iterating the Dyson
//! equation until the self-energy stops moving. Every step of that iteration
//! is a product in `(τ, r)` sandwiched between basis transforms, so the cost
//! of a step is set by the size of the basis rather than by a frequency
//! cutoff.

use num_complex::Complex64;
use sparse_ir::Error;

use crate::lattice::Lattice;
use crate::mesh::evaluate_rows;
use crate::roots::{RootError, brent};

/// Absolute tolerance of the chemical-potential search, the same number
/// SciPy's `brentq` uses by default.
const XTOL: f64 = 2e-12;
const MAX_ITER: usize = 100;

/// Why a FLEX calculation could not finish.
#[derive(Debug)]
pub enum FlexError {
    /// The chemical-potential search failed.
    Root(RootError),
    /// The library refused something.
    Basis(Error),
}

impl std::fmt::Display for FlexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Root(e) => write!(f, "chemical potential search failed: {e}"),
            Self::Basis(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FlexError {}

impl From<RootError> for FlexError {
    fn from(value: RootError) -> Self {
        Self::Root(value)
    }
}

impl From<Error> for FlexError {
    fn from(value: Error) -> Self {
        Self::Basis(value)
    }
}

/// How the loop is run.
#[derive(Clone, Copy)]
pub struct Settings {
    /// Hubbard interaction.
    pub u: f64,
    /// Electron filling per site, summed over both spins.
    pub filling: f64,
    /// Fraction of the new Green's function kept at each step.
    pub mix: f64,
    /// Number of self-consistency steps. The loop always runs all of them:
    /// the notebook stops early on a tolerance, but a fixed count makes the
    /// result independent of where exactly the tolerance is crossed, and the
    /// residual is reported instead.
    pub iterations: usize,
    /// Cap on the interaction-renormalisation loop, which may not terminate
    /// on its own.
    pub renormalisation_iterations: usize,
}

/// A converged (or at least well-iterated) FLEX solution.
pub struct Solver<'a> {
    lattice: &'a Lattice,
    settings: Settings,
    /// The interaction actually used, which the renormalisation loop may
    /// lower below `settings.u` while it is running.
    u: f64,
    mu: f64,
    /// `Σ(iν, k)`, row-major as `(fermionic frequency, momentum)`.
    sigma: Vec<Complex64>,
    /// `G(iν, k)`, same layout.
    gkio: Vec<Complex64>,
    /// `G(τ, r)`, row-major as `(imaginary time, lattice vector)`.
    grit: Vec<Complex64>,
    /// `χ⁰(iν^B, q)`, row-major as `(bosonic frequency, momentum)`.
    ckio: Vec<Complex64>,
    chi_spin: Vec<Complex64>,
    chi_charge: Vec<Complex64>,
    /// Number of steps the renormalisation loop took, zero if it never ran.
    renormalisation_steps: usize,
    /// `Σ_k |Σ − Σ_old| / Σ_k |Σ|` after the last step.
    residual: f64,
}

impl<'a> Solver<'a> {
    /// Starts from a given self-energy, which is how a temperature scan hands
    /// one temperature's answer to the next.
    pub fn new(
        lattice: &'a Lattice,
        settings: Settings,
        sigma_init: Vec<Complex64>,
    ) -> Result<Self, FlexError> {
        let nk = lattice.nk();
        assert_eq!(sigma_init.len(), lattice.mesh_f().n_wn() * nk);
        let mut solver = Self {
            lattice,
            settings,
            u: settings.u,
            mu: 0.0,
            sigma: sigma_init,
            gkio: Vec::new(),
            grit: Vec::new(),
            ckio: Vec::new(),
            chi_spin: Vec::new(),
            chi_charge: Vec::new(),
            renormalisation_steps: 0,
            residual: f64::NAN,
        };
        solver.mu = solver.find_chemical_potential()?;
        solver.green(solver.mu);
        solver.real_space_green()?;
        solver.irreducible_susceptibility()?;
        Ok(solver)
    }

    /// Starts from `Σ = 0`, which is the non-interacting Green's function.
    pub fn from_scratch(lattice: &'a Lattice, settings: Settings) -> Result<Self, FlexError> {
        let zero = vec![Complex64::default(); lattice.mesh_f().n_wn() * lattice.nk()];
        Self::new(lattice, settings, zero)
    }

    /// Runs the self-consistency loop.
    pub fn solve(&mut self) -> Result<(), FlexError> {
        // The RPA denominator `1 − U χ⁰` must stay positive. If the starting
        // point violates it, the interaction is walked up towards `U` instead
        // of being switched on at full strength, which is the notebook's
        // `U_renormalization`.
        if self.max_chi() * self.settings.u >= 1.0 {
            self.renormalise()?;
        }
        for _ in 0..self.settings.iterations {
            let previous = self.sigma.clone();
            self.step()?;
            let moved: f64 = self
                .sigma
                .iter()
                .zip(&previous)
                .map(|(a, b)| (a - b).norm())
                .sum();
            let total: f64 = self.sigma.iter().map(|s| s.norm()).sum();
            self.residual = moved / total;
        }
        Ok(())
    }

    /// One pass of the Dyson equation.
    fn step(&mut self) -> Result<(), FlexError> {
        let previous = self.gkio.clone();
        let interaction = self.interaction()?;
        self.sigma = self.self_energy(&interaction)?;

        self.mu = self.find_chemical_potential()?;
        self.green(self.mu);
        let mix = self.settings.mix;
        for (new, old) in self.gkio.iter_mut().zip(&previous) {
            *new = mix * *new + (1.0 - mix) * old;
        }

        self.real_space_green()?;
        self.irreducible_susceptibility()?;
        Ok(())
    }

    /// Walks a too-large interaction up towards its target value.
    ///
    /// Each pass solves one FLEX step at the largest interaction that keeps
    /// `U χ⁰ < 1`, which shrinks `χ⁰`; the target `U` is then retried. The
    /// loop is capped because there is no guarantee it ever succeeds — if the
    /// system really is magnetically ordered, it cannot.
    fn renormalise(&mut self) -> Result<(), FlexError> {
        let target = self.settings.u;
        while target * self.max_chi() >= 1.0 {
            self.renormalisation_steps += 1;
            self.u = target / (self.max_chi() * target + 0.01);
            self.step()?;
            self.u = target;
            if self.renormalisation_steps == self.settings.renormalisation_iterations {
                break;
            }
        }
        Ok(())
    }

    /// `G(iν, k) = [iν − (ε_k − μ) − Σ]⁻¹`.
    fn green(&mut self, mu: f64) {
        let nk = self.lattice.nk();
        let ek = self.lattice.dispersion();
        self.gkio.clear();
        self.gkio.reserve(self.lattice.nu().len() * nk);
        for (i, &nu) in self.lattice.nu().iter().enumerate() {
            for (k, &e) in ek.iter().enumerate() {
                let denominator = Complex64::new(-(e - mu), nu) - self.sigma[i * nk + k];
                self.gkio.push(1.0 / denominator);
            }
        }
    }

    fn real_space_green(&mut self) -> Result<(), FlexError> {
        let nk = self.lattice.nk();
        let in_real_space = self.lattice.grid().k_to_r(&self.gkio);
        self.grit = self.lattice.mesh_f().wn_to_tau(&in_real_space, nk)?;
        Ok(())
    }

    /// `χ⁰(τ, r) = G(τ, r) G(β − τ, r)`, then back to `(iν^B, q)`.
    fn irreducible_susceptibility(&mut self) -> Result<(), FlexError> {
        let nk = self.lattice.nk();
        let reversed = self.lattice.mesh_f().reverse_tau(&self.grit, nk);
        let product: Vec<Complex64> = self
            .grit
            .iter()
            .zip(&reversed)
            .map(|(a, b)| a * b)
            .collect();
        let in_momentum = self.lattice.grid().r_to_k(&product);
        self.ckio = self.lattice.mesh_b().tau_to_wn(&in_momentum, nk)?;
        Ok(())
    }

    /// `V(τ, r)` from the spin and charge susceptibilities.
    ///
    /// The constant Hartree term `~U` is left out: it is frequency
    /// independent, so the basis cannot represent it compactly, and in a
    /// single band it is absorbed into the chemical potential.
    fn interaction(&mut self) -> Result<Vec<Complex64>, FlexError> {
        let nk = self.lattice.nk();
        let u = self.u;
        self.chi_spin = self.ckio.iter().map(|c| c / (1.0 - u * c)).collect();
        self.chi_charge = self.ckio.iter().map(|c| c / (1.0 + u * c)).collect();
        let v: Vec<Complex64> = (0..self.ckio.len())
            .map(|i| u * u * (1.5 * self.chi_spin[i] + 0.5 * self.chi_charge[i] - self.ckio[i]))
            .collect();
        let in_real_space = self.lattice.grid().k_to_r(&v);
        Ok(self.lattice.mesh_b().wn_to_tau(&in_real_space, nk)?)
    }

    /// `Σ(iν, k)` from `V(τ, r) G(τ, r)`.
    fn self_energy(&self, interaction: &[Complex64]) -> Result<Vec<Complex64>, FlexError> {
        let nk = self.lattice.nk();
        let product: Vec<Complex64> = interaction
            .iter()
            .zip(&self.grit)
            .map(|(v, g)| v * g)
            .collect();
        let in_momentum = self.lattice.grid().r_to_k(&product);
        Ok(self.lattice.mesh_f().tau_to_wn(&in_momentum, nk)?)
    }

    /// `n = 2 [1 + Re G(τ = 0⁻)]`, the zone average through the basis.
    fn filling_at(&mut self, mu: f64) -> Result<f64, Error> {
        let nk = self.lattice.nk();
        self.green(mu);
        let averaged: Vec<Complex64> = (0..self.lattice.mesh_f().n_wn())
            .map(|i| self.gkio[i * nk..(i + 1) * nk].iter().sum::<Complex64>() / nk as f64)
            .collect();
        let coefficients = self.lattice.mesh_f().wn_to_l(&averaged, 1)?;
        let g_at_zero = evaluate_rows(self.lattice.uf_at_zero(), &coefficients, 1)[0];
        Ok(2.0 * (1.0 + g_at_zero.re))
    }

    fn find_chemical_potential(&mut self) -> Result<f64, FlexError> {
        let ek = self.lattice.dispersion();
        let lo = 3.0 * ek.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = 3.0 * ek.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let target = self.settings.filling;
        // Brent's method has no way to carry an error out, so the first one is
        // remembered and the search is left to finish on nonsense.
        let mut failure: Option<Error> = None;
        let mut probe = |mu: f64| match self.filling_at(mu) {
            Ok(n) => n - target,
            Err(e) => {
                failure.get_or_insert(e);
                f64::NAN
            }
        };
        let root = brent(&mut probe, lo, hi, XTOL, MAX_ITER);
        match failure {
            Some(e) => Err(FlexError::Basis(e)),
            None => Ok(root?),
        }
    }

    fn max_chi(&self) -> f64 {
        self.ckio
            .iter()
            .map(|c| c.norm())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn lattice(&self) -> &Lattice {
        self.lattice
    }

    pub fn mu(&self) -> f64 {
        self.mu
    }

    pub fn green_function(&self) -> &[Complex64] {
        &self.gkio
    }

    pub fn self_energy_values(&self) -> &[Complex64] {
        &self.sigma
    }

    pub fn chi_0(&self) -> &[Complex64] {
        &self.ckio
    }

    pub fn chi_spin(&self) -> &[Complex64] {
        &self.chi_spin
    }

    pub fn chi_charge(&self) -> &[Complex64] {
        &self.chi_charge
    }

    /// Largest `|χ_sp|` anywhere on the grid: the standard measure of how
    /// close the paramagnet is to ordering.
    pub fn max_chi_spin(&self) -> f64 {
        self.chi_spin
            .iter()
            .map(|c| c.re)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn renormalisation_steps(&self) -> usize {
        self.renormalisation_steps
    }

    pub fn residual(&self) -> f64 {
        self.residual
    }
}

/// The linearised Eliashberg equation in the singlet channel, solved for its
/// largest eigenvalue by the power method.
///
/// `λ = 1` marks the superconducting transition, so following `λ(T)` up to
/// one locates `T_c` without ever entering the symmetry-broken state.
pub struct GapSolver {
    /// `Δ(iν, k)`, normalised, row-major as `(fermionic frequency, momentum)`.
    delta: Vec<Complex64>,
    /// The `d`-wave seed `cos k_x − cos k_y`, unnormalised.
    seed: Vec<f64>,
    /// `F(iν, k) = −|G|² Δ` from the last step.
    anomalous: Vec<Complex64>,
    /// `V^S(τ, r)`, fixed by the FLEX solution.
    interaction: Vec<Complex64>,
    green: Vec<Complex64>,
    lambda: f64,
    iterations: usize,
}

impl GapSolver {
    /// Takes the susceptibilities and the Green's function of a solved FLEX
    /// problem, and seeds the power method with a `d_{x²−y²}` gap.
    pub fn new(solver: &Solver<'_>) -> Result<Self, FlexError> {
        use std::f64::consts::TAU;
        let lattice = solver.lattice();
        let nk = lattice.nk();
        let n_wn = lattice.mesh_f().n_wn();

        let seed: Vec<f64> = (0..nk)
            .map(|index| {
                let (k1, k2) = lattice.grid().coordinates(index);
                (TAU * k1).cos() - (TAU * k2).cos()
            })
            .collect();
        let mut delta: Vec<Complex64> = (0..n_wn)
            .flat_map(|_| seed.iter().map(|&d| Complex64::new(d, 0.0)))
            .collect();
        normalise(&mut delta);

        // The singlet vertex differs from the one in the self-energy: the
        // charge fluctuations enter with the opposite sign, and the constant
        // Hartree term drops out because a `d`-wave gap sums to zero over the
        // zone.
        let u = solver.u;
        let v: Vec<Complex64> = (0..solver.chi_spin.len())
            .map(|i| u * u * (1.5 * solver.chi_spin[i] - 0.5 * solver.chi_charge[i]))
            .collect();
        let in_real_space = lattice.grid().k_to_r(&v);
        let interaction = lattice.mesh_b().wn_to_tau(&in_real_space, nk)?;

        Ok(Self {
            delta,
            seed,
            anomalous: Vec::new(),
            interaction,
            green: solver.gkio.clone(),
            lambda: 0.0,
            iterations: 0,
        })
    }

    /// Runs the power method until `λ` stops moving by more than `tol`, or
    /// for at most `max_iterations` steps.
    ///
    /// Unlike [`Solver::solve`] this one must stop early, and the stopping
    /// rule has to be a tolerance rather than a fixed count. Dropping the
    /// Hartree term from the singlet vertex is exact in the `d`-wave channel
    /// but not outside it, and it leaves the operator with a spurious mode
    /// whose eigenvalue is larger in magnitude than `λ_d`. The seed has an
    /// almost vanishing overlap with that mode, so the iterate sits on the
    /// `d`-wave answer for tens of steps — and then leaves it. The tolerance
    /// is crossed long before that happens, with three orders of magnitude to
    /// spare, so the stopping step is not in doubt; the number of steps taken
    /// is reported so that a change in it cannot pass unnoticed.
    pub fn solve(
        &mut self,
        lattice: &Lattice,
        max_iterations: usize,
        tol: f64,
    ) -> Result<(), FlexError> {
        let nk = lattice.nk();
        for _ in 0..max_iterations {
            let lambda_old = self.lambda;
            let previous = self.delta.clone();

            self.anomalous = self
                .green
                .iter()
                .zip(&self.delta)
                .map(|(g, d)| -(g * g.conj()) * d)
                .collect();
            let in_real_space = lattice.grid().k_to_r(&self.anomalous);
            let frit = lattice.mesh_f().wn_to_tau(&in_real_space, nk)?;

            let product: Vec<Complex64> = self
                .interaction
                .iter()
                .zip(&frit)
                .map(|(v, f)| v * f)
                .collect();
            let in_momentum = lattice.grid().r_to_k(&product);
            let mut delta = lattice.mesh_f().tau_to_wn(&in_momentum, nk)?;

            // The Rayleigh quotient against the previous (normalised) vector.
            self.lambda = delta
                .iter()
                .zip(&previous)
                .map(|(new, old)| (new.conj() * old).re)
                .sum();
            normalise(&mut delta);
            self.delta = delta;

            self.iterations += 1;
            if (self.lambda - lambda_old).abs() < tol {
                break;
            }
        }
        Ok(())
    }

    /// How many power-method steps were taken.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn gap(&self) -> &[Complex64] {
        &self.delta
    }

    pub fn seed(&self) -> &[f64] {
        &self.seed
    }

    pub fn anomalous_green(&self) -> &[Complex64] {
        &self.anomalous
    }
}

fn normalise(values: &mut [Complex64]) {
    let norm = values.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    for value in values.iter_mut() {
        *value /= norm;
    }
}
