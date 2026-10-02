//! Eliashberg theory for the Jahn-Teller-Hubbard model, shared by
//! `eliashberg_holstein` and `eliashberg_holstein_scan`.
//!
//! Ported from the Python notebook `eliashberg_holstein_py.ipynb` of
//! sparse-ir-tutorial-v2
//! (<https://spm-lab.github.io/sparse-ir-tutorial-v2/src/eliashberg_holstein_py.html>), whose
//! authors are Shintaro Hoshino and Hiroshi Shinaoka.
//!
//! The electrons live on a semicircular density of states and are coupled to
//! a local phonon. Both propagators are dressed self-consistently, and the
//! superconducting state is described by the anomalous `F` and the gap
//! function `Δ` alongside the usual `G` and `Σ`. Every product is local, so
//! unlike the lattice examples there is no momentum index at all: each
//! quantity is one array over sampling points, and a step of the loop is four
//! transforms between `τ` and the Matsubara frequencies.

use num_complex::Complex64;
use sparse_ir::Error;

use crate::lattice::Bases;
use crate::mesh::evaluate_rows;
use crate::quad::gauss_legendre;
use crate::semicircle::shifted_semicircle;

/// The model, and how the loop is run.
#[derive(Clone, Copy)]
pub struct Settings {
    /// Half bandwidth of the semicircular density of states.
    pub d: f64,
    pub u: f64,
    /// Hund's coupling, zero for the single-orbital Holstein-Hubbard model.
    pub j: f64,
    /// Phonon frequency.
    pub omega0: f64,
    /// Electron-phonon coupling `g₀ = √(3 λ₀ ω₀ / 4)`.
    pub g: f64,
    pub mu: f64,
    /// Order of the Gauss-Legendre rule over the density of states.
    pub deg_leggauss: usize,
    pub mixing: f64,
    pub max_iterations: usize,
    /// The loop stops when neither `Σ` nor `Δ` moves by more than this.
    pub atol: f64,
}

/// `g₀` from the dimensionless coupling `λ₀`.
pub fn coupling(lambda0: f64, omega0: f64) -> f64 {
    (3.0 * lambda0 * omega0 / 4.0).sqrt()
}

/// One self-consistent solution, and everything the example reports from it.
pub struct Solver<'a> {
    bases: &'a Bases,
    settings: Settings,
    /// Nodes of the density-of-states integral, and `ρ(ω)` times the weights.
    omega: Vec<f64>,
    omega_coeff: Vec<f64>,
    iv_f: Vec<Complex64>,
    iv_b: Vec<Complex64>,
    /// The bare phonon propagator `D₀(iν^B) = 2ω₀/((iν^B)² − ω₀²)`.
    d0_iv: Vec<Complex64>,
    /// Index of the smallest positive sampling time `τ₀`. The notebook's
    /// `F(0⁺)` (and the phonon term of the energy) is read there: an
    /// approximation to the `τ → 0⁺` limit, not the limit itself.
    tau_zero_plus: usize,
    sigma: Vec<Complex64>,
    delta: Vec<Complex64>,
    g_iv: Vec<Complex64>,
    f_iv: Vec<Complex64>,
    d_iv: Vec<Complex64>,
    phi_iv: Vec<Complex64>,
    d_tau: Vec<Complex64>,
    iterations: usize,
}

impl<'a> Solver<'a> {
    /// Starts from the given self-energy and `Δ = 1`, as the notebook does.
    ///
    /// The starting `Σ` has to be perturbed away from zero: the normal state
    /// is a solution of these equations too, and a loop started exactly on it
    /// stays there. The perturbation is read from a committed file rather
    /// than drawn, so that this and the Python reference start from
    /// bit-identical numbers.
    pub fn new(bases: &'a Bases, settings: Settings, sigma: Vec<Complex64>) -> Self {
        let n_wn_f = bases.mesh_f().n_wn();
        assert_eq!(
            sigma.len(),
            n_wn_f,
            "the starting self-energy has {} values but the basis needs {n_wn_f}",
            sigma.len()
        );
        let (omega, weights) = gauss_legendre(settings.deg_leggauss, -settings.d, settings.d);
        let omega_coeff = omega
            .iter()
            .zip(&weights)
            .map(|(&w, &weight)| shifted_semicircle(w, 0.0, settings.d, 1.0) * weight)
            .collect();
        let iv_f: Vec<Complex64> = bases
            .nu()
            .iter()
            .map(|&nu| Complex64::new(0.0, nu))
            .collect();
        let beta = bases.mesh_b().beta();
        let iv_b: Vec<Complex64> = bases
            .mesh_b()
            .wn()
            .iter()
            .map(|w| Complex64::new(0.0, w.n() as f64 * std::f64::consts::PI / beta))
            .collect();
        let d0_iv = iv_b
            .iter()
            .map(|&iv| 2.0 * settings.omega0 / (iv * iv - settings.omega0 * settings.omega0))
            .collect();
        let tau_zero_plus = smallest_positive(bases.mesh_f().tau_points());

        Self {
            bases,
            settings,
            omega,
            omega_coeff,
            iv_f,
            iv_b,
            d0_iv,
            tau_zero_plus,
            delta: vec![Complex64::new(1.0, 0.0); n_wn_f],
            sigma,
            g_iv: Vec::new(),
            f_iv: Vec::new(),
            d_iv: Vec::new(),
            phi_iv: Vec::new(),
            d_tau: Vec::new(),
            iterations: 0,
        }
    }

    /// Starts the gap from a given function rather than from `Δ = 1`.
    ///
    /// The scan walks down in temperature and hands each solution to the next
    /// step, which is both faster and what keeps the sweep on one branch of
    /// the solution rather than jumping between them.
    pub fn set_gap(&mut self, delta: Vec<Complex64>) {
        assert_eq!(
            delta.len(),
            self.delta.len(),
            "the starting gap has {} values but the basis needs {}",
            delta.len(),
            self.delta.len()
        );
        self.delta = delta;
    }

    /// Runs the self-consistency loop, and returns whether it converged.
    pub fn solve(&mut self) -> Result<bool, Error> {
        let mesh_f = self.bases.mesh_f();
        let mesh_b = self.bases.mesh_b();
        let mixing = self.settings.mixing;
        let four_g2 = 4.0 * self.settings.g * self.settings.g;

        for _ in 0..self.settings.max_iterations {
            self.iterations += 1;
            // ANCHOR: to_tau
            let (g_iv, f_iv) = self.green();
            self.g_iv = g_iv;
            self.f_iv = f_iv;
            let mut g_tau = mesh_f.wn_to_tau(&self.g_iv, 1)?;
            let f_tau = mesh_f.wn_to_tau(&self.f_iv, 1)?;
            // ANCHOR_END: to_tau

            // Particle-hole symmetry is a symmetry of the solution at half
            // filling, but not of every iterate; imposing it on `G` keeps the
            // loop from drifting away from the state it is meant to find.
            let reversed = mesh_f.reverse_tau(&g_tau, 1);
            for (value, mirror) in g_tau.iter_mut().zip(&reversed) {
                *value = 0.5 * (*value + mirror);
            }
            clamp_to_negative(&mut g_tau, mesh_f.tau_points());

            // ANCHOR: phonon
            // `Π(τ) = −4g² [G(τ)G(β − τ) + F(τ)²]`, fitted as a bosonic
            // function of the shared τ grid.
            let reversed = mesh_f.reverse_tau(&g_tau, 1);
            let phi_tau: Vec<Complex64> = (0..g_tau.len())
                .map(|i| -four_g2 * (g_tau[i] * reversed[i] + f_tau[i] * f_tau[i]))
                .collect();
            self.phi_iv = mesh_b.tau_to_wn(&phi_tau, 1)?;
            // `Π` is real by construction; what the transforms leave in the
            // imaginary part is round-off, and dropping it keeps `D` real.
            for value in &mut self.phi_iv {
                value.im = 0.0;
            }

            self.d_iv = (0..self.phi_iv.len())
                .map(|i| 1.0 / (1.0 / self.d0_iv[i] - self.phi_iv[i]))
                .collect();
            self.d_tau = mesh_b.wn_to_tau(&self.d_iv, 1)?;
            // ANCHOR_END: phonon

            // ANCHOR: electron
            // `Σ(τ) = −4g² D(τ) G(τ)`.
            let sigma_tau: Vec<Complex64> = (0..g_tau.len())
                .map(|i| -four_g2 * self.d_tau[i] * g_tau[i])
                .collect();
            let sigma_new = mesh_f.tau_to_wn(&sigma_tau, 1)?;

            // `Δ(τ) = U_eff(τ) F(τ)`. The instantaneous part of `U_eff` is
            // `(U + 2J)δ(τ)`, which contributes the constant `(U + 2J)F(0⁺)`;
            // as in the notebook, `F(0⁺)` is approximated by `F(τ₀)` at the
            // smallest positive sampling time.
            let delta_tau: Vec<Complex64> = (0..f_tau.len())
                .map(|i| four_g2 * self.d_tau[i] * f_tau[i])
                .collect();
            let instantaneous =
                (self.settings.u + 2.0 * self.settings.j) * f_tau[self.tau_zero_plus];
            let delta_new: Vec<Complex64> = mesh_f
                .tau_to_wn(&delta_tau, 1)?
                .into_iter()
                .map(|value| value + instantaneous)
                .collect();
            // ANCHOR_END: electron

            let moved =
                max_deviation(&sigma_new, &self.sigma).max(max_deviation(&delta_new, &self.delta));

            for (old, new) in self.sigma.iter_mut().zip(&sigma_new) {
                *old = (1.0 - mixing) * *old + mixing * new;
            }
            for (old, new) in self.delta.iter_mut().zip(&delta_new) {
                *old = (1.0 - mixing) * *old + mixing * new;
                old.im = 0.0;
            }
            // `Δ(iν) = Δ(−iν)` for a singlet gap, again a symmetry of the
            // solution that is worth imposing on every iterate.
            let n = self.delta.len();
            for i in 0..n / 2 {
                let averaged = 0.5 * (self.delta[i] + self.delta[n - 1 - i]);
                self.delta[i] = averaged;
                self.delta[n - 1 - i] = averaged;
            }

            if moved < self.settings.atol {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `G(iν)` and `F(iν)` from the current `Σ` and `Δ`, integrated over the
    /// density of states.
    fn green(&self) -> (Vec<Complex64>, Vec<Complex64>) {
        let mut g_iv = Vec::with_capacity(self.iv_f.len());
        let mut f_iv = Vec::with_capacity(self.iv_f.len());
        for i in 0..self.iv_f.len() {
            let xi = self.iv_f[i] + self.settings.mu - self.sigma[i];
            let head = xi * xi - self.delta[i] * self.delta[i];
            let mut g = Complex64::default();
            let mut f = Complex64::default();
            for (&omega, &coefficient) in self.omega.iter().zip(&self.omega_coeff) {
                let denominator = head - omega * omega;
                g += coefficient * (xi + omega) / denominator;
                f += coefficient * self.delta[i] / denominator;
            }
            g_iv.push(g);
            f_iv.push(f);
        }
        (g_iv, f_iv)
    }

    /// The internal energy `⟨H⟩`, whose temperature derivative is the
    /// specific heat.
    ///
    /// Each of the three terms is a Matsubara sum, which in the basis is a fit
    /// followed by one evaluation at `τ = 0` — the same trick the TPSC sum
    /// rules use. The constant subtracted inside each fit (`iνG → 1`) is the
    /// part the basis cannot represent; what is left decays at least as `1/iν`.
    ///
    /// `uf_at_zero` evaluates at `+0.0`, i.e. the `τ → 0⁺` side, whereas a
    /// convergence factor `e^{iν0⁺}` would ask for `0⁻`. The two sides differ
    /// by the `1/iν` coefficient of the fitted function. For `e1` and `e2`
    /// that coefficient vanishes here (`μ = 0`, a symmetric density of states,
    /// `Σ(iν) → 0`), and the two sides agree to ~1e-14, so the choice does not
    /// matter in this model. The phonon term is read at the smallest positive
    /// sampling time `τ₀`, as the notebook does; it is bosonic and continuous
    /// at `τ = 0`, so this approximates its `τ = 0` value.
    pub fn internal_energy(&self) -> Result<f64, Error> {
        let mesh_f = self.bases.mesh_f();
        let mesh_b = self.bases.mesh_b();

        let e1: Vec<Complex64> = (0..self.iv_f.len())
            .map(|i| self.iv_f[i] * self.g_iv[i] - 1.0)
            .collect();
        let e2: Vec<Complex64> = (0..self.iv_f.len())
            .map(|i| {
                let xi = self.iv_f[i] - self.sigma[i];
                self.g_iv[i] * (xi * xi - self.delta[i] * self.delta[i]) / xi - 1.0
            })
            .collect();
        let at_zero = |values: &[Complex64]| -> Result<Complex64, Error> {
            let coefficients = mesh_f.wn_to_l(values, 1)?;
            Ok(evaluate_rows(self.bases.uf_at_zero(), &coefficients, 1)[0])
        };

        // The phonon term is read at the smallest positive sampling time `τ₀`
        // rather than at `τ = 0`, which is what the notebook does.
        let f2: Vec<Complex64> = (0..self.iv_b.len())
            .map(|i| {
                (self.iv_b[i] * self.iv_b[i] * self.d_iv[i] - 2.0 * self.settings.omega0)
                    / (self.settings.omega0 * self.settings.omega0)
            })
            .collect();
        let f2 = mesh_b.wn_to_tau(&f2, 1)?;

        let total = at_zero(&e1)? + at_zero(&e2)? - self.settings.omega0 * f2[self.tau_zero_plus];
        Ok(3.0 * total.re)
    }

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn self_energy(&self) -> &[Complex64] {
        &self.sigma
    }

    pub fn gap(&self) -> &[Complex64] {
        &self.delta
    }

    pub fn green_function(&self) -> &[Complex64] {
        &self.g_iv
    }

    pub fn anomalous_green_function(&self) -> &[Complex64] {
        &self.f_iv
    }

    pub fn phonon_propagator(&self) -> &[Complex64] {
        &self.d_iv
    }

    pub fn phonon_propagator_tau(&self) -> &[Complex64] {
        &self.d_tau
    }

    pub fn polarisation(&self) -> &[Complex64] {
        &self.phi_iv
    }
}

/// Index of the smallest positive sampling time.
///
/// The notebook reads `F(0⁺)` as the first element of an array of sampling
/// times on `[0, β)`. Here the times run over `[−β/2, β/2]`, so the same
/// point has to be looked up rather than indexed.
fn smallest_positive(points: &[f64]) -> usize {
    points
        .iter()
        .enumerate()
        .filter(|&(_, &tau)| tau > 0.0)
        .min_by(|a, b| a.1.total_cmp(b.1))
        .expect("the sampling grid has a positive time")
        .0
}

/// `G(τ) ≤ 0` is exact on `[0, β)`, so anything above it is round-off; the
/// notebook clamps it away with `g_tau[g_tau > 0] = 0`.
///
/// The statement is about the values on `[0, β)`, which is where the notebook
/// samples. Here the grid is `[−β/2, β/2]`, and on its negative half `G` is
/// positive for exactly the reason that makes the clamp correct — `G(τ) =
/// −G(τ + β)` — so the sign has to be undone before the comparison and the
/// clamp applied to the value the notebook would have seen.
///
/// NumPy orders complex numbers lexicographically, which is the comparison
/// reproduced here.
fn clamp_to_negative(values: &mut [Complex64], points: &[f64]) {
    assert_eq!(
        values.len(),
        points.len(),
        "one value per sampling time is needed to fold the comparison"
    );
    // ANCHOR: clamp
    for (value, &tau) in values.iter_mut().zip(points) {
        let folded = if tau > 0.0 { *value } else { -*value };
        if folded.re > 0.0 || (folded.re == 0.0 && folded.im > 0.0) {
            *value = Complex64::default();
        }
    }
    // ANCHOR_END: clamp
}

/// `max |a − b|`, the movement the loop stops on.
fn max_deviation(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).norm())
        .fold(0.0, f64::max)
}
