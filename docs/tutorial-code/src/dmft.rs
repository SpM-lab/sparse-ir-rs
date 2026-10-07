//! The DMFT loop of the `dmft_ipt` examples.
//!
//! Dynamical mean-field theory on the Bethe lattice with the impurity problem
//! solved by iterated perturbation theory. Both `dmft_ipt` and
//! `dmft_ipt_scan` run exactly this loop; they differ only in how hard they
//! converge it and how many interaction strengths they run it for.
//!
//! At half filling with a symmetric density of states the exact self-energy
//! is particle-hole symmetric: `Σ(iν)` is purely imaginary and odd in `ν`.
//! The round trip through the basis preserves that only up to rounding, and
//! the symmetric solution is an *unstable* fixed point of the loop with
//! respect to perturbations that break the symmetry: left alone, a `10⁻¹⁶`
//! asymmetry grows by about a factor of ten every hundred iterations until it
//! takes over. So [`Dmft::solve`] projects every new self-energy back onto the
//! symmetric subspace; [`Dmft::solve_with`] can switch that off to show what
//! happens without it.

use num_complex::Complex64;
use sparse_ir::{Error, Fermionic, FiniteTempBasis, LogisticKernel};

use crate::{IrMesh, shifted_semicircle_overlaps};

/// The proportion of the new self-energy mixed in at each step.
pub const MIX: f64 = 0.25;

/// The basis, its sampling grids, and the Bethe lattice they describe.
pub struct Dmft {
    basis: FiniteTempBasis<LogisticKernel, Fermionic>,
    mesh: IrMesh<Fermionic>,
    /// The half-bandwidth; the hopping is `t = D/2`.
    d: f64,
    iwn: Vec<Complex64>,
    /// Index of the lowest positive Matsubara frequency, `n = 1`.
    iw0: usize,
    /// `mirror[i]` is the index of the frequency `−ν_i`.
    mirror: Vec<usize>,
}

/// Whether the loop enforces the particle-hole symmetry of the half-filled
/// model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Symmetry {
    /// Project each new `Σ(iν)` onto `i·½[Im Σ(iν) − Im Σ(−iν)]`.
    ParticleHole,
    /// Leave the self-energy as the impurity solver returns it.
    Unconstrained,
}

/// What one run of the loop leaves behind.
pub struct Solution {
    pub green: Vec<Complex64>,
    pub self_energy: Vec<Complex64>,
    /// The relative change of the self-energy at each iteration.
    pub residuals: Vec<f64>,
    /// How far `Σ(iν)` is from particle-hole symmetric at each iteration:
    /// the largest `|Σ(iν) − i·½[Im Σ(iν) − Im Σ(−iν)]|`.
    pub asymmetry: Vec<f64>,
}

impl Dmft {
    pub fn new(beta: f64, d: f64, eps: f64) -> Result<Self, Error> {
        let wmax = 2.0 * d;
        let kernel = LogisticKernel::new(beta * wmax)?;
        let basis =
            FiniteTempBasis::<LogisticKernel, Fermionic>::new(kernel, beta, Some(eps), None)?;
        let mesh = IrMesh::<Fermionic>::new(&basis)?;
        let iwn: Vec<Complex64> = mesh
            .wn()
            .iter()
            .map(|w| Complex64::new(0.0, w.value(beta)))
            .collect();
        let iw0 = mesh
            .wn()
            .iter()
            .position(|w| w.n() == 1)
            .expect("the fermionic sampling frequencies always include n = 1");
        // The default Matsubara sampling points come in ± pairs.
        let ns: Vec<i64> = mesh.wn().iter().map(|w| w.n()).collect();
        let mirror = ns
            .iter()
            .map(|&n| {
                ns.iter()
                    .position(|&m| m == -n)
                    .expect("the fermionic sampling frequencies come in ± pairs")
            })
            .collect();
        Ok(Self {
            basis,
            mesh,
            d,
            iwn,
            iw0,
            mirror,
        })
    }

    pub fn basis(&self) -> &FiniteTempBasis<LogisticKernel, Fermionic> {
        &self.basis
    }

    pub fn mesh(&self) -> &IrMesh<Fermionic> {
        &self.mesh
    }

    /// `G⁰(iν)` of the semicircular density of states, through its spectral
    /// representation `g_l = −s_l ρ_l`, with `ρ_l = ∫dω v_l(ω) ρ(ω)` computed by
    /// Gauss-Legendre quadrature after a substitution that removes the
    /// square-root band edges (see [`shifted_semicircle_overlaps`]).
    pub fn noninteracting(&self) -> Result<Vec<Complex64>, Error> {
        // ANCHOR: noninteracting
        let rho_l = shifted_semicircle_overlaps(&self.basis, 0.0, self.d, 1.0);
        let g_l: Vec<Complex64> = self
            .basis
            .s()
            .iter()
            .zip(&rho_l)
            .map(|(s, rho)| Complex64::new(-s * rho, 0.0))
            .collect();
        self.mesh.l_to_wn(&g_l, 1)
        // ANCHOR_END: noninteracting
    }

    /// The DMFT loop, with particle-hole symmetry enforced.
    ///
    /// Each iteration is the impurity solver `Σ(τ) = U² 𝒢(τ)³` — one round
    /// trip through the basis — followed by the Dyson equation and the
    /// Bethe-lattice self-consistency `𝒢⁻¹ = iν − t² G_loc`.
    pub fn solve(
        &self,
        g_loc: &[Complex64],
        u: f64,
        maxiter: usize,
        tol: f64,
    ) -> Result<Solution, Error> {
        self.solve_with(g_loc, u, maxiter, tol, Symmetry::ParticleHole)
    }

    /// The DMFT loop, with the symmetry projection chosen by `symmetry`.
    pub fn solve_with(
        &self,
        g_loc: &[Complex64],
        u: f64,
        maxiter: usize,
        tol: f64,
        symmetry: Symmetry,
    ) -> Result<Solution, Error> {
        let t = self.d / 2.0;
        let self_consistency = |g_loc: &[Complex64]| -> Vec<Complex64> {
            self.iwn
                .iter()
                .zip(g_loc)
                .map(|(iw, g)| (iw - t * t * g).inv())
                .collect()
        };

        let mut sigma = vec![Complex64::default(); self.iwn.len()];
        let mut g_loc = g_loc.to_vec();
        let mut g_weiss = self_consistency(&g_loc);
        let mut residuals = Vec::new();
        let mut asymmetry = Vec::new();

        for _ in 0..maxiter {
            let previous = sigma.clone();

            // ANCHOR: ipt_step
            // Σ(τ) = U² 𝒢(τ)³, which is the whole impurity solver.
            let g_tau = self.mesh.wn_to_tau(&g_weiss, 1)?;
            let sigma_tau: Vec<Complex64> = g_tau.iter().map(|g| u * u * g * g * g).collect();
            let mut fresh = self.mesh.tau_to_wn(&sigma_tau, 1)?;
            // ANCHOR_END: ipt_step
            // ANCHOR: symmetrize
            if symmetry == Symmetry::ParticleHole {
                fresh = self.particle_hole_symmetric(&fresh);
            }
            // ANCHOR_END: symmetrize
            for (slot, new) in sigma.iter_mut().zip(&fresh) {
                *slot = new * MIX + *slot * (1.0 - MIX);
            }

            g_loc = g_weiss
                .iter()
                .zip(&sigma)
                .map(|(g, s)| (g.inv() - s).inv())
                .collect();
            g_weiss = self_consistency(&g_loc);

            // At `U = 0` the self-energy stays identically zero, and the
            // relative change of nothing is not a number; that case is
            // converged.
            let scale: f64 = sigma.iter().map(|s| s.norm()).sum();
            let change: f64 = sigma
                .iter()
                .zip(&previous)
                .map(|(a, b)| (a - b).norm())
                .sum();
            residuals.push(if scale > 0.0 { change / scale } else { 0.0 });
            asymmetry.push(
                sigma
                    .iter()
                    .zip(self.particle_hole_symmetric(&sigma))
                    .map(|(a, b)| (a - b).norm())
                    .fold(0.0, f64::max),
            );
            if residuals[residuals.len() - 1] < tol {
                break;
            }
        }

        Ok(Solution {
            green: g_loc,
            self_energy: sigma,
            residuals,
            asymmetry,
        })
    }

    /// The particle-hole symmetric part of `Σ(iν)`: purely imaginary and odd,
    /// `i·½[Im Σ(iν) − Im Σ(−iν)]`.
    pub fn particle_hole_symmetric(&self, sigma: &[Complex64]) -> Vec<Complex64> {
        self.mirror
            .iter()
            .zip(sigma)
            .map(|(&j, s)| Complex64::new(0.0, 0.5 * (s.im - sigma[j].im)))
            .collect()
    }

    /// `Z` from the slope of `Im Σ` between the two lowest positive
    /// frequencies: a finite difference, `∂ Im Σ/∂ν ≈ (Im Σ(iν₃) − Im Σ(iν₁))
    /// / (2π/β)` with the reduced indices `n = 1, 3`.
    ///
    /// A negative slope means the self-energy turns upwards as `ν → 0`, which
    /// is the insulator; `Z` is zero there rather than negative.
    pub fn renormalisation(&self, sigma: &[Complex64]) -> f64 {
        let slope = (sigma[self.iw0 + 1] - sigma[self.iw0]).im;
        let beta = self.mesh.beta();
        let z = 1.0 / (1.0 - slope * beta / (2.0 * std::f64::consts::PI));
        z.max(0.0)
    }

    /// The `n` of each sampling frequency, as a column.
    pub fn frequency_column(&self) -> Vec<f64> {
        self.mesh.wn().iter().map(|w| w.n() as f64).collect()
    }
}
