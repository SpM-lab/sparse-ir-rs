//! Finite temperature basis for SparseIR
//!
//! This module provides the `FiniteTempBasis` type which represents the
//! intermediate representation (IR) basis for a given temperature.

use std::sync::Arc;

use crate::error::{
    Error, require_accuracy, require_nonzero_size, require_positive_finite, require_threshold,
};
use crate::kernel::{CentrosymmKernel, KernelProperties, LogisticKernel};
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector, default_sampling_points};
use crate::polyfourier::PiecewiseLegendreFTVector;
use crate::sve::{SVEResult, TworkType, compute_sve};
use crate::traits::{Bosonic, Fermionic, StatisticsType};

// Re-export Statistics enum for C-API
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Statistics {
    Fermionic,
    Bosonic,
}

/// Finite temperature basis for imaginary time/frequency Green's functions
///
/// For a continuation kernel `K` from real frequencies `ω ∈ [-ωmax, ωmax]` to
/// imaginary time `τ ∈ [0, β]`, this type stores the truncated singular
/// value expansion or IR basis:
///
/// ```text
/// K(τ, ω) ≈ sum(u[l](τ) * s[l] * v[l](ω) for l in 0..L)
/// ```
///
/// This basis is inferred from a reduced form by appropriate scaling of
/// the variables.
///
/// # Type Parameters
///
/// * `K` - Kernel type implementing `KernelProperties + CentrosymmKernel`
/// * `S` - Statistics type (`Fermionic` or `Bosonic`)
#[derive(Clone)]
pub struct FiniteTempBasis<K, S>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    /// The kernel used to construct this basis
    kernel: K,

    /// The SVE result (in scaled variables)
    sve_result: Arc<SVEResult>,

    /// Accuracy of the basis (relative error)
    accuracy: f64,

    /// Inverse temperature β
    beta: f64,

    /// Left singular functions on imaginary time axis τ ∈ [0, β]
    /// Arc for efficient sharing (large immutable data)
    u: Arc<PiecewiseLegendrePolyVector>,

    /// Right singular functions on real frequency axis ω ∈ [-ωmax, ωmax]
    /// Arc for efficient sharing (large immutable data)
    v: Arc<PiecewiseLegendrePolyVector>,

    /// Singular values
    s: Vec<f64>,

    /// Left singular functions on Matsubara frequency axis (Fourier transform of u)
    /// Arc for efficient sharing (large immutable data)
    uhat: Arc<PiecewiseLegendreFTVector<S>>,

    /// Full uhat (before truncation to basis size)
    /// Arc for efficient sharing (large immutable data, used for Matsubara sampling)
    uhat_full: Arc<PiecewiseLegendreFTVector<S>>,

    _phantom: std::marker::PhantomData<S>,
}

/// Check that `sve_result` is an SVE on [-1, 1] × [-1, 1], the domain of
/// the kernels in the scaled variables x = 2τ/β - 1 and y = ω/ωmax
///
/// The tolerance is the one of the Fourier transform of the basis functions
/// (`PiecewiseLegendreFT::new`), which requires the interval [-1, 1].
fn check_unit_domain(sve_result: &SVEResult) -> Result<(), Error> {
    let domain =
        |funcs: &PiecewiseLegendrePolyVector| funcs.get_polys().first().map(|p| (p.xmin, p.xmax));
    let (Some(u), Some(v)) = (domain(&sve_result.u), domain(&sve_result.v)) else {
        return Err(Error::EmptyInput { name: "sve_result" });
    };
    let is_unit = |(lo, hi): (f64, f64)| (lo + 1.0).abs() <= 1e-12 && (hi - 1.0).abs() <= 1e-12;
    if is_unit(u) && is_unit(v) {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name: "sve_result",
            value: format!("an SVE on [{:?}, {:?}] × [{:?}, {:?}]", u.0, u.1, v.0, v.1),
            reason: "must be an SVE on [-1, 1] × [-1, 1]".to_string(),
        })
    }
}

/// Knots and widths of `poly` with the first and last knot set exactly to
/// -1 and 1
///
/// `check_unit_domain` accepts an SVE whose domain differs from [-1, 1] by up
/// to 1e-12 (e.g. from `spir_sve_result_from_matrix`). Scaled to τ and ω,
/// such knots end inside [0, β] and [-ωmax, ωmax], where evaluating at the
/// ends panics. The width of an end segment is recomputed only if its knot
/// moved, so an SVE with exact ends keeps its knots and widths bit for bit.
fn unit_knots_and_widths(poly: &PiecewiseLegendrePoly) -> (Vec<f64>, Vec<f64>) {
    let mut knots = poly.knots.clone();
    let mut widths = poly.delta_x.clone();
    let last = knots.len() - 1;
    if knots[0] != -1.0 {
        knots[0] = -1.0;
        widths[0] = knots[1] - knots[0];
    }
    if knots[last] != 1.0 {
        knots[last] = 1.0;
        widths[last - 1] = knots[last] - knots[last - 1];
    }
    (knots, widths)
}

impl<K, S> FiniteTempBasis<K, S>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType,
{
    // ========== Getters ==========

    /// Get a reference to the kernel
    pub fn kernel(&self) -> &K {
        &self.kernel
    }

    /// Get the SVE result the basis was built from
    ///
    /// It is not truncated to the basis size: a basis limited by `max_size`
    /// or `epsilon` keeps all singular functions of the SVE.
    pub fn sve_result(&self) -> &Arc<SVEResult> {
        &self.sve_result
    }

    /// Get the accuracy of the basis
    pub fn accuracy(&self) -> f64 {
        self.accuracy
    }

    /// Get the inverse temperature β
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Get the left singular functions (u) on imaginary time axis
    pub fn u(&self) -> &Arc<PiecewiseLegendrePolyVector> {
        &self.u
    }

    /// Get the right singular functions (v) on real frequency axis
    pub fn v(&self) -> &Arc<PiecewiseLegendrePolyVector> {
        &self.v
    }

    /// Get the singular values
    pub fn s(&self) -> &[f64] {
        &self.s
    }

    /// Get the left singular functions on Matsubara frequency axis (uhat)
    pub fn uhat(&self) -> &Arc<PiecewiseLegendreFTVector<S>> {
        &self.uhat
    }

    /// Get the full uhat (before truncation)
    ///
    /// Holds the Matsubara transforms of all singular functions of
    /// [`sve_result`](Self::sve_result), not only of the `size()` basis
    /// functions; the default Matsubara sampling points use them.
    pub fn uhat_full(&self) -> &Arc<PiecewiseLegendreFTVector<S>> {
        &self.uhat_full
    }

    // ========== Other methods ==========

    /// Get the frequency cutoff ωmax
    pub fn wmax(&self) -> f64 {
        self.kernel.lambda() / self.beta
    }

    /// Get default Matsubara sampling points as i64 indices (for C-API)
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] if the basis functions have no definite parity
    /// (an SVE that is not centrosymmetric, e.g. from `compute_sve_general`;
    /// #183)
    pub fn default_matsubara_sampling_points_i64(
        &self,
        positive_only: bool,
    ) -> Result<Vec<i64>, Error>
    where
        S: 'static,
    {
        Ok(self
            .default_matsubara_sampling_points(positive_only)?
            .into_iter()
            .map(|f| f.n())
            .collect())
    }

    /// Get default Matsubara sampling points as i64 indices with mitigate parameter (for C-API)
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] if the basis functions have no definite parity
    /// (an SVE that is not centrosymmetric, e.g. from `compute_sve_general`;
    /// #183)
    pub fn default_matsubara_sampling_points_i64_with_mitigate(
        &self,
        positive_only: bool,
        mitigate: bool,
        n_points: usize,
    ) -> Result<Vec<i64>, Error>
    where
        S: 'static,
    {
        if !self.kernel().is_centrosymmetric() {
            return Err(Error::NotSupported {
                what: "default Matsubara sampling points of a basis whose kernel is not \
                       centrosymmetric: they rely on the parity of the basis functions"
                    .to_string(),
            });
        }
        let fence = mitigate;
        let freqs = Self::default_matsubara_sampling_points_impl(
            &self.uhat_full,
            n_points,
            fence,
            positive_only,
        )?;
        Ok(freqs.into_iter().map(|f| f.n()).collect())
    }

    /// Create a new FiniteTempBasis
    ///
    /// # Arguments
    ///
    /// * `kernel` - Kernel implementing `KernelProperties + CentrosymmKernel`
    /// * `beta` - Inverse temperature (β > 0)
    /// * `epsilon` - Accuracy of the basis, in (0, 1). `None` selects the best
    ///   accuracy of the working precision (about 1.6e-16).
    /// * `max_size` - Maximum number of basis functions (optional). It limits
    ///   the basis, not the SVE: the SVE is computed and kept in full, as in
    ///   [`from_sve_result`](Self::from_sve_result) with an untruncated SVE.
    ///   The default sampling points and [`accuracy`](Self::accuracy) of the
    ///   basis use the singular functions beyond it.
    ///
    /// # Returns
    ///
    /// A new FiniteTempBasis
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidParameter`] if `beta` is not positive and finite,
    ///   `epsilon` is not in (0, 1), or `max_size` is `Some(0)`. These are
    ///   checked before the SVE is computed.
    /// * The errors of [`compute_sve`]: [`Error::NonFiniteInput`] if the
    ///   discretized kernel has a non-finite entry, [`Error::DecompositionFailed`]
    ///   if an SVD of the SVE fails
    /// * The errors of [`from_sve_result`](Self::from_sve_result)
    pub fn new(
        kernel: K,
        beta: f64,
        epsilon: Option<f64>,
        max_size: Option<usize>,
    ) -> Result<Self, Error> {
        // Validate before the (expensive) SVE
        require_positive_finite("beta", beta)?;
        require_accuracy("epsilon", epsilon)?;
        require_nonzero_size("max_size", max_size)?;

        // Compute the SVE without a size limit; `from_sve_result` truncates
        // only the basis to `max_size`. The default sampling points of a basis
        // of size L are the roots of u_L (tau) and v_L (omega) and the sign
        // changes of uhat_L or uhat_{L+1} (Matsubara), and its accuracy is
        // s_L / s_0. An SVE truncated to `max_size` functions lacks them, and
        // the point selection would fall back to the extrema of the last
        // function (issue #285).
        //
        // Convention-matched with SparseIR.jl 1.1.4 (src/basis.jl,
        // `FiniteTempBasis(statistics, β, ωmax, ε; max_size, kernel,
        // sve_result=SVEResult(kernel; ε))`): its default SVE takes no `lmax`,
        // and `part(sve_result; ε, max_size)` truncates the basis only. No
        // code was ported.
        let sve_result = compute_sve(
            kernel.clone(),
            epsilon,
            None, // cutoff
            None, // no limit on the number of singular values
            TworkType::Auto,
        )?;

        Self::from_sve_result(kernel, beta, sve_result, epsilon, max_size)
    }

    /// Create basis from existing SVE result
    ///
    /// This is useful when you want to reuse the same SVE computation
    /// for both fermionic and bosonic bases.
    ///
    /// `max_size` (and `epsilon`) truncate the basis functions and singular
    /// values only. `sve_result` is kept as given: the default sampling points
    /// and [`accuracy`](Self::accuracy) use its singular functions beyond the
    /// basis, so pass an untruncated SVE to get the points of SparseIR.jl.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidParameter`] if `beta` is not positive and finite,
    ///   `epsilon` is not in [0, 1) (0 keeps every singular value), `max_size`
    ///   is `Some(0)`, or `sve_result` is not an SVE on [-1, 1] × [-1, 1]
    /// * [`Error::EmptyInput`] if `sve_result` has no singular functions
    /// * The errors of [`SVEResult::part`] (for an `SVEResult` whose public
    ///   fields break its invariants)
    pub fn from_sve_result(
        kernel: K,
        beta: f64,
        sve_result: SVEResult,
        epsilon: Option<f64>,
        max_size: Option<usize>,
    ) -> Result<Self, Error> {
        require_positive_finite("beta", beta)?;
        require_threshold("epsilon", epsilon)?;
        require_nonzero_size("max_size", max_size)?;
        check_unit_domain(&sve_result)?;

        // Get truncated u, s, v from SVE result
        let (u_sve, s_sve, v_sve) = sve_result.part(epsilon, max_size)?;

        // Calculate accuracy
        let accuracy = if sve_result.s.len() > s_sve.len() {
            sve_result.s[s_sve.len()] / sve_result.s[0]
        } else {
            sve_result.s[sve_result.s.len() - 1] / sve_result.s[0]
        };

        // Get kernel parameters
        let lambda = kernel.lambda();
        let omega_max = lambda / beta;

        // Scale polynomials to new variables
        // tau = β/2 * (x + 1), w = ωmax * y

        // Transform u: x ∈ [-1, 1] → τ ∈ [0, β]
        let (x_knots, x_widths) = unit_knots_and_widths(&u_sve.get_polys()[0]);
        let u_knots: Vec<f64> = x_knots.iter().map(|&x| beta / 2.0 * (x + 1.0)).collect();
        let u_delta_x: Vec<f64> = x_widths.iter().map(|&dx| beta / 2.0 * dx).collect();
        let u_symm: Vec<i32> = u_sve.get_polys().iter().map(|p| p.symm).collect();

        let u = u_sve.rescale_domain(u_knots, Some(u_delta_x), Some(u_symm))?;

        // Transform v: y ∈ [-1, 1] → ω ∈ [-ωmax, ωmax]
        let (y_knots, y_widths) = unit_knots_and_widths(&v_sve.get_polys()[0]);
        let v_knots: Vec<f64> = y_knots.iter().map(|&y| omega_max * y).collect();
        let v_delta_x: Vec<f64> = y_widths.iter().map(|&dy| omega_max * dy).collect();
        let v_symm: Vec<i32> = v_sve.get_polys().iter().map(|p| p.symm).collect();

        let v = v_sve.rescale_domain(v_knots, Some(v_delta_x), Some(v_symm))?;

        // Scale singular values to τ = β(x + 1)/2 and ω = ωmax y. A kernel with
        // `ypower` carries that power of y = ω/ωmax, so its physical form is
        // K(τ, ω) = ωmax^ypower K(x, y) and
        // s_scaled = sqrt(β/2 * ωmax) * ωmax^ypower * s_sve,
        // e.g. S_l = sqrt(β ωmax³/2) s_l for RegularizedBoseKernel (irbasis
        // paper, Chikano et al., CPC 240, 181 (2019), arXiv:1807.05237, Eq. (25)).
        let ypower = kernel.ypower();
        let scale_factor = (beta / 2.0 * omega_max).sqrt() * omega_max.powi(ypower);
        let s: Vec<f64> = s_sve.iter().map(|&x| scale_factor * x).collect();

        // Construct uhat (Fourier transform of u)
        // HACK: Fourier transforms only work on unit interval, so we scale the data
        let uhat_base_full = sve_result.u.scale_data(beta.sqrt());
        let conv_rad = kernel.conv_radius();

        // Create statistics marker instance using Default trait
        // S is a zero-sized type (ZST) like Fermionic or Bosonic
        let stat_marker = S::default();

        let uhat_full = PiecewiseLegendreFTVector::<S>::from_poly_vector(
            &uhat_base_full,
            stat_marker,
            Some(conv_rad),
        )?;

        // Truncate uhat to basis size
        let uhat_polyvec: Vec<_> = uhat_full.polyvec.iter().take(s.len()).cloned().collect();
        let uhat = PiecewiseLegendreFTVector::from_vector(uhat_polyvec);

        Ok(Self {
            kernel,
            sve_result: Arc::new(sve_result),
            accuracy,
            beta,
            u: Arc::new(u),
            v: Arc::new(v),
            s,
            uhat: Arc::new(uhat),
            uhat_full: Arc::new(uhat_full),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the size of the basis (number of basis functions)
    pub fn size(&self) -> usize {
        self.s.len()
    }

    /// Get the cutoff parameter Λ = β * ωmax
    pub fn lambda(&self) -> f64 {
        self.kernel.lambda()
    }

    /// Get the frequency cutoff ωmax
    pub fn omega_max(&self) -> f64 {
        self.lambda() / self.beta
    }

    /// Get significance of each singular value (s[i] / s[0])
    pub fn significance(&self) -> Vec<f64> {
        let s0 = self.s[0];
        self.s.iter().map(|&s| s / s0).collect()
    }

    /// Get default tau sampling points
    ///
    /// Returns sampling points in imaginary time τ ∈ [-β/2, β/2].
    ///
    /// Roots are found with symmetry exploitation (matching Python 1.x / Julia v1),
    /// then mapped to [-β/2, β/2] by folding τ_physical ∈ [0, β] around β/2.
    ///
    /// # Errors
    ///
    /// * [`Error::NotSupported`] if the default points are not defined for
    ///   this basis: its SVE has so few singular functions that the last one
    ///   has no extrema (e.g. `compute_sve` with `max_num_svals = Some(2)`)
    pub fn default_tau_sampling_points(&self) -> Result<Vec<f64>, Error> {
        let points = self.default_tau_sampling_points_size_requested(self.size())?;
        let basis_size = self.size();
        if points.len() < basis_size {
            debug_warn!(
                "Number of tau sampling points ({}) is less than basis size ({}). \
                 Basis parameters: beta={}, wmax={}, epsilon={:.2e}",
                points.len(),
                basis_size,
                self.beta,
                self.wmax(),
                self.accuracy()
            );
        }
        Ok(points)
    }

    /// Get default tau sampling points with a requested size
    ///
    /// Returns sampling points in τ ∈ [-β/2, β/2].
    ///
    /// # Errors
    ///
    /// * [`Error::NotSupported`] if the default points are not defined for
    ///   this basis: its SVE has so few singular functions that the last one
    ///   has no extrema (e.g. `compute_sve` with `max_num_svals = Some(2)`)
    pub fn default_tau_sampling_points_size_requested(
        &self,
        size_requested: usize,
    ) -> Result<Vec<f64>, Error> {
        let x = default_sampling_points(&self.sve_result.u, "u", size_requested)?;
        let half_beta = self.beta / 2.0;
        // Map roots to physical tau ∈ [0, β], then fold to [-β/2, β/2]
        let mut smpl_taus: Vec<f64> = x
            .iter()
            .map(|&xi| {
                let tau = half_beta * (xi + 1.0);
                if tau <= half_beta {
                    tau
                } else {
                    tau - self.beta
                }
            })
            .collect();
        smpl_taus.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(smpl_taus)
    }

    /// Get default Matsubara frequency sampling points
    ///
    /// Returns sampling points as MatsubaraFreq objects: the sign changes of the
    /// first discarded Matsubara basis function (its extrema when that function
    /// is not available); bosonic sets always include n = 0.
    ///
    /// # Arguments
    /// * `positive_only` - If true, returns only non-negative frequencies
    ///
    /// # Returns
    /// Vector of Matsubara frequency sampling points
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] if the basis functions have no definite parity
    /// (an SVE that is not centrosymmetric, e.g. from `compute_sve_general`;
    /// #183)
    pub fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<crate::freq::MatsubaraFreq<S>>, Error>
    where
        S: 'static,
    {
        if !self.kernel().is_centrosymmetric() {
            return Err(Error::NotSupported {
                what: "default Matsubara sampling points of a basis whose kernel is not \
                       centrosymmetric: they rely on the parity of the basis functions"
                    .to_string(),
            });
        }
        let fence = false;
        let points = Self::default_matsubara_sampling_points_impl(
            &self.uhat_full,
            self.size(),
            fence,
            positive_only,
        )?;
        let basis_size = self.size();
        // For positive_only=true, we need 2*n_sampling_points >= basis_size
        // For positive_only=false, we need n_sampling_points >= basis_size
        let effective_points = if positive_only {
            2 * points.len()
        } else {
            points.len()
        };
        if effective_points < basis_size {
            debug_warn!(
                "Number of Matsubara sampling points ({}{}) is less than basis size ({}). \
                 Basis parameters: beta={}, wmax={}, epsilon={:.2e}",
                points.len(),
                if positive_only { " × 2" } else { "" },
                basis_size,
                self.beta,
                self.wmax(),
                self.accuracy()
            );
        }
        Ok(points)
    }

    /// Fence Matsubara sampling points to improve conditioning
    ///
    /// This function adds additional sampling points near the outer frequencies
    /// to improve the conditioning of the sampling matrix. This is particularly
    /// important for Matsubara sampling where we cannot freely choose sampling points.
    ///
    /// Implementation matches C++ version in `basis.hpp` (lines 407-452).
    fn fence_matsubara_sampling(
        omega_n: &mut Vec<crate::freq::MatsubaraFreq<S>>,
        positive_only: bool,
    ) where
        S: StatisticsType + 'static,
    {
        use crate::freq::{BosonicFreq, MatsubaraFreq};

        if omega_n.is_empty() {
            return;
        }

        // Collect outer frequencies
        let mut outer_frequencies = Vec::new();
        if positive_only {
            outer_frequencies.push(omega_n[omega_n.len() - 1]);
        } else {
            outer_frequencies.push(omega_n[0]);
            outer_frequencies.push(omega_n[omega_n.len() - 1]);
        }

        for wn_outer in outer_frequencies {
            let outer_val = wn_outer.n();
            // In SparseIR.jl-v1, ωn_diff is always created as BosonicFreq
            // This ensures diff_val is always even (valid for Bosonic)
            let mut diff_val = 2 * (0.025 * outer_val as f64).round() as i64;

            // Handle edge case: if diff_val is 0, set it to 2 (minimum even value for Bosonic)
            if diff_val == 0 {
                diff_val = 2;
            }

            // Get the n value from BosonicFreq (same as diff_val since it's even)
            let wn_diff = BosonicFreq::new(diff_val).unwrap().n();

            // Sign function: returns +1 if n > 0, -1 if n < 0, 0 if n == 0
            // Matches C++ implementation: (a.get_n() > 0) - (a.get_n() < 0)
            let sign_val = if outer_val > 0 {
                1
            } else if outer_val < 0 {
                -1
            } else {
                0
            };

            // Check original size before adding (C++ checks wn.size() before each push)
            let original_size = omega_n.len();
            if original_size >= 20 {
                // For Fermionic: wn_outer.n is odd, wn_diff is even, so wn_outer.n ± wn_diff is odd (valid)
                // For Bosonic: wn_outer.n is even, wn_diff is even, so wn_outer.n ± wn_diff is even (valid)
                let new_n = outer_val - sign_val * wn_diff;
                if let Ok(new_freq) = MatsubaraFreq::<S>::new(new_n) {
                    omega_n.push(new_freq);
                }
            }
            if original_size >= 42 {
                let new_n = outer_val + sign_val * wn_diff;
                if let Ok(new_freq) = MatsubaraFreq::<S>::new(new_n) {
                    omega_n.push(new_freq);
                }
            }
        }

        // Sort and remove duplicates using BTreeSet
        let omega_n_set: std::collections::BTreeSet<MatsubaraFreq<S>> = omega_n.drain(..).collect();
        *omega_n = omega_n_set.into_iter().collect();
    }

    /// Default Matsubara sampling points for a basis of size `l` from the
    /// Matsubara basis functions `uhat_full`: the sign changes of
    /// `uhat_full[l]` (after the parity adjustment of `l`), or the extrema of
    /// the last function when `uhat_full` has no function `l`; bosonic sets
    /// always include n = 0. `fence` adds points near the outer frequencies.
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `uhat_full` is empty
    /// * [`Error::NotSupported`] if the functions have no definite parity
    ///   (symm = 0, as from an SVE that is not centrosymmetric, #183)
    pub fn default_matsubara_sampling_points_impl(
        uhat_full: &PiecewiseLegendreFTVector<S>,
        l: usize,
        fence: bool,
        positive_only: bool,
    ) -> Result<Vec<crate::freq::MatsubaraFreq<S>>, Error>
    where
        S: StatisticsType + 'static,
    {
        use crate::freq::MatsubaraFreq;
        use crate::polyfourier::{find_extrema, sign_changes};
        use std::collections::BTreeSet;

        if uhat_full.is_empty() {
            return Err(Error::EmptyInput { name: "uhat_full" });
        }

        let mut l_requested = l;

        // Adjust l_requested based on statistics (same as C++)
        if S::STATISTICS == crate::traits::Statistics::Fermionic && l_requested % 2 != 0 {
            l_requested += 1;
        } else if S::STATISTICS == crate::traits::Statistics::Bosonic && l_requested % 2 == 0 {
            l_requested += 1;
        }

        // Choose sign_changes or find_extrema based on l_requested
        let mut omega_n = if l_requested < uhat_full.len() {
            sign_changes(&uhat_full[l_requested], positive_only)?
        } else {
            find_extrema(&uhat_full[uhat_full.len() - 1], positive_only)?
        };

        // For bosons, include zero frequency explicitly to prevent conditioning issues
        if S::STATISTICS == crate::traits::Statistics::Bosonic {
            omega_n.push(MatsubaraFreq::<S>::new(0).unwrap());
        }

        // Sort and remove duplicates using BTreeSet
        let omega_n_set: BTreeSet<MatsubaraFreq<S>> = omega_n.into_iter().collect();
        let mut omega_n: Vec<MatsubaraFreq<S>> = omega_n_set.into_iter().collect();

        // Check expected size
        let expected_size = if positive_only {
            l_requested.div_ceil(2)
        } else {
            l_requested
        };

        if omega_n.len() != expected_size {
            debug_warn!(
                "Requested {} sampling frequencies for basis size L = {}, but got {}.",
                expected_size,
                l,
                omega_n.len()
            );
        }

        // Apply fencing if requested (same as C++ implementation)
        if fence {
            Self::fence_matsubara_sampling(&mut omega_n, positive_only);
        }

        Ok(omega_n)
    }
    /// Get default omega (real frequency) sampling points
    ///
    /// Returns sampling points on the real-frequency axis ω ∈ [-ωmax, ωmax].
    /// These are used as pole locations for the Discrete Lehmann Representation (DLR).
    ///
    /// The sampling points are chosen as the roots of the L-th basis function
    /// in the spectral domain (v), which provides near-optimal conditioning.
    ///
    /// # Returns
    /// Vector of real-frequency sampling points in [-ωmax, ωmax]
    ///
    /// # Errors
    ///
    /// * [`Error::NotSupported`] if the default points are not defined for
    ///   this basis: its SVE has so few singular functions that the last one
    ///   has no extrema (e.g. `compute_sve` with `max_num_svals = Some(2)`)
    pub fn default_omega_sampling_points(&self) -> Result<Vec<f64>, Error> {
        let sz = self.size();

        // Use UNTRUNCATED sve_result.v (same as C++)
        // C++: default_sampling_points(*(sve_result->v), sz)
        let y = default_sampling_points(&self.sve_result.v, "v", sz)?;

        // Scale to [-ωmax, ωmax]
        let wmax = self.kernel.lambda() / self.beta;
        let omega_points: Vec<f64> = y.into_iter().map(|yi| wmax * yi).collect();

        Ok(omega_points)
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<K, S> crate::basis_trait::Basis<S> for FiniteTempBasis<K, S>
where
    K: KernelProperties + CentrosymmKernel + Clone + 'static,
    S: StatisticsType + 'static,
{
    type Kernel = K;

    fn kernel(&self) -> &Self::Kernel {
        &self.kernel
    }

    fn beta(&self) -> f64 {
        self.beta
    }

    fn wmax(&self) -> f64 {
        self.kernel.lambda() / self.beta
    }

    fn lambda(&self) -> f64 {
        self.kernel.lambda()
    }

    fn size(&self) -> usize {
        self.size()
    }

    fn accuracy(&self) -> f64 {
        self.accuracy
    }

    fn significance(&self) -> Vec<f64> {
        if let Some(&first_s) = self.s.first() {
            self.s.iter().map(|&s| s / first_s).collect()
        } else {
            vec![]
        }
    }

    fn svals(&self) -> Vec<f64> {
        self.s.clone()
    }

    fn default_tau_sampling_points(&self) -> Result<Vec<f64>, Error> {
        self.default_tau_sampling_points()
    }

    fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<crate::freq::MatsubaraFreq<S>>, Error> {
        self.default_matsubara_sampling_points(positive_only)
    }

    fn evaluate_tau(&self, tau: &[f64]) -> Result<mdarray::DTensor<f64, 2>, Error> {
        use crate::taufuncs::normalize_tau;
        use mdarray::DTensor;

        let basis_size = self.size();

        // Normalize every τ ∈ [-β, β] to [0, β] with the statistics sign
        // first; this rejects a τ outside [-β, β] and NaN. The u polynomials
        // are on [0, β] exactly (from_sve_result sets their ends), so
        // evaluating them cannot fail.
        let normalized = tau
            .iter()
            .map(|&t| normalize_tau::<S>(t, self.beta))
            .collect::<Result<Vec<(f64, f64)>, Error>>()?;
        if normalized.is_empty() {
            // mdarray 0.7.2 runs the closure of from_fn for a zero extent
            // (https://github.com/fre-hu/mdarray/issues/21).
            return Ok(DTensor::<f64, 2>::from_elem([0, basis_size], 0.0));
        }

        // Result: matrix[i, l] = u_l(tau[i])
        Ok(DTensor::<f64, 2>::from_fn(
            [normalized.len(), basis_size],
            |idx| {
                let (tau_norm, sign) = normalized[idx[0]];
                sign * self.u[idx[1]].evaluate(tau_norm)
            },
        ))
    }

    fn evaluate_matsubara(
        &self,
        freqs: &[crate::freq::MatsubaraFreq<S>],
    ) -> Result<mdarray::DTensor<num_complex::Complex<f64>, 2>, Error> {
        use mdarray::DTensor;
        use num_complex::Complex;

        let n_points = freqs.len();
        let basis_size = self.size();
        if n_points == 0 {
            // See evaluate_tau (mdarray#21).
            return Ok(DTensor::<Complex<f64>, 2>::from_elem(
                [0, basis_size],
                Complex::new(0.0, 0.0),
            ));
        }

        // Evaluate each basis function at all Matsubara frequencies
        // Result: matrix[i, l] = uhat_l(iν[i])
        Ok(DTensor::<Complex<f64>, 2>::from_fn(
            [n_points, basis_size],
            |idx| {
                let i = idx[0]; // frequency index
                let l = idx[1]; // basis function index
                self.uhat[l].evaluate(&freqs[i])
            },
        ))
    }

    fn evaluate_omega(&self, omega: &[f64]) -> Result<mdarray::DTensor<f64, 2>, Error> {
        use mdarray::DTensor;

        let basis_size = self.size();
        // The v polynomials are on [-ωmax, ωmax] exactly (from_sve_result
        // sets their ends).
        let domain = (self.v.xmin(), self.v.xmax());
        if let Some(&w) = omega.iter().find(|&&w| !(w >= domain.0 && w <= domain.1)) {
            return Err(Error::OutOfDomain {
                name: "omega",
                value: w,
                domain,
            });
        }
        if omega.is_empty() {
            // See evaluate_tau (mdarray#21).
            return Ok(DTensor::<f64, 2>::from_elem([0, basis_size], 0.0));
        }

        // Result: matrix[i, l] = V_l(omega[i])
        Ok(DTensor::<f64, 2>::from_fn(
            [omega.len(), basis_size],
            |idx| self.v[idx[1]].evaluate(omega[idx[0]]),
        ))
    }

    fn default_omega_sampling_points(&self) -> Result<Vec<f64>, Error> {
        self.default_omega_sampling_points()
    }
}

// ============================================================================
// Type aliases
// ============================================================================

/// Type alias for fermionic basis with LogisticKernel
pub type FermionicBasis = FiniteTempBasis<LogisticKernel, Fermionic>;

/// Type alias for bosonic basis with LogisticKernel
pub type BosonicBasis = FiniteTempBasis<LogisticKernel, Bosonic>;

#[cfg(test)]
#[path = "basis_tests.rs"]
mod basis_tests;
