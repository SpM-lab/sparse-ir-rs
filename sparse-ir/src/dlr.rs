//! Discrete Lehmann Representation (DLR)
//!
//! This module provides the Discrete Lehmann Representation (DLR) basis,
//! which represents Green's functions as a linear combination of poles on the
//! real-frequency axis.

use crate::fitter::RealMatrixFitter;
use crate::freq::MatsubaraFreq;
use crate::gemm::GemmBackendHandle;
use crate::kernel::AbstractKernel;
use crate::traits::{Statistics, StatisticsType};
use mdarray::DTensor;
use num_complex::Complex;
use std::marker::PhantomData;

/// Generic single-pole Green's function at imaginary time τ
///
/// Computes G(τ) for either fermionic or bosonic statistics based on the type parameter S.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// use sparse_ir::traits::Fermionic;
/// let g_f = gtau_single_pole::<Fermionic>(0.5, 5.0, 1.0);
///
/// use sparse_ir::traits::Bosonic;
/// let g_b = gtau_single_pole::<Bosonic>(0.5, 5.0, 1.0);
/// ```
pub fn gtau_single_pole<S: StatisticsType>(tau: f64, omega: f64, beta: f64) -> f64 {
    match S::STATISTICS {
        Statistics::Fermionic => fermionic_single_pole(tau, omega, beta),
        Statistics::Bosonic => bosonic_single_pole(tau, omega, beta),
    }
}

/// Compute fermionic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = -exp(-ω×τ) / (1 + exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports extended τ ranges with anti-periodic boundary conditions:
/// - G(τ + β) = -G(τ) (fermionic anti-periodicity)
/// - Valid for τ ∈ (-β, 2β)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = fermionic_single_pole(tau, omega, beta);
/// ```
pub fn fermionic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    use crate::taufuncs::normalize_tau;
    use crate::traits::Fermionic;

    // Normalize τ to [0, β] and track sign from anti-periodicity
    // G(τ + β) = -G(τ) for fermions
    let (tau_normalized, sign) = normalize_tau::<Fermionic>(tau, beta);

    sign * (-(-omega * tau_normalized).exp() / (1.0 + (-beta * omega).exp()))
}

/// Compute bosonic single-pole Green's function at imaginary time τ
///
/// Evaluates G(τ) = exp(-ω×τ) / (1 - exp(-β×ω)) for a single pole at frequency ω.
///
/// Supports extended τ ranges with periodic boundary conditions:
/// - G(τ + β) = G(τ) (bosonic periodicity)
/// - Valid for τ ∈ (-β, 2β)
///
/// # Arguments
/// * `tau` - Imaginary time (can be outside [0, β))
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Real-valued Green's function G(τ)
///
/// # Example
/// ```ignore
/// let beta = 1.0;
/// let omega = 5.0;
/// let tau = 0.5 * beta;
/// let g = bosonic_single_pole(tau, omega, beta);
/// ```
pub fn bosonic_single_pole(tau: f64, omega: f64, beta: f64) -> f64 {
    use crate::taufuncs::normalize_tau;
    use crate::traits::Bosonic;

    // Normalize τ to [0, β] using periodicity
    // G(τ + β) = G(τ) for bosons
    let tau_normalized = normalize_tau::<Bosonic>(tau, beta).0;

    (-omega * tau_normalized).exp() / (1.0 - (-beta * omega).exp())
}

/// Generic single-pole Green's function at Matsubara frequency
///
/// Computes G(iωn) = 1/(iωn - ω) for a single pole at frequency ω.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `matsubara_freq` - Matsubara frequency
/// * `omega` - Pole position (real frequency)
/// * `beta` - Inverse temperature
///
/// # Returns
/// Complex-valued Green's function G(iωn)
pub fn giwn_single_pole<S: StatisticsType>(
    matsubara_freq: &MatsubaraFreq<S>,
    omega: f64,
    beta: f64,
) -> Complex<f64> {
    // G(iωn) = 1/(iωn - ω)
    let wn = matsubara_freq.value(beta);
    let denominator = Complex::new(0.0, 1.0) * wn - Complex::new(omega, 0.0);
    Complex::new(1.0, 0.0) / denominator
}

// ============================================================================
// Discrete Lehmann Representation
// ============================================================================

/// Discrete Lehmann Representation (DLR)
///
/// The DLR is a variant of the IR basis based on a "sketching" of the analytic
/// continuation kernel K. Instead of using singular value expansion, it represents
/// Green's functions as a linear combination of poles on the real-frequency axis:
///
/// ```text
/// G(iν) = Σ_i a[i] * reg[i] / (iν - ω[i])
/// ```
///
/// where:
/// - `ω[i]` are pole positions on the real axis
/// - `a[i]` are expansion coefficients
/// - `reg[i]` are regularization factors (1 for fermions, tanh(βω/2) for bosons)
///
/// Note: DLR always uses LogisticKernel-type weights, regardless of the IR basis kernel type.
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
pub struct DiscreteLehmannRepresentation<S>
where
    S: StatisticsType,
{
    /// Pole positions on the real-frequency axis ω ∈ [-ωmax, ωmax]
    pub poles: Vec<f64>,

    /// Inverse temperature β
    pub beta: f64,

    /// Maximum frequency ωmax
    pub wmax: f64,

    /// LogisticKernel used for weight computations
    /// DLR always uses LogisticKernel regardless of the IR basis kernel type
    kernel: crate::kernel::LogisticKernel,

    /// Accuracy of the representation
    pub accuracy: f64,

    /// Inverse weights for each pole: inv_weight[i] = 1 / weight[i]
    /// Always computed using LogisticKernel:
    /// - Fermionic: inv_weight = 1.0
    /// - Bosonic: inv_weight = tanh(β·ω/2)
    pub inv_weights: Vec<f64>,

    /// Fitting matrix from IR: fitmat = -s · V(poles)
    /// Used for to_IR transformation
    fitmat: DTensor<f64, 2>,

    /// Fitter for from_IR transformation (uses SVD of fitmat)
    fitter: RealMatrixFitter,

    /// Marker for statistics type
    _phantom: PhantomData<S>,
}

impl<S> DiscreteLehmannRepresentation<S>
where
    S: StatisticsType,
{
    /// Create DLR from IR basis with custom poles
    ///
    /// Note: Always uses LogisticKernel-type weights, regardless of the basis kernel type.
    ///
    /// # Arguments
    /// * `basis` - The IR basis to construct DLR from
    /// * `poles` - Pole positions on the real-frequency axis
    ///
    /// # Returns
    /// A new DLR representation
    pub fn with_poles<K>(
        basis: &impl crate::basis_trait::Basis<S, Kernel = K>,
        poles: Vec<f64>,
    ) -> Self
    where
        S: 'static,
        K: crate::kernel::KernelProperties + Clone,
    {
        use crate::kernel::{KernelProperties, LogisticKernel};

        let beta = basis.beta();
        let wmax = basis.wmax();
        let accuracy = basis.accuracy();

        // Compute fitting matrix: fitmat = -s · V(poles)
        // This transforms DLR coefficients to IR coefficients
        let v_at_poles = basis.evaluate_omega(&poles); // shape: [n_poles, basis_size]
        let s = basis.svals(); // Non-normalized singular values (same as C++)

        let basis_size = basis.size();
        let n_poles = poles.len();

        // fitmat[l, i] = -s[l] * V_l(pole[i])
        // C++: fitmat = (-A_array * s_array.replicate(1, A.cols())).matrix()
        let fitmat = DTensor::<f64, 2>::from_fn([basis_size, n_poles], |idx| {
            let l = idx[0];
            let i = idx[1];
            -s[l] * v_at_poles[[i, l]]
        });

        // Create fitter for from_IR (inverse operation)
        let fitter = RealMatrixFitter::new(fitmat.clone());

        // Compute inverse weights for each pole using LogisticKernel
        // (regardless of the basis kernel type)
        let lambda = beta * wmax;
        let logistic_kernel = LogisticKernel::new(lambda);
        let inv_weights: Vec<f64> = poles
            .iter()
            .map(|&pole| logistic_kernel.inv_weight::<S>(beta, pole))
            .collect();

        Self {
            poles,
            beta,
            wmax,
            kernel: logistic_kernel,
            accuracy,
            inv_weights,
            fitmat,
            fitter,
            _phantom: PhantomData,
        }
    }

    /// Create DLR from IR basis with default pole locations
    ///
    /// Uses the default omega sampling points from the basis.
    ///
    /// # Arguments
    /// * `basis` - The IR basis to construct DLR from
    ///
    /// # Returns
    /// A new DLR representation with default poles
    pub fn new<K>(basis: &impl crate::basis_trait::Basis<S, Kernel = K>) -> Self
    where
        S: 'static,
        K: crate::kernel::KernelProperties + Clone,
    {
        let poles = basis.default_omega_sampling_points();
        assert!(
            basis.size() <= poles.len(),
            "The number of poles must be greater than or equal to the basis size"
        );
        Self::with_poles(basis, poles)
    }

    // ========================================================================
    // Public API (generic, user-friendly)
    // ========================================================================

    /// Convert IR coefficients to DLR (N-dimensional, generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `gl` - IR coefficients as N-D tensor
    /// * `dim` - Dimension along which to transform
    ///
    /// # Returns
    /// DLR coefficients as N-D tensor
    pub fn from_ir_nd<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        gl: &mdarray::Tensor<T, mdarray::DynRank>,
        dim: usize,
    ) -> mdarray::Tensor<T, mdarray::DynRank>
    where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + From<f64>
            + Copy
            + Default
            + 'static,
    {
        use mdarray::{DTensor, Shape};

        let mut gl_shape = vec![];
        gl.shape().with_dims(|dims| {
            gl_shape.extend_from_slice(dims);
        });

        let basis_size = gl_shape[dim];
        assert_eq!(
            basis_size,
            self.fitmat.shape().0,
            "IR basis size mismatch: expected {}, got {}",
            self.fitmat.shape().0,
            basis_size
        );

        // Move target dimension to position 0
        let gl_dim0 = crate::sampling::movedim(gl, dim, 0);

        // Reshape to 2D
        let extra_size = gl_dim0.len() / basis_size;
        let gl_2d_dyn = gl_dim0.reshape(&[basis_size, extra_size][..]).to_tensor();

        let gl_2d = DTensor::<T, 2>::from_fn([basis_size, extra_size], |idx| {
            gl_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Fit using fitter's generic 2D method
        let g_dlr_2d = self.fitter.fit_2d_generic::<T>(backend, &gl_2d);

        // Reshape back
        let n_poles = self.poles.len();
        let mut g_dlr_shape = vec![n_poles];
        gl_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                g_dlr_shape.push(dims[i]);
            }
        });

        let g_dlr_dim0 = g_dlr_2d.into_dyn().reshape(&g_dlr_shape[..]).to_tensor();
        crate::sampling::movedim(&g_dlr_dim0, 0, dim)
    }

    /// Convert DLR coefficients to IR (N-dimensional, generic over real/complex)
    ///
    /// # Type Parameters
    /// * `T` - Element type (f64 or Complex<f64>)
    ///
    /// # Arguments
    /// * `g_dlr` - DLR coefficients as N-D tensor
    /// * `dim` - Dimension along which to transform
    ///
    /// # Returns
    /// IR coefficients as N-D tensor
    pub fn to_ir_nd<T>(
        &self,
        backend: Option<&GemmBackendHandle>,
        g_dlr: &mdarray::Tensor<T, mdarray::DynRank>,
        dim: usize,
    ) -> mdarray::Tensor<T, mdarray::DynRank>
    where
        T: num_complex::ComplexFloat
            + faer_traits::ComplexField
            + From<f64>
            + Copy
            + Default
            + 'static,
    {
        use mdarray::{DTensor, Shape};

        let mut g_dlr_shape = vec![];
        g_dlr.shape().with_dims(|dims| {
            g_dlr_shape.extend_from_slice(dims);
        });

        let n_poles = g_dlr_shape[dim];
        assert_eq!(
            n_poles,
            self.poles.len(),
            "DLR size mismatch: expected {}, got {}",
            self.poles.len(),
            n_poles
        );

        // Move target dimension to position 0
        let g_dlr_dim0 = crate::sampling::movedim(g_dlr, dim, 0);

        // Reshape to 2D
        let extra_size = g_dlr_dim0.len() / n_poles;
        let g_dlr_2d_dyn = g_dlr_dim0.reshape(&[n_poles, extra_size][..]).to_tensor();

        let g_dlr_2d = DTensor::<T, 2>::from_fn([n_poles, extra_size], |idx| {
            g_dlr_2d_dyn[&[idx[0], idx[1]][..]]
        });

        // Evaluate using fitter's generic 2D method
        let gl_2d = self.fitter.evaluate_2d_generic::<T>(backend, &g_dlr_2d);

        // Reshape back
        let basis_size = self.fitmat.shape().0;
        let mut gl_shape = vec![basis_size];
        g_dlr_dim0.shape().with_dims(|dims| {
            for i in 1..dims.len() {
                gl_shape.push(dims[i]);
            }
        });

        let gl_dim0 = gl_2d.into_dyn().reshape(&gl_shape[..]).to_tensor();
        crate::sampling::movedim(&gl_dim0, 0, dim)
    }
}

// ============================================================================
// Basis trait implementation for DLR
// ============================================================================

impl<S> crate::basis_trait::Basis<S> for DiscreteLehmannRepresentation<S>
where
    S: StatisticsType + 'static,
{
    type Kernel = crate::kernel::LogisticKernel;

    fn kernel(&self) -> &Self::Kernel {
        // DLR always uses LogisticKernel for weight computations
        &self.kernel
    }

    fn beta(&self) -> f64 {
        self.beta
    }

    fn wmax(&self) -> f64 {
        self.wmax
    }

    fn lambda(&self) -> f64 {
        self.beta * self.wmax
    }

    fn size(&self) -> usize {
        self.poles.len()
    }

    fn accuracy(&self) -> f64 {
        self.accuracy
    }

    fn significance(&self) -> Vec<f64> {
        // All poles are equally significant in DLR
        vec![1.0; self.poles.len()]
    }

    fn svals(&self) -> Vec<f64> {
        // All poles are equally significant in DLR (no singular value concept)
        vec![1.0; self.poles.len()]
    }

    fn default_tau_sampling_points(&self) -> Vec<f64> {
        // TODO: Could use the original IR basis sampling points
        // For now, return empty - not well-defined for DLR
        vec![]
    }

    fn default_matsubara_sampling_points(
        &self,
        _positive_only: bool,
    ) -> Vec<crate::freq::MatsubaraFreq<S>> {
        // TODO: Could use the original IR basis sampling points
        // For now, return empty - not well-defined for DLR
        vec![]
    }

    fn evaluate_tau(&self, tau: &[f64]) -> mdarray::DTensor<f64, 2> {
        use crate::taufuncs::normalize_tau;
        use mdarray::DTensor;

        let n_points = tau.len();
        let n_poles = self.poles.len();

        // Evaluate tau-domain DLR basis functions:
        //   u_i(τ) = sign * ( -K_logistic(x, y_i) )
        // where x = 2τ/β - 1, y_i = pole_i/ωmax and
        //   sign encodes (anti-)periodicity:
        //     Fermionic: G(τ + β) = -G(τ)
        //     Bosonic:  G(τ + β) =  G(τ)
        //
        // NOTE: Statistics-dependent regularization factors
        // (tanh(βω/2) for bosons) are applied only in the
        // Matsubara representation via `inv_weights` and
        // do NOT enter the tau basis functions themselves.

        let mut result = DTensor::<f64, 2>::zeros([n_points, n_poles]);

        // Loop over tau points: for each tau, normalize once and compute for all poles
        for i in 0..n_points {
            let tau_val = tau[i];

            // Normalize tau to [0, β] with statistics-dependent sign
            let (tau_norm, sign) = normalize_tau::<S>(tau_val, self.beta);

            // Compute x coordinate once for this tau
            let x = 2.0 * tau_norm / self.beta - 1.0;

            // Loop over poles: sign is the same for all poles at this tau
            for j in 0..n_poles {
                let pole = self.poles[j];
                let y = pole / self.wmax;

                // Compute kernel value
                let kernel_val = self.kernel.compute(x, y);

                // Tau basis: u_i(τ) = sign * (-K(x, y_i))
                let value = sign * (-kernel_val);
                result[[i, j]] += value;
            }
        }

        result
    }

    fn evaluate_matsubara(
        &self,
        freqs: &[crate::freq::MatsubaraFreq<S>],
    ) -> mdarray::DTensor<num_complex::Complex<f64>, 2> {
        use mdarray::DTensor;
        use num_complex::Complex;

        let n_points = freqs.len();
        let n_poles = self.poles.len();

        // Evaluate MatsubaraPoles basis functions
        DTensor::<Complex<f64>, 2>::from_fn([n_points, n_poles], |idx| {
            let freq = &freqs[idx[0]];
            let pole = self.poles[idx[1]];
            let inv_weight = self.inv_weights[idx[1]];

            // iν = i * π * (2n + ζ) / β
            let iv = freq.value_imaginary(self.beta);

            // u_i(iν) = inv_weight / (iν - pole_i)
            // Fermionic: inv_weight = 1.0
            // Bosonic: inv_weight = tanh(β·pole_i/2)
            Complex::new(inv_weight, 0.0) / (iv - Complex::new(pole, 0.0))
        })
    }

    fn evaluate_omega(&self, omega: &[f64]) -> mdarray::DTensor<f64, 2> {
        use mdarray::DTensor;

        let n_points = omega.len();
        let n_poles = self.poles.len();

        // For DLR, this could return identity or delta function
        // For now, return zeros (not well-defined)
        DTensor::<f64, 2>::from_fn([n_points, n_poles], |_idx| 0.0)
    }

    fn default_omega_sampling_points(&self) -> Vec<f64> {
        // DLR poles ARE the omega sampling points
        self.poles.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Bosonic, Fermionic};

    /// Generic test for periodicity/anti-periodicity
    fn test_periodicity_generic<S: StatisticsType>(expected_sign: f64, stat_name: &str) {
        let beta = 1.0;
        let omega = 5.0;

        // Test periodicity by comparing G(τ) with G(τ-β)
        // Since normalize_tau is restricted to [-β, β], we test:
        // For τ ∈ (0, β]: compare G(τ) with G(τ-β)
        // For fermions: G(τ) should equal -G(τ-β)
        // For bosons: G(τ) should equal G(τ-β)
        for tau in [0.1, 0.3, 0.7] {
            let g_tau = gtau_single_pole::<S>(tau, omega, beta);
            let g_tau_minus_beta = gtau_single_pole::<S>(tau - beta, omega, beta);

            // For fermions: G(τ) = -G(τ-β) → G(τ-β) = -G(τ)
            // For bosons: G(τ) = G(τ-β)
            let expected = expected_sign * g_tau;

            assert!(
                (expected - g_tau_minus_beta).abs() < 1e-14,
                "{} periodicity violated at τ={}: G(τ)={}, G(τ-β)={}, expected={}",
                stat_name,
                tau,
                g_tau,
                g_tau_minus_beta,
                expected
            );
        }
    }

    #[test]
    fn test_fermionic_antiperiodicity() {
        // Fermions: G(τ+β) = -G(τ)
        test_periodicity_generic::<Fermionic>(-1.0, "Fermionic");
    }

    #[test]
    fn test_bosonic_periodicity() {
        // Bosons: G(τ+β) = G(τ)
        test_periodicity_generic::<Bosonic>(1.0, "Bosonic");
    }

    #[test]
    fn test_generic_function_matches_specific() {
        let beta = 1.0;
        let omega = 5.0;
        let tau = 0.5;

        // Test that generic function matches specific functions
        let g_f_specific = fermionic_single_pole(tau, omega, beta);
        let g_f_generic = gtau_single_pole::<Fermionic>(tau, omega, beta);

        let g_b_specific = bosonic_single_pole(tau, omega, beta);
        let g_b_generic = gtau_single_pole::<Bosonic>(tau, omega, beta);

        assert!(
            (g_f_specific - g_f_generic).abs() < 1e-14,
            "Fermionic: specific={}, generic={}",
            g_f_specific,
            g_f_generic
        );
        assert!(
            (g_b_specific - g_b_generic).abs() < 1e-14,
            "Bosonic: specific={}, generic={}",
            g_b_specific,
            g_b_generic
        );
    }
}

#[cfg(test)]
#[path = "dlr_tests.rs"]
mod dlr_tests;
