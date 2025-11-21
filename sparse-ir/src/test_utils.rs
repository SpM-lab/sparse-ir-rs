//! Common test utilities

use mdarray::{DynRank, Tensor};
use num_complex::Complex;

/// Move axis from position src to position dst
///
/// Equivalent to numpy.moveaxis or libsparseir's movedim.
pub fn movedim<T: Clone>(arr: &Tensor<T, DynRank>, src: usize, dst: usize) -> Tensor<T, DynRank> {
    let rank = arr.rank();
    assert!(
        src < rank && dst < rank,
        "src={}, dst={} must be < rank={}",
        src,
        dst,
        rank
    );

    if src == dst {
        return arr.clone();
    }

    // Create permutation: move src to dst
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(src);
    perm.insert(dst, src);

    arr.permute(&perm[..]).to_tensor()
}

/// Simple deterministic pseudo-random number generator (LCG)
///
/// Linear Congruential Generator for reproducible random numbers in tests.
/// Uses the common parameters: a = 1664525, c = 1013904223 (from Numerical Recipes)
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new RNG with the given seed
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Generate next f64 in range [0, 1) (base method)
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        // Map to [0, 1) using upper 48 bits
        ((self.state >> 16) as f64) / ((1u64 << 48) as f64)
    }

    /// Generate next value of type T (generic)
    pub fn next<T: RandomGenerate>(&mut self) -> T {
        T::generate(self)
    }
}

/// Trait for types that can be randomly generated from SimpleRng
pub trait RandomGenerate {
    fn generate(rng: &mut SimpleRng) -> Self;
}

impl RandomGenerate for f64 {
    fn generate(rng: &mut SimpleRng) -> Self {
        rng.next_f64()
    }
}

impl RandomGenerate for Complex<f64> {
    fn generate(rng: &mut SimpleRng) -> Self {
        // Generate independent real and imaginary parts
        let re = rng.next_f64();
        let im = rng.next_f64();
        Complex::new(re, im)
    }
}

/// Trait for computing error magnitude (real value)
pub trait ErrorNorm {
    /// Compute the error magnitude as f64
    fn error_norm(self) -> f64;
}

/// Trait for converting f64 to T (identity for f64, promote for Complex<f64>)
pub trait ConvertFromReal {
    fn from_real(value: f64) -> Self;
}

impl ConvertFromReal for f64 {
    fn from_real(value: f64) -> Self {
        value
    }
}

impl ConvertFromReal for Complex<f64> {
    fn from_real(value: f64) -> Self {
        Complex::new(value, 0.0)
    }
}

impl ErrorNorm for f64 {
    fn error_norm(self) -> f64 {
        self.abs()
    }
}

impl ErrorNorm for Complex<f64> {
    fn error_norm(self) -> f64 {
        self.norm() // sqrt(re^2 + im^2)
    }
}

/// Generate test data for both τ and Matsubara sampling (1D vectors)
///
/// Creates random coefficients and computes corresponding Green's function values
/// at both τ sampling points and Matsubara frequencies using a single-pole model.
///
/// # Type Parameters
/// * `T` - Element type for coefficients (f64 or Complex<f64>)
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `basis` - Finite temperature basis
/// * `tau_points` - τ sampling points
/// * `matsubara_freqs` - Matsubara frequency sampling points
/// * `seed` - Random seed for reproducible test data
///
/// # Returns
/// Tuple of (coefficients, τ values, Matsubara values)
pub fn generate_test_data_tau_and_matsubara<T, S, K>(
    basis: &crate::FiniteTempBasis<K, S>,
    tau_points: &[f64],
    matsubara_freqs: &[crate::freq::MatsubaraFreq<S>],
    seed: u64,
) -> (Vec<T>, Vec<T>, Vec<Complex<f64>>)
where
    T: RandomGenerate + ConvertFromReal + Clone + std::ops::Mul<f64, Output = T>,
    S: crate::traits::StatisticsType,
    K: crate::kernel::KernelProperties + crate::kernel::CentrosymmKernel + Clone + 'static,
{
    use crate::giwn_single_pole;

    let mut rng = SimpleRng::new(seed);
    let basis_size = basis.size();
    let beta = basis.beta;
    let wmax = basis.kernel.lambda() / beta;

    // Choose a pole position (arbitrary choice within reasonable range)
    let omega = wmax * 0.5; // Pole at half of wmax

    // Generate random coefficients scaled by singular values
    let coeffs: Vec<T> = (0..basis_size)
        .map(|l| rng.next::<T>() * basis.s[l])
        .collect();

    // Compute G(τ) at τ sampling points
    let gtau_values: Vec<T> = tau_points
        .iter()
        .map(|&tau| {
            let g = crate::gtau_single_pole::<S>(tau, omega, beta);
            T::from_real(g)
        })
        .collect();

    // Compute G(iωn) at Matsubara frequencies
    let giwn_values: Vec<Complex<f64>> = matsubara_freqs
        .iter()
        .map(|freq| giwn_single_pole::<S>(freq, omega, beta))
        .collect();

    (coeffs, gtau_values, giwn_values)
}

/// Generate N-dimensional test data for τ and/or Matsubara sampling using single-pole model
///
/// Creates N-dimensional tensors of coefficients and Green's function values at sampling points
/// using a single-pole model with random pole position per extra dimension.
///
/// # Type Parameters
/// * `T` - Element type for coefficients (f64 or Complex<f64>)
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `basis` - Finite temperature basis
/// * `tau_points` - τ sampling points (optional, pass empty slice if not needed)
/// * `matsubara_freqs` - Matsubara frequency sampling points (optional, pass empty slice if not needed)
/// * `seed` - Random seed for reproducible test data
/// * `extra_dims` - Extra dimensions beyond the sampling dimension
///
/// # Returns
/// Tuple of (coefficients tensor, τ values tensor, Matsubara values tensor)
/// - Coefficients: shape [basis_size, ...extra_dims]
/// - τ values: shape [n_tau, ...extra_dims]
/// - Matsubara values: shape [n_matsubara, ...extra_dims]
pub fn generate_nd_test_data<T, S, K>(
    basis: &crate::FiniteTempBasis<K, S>,
    tau_points: &[f64],
    matsubara_freqs: &[crate::freq::MatsubaraFreq<S>],
    seed: u64,
    extra_dims: &[usize],
) -> (
    Tensor<T, DynRank>,
    Tensor<T, DynRank>,
    Tensor<num_complex::Complex<f64>, DynRank>,
)
where
    T: RandomGenerate
        + ConvertFromReal
        + Clone
        + std::ops::Mul<f64, Output = T>
        + Default
        + From<f64>
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>,
    S: crate::traits::StatisticsType,
    K: crate::kernel::KernelProperties + crate::kernel::CentrosymmKernel + Clone + 'static,
{
    use crate::{giwn_single_pole, gtau_single_pole};

    let mut rng = SimpleRng::new(seed);
    let beta = basis.beta;
    let wmax = basis.kernel.lambda() / beta;
    let basis_size = basis.size();

    // Calculate total number of extra elements
    let total_extra: usize = extra_dims.iter().product();
    let total_extra = if total_extra == 0 { 1 } else { total_extra };

    // Create tensors
    let mut coeffs_shape = vec![basis_size];
    coeffs_shape.extend_from_slice(extra_dims);
    let mut coeffs: Tensor<T, DynRank> = Tensor::zeros(&coeffs_shape[..]);

    let mut gtau_shape = vec![tau_points.len()];
    gtau_shape.extend_from_slice(extra_dims);
    let mut gtau_values: Tensor<T, DynRank> = Tensor::zeros(&gtau_shape[..]);

    let mut giwn_shape = vec![matsubara_freqs.len()];
    giwn_shape.extend_from_slice(extra_dims);
    let mut giwn_values: Tensor<Complex<f64>, DynRank> = Tensor::zeros(&giwn_shape[..]);

    // Generate data for each extra index
    for flat_idx in 0..total_extra {
        // Convert flat index to multi-index
        let mut extra_idx = Vec::new();
        let mut remainder = flat_idx;
        for &dim_size in extra_dims.iter().rev() {
            extra_idx.push(remainder % dim_size);
            remainder /= dim_size;
        }
        extra_idx.reverse();

        // Generate random coefficients for this slice
        for l in 0..basis_size {
            let random_base: T = rng.next();
            let random_centered = random_base * T::from(2.0) - T::from(1.0);
            let scaled_coeff = random_centered * basis.s[l];

            let mut full_idx = vec![l];
            full_idx.extend_from_slice(&extra_idx);
            coeffs[&full_idx[..]] = scaled_coeff;
        }

        // Random pole position for this slice
        let omega = wmax * (0.3 + rng.next_f64() * 0.4); // Between 0.3*wmax and 0.7*wmax

        // Compute G(τ) values
        for (i, &tau) in tau_points.iter().enumerate() {
            let g = gtau_single_pole::<S>(tau, omega, beta);
            let mut full_idx = vec![i];
            full_idx.extend_from_slice(&extra_idx);
            gtau_values[&full_idx[..]] = T::from_real(g);
        }

        // Compute G(iωn) values
        for (i, freq) in matsubara_freqs.iter().enumerate() {
            let g = giwn_single_pole::<S>(freq, omega, beta);
            let mut full_idx = vec![i];
            full_idx.extend_from_slice(&extra_idx);
            giwn_values[&full_idx[..]] = g;
        }
    }

    (coeffs, gtau_values, giwn_values)
}
