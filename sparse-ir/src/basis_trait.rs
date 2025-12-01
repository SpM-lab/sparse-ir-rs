//! Basis trait for IR and DLR representations
//!
//! This module provides a common trait for different basis representations
//! (IR basis, DLR basis, augmented basis, etc.) in imaginary-time/frequency domains.

use crate::freq::MatsubaraFreq;
use crate::kernel::KernelProperties;
use crate::traits::StatisticsType;

/// Common trait for basis representations in imaginary-time/frequency domains
///
/// This trait abstracts over different basis representations:
/// - `FiniteTempBasis`: IR (Intermediate Representation) basis
/// - `DiscreteLehmannRepresentation`: DLR basis
/// - `AugmentedBasis`: IR basis with additional functions
///
/// Each basis provides:
/// - Physical parameters (β, ωmax, Λ)
/// - Basis size and accuracy information
/// - Default sampling points for τ and Matsubara frequencies
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
pub trait Basis<S: StatisticsType> {
    /// Associated kernel type
    type Kernel: KernelProperties;

    /// Get reference to the kernel
    ///
    /// # Returns
    /// Reference to the kernel used to construct this basis
    fn kernel(&self) -> &Self::Kernel;

    /// Inverse temperature β
    ///
    /// # Returns
    /// The inverse temperature in units where ℏ = kB = 1
    fn beta(&self) -> f64;

    /// Maximum frequency ωmax
    ///
    /// The basis functions are designed to accurately represent
    /// spectral functions with support in [-ωmax, ωmax].
    ///
    /// # Returns
    /// The maximum frequency cutoff
    fn wmax(&self) -> f64;

    /// Kernel parameter Λ = β × ωmax
    ///
    /// This is the dimensionless parameter that controls the basis.
    ///
    /// # Returns
    /// The dimensionless parameter Λ
    fn lambda(&self) -> f64 {
        self.beta() * self.wmax()
    }

    /// Number of basis functions
    ///
    /// # Returns
    /// The size of the basis (number of basis functions)
    fn size(&self) -> usize;

    /// Accuracy of the basis
    ///
    /// Upper bound to the relative error of representing a propagator
    /// with the given number of basis functions.
    ///
    /// # Returns
    /// A number between 0 and 1 representing the accuracy
    fn accuracy(&self) -> f64;

    /// Significance of each basis function
    ///
    /// Returns a vector where `σ[i]` (0 ≤ σ[i] ≤ 1) is the significance
    /// level of the i-th basis function. If ε is the desired accuracy,
    /// then any basis function where σ[i] < ε can be neglected.
    ///
    /// For the IR basis: σ[i] = s[i] / s[0]
    /// For the DLR basis: σ[i] = 1.0 (all poles equally significant)
    ///
    /// # Returns
    /// Vector of significance values for each basis function
    fn significance(&self) -> Vec<f64>;

    /// Get singular values (non-normalized)
    ///
    /// Returns the singular values s[i] from the SVE decomposition.
    /// These are the absolute values, not normalized by s[0].
    ///
    /// # Returns
    /// Vector of singular values
    fn svals(&self) -> Vec<f64>;

    /// Get default tau sampling points
    ///
    /// Returns sampling points in imaginary time τ ∈ [-β/2, β/2].
    /// These are chosen to provide near-optimal conditioning of the
    /// sampling matrix.
    ///
    /// # Returns
    /// Vector of tau sampling points
    fn default_tau_sampling_points(&self) -> Vec<f64>;

    /// Get default Matsubara sampling points
    ///
    /// Returns sampling points in Matsubara frequency space.
    /// These are chosen to provide near-optimal conditioning.
    ///
    /// # Arguments
    /// * `positive_only` - If true, only return non-negative frequencies
    ///
    /// # Returns
    /// Vector of Matsubara frequency sampling points
    fn default_matsubara_sampling_points(&self, positive_only: bool) -> Vec<MatsubaraFreq<S>>
    where
        S: 'static;

    /// Evaluate basis functions at imaginary time points
    ///
    /// Computes the value of basis functions at given τ points.
    /// For IR basis: u_l(τ)
    /// For DLR basis: sum over poles weighted by basis coefficients
    ///
    /// # Arguments
    /// * `tau` - Imaginary time points where τ ∈ [0, β] (or extended range for DLR)
    ///
    /// # Returns
    /// Matrix of shape [tau.len(), self.size()] where result[i, l] = u_l(τ_i)
    fn evaluate_tau(&self, tau: &[f64]) -> mdarray::DTensor<f64, 2>;

    /// Evaluate basis functions at Matsubara frequencies
    ///
    /// Computes the value of basis functions at given Matsubara frequencies.
    /// For IR basis: û_l(iωn)
    /// For DLR basis: basis functions in Matsubara space
    ///
    /// # Arguments
    /// * `freqs` - Matsubara frequencies
    ///
    /// # Returns
    /// Matrix of shape [freqs.len(), self.size()] where result[i, l] = û_l(iωn_i)
    fn evaluate_matsubara(
        &self,
        freqs: &[MatsubaraFreq<S>],
    ) -> mdarray::DTensor<num_complex::Complex<f64>, 2>
    where
        S: 'static;

    /// Evaluate spectral basis functions at real frequencies
    ///
    /// Computes the value of spectral basis functions at given real frequencies.
    /// For IR basis: V_l(ω)
    /// For DLR basis: may return identity or specific representation
    ///
    /// # Arguments
    /// * `omega` - Real frequency points in [-ωmax, ωmax]
    ///
    /// # Returns
    /// Matrix of shape [omega.len(), self.size()] where result[i, l] = V_l(ω_i)
    fn evaluate_omega(&self, omega: &[f64]) -> mdarray::DTensor<f64, 2>;

    /// Get default omega (real frequency) sampling points
    ///
    /// Returns sampling points on the real-frequency axis ω ∈ [-ωmax, ωmax].
    /// These are used as pole locations for the Discrete Lehmann Representation (DLR).
    ///
    /// The sampling points are chosen as the roots/extrema of the L-th basis function
    /// in the spectral domain, providing near-optimal conditioning.
    ///
    /// # Returns
    /// Vector of real-frequency sampling points in [-ωmax, ωmax]
    fn default_omega_sampling_points(&self) -> Vec<f64>;
}

#[cfg(test)]
#[path = "basis_trait_tests.rs"]
mod tests;
