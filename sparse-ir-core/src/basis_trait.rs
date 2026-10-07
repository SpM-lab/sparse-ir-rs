//! Basis trait for IR and DLR representations
//!
//! This module provides a common trait for different basis representations
//! (IR basis, DLR basis, augmented basis, etc.) in imaginary-time/frequency domains.

use crate::error::Error;
use crate::freq::MatsubaraFreq;
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
    /// Returns the singular values S_l of the basis in physical units; for
    /// `FiniteTempBasis`, S_l = sqrt(β ωmax/2) ωmax^ypower s_l with s_l those of
    /// the SVE.
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
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] if the basis has no default tau sampling points,
    /// e.g. an IR basis whose SVE has too few singular functions (see
    /// `FiniteTempBasis::default_tau_sampling_points`)
    fn default_tau_sampling_points(&self) -> Result<Vec<f64>, Error>;

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
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] if the basis is a DLR (use the points of its IR
    /// basis), or its basis functions have no definite parity (an SVE that is
    /// not centrosymmetric, #183)
    fn default_matsubara_sampling_points(
        &self,
        positive_only: bool,
    ) -> Result<Vec<MatsubaraFreq<S>>, Error>
    where
        S: 'static;

    /// Evaluate basis functions at imaginary time points
    ///
    /// Computes the value of basis functions at given τ points.
    /// For IR basis: u_l(τ)
    /// For DLR basis: the pole functions u_p(τ), one column per pole
    ///
    /// # Arguments
    /// * `tau` - Imaginary time points τ ∈ [-β, β]; negative τ uses the
    ///   (anti)periodicity of the statistics
    ///
    /// # Returns
    /// Matrix of shape [tau.len(), self.size()] where result[i, l] = u_l(τ_i)
    ///
    /// An empty `tau` gives a `[0, size]` matrix.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfDomain`] if a τ is outside [-β, β] or NaN; no value is
    /// computed then.
    fn evaluate_tau(&self, tau: &[f64]) -> Result<crate::Matrix<f64>, Error>;

    /// Evaluate basis functions at Matsubara frequencies
    ///
    /// Computes the value of basis functions at given Matsubara frequencies.
    /// For IR basis: û_l(iν)
    /// For DLR basis: basis functions in Matsubara space
    ///
    /// # Arguments
    /// * `freqs` - Matsubara frequencies
    ///
    /// # Returns
    /// Matrix of shape [freqs.len(), self.size()] where result[i, l] = û_l(iν_i)
    ///
    /// An empty `freqs` gives a `[0, size]` matrix.
    ///
    /// # Errors
    ///
    /// Implementors may return errors. The bases of this crate
    /// (`FiniteTempBasis` and `DiscreteLehmannRepresentation`)
    /// never do: every `MatsubaraFreq<S>` has the parity of the statistics, so
    /// every frequency can be evaluated.
    fn evaluate_matsubara(
        &self,
        freqs: &[MatsubaraFreq<S>],
    ) -> Result<crate::Matrix<num_complex::Complex<f64>>, Error>
    where
        S: 'static;

    /// Evaluate spectral basis functions at real frequencies
    ///
    /// Computes the value of spectral basis functions at given real frequencies.
    /// For IR basis: V_l(ω)
    /// Not supported for the DLR basis (see Errors)
    ///
    /// # Arguments
    /// * `omega` - Real frequency points in [-ωmax, ωmax]
    ///
    /// # Returns
    /// Matrix of shape [omega.len(), self.size()] where result[i, l] = V_l(ω_i)
    ///
    /// An empty `omega` gives a `[0, size]` matrix.
    ///
    /// # Errors
    ///
    /// * [`Error::OutOfDomain`] if an ω is outside [-ωmax, ωmax] or NaN
    /// * [`Error::NotSupported`] for the DLR basis
    fn evaluate_omega(&self, omega: &[f64]) -> Result<crate::Matrix<f64>, Error>;

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
    ///
    /// # Errors
    ///
    /// [`Error::NotSupported`] if the basis is an IR basis whose SVE has too
    /// few singular functions (the DLR returns its poles)
    fn default_omega_sampling_points(&self) -> Result<Vec<f64>, Error>;
}
