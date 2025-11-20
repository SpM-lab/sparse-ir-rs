//! Imaginary time τ normalization utilities
//!
//! This module provides utility functions for normalizing imaginary time τ
//! with appropriate boundary conditions (periodic/anti-periodic).
//!
//! These utilities are used internally (e.g., in DLR implementation) and
//! can also be used by external C-API or FFI layers.

use crate::traits::{Statistics, StatisticsType};

/// Check if τ is outside the normal range [0, β]
///
/// Returns whether τ is in an "odd" period (outside [0, β]).
/// This determines the sign for anti-periodic (fermionic) functions.
///
/// # Arguments
/// * `tau` - Imaginary time
/// * `beta` - Inverse temperature
///
/// # Returns
/// * `true` if τ < 0 or τ > β (odd period), `false` otherwise
#[inline]
#[allow(dead_code)]
fn is_odd_period(tau: f64, beta: f64) -> bool {
    tau < 0.0 || tau > beta
}

/// Normalize τ to the range [0, β] with statistics-dependent boundary conditions
///
/// Handles boundary conditions based on statistics:
/// - Fermions: Anti-periodic G(τ + β) = -G(τ)
/// - Bosons: Periodic G(τ + β) = G(τ)
///
/// # Type Parameters
/// * `S` - Statistics type (Fermionic or Bosonic)
///
/// # Arguments
/// * `tau` - Imaginary time in range [-β, β]
/// * `beta` - Inverse temperature
///
/// # Returns
/// * `(tau_normalized, sign)` - Normalized τ ∈ [0, β] and sign factor
///
/// # Panics
/// Panics if `tau` is outside [-β, β]
///
/// # Boundary Interpretation
/// * `β` is interpreted as `β-` (left limit at β): `tau == beta` stays in normal range
/// * `-β` wraps to 0 for bosons, or to β with sign flip for fermions
/// * `-0.0` (negative zero) is treated as being in the odd period for fermions
///
/// # Special Cases
/// For Fermionic statistics:
/// * `tau = -0.0` (negative zero) → `(tau_normalized = β, sign = -1.0)`
/// * `tau = 0.0` (positive zero) → `(tau_normalized = 0.0, sign = 1.0)`
/// * `tau ∈ [-β, 0)` → wraps to [0, β] with `sign = -1.0`
///
/// For Bosonic statistics:
/// * `tau = -0.0` (negative zero) → `(tau_normalized = β, sign = 1.0)` (periodic)
/// * `tau = 0.0` (positive zero) → `(tau_normalized = 0.0, sign = 1.0)`
/// * `tau ∈ [-β, 0)` → wraps to [0, β] with `sign = 1.0`
///
/// # Examples
/// ```ignore
/// use crate::taufuncs::normalize_tau;
/// use crate::traits::Fermionic;
///
/// // Normal negative value
/// let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.3, 1.0);
/// assert!((tau_norm - 0.7).abs() < 1e-14);
/// assert_eq!(sign, -1.0);
///
/// // Negative zero
/// let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.0, 1.0);
/// assert!((tau_norm - 1.0).abs() < 1e-14);
/// assert_eq!(sign, -1.0);
/// ```
pub fn normalize_tau<S: StatisticsType>(tau: f64, beta: f64) -> (f64, f64) {
    // Normalize τ ∈ [-β, β] to [0, β]
    // Panics if tau is outside this range

    if tau < -beta || tau > beta {
        panic!(
            "tau = {} is outside allowed range [-beta = {}, beta = {}]",
            tau, -beta, beta
        );
    }

    // Special handling for negative zero
    if tau.is_sign_negative() && tau == 0.0 {
        // tau = -0.0
        return match S::STATISTICS {
            Statistics::Fermionic => (beta, -1.0), // Anti-periodic: wraps to beta with sign flip
            Statistics::Bosonic => (beta, 1.0),    // Periodic: wraps to beta with sign unchanged
        };
    }

    // If already in [0, β], return as-is with sign = 1
    if tau >= 0.0 && tau <= beta {
        return (tau, 1.0);
    }

    // tau ∈ [-β, 0): wrap to [0, β]
    // For tau ∈ (-β, 0): tau_normalized = tau + β
    // For tau = -β: wraps to 0
    let tau_normalized = tau + beta;

    // Sign depends on statistics
    let sign = match S::STATISTICS {
        Statistics::Fermionic => -1.0, // Anti-periodic: sign flips
        Statistics::Bosonic => 1.0,    // Periodic: sign stays
    };

    (tau_normalized, sign)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{Bosonic, Fermionic};

    #[test]
    fn test_is_odd_period() {
        let beta = 1.0;

        // Normal range
        assert!(!is_odd_period(0.0, beta));
        assert!(!is_odd_period(0.5, beta));
        assert!(!is_odd_period(beta, beta)); // β is β-

        // Odd periods
        assert!(is_odd_period(-0.1, beta));
        assert!(is_odd_period(1.1, beta));
    }

    #[test]
    fn test_normalize_tau_generic_fermionic() {
        let beta = 1.0;

        // Normal range
        let (tau_norm, sign) = normalize_tau::<Fermionic>(0.5, beta);
        assert!((tau_norm - 0.5).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // At β (interpreted as β-)
        let (tau_norm, sign) = normalize_tau::<Fermionic>(beta, beta);
        assert!((tau_norm - beta).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // At 0
        let (tau_norm, sign) = normalize_tau::<Fermionic>(0.0, beta);
        assert!(tau_norm.abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // Negative range
        let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.3, beta);
        assert!((tau_norm - 0.7).abs() < 1e-14);
        assert!((sign - (-1.0)).abs() < 1e-14);

        // Test -β: wraps to 0 with sign flip
        let (tau_norm, sign) = normalize_tau::<Fermionic>(-beta, beta);
        assert!(tau_norm.abs() < 1e-14); // wraps to 0
        assert!((sign - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_normalize_tau_negative_zero_fermionic() {
        // Test negative zero: -0.0 should be treated as being in odd period
        let beta = 1.0;

        // Negative zero should normalize to beta with sign = -1
        let (tau_norm, sign) = normalize_tau::<Fermionic>(-0.0, beta);
        assert!(
            (tau_norm - beta).abs() < 1e-14,
            "Expected tau_normalized = {}, got {}",
            beta,
            tau_norm
        );
        assert!(
            (sign - (-1.0)).abs() < 1e-14,
            "Expected sign = -1.0, got {}",
            sign
        );

        // Positive zero should stay at 0 with sign = 1
        let (tau_norm, sign) = normalize_tau::<Fermionic>(0.0, beta);
        assert!(
            tau_norm.abs() < 1e-14,
            "Expected tau_normalized = 0.0, got {}",
            tau_norm
        );
        assert!(
            (sign - 1.0).abs() < 1e-14,
            "Expected sign = 1.0, got {}",
            sign
        );

        // Verify that Rust distinguishes -0.0 from 0.0
        assert_ne!(
            (-0.0_f64).to_bits(),
            0.0_f64.to_bits(),
            "Rust should distinguish -0.0 from 0.0"
        );
    }

    #[test]
    fn test_normalize_tau_generic_bosonic() {
        let beta = 1.0;

        // Normal range
        let (tau_norm, sign) = normalize_tau::<Bosonic>(0.5, beta);
        assert!((tau_norm - 0.5).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // At β (interpreted as β-)
        let (tau_norm, sign) = normalize_tau::<Bosonic>(beta, beta);
        assert!((tau_norm - beta).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // At 0
        let (tau_norm, sign) = normalize_tau::<Bosonic>(0.0, beta);
        assert!(tau_norm.abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // Negative range
        let (tau_norm, sign) = normalize_tau::<Bosonic>(-0.3, beta);
        assert!((tau_norm - 0.7).abs() < 1e-14);
        assert!((sign - 1.0).abs() < 1e-14);

        // Test -β: wraps to 0, sign stays 1 (periodic)
        let (tau_norm, sign) = normalize_tau::<Bosonic>(-beta, beta);
        assert!(tau_norm.abs() < 1e-14); // wraps to 0
        assert!((sign - 1.0).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "outside allowed range")]
    fn test_normalize_tau_out_of_range_positive() {
        let beta = 1.0;
        // Should panic for tau > beta
        let _ = normalize_tau::<Fermionic>(1.5, beta);
    }

    #[test]
    #[should_panic(expected = "outside allowed range")]
    fn test_normalize_tau_out_of_range_negative() {
        let beta = 1.0;
        // Should panic for tau < -beta
        let _ = normalize_tau::<Fermionic>(-1.5, beta);
    }
}
