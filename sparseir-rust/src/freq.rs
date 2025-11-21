//! Matsubara frequency implementation for SparseIR
//!
//! This module provides Matsubara frequency types for both fermionic and
//! bosonic statistics, matching the C++ implementation in freq.hpp.

use num_complex::Complex64;
use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::fmt;
use std::ops::{Add, Neg, Sub};

use crate::traits::{Bosonic, Fermionic, Statistics, StatisticsType};

/// Matsubara frequency for a specific statistics type
///
/// This represents a Matsubara frequency ω_n = (2n + ζ)π/β where:
/// - n is the Matsubara index (integer)
/// - ζ is the statistics parameter (1 for fermionic, 0 for bosonic)
/// - β is the inverse temperature
///
/// The statistics type S is checked at compile time to ensure type safety.
#[derive(Debug, Clone, Copy)]
pub struct MatsubaraFreq<S: StatisticsType> {
    n: i64,
    _phantom: std::marker::PhantomData<S>,
}

// Type aliases for convenience
pub type FermionicFreq = MatsubaraFreq<Fermionic>;
pub type BosonicFreq = MatsubaraFreq<Bosonic>;

impl<S: StatisticsType> MatsubaraFreq<S> {
    /// Get the Matsubara index n
    pub fn n(&self) -> i64 {
        self.n
    }

    /// Create a new Matsubara frequency
    ///
    /// # Arguments
    /// * `n` - The Matsubara index
    ///
    /// # Returns
    /// * `Ok(MatsubaraFreq)` if the frequency is valid for the statistics type
    /// * `Err(String)` if the frequency is not allowed (e.g., even n for fermionic)
    ///
    /// # Examples
    /// ```
    /// use sparse_ir::freq::{FermionicFreq, BosonicFreq};
    ///
    /// let fermionic = FermionicFreq::new(1).unwrap();  // OK: odd n for fermionic
    /// let bosonic = BosonicFreq::new(0).unwrap();      // OK: even n for bosonic
    /// ```
    pub fn new(n: i64) -> Result<Self, String> {
        // Check if the frequency is allowed for this statistics type
        let allowed = match S::STATISTICS {
            Statistics::Fermionic => n % 2 != 0, // Fermionic: odd n only
            Statistics::Bosonic => n % 2 == 0,   // Bosonic: even n only
        };

        if !allowed {
            return Err(format!(
                "Frequency n={} is not allowed for {} statistics",
                n,
                S::STATISTICS.as_str()
            ));
        }

        Ok(Self {
            n,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create a Matsubara frequency without validation
    ///
    /// # Safety
    /// This function bypasses the validation in `new()`. Only use when you're
    /// certain the frequency is valid for the statistics type.
    pub unsafe fn new_unchecked(n: i64) -> Self {
        Self {
            n,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the Matsubara index
    pub fn get_n(&self) -> i64 {
        self.n
    }

    /// Compute the real frequency value: n * π / β
    ///
    /// # Arguments
    /// * `beta` - Inverse temperature
    ///
    /// # Returns
    /// The real frequency value
    pub fn value(&self, beta: f64) -> f64 {
        self.n as f64 * std::f64::consts::PI / beta
    }

    /// Compute the imaginary frequency value: i * n * π / β
    ///
    /// # Arguments
    /// * `beta` - Inverse temperature
    ///
    /// # Returns
    /// The imaginary frequency value as a complex number
    pub fn value_imaginary(&self, beta: f64) -> Complex64 {
        Complex64::new(0.0, self.value(beta))
    }

    /// Get the statistics type
    pub fn statistics(&self) -> Statistics {
        S::STATISTICS
    }

    /// Convert to i64 (for compatibility with C++ operator long long())
    pub fn into_i64(self) -> i64 {
        self.n
    }
}

// Default implementations
impl Default for FermionicFreq {
    fn default() -> Self {
        // Default fermionic frequency is n=1 (smallest positive odd frequency)
        unsafe { Self::new_unchecked(1) }
    }
}

impl Default for BosonicFreq {
    fn default() -> Self {
        // Default bosonic frequency is n=0 (zero frequency)
        unsafe { Self::new_unchecked(0) }
    }
}

// Conversion to i64
impl<S: StatisticsType> From<MatsubaraFreq<S>> for i64 {
    fn from(freq: MatsubaraFreq<S>) -> Self {
        freq.n
    }
}

// Operator overloading: Addition
impl<S: StatisticsType> Add for MatsubaraFreq<S> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        unsafe { Self::new_unchecked(self.n + other.n) }
    }
}

// Operator overloading: Subtraction
impl<S: StatisticsType> Sub for MatsubaraFreq<S> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        unsafe { Self::new_unchecked(self.n - other.n) }
    }
}

// Operator overloading: Negation
impl<S: StatisticsType> Neg for MatsubaraFreq<S> {
    type Output = Self;

    fn neg(self) -> Self {
        unsafe { Self::new_unchecked(-self.n) }
    }
}

// Comparison operators
impl<S: StatisticsType> PartialEq for MatsubaraFreq<S> {
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n
    }
}

impl<S: StatisticsType> Eq for MatsubaraFreq<S> {}

impl<S: StatisticsType> PartialOrd for MatsubaraFreq<S> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<S: StatisticsType> Ord for MatsubaraFreq<S> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.n.cmp(&other.n)
    }
}

// Hash implementation (needed for some collections)
impl<S: StatisticsType> std::hash::Hash for MatsubaraFreq<S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.n.hash(state);
    }
}

// Display implementation
impl<S: StatisticsType> fmt::Display for MatsubaraFreq<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.n {
            0 => write!(f, "0"),
            1 => write!(f, "π/β"),
            -1 => write!(f, "-π/β"),
            n => write!(f, "{}π/β", n),
        }
    }
}

// Utility functions
/// Get the sign of a Matsubara frequency
#[allow(dead_code)]
pub(crate) fn sign<S: StatisticsType>(freq: &MatsubaraFreq<S>) -> i32 {
    if freq.n > 0 {
        1
    } else if freq.n < 0 {
        -1
    } else {
        0
    }
}

/// Get the fermionic sign based on statistics type
#[allow(dead_code)]
pub(crate) fn fermionic_sign<S: StatisticsType>() -> i32 {
    match S::STATISTICS {
        Statistics::Fermionic => -1,
        Statistics::Bosonic => 1,
    }
}

/// Create a zero frequency (bosonic only)
#[allow(dead_code)]
pub(crate) fn zero() -> BosonicFreq {
    unsafe { BosonicFreq::new_unchecked(0) }
}

/// Check if a frequency is zero
#[allow(dead_code)]
pub(crate) fn is_zero<S: StatisticsType>(freq: &MatsubaraFreq<S>) -> bool {
    match S::STATISTICS {
        Statistics::Fermionic => false, // Fermionic frequencies are never zero
        Statistics::Bosonic => freq.n == 0,
    }
}

/// Compare two Matsubara frequencies of potentially different statistics types
#[allow(dead_code)]
pub(crate) fn is_less<S1: StatisticsType, S2: StatisticsType>(
    a: &MatsubaraFreq<S1>,
    b: &MatsubaraFreq<S2>,
) -> bool {
    a.get_n() < b.get_n()
}

/// Factory function to create Statistics from zeta value
#[allow(dead_code)]
pub(crate) fn create_statistics(zeta: i64) -> Result<Statistics, String> {
    match zeta {
        1 => Ok(Statistics::Fermionic),
        0 => Ok(Statistics::Bosonic),
        _ => Err(format!("Unknown statistics type: zeta={}", zeta)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fermionic_frequency_creation() {
        // Valid fermionic frequencies (odd n)
        let freq1 = FermionicFreq::new(1).unwrap();
        assert_eq!(freq1.get_n(), 1);

        let freq3 = FermionicFreq::new(3).unwrap();
        assert_eq!(freq3.get_n(), 3);

        let freq_neg1 = FermionicFreq::new(-1).unwrap();
        assert_eq!(freq_neg1.get_n(), -1);

        // Invalid fermionic frequencies (even n)
        assert!(FermionicFreq::new(0).is_err());
        assert!(FermionicFreq::new(2).is_err());
        assert!(FermionicFreq::new(-2).is_err());
    }

    #[test]
    fn test_bosonic_frequency_creation() {
        // Valid bosonic frequencies (even n)
        let freq0 = BosonicFreq::new(0).unwrap();
        assert_eq!(freq0.get_n(), 0);

        let freq2 = BosonicFreq::new(2).unwrap();
        assert_eq!(freq2.get_n(), 2);

        let freq_neg2 = BosonicFreq::new(-2).unwrap();
        assert_eq!(freq_neg2.get_n(), -2);

        // Invalid bosonic frequencies (odd n)
        assert!(BosonicFreq::new(1).is_err());
        assert!(BosonicFreq::new(3).is_err());
        assert!(BosonicFreq::new(-1).is_err());
    }

    #[test]
    fn test_frequency_values() {
        let beta = 2.0;
        let pi = std::f64::consts::PI;

        let fermionic = FermionicFreq::new(1).unwrap();
        assert!((fermionic.value(beta) - pi / 2.0).abs() < 1e-14);

        let bosonic = BosonicFreq::new(0).unwrap();
        assert_eq!(bosonic.value(beta), 0.0);

        let bosonic2 = BosonicFreq::new(2).unwrap();
        assert!((bosonic2.value(beta) - pi).abs() < 1e-14);
    }

    #[test]
    fn test_imaginary_values() {
        let beta = 2.0;
        let pi = std::f64::consts::PI;

        let fermionic = FermionicFreq::new(1).unwrap();
        let imag = fermionic.value_imaginary(beta);
        assert!((imag.re - 0.0).abs() < 1e-14);
        assert!((imag.im - pi / 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_operator_overloading() {
        let freq1 = FermionicFreq::new(1).unwrap();
        let freq3 = FermionicFreq::new(3).unwrap();

        // Addition
        let sum = freq1 + freq3;
        assert_eq!(sum.get_n(), 4);

        // Subtraction
        let diff = freq3 - freq1;
        assert_eq!(diff.get_n(), 2);

        // Negation
        let neg = -freq1;
        assert_eq!(neg.get_n(), -1);
    }

    #[test]
    fn test_comparison_operators() {
        let freq1 = FermionicFreq::new(1).unwrap();
        let freq3 = FermionicFreq::new(3).unwrap();
        let freq1_copy = FermionicFreq::new(1).unwrap();

        assert_eq!(freq1, freq1_copy);
        assert_ne!(freq1, freq3);
        assert!(freq1 < freq3);
        assert!(freq3 > freq1);
        assert!(freq1 <= freq1_copy);
        assert!(freq1 >= freq1_copy);
    }

    #[test]
    fn test_utility_functions() {
        let fermionic = FermionicFreq::new(1).unwrap();
        let bosonic = BosonicFreq::new(0).unwrap();
        let bosonic2 = BosonicFreq::new(2).unwrap();

        // Sign function
        assert_eq!(sign(&fermionic), 1);
        assert_eq!(sign(&bosonic), 0);
        assert_eq!(sign(&-fermionic), -1);

        // Fermionic sign
        assert_eq!(fermionic_sign::<Fermionic>(), -1);
        assert_eq!(fermionic_sign::<Bosonic>(), 1);

        // Zero frequency
        let zero_freq = zero();
        assert_eq!(zero_freq.get_n(), 0);

        // Is zero
        assert!(!is_zero(&fermionic));
        assert!(is_zero(&bosonic));
        assert!(!is_zero(&bosonic2));

        // Is less
        assert!(is_less(&fermionic, &bosonic2));
        assert!(!is_less(&bosonic2, &fermionic));
    }

    #[test]
    fn test_statistics_creation() {
        assert_eq!(create_statistics(1).unwrap(), Statistics::Fermionic);
        assert_eq!(create_statistics(0).unwrap(), Statistics::Bosonic);
        assert!(create_statistics(2).is_err());
    }

    #[test]
    fn test_display() {
        let freq1 = FermionicFreq::new(1).unwrap();
        let freq_neg1 = FermionicFreq::new(-1).unwrap();
        let freq0 = BosonicFreq::new(0).unwrap();
        let freq2 = BosonicFreq::new(2).unwrap();

        assert_eq!(format!("{}", freq1), "π/β");
        assert_eq!(format!("{}", freq_neg1), "-π/β");
        assert_eq!(format!("{}", freq0), "0");
        assert_eq!(format!("{}", freq2), "2π/β");
    }

    #[test]
    fn test_default_implementations() {
        let default_fermionic = FermionicFreq::default();
        assert_eq!(default_fermionic.get_n(), 1);

        let default_bosonic = BosonicFreq::default();
        assert_eq!(default_bosonic.get_n(), 0);
    }

    #[test]
    fn test_conversion_to_i64() {
        let freq = FermionicFreq::new(3).unwrap();
        let n: i64 = freq.into();
        assert_eq!(n, 3);

        let n_direct = freq.into_i64();
        assert_eq!(n_direct, 3);
    }

    #[test]
    fn test_sign() {
        let fermionic_pos = FermionicFreq::new(3).unwrap();
        assert_eq!(sign(&fermionic_pos), 1);

        let fermionic_neg = FermionicFreq::new(-5).unwrap();
        assert_eq!(sign(&fermionic_neg), -1);

        let bosonic_zero = BosonicFreq::new(0).unwrap();
        assert_eq!(sign(&bosonic_zero), 0);

        let bosonic_pos = BosonicFreq::new(4).unwrap();
        assert_eq!(sign(&bosonic_pos), 1);
    }

    #[test]
    fn test_fermionic_sign() {
        assert_eq!(fermionic_sign::<Fermionic>(), -1);
        assert_eq!(fermionic_sign::<Bosonic>(), 1);
    }

    #[test]
    fn test_zero() {
        let zero_freq = zero();
        assert_eq!(zero_freq.get_n(), 0);
        assert_eq!(zero_freq.value(1.0), 0.0);
    }

    #[test]
    fn test_is_zero() {
        // Fermionic frequencies are never zero
        let fermionic = FermionicFreq::new(1).unwrap();
        assert!(!is_zero(&fermionic));

        // Bosonic zero frequency
        let bosonic_zero = BosonicFreq::new(0).unwrap();
        assert!(is_zero(&bosonic_zero));

        // Non-zero bosonic frequency
        let bosonic_nonzero = BosonicFreq::new(2).unwrap();
        assert!(!is_zero(&bosonic_nonzero));
    }

    #[test]
    fn test_is_less() {
        let freq1 = FermionicFreq::new(1).unwrap();
        let freq3 = FermionicFreq::new(3).unwrap();
        assert!(is_less(&freq1, &freq3));
        assert!(!is_less(&freq3, &freq1));

        // Compare different statistics types
        let fermionic = FermionicFreq::new(1).unwrap();
        let bosonic = BosonicFreq::new(2).unwrap();
        assert!(is_less(&fermionic, &bosonic));

        // Same frequency
        assert!(!is_less(&freq1, &freq1));
    }

    #[test]
    fn test_create_statistics() {
        // Fermionic (zeta = 1)
        let fermionic = create_statistics(1).unwrap();
        assert_eq!(fermionic, Statistics::Fermionic);

        // Bosonic (zeta = 0)
        let bosonic = create_statistics(0).unwrap();
        assert_eq!(bosonic, Statistics::Bosonic);

        // Invalid zeta
        assert!(create_statistics(2).is_err());
        assert!(create_statistics(-1).is_err());
    }
}
