//! Type definitions for SVE computation

use simba::scalar::ComplexField;

/// Working precision type for SVE computations
///
/// Values match the C-API constants defined in sparseir.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TworkType {
    /// Use double precision (64-bit)
    Float64 = 0, // SPIR_TWORK_FLOAT64
    /// Use extended precision (128-bit double-double)
    Float64X2 = 1, // SPIR_TWORK_FLOAT64X2
    /// Automatically choose precision based on epsilon
    Auto = -1, // SPIR_TWORK_AUTO
}

/// SVD computation strategy
///
/// Values match the C-API constants defined in sparseir.h
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVDStrategy {
    /// Fast computation
    Fast = 0, // SPIR_SVDSTRAT_FAST
    /// Accurate computation
    Accurate = 1, // SPIR_SVDSTRAT_ACCURATE
    /// Automatically choose strategy
    Auto = -1, // SPIR_SVDSTRAT_AUTO
}

/// Determine safe epsilon and working precision
///
/// This function determines the safe epsilon value based on the working precision,
/// and automatically selects the working precision if TworkType::Auto is specified.
///
/// # Arguments
///
/// * `epsilon` - Required accuracy (must be non-negative)
/// * `twork` - Working precision type (Auto for automatic selection)
/// * `svd_strategy` - SVD computation strategy (Auto for automatic selection)
///
/// # Returns
///
/// Tuple of (safe_epsilon, actual_twork, actual_svd_strategy)
///
/// # Panics
///
/// Panics if epsilon is negative
pub fn safe_epsilon(
    epsilon: f64,
    twork: TworkType,
    svd_strategy: SVDStrategy,
) -> (f64, TworkType, SVDStrategy) {
    // Check for negative epsilon (following C++ implementation)
    if epsilon < 0.0 {
        panic!("eps_required must be non-negative");
    }

    // First, choose the working dtype based on the eps required
    let twork_actual = match twork {
        TworkType::Auto => {
            if epsilon.is_nan() || epsilon < 1e-8 {
                TworkType::Float64X2 // MAX_DTYPE equivalent
            } else {
                TworkType::Float64
            }
        }
        other => other,
    };

    // Next, work out the actual epsilon
    let safe_eps = match twork_actual {
        TworkType::Float64 => {
            // This is technically a bit too low (the true value is about 1.5e-8),
            // but it's not too far off and easier to remember for the user.
            1e-8
        }
        TworkType::Float64X2 => {
            // sqrt(Df64 epsilon) ≈ sqrt(2.465e-32) ≈ 1.57e-16
            use crate::numeric::CustomNumeric;
            crate::Df64::epsilon().sqrt().to_f64()
        }
        _ => 1e-8,
    };

    // Work out the SVD strategy to be used
    let svd_strategy_actual = match svd_strategy {
        SVDStrategy::Auto => {
            if !epsilon.is_nan() && epsilon < safe_eps {
                // TODO: Add warning output like C++
                SVDStrategy::Accurate
            } else {
                SVDStrategy::Fast
            }
        }
        other => other,
    };

    (safe_eps, twork_actual, svd_strategy_actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_epsilon_auto_float64() {
        let (safe_eps, twork, _) = safe_epsilon(1e-7, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(twork, TworkType::Float64);
        assert_eq!(safe_eps, 1e-8);
    }

    #[test]
    fn test_safe_epsilon_auto_float64x2() {
        let (safe_eps, twork, _) = safe_epsilon(1e-10, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(twork, TworkType::Float64X2);
        // sqrt(Df64 epsilon) ≈ 1.57e-16
        assert!((safe_eps - 1.5700924586837752e-16).abs() < 1e-20);
    }

    #[test]
    fn test_safe_epsilon_explicit_precision() {
        let (safe_eps, twork, _) = safe_epsilon(1e-7, TworkType::Float64X2, SVDStrategy::Auto);
        assert_eq!(twork, TworkType::Float64X2);
        // sqrt(Df64 epsilon) ≈ 1.57e-16
        assert!((safe_eps - 1.5700924586837752e-16).abs() < 1e-20);
    }

    #[test]
    fn test_svd_strategy_auto_accurate() {
        // epsilon = 1e-20 < 1.57e-16 (safe_eps for Float64X2) → Accurate
        let (_, _, strategy) = safe_epsilon(1e-20, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(strategy, SVDStrategy::Accurate);
    }

    #[test]
    fn test_svd_strategy_auto_fast() {
        let (_, _, strategy) = safe_epsilon(1e-7, TworkType::Auto, SVDStrategy::Auto);
        assert_eq!(strategy, SVDStrategy::Fast);
    }

    #[test]
    #[should_panic(expected = "eps_required must be non-negative")]
    fn test_negative_epsilon_panics() {
        safe_epsilon(-1.0, TworkType::Auto, SVDStrategy::Auto);
    }
}
