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

/// Determine safe epsilon and working precision
///
/// This function determines the safe epsilon value based on the working precision,
/// and automatically selects the working precision if TworkType::Auto is specified.
///
/// # Arguments
///
/// * `epsilon` - Required accuracy (non-negative); `None` selects the best
///   accuracy of the working precision
/// * `twork` - Working precision type (Auto for automatic selection)
///
/// # Returns
///
/// Tuple of (safe_epsilon, actual_twork)
///
/// # Panics
///
/// Panics if epsilon is negative or NaN. [`compute_sve`](crate::sve::compute_sve)
/// checks it first.
pub(crate) fn safe_epsilon(epsilon: Option<f64>, twork: TworkType) -> (f64, TworkType) {
    // Check for a negative or NaN epsilon (following the C++ implementation
    // for negative values; NaN used to select the automatic accuracy)
    if let Some(eps) = epsilon.filter(|eps| !(*eps >= 0.0)) {
        panic!("eps_required must be non-negative, got {eps:?}");
    }

    // First, choose the working dtype based on the eps required
    let twork_actual = match twork {
        TworkType::Auto => match epsilon {
            Some(eps) if eps >= 1e-8 => TworkType::Float64,
            _ => TworkType::Float64X2, // MAX_DTYPE equivalent
        },
        other => other,
    };

    // Next, work out the actual epsilon.
    // The precision floor is the smallest epsilon achievable with the chosen
    // working type.  The returned epsilon is the *larger* of the user's
    // request and this floor so that (a) we never promise more accuracy than
    // the arithmetic can deliver and (b) a user who asks for *less* accuracy
    // actually gets what they asked for.
    let precision_floor = match twork_actual {
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
    let safe_eps = epsilon.map_or(precision_floor, |eps| eps.max(precision_floor));

    (safe_eps, twork_actual)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_epsilon_auto_float64() {
        // epsilon=1e-7 > floor=1e-8 → safe_eps should honour the user's request
        let (safe_eps, twork) = safe_epsilon(Some(1e-7), TworkType::Auto);
        assert_eq!(twork, TworkType::Float64);
        assert_eq!(safe_eps, 1e-7);
    }

    #[test]
    fn test_safe_epsilon_auto_float64x2() {
        // epsilon=1e-10 > floor≈1.57e-16 → safe_eps should be the user's epsilon
        let (safe_eps, twork) = safe_epsilon(Some(1e-10), TworkType::Auto);
        assert_eq!(twork, TworkType::Float64X2);
        assert_eq!(safe_eps, 1e-10);
    }

    #[test]
    fn test_safe_epsilon_explicit_precision() {
        // epsilon=1e-7 > floor≈1.57e-16 → safe_eps should honour the user's epsilon
        let (safe_eps, twork) = safe_epsilon(Some(1e-7), TworkType::Float64X2);
        assert_eq!(twork, TworkType::Float64X2);
        assert_eq!(safe_eps, 1e-7);
    }

    #[test]
    #[should_panic(expected = "eps_required must be non-negative")]
    fn test_negative_epsilon_panics() {
        safe_epsilon(Some(-1.0), TworkType::Auto);
    }

    /// NaN is not an accuracy: it used to select the automatic accuracy.
    /// compute_sve rejects it first; the public helper panics like it does
    /// for a negative epsilon.
    #[test]
    #[should_panic(expected = "eps_required must be non-negative, got NaN")]
    fn test_nan_epsilon_panics() {
        safe_epsilon(Some(f64::NAN), TworkType::Auto);
    }
}
