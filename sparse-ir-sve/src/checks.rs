//! Parameter checks shared by the `sparse-ir` crates

use crate::error::Error;

/// `Ok` if `value` is positive and finite
pub fn require_positive_finite(name: &'static str, value: f64) -> Result<(), Error> {
    if value > 0.0 && value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value: format!("{value:?}"),
            reason: "must be positive and finite".to_string(),
        })
    }
}

/// `Ok` if `value` is finite
pub fn require_finite(name: &'static str, value: f64) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::InvalidParameter {
            name,
            value: format!("{value:?}"),
            reason: "must be finite".to_string(),
        })
    }
}

/// `Ok` for an accuracy that selects the SVE: `None` (automatic) or a value
/// in (0, 1)
pub fn require_accuracy(name: &'static str, epsilon: Option<f64>) -> Result<(), Error> {
    match epsilon {
        Some(eps) if !(eps > 0.0 && eps < 1.0) => Err(Error::InvalidParameter {
            name,
            value: format!("{eps:?}"),
            reason: "must be in (0, 1)".to_string(),
        }),
        _ => Ok(()),
    }
}

/// `Ok` for a relative truncation threshold: `None` or a value in [0, 1),
/// where 0 keeps every singular value
pub fn require_threshold(name: &'static str, epsilon: Option<f64>) -> Result<(), Error> {
    match epsilon {
        Some(eps) if !(eps >= 0.0 && eps < 1.0) => Err(Error::InvalidParameter {
            name,
            value: format!("{eps:?}"),
            reason: "must be in [0, 1)".to_string(),
        }),
        _ => Ok(()),
    }
}

/// `Ok` unless `size` is `Some(0)`
pub fn require_nonzero_size(name: &'static str, size: Option<usize>) -> Result<(), Error> {
    match size {
        Some(0) => Err(Error::InvalidParameter {
            name,
            value: "0".to_string(),
            reason: "must be positive".to_string(),
        }),
        _ => Ok(()),
    }
}
