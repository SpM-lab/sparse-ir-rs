//! Error type of the public API
//!
//! The public functions of this crate that return a `Result` use [`Error`]
//! as the error type. Each variant carries the values that caused it, so
//! that its message locates the problem. [`Error::kind`] sorts the variants
//! into the categories that the C API reports as status codes.

use crate::gemm::GemmError;
use crate::traits::Statistics;

/// Which array of an operation has the wrong shape, in
/// [`Error::ShapeMismatch`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArrayRole {
    /// An array that the operation reads, e.g. the coefficients of an
    /// evaluation or a given sampling matrix
    Input,
    /// The array that the operation writes to, e.g. `out` of `evaluate_nd_to`
    Output,
}

impl std::fmt::Display for ArrayRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ArrayRole::Input => "input",
            ArrayRole::Output => "output",
        })
    }
}

/// Error returned by the fallible public functions of this crate
///
/// More variants may be added without a major version bump. To handle a category of
/// errors rather than one variant, match on [`Error::kind`].
///
/// Errors compare equal when all their fields do. An error that holds a NaN
/// value is therefore not equal to itself; match on the variant instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A parameter is outside its valid range.
    #[error("invalid {name} = {value}: {reason}")]
    InvalidParameter {
        /// Name of the parameter, as in the documentation of the function
        name: &'static str,
        /// The rejected value, formatted for the message
        value: String,
        /// The condition that the value violates, e.g. "must be in (0, 1)"
        reason: String,
    },
    /// A point lies outside the domain of a function, e.g. τ outside [-β, β]
    /// or x outside the knots of a polynomial. NaN lies outside every domain.
    #[error("{name} = {value:?} is outside the domain [{:?}, {:?}]", .domain.0, .domain.1)]
    OutOfDomain {
        /// Name of the argument, as in the documentation of the function
        name: &'static str,
        /// The rejected point
        value: f64,
        /// The closed interval `(lower, upper)` of valid points
        domain: (f64, f64),
    },
    /// A Matsubara frequency index is not allowed: `n` must be odd for
    /// fermionic and even for bosonic statistics, and non-negative for a
    /// positive-only sampling. The message says which: a wrong parity, or a
    /// negative `n` of the right parity.
    #[error("{}", invalid_matsubara_index_message(.n, .statistics))]
    InvalidMatsubaraIndex {
        /// The rejected index
        n: i64,
        /// The statistics that `n` was checked against
        statistics: Statistics,
    },
    /// An input that must not be empty is empty.
    #[error("{name} must not be empty")]
    EmptyInput {
        /// Name of the input, as in the documentation of the function
        name: &'static str,
    },
    /// An input contains NaN or an infinity.
    #[error("{name} has the non-finite entry {value} at index {index:?}")]
    NonFiniteInput {
        /// Name of the input, as in the documentation of the function
        name: &'static str,
        /// Index of the first non-finite entry (`[row, column]` for a matrix)
        index: Vec<usize>,
        /// That entry, converted to `f64`
        value: f64,
    },
    /// The kernel does not support the requested statistics, e.g.
    /// `RegularizedBoseKernel` with fermionic statistics.
    #[error(
        "kernel does not support the requested statistics: kernels with ypower = 1 \
         (e.g. RegularizedBoseKernel) require bosonic statistics"
    )]
    KernelStatisticsMismatch,
    /// The basis has fewer default poles than functions, so its default poles
    /// cannot define a DLR. This can happen with certain kernels (e.g.
    /// `RegularizedBoseKernel`) because of the limited precision of the root
    /// finding.
    #[error("number of default poles ({n_poles}) is less than the basis size ({basis_size})")]
    InsufficientDefaultPoles {
        /// Basis size
        basis_size: usize,
        /// Number of default poles found
        n_poles: usize,
    },
    /// An axis argument is not an axis of the array, e.g. `dim` of
    /// `evaluate_nd` for an array of rank `dim` or less.
    #[error("axis {axis} is out of range for an array of rank {rank}")]
    AxisOutOfRange {
        /// The rejected axis
        axis: usize,
        /// The rank of the array
        rank: usize,
    },
    /// An array has the wrong shape. For a slice, the shapes have one entry,
    /// its length.
    #[error("{which} has the shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// Which array: an input, or the output
        which: ArrayRole,
        /// The shape the operation needs
        expected: Vec<usize>,
        /// The shape of the array
        actual: Vec<usize>,
    },
    /// The operation is not defined for this input, e.g. default Matsubara
    /// sampling points for basis functions without a definite parity (#183).
    #[error("not supported: {what}")]
    NotSupported {
        /// What is not supported, and why
        what: String,
    },
    /// A matrix decomposition failed, e.g. the SVD iteration did not converge.
    #[error("decomposition failed: {reason}")]
    DecompositionFailed {
        /// What failed, with the values involved
        reason: String,
    },
    /// The GEMM backend rejected a call, e.g. a dimension exceeds the integer
    /// range of an injected BLAS.
    #[error(transparent)]
    Gemm(#[from] GemmError),
    /// A tenferro tensor operation failed.
    #[error("tensor operation failed: {reason}")]
    Tensor {
        /// The message of the tenferro error
        reason: String,
    },
}

impl From<tenferro_tensor::Error> for Error {
    fn from(e: tenferro_tensor::Error) -> Self {
        Error::Tensor {
            reason: e.to_string(),
        }
    }
}

/// Category of an [`Error`]
///
/// Each category corresponds to one failure status code of the C API, shown
/// in parentheses.
///
/// Unlike [`Error`], this enum is exhaustive: a new category would need a new
/// C status code, so adding one is a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// An argument has an invalid value (`SPIR_INVALID_ARGUMENT`).
    InvalidArgument,
    /// An axis or a rank is invalid (`SPIR_INVALID_DIMENSION`).
    InvalidDimension,
    /// An input array has the wrong shape (`SPIR_INPUT_DIMENSION_MISMATCH`).
    InputDimensionMismatch,
    /// An output array has the wrong shape (`SPIR_OUTPUT_DIMENSION_MISMATCH`).
    OutputDimensionMismatch,
    /// The operation is not supported for these arguments
    /// (`SPIR_NOT_SUPPORTED`).
    NotSupported,
    /// A failure that invalid input does not explain
    /// (`SPIR_INTERNAL_ERROR`).
    Internal,
}

impl Error {
    /// Category of the error
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::InvalidParameter { .. }
            | Error::OutOfDomain { .. }
            | Error::InvalidMatsubaraIndex { .. }
            | Error::EmptyInput { .. }
            | Error::NonFiniteInput { .. }
            | Error::InsufficientDefaultPoles { .. } => ErrorKind::InvalidArgument,
            Error::KernelStatisticsMismatch | Error::NotSupported { .. } => ErrorKind::NotSupported,
            Error::AxisOutOfRange { .. } => ErrorKind::InvalidDimension,
            Error::ShapeMismatch {
                which: ArrayRole::Input,
                ..
            } => ErrorKind::InputDimensionMismatch,
            Error::ShapeMismatch {
                which: ArrayRole::Output,
                ..
            } => ErrorKind::OutputDimensionMismatch,
            Error::Gemm(GemmError::DimensionOverflow { .. }) => ErrorKind::InvalidArgument,
            Error::DecompositionFailed { .. }
            | Error::Gemm(GemmError::InvalidArgument(_))
            | Error::Tensor { .. } => ErrorKind::Internal,
        }
    }
}

/// Message of [`Error::InvalidMatsubaraIndex`]: an `n` of the parity of
/// `statistics` can only have been rejected for being negative (in a
/// positive-only sampling)
fn invalid_matsubara_index_message(n: &i64, statistics: &Statistics) -> String {
    let parity_ok = match statistics {
        Statistics::Fermionic => n.rem_euclid(2) == 1,
        Statistics::Bosonic => n.rem_euclid(2) == 0,
    };
    if parity_ok && *n < 0 {
        format!(
            "Matsubara frequency n = {n} is negative, but only non-negative frequencies are \
             allowed here (positive-only sampling)"
        )
    } else {
        format!(
            "Matsubara frequency n = {n} is not allowed for {} statistics",
            statistics.as_str()
        )
    }
}

/// `Ok` if `value` is positive and finite
pub(crate) fn require_positive_finite(name: &'static str, value: f64) -> Result<(), Error> {
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
pub(crate) fn require_finite(name: &'static str, value: f64) -> Result<(), Error> {
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
pub(crate) fn require_accuracy(name: &'static str, epsilon: Option<f64>) -> Result<(), Error> {
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
pub(crate) fn require_threshold(name: &'static str, epsilon: Option<f64>) -> Result<(), Error> {
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
pub(crate) fn require_nonzero_size(name: &'static str, size: Option<usize>) -> Result<(), Error> {
    match size {
        Some(0) => Err(Error::InvalidParameter {
            name,
            value: "0".to_string(),
            reason: "must be positive".to_string(),
        }),
        _ => Ok(()),
    }
}

/// Result type of the fallible public functions of this crate
pub type Result<T, E = Error> = std::result::Result<T, E>;

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;
