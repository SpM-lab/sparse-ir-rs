//! Error type of the public API
//!
//! The public functions of this crate that return a `Result` use [`Error`]
//! as the error type. Each variant carries the values that caused it, so
//! that its message locates the problem. The `sparse-ir` crates convert it
//! into their own error type variant by variant.

/// Error returned by the fallible public functions of this crate
///
/// More variants may be added without a major version bump.
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
    /// A point lies outside the domain of a function, e.g. x outside the
    /// knots of a polynomial. NaN lies outside every domain.
    #[error("{name} = {value:?} is outside the domain [{:?}, {:?}]", .domain.0, .domain.1)]
    OutOfDomain {
        /// Name of the argument, as in the documentation of the function
        name: &'static str,
        /// The rejected point
        value: f64,
        /// The closed interval `(lower, upper)` of valid points
        domain: (f64, f64),
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
    /// The operation is not defined for this input, e.g. the even/odd SVE of
    /// a kernel that is not centrosymmetric.
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
}

/// Result type of the fallible public functions of this crate
pub type Result<T, E = Error> = std::result::Result<T, E>;
