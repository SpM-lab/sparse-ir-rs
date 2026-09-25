//! Status codes of core errors
//!
//! Core errors become C status codes only here, so that every function of the
//! C API reports the same error with the same status.

use crate::{
    SPIR_INPUT_DIMENSION_MISMATCH, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT,
    SPIR_INVALID_DIMENSION, SPIR_NOT_SUPPORTED, SPIR_OUTPUT_DIMENSION_MISMATCH, StatusCode,
};
use sparse_ir::{Error, ErrorKind};

/// Status code that reports `err` to C callers
pub(crate) fn status_from(err: &Error) -> StatusCode {
    status_of_kind(err.kind())
}

/// Status code of each error category
fn status_of_kind(kind: ErrorKind) -> StatusCode {
    match kind {
        ErrorKind::InvalidArgument => SPIR_INVALID_ARGUMENT,
        ErrorKind::InvalidDimension => SPIR_INVALID_DIMENSION,
        ErrorKind::InputDimensionMismatch => SPIR_INPUT_DIMENSION_MISMATCH,
        ErrorKind::OutputDimensionMismatch => SPIR_OUTPUT_DIMENSION_MISMATCH,
        ErrorKind::NotSupported => SPIR_NOT_SUPPORTED,
        ErrorKind::Internal => SPIR_INTERNAL_ERROR,
    }
}

#[cfg(test)]
mod tests {
    // The status codes, `StatusCode`, `Error` and `ErrorKind` come from the
    // imports of the parent module (Step 3).
    use super::*;
    use crate::SPIR_COMPUTATION_SUCCESS;
    use sparse_ir::Statistics;

    /// Every error category has its own failure status.
    #[test]
    fn test_status_of_every_kind() {
        let table = [
            (ErrorKind::InvalidArgument, SPIR_INVALID_ARGUMENT),
            (ErrorKind::InvalidDimension, SPIR_INVALID_DIMENSION),
            (
                ErrorKind::InputDimensionMismatch,
                SPIR_INPUT_DIMENSION_MISMATCH,
            ),
            (
                ErrorKind::OutputDimensionMismatch,
                SPIR_OUTPUT_DIMENSION_MISMATCH,
            ),
            (ErrorKind::NotSupported, SPIR_NOT_SUPPORTED),
            (ErrorKind::Internal, SPIR_INTERNAL_ERROR),
        ];
        for (kind, status) in table {
            assert_eq!(status_of_kind(kind), status, "{kind:?}");
            assert_ne!(status, SPIR_COMPUTATION_SUCCESS, "{kind:?}");
        }
        let mut statuses: Vec<StatusCode> = table.iter().map(|&(_, status)| status).collect();
        statuses.sort();
        statuses.dedup();
        assert_eq!(statuses.len(), table.len());
    }

    /// The status of one error of every variant that the core returns
    #[test]
    fn test_status_of_every_variant() {
        let table = [
            (
                Error::InvalidParameter {
                    name: "rtol",
                    value: "NaN".to_string(),
                    reason: "must be in (0, 1)".to_string(),
                },
                SPIR_INVALID_ARGUMENT,
            ),
            (
                Error::InvalidMatsubaraIndex {
                    n: 2,
                    statistics: Statistics::Fermionic,
                },
                SPIR_INVALID_ARGUMENT,
            ),
            (Error::EmptyInput { name: "poles" }, SPIR_INVALID_ARGUMENT),
            (
                Error::NonFiniteInput {
                    name: "matrix",
                    index: vec![1, 2],
                    value: f64::NAN,
                },
                SPIR_INVALID_ARGUMENT,
            ),
            (Error::KernelStatisticsMismatch, SPIR_NOT_SUPPORTED),
            (
                Error::InsufficientDefaultPoles {
                    basis_size: 6,
                    n_poles: 3,
                },
                SPIR_INVALID_ARGUMENT,
            ),
            (
                Error::DecompositionFailed {
                    reason: "the SVD did not converge within 50 iterations".to_string(),
                },
                SPIR_INTERNAL_ERROR,
            ),
        ];
        for (err, status) in &table {
            assert_eq!(status_from(err), *status, "{err:?}");
        }
    }
}
