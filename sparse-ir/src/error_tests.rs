//! Tests for the crate error type

use super::{Error, ErrorKind};
use crate::traits::Statistics;

/// One error of every variant, with its kind and its message
fn every_variant() -> Vec<(Error, ErrorKind, &'static str)> {
    vec![
        (
            Error::InvalidParameter {
                name: "rtol",
                value: "NaN".to_string(),
                reason: "must be in (0, 1)".to_string(),
            },
            ErrorKind::InvalidArgument,
            "invalid rtol = NaN: must be in (0, 1)",
        ),
        (
            Error::InvalidMatsubaraIndex {
                n: 2,
                statistics: Statistics::Fermionic,
            },
            ErrorKind::InvalidArgument,
            "Matsubara frequency n = 2 is not allowed for fermionic statistics",
        ),
        (
            Error::EmptyInput { name: "poles" },
            ErrorKind::InvalidArgument,
            "poles must not be empty",
        ),
        (
            Error::NonFiniteInput {
                name: "matrix",
                index: vec![1, 2],
                value: f64::INFINITY,
            },
            ErrorKind::InvalidArgument,
            "matrix has the non-finite entry inf at index [1, 2]",
        ),
        (
            Error::KernelStatisticsMismatch,
            ErrorKind::NotSupported,
            "kernel does not support the requested statistics: kernels with ypower = 1 \
             (e.g. RegularizedBoseKernel) require bosonic statistics",
        ),
        (
            Error::InsufficientDefaultPoles {
                basis_size: 12,
                n_poles: 7,
            },
            ErrorKind::InvalidArgument,
            "number of default poles (7) is less than the basis size (12)",
        ),
        (
            Error::DecompositionFailed {
                reason: "the SVD did not converge within 50 iterations".to_string(),
            },
            ErrorKind::Internal,
            "decomposition failed: the SVD did not converge within 50 iterations",
        ),
    ]
}

#[test]
fn test_kind_of_every_variant() {
    for (err, kind, _) in every_variant() {
        assert_eq!(err.kind(), kind, "{err:?}");
    }
}

/// The messages carry the observed values (rules/rust.md, "Typed Errors In
/// The Rust Core").
#[test]
fn test_message_of_every_variant() {
    for (err, _, message) in every_variant() {
        assert_eq!(err.to_string(), message, "{err:?}");
    }
}

/// `Error` can be returned with `?` as a boxed `Send + Sync` error, e.g. to
/// `anyhow`, and downcast back to the typed error.
#[test]
fn test_error_boxes_and_downcasts() {
    fn fails() -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        Err(Error::EmptyInput { name: "poles" })?;
        Ok(())
    }
    let err = fails().unwrap_err();
    assert_eq!(err.to_string(), "poles must not be empty");
    assert!(err.source().is_none());
    assert_eq!(
        err.downcast_ref::<Error>(),
        Some(&Error::EmptyInput { name: "poles" })
    );
}
