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
        (
            Error::OutOfDomain {
                name: "tau",
                value: 1.5,
                domain: (-1.0, 1.0),
            },
            ErrorKind::InvalidArgument,
            "tau = 1.5 is outside the domain [-1.0, 1.0]",
        ),
        (
            Error::NotSupported {
                what: "default Matsubara sampling points of functions with symm = 0".to_string(),
            },
            ErrorKind::NotSupported,
            "not supported: default Matsubara sampling points of functions with symm = 0",
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

/// InvalidMatsubaraIndex says why n is not allowed: a wrong parity, or a
/// negative n where only non-negative frequencies are allowed (positive-only
/// samplings, #247). The fields are those of spec §3; the message follows
/// from n and the statistics.
#[test]
fn test_invalid_matsubara_index_messages() {
    let message = |n, statistics| Error::InvalidMatsubaraIndex { n, statistics }.to_string();
    assert_eq!(
        message(2, Statistics::Fermionic),
        "Matsubara frequency n = 2 is not allowed for fermionic statistics"
    );
    assert_eq!(
        message(-3, Statistics::Bosonic),
        "Matsubara frequency n = -3 is not allowed for bosonic statistics"
    );
    assert_eq!(
        message(-3, Statistics::Fermionic),
        "Matsubara frequency n = -3 is negative, but only non-negative frequencies are \
         allowed here (positive-only sampling)"
    );
    assert_eq!(
        message(-2, Statistics::Bosonic),
        "Matsubara frequency n = -2 is negative, but only non-negative frequencies are \
         allowed here (positive-only sampling)"
    );
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

/// Position of each variant in `every_variant`. The match has no wildcard
/// arm, so adding a variant to `Error` fails to compile here: give it the
/// next number, raise `VARIANT_COUNT`, and add it to `every_variant` and to
/// the table of `sparse-ir-capi/src/status.rs`.
fn variant_number(err: &Error) -> usize {
    match err {
        Error::InvalidParameter { .. } => 0,
        Error::InvalidMatsubaraIndex { .. } => 1,
        Error::EmptyInput { .. } => 2,
        Error::NonFiniteInput { .. } => 3,
        Error::KernelStatisticsMismatch => 4,
        Error::InsufficientDefaultPoles { .. } => 5,
        Error::DecompositionFailed { .. } => 6,
        Error::OutOfDomain { .. } => 7,
        Error::NotSupported { .. } => 8,
    }
}

const VARIANT_COUNT: usize = 9;

#[test]
fn test_every_variant_lists_each_variant_once() {
    let mut numbers: Vec<usize> = every_variant()
        .iter()
        .map(|(err, _, _)| variant_number(err))
        .collect();
    numbers.sort();
    assert_eq!(numbers, (0..VARIANT_COUNT).collect::<Vec<_>>());
}

#[test]
fn test_parameter_checks() {
    use super::{
        require_accuracy, require_finite, require_nonzero_size, require_positive_finite,
        require_threshold,
    };

    let invalid = |name: &'static str, value: &str, reason: &str| Error::InvalidParameter {
        name,
        value: value.to_string(),
        reason: reason.to_string(),
    };

    assert_eq!(require_positive_finite("beta", 1e-300), Ok(()));
    for (value, shown) in [
        (0.0, "0.0"),
        (-1.0, "-1.0"),
        (f64::NAN, "NaN"),
        (f64::INFINITY, "inf"),
    ] {
        assert_eq!(
            require_positive_finite("beta", value),
            Err(invalid("beta", shown, "must be positive and finite"))
        );
    }

    for ok in [None, Some(1e-300), Some(0.5), Some(1.0 - f64::EPSILON)] {
        assert_eq!(require_accuracy("epsilon", ok), Ok(()));
    }
    for (value, shown) in [
        (0.0, "0.0"),
        (1.0, "1.0"),
        (-1e-3, "-0.001"),
        (f64::NAN, "NaN"),
    ] {
        assert_eq!(
            require_accuracy("epsilon", Some(value)),
            Err(invalid("epsilon", shown, "must be in (0, 1)"))
        );
    }

    for ok in [None, Some(0.0), Some(0.5)] {
        assert_eq!(require_threshold("eps", ok), Ok(()));
    }
    for (value, shown) in [(1.0, "1.0"), (-1e-3, "-0.001"), (f64::INFINITY, "inf")] {
        assert_eq!(
            require_threshold("eps", Some(value)),
            Err(invalid("eps", shown, "must be in [0, 1)"))
        );
    }

    assert_eq!(require_nonzero_size("max_size", None), Ok(()));
    assert_eq!(require_nonzero_size("max_size", Some(1)), Ok(()));
    assert_eq!(
        require_nonzero_size("max_size", Some(0)),
        Err(invalid("max_size", "0", "must be positive"))
    );

    assert_eq!(require_finite("omega", -1e300), Ok(()));
    for (value, shown) in [(f64::NAN, "NaN"), (f64::NEG_INFINITY, "-inf")] {
        assert_eq!(
            require_finite("omega", value),
            Err(invalid("omega", shown, "must be finite"))
        );
    }
}
