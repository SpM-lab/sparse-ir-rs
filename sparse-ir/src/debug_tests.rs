//! Tests for the `SPARSEIR_DEBUG` predicate.
//!
//! These call the pure predicate instead of setting `SPARSEIR_DEBUG`: the
//! process environment is shared with the tests running in parallel.

use super::enables_debug;
use std::ffi::OsStr;

fn enables(value: &str) -> bool {
    enables_debug(Some(OsStr::new(value)))
}

/// pylibsparseir enables debug output when
/// `os.environ.get("SPARSEIR_DEBUG", "").lower() in ("1", "true", "yes", "on")`.
#[test]
fn test_accepted_values_enable_debug() {
    for value in [
        "1", "true", "TRUE", "True", "tRuE", "yes", "YES", "Yes", "on", "ON", "On",
    ] {
        assert!(
            enables(value),
            "SPARSEIR_DEBUG={:?} must enable debug output",
            value
        );
    }
}

#[test]
fn test_other_values_disable_debug() {
    for value in [
        // Falsy spellings, including the "0" that enabled output before
        "",
        "0",
        "false",
        "FALSE",
        "no",
        "off",
        // Other numbers and abbreviations
        "2",
        "01",
        "-1",
        "t",
        "y",
        "enable",
        "debug",
        // Accepted values with extra characters (Python does not strip)
        " 1",
        "1 ",
        " true",
        "true\n",
        "yes please",
        "on,",
        // Non-ASCII look-alikes (fullwidth forms)
        "\u{FF11}",
        "\u{FF34}\u{FF32}\u{FF35}\u{FF25}",
    ] {
        assert!(
            !enables(value),
            "SPARSEIR_DEBUG={:?} must not enable debug output",
            value
        );
    }
}

#[test]
fn test_unset_variable_disables_debug() {
    assert!(!enables_debug(None));
}

#[cfg(unix)]
#[test]
fn test_non_unicode_value_disables_debug() {
    use std::os::unix::ffi::OsStrExt;

    for bytes in [&b"\xff"[..], &b"1\xff"[..], &b"on\x80"[..]] {
        assert!(
            !enables_debug(Some(OsStr::from_bytes(bytes))),
            "SPARSEIR_DEBUG={:?} must not enable debug output",
            bytes
        );
    }
}
