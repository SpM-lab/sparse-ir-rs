//! Opt-in debug diagnostics, controlled by the `SPARSEIR_DEBUG` environment
//! variable.

use std::ffi::OsStr;

/// Environment variable that enables debug diagnostics.
const DEBUG_ENV_VAR: &str = "SPARSEIR_DEBUG";

/// Values of [`DEBUG_ENV_VAR`] that enable debug diagnostics, compared ASCII
/// case-insensitively. pylibsparseir accepts the same values.
const ENABLED_VALUES: [&str; 4] = ["1", "true", "yes", "on"];

/// Returns `true` if the `SPARSEIR_DEBUG` environment variable enables debug
/// diagnostics.
///
/// Debug diagnostics are enabled only when `SPARSEIR_DEBUG` is `1`, `true`,
/// `yes` or `on`, in any letter case, the same values pylibsparseir accepts.
/// Any other value, including `0`, `false`, an empty string, or an accepted
/// value with surrounding whitespace, leaves them disabled, as does an unset
/// variable.
///
/// The variable is read on every call. [`debug_warn!`](crate::debug_warn!)
/// and the debug macros of `sparse-ir-capi` all use this function.
pub fn is_debug_enabled() -> bool {
    enables_debug(std::env::var_os(DEBUG_ENV_VAR).as_deref())
}

/// Whether a value of `SPARSEIR_DEBUG` (`None` if the variable is unset)
/// enables debug diagnostics.
fn enables_debug(value: Option<&OsStr>) -> bool {
    // A value that is not valid Unicode cannot equal an accepted value.
    value.and_then(OsStr::to_str).is_some_and(|value| {
        ENABLED_VALUES
            .iter()
            .any(|enabled| value.eq_ignore_ascii_case(enabled))
    })
}

#[cfg(test)]
#[path = "debug_tests.rs"]
mod debug_tests;
