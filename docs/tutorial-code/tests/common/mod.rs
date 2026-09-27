//! Shared helpers for the tutorial's integration tests.
//!
//! Each test binary compiles its own copy of this module and uses only part of
//! it, so unused items here are expected rather than a sign of dead code.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sparse_ir_tutorial::{Table, read_table};

/// Whether the tests that need the examples to have run should run.
///
/// Running every example takes minutes, so `cargo test` on its own only
/// compiles them and checks the library's own units. Set
/// `SPARSEIR_TUTORIAL_RUN=1` — as `scripts/check.sh --run` and CI do — to run
/// the examples and check their numbers.
pub fn examples_requested() -> bool {
    std::env::var_os("SPARSEIR_TUTORIAL_RUN").is_some_and(|value| value != "0")
}

/// Whether the `_scan` examples should run as well.
///
/// The scans repeat a self-consistent calculation over a grid of parameters,
/// which costs minutes rather than seconds. They run in the scheduled
/// applied-examples job and under `scripts/check.sh --run --scans`, never on a
/// pull request.
pub fn scans_requested() -> bool {
    examples_requested()
        && std::env::var_os("SPARSEIR_TUTORIAL_SCANS").is_some_and(|value| value != "0")
}

pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Where an example wrote its output, honouring the same override the
/// examples themselves honour.
pub fn data_dir() -> PathBuf {
    match std::env::var_os("SPARSEIR_TUTORIAL_DATA_DIR") {
        Some(value) => PathBuf::from(value),
        None => crate_dir().join("data"),
    }
}

/// Reads `data/<example>/<name>.csv`, the output of a run.
pub fn output(example: &str, name: &str) -> Table {
    read(
        &data_dir().join(example).join(format!("{name}.csv")),
        example,
    )
}

/// Reads `reference/<example>/<name>.csv`, written by
/// `scripts/make_reference.py` from the published Python implementation.
pub fn reference(example: &str, name: &str) -> Table {
    read(
        &crate_dir()
            .join("reference")
            .join(example)
            .join(format!("{name}.csv")),
        example,
    )
}

fn read(path: &Path, example: &str) -> Table {
    read_table(path).unwrap_or_else(|e| {
        panic!("{e}\nRun the example first:\n    cargo run --profile ci --bin {example}")
    })
}

/// One comparison a verification test made.
struct Record {
    example: String,
    column: String,
    tolerance: f64,
    deviation: f64,
    passed: bool,
}

static SUMMARY: Mutex<Vec<Record>> = Mutex::new(Vec::new());

/// Records one comparison and rewrites `data/verification-summary.json`.
///
/// The file is rewritten on every call rather than at the end, because a test
/// that fails never reaches an end: the comparison that failed has to be in
/// the file, and it is the last one written.
fn record(example: &str, column: &str, tolerance: f64, deviation: f64, passed: bool) {
    let mut summary = SUMMARY.lock().expect("the summary mutex was poisoned");
    summary.push(Record {
        example: example.to_string(),
        column: column.to_string(),
        tolerance,
        deviation,
        passed,
    });
    let mut json = String::from("[\n");
    for (index, record) in summary.iter().enumerate() {
        let comma = if index + 1 == summary.len() { "" } else { "," };
        json.push_str(&format!(
            "  {{\"example\": \"{}\", \"column\": \"{}\", \"tolerance\": {:e}, \
             \"deviation\": {:e}, \"passed\": {}}}{comma}\n",
            record.example, record.column, record.tolerance, record.deviation, record.passed
        ));
    }
    json.push_str("]\n");
    let directory = data_dir();
    let _ = std::fs::create_dir_all(&directory);
    let _ = std::fs::write(directory.join("verification-summary.json"), json);
}

/// Which example a table came from, taken from its provenance comment
/// (`# example=<name> ...`).
fn example_of(table: &Table) -> String {
    table
        .comment
        .split_whitespace()
        .find_map(|field| field.strip_prefix("example="))
        .unwrap_or("unknown")
        .to_string()
}

/// Asserts that two columns agree to `tol`, measured against the largest value
/// in the reference column.
///
/// Comparing against the largest value rather than against each entry is what
/// these quantities need: the IR coefficients fall off over fifteen orders of
/// magnitude, and the small ones are only ever accurate in absolute terms.
pub fn assert_close(actual: &Table, expected: &Table, column: &str, tol: f64) {
    let expected_table = expected;
    let actual = actual.expect_column(column);
    let expected = expected.expect_column(column);
    assert_eq!(
        actual.len(),
        expected.len(),
        "column `{column}`: {} values but the reference has {}",
        actual.len(),
        expected.len()
    );

    let scale = expected.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let scale = if scale > 0.0 { scale } else { 1.0 };

    let mut worst = (0usize, 0.0_f64);
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let deviation = (a - e).abs() / scale;
        if deviation > worst.1 {
            worst = (index, deviation);
        }
    }
    record(
        &example_of(expected_table),
        column,
        tol,
        worst.1,
        worst.1 <= tol,
    );
    assert!(
        worst.1 <= tol,
        "column `{column}`: relative deviation {:.3e} exceeds {tol:.3e} at row {} ({} vs {})",
        worst.1,
        worst.0,
        actual[worst.0],
        expected[worst.0]
    );
}

/// Asserts that two columns of whole numbers agree exactly.
pub fn assert_exact_integers(actual: &Table, expected: &Table, column: &str) {
    let example = example_of(expected);
    let actual = actual.expect_column(column);
    let expected = expected.expect_column(column);
    assert_eq!(actual.len(), expected.len(), "column `{column}`: length");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let identical = a.to_bits() == e.to_bits();
        if !identical || index + 1 == actual.len() {
            record(
                &example,
                column,
                0.0,
                if identical { 0.0 } else { 1.0 },
                identical,
            );
        }
        assert!(
            identical,
            "column `{column}`: row {index} is {a} but the reference says {e}"
        );
    }
}

/// Asserts that `column` is numerical noise: small against the largest value
/// of `scale_column` in the same table.
///
/// Use it for a quantity that vanishes by symmetry. There is nothing to
/// compare against a reference there — both implementations produce rounding
/// error around zero, and the two lots of rounding error have no reason to
/// agree. What is worth asserting is that it really is rounding error.
///
/// The tolerances that go with it are loose for the same reason. How large
/// the noise around zero comes out depends on the machine's BLAS and libm —
/// the same run is an order of magnitude noisier under OpenBLAS on x86 than
/// under Accelerate on arm64 — so the assertion is written to catch a
/// quantity that does not vanish at all, not to pin the noise level down.
pub fn assert_negligible(table: &Table, column: &str, scale_column: &str, tol: f64) {
    let scale = table
        .expect_column(scale_column)
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let worst = table
        .expect_column(column)
        .iter()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let deviation = if scale > 0.0 { worst / scale } else { worst };
    record(
        &example_of(table),
        column,
        tol,
        deviation,
        worst <= tol * scale,
    );
    assert!(
        worst <= tol * scale,
        "column `{column}`: largest value {worst:.3e} is not negligible \
         against |{scale_column}|max = {scale:.3e}"
    );
}
