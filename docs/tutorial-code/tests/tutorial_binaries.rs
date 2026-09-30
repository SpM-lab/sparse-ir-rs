//! Runs every tutorial example and checks that it finishes.
//!
//! The examples are what produce the numbers `verification.rs` checks and the
//! figures the book shows, so "it still runs" is the first thing to know.
//! Skipped unless `SPARSEIR_TUTORIAL_RUN` is set; see `common::examples_requested`.

mod common;

use std::process::Command;

/// One entry per binary in `src/bin/`. `env!("CARGO_BIN_EXE_…")` makes Cargo
/// build it for this test, so the list cannot go stale without a compile
/// error.
const EXAMPLES: &[(&str, &str)] = &[
    (
        "sparse_sampling_demo",
        env!("CARGO_BIN_EXE_sparse_sampling_demo"),
    ),
    ("transformation", env!("CARGO_BIN_EXE_transformation")),
    ("dlr", env!("CARGO_BIN_EXE_dlr")),
    ("minipole", env!("CARGO_BIN_EXE_minipole")),
    ("spm", env!("CARGO_BIN_EXE_spm")),
    (
        "analytic_continuation",
        env!("CARGO_BIN_EXE_analytic_continuation"),
    ),
    (
        "second_order_perturbation",
        env!("CARGO_BIN_EXE_second_order_perturbation"),
    ),
    ("gw", env!("CARGO_BIN_EXE_gw")),
    ("liechtenstein", env!("CARGO_BIN_EXE_liechtenstein")),
    (
        "orbital_magnetic_susceptibility",
        env!("CARGO_BIN_EXE_orbital_magnetic_susceptibility"),
    ),
    ("dmft_ipt", env!("CARGO_BIN_EXE_dmft_ipt")),
    ("tpsc", env!("CARGO_BIN_EXE_tpsc")),
    // A scan by name, but fifty-one solves of a 24 × 24 lattice take a
    // fraction of a second, so it stays where pull requests can see it.
    ("tpsc_scan", env!("CARGO_BIN_EXE_tpsc_scan")),
    ("flex", env!("CARGO_BIN_EXE_flex")),
    (
        "eliashberg_holstein",
        env!("CARGO_BIN_EXE_eliashberg_holstein"),
    ),
];

/// The examples that repeat a whole self-consistent calculation over a grid of
/// parameters. They are minutes rather than seconds, so they run in the
/// scheduled applied-examples job instead of on every pull request; see
/// `common::scans_requested`.
const SCANS: &[(&str, &str)] = &[
    ("dmft_ipt_scan", env!("CARGO_BIN_EXE_dmft_ipt_scan")),
    ("flex_scan", env!("CARGO_BIN_EXE_flex_scan")),
    (
        "eliashberg_holstein_scan",
        env!("CARGO_BIN_EXE_eliashberg_holstein_scan"),
    ),
];

#[test]
fn every_example_runs() {
    if !common::examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to run the examples");
        return;
    }

    run(EXAMPLES);
}

#[test]
fn every_scan_runs() {
    if !common::scans_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_SCANS=1 to run the parameter scans");
        return;
    }

    run(SCANS);
}

fn run(examples: &[(&str, &str)]) {
    for (name, binary) in examples {
        let output = Command::new(binary)
            .output()
            .unwrap_or_else(|e| panic!("could not start `{name}` ({binary}): {e}"));
        assert!(
            output.status.success(),
            "`{name}` exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}
