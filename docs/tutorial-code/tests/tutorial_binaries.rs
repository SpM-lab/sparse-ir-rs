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
];

#[test]
fn every_example_runs() {
    if !common::examples_requested() {
        eprintln!("skipped: set SPARSEIR_TUTORIAL_RUN=1 to run the examples");
        return;
    }

    for (name, binary) in EXAMPLES {
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
