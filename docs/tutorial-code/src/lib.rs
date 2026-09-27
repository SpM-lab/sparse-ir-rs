//! Helpers shared by the runnable examples of the sparse-ir Rust tutorial.
//!
//! Every example is a binary under `src/bin/`. A binary computes, writes its
//! numbers to CSV under [`data_dir`], and prints nothing that a test needs to
//! parse. The plotting scripts in `docs/plotting/` and the verification tests
//! in `tests/` both read those CSV files, so the pictures in the book and the
//! numbers the CI compares always come from the same run.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub mod csv;
pub mod dmft;
pub mod elliptic;
pub mod linalg;
pub mod mesh;
pub mod optimize;
pub mod quad;
pub mod semicircle;
pub mod spectra;

pub use csv::{Table, read_table, write_table};
pub use elliptic::{ellipe, ellipk};
pub use linalg::{Eigen2, Hermitian2};
pub use mesh::{IrMesh, MomentumGrid, evaluate_rows, reverse_tau_rows, tau_reversal};
pub use optimize::{FistaReport, fista, soft_threshold, soft_threshold_nonneg};
pub use quad::integrate_segments;
pub use semicircle::{
    semicircle, semicircle_coefficients, semicircle_overlaps, shifted_semicircle,
    shifted_semicircle_overlaps,
};
pub use spectra::three_gaussians;

/// Directory the examples write their CSV output to.
///
/// `SPARSEIR_TUTORIAL_DATA_DIR` overrides it; otherwise it is
/// `docs/tutorial-code/data/`, next to this crate's manifest. The directory is
/// created if it does not exist.
pub fn data_dir() -> io::Result<PathBuf> {
    let dir = match std::env::var_os("SPARSEIR_TUTORIAL_DATA_DIR") {
        Some(value) => PathBuf::from(value),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("data"),
    };
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Path of one output file of the example named `example`.
///
/// The examples keep their files apart so that running one never overwrites
/// another one's numbers: `data/<example>/<name>.csv`.
pub fn output_path(example: &str, name: &str) -> io::Result<PathBuf> {
    let dir = data_dir()?.join(example);
    fs::create_dir_all(&dir)?;
    Ok(dir.join(format!("{name}.csv")))
}

/// Path of one committed input file of the example named `example`.
///
/// A couple of examples start from data rather than from a formula. That data
/// is committed under `input/<example>/<name>.csv` so a run never depends on
/// the network, and so the Rust example and its Python reference start from
/// exactly the same numbers.
pub fn input_path(example: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("input")
        .join(example)
        .join(format!("{name}.csv"))
}

/// The provenance line every CSV carries as its first line.
pub fn provenance(example: &str) -> String {
    let mut line = String::new();
    write!(
        line,
        "# example={example} sparse-ir={}",
        sparse_ir_version()
    )
    .expect("writing to a String cannot fail");
    line
}

/// Version of the `sparse-ir` crate the examples were built against.
///
/// Read from the path dependency's manifest at build time, so a CSV always
/// says which library produced it.
pub fn sparse_ir_version() -> &'static str {
    // `sparse-ir` does not export its version, and Cargo only sets
    // CARGO_PKG_VERSION for this crate, so read it from the dependency's
    // metadata that Cargo does expose.
    env!("SPARSE_IR_VERSION")
}
