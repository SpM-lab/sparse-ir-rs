//! Example: Using sparse-ir-rs native Rust API (without C-API)
//!
//! This demonstrates how to use the high-level Rust API directly,
//! achieving the same functionality as the C-API version but with
//! a cleaner, more idiomatic Rust interface.

use sparse_ir::{Fermionic, FiniteTempBasis, LogisticKernel, TauSampling};

fn main() {
    let t = 0.1; // temperature
    let beta = 1.0 / t;
    let wmax = 10.0;
    let ir_tol = 1e-10;

    // Step 1: Create kernel
    let lambda = beta * wmax;
    let kernel = LogisticKernel::new(lambda);
    println!("✅ Created kernel with lambda = {}", lambda);

    // Step 2 & 3: Create fermionic basis (SVE is computed internally)
    // No need to manually compute SVE - it's done automatically in FiniteTempBasis::new
    let basis = FiniteTempBasis::<LogisticKernel, Fermionic>::new(
        kernel,
        beta,
        Some(ir_tol), // epsilon
        None,         // max_size: None means no limit
    );
    println!(
        "✅ Created fermionic basis (size = {}, tolerance = {})",
        basis.size(),
        ir_tol
    );

    // Step 4: Create tau sampling with custom sampling points
    let sampling_points = vec![0.0];
    let smpl_tau0 = TauSampling::<Fermionic>::with_sampling_points(&basis, sampling_points.clone());
    println!(
        "✅ Created tau sampling with sampling_points = {:?}",
        sampling_points
    );

    // Example: Demonstrate some additional capabilities
    println!("\n--- Additional Information ---");
    println!("Basis accuracy: {:.2e}", basis.accuracy());
    println!("Lambda (β * ωmax): {}", basis.lambda());
    println!("ωmax: {}", basis.wmax());
    println!("Sampling points count: {}", smpl_tau0.n_sampling_points());

    // Get default tau sampling points for comparison
    let default_taus = basis.default_tau_sampling_points();
    println!("Default tau sampling points: {} points", default_taus.len());

    // No manual cleanup needed - Rust's Drop handles everything automatically
    println!("\n✅ Successfully completed (resources cleaned up automatically)");
}
