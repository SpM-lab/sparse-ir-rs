//! Example: Using sparse-ir-rs native Rust API (without C-API)
//!
//! This demonstrates how to use the high-level Rust API directly,
//! achieving the same functionality as the C-API version but with
//! a cleaner, more idiomatic Rust interface.

use sparse_ir::{Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, StatisticsType};

fn test_sampling<S: StatisticsType + 'static>(stat_name: &str, beta: f64, wmax: f64, ir_tol: f64) {
    println!("\n======== {} Statistics ========", stat_name);

    // Create kernel and basis
    let lambda = beta * wmax;
    let kernel = LogisticKernel::new(lambda);
    let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(ir_tol), None);

    println!(
        "✅ Created {} basis (size = {}, tolerance = {:.2e})",
        stat_name.to_lowercase(),
        basis.size(),
        ir_tol
    );

    // Tau Sampling
    println!("\n--- Tau Sampling ---");
    let tau_sampling_points = basis.default_tau_sampling_points();
    println!(
        "  Sampling points: {} (basis size: {})",
        tau_sampling_points.len(),
        basis.size()
    );
    if tau_sampling_points.len() >= basis.size() {
        println!("  ✅ OK: {} >= {}", tau_sampling_points.len(), basis.size());
    } else {
        println!(
            "  ⚠️  WARNING: {} < {}",
            tau_sampling_points.len(),
            basis.size()
        );
    }

    // Matsubara Sampling (all frequencies)
    println!("\n--- Matsubara Sampling (positive_only=false) ---");
    let matsubara_all = basis.default_matsubara_sampling_points(false);
    println!(
        "  Sampling points: {} (basis size: {})",
        matsubara_all.len(),
        basis.size()
    );
    if matsubara_all.len() >= basis.size() {
        println!("  ✅ OK: {} >= {}", matsubara_all.len(), basis.size());
    } else {
        println!("  ⚠️  WARNING: {} < {}", matsubara_all.len(), basis.size());
    }

    // Matsubara Sampling (positive only)
    println!("\n--- Matsubara Sampling (positive_only=true) ---");
    let matsubara_pos = basis.default_matsubara_sampling_points(true);
    let effective = 2 * matsubara_pos.len();
    println!(
        "  Sampling points: {} (effective: 2×{} = {}, basis size: {})",
        matsubara_pos.len(),
        matsubara_pos.len(),
        effective,
        basis.size()
    );
    if effective >= basis.size() {
        println!("  ✅ OK: {} >= {}", effective, basis.size());
    } else {
        println!("  ⚠️  WARNING: {} < {}", effective, basis.size());
    }
}

fn main() {
    let t = 0.1; // temperature
    let beta = 1.0 / t;
    let wmax = 10.0;
    let ir_tol = 1e-10;

    println!("=== Test Parameters ===");
    println!("Temperature: {}", t);
    println!("Beta: {}", beta);
    println!("ωmax: {}", wmax);
    println!("Lambda (β * ωmax): {}", beta * wmax);
    println!("IR tolerance: {:.2e}", ir_tol);

    // Test Fermionic
    test_sampling::<Fermionic>("Fermionic", beta, wmax, ir_tol);

    // Test Bosonic
    test_sampling::<Bosonic>("Bosonic", beta, wmax, ir_tol);

    println!("\n========================================");
    println!("✅ All tests completed successfully!");
}
