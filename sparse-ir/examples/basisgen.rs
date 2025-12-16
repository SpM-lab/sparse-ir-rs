//! Basis generation test
//!
//! This example tests basis generation for various parameters and verifies
//! that sampling points satisfy the required conditions.

use sparse_ir::{Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, StatisticsType};

/// Check that sampling points satisfy the required conditions
fn check_sampling_points<S: StatisticsType + 'static>(
    basis: &FiniteTempBasis<LogisticKernel, S>,
    stat_name: &str,
    beta: f64,
    wmax: f64,
    epsilon: Option<f64>,
) {
    let basis_size = basis.size();
    let eps_str = epsilon
        .map(|e| format!("{:.0e}", e))
        .unwrap_or_else(|| "default".to_string());

    println!(
        "\n--- {} | beta={}, wmax={}, eps={} | basis_size={} ---",
        stat_name, beta, wmax, eps_str, basis_size
    );

    // Check tau sampling
    let tau_points = basis.default_tau_sampling_points();
    print!("  Tau sampling: {} points", tau_points.len());
    if tau_points.len() >= basis_size {
        println!(" ✅ OK ({} >= {})", tau_points.len(), basis_size);
    } else {
        println!(" ❌ FAIL ({} < {})", tau_points.len(), basis_size);
        panic!(
            "Tau sampling check failed: {} < {} for {} beta={} wmax={} eps={}",
            tau_points.len(),
            basis_size,
            stat_name,
            beta,
            wmax,
            eps_str
        );
    }

    // Check Matsubara sampling (positive_only=false)
    let matsubara_all = basis.default_matsubara_sampling_points(false);
    print!("  Matsubara (all): {} points", matsubara_all.len());
    if matsubara_all.len() >= basis_size {
        println!(" ✅ OK ({} >= {})", matsubara_all.len(), basis_size);
    } else {
        println!(" ❌ FAIL ({} < {})", matsubara_all.len(), basis_size);
        panic!(
            "Matsubara sampling (all) check failed: {} < {} for {} beta={} wmax={} eps={}",
            matsubara_all.len(),
            basis_size,
            stat_name,
            beta,
            wmax,
            eps_str
        );
    }

    // Check Matsubara sampling (positive_only=true)
    let matsubara_pos = basis.default_matsubara_sampling_points(true);
    let effective = 2 * matsubara_pos.len();
    print!(
        "  Matsubara (pos): {} points (effective: {})",
        matsubara_pos.len(),
        effective
    );
    if effective >= basis_size {
        println!(" ✅ OK ({} >= {})", effective, basis_size);
    } else {
        println!(" ❌ FAIL ({} < {})", effective, basis_size);
        panic!(
            "Matsubara sampling (positive_only) check failed: {} < {} for {} beta={} wmax={} eps={}",
            effective, basis_size, stat_name, beta, wmax, eps_str
        );
    }
}

/// Test for a single statistics type
fn test_statistics<S: StatisticsType + 'static>(
    stat_name: &str,
    betas: &[f64],
    wmax: f64,
    epsilons: &[Option<f64>],
) {
    println!("\n======== Testing {} Statistics ========", stat_name);

    for &beta in betas {
        for &epsilon in epsilons {
            let lambda = beta * wmax;
            let kernel = LogisticKernel::new(lambda);
            let basis = FiniteTempBasis::<LogisticKernel, S>::new(
                kernel, beta, epsilon, None, // max_size: None means no limit
            );

            check_sampling_points(&basis, stat_name, beta, wmax, epsilon);
        }
    }
}

fn main() {
    println!("=== Basis Generation and Sampling Point Validation ===");
    println!("\nParameters:");
    println!("  wmax: 1.0");
    println!("  beta: 10, 100, 1000, 10000, 100000");
    println!("  epsilon: 1e-8, default (None)");

    let wmax = 1.0;
    let betas = vec![10.0, 100.0, 1000.0, 10000.0, 100000.0];
    let epsilons = vec![Some(1e-8), None];

    // Test Fermionic
    test_statistics::<Fermionic>("Fermionic", &betas, wmax, &epsilons);

    // Test Bosonic
    test_statistics::<Bosonic>("Bosonic", &betas, wmax, &epsilons);

    println!("\n========================================");
    println!("✅ All checks passed!");
}
