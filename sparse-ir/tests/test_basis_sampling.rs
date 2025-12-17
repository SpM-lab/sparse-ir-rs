//! Basis generation test as Rust test code
//!
//! This test verifies that sampling points satisfy the required conditions
//! for various basis generation parameters.

use sparse_ir::{Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, StatisticsType};

/// Check that sampling points satisfy the required conditions
fn check_sampling_points<S: StatisticsType + 'static>(
    basis: &FiniteTempBasis<LogisticKernel, S>,
    beta: f64,
    wmax: f64,
    epsilon: Option<f64>,
) {
    let basis_size = basis.size();

    // Check tau sampling
    let tau_points = basis.default_tau_sampling_points();
    assert!(
        tau_points.len() >= basis_size,
        "Tau sampling check failed: {} < {} for beta={} wmax={} eps={:?}",
        tau_points.len(),
        basis_size,
        beta,
        wmax,
        epsilon
    );

    // Check Matsubara sampling (positive_only=false)
    let matsubara_all = basis.default_matsubara_sampling_points(false);
    assert!(
        matsubara_all.len() >= basis_size,
        "Matsubara sampling (all) check failed: {} < {} for beta={} wmax={} eps={:?}",
        matsubara_all.len(),
        basis_size,
        beta,
        wmax,
        epsilon
    );

    // Check Matsubara sampling (positive_only=true)
    let matsubara_pos = basis.default_matsubara_sampling_points(true);
    let effective = 2 * matsubara_pos.len();
    assert!(
        effective >= basis_size,
        "Matsubara sampling (positive_only) check failed: {} < {} for beta={} wmax={} eps={:?}",
        effective,
        basis_size,
        beta,
        wmax,
        epsilon
    );
}

/// Test for a single statistics type
fn test_statistics<S: StatisticsType + 'static>(
    betas: &[f64],
    wmax: f64,
    epsilons: &[Option<f64>],
) {
    for &beta in betas {
        for &epsilon in epsilons {
            let lambda = beta * wmax;
            let kernel = LogisticKernel::new(lambda);
            let basis = FiniteTempBasis::<LogisticKernel, S>::new(
                kernel, beta, epsilon, None, // max_size: None means no limit
            );

            check_sampling_points::<S>(&basis, beta, wmax, epsilon);
        }
    }
}

#[test]
fn test_basis_sampling() {
    let wmax = 1.0;
    // let betas = vec![10.0, 100.0, 1000.0, 10000.0, 100000.0];
    let betas = vec![10.0, 100.0, 1000.0];
    let epsilons = vec![Some(1e-8), None];

    // Test Fermionic
    test_statistics::<Fermionic>(&betas, wmax, &epsilons);

    // Test Bosonic
    test_statistics::<Bosonic>(&betas, wmax, &epsilons);
}

#[cfg(test)]
mod high_api_sampling_tests {
    use sparse_ir::{Bosonic, Fermionic, FiniteTempBasis, LogisticKernel, StatisticsType};

    fn test_sampling<S: StatisticsType + 'static>(beta: f64, wmax: f64, ir_tol: f64) {
        // カーネルと基底の生成
        let lambda = beta * wmax;
        let kernel = LogisticKernel::new(lambda);
        let basis = FiniteTempBasis::<LogisticKernel, S>::new(kernel, beta, Some(ir_tol), None);

        // Tau Sampling
        let tau_sampling_points = basis.default_tau_sampling_points();
        assert!(
            tau_sampling_points.len() >= basis.size(),
            "Tau sampling points insufficient: {} < {}",
            tau_sampling_points.len(),
            basis.size()
        );

        // Matsubara Sampling (all frequencies)
        let matsubara_all = basis.default_matsubara_sampling_points(false);
        assert!(
            matsubara_all.len() >= basis.size(),
            "Matsubara (all) sampling points insufficient: {} < {}",
            matsubara_all.len(),
            basis.size()
        );

        // Matsubara Sampling (positive only)
        let matsubara_pos = basis.default_matsubara_sampling_points(true);
        let effective = 2 * matsubara_pos.len();
        assert!(
            effective >= basis.size(),
            "Matsubara (positive only) sampling points insufficient: 2*{} = {} < {}",
            matsubara_pos.len(),
            effective,
            basis.size()
        );
    }

    #[test]
    fn test_high_api_sampling() {
        // テスト条件
        let t = 0.1;
        let beta = 1.0 / t;
        let wmax = 10.0;
        let ir_tol = 1e-10;

        // Fermionic
        test_sampling::<Fermionic>(beta, wmax, ir_tol);

        // Bosonic
        test_sampling::<Bosonic>(beta, wmax, ir_tol);
    }
}
