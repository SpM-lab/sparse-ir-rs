//! Round-trip tests demonstrating DLR/IR/sampling cycle
//!
//! This example is a Rust port of the C++ integration test `cinterface_integration.cxx`,
//! but uses the native Rust API instead of the C-API. It demonstrates:
//! - Creating IR basis from kernel
//! - Converting between DLR and IR representations
//! - Evaluating Green's functions on tau and Matsubara grids
//! - Round-trip consistency checks (tau ↔ IR ↔ Matsubara)
//!
//! Run with: `cargo run --example roundtrip`
//! Or use the wrapper script: `./examples/run_roundtrip.sh` (saves log to logs/roundtrip_*.log)

use num_complex::{Complex, ComplexFloat};
use sparse_ir::{
    Bosonic, DiscreteLehmannRepresentation, Fermionic, FiniteTempBasis, LogisticKernel, Matrix,
    MatsubaraSampling, RegularizedBoseKernel, TauSampling, TensorScalar, TypedTensor,
    basis_trait::Basis,
    kernel::{CentrosymmKernel, KernelProperties},
    sve::{SVEResult, TworkType, compute_sve},
    traits::StatisticsType,
};
use std::ops::Sub;

/// Compute maximum relative error between two tensors
///
/// This matches the C++ implementation: computes max(|a - b|) / max(|a|)
/// rather than max(|a - b| / |a|) for each element.
///
/// Works with both real (`f64`) and complex (`Complex<f64>`) tensors.
fn max_relative_error<T>(a: &TypedTensor<T>, b: &TypedTensor<T>) -> f64
where
    T: TensorScalar + ComplexFloat<Real = f64> + Sub<Output = T>,
{
    assert_eq!(a.shape(), b.shape());
    let (a, b) = (a.host_data().unwrap(), b.host_data().unwrap());
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0.0_f64, |acc, (x, y)| acc.max((*x - *y).abs()));
    let max_ref = a.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()));

    // Avoid division by zero (behavior similar to C++ helper)
    if max_ref < 1e-15 {
        max_diff
    } else {
        max_diff / max_ref
    }
}

/// Compute maximum relative error between two real tensors
fn max_relative_error_real(a: &TypedTensor<f64>, b: &TypedTensor<f64>) -> f64 {
    max_relative_error(a, b)
}

/// Compute maximum relative error between two complex tensors
fn max_relative_error_complex(a: &TypedTensor<Complex<f64>>, b: &TypedTensor<Complex<f64>>) -> f64 {
    max_relative_error(a, b)
}

/// Promote a real tensor to complex element-wise.
fn to_complex(t: &TypedTensor<f64>) -> TypedTensor<Complex<f64>> {
    let data = t
        .host_data()
        .unwrap()
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    TypedTensor::from_vec_col_major(t.shape().to_vec(), data).unwrap()
}

/// Get dimensions for N-dimensional tensor with target_dim at specified position
/// Similar to C++ _get_dims function
fn get_dims(target_dim_size: usize, extra_dims: &[usize], target_dim: usize) -> Vec<usize> {
    let ndim = 1 + extra_dims.len();
    let mut dims = vec![0; ndim];
    dims[target_dim] = target_dim_size;
    let mut pos = 0;
    for i in 0..ndim {
        if i == target_dim {
            continue;
        }
        dims[i] = extra_dims[pos];
        pos += 1;
    }
    dims
}

/// Contract a multi-dimensional tensor along a specific dimension with a 2D matrix
///
/// This function performs: result = matrix @ coeffs (along target_dim)
/// where `matrix` is a 2D matrix [n_points, n_poles] and `coeffs` is an N-dimensional
/// column-major tensor with size `n_poles` along `target_dim`. It is a plain
/// reference loop, independent of the library's GEMM path.
fn contract_along_dim<T>(
    matrix: &Matrix<T>,
    coeffs: &TypedTensor<T>,
    target_dim: usize,
) -> TypedTensor<T>
where
    T: TensorScalar + ComplexFloat + Copy,
{
    let (n_points, n_poles) = (matrix.shape()[0], matrix.shape()[1]);
    let shape = coeffs.shape().to_vec();
    assert!(target_dim < shape.len(), "invalid target_dim");
    assert_eq!(shape[target_dim], n_poles, "size mismatch along target_dim");
    let pre: usize = shape[..target_dim].iter().product();
    let post: usize = shape[target_dim + 1..].iter().product();
    let a = matrix.host_data().unwrap();
    let x = coeffs.host_data().unwrap();
    let mut y = vec![T::zero(); pre * n_points * post];
    for q in 0..post {
        for i in 0..n_points {
            for p in 0..pre {
                let mut sum = T::zero();
                for k in 0..n_poles {
                    sum = sum + a[i + n_points * k] * x[p + pre * (k + n_poles * q)];
                }
                y[p + pre * (i + n_points * q)] = sum;
            }
        }
    }
    let mut out_shape = shape;
    out_shape[target_dim] = n_points;
    TypedTensor::from_vec_col_major(out_shape, y).unwrap()
}

/// Generate random DLR coefficient for a given pole
///
/// This function generates a reproducible random coefficient based on the seed
/// and multi-dimensional index. The coefficient is scaled by the pole value
/// to ensure appropriate magnitude.
///
/// # Arguments
/// * `seed` - Random seed for reproducibility
/// * `idx` - Multi-dimensional index array (e.g., [i, j, k] for a 3D tensor)
/// * `pole` - DLR pole value used for scaling
///
/// # Returns
/// Random coefficient in range [-sqrt(|pole|), sqrt(|pole|))
fn random_pole_coeff(seed: u64, idx: &[usize], pole: f64) -> f64 {
    // Compute a unique flat index from multi-dimensional indices
    let mut index = 0u64;
    for &dim_val in idx.iter() {
        index = index.wrapping_mul(1000).wrapping_add(dim_val as u64);
    }
    // Generate random value in [0, 1]
    let mut x = seed.wrapping_add(index);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let random_val = (x as f64) / (u64::MAX as f64);
    // Scale to [-sqrt(|pole|), sqrt(|pole|))
    (2.0 * random_val - 1.0) * pole.abs().sqrt()
}

/// Create a tensor filled with random DLR coefficients
///
/// This function creates a multi-dimensional tensor with random coefficients
/// generated using `random_pole_coeff` over a 2D [n_poles, n_rest] index space and lays it out with the pole dimension at
/// the requested `target_dim`.
///
/// # Arguments
/// * `n_poles` - Number of DLR poles (size of the target dimension)
/// * `extra_dims` - Extra dimensions beyond the sampling dimension (e.g., [2, 3, 4] for 4D)
/// * `seed` - Random seed for reproducibility
/// * `poles` - Array of DLR pole values (length must match n_poles)
/// * `target_dim` - Dimension index along which poles are applied
///
/// # Returns
/// A `TypedTensor<f64>` filled with random coefficients
fn create_random_dlr_coeffs(
    n_poles: usize,
    extra_dims: &[usize],
    seed: u64,
    poles: &[f64],
    target_dim: usize,
) -> TypedTensor<f64> {
    // Compute full dimensions using get_dims
    let dims = get_dims(n_poles, extra_dims, target_dim);
    let ndim = dims.len();
    assert!(ndim >= 1, "dims must have at least one dimension");
    assert!(target_dim < ndim, "invalid target_dim");

    let total_size: usize = dims.iter().product();

    // The random generator uses the index [i, j] of the pole index and the
    // "rest" index (column-major over the remaining axes), so the values do
    // not depend on where the pole axis sits.
    let pre: usize = dims[..target_dim].iter().product();
    let data = (0..total_size)
        .map(|lin| {
            let p = lin % pre;
            let i = (lin / pre) % n_poles;
            let q = lin / (pre * n_poles);
            random_pole_coeff(seed, &[i, p + pre * q], poles[i])
        })
        .collect();
    TypedTensor::from_vec_col_major(dims, data).unwrap()
}

/// Run the integration example for a specific configuration
///
/// This function demonstrates the complete DLR/IR/sampling cycle with multiple round-trip tests:
///
/// ## Round-trip tests performed:
///
/// 1. **DLR ↔ IR conversion** (Steps 5 & 9):
///    - DLR coefficients → IR coefficients (Step 5)
///    - IR coefficients → DLR coefficients (Step 9)
///    - Verifies that DLR and IR representations are equivalent
///
/// 2. **tau → IR → Matsubara** (Step 8):
///    - Start from Green's function values on tau grid (g_tau_ir)
///    - Fit to recover IR coefficients (tau → IR)
///    - Evaluate recovered IR coefficients on Matsubara grid (IR → Matsubara)
///    - Compares with original Matsubara values to verify consistency
///
/// 3. **Cross-representation evaluation** (Steps 6 & 7):
///    - Evaluates Green's function on tau grid from both IR and DLR coefficients
///    - Evaluates Green's function on Matsubara grid from both IR and DLR coefficients
///    - Verifies that both representations produce identical results
///
/// ## Parameters:
///
/// * `beta` - Inverse temperature
/// * `omega_max` - Maximum frequency cutoff
/// * `epsilon` - Target accuracy for basis construction
/// * `tol` - Tolerance for error comparisons
/// * `extra_dims` - Extra dimensions beyond the sampling dimension (e.g., [2, 3, 4] for 4D)
/// * `target_dim` - Dimension along which to perform transformations (0-indexed)
/// * `positive_only` - If true, only use non-negative Matsubara frequencies
/// * `kernel` - Pre-computed kernel (shared across all tests)
/// * `sve` - Pre-computed SVE result (shared across all tests)
fn run_integration_example_single<K, S>(
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    tol: f64,
    extra_dims: &[usize],
    target_dim: usize,
    positive_only: bool,
    kernel: &K,
    sve: &SVEResult,
) where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
    S: StatisticsType + 'static,
{
    let ndim = 1 + extra_dims.len();
    let stat_name = if S::STATISTICS == sparse_ir::traits::Statistics::Fermionic {
        "Fermionic"
    } else {
        "Bosonic"
    };
    println!("========================================");
    println!(
        "Integration Example ({}D, target_dim={}, {}, positive_only={})",
        ndim, target_dim, stat_name, positive_only
    );
    println!("========================================");
    println!("Parameters:");
    println!("  beta = {}", beta);
    println!("  omega_max = {}", omega_max);
    println!("  epsilon = {}", epsilon);
    println!("  tolerance = {}", tol);
    println!("  extra_dims = {:?}", extra_dims);
    println!("  target_dim = {}", target_dim);
    println!("  statistics = {}", stat_name);
    println!("  positive_only = {}", positive_only);
    println!();

    // Step 1: Create basis from pre-computed SVE
    println!("Step 1: Creating kernel and IR basis...");
    let basis = FiniteTempBasis::<_, S>::from_sve_result(
        kernel.clone(),
        beta,
        sve.clone(),
        Some(epsilon),
        None,
    );
    let basis_size = basis.size();
    println!("  Basis size: {}", basis_size);
    println!();

    // Step 2: Create tau and Matsubara sampling
    println!("Step 2: Creating sampling objects...");
    let tau_points = basis.default_tau_sampling_points();
    let n_tau = tau_points.len();
    println!("  Number of tau points: {}", n_tau);
    let tau_sampling = TauSampling::<S>::with_sampling_points(&basis, tau_points.clone());

    let matsubara_points = basis.default_matsubara_sampling_points(positive_only);
    let n_matsubara = matsubara_points.len();
    println!("  Number of Matsubara points: {}", n_matsubara);
    let matsubara_sampling =
        MatsubaraSampling::<S>::with_sampling_points(&basis, matsubara_points.clone());
    println!();

    // Step 3: Create DLR from IR basis
    println!("Step 3: Creating DLR representation...");
    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis).expect("failed to build DLR");
    let n_poles = dlr.poles.len();
    println!("  Number of DLR poles: {}", n_poles);
    println!();

    // Step 4: Generate random DLR coefficients
    println!("Step 4: Generating random DLR coefficients...");
    // Create N-dimensional tensor for DLR coefficients with target_dim at specified position.
    let seed = 982743u64;
    let dlr_coeffs = create_random_dlr_coeffs(n_poles, extra_dims, seed, &dlr.poles, target_dim);
    println!(
        "  Generated DLR coefficients with shape: {:?}",
        dlr_coeffs.shape()
    );
    println!();

    // Step 5: Convert DLR to IR
    println!("Step 5: Converting DLR coefficients to IR...");
    let ir_coeffs = dlr.to_ir_nd(None, &dlr_coeffs, target_dim).unwrap();
    println!("  IR coefficients shape: {:?}", ir_coeffs.shape());
    println!();

    // Step 6: Evaluate on tau grid from both DLR and IR
    println!("Step 6: Evaluating on tau grid...");
    // From IR coefficients
    let g_tau_ir = tau_sampling
        .evaluate_nd(None, &ir_coeffs, target_dim)
        .unwrap();
    println!("  g_tau_ir shape: {:?}", g_tau_ir.shape());

    // From DLR coefficients (evaluate DLR basis functions at tau points)
    // Use Basis trait to call evaluate_tau
    let dlr_u_tau = <DiscreteLehmannRepresentation<S> as Basis<S>>::evaluate_tau(&dlr, &tau_points);
    // For multi-dimensional case, we need to evaluate DLR at each tau point
    // and then contract with DLR coefficients along the target_dim
    let g_tau_dlr = contract_along_dim(&dlr_u_tau, &dlr_coeffs, target_dim);

    // Compare
    let tau_error = max_relative_error_real(&g_tau_ir, &g_tau_dlr);
    println!("  Max relative error (IR vs DLR on tau): {:.2e}", tau_error);
    if tau_error > tol {
        println!("  WARNING: Error exceeds tolerance!");
    }
    println!();

    // Step 7: Evaluate on Matsubara grid from both DLR and IR
    println!("Step 7: Evaluating on Matsubara grid...");
    // From IR coefficients: evaluate_nd now accepts f64 directly
    let g_iw_ir = matsubara_sampling
        .evaluate_nd(None, &ir_coeffs, target_dim)
        .unwrap();
    println!("  g_iw_ir shape: {:?}", g_iw_ir.shape());

    // From DLR coefficients (evaluate DLR basis functions at Matsubara frequencies)
    // Use Basis trait to call evaluate_matsubara
    let dlr_uhat_matsu =
        <DiscreteLehmannRepresentation<S> as Basis<S>>::evaluate_matsubara(&dlr, &matsubara_points);
    // For multi-dimensional case, similar to tau evaluation
    // Convert real DLR coefficients to complex for matrix multiplication
    let dlr_coeffs_complex = to_complex(&dlr_coeffs);
    let g_iw_dlr = contract_along_dim(&dlr_uhat_matsu, &dlr_coeffs_complex, target_dim);

    // Compare
    let matsubara_error = max_relative_error_complex(&g_iw_ir, &g_iw_dlr);
    println!(
        "  Max relative error (IR vs DLR on Matsubara): {:.2e}",
        matsubara_error
    );
    if matsubara_error > tol {
        println!("  WARNING: Error exceeds tolerance!");
    }
    println!();

    // Step 8: Round-trip test: tau → IR → Matsubara
    println!("Step 8: Round-trip test (tau → IR → Matsubara)...");
    // Fit IR coefficients directly from g_tau_ir (values on tau grid)
    let ir_coeffs_recovered = tau_sampling.fit_nd(None, &g_tau_ir, target_dim).unwrap();
    println!(
        "  Recovered IR coefficients shape: {:?}",
        ir_coeffs_recovered.shape()
    );

    // Compare recovered IR coefficients with original
    let ir_recovery_error = max_relative_error_real(&ir_coeffs, &ir_coeffs_recovered);
    println!(
        "  Max relative error (IR recovery): {:.2e}",
        ir_recovery_error
    );
    if ir_recovery_error > tol {
        println!("  WARNING: Error exceeds tolerance!");
    }

    // Now evaluate recovered IR coefficients on Matsubara grid
    // Cast recovered real IR coefficients to complex tensor element-wise
    let ir_coeffs_recovered_complex = to_complex(&ir_coeffs_recovered);
    let g_iw_ir_reconst = matsubara_sampling
        .evaluate_nd(None, &ir_coeffs_recovered_complex, target_dim)
        .unwrap();

    // Compare with original g_iw_ir
    let roundtrip_error = max_relative_error_complex(&g_iw_ir, &g_iw_ir_reconst);
    println!(
        "  Max relative error (Matsubara round-trip): {:.2e}",
        roundtrip_error
    );
    if roundtrip_error > tol {
        println!("  WARNING: Error exceeds tolerance!");
    }
    println!();

    // Step 9: Round-trip test: DLR → IR → DLR
    println!("Step 9: Round-trip test (DLR → IR → DLR)...");
    let dlr_coeffs_recovered = dlr.from_ir_nd(None, &ir_coeffs, target_dim).unwrap();
    let dlr_recovery_error = max_relative_error_real(&dlr_coeffs, &dlr_coeffs_recovered);
    println!(
        "  Max relative error (DLR recovery): {:.2e}",
        dlr_recovery_error
    );
    if dlr_recovery_error > tol {
        println!("  WARNING: Error exceeds tolerance!");
    }
    println!();

    println!("========================================");
    println!(
        "Summary ({}D, target_dim={}, {}, positive_only={}):",
        ndim, target_dim, stat_name, positive_only
    );
    println!("  Tau evaluation error (IR vs DLR): {:.2e}", tau_error);
    println!(
        "  Matsubara evaluation error (IR vs DLR): {:.2e}",
        matsubara_error
    );
    println!("  IR recovery error (tau fit): {:.2e}", ir_recovery_error);
    println!("  Matsubara round-trip error: {:.2e}", roundtrip_error);
    println!("  DLR recovery error: {:.2e}", dlr_recovery_error);
    println!("========================================");
}

/// Run integration examples for multiple configurations
///
/// This function uses a unified nested loop structure similar to the C++ test,
/// iterating over all combinations of:
/// - Statistics type (Fermionic, Bosonic)
/// - positive_only (false, true)
/// - extra_dims ({}, {2, 3, 4})
/// - target_dim (conditional range based on extra_dims)
fn run_integration_example(beta: f64, omega_max: f64, epsilon: f64, tol: f64) {
    // Create kernel once for all tests
    let lambda = beta * omega_max;
    let kernel = LogisticKernel::new(lambda);

    // Create SVE once for all tests
    println!();
    println!(
        "Testing with beta = {}, omega_max = {}, epsilon = {}",
        beta, omega_max, epsilon
    );
    println!("Computing SVE for all tests");
    let sve = compute_sve(kernel.clone(), epsilon, None, None, TworkType::Auto);
    println!("SVE computed");
    println!();

    // Unified nested loop structure for all test combinations
    // Iterate over statistics types
    run_integration_example_for_stat::<LogisticKernel, Fermionic>(
        beta, omega_max, epsilon, tol, &kernel, &sve,
    );
    run_integration_example_for_stat::<LogisticKernel, Bosonic>(
        beta, omega_max, epsilon, tol, &kernel, &sve,
    );
}

/// Run integration examples for RegularizedBoseKernel (Bosonic only)
///
/// This function uses a unified nested loop structure similar to the C++ test,
/// but only tests Bosonic statistics since RegularizedBoseKernel does not
/// support Fermionic statistics.
fn run_integration_example_regularized_bose(beta: f64, omega_max: f64, epsilon: f64, tol: f64) {
    // Create kernel once for all tests.
    //
    // IMPORTANT:
    //   For very large lambda (e.g., 1e5), the even/odd reduced kernels become
    //   numerically almost identical. This makes the SVE basis for even/odd
    //   sectors nearly the same, so default_omega_sampling_points() can return
    //   effectively duplicate sampling points (poles < basis_size), which
    //   breaks DLR construction.
    //
    //   To avoid this, we keep lambda at a moderate value (1e2) for
    //   RegularizedBoseKernel tests.
    let _lambda_physical = beta * omega_max;
    let lambda = 1e2;
    let kernel = RegularizedBoseKernel::new(lambda);

    // Create SVE once for all tests
    println!();
    println!(
        "Testing RegularizedBoseKernel with beta = {}, omega_max = {}, lambda = {}, epsilon = {}",
        beta, omega_max, lambda, epsilon
    );
    println!("Computing SVE for all tests");
    let sve = compute_sve(kernel.clone(), epsilon, None, None, TworkType::Auto);
    println!("SVE computed");
    println!();

    // Unified nested loop structure for all test combinations
    // Only test Bosonic statistics (RegularizedBoseKernel does not support Fermionic)
    run_integration_example_for_stat::<RegularizedBoseKernel, Bosonic>(
        beta, omega_max, epsilon, tol, &kernel, &sve,
    );
}

/// Run integration examples for a specific statistics type
fn run_integration_example_for_stat<K, S>(
    beta: f64,
    omega_max: f64,
    epsilon: f64,
    tol: f64,
    kernel: &K,
    sve: &SVEResult,
) where
    K: CentrosymmKernel + KernelProperties + Clone + 'static,
    S: StatisticsType + 'static,
{
    for positive_only in [false, true] {
        println!("positive_only = {}", positive_only);

        // Iterate over extra_dims configurations
        let extra_dims_configs: Vec<Vec<usize>> = vec![vec![], vec![2, 3, 4]];
        for extra_dims in extra_dims_configs {
            let ndim = 1 + extra_dims.len();

            // Determine target_dim range based on extra_dims
            let target_dim_start = 0;
            let target_dim_end = if extra_dims.is_empty() { 0 } else { ndim - 1 };

            for target_dim in target_dim_start..=target_dim_end {
                run_integration_example_single::<K, S>(
                    beta,
                    omega_max,
                    epsilon,
                    tol,
                    &extra_dims,
                    target_dim,
                    positive_only,
                    kernel,
                    sve,
                );
                println!();
            }
        }
    }
}

fn main() {
    // Use parameters similar to the C++ test
    let beta = 1e4;
    let omega_max = 2.0;
    let epsilon = 1e-10;
    let tol = 10.0 * epsilon;

    run_integration_example(beta, omega_max, epsilon, tol);

    // Also test RegularizedBoseKernel (Bosonic only) with moderate lambda.
    //
    // For very large lambda, the even/odd reduced kernels become almost
    // identical, which makes the even/odd SVE bases nearly the same.
    // In that regime, default_omega_sampling_points() can return effectively
    // duplicate sampling points so that the number of distinct poles is
    // smaller than the basis size and DLR construction fails.
    //
    // To avoid this pathology we keep lambda = 1e2 for this test, regardless
    // of beta * omega_max.
    run_integration_example_regularized_bose(beta, omega_max, epsilon, tol);
}
