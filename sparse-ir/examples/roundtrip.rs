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

use mdarray::{DTensor, DynRank, Shape, Tensor, expr};
use num_complex::{Complex, ComplexFloat};
use sparse_ir::{
    Bosonic, DiscreteLehmannRepresentation, Fermionic, FiniteTempBasis, LogisticKernel,
    MatsubaraSampling, RegularizedBoseKernel, TauSampling,
    basis_trait::Basis,
    gemm::matmul_par,
    kernel::{CentrosymmKernel, KernelProperties},
    sampling::movedim,
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
fn max_relative_error<T>(a: &Tensor<T, DynRank>, b: &Tensor<T, DynRank>) -> f64
where
    T: ComplexFloat<Real = f64> + Sub<Output = T>,
{
    // Compute max |a - b|
    let max_diff = expr::fold(expr::zip(a.expr(), b.expr()), 0.0_f64, |acc, (x, y)| {
        acc.max((*x - *y).abs())
    });

    // Compute max |a|
    let max_ref = expr::fold(a.expr(), 0.0_f64, |acc, x| acc.max(x.abs()));

    // Avoid division by zero (behavior similar to C++ helper)
    if max_ref < 1e-15 {
        max_diff
    } else {
        max_diff / max_ref
    }
}

/// Compute maximum relative error between two real tensors
///
/// Convenience wrapper for `max_relative_error` with `f64`.
fn max_relative_error_real(a: &Tensor<f64, DynRank>, b: &Tensor<f64, DynRank>) -> f64 {
    max_relative_error(a, b)
}

/// Compute maximum relative error between two complex tensors
///
/// Convenience wrapper for `max_relative_error` with `Complex<f64>`.
fn max_relative_error_complex(
    a: &Tensor<Complex<f64>, DynRank>,
    b: &Tensor<Complex<f64>, DynRank>,
) -> f64 {
    max_relative_error(a, b)
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
/// tensor with size `n_poles` along `target_dim`.
///
/// # Arguments
/// * `matrix` - 2D transformation matrix [n_points, n_poles]
/// * `coeffs` - N-dimensional tensor of coefficients with size `n_poles` along `target_dim`
/// * `target_dim` - Dimension along which to contract (must have size = n_poles)
///
/// # Returns
/// N-dimensional tensor with size `n_points` along `target_dim` instead of `n_poles`
fn contract_along_dim<T>(
    matrix: &DTensor<T, 2>,
    coeffs: &Tensor<T, DynRank>,
    target_dim: usize,
) -> Tensor<T, DynRank>
where
    T: num_complex::ComplexFloat + faer_traits::ComplexField + num_traits::One + Copy + 'static,
{
    let (n_points, n_poles) = *matrix.shape();
    let rank = coeffs.rank();
    assert!(
        target_dim < rank,
        "target_dim {} must be < rank {}",
        target_dim,
        rank
    );
    assert_eq!(
        coeffs.shape().dim(target_dim),
        n_poles,
        "coeffs.shape().dim({}) = {} must equal n_poles = {}",
        target_dim,
        coeffs.shape().dim(target_dim),
        n_poles
    );

    // 1. Move target dimension to position 0
    let coeffs_dim0 = movedim(coeffs, target_dim, 0);

    // 2. Reshape to 2D: [n_poles, extra_size]
    let extra_size = coeffs_dim0.len() / n_poles;
    let coeffs_2d_dyn = coeffs_dim0.reshape(&[n_poles, extra_size][..]).to_tensor();

    // 3. Convert DynRank to fixed Rank<2> for matmul_par
    let coeffs_2d = DTensor::<T, 2>::from_fn([n_poles, extra_size], |idx| {
        coeffs_2d_dyn[&[idx[0], idx[1]][..]]
    });

    // 4. Matrix multiply: result_2d = matrix @ coeffs_2d
    let result_2d = matmul_par(matrix, &coeffs_2d, None);

    // 5. Reshape back to N-D with n_points at position 0
    let mut result_shape = vec![n_points];
    coeffs_dim0.shape().with_dims(|dims| {
        for i in 1..dims.len() {
            result_shape.push(dims[i]);
        }
    });

    let result_dim0 = result_2d.into_dyn().reshape(&result_shape[..]).to_tensor();

    // 6. Move dimension 0 back to original position target_dim
    movedim(&result_dim0, 0, target_dim)
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
/// generated using `random_pole_coeff`. It first fills a 2D array of shape
/// [n_poles, n_rest], then reshapes and moves the pole dimension to the
/// requested `target_dim`. This avoids the `Tensor::from_fn` bug with
/// `DynRank` shapes.
///
/// # Arguments
/// * `n_poles` - Number of DLR poles (size of the target dimension)
/// * `extra_dims` - Extra dimensions beyond the sampling dimension (e.g., [2, 3, 4] for 4D)
/// * `seed` - Random seed for reproducibility
/// * `poles` - Array of DLR pole values (length must match n_poles)
/// * `target_dim` - Dimension index along which poles are applied
///
/// # Returns
/// A `Tensor<f64, DynRank>` filled with random coefficients
fn create_random_dlr_coeffs(
    n_poles: usize,
    extra_dims: &[usize],
    seed: u64,
    poles: &[f64],
    target_dim: usize,
) -> Tensor<f64, DynRank> {
    // Compute full dimensions using get_dims
    let dims = get_dims(n_poles, extra_dims, target_dim);
    let ndim = dims.len();
    assert!(ndim >= 1, "dims must have at least one dimension");
    assert!(target_dim < ndim, "invalid target_dim");

    // Product of all dimensions and product of the rest (excluding target_dim)
    let total_size: usize = dims.iter().product();
    let n_rest = total_size / n_poles;

    // First create a 2D tensor [n_poles, n_rest] and fill it with random coefficients.
    //
    // The random generator uses the 2D index [i, j] so that the random value
    // depends on the pole index and the "rest" index, but not on the full
    // N-dimensional layout.
    let mut coeffs_2d = DTensor::<f64, 2>::from_elem([n_poles, n_rest], 0.0);
    for i in 0..n_poles {
        let pole = poles[i];
        for j in 0..n_rest {
            let idx_2d = [i, j];
            coeffs_2d[[i, j]] = random_pole_coeff(seed, &idx_2d, pole);
        }
    }

    // Now reshape [n_poles, n_rest] to an N-dimensional tensor where the pole
    // dimension is at position 0: shape_dim0 = [n_poles, extra_dims...].
    let mut shape_dim0 = Vec::with_capacity(ndim);
    shape_dim0.push(n_poles);
    for (k, &d) in dims.iter().enumerate() {
        if k == target_dim {
            continue;
        }
        shape_dim0.push(d);
    }

    let coeffs_dim0 = coeffs_2d.into_dyn().reshape(&shape_dim0[..]).to_tensor();

    // Finally, move the pole dimension from position 0 to target_dim so that
    // the resulting tensor has the desired shape `dims`.
    movedim(&coeffs_dim0, 0, target_dim)
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
    let dlr = DiscreteLehmannRepresentation::<S>::new(&basis);
    let n_poles = dlr.poles.len();
    println!("  Number of DLR poles: {}", n_poles);
    println!();

    // Step 4: Generate random DLR coefficients
    println!("Step 4: Generating random DLR coefficients...");
    // Create N-dimensional tensor for DLR coefficients with target_dim at specified position.
    // We use a dedicated helper to avoid the `Tensor::from_fn` bug with DynRank shapes.
    let seed = 982743u64;
    let dlr_coeffs = create_random_dlr_coeffs(n_poles, extra_dims, seed, &dlr.poles, target_dim);
    println!(
        "  Generated DLR coefficients with shape: {:?}",
        dlr_coeffs.shape().dims()
    );
    println!();

    // Step 5: Convert DLR to IR
    println!("Step 5: Converting DLR coefficients to IR...");
    let ir_coeffs = dlr.to_ir_nd(None, &dlr_coeffs, target_dim);
    println!("  IR coefficients shape: {:?}", ir_coeffs.shape().dims());
    println!();

    // Step 6: Evaluate on tau grid from both DLR and IR
    println!("Step 6: Evaluating on tau grid...");
    // From IR coefficients
    let g_tau_ir = tau_sampling.evaluate_nd(None, &ir_coeffs, target_dim);
    println!("  g_tau_ir shape: {:?}", g_tau_ir.shape().dims());

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
    let g_iw_ir = matsubara_sampling.evaluate_nd(None, &ir_coeffs, target_dim);
    println!("  g_iw_ir shape: {:?}", g_iw_ir.shape().dims());

    // From DLR coefficients (evaluate DLR basis functions at Matsubara frequencies)
    // Use Basis trait to call evaluate_matsubara
    let dlr_uhat_matsu =
        <DiscreteLehmannRepresentation<S> as Basis<S>>::evaluate_matsubara(&dlr, &matsubara_points);
    // For multi-dimensional case, similar to tau evaluation
    // Convert real DLR coefficients to complex for matrix multiplication
    let dlr_coeffs_complex: Tensor<Complex<f64>, DynRank> =
        expr::FromExpression::from_expr(expr::map(dlr_coeffs.expr(), |x| Complex::new(*x, 0.0)));
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
    let ir_coeffs_recovered = tau_sampling.fit_nd(None, &g_tau_ir, target_dim);
    println!(
        "  Recovered IR coefficients shape: {:?}",
        ir_coeffs_recovered.shape().dims()
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
    let ir_coeffs_recovered_complex: Tensor<Complex<f64>, DynRank> =
        expr::FromExpression::from_expr(expr::map(ir_coeffs_recovered.expr(), |x| {
            Complex::new(*x, 0.0)
        }));
    let g_iw_ir_reconst =
        matsubara_sampling.evaluate_nd(None, &ir_coeffs_recovered_complex, target_dim);

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
    let dlr_coeffs_recovered = dlr.from_ir_nd(None, &ir_coeffs, target_dim);
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
