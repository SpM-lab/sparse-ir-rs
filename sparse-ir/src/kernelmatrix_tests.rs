//! Tests for kernel matrix discretization and interpolation

use crate::Df64;
use crate::gauss::legendre;
use crate::kernel::{
    CentrosymmKernel, KernelProperties, LogisticKernel, LogisticSVEHints, RegularizedBoseKernel,
    RegularizedBoseSVEHints, SVEHints, SymmetryType,
};
use crate::kernelmatrix::{InterpolatedKernel, matrix_from_gauss};
use crate::numeric::CustomNumeric;

// ========================================================================
// Basic matrix_from_gauss tests
// ========================================================================

#[test]
fn test_matrix_from_gauss_basic() {
    // 2x2の小さな行列で基本動作確認
    let kernel = LogisticKernel::new(1.0);
    let gauss_x = legendre::<f64>(2).reseat(0.0, 1.0);
    let gauss_y = legendre::<f64>(2).reseat(0.0, 1.0);
    let matrix = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);

    assert_eq!(matrix.matrix.shape().0, 2);
    assert_eq!(matrix.matrix.shape().1, 2);
}

#[test]
fn test_matrix_from_gauss_sizes() {
    let kernel = LogisticKernel::new(1.0);

    for n in [2, 4, 8] {
        let gauss_x = legendre::<f64>(n).reseat(0.0, 1.0);
        let gauss_y = legendre::<f64>(n).reseat(0.0, 1.0);
        let matrix = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);

        assert_eq!(matrix.matrix.shape().0, n);
        assert_eq!(matrix.matrix.shape().1, n);
    }
}

// ========================================================================
// Kernel interpolation tests
// ========================================================================

/// Generic test for kernel interpolation precision
///
/// This test creates a discretized kernel and interpolated kernel, then
/// evaluates interpolation error at various points within each segment.
///
/// Generic over:
/// - T: Numeric type (f64, Df64, etc.)
/// - K: Kernel type (LogisticKernel, RegularizedBoseKernel, etc.)
/// - H: SVEHints type for the kernel
fn test_kernel_interpolation_precision_generic<T, K>(
    kernel: K,
    hints: impl SVEHints<T>,
    kernel_name: &str,
    epsilon: f64,
    tolerance_abs: f64,
    tolerance_rel: f64,
    symmetry_type: SymmetryType,
) where
    T: CustomNumeric + Clone + std::fmt::Debug + Send + Sync + 'static,
    K: CentrosymmKernel + KernelProperties + Clone,
{
    // Create segments for discretization
    let segments_x: Vec<T> = hints.segments_x();
    let segments_y: Vec<T> = hints.segments_y();

    println!("\n=== {} Interpolation Test ===", kernel_name);
    println!("Lambda = {}, Epsilon = {:.0e}", kernel.lambda(), epsilon);
    println!(
        "Segments: {}x{}",
        segments_x.len() - 1,
        segments_y.len() - 1
    );

    // Create interpolated kernel directly from kernel and segments
    let gauss_per_cell = hints.ngauss(); // Get polynomial degree from SVE hints

    println!("Gauss points per cell: {}", gauss_per_cell);

    // Create interpolated kernel directly (no composite Gauss rules needed)
    let interpolated = InterpolatedKernel::from_kernel_and_segments(
        &kernel,
        segments_x.clone(),
        segments_y.clone(),
        gauss_per_cell,
        symmetry_type,
    );

    println!(
        "Interpolated kernel created: {}x{} cells",
        interpolated.n_cells_x(),
        interpolated.n_cells_y()
    );

    // Test points within each segment (relative coordinates)
    let test_points_relative = vec![(0.1, 0.2), (0.4, 0.3), (0.7, 0.6), (0.9, 0.8)];

    let mut max_error: T = T::from_f64_unchecked(0.0);
    let mut max_rel_error: T = T::from_f64_unchecked(0.0);
    let mut total_tests = 0;

    // Test each cell
    for i in 0..interpolated.n_cells_x() {
        for j in 0..interpolated.n_cells_y() {
            let x_min = interpolated.segments_x[i];
            let x_max = interpolated.segments_x[i + 1];
            let y_min = interpolated.segments_y[j];
            let y_max = interpolated.segments_y[j + 1];

            // Test points in this cell
            for &(rel_x, rel_y) in &test_points_relative {
                let x = x_min + T::from_f64_unchecked(rel_x) * (x_max - x_min);
                let y = y_min + T::from_f64_unchecked(rel_y) * (y_max - y_min);

                // Skip if point is outside kernel domain
                if x > T::from_f64_unchecked(kernel.xmax())
                    || y > T::from_f64_unchecked(kernel.ymax())
                {
                    continue;
                }

                // Direct kernel evaluation with specified symmetry type
                let direct_value: T = kernel.compute_reduced(x, y, symmetry_type);

                // Interpolated value
                let interpolated_value: T = interpolated.evaluate(x, y);

                let error: T = (direct_value - interpolated_value).abs_as_same_type();
                let rel_error = if direct_value.abs_as_same_type() > T::from_f64_unchecked(1e-12) {
                    error / direct_value.abs_as_same_type()
                } else {
                    error
                };

                max_error = max_error.max(error);
                max_rel_error = max_rel_error.max(rel_error);
                total_tests += 1;
            }
        }
    }

    println!("Max absolute error: {:.6e}", max_error.to_f64());
    println!("Max relative error: {:.6e}", max_rel_error.to_f64());
    println!("Total test points: {}", total_tests);
    println!(
        "Tolerance: abs={:.0e}, rel={:.0e}",
        tolerance_abs, tolerance_rel
    );

    assert!(
        max_error < T::from_f64_unchecked(tolerance_abs),
        "Max absolute error {:.6e} exceeds tolerance {:.0e}",
        max_error.to_f64(),
        tolerance_abs
    );
    assert!(
        max_rel_error < T::from_f64_unchecked(tolerance_rel),
        "Max relative error {:.6e} exceeds tolerance {:.0e}",
        max_rel_error.to_f64(),
        tolerance_rel
    );
    assert!(total_tests > 0, "Should have at least some test points");
}

// ========================================================================
// Unified kernel interpolation test framework
// ========================================================================

/// Test kernel interpolation for both Even and Odd symmetries
fn test_kernel_interpolation_both_symmetries<T, K, H>(
    kernel: K,
    hints_factory: impl Fn(K, f64) -> H,
    kernel_name: &str,
    epsilon: f64,
    tolerance_abs_even: f64,
    tolerance_rel_even: f64,
    tolerance_abs_odd: f64,
    tolerance_rel_odd: f64,
) where
    T: CustomNumeric + Clone + std::fmt::Debug + Send + Sync + 'static,
    K: CentrosymmKernel + KernelProperties + Clone,
    H: SVEHints<T>,
{
    let hints = hints_factory(kernel.clone(), epsilon);

    // Test Even symmetry
    test_kernel_interpolation_precision_generic::<T, _>(
        kernel.clone(),
        hints_factory(kernel.clone(), epsilon),
        &format!("{} Even", kernel_name),
        epsilon,
        tolerance_abs_even,
        tolerance_rel_even,
        SymmetryType::Even,
    );

    // Test Odd symmetry
    test_kernel_interpolation_precision_generic::<T, _>(
        kernel,
        hints,
        &format!("{} Odd", kernel_name),
        epsilon,
        tolerance_abs_odd,
        tolerance_rel_odd,
        SymmetryType::Odd,
    );
}

// ========================================================================
// LogisticKernel interpolation tests
// ========================================================================

/// Test LogisticKernel interpolation precision with f64
#[test]
fn test_logistic_kernel_interpolation_f64() {
    test_kernel_interpolation_both_symmetries::<f64, _, _>(
        LogisticKernel::new(100.0),
        LogisticSVEHints::new,
        "LogisticKernel (f64)",
        1e-12, // epsilon
        1e-12, // tolerance_abs_even
        1e-10, // tolerance_rel_even
        1e-12, // tolerance_abs_odd
        1e-10, // tolerance_rel_odd
    );
}

/// Test LogisticKernel interpolation precision with Df64
#[test]
fn test_logistic_kernel_interpolation_twofloat() {
    test_kernel_interpolation_both_symmetries::<Df64, _, _>(
        LogisticKernel::new(100.0),
        LogisticSVEHints::new,
        "LogisticKernel (Df64)",
        1e-12, // epsilon
        1e-11, // tolerance_abs_even
        1e-10, // tolerance_rel_even
        1e-11, // tolerance_abs_odd
        1e-10, // tolerance_rel_odd
    );
}

// ========================================================================
// RegularizedBoseKernel interpolation tests
// ========================================================================

/// Test RegularizedBoseKernel interpolation precision with f64
#[test]
fn test_regularized_bose_kernel_interpolation_f64() {
    test_kernel_interpolation_both_symmetries::<f64, _, _>(
        RegularizedBoseKernel::new(10.0),
        RegularizedBoseSVEHints::new,
        "RegularizedBoseKernel (f64)",
        1e-4,  // epsilon
        1e-12, // tolerance_abs_even (excellent precision)
        1e-10, // tolerance_rel_even
        1e-12, // tolerance_abs_odd (should be same as even)
        1e-10, // tolerance_rel_odd (should be same as even)
    );
}

/// Test RegularizedBoseKernel interpolation precision with Df64
#[test]
fn test_regularized_bose_kernel_interpolation_twofloat() {
    test_kernel_interpolation_both_symmetries::<Df64, _, _>(
        RegularizedBoseKernel::new(10.0),
        RegularizedBoseSVEHints::new,
        "RegularizedBoseKernel (Df64)",
        1e-4,  // epsilon
        1e-11, // tolerance_abs_even
        1e-10, // tolerance_rel_even
        1e-11, // tolerance_abs_odd
        1e-10, // tolerance_rel_odd
    );
}
