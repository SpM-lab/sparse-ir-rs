use super::*;
use dashu_base::Approximation;
use dashu_float::{Context, DBig, round::mode::HalfAway};
use std::str::FromStr;
use crate::Df64;

// Configuration for precision tests
const DBIG_DIGITS: usize = 100;
const TOLERANCE_F64: f64 = 1e-15;
const TOLERANCE_DF64: f64 = 1e-30;

/// Trait for kernel computation using high-precision DBig arithmetic
trait KernelDbigCompute {
    /// Compute kernel value using DBig for high-precision reference
    fn compute_dbig(lambda: DBig, x: DBig, y: DBig, ctx: &Context<HalfAway>) -> DBig;
}

/// Convert f64 to DBig with high precision
fn f64_to_dbig(val: f64, precision: usize) -> DBig {
    let val_str = format!("{:.30e}", val);
    DBig::from_str(&val_str)
        .unwrap()
        .with_precision(precision)
        .unwrap()
}

/// Extract f64 from Approximation
fn extract_f64(approx: Approximation<f64, dashu_float::round::Rounding>) -> f64 {
    match approx {
        Approximation::Exact(val) => val,
        Approximation::Inexact(val, _) => val,
    }
}

// Implement KernelDbigCompute for LogisticKernel
impl KernelDbigCompute for LogisticKernel {
    fn compute_dbig(lambda: DBig, x: DBig, y: DBig, ctx: &Context<HalfAway>) -> DBig {
        // K(x, y) = exp(-Λy(x + 1)/2) / (1 + exp(-Λy))
        let one = f64_to_dbig(1.0, ctx.precision());
        let two = f64_to_dbig(2.0, ctx.precision());

        let numerator = (-lambda.clone() * y.clone() * (x + one.clone()) / two).exp();
        let denominator = one + (-lambda * y).exp();

        numerator / denominator
    }
}

// Implement KernelDbigCompute for RegularizedBoseKernel
impl KernelDbigCompute for RegularizedBoseKernel {
    fn compute_dbig(lambda: DBig, x: DBig, y: DBig, ctx: &Context<HalfAway>) -> DBig {
        // K(x, y) = y * exp(-Λy(x+1)/2) / (1 - exp(-Λy))
        let one = f64_to_dbig(1.0, ctx.precision());
        let half = f64_to_dbig(0.5, ctx.precision());

        // Handle y ≈ 0 using Taylor expansion: K(x,y) = 1/Λ - xy/2 + O(y²)
        let y_f64 = extract_f64(y.to_f64());
        if y_f64.abs() < 1e-100 {
            let term0 = one.clone() / lambda.clone();
            let term1 = half * x.clone() * y.clone();
            return term0 - term1;
        }
        
        // exp(-Λy(x+1)/2)
        let exp_arg = -lambda.clone() * y.clone() * (x.clone() + one.clone()) * half;
        let numerator = y.clone() * exp_arg.exp();
        
        // 1 - exp(-Λy)
        let exp_neg_lambda_y = (-lambda * y.clone()).exp();
        let denominator = one - exp_neg_lambda_y;
        
        // K(x, y) = numerator / denominator
        numerator / denominator
    }
}

/// Generic test for kernel computation precision
///
/// Compares kernel implementation against high-precision DBig reference.
///
/// # Type Parameters
/// * `K` - Kernel type implementing CentrosymmKernel and KernelDbigCompute
/// * `T` - Numeric type to test (f64, Df64, etc.)
fn test_kernel_compute_precision_generic<K, T>(
    kernel: &K,
    lambda: f64,
    x_dd: Df64,
    y_dd: Df64,
    tolerance: f64,
    kernel_name: &str,
) where
    K: CentrosymmKernel + KernelDbigCompute,
    T: CustomNumeric,
{
    // Convert inputs to type T
    // For f64, convert from Df64; for Df64, use directly
    let x_t: T = T::convert_from(x_dd);
    let y_t: T = T::convert_from(y_dd);
    let result_t = kernel.compute(x_t, y_t);

    // DBig version (high precision reference)
    // Convert Df64 to DBig using BigFloat conversion pattern
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    let lambda_dbig = f64_to_dbig(lambda, ctx.precision());
    // Convert Df64 to DBig: use the full precision value
    let x_dbig = {
        let x_str = format!("{:.30e}", x_dd.hi() + x_dd.lo());
        DBig::from_str(&x_str)
            .unwrap()
            .with_precision(ctx.precision())
            .unwrap()
    };
    let y_dbig = {
        let y_str = format!("{:.30e}", y_dd.hi() + y_dd.lo());
        DBig::from_str(&y_str)
            .unwrap()
            .with_precision(ctx.precision())
            .unwrap()
    };
    let result_dbig = K::compute_dbig(lambda_dbig, x_dbig, y_dbig, &ctx);

    // Convert DBig result to f64
    let result_dbig_f64 = extract_f64(result_dbig.to_f64());
    let result_t_f64 = result_t.to_f64();

    // Compare using absolute error
    let diff = (result_t_f64 - result_dbig_f64).abs();
    let rel_error = if result_dbig_f64.abs() > 1e-300 {
        diff / result_dbig_f64.abs()
    } else {
        diff
    };

    let x_val = x_dd.hi() + x_dd.lo();
    let y_val = y_dd.hi() + y_dd.lo();
    assert!(
        diff < tolerance,
        "{}: precision test failed for lambda={}, x={}, y={}\n  Expected: {}\n  Got: {}\n  Absolute error: {} (tolerance: {})\n  Relative error: {}",
        kernel_name,
        lambda,
        x_val,
        y_val,
        result_dbig_f64,
        result_t_f64,
        diff,
        tolerance,
        rel_error
    );
}

/// Generic test for kernel computation precision (non-centrosymmetric)
fn test_kernel_compute_precision_generic_noncentrosymm<K, T>(
    kernel: &K,
    x_dd: Df64,
    y_dd: Df64,
    tolerance: f64,
    kernel_name: &str,
) where
    K: AbstractKernel + KernelDbigCompute,
    T: CustomNumeric,
{
    // Convert inputs to type T
    // For f64, convert from Df64; for Df64, use directly
    let x_t: T = T::convert_from(x_dd);
    let y_t: T = T::convert_from(y_dd);
    let result_t = kernel.compute(x_t, y_t);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    // For non-centrosymmetric kernels, we need lambda from kernel properties
    // This is a simplified test - in practice, non-centrosymmetric kernels
    // might not have lambda
    let lambda_dbig = f64_to_dbig(10.0, ctx.precision()); // Default lambda for test
    // Convert Df64 to DBig: use the full precision value
    let x_dbig = {
        let x_str = format!("{:.30e}", x_dd.hi() + x_dd.lo());
        DBig::from_str(&x_str)
            .unwrap()
            .with_precision(ctx.precision())
            .unwrap()
    };
    let y_dbig = {
        let y_str = format!("{:.30e}", y_dd.hi() + y_dd.lo());
        DBig::from_str(&y_str)
            .unwrap()
            .with_precision(ctx.precision())
            .unwrap()
    };
    let result_dbig = K::compute_dbig(lambda_dbig, x_dbig, y_dbig, &ctx);

    // Convert DBig result to f64
    let result_dbig_f64 = extract_f64(result_dbig.to_f64());
    let result_t_f64 = result_t.to_f64();

    // Compare using absolute error
    let diff = (result_t_f64 - result_dbig_f64).abs();
    let rel_error = if result_dbig_f64.abs() > 1e-300 {
        diff / result_dbig_f64.abs()
    } else {
        diff
    };

    let x_val = x_dd.hi() + x_dd.lo();
    let y_val = y_dd.hi() + y_dd.lo();
    assert!(
        diff < tolerance,
        "{}: precision test failed for x={}, y={}\n  Expected: {}\n  Got: {}\n  Absolute error: {} (tolerance: {})\n  Relative error: {}",
        kernel_name,
        x_val,
        y_val,
        result_dbig_f64,
        result_t_f64,
        diff,
        tolerance,
        rel_error
    );
}

/// High-precision compute_reduced implementation using DBig (generic)
fn compute_reduced_dbig<K: KernelDbigCompute>(
    lambda: DBig,
    x: DBig,
    y: DBig,
    symmetry: SymmetryType,
    ctx: &Context<HalfAway>,
) -> DBig {
    match symmetry {
        SymmetryType::Even => {
            // K(x, y) + K(x, -y)
            let k_plus = K::compute_dbig(lambda.clone(), x.clone(), y.clone(), ctx);
            let k_minus = K::compute_dbig(lambda, x, -y, ctx);
            k_plus + k_minus
        }
        SymmetryType::Odd => {
            // K(x, y) - K(x, -y)
            let k_plus = K::compute_dbig(lambda.clone(), x.clone(), y.clone(), ctx);
            let k_minus = K::compute_dbig(lambda, x, -y, ctx);
            k_plus - k_minus
        }
    }
}

/// Generic test for compute_reduced precision
fn test_kernel_compute_reduced_precision_generic<K, T>(
    kernel: &K,
    lambda: f64,
    x_dd: Df64,
    y_dd: Df64,
    symmetry: SymmetryType,
    tolerance: f64,
    kernel_name: &str,
) where
    K: CentrosymmKernel + KernelDbigCompute,
    T: CustomNumeric,
{
    // Convert inputs to type T
    // For f64, convert from Df64; for Df64, use directly
    let x_t: T = T::convert_from(x_dd);
    let y_t: T = T::convert_from(y_dd);
    let result_t = kernel.compute_reduced(x_t, y_t, symmetry);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    let lambda_dbig = f64_to_dbig(lambda, ctx.precision());
    // Convert Df64 to DBig: use the full precision value
    let x_dbig = {
        let x_str = format!("{:.30e}", x_dd.hi() + x_dd.lo());
        DBig::from_str(&x_str)
            .unwrap()
            .with_precision(ctx.precision())
            .unwrap()
    };
    let y_dbig = {
        let y_str = format!("{:.30e}", y_dd.hi() + y_dd.lo());
        DBig::from_str(&y_str)
            .unwrap()
            .with_precision(ctx.precision())
            .unwrap()
    };
    let result_dbig = compute_reduced_dbig::<K>(lambda_dbig, x_dbig, y_dbig, symmetry, &ctx);

    // Convert DBig result to f64
    let result_dbig_f64 = extract_f64(result_dbig.to_f64());
    let result_t_f64 = result_t.to_f64();

    // Compare using absolute error
    let diff = (result_t_f64 - result_dbig_f64).abs();
    let rel_error = if result_dbig_f64.abs() > 1e-300 {
        diff / result_dbig_f64.abs()
    } else {
        diff
    };

    let x_val = x_dd.hi() + x_dd.lo();
    let y_val = y_dd.hi() + y_dd.lo();
    assert!(
        diff < tolerance,
        "{}: compute_reduced precision test failed for lambda={}, x={}, y={}, symmetry={:?}\n  Expected: {}\n  Got: {}\n  Absolute error: {} (tolerance: {})\n  Relative error: {}",
        kernel_name,
        lambda,
        x_val,
        y_val,
        symmetry,
        result_dbig_f64,
        result_t_f64,
        diff,
        tolerance,
        rel_error
    );
}

/// Generic test helper for different lambda values
/// Tests both compute() and compute_reduced() for Even and Odd symmetries
/// Works with any CentrosymmKernel that implements KernelDbigCompute
fn test_kernel_precision_different_lambdas<K, F, T>(
    kernel_constructor: F,
    kernel_name: &str,
    test_points: &[(T, T)],
    lambdas: &[f64],
) where
    K: CentrosymmKernel + KernelDbigCompute + Clone,
    F: Fn(f64) -> K,
    T: CustomNumeric + Copy,
{
    for &lambda in lambdas {
        let kernel = kernel_constructor(lambda);
        for &(x, y) in test_points {
            // Convert test points to Df64 (as they would be stored in Gauss quadrature rules)
            let x_dd = Df64::convert_from(x);
            let y_dd = Df64::convert_from(y);
            
            // Test compute() precision
            test_kernel_compute_precision_generic::<K, f64>(
                &kernel, lambda, x_dd, y_dd, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_precision_generic::<K, Df64>(
                &kernel, lambda, x_dd, y_dd, TOLERANCE_DF64, kernel_name,
            );
            
            // Test compute_reduced() precision for Even symmetry
            test_kernel_compute_reduced_precision_generic::<K, f64>(
                &kernel, lambda, x_dd, y_dd, SymmetryType::Even, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_reduced_precision_generic::<K, Df64>(
                &kernel, lambda, x_dd, y_dd, SymmetryType::Even, TOLERANCE_DF64, kernel_name,
            );
            
            // Test compute_reduced() precision for Odd symmetry
            test_kernel_compute_reduced_precision_generic::<K, f64>(
                &kernel, lambda, x_dd, y_dd, SymmetryType::Odd, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_reduced_precision_generic::<K, Df64>(
                &kernel, lambda, x_dd, y_dd, SymmetryType::Odd, TOLERANCE_DF64, kernel_name,
            );
        }
    }
}

/// Test kernel precision at critical points that revealed the double-double precision bug
/// This test specifically targets the issue where abs_as_same_type was losing precision
#[test]
fn test_logistic_kernel_precision_critical_points() {
    // Critical test points that revealed the precision bug
    // x = y = 0.01 was the point where the bug was discovered
    // Store test points as Df64 (as they would be stored in Gauss quadrature rules)
    let test_points_dd: [(Df64, Df64); 10] = [
        (Df64::from(0.0), Df64::from(0.0)),      // Origin
        (Df64::from(0.01), Df64::from(0.01)),    // Small values (bug discovery point)
        (Df64::from(0.1), Df64::from(0.1)),      // Medium-small values
        (Df64::from(0.5), Df64::from(0.5)),      // Medium values
        (Df64::from(0.9), Df64::from(0.9)),      // Near boundary
        (Df64::from(0.01), Df64::from(0.1)),     // Asymmetric small x
        (Df64::from(0.1), Df64::from(0.01)),     // Asymmetric small y
        (Df64::from(0.99), Df64::from(0.99)),    // Very near boundary (symmetric)
        (Df64::from(0.99), Df64::from(0.01)),    // Very near boundary (asymmetric large x)
        (Df64::from(0.01), Df64::from(0.99)),    // Very near boundary (asymmetric large y)
    ];
    
    let lambdas = [10.0, 1e2, 1e3, 1e4];
    
    // Test with Df64 test points
    test_kernel_precision_different_lambdas::<LogisticKernel, _, Df64>(
        |lambda| LogisticKernel::new(lambda),
        "LogisticKernel",
        &test_points_dd,
        &lambdas,
    );
    
    // Also test with f64 test points for completeness
    // Convert from Df64 to f64
    let test_points_f64: [(f64, f64); 10] = test_points_dd.map(|(x_dd, y_dd)| {
        (x_dd.hi() + x_dd.lo(), y_dd.hi() + y_dd.lo())
    });
    
    test_kernel_precision_different_lambdas::<LogisticKernel, _, f64>(
        |lambda| LogisticKernel::new(lambda),
        "LogisticKernel",
        &test_points_f64,
        &lambdas,
    );
}

/// Test RegularizedBoseKernel precision at critical points
/// Uses the same generic test framework to ensure consistency
#[test]
#[ignore]
fn test_regularized_bose_kernel_precision_critical_points() {
    // Store test points as Df64 (as they would be stored in Gauss quadrature rules)
    let test_points_dd: [(Df64, Df64); 10] = [
        (Df64::from(0.0), Df64::from(0.0)),      // Origin
        (Df64::from(0.01), Df64::from(0.01)),    // Small values
        (Df64::from(0.1), Df64::from(0.1)),      // Medium-small values
        (Df64::from(0.5), Df64::from(0.5)),      // Medium values
        (Df64::from(0.9), Df64::from(0.9)),      // Near boundary
        (Df64::from(0.01), Df64::from(0.1)),     // Asymmetric small x
        (Df64::from(0.1), Df64::from(0.01)),     // Asymmetric small y
        (Df64::from(0.99), Df64::from(0.99)),    // Very near boundary (symmetric)
        (Df64::from(0.99), Df64::from(0.01)),    // Very near boundary (asymmetric large x)
        (Df64::from(0.01), Df64::from(0.99)),    // Very near boundary (asymmetric large y)
    ];
    
    let lambdas = [10.0, 1e2, 1e3];
    
    // Test with Df64 test points
    test_kernel_precision_different_lambdas::<RegularizedBoseKernel, _, Df64>(
        |lambda| RegularizedBoseKernel::new(lambda),
        "RegularizedBoseKernel",
        &test_points_dd,
        &lambdas,
    );
    
    // Also test with f64 test points for completeness
    // Convert from Df64 to f64
    let test_points_f64: [(f64, f64); 10] = test_points_dd.map(|(x_dd, y_dd)| {
        (x_dd.hi() + x_dd.lo(), y_dd.hi() + y_dd.lo())
    });
    
    test_kernel_precision_different_lambdas::<RegularizedBoseKernel, _, f64>(
        |lambda| RegularizedBoseKernel::new(lambda),
        "RegularizedBoseKernel",
        &test_points_f64,
        &lambdas,
    );
}

// Simple non-centrosymmetric kernel for testing AbstractKernel trait
// K(x, y) = x + y (clearly non-centrosymmetric: K(-x, -y) = -x - y = -(x + y) != x + y = K(x, y))
#[derive(Debug, Clone, Copy)]
struct SimpleNonCentrosymmKernel {
    lambda: f64,
}

impl SimpleNonCentrosymmKernel {
    fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl AbstractKernel for SimpleNonCentrosymmKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        x + y
    }

    fn is_centrosymmetric(&self) -> bool {
        false
    }
}

#[test]
fn test_noncentrosymm_kernel_is_centrosymmetric() {
    let kernel = SimpleNonCentrosymmKernel::new(10.0);
    assert!(!kernel.is_centrosymmetric());
}

#[test]
fn test_noncentrosymm_kernel_compute() {
    let kernel = SimpleNonCentrosymmKernel::new(10.0);

    // Test basic computation: K(x, y) = x + y
    let result1: f64 = kernel.compute(0.5, 0.5);
    assert!((result1 - 1.0).abs() < 1e-10);

    let result2: f64 = kernel.compute(-0.5, 0.5);
    assert!((result2 - 0.0).abs() < 1e-10);

    // Verify non-centrosymmetry: K(-x, -y) != K(x, y)
    // K(0.5, 0.5) = 1.0, K(-0.5, -0.5) = -1.0
    let result3: f64 = kernel.compute(0.5, 0.5);
    let result4: f64 = kernel.compute(-0.5, -0.5);
    assert!((result3 - result4).abs() > 1e-10); // Should be different (1.0 vs -1.0)
}

