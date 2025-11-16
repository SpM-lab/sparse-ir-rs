use super::*;
use dashu_base::{Abs, Approximation};
use dashu_float::{Context, DBig, round::mode::HalfAway};
use std::str::FromStr;
use crate::Df64;

// Configuration for precision tests
const DBIG_DIGITS: usize = 300;
const TOLERANCE_F64: f64 = 1e-14;
const TOLERANCE_DF64: f64 = 1e-25;

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
        // K(x, y) = -1/Λ * exp(-|v| * (v >= 0 ? u_plus : u_minus)) * (|v| / (exp(-|v|) - 1))
        // where v = Λy, u_plus = (x+1)/2, u_minus = (1-x)/2
        
        let one = f64_to_dbig(1.0, ctx.precision());
        let half = f64_to_dbig(0.5, ctx.precision());
        
        // u_plus = (x + 1) / 2, u_minus = (1 - x) / 2
        let u_plus = (x.clone() + one.clone()) * half.clone();
        let u_minus = (one.clone() - x.clone()) * half.clone();
        
        let v = lambda.clone() * y.clone();
        let absv = v.clone().abs();
        
        // Handle y ≈ 0 using Taylor expansion
        // K(x,y) = 1/Λ - xy/2 + (1/24)Λ(3x² - 1)y² + O(y³)
        // For |Λy| < 2e-14, use first-order approximation
        let absv_f64 = extract_f64(absv.to_f64());
        if absv_f64 < 2e-14 {
            let term0 = one.clone() / lambda.clone();
            let term1 = half.clone() * x.clone() * y.clone();
            return term0 - term1;
        }
        
        // enum_val = exp(-|v| * (v >= 0 ? u_plus : u_minus))
        let enum_val = if extract_f64(v.to_f64()) >= 0.0 {
            (-absv.clone() * u_plus).exp()
        } else {
            (-absv.clone() * u_minus).exp()
        };
        
        // Handle v / (exp(v) - 1) with numerical stability using expm1 pattern
        // denom = absv / (exp(-absv) - 1)
        let exp_neg_absv = (-absv.clone()).exp();
        let denom = absv / (exp_neg_absv - one.clone());
        
        // K(x, y) = -1/Λ * enum_val * denom
        -one / lambda * enum_val * denom
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
    x: f64,
    y: f64,
    tolerance: f64,
    kernel_name: &str,
) where
    K: CentrosymmKernel + KernelDbigCompute,
    T: CustomNumeric,
{
    // Convert inputs to type T
    let x_t: T = T::from_f64_unchecked(x);
    let y_t: T = T::from_f64_unchecked(y);
    let result_t = kernel.compute(x_t, y_t);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    let lambda_dbig = f64_to_dbig(lambda, ctx.precision());
    let x_dbig = f64_to_dbig(x, ctx.precision());
    let y_dbig = f64_to_dbig(y, ctx.precision());
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

    assert!(
        diff < tolerance,
        "{}: precision test failed for lambda={}, x={}, y={}\n  Expected: {}\n  Got: {}\n  Absolute error: {} (tolerance: {})\n  Relative error: {}",
        kernel_name,
        lambda,
        x,
        y,
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
    x: f64,
    y: f64,
    tolerance: f64,
    kernel_name: &str,
) where
    K: AbstractKernel + KernelDbigCompute,
    T: CustomNumeric,
{
    // Convert inputs to type T
    let x_t: T = T::from_f64_unchecked(x);
    let y_t: T = T::from_f64_unchecked(y);
    let result_t = kernel.compute(x_t, y_t);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    // For non-centrosymmetric kernels, we need lambda from kernel properties
    // This is a simplified test - in practice, non-centrosymmetric kernels
    // might not have lambda
    let lambda_dbig = f64_to_dbig(10.0, ctx.precision()); // Default lambda for test
    let x_dbig = f64_to_dbig(x, ctx.precision());
    let y_dbig = f64_to_dbig(y, ctx.precision());
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

    assert!(
        diff < tolerance,
        "{}: precision test failed for x={}, y={}\n  Expected: {}\n  Got: {}\n  Absolute error: {} (tolerance: {})\n  Relative error: {}",
        kernel_name,
        x,
        y,
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
    x: f64,
    y: f64,
    symmetry: SymmetryType,
    tolerance: f64,
    kernel_name: &str,
) where
    K: CentrosymmKernel + KernelDbigCompute,
    T: CustomNumeric,
{
    // Convert inputs to type T
    let x_t: T = T::from_f64_unchecked(x);
    let y_t: T = T::from_f64_unchecked(y);
    let result_t = kernel.compute_reduced(x_t, y_t, symmetry);

    // DBig version (high precision reference)
    let ctx = Context::<HalfAway>::new(DBIG_DIGITS);
    let lambda_dbig = f64_to_dbig(lambda, ctx.precision());
    let x_dbig = f64_to_dbig(x, ctx.precision());
    let y_dbig = f64_to_dbig(y, ctx.precision());
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

    assert!(
        diff < tolerance,
        "{}: compute_reduced precision test failed for lambda={}, x={}, y={}, symmetry={:?}\n  Expected: {}\n  Got: {}\n  Absolute error: {} (tolerance: {})\n  Relative error: {}",
        kernel_name,
        lambda,
        x,
        y,
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
fn test_kernel_precision_different_lambdas<K, F>(
    kernel_constructor: F,
    kernel_name: &str,
    test_points: &[(f64, f64)],
    lambdas: &[f64],
) where
    K: CentrosymmKernel + KernelDbigCompute + Clone,
    F: Fn(f64) -> K,
{
    for &lambda in lambdas {
        let kernel = kernel_constructor(lambda);
        for &(x, y) in test_points {
            // Test compute() precision
            test_kernel_compute_precision_generic::<K, f64>(
                &kernel, lambda, x, y, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_precision_generic::<K, Df64>(
                &kernel, lambda, x, y, TOLERANCE_DF64, kernel_name,
            );
            
            // Test compute_reduced() precision for Even symmetry
            test_kernel_compute_reduced_precision_generic::<K, f64>(
                &kernel, lambda, x, y, SymmetryType::Even, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_reduced_precision_generic::<K, Df64>(
                &kernel, lambda, x, y, SymmetryType::Even, TOLERANCE_DF64, kernel_name,
            );
            
            // Test compute_reduced() precision for Odd symmetry
            test_kernel_compute_reduced_precision_generic::<K, f64>(
                &kernel, lambda, x, y, SymmetryType::Odd, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_reduced_precision_generic::<K, Df64>(
                &kernel, lambda, x, y, SymmetryType::Odd, TOLERANCE_DF64, kernel_name,
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
    let test_points = [
        (0.0, 0.0),      // Origin
        (0.01, 0.01),    // Small values (bug discovery point)
        (0.1, 0.1),      // Medium-small values
        (0.5, 0.5),      // Medium values
        (0.9, 0.9),      // Near boundary
        (0.01, 0.1),     // Asymmetric small x
        (0.1, 0.01),     // Asymmetric small y
    ];
    
    let lambdas = [10.0, 1e2, 1e3, 1e4];
    
    test_kernel_precision_different_lambdas::<LogisticKernel, _>(
        |lambda| LogisticKernel::new(lambda),
        "LogisticKernel",
        &test_points,
        &lambdas,
    );
}

/// Test RegularizedBoseKernel precision at critical points
/// Uses the same generic test framework to ensure consistency
#[test]
#[ignore]
fn test_regularized_bose_kernel_precision_critical_points() {
    let test_points = [
        (0.0, 0.0),      // Origin
        (0.01, 0.01),    // Small values
        (0.1, 0.1),      // Medium-small values
        (0.5, 0.5),      // Medium values
        (0.9, 0.9),      // Near boundary
        (0.01, 0.1),     // Asymmetric small x
        (0.1, 0.01),     // Asymmetric small y
    ];
    
    let lambdas = [10.0, 1e2, 1e3];
    
    test_kernel_precision_different_lambdas::<RegularizedBoseKernel, _>(
        |lambda| RegularizedBoseKernel::new(lambda),
        "RegularizedBoseKernel",
        &test_points,
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

