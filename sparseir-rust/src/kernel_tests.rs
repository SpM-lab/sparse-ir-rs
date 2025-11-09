use super::*;
use crate::traits::{Bosonic, Fermionic};
use dashu_base::Approximation;
use dashu_float::{Context, DBig, round::mode::HalfAway};
use std::str::FromStr;
use crate::Df64;

// Configuration for precision tests
const DBIG_DIGITS: usize = 100;
const TOLERANCE_F64: f64 = 1e-12;
const TOLERANCE_TWOFLOAT: f64 = 1e-12; // TODO: MAKE IT TIGHTER

/// Trait for kernel computation using high-precision DBig arithmetic
trait KernelDbigCompute {
    /// Compute kernel value using DBig for high-precision reference
    fn compute_dbig(lambda: DBig, x: DBig, y: DBig, ctx: &Context<HalfAway>) -> DBig;
}

/// Convert f64 to DBig with high precision
fn f64_to_dbig(val: f64, precision: usize) -> DBig {
    let val_str = format!("{:.17e}", val);
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
        // K(x, y) = y * exp(-Λ y (x + 1) / 2) / (1 - exp(-Λ y))

        // Special case: y ≈ 0 → K(x, 0) = 1/Λ (independent of x)
        let y_f64 = extract_f64(y.to_f64());
        let lambda_f64 = extract_f64(lambda.to_f64());
        if y_f64.abs() < 1e-100 {
            // For very small y, use the limit: lim_{y→0} K(x,y) = 1/Λ
            return f64_to_dbig(1.0 / lambda_f64, ctx.precision());
        }

        let one = f64_to_dbig(1.0, ctx.precision());
        let two = f64_to_dbig(2.0, ctx.precision());

        let exponent = -lambda.clone() * y.clone() * (x + one.clone()) / two;
        let numerator = y.clone() * exponent.exp();
        let denominator = one - (-lambda * y.clone()).exp();

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

    // Compare
    let diff = (result_t_f64 - result_dbig_f64).abs();
    let rel_error = if result_dbig_f64.abs() > 1e-300 {
        diff / result_dbig_f64.abs()
    } else {
        diff
    };

    assert!(
        rel_error < tolerance,
        "{}: precision test failed for lambda={}, x={}, y={}\n  Expected: {}\n  Got: {}\n  Relative error: {}",
        kernel_name,
        lambda,
        x,
        y,
        result_dbig_f64,
        result_t_f64,
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

    // Compare
    let diff = (result_t_f64 - result_dbig_f64).abs();
    let rel_error = if result_dbig_f64.abs() > 1e-300 {
        diff / result_dbig_f64.abs()
    } else {
        diff
    };

    assert!(
        rel_error < tolerance,
        "{}: precision test failed for x={}, y={}\n  Expected: {}\n  Got: {}\n  Relative error: {}",
        kernel_name,
        x,
        y,
        result_dbig_f64,
        result_t_f64,
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

    // Compare
    let diff = (result_t_f64 - result_dbig_f64).abs();
    let rel_error = if result_dbig_f64.abs() > 1e-300 {
        diff / result_dbig_f64.abs()
    } else {
        diff
    };

    assert!(
        rel_error < tolerance,
        "{}: compute_reduced precision test failed for lambda={}, x={}, y={}, symmetry={:?}\n  Expected: {}\n  Got: {}\n  Relative error: {}",
        kernel_name,
        lambda,
        x,
        y,
        symmetry,
        result_dbig_f64,
        result_t_f64,
        rel_error
    );
}

/// Generic test helper for different lambda values
fn test_kernel_precision_different_lambdas<F>(
    kernel_constructor: F,
    kernel_name: &str,
    test_points: &[(f64, f64)],
) where
    F: Fn(f64) -> LogisticKernel,
{
    let lambdas = [10.0, 1e2, 1e3, 1e4];
    for &lambda in &lambdas {
        let kernel = kernel_constructor(lambda);
        for &(x, y) in test_points {
            test_kernel_compute_precision_generic::<LogisticKernel, f64>(
                &kernel, lambda, x, y, TOLERANCE_F64, kernel_name,
            );
            test_kernel_compute_precision_generic::<LogisticKernel, Df64>(
                &kernel, lambda, x, y, TOLERANCE_TWOFLOAT, kernel_name,
            );
        }
    }
}

// Simple non-centrosymmetric kernel for testing
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

impl KernelProperties for SimpleNonCentrosymmKernel {
    type SVEHintsType<T> = SimpleNonCentrosymmSVEHints<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn ypower(&self) -> i32 {
        0
    }

    fn conv_radius(&self) -> f64 {
        std::f64::INFINITY
    }

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn weight<S: StatisticsType + 'static>(&self, _beta: f64, _omega: f64) -> f64 {
        1.0
    }

    fn inv_weight<S: StatisticsType + 'static>(&self, _beta: f64, _omega: f64) -> f64 {
        1.0
    }

    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        SimpleNonCentrosymmSVEHints::new(*self, epsilon)
    }
}

#[derive(Debug, Clone)]
struct SimpleNonCentrosymmSVEHints<T> {
    kernel: SimpleNonCentrosymmKernel,
    epsilon: f64,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> SimpleNonCentrosymmSVEHints<T>
where
    T: Copy + Debug + Send + Sync,
{
    pub fn new(kernel: SimpleNonCentrosymmKernel, epsilon: f64) -> Self {
        Self {
            kernel,
            epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T> SVEHints<T> for SimpleNonCentrosymmSVEHints<T>
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        // Return full domain [-1, 1] for non-centrosymmetric kernel
        vec![
            T::from_f64_unchecked(-1.0),
            T::from_f64_unchecked(0.0),
            T::from_f64_unchecked(1.0),
        ]
    }

    fn segments_y(&self) -> Vec<T> {
        // Return full domain [-1, 1] for non-centrosymmetric kernel
        vec![
            T::from_f64_unchecked(-1.0),
            T::from_f64_unchecked(0.0),
            T::from_f64_unchecked(1.0),
        ]
    }

    fn nsvals(&self) -> usize {
        10
    }

    fn ngauss(&self) -> usize {
        if self.epsilon >= 1e-8 {
            10
        } else {
            16
        }
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

#[test]
fn test_noncentrosymm_kernel_sve_hints() {
    let kernel = SimpleNonCentrosymmKernel::new(10.0);
    let hints = kernel.sve_hints::<f64>(1e-6);

    let segments_x = hints.segments_x();
    let segments_y = hints.segments_y();

    // Should include negative values for non-centrosymmetric kernel
    assert!(segments_x[0] < 0.0);
    assert!(segments_y[0] < 0.0);
    assert!(segments_x.last().unwrap() > &0.0);
    assert!(segments_y.last().unwrap() > &0.0);

    assert!(hints.nsvals() > 0);
    assert!(hints.ngauss() > 0);
}

#[test]
fn test_compute_sve_general_noncentrosymm() {
    use crate::sve::{compute_sve_general, TworkType};

    let kernel = SimpleNonCentrosymmKernel::new(10.0);
    let epsilon = 1e-6;

    let result = compute_sve_general(
        kernel,
        epsilon,
        None,
        None,
        TworkType::Auto,
    );

    // Verify result is valid
    assert!(result.s.len() > 0);
    assert!(result.u.get_polys().len() > 0);
    assert!(result.v.get_polys().len() > 0);

    // Singular values should be non-negative and decreasing
    for i in 0..result.s.len() - 1 {
        assert!(result.s[i] >= result.s[i + 1]);
    }
}

#[test]
fn test_matrix_from_gauss_noncentrosymmetric() {
    use crate::kernelmatrix::matrix_from_gauss_noncentrosymmetric;
    use crate::gauss::legendre_generic;

    let kernel = SimpleNonCentrosymmKernel::new(10.0);
    let hints = kernel.sve_hints::<f64>(1e-6);

    let segments_x = hints.segments_x();
    let segments_y = hints.segments_y();

    let rule = legendre_generic::<f64>(10);
    let gauss_x = rule.piecewise(&segments_x);
    let gauss_y = rule.piecewise(&segments_y);

    let discretized = matrix_from_gauss_noncentrosymmetric(
        &kernel,
        &gauss_x,
        &gauss_y,
        &hints,
    );

    // Verify matrix dimensions
    assert_eq!(discretized.matrix.shape().0, gauss_x.x.len());
    assert_eq!(discretized.matrix.shape().1, gauss_y.x.len());

    // Verify some matrix values (for K(x, y) = x + y)
    for i in 0..gauss_x.x.len().min(5) {
        for j in 0..gauss_y.x.len().min(5) {
            let expected = gauss_x.x[i] + gauss_y.x[j];
            let actual = discretized.matrix[[i, j]];
            assert!((actual - expected).abs() < 1e-10,
                "Matrix value mismatch at [{}, {}]: expected {}, got {}", i, j, expected, actual);
        }
    }
}
