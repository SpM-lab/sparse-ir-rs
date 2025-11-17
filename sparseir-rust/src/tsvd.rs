//! High-precision truncated SVD implementation using nalgebra
//!
//! This module provides QR + SVD based truncated SVD decomposition
//! with support for extended precision arithmetic.

use nalgebra::{DMatrix, DVector, ComplexField, RealField};
use nalgebra::linalg::ColPivQR;
use num_traits::{Zero, One, ToPrimitive};
use crate::Df64;
use crate::numeric::CustomNumeric;
use mdarray::DTensor;

/// Result of SVD decomposition
#[derive(Debug, Clone)]
pub struct SVDResult<T> {
    /// Left singular vectors (m × rank)
    pub u: DMatrix<T>,
    /// Singular values (rank)
    pub s: DVector<T>,
    /// Right singular vectors (n × rank)
    pub v: DMatrix<T>,
    /// Effective rank
    pub rank: usize,
}

/// Configuration for TSVD computation
#[derive(Debug, Clone)]
pub struct TSVDConfig<T> {
    /// Relative tolerance for rank determination
    pub rtol: T,
}

impl<T> TSVDConfig<T> {
    pub fn new(rtol: T) -> Self {
        Self { rtol }
    }
}

/// Error types for TSVD computation
#[derive(Debug, thiserror::Error)]
pub enum TSVDError {
    #[error("Matrix is empty")]
    EmptyMatrix,
    #[error("Invalid tolerance: {0}")]
    InvalidTolerance(String),
}

/// Get appropriate epsilon value for SVD convergence based on type
///
/// Returns the machine epsilon (EPSILON constant) for the given type.
/// This is preferred over approx::AbsDiffEq::default_epsilon() because
/// the latter may return MIN_POSITIVE for Df64, which is too small and
/// causes excessive iterations in SVD.
#[inline]
fn get_epsilon_for_svd<T: RealField + Copy>() -> T {
    use std::any::TypeId;
    
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // f64::EPSILON ≈ 2.22e-16
        unsafe { std::ptr::read(&f64::EPSILON as *const f64 as *const T) }
    } else if TypeId::of::<T>() == TypeId::of::<crate::Df64>() {
        // Df64::EPSILON ≈ 2.465e-32
        unsafe { std::ptr::read(&crate::Df64::EPSILON as *const crate::Df64 as *const T) }
    } else {
        // Fallback: use a reasonable default
        T::from_f64(1e-15).unwrap_or(T::one() * T::from_f64(1e-15).unwrap_or(T::one()))
    }
}

/// Perform SVD decomposition using nalgebra with sorted singular values
///
/// Note: This computes ALL singular values, not a truncated SVD.
/// The truncation happens after SVD computation based on rtol.
///
/// # Arguments
/// * `matrix` - Input matrix (m × n)
/// * `rtol` - Relative tolerance for rank determination (used for rank calculation, not SVD convergence)
///
/// # Returns
/// * `SVDResult` - Truncated SVD result with U, S, V matrices and rank
pub fn svd_decompose<T>(matrix: &DMatrix<T>, rtol: f64) -> SVDResult<T>
where
    T: ComplexField + RealField + Copy + nalgebra::RealField + ToPrimitive,
{
    // Use type-appropriate epsilon for SVD convergence
    // For f64: f64::EPSILON (約 2.22e-16)
    // For Df64: Df64::EPSILON (約 2.465e-32)
    // Note: We use EPSILON (machine epsilon) instead of default_epsilon() (from approx trait)
    // because default_epsilon() may return MIN_POSITIVE for Df64, which is too small.
    let eps = get_epsilon_for_svd::<T>();
    
    // Perform FULL SVD decomposition with explicit epsilon
    // try_svd automatically sorts singular values in descending order
    // max_niter = 0 means iterate indefinitely until convergence
    let svd = matrix
        .clone()
        .try_svd(true, true, eps, 0)
        .expect("SVD computation failed");

    // Extract U, S, V matrices (already sorted by nalgebra)
    let u_matrix = svd.u.unwrap();
    let s_vector = svd.singular_values;  // Sorted in descending order
    let v_t_matrix = svd.v_t.unwrap();

    // Calculate effective rank from sorted singular values
    // Early termination is possible in rank calculation because values are sorted
    let rank = calculate_rank_from_vector(&s_vector, rtol);

    // Convert to thin SVD (truncate to effective rank)
    let u = DMatrix::from(u_matrix.columns(0, rank));
    let s = DVector::from(s_vector.rows(0, rank));
    let v = DMatrix::from(v_t_matrix.rows(0, rank).transpose());

    SVDResult {
        u,
        s,
        v,
        rank,
    }
}

/// Calculate effective rank from sorted singular values
///
/// # Arguments
/// * `singular_values` - Vector of singular values (sorted in descending order)
/// * `rtol` - Relative tolerance for rank determination
///
/// # Returns
/// * `usize` - Effective rank
///
/// # Note
/// Since singular values are sorted in descending order by try_svd,
/// this function can terminate early when a value below the threshold is found.
fn calculate_rank_from_vector<T>(singular_values: &DVector<T>, rtol: f64) -> usize
where
    T: RealField + Copy + ToPrimitive,
{
    if singular_values.is_empty() {
        return 0;
    }

    // First element is the maximum (sorted in descending order)
    let max_sv = singular_values[0];
    let threshold = max_sv * T::from_f64(rtol).unwrap_or(T::zero());

    let mut rank = 0;
    for &sv in singular_values.iter() {
        if sv > threshold {
            rank += 1;
        } else {
            // Early termination: since values are sorted, all remaining values are also below threshold
            break;
        }
    }

    rank
}

/// Calculate rank from R matrix diagonal elements
fn calculate_rank_from_r<T: RealField>(
    r_matrix: &DMatrix<T>,
    rtol: T,
) -> usize
where
    T: ComplexField + RealField + Copy,
{
    let dim = r_matrix.nrows().min(r_matrix.ncols());
    let mut rank = dim;

    // Find the maximum diagonal element
    let mut max_diag_abs = Zero::zero();
    for i in 0..dim {
        let diag_abs = ComplexField::abs(r_matrix[(i, i)]);
        if diag_abs > max_diag_abs {
            max_diag_abs = diag_abs;
        }
    }

    // If max_diag_abs is zero, rank is zero
    if max_diag_abs == Zero::zero() {
        return 0;
    }

    // Check each diagonal element
    for i in 0..dim {
        let diag_abs = ComplexField::abs(r_matrix[(i, i)]);

        // Check if the diagonal element is too small relative to the maximum diagonal element
        if diag_abs < rtol * max_diag_abs {
            rank = i;
            break;
        }
    }

    rank
}

/// Main TSVD function using QR + SVD approach
///
/// Computes the truncated SVD using the algorithm:
/// 1. Apply QR decomposition to A to get Q and R
/// 2. Compute SVD of R
/// 3. Reconstruct final U and V matrices
///
/// # Arguments
/// * `matrix` - Input matrix (m × n)
/// * `config` - TSVD configuration
///
/// # Returns
/// * `SVDResult` - Truncated SVD result
pub fn tsvd<T>(
    matrix: &DMatrix<T>,
    config: TSVDConfig<T>,
) -> Result<SVDResult<T>, TSVDError>
where
    T: ComplexField + RealField + Copy + nalgebra::RealField + std::fmt::Debug + ToPrimitive + CustomNumeric,
{
    let (m, n) = matrix.shape();

    if m == 0 || n == 0 {
        return Err(TSVDError::EmptyMatrix);
    }

    if config.rtol <= Zero::zero() || config.rtol >= One::one() {
        return Err(TSVDError::InvalidTolerance(format!(
            "Tolerance must be in (0, 1), got {:?}",
            config.rtol
        )));
    }

    // Step 1: Apply QR decomposition to A using nalgebra
    let qr = ColPivQR::new(matrix.clone());
    let q_matrix = qr.q();
    let r_matrix = qr.r();
    let permutation = qr.p();

    // Step 2: Apply QR-based rank estimation first
    // Use type-specific epsilon for QR diagonal elements (more conservative than rtol)
    let qr_rank = calculate_rank_from_r(&r_matrix, T::from_f64_unchecked(2.0) * get_epsilon_for_svd::<T>());
    
    if qr_rank == 0 {
        // Matrix has zero rank
        return Ok(SVDResult {
            u: DMatrix::zeros(m, 0),
            s: DVector::zeros(0),
            v: DMatrix::zeros(n, 0),
            rank: 0,
        });
    }

    // Step 3: Truncate R to estimated rank and apply SVD
    let r_truncated: DMatrix<T> = r_matrix.rows(0, qr_rank).into();
    // Use rtol directly as T
    let rtol_t = config.rtol;
    let rtol_f64 = rtol_t.to_f64();
    let svd_result = svd_decompose(&r_truncated, rtol_f64);
    
    if svd_result.rank == 0 {
        // Matrix has zero rank
        return Ok(SVDResult {
            u: DMatrix::zeros(m, 0),
            s: DVector::zeros(0),
            v: DMatrix::zeros(n, 0),
            rank: 0,
        });
    }

    // Step 4: Reconstruct full SVD
    // U = Q * U_R (Q is (m, qr_rank), U_R is (qr_rank, svd_result.rank))
    let q_truncated: DMatrix<T> = q_matrix.columns(0, qr_rank).into();
    let u_full = &q_truncated * &svd_result.u;

    // V = P^T * V_R (apply inverse permutation matrix)
    // Since A*P = Q*R, we have A = Q*R*P^T
    // After SVD of R: A = Q*U_R*S_R*V_R^T*P^T = U*S*V^T
    // where V^T = V_R^T*P^T, so V = P^(-1)*V_R = P^T*V_R
    let mut v_full = svd_result.v.clone();
    permutation.inv_permute_rows(&mut v_full);

    // S_full = S_R (already correct size)
    let s_full = svd_result.s.clone();

    Ok(SVDResult {
        u: u_full,
        s: s_full,
        v: v_full,
        rank: svd_result.rank,
    })
}

/// Convenience function for f64 TSVD
pub fn tsvd_f64(matrix: &DMatrix<f64>, rtol: f64) -> Result<SVDResult<f64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

/// Convenience function for Df64 TSVD
pub fn tsvd_df64(matrix: &DMatrix<Df64>, rtol: Df64) -> Result<SVDResult<Df64>, TSVDError> {
    tsvd(matrix, TSVDConfig::new(rtol))
}

/// Convenience function for Df64 TSVD from f64 matrix
pub fn tsvd_df64_from_f64(matrix: &DMatrix<f64>, rtol: f64) -> Result<SVDResult<Df64>, TSVDError> {
    let matrix_df64 = DMatrix::from_fn(matrix.nrows(), matrix.ncols(), |i, j| {
        Df64::from(matrix[(i, j)])
    });
    let rtol_df64 = Df64::from(rtol);
    tsvd(&matrix_df64, TSVDConfig::new(rtol_df64))
}

/// Compute SVD for DTensor using nalgebra-based TSVD
///
/// Supports both f64 and Df64 types. Uses nalgebra TSVD backend for both.
pub fn compute_svd_dtensor<T: CustomNumeric + 'static>(
    matrix: &DTensor<T, 2>,
) -> (DTensor<T, 2>, Vec<T>, DTensor<T, 2>) {
    use std::any::TypeId;
    use nalgebra::DMatrix;

    // Dispatch based on type: convert to appropriate DMatrix type
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // Convert to DMatrix<f64>
        let matrix_f64 = DMatrix::from_fn(matrix.shape().0, matrix.shape().1, |i, j| {
            CustomNumeric::to_f64(matrix[[i, j]])
        });

        // Use TSVD with appropriate tolerance for f64
        let rtol = 2.0 * f64::EPSILON;
        let result = tsvd(&matrix_f64, TSVDConfig::new(rtol)).expect("TSVD computation failed");


        // Convert back to DTensor<T>
        let u = DTensor::<T, 2>::from_fn([result.u.nrows(), result.u.ncols()], |idx| {
            let [i, j] = [idx[0], idx[1]];
            T::from_f64_unchecked(result.u[(i, j)])
        });

        let s: Vec<T> = result.s.iter().map(|x| T::from_f64_unchecked(*x)).collect();

        let v = DTensor::<T, 2>::from_fn([result.v.nrows(), result.v.ncols()], |idx| {
            let [i, j] = [idx[0], idx[1]];
            T::from_f64_unchecked(result.v[(i, j)])
        });

        (u, s, v)
    } else if TypeId::of::<T>() == TypeId::of::<Df64>() {
        // Convert to DMatrix<Df64> without going through f64 to preserve precision
        // TypeId check ensures T == Df64 at runtime, so we can safely cast
        let matrix_df64: DMatrix<Df64> = DMatrix::from_fn(matrix.shape().0, matrix.shape().1, |i, j| {
            // Safe: TypeId check guarantees T == Df64
            unsafe { std::mem::transmute_copy(&matrix[[i, j]]) }
        });

        // Use TSVD with appropriate tolerance for Df64
        let rtol = Df64::from(2.0) * Df64::epsilon();
        let result = tsvd_df64(&matrix_df64, rtol).expect("TSVD computation failed");

        // Convert back to DTensor<T> without going through f64 to preserve Df64 precision
        let u = DTensor::<T, 2>::from_fn([result.u.nrows(), result.u.ncols()], |idx| {
            let [i, j] = [idx[0], idx[1]];
            T::convert_from(result.u[(i, j)])
        });

        let s: Vec<T> = result
            .s
            .iter()
            .map(|x| T::convert_from(*x))
            .collect();

        let v = DTensor::<T, 2>::from_fn([result.v.nrows(), result.v.ncols()], |idx| {
            let [i, j] = [idx[0], idx[1]];
            T::convert_from(result.v[(i, j)])
        });

        (u, s, v)
    } else {
        panic!("SVD is only implemented for f64 and Df64");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use num_traits::cast::ToPrimitive;

    #[test]
    fn test_svd_identity_matrix() {
        let matrix = DMatrix::<f64>::identity(3, 3);
        let result = svd_decompose(&matrix, 1e-12);
        
        assert_eq!(result.rank, 3);
        assert_eq!(result.s.len(), 3);
        assert_eq!(result.u.nrows(), 3);
        assert_eq!(result.u.ncols(), 3);
        assert_eq!(result.v.nrows(), 3);
        assert_eq!(result.v.ncols(), 3);
    }

    #[test]
    fn test_tsvd_identity_matrix() {
        let matrix = DMatrix::<f64>::identity(3, 3);
        let result = tsvd_f64(&matrix, 1e-12).unwrap();
        
        assert_eq!(result.rank, 3);
        assert_eq!(result.s.len(), 3);
    }

    #[test]
    fn test_tsvd_rank_one() {
        let matrix = DMatrix::<f64>::from_fn(3, 3, |i, j| (i + 1) as f64 * (j + 1) as f64);
        let result = tsvd_f64(&matrix, 1e-12).unwrap();
        
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn test_tsvd_empty_matrix() {
        let matrix = DMatrix::<f64>::zeros(0, 0);
        let result = tsvd_f64(&matrix, 1e-12);
        
        assert!(matches!(result, Err(TSVDError::EmptyMatrix)));
    }

    /// Create Hilbert matrix of size n x n with generic type
    /// H[i,j] = 1 / (i + j + 1)
    fn create_hilbert_matrix_generic<T>(n: usize) -> DMatrix<T>
    where
        T: nalgebra::RealField + From<f64> + Copy + std::ops::Div<Output = T>,
    {
        DMatrix::from_fn(n, n, |i, j| {
            // For high precision types like Df64, we need to do the division in type T
            // to preserve precision, not in f64
            T::one() / T::from((i + j + 1) as f64)
        })
    }

    /// Reconstruct matrix from SVD with generic type: A = U * S * V^T
    fn reconstruct_matrix_generic<T>(
        u: &DMatrix<T>,
        s: &nalgebra::DVector<T>,
        v: &DMatrix<T>,
    ) -> DMatrix<T>
    where
        T: nalgebra::RealField + Copy,
    {
        // A = U * S * V^T
        // U: (m × k), S: (k), V: (n × k)
        // Result: (m × n)
        u * &DMatrix::from_diagonal(s) * &v.transpose()
    }

    /// Calculate Frobenius norm of matrix with generic type
    fn frobenius_norm_generic<T>(matrix: &DMatrix<T>) -> f64
    where
        T: nalgebra::RealField + Copy + ToPrimitive,
    {
        let mut sum = 0.0;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let val = matrix[(i, j)].to_f64().unwrap_or(0.0);
                sum += val * val;
            }
        }
        sum.sqrt()
    }

    /// Generic Hilbert matrix reconstruction test
    fn test_hilbert_reconstruction_generic<T>(n: usize, rtol: f64, expected_max_error: f64)
    where
        T: nalgebra::RealField + From<f64> + Copy + ToPrimitive + std::fmt::Debug + crate::numeric::CustomNumeric,
    {
        let h = create_hilbert_matrix_generic::<T>(n);

        // Compute TSVD with specified tolerance
        let config = TSVDConfig::new(T::from(rtol));
        let result = tsvd(&h, config).unwrap();

        // Reconstruct matrix
        let h_reconstructed = reconstruct_matrix_generic(&result.u, &result.s, &result.v);
        
        // Calculate reconstruction error (in the same type T to preserve precision)
        let error_matrix = &h - &h_reconstructed;
        let error_norm = frobenius_norm_generic(&error_matrix);
        let relative_error = error_norm / frobenius_norm_generic(&h);

        // Check that reconstruction error is within expected bounds
        assert!(relative_error <= expected_max_error, 
                "Relative reconstruction error {} exceeds expected maximum {}", 
                relative_error, expected_max_error);
    }

    #[test]
    fn test_hilbert_5x5_f64_reconstruction() {
        test_hilbert_reconstruction_generic::<f64>(5, 1e-12, 1e-14);
    }

    #[test]
    fn test_hilbert_5x5_df64_reconstruction() {
        test_hilbert_reconstruction_generic::<Df64>(5, 1e-28, 1e-28);
    }

    #[test]
    fn test_hilbert_10x10_f64_reconstruction() {
        test_hilbert_reconstruction_generic::<f64>(10, 1e-12, 1e-12);
    }

    #[test]
    fn test_hilbert_10x10_df64_reconstruction() {
        // Note: 10x10 Hilbert matrix has very large condition number (~1e13)
        // Even with Df64, reconstruction is limited by nalgebra's matrix operations
        // which may not fully utilize Df64's precision in intermediate calculations
        test_hilbert_reconstruction_generic::<Df64>(10, 1e-28, 1e-30);
    }

    #[test]
    fn test_hilbert_100x100_f64_reconstruction() {
        // Large matrix test with f64 - expect reasonable performance
        test_hilbert_reconstruction_generic::<f64>(100, 1e-12, 1e-12);
    }

    #[test]
    fn test_hilbert_100x100_df64_reconstruction() {
        // Large matrix test with Df64 - expect high precision but longer execution time
        test_hilbert_reconstruction_generic::<Df64>(100, 1e-28, 1e-28);
    }
}
