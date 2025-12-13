//! Common utilities for fitters
//!
//! This module contains shared helper functions and SVD structures
//! used by all fitter implementations.

use mdarray::{DTensor, DynRank, Shape, Slice};
use num_complex::Complex;

// ============================================================================
// Permutation helpers
// ============================================================================

/// Generate permutation to move dimension `dim` to position 0
///
/// For example, with rank=4 and dim=2:
/// - Result: [2, 0, 1, 3]
pub(crate) fn make_perm_to_front(rank: usize, dim: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(dim);
    for i in 0..rank {
        if i != dim {
            perm.push(i);
        }
    }
    perm
}

// ============================================================================
// Complex-Real reinterpretation helpers
// ============================================================================

/// Reinterpret a Complex<f64> slice as a f64 view with an extra dimension of size 2
///
/// Complex<f64> array `[d0, d1, ..., dN]` becomes f64 array `[d0, d1, ..., dN, 2]`
/// where the last dimension contains [re, im] pairs.
pub(crate) fn complex_slice_as_real<'a>(
    coeffs: &'a Slice<Complex<f64>, DynRank>,
) -> mdarray::View<'a, f64, DynRank, mdarray::Dense> {
    // Build new shape: [..., 2]
    let mut new_shape: Vec<usize> = Vec::with_capacity(coeffs.rank() + 1);
    coeffs.shape().with_dims(|dims| {
        for d in dims {
            new_shape.push(*d);
        }
    });
    new_shape.push(2);

    unsafe {
        let shape: DynRank = Shape::from_dims(&new_shape[..]);
        let mapping = mdarray::DenseMapping::new(shape);
        mdarray::View::new_unchecked(coeffs.as_ptr() as *const f64, mapping)
    }
}

/// Reinterpret a mutable Complex<f64> slice as a mutable f64 view with an extra dimension of size 2
///
/// Complex<f64> array `[d0, d1, ..., dN]` becomes f64 array `[d0, d1, ..., dN, 2]`
/// where the last dimension contains [re, im] pairs.
#[allow(dead_code)]
pub(crate) fn complex_slice_mut_as_real<'a>(
    out: &'a mut Slice<Complex<f64>, DynRank>,
) -> mdarray::ViewMut<'a, f64, DynRank, mdarray::Dense> {
    // Build new shape: [..., 2]
    let mut new_shape: Vec<usize> = Vec::with_capacity(out.rank() + 1);
    out.shape().with_dims(|dims| {
        for d in dims {
            new_shape.push(*d);
        }
    });
    new_shape.push(2);

    unsafe {
        let shape: DynRank = Shape::from_dims(&new_shape[..]);
        let mapping = mdarray::DenseMapping::new(shape);
        mdarray::ViewMut::new_unchecked(out.as_mut_ptr() as *mut f64, mapping)
    }
}

// ============================================================================
// SVD structures
// ============================================================================

/// SVD decomposition for real matrices
pub(crate) struct RealSVD {
    pub ut: DTensor<f64, 2>, // (min_dim, n_rows) - U^T
    pub s: Vec<f64>,         // (min_dim,)
    pub v: DTensor<f64, 2>,  // (n_cols, min_dim) - V (transpose of V^T)
}

impl RealSVD {
    pub fn new(u: DTensor<f64, 2>, s: Vec<f64>, vt: DTensor<f64, 2>) -> Self {
        // Check dimensions
        let (_, u_cols) = *u.shape();
        let (vt_rows, _) = *vt.shape();
        let min_dim = s.len();

        assert_eq!(
            u_cols, min_dim,
            "u.cols()={} must equal s.len()={}",
            u_cols, min_dim
        );
        assert_eq!(
            vt_rows, min_dim,
            "vt.rows()={} must equal s.len()={}",
            vt_rows, min_dim
        );

        // Create ut and v from u and vt
        let ut = u.transpose().to_tensor(); // (min_dim, n_rows)
        let v = vt.transpose().to_tensor(); // (n_cols, min_dim)

        // Verify v.cols() == s.len() (v.shape().1 is the second dimension, which is min_dim)
        assert_eq!(
            v.shape().1,
            min_dim,
            "v.cols()={} must equal s.len()={}",
            v.shape().1,
            min_dim
        );

        Self { ut, s, v }
    }
}

/// SVD decomposition for complex matrices
pub(crate) struct ComplexSVD {
    pub ut: DTensor<Complex<f64>, 2>, // (min_dim, n_rows) - U^H
    pub s: Vec<f64>,                  // (min_dim,) - singular values are real
    pub v: DTensor<Complex<f64>, 2>,  // (n_cols, min_dim) - V (transpose of V^T)
}

impl ComplexSVD {
    pub fn new(u: DTensor<Complex<f64>, 2>, s: Vec<f64>, vt: DTensor<Complex<f64>, 2>) -> Self {
        // Check dimensions
        let (u_rows, u_cols) = *u.shape();
        let (vt_rows, _) = *vt.shape();
        let min_dim = s.len();

        assert_eq!(
            u_cols, min_dim,
            "u.cols()={} must equal s.len()={}",
            u_cols, min_dim
        );
        assert_eq!(
            vt_rows, min_dim,
            "vt.rows()={} must equal s.len()={}",
            vt_rows, min_dim
        );

        // Create ut (U^H, conjugate transpose) and v from u and vt
        let ut = DTensor::<Complex<f64>, 2>::from_fn([u_cols, u_rows], |idx| {
            u[[idx[1], idx[0]]].conj() // conjugate transpose: U^H
        });
        let v = vt.transpose().to_tensor(); // (n_cols, min_dim)

        // Verify v.cols() == s.len() (v.shape().1 is the second dimension, which is min_dim)
        assert_eq!(
            v.shape().1,
            min_dim,
            "v.cols()={} must equal s.len()={}",
            v.shape().1,
            min_dim
        );

        Self { ut, s, v }
    }
}

// ============================================================================
// SVD computation functions
// ============================================================================

/// Compute SVD of a real matrix using mdarray-linalg
pub(crate) fn compute_real_svd(matrix: &DTensor<f64, 2>) -> RealSVD {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    let mut a = matrix.clone();
    let SVDDecomp { u, s, vt } = Faer.svd(&mut *a).expect("SVD computation failed");

    // Extract singular values from first row
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]]).collect();

    // Trim u and vt to min_dim
    // u: (n_rows, n_cols) -> (n_rows, min_dim) - take first min_dim columns
    // vt: (n_rows, n_cols) -> (min_dim, n_cols) - take first min_dim rows
    let u_trimmed = u.view(.., ..min_dim).to_tensor();
    let vt_trimmed = vt.view(..min_dim, ..).to_tensor();

    RealSVD::new(u_trimmed, s_vec, vt_trimmed)
}

/// Compute SVD of a complex matrix directly
pub(crate) fn compute_complex_svd(matrix: &DTensor<Complex<f64>, 2>) -> ComplexSVD {
    use mdarray_linalg::prelude::SVD;
    use mdarray_linalg::svd::SVDDecomp;
    use mdarray_linalg_faer::Faer;

    // Use matrix directly (Complex<f64> is compatible with faer's c64)
    let mut matrix_c64 = matrix.clone();

    // Compute complex SVD directly
    let SVDDecomp { u, s, vt } = Faer
        .svd(&mut *matrix_c64)
        .expect("Complex SVD computation failed");

    // Extract singular values from first row (they are real even though stored as Complex)
    let min_dim = s.shape().0.min(s.shape().1);
    let s_vec: Vec<f64> = (0..min_dim).map(|i| s[[0, i]].re).collect();

    // Trim u and vt to min_dim
    // u: (n_rows, n_cols) -> (n_rows, min_dim) - take first min_dim columns
    // vt: (n_rows, n_cols) -> (min_dim, n_cols) - take first min_dim rows
    let u_trimmed = u.view(.., ..min_dim).to_tensor();
    let vt_trimmed = vt.view(..min_dim, ..).to_tensor();

    ComplexSVD::new(u_trimmed, s_vec, vt_trimmed)
}

// ============================================================================
// Complex-Real conversion helpers
// ============================================================================

/// Combine real and imaginary parts into complex tensor
pub(crate) fn combine_complex(
    re: &DTensor<f64, 2>,
    im: &DTensor<f64, 2>,
) -> DTensor<Complex<f64>, 2> {
    let (n_points, extra_size) = *re.shape();
    DTensor::<Complex<f64>, 2>::from_fn([n_points, extra_size], |idx| {
        Complex::new(re[idx], im[idx])
    })
}

/// Extract real parts from complex tensor (for coefficients)
pub(crate) fn extract_real_parts_coeffs(coeffs_2d: &DTensor<Complex<f64>, 2>) -> DTensor<f64, 2> {
    let (basis_size, extra_size) = *coeffs_2d.shape();
    DTensor::<f64, 2>::from_fn([basis_size, extra_size], |idx| coeffs_2d[idx].re)
}
