//! Utility functions for C API
//!
//! This module provides helpers for memory-order handling, zero-copy tensor
//! views over caller buffers, and mapping library errors to status codes.

#[allow(unused_imports)] // Used in test code
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INPUT_DIMENSION_MISMATCH, SPIR_INTERNAL_ERROR,
    SPIR_INVALID_ARGUMENT, SPIR_INVALID_DIMENSION, SPIR_NOT_SUPPORTED, SPIR_ORDER_COLUMN_MAJOR,
    SPIR_ORDER_ROW_MAJOR, SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2, StatusCode,
};
use sparse_ir::numeric::CustomNumeric; // Used in test code
use sparse_ir::{Matrix, TensorScalar, TypedTensor, TypedTensorView, TypedTensorViewMut};

/// Check if SPARSEIR_DEBUG environment variable is set
///
/// Returns true if SPARSEIR_DEBUG is set to any non-empty value.
pub fn is_debug_enabled() -> bool {
    std::env::var("SPARSEIR_DEBUG").is_ok()
}

/// Memory layout order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryOrder {
    RowMajor,    // Rightmost dimension varies fastest (C, Python)
    ColumnMajor, // Leftmost dimension varies fastest (Fortran, Julia, MATLAB)
}

impl MemoryOrder {
    /// Convert from C int to MemoryOrder
    pub fn from_c_int(order: libc::c_int) -> Result<Self, ()> {
        match order {
            SPIR_ORDER_ROW_MAJOR => Ok(Self::RowMajor),
            SPIR_ORDER_COLUMN_MAJOR => Ok(Self::ColumnMajor),
            _ => Err(()),
        }
    }
}

/// Map a library error to a C API status code.
pub(crate) fn status_from_error(err: &sparse_ir::Error) -> StatusCode {
    use sparse_ir::Error;
    match err {
        Error::Unsupported(_) => SPIR_NOT_SUPPORTED,
        Error::ShapeMismatch(_) => SPIR_INPUT_DIMENSION_MISMATCH,
        Error::AxisOutOfRange { .. } => SPIR_INVALID_DIMENSION,
        Error::InvalidArgument(_) | Error::Layout(_) => SPIR_INVALID_ARGUMENT,
        _ => SPIR_INTERNAL_ERROR,
    }
}

/// Convert dimensions and target_dim to the column-major convention used by
/// sparse-ir tensors.
///
/// A row-major buffer with dims `[d0, ..., dn]` is the same memory as a
/// column-major buffer with dims `[dn, ..., d0]`, so row-major input is
/// handled by reversing the dimensions and mirroring `target_dim`; no data is
/// moved.
///
/// # Example
/// ```text
/// // Python: dims=[5, 3], target_dim=0, order=ROW_MAJOR
/// convert_dims_for_col_major(&[5, 3], 0, MemoryOrder::RowMajor)
/// // Returns: ([3, 5], 1)
/// ```
pub fn convert_dims_for_col_major(
    dims: &[usize],
    target_dim: usize,
    order: MemoryOrder,
) -> (Vec<usize>, usize) {
    match order {
        MemoryOrder::ColumnMajor => (dims.to_vec(), target_dim),
        MemoryOrder::RowMajor => {
            let mut rev_dims = dims.to_vec();
            rev_dims.reverse();
            (rev_dims, dims.len() - 1 - target_dim)
        }
    }
}

/// Compact column-major strides for `dims`, or `None` on overflow.
fn col_major_strides(dims: &[usize]) -> Option<Vec<isize>> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut acc: usize = 1;
    for &d in dims {
        strides.push(isize::try_from(acc).ok()?);
        acc = acc.checked_mul(d)?;
    }
    isize::try_from(acc).ok()?;
    Some(strides)
}

/// Number of elements of `dims`, or an error on overflow.
fn checked_len(dims: &[usize]) -> sparse_ir::Result<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| sparse_ir::Error::InvalidArgument(format!("dims {dims:?} overflow")))
}

/// Read a matrix from a raw pointer into a column-major [`Matrix`].
///
/// # Safety
/// `ptr` must be valid for reads of `nrows * ncols` elements.
pub(crate) unsafe fn read_matrix<T: TensorScalar + Copy>(
    ptr: *const T,
    nrows: usize,
    ncols: usize,
    order: MemoryOrder,
) -> sparse_ir::Result<Matrix<T>> {
    let len = checked_len(&[nrows, ncols])?;
    // SAFETY: the caller guarantees `ptr` is valid for `len` reads.
    let src = unsafe { std::slice::from_raw_parts(ptr, len) };
    let data = match order {
        MemoryOrder::ColumnMajor => src.to_vec(),
        MemoryOrder::RowMajor => {
            let mut data = Vec::with_capacity(len);
            for j in 0..ncols {
                data.extend((0..nrows).map(|i| src[i * ncols + j]));
            }
            data
        }
    };
    Ok(TypedTensor::from_vec_col_major([nrows, ncols], data)?)
}

/// Copy a column-major tensor's storage into a new column-major tensor with
/// the given dimensions.
///
/// Used together with [`convert_dims_for_col_major`]: the dims passed here are
/// already in column-major convention, so this is a flat copy.
///
/// # Safety
/// `ptr` must be valid for reads of `product(dims)` elements.
pub(crate) unsafe fn read_tensor_col_major<T: TensorScalar + Copy>(
    ptr: *const T,
    dims: &[usize],
) -> sparse_ir::Result<TypedTensor<T>> {
    let len = checked_len(dims)?;
    // SAFETY: the caller guarantees `ptr` is valid for `len` reads.
    let src = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(TypedTensor::from_vec_col_major(
        dims.to_vec(),
        src.to_vec(),
    )?)
}

/// Copy a tensor's column-major storage to a C array.
///
/// # Safety
/// `out` must be valid for writes of `tensor.n_elements()` elements.
pub(crate) unsafe fn copy_tensor_to_c_array<T: TensorScalar + Copy>(
    tensor: &TypedTensor<T>,
    out: *mut T,
) -> sparse_ir::Result<()> {
    let data = tensor.host_data()?;
    // SAFETY: the caller guarantees `out` is valid for `data.len()` writes,
    // and `out` cannot alias the freshly computed tensor.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), out, data.len()) };
    Ok(())
}

/// Build output dimensions by replacing target_dim with new_size
pub(crate) fn build_output_dims(
    input_dims: &[usize],
    target_dim: usize,
    new_size: usize,
) -> Vec<usize> {
    let mut out_dims = input_dims.to_vec();
    out_dims[target_dim] = new_size;
    out_dims
}

/// Create an immutable compact column-major view over a raw buffer.
///
/// Zero-copy. For row-major data, pass dims converted by
/// [`convert_dims_for_col_major`].
///
/// # Safety
/// - `ptr` must be valid for reads of `product(dims)` elements
/// - The memory must remain valid and unmodified for `'a`
pub(crate) unsafe fn create_view_from_ptr<'a, T: 'static>(
    ptr: *const T,
    dims: &[usize],
) -> sparse_ir::Result<TypedTensorView<'a, T>> {
    let len = checked_len(dims)?;
    let strides = col_major_strides(dims)
        .ok_or_else(|| sparse_ir::Error::InvalidArgument(format!("dims {dims:?} overflow")))?;
    // SAFETY: the caller guarantees `ptr` is valid for `len` reads for `'a`.
    let data = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(TypedTensorView::from_slice(dims, strides, 0, data)?)
}

/// Create a mutable compact column-major view over a raw buffer.
///
/// Zero-copy. For row-major data, pass dims converted by
/// [`convert_dims_for_col_major`].
///
/// # Safety
/// - `ptr` must be valid for writes of `product(dims)` elements
/// - The memory must remain valid for `'a` and must not alias any other
///   live reference
pub(crate) unsafe fn create_view_mut_from_ptr<'a, T: 'static>(
    ptr: *mut T,
    dims: &[usize],
) -> sparse_ir::Result<TypedTensorViewMut<'a, T>> {
    let len = checked_len(dims)?;
    let strides = col_major_strides(dims)
        .ok_or_else(|| sparse_ir::Error::InvalidArgument(format!("dims {dims:?} overflow")))?;
    // SAFETY: the caller guarantees `ptr` is valid, unaliased, and writable
    // for `len` elements for `'a`.
    let data = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    Ok(TypedTensorViewMut::from_slice(dims, strides, 0, data)?)
}

/// Choose the working type (Twork) based on epsilon value
///
/// This function determines the appropriate working precision type based on the
/// target accuracy epsilon. It follows the same logic as SPIR_TWORK_AUTO:
/// - Returns SPIR_TWORK_FLOAT64X2 if epsilon < 1e-8 or epsilon is NaN
/// - Returns SPIR_TWORK_FLOAT64 otherwise
///
/// # Arguments
/// * `epsilon` - Target accuracy (must be non-negative, or NaN for auto-selection)
///
/// # Returns
/// Working type constant:
/// - SPIR_TWORK_FLOAT64 (0): Use double precision (64-bit)
/// - SPIR_TWORK_FLOAT64X2 (1): Use extended precision (128-bit)
#[unsafe(no_mangle)]
pub extern "C" fn spir_choose_working_type(epsilon: f64) -> libc::c_int {
    if epsilon.is_nan() || epsilon < 1e-8 {
        SPIR_TWORK_FLOAT64X2
    } else {
        SPIR_TWORK_FLOAT64
    }
}

/// Compute piecewise Gauss-Legendre quadrature rule (double precision)
///
/// Generates a piecewise Gauss-Legendre quadrature rule with n points per segment.
/// The rule is concatenated across all segments, with points and weights properly
/// scaled for each segment interval.
///
/// # Arguments
/// * `n` - Number of Gauss points per segment (must be >= 1)
/// * `segments` - Array of segment boundaries (n_segments + 1 elements).
///                Must be monotonically increasing.
/// * `n_segments` - Number of segments (must be >= 1)
/// * `x` - Output array for Gauss points (size n * n_segments). Must be pre-allocated.
/// * `w` - Output array for Gauss weights (size n * n_segments). Must be pre-allocated.
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Status code:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - Non-zero error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn spir_gauss_legendre_rule_piecewise_double(
    n: libc::c_int,
    segments: *const f64,
    n_segments: libc::c_int,
    x: *mut f64,
    w: *mut f64,
    status: *mut crate::StatusCode,
) -> crate::StatusCode {
    use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT};
    use sparse_ir::legendre;
    use std::panic::catch_unwind;

    if status.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if segments.is_null() || x.is_null() || w.is_null() {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return SPIR_INVALID_ARGUMENT;
    }

    if n < 1 || n_segments < 1 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| {
        // Convert segments to Vec
        let segments_slice =
            unsafe { std::slice::from_raw_parts(segments, (n_segments + 1) as usize) };
        let segs_vec = segments_slice.to_vec();

        // Verify segments are monotonically increasing
        for i in 1..segs_vec.len() {
            if segs_vec[i] <= segs_vec[i - 1] {
                unsafe {
                    *status = SPIR_INVALID_ARGUMENT;
                }
                return SPIR_INVALID_ARGUMENT;
            }
        }

        // Generate base rule with DDouble precision, then convert to double
        let rule_dd = legendre::<sparse_ir::Df64>(n as usize);
        let rule = sparse_ir::gauss::Rule::from_vectors(
            rule_dd.x.iter().map(|&x| x.to_f64()).collect(),
            rule_dd.w.iter().map(|&w| w.to_f64()).collect(),
            rule_dd.a.to_f64(),
            rule_dd.b.to_f64(),
        );

        // Create piecewise rule
        let piecewise_rule = rule.piecewise(&segs_vec);

        // Copy to output arrays
        for i in 0..piecewise_rule.x.len() {
            unsafe {
                *x.add(i) = piecewise_rule.x[i];
                *w.add(i) = piecewise_rule.w[i];
            }
        }

        unsafe {
            *status = SPIR_COMPUTATION_SUCCESS;
        }
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or_else(|_| {
        unsafe {
            *status = SPIR_INTERNAL_ERROR;
        }
        SPIR_INTERNAL_ERROR
    })
}

/// Compute piecewise Gauss-Legendre quadrature rule (DDouble precision)
///
/// Generates a piecewise Gauss-Legendre quadrature rule with n points per segment,
/// computed using extended precision (DDouble). Returns high and low parts separately
/// for maximum precision.
///
/// # Arguments
/// * `n` - Number of Gauss points per segment (must be >= 1)
/// * `segments` - Array of segment boundaries (n_segments + 1 elements).
///                Must be monotonically increasing.
/// * `n_segments` - Number of segments (must be >= 1)
/// * `x_high` - Output array for high part of Gauss points (size n * n_segments).
///              Must be pre-allocated.
/// * `x_low` - Output array for low part of Gauss points (size n * n_segments).
///             Must be pre-allocated.
/// * `w_high` - Output array for high part of Gauss weights (size n * n_segments).
///              Must be pre-allocated.
/// * `w_low` - Output array for low part of Gauss weights (size n * n_segments).
///            Must be pre-allocated.
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Status code:
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - Non-zero error code on failure
#[unsafe(no_mangle)]
pub extern "C" fn spir_gauss_legendre_rule_piecewise_ddouble(
    n: libc::c_int,
    segments: *const f64,
    n_segments: libc::c_int,
    x_high: *mut f64,
    x_low: *mut f64,
    w_high: *mut f64,
    w_low: *mut f64,
    status: *mut crate::StatusCode,
) -> crate::StatusCode {
    use crate::{SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_INVALID_ARGUMENT};
    use sparse_ir::legendre;
    use std::panic::catch_unwind;

    if status.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }

    if segments.is_null()
        || x_high.is_null()
        || x_low.is_null()
        || w_high.is_null()
        || w_low.is_null()
    {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return SPIR_INVALID_ARGUMENT;
    }

    if n < 1 || n_segments < 1 {
        unsafe {
            *status = SPIR_INVALID_ARGUMENT;
        }
        return SPIR_INVALID_ARGUMENT;
    }

    let result = catch_unwind(|| {
        // Convert segments to Vec
        let segments_slice =
            unsafe { std::slice::from_raw_parts(segments, (n_segments + 1) as usize) };
        let segs_vec: Vec<sparse_ir::Df64> = segments_slice
            .iter()
            .map(|&x| sparse_ir::Df64::new(x))
            .collect();

        // Verify segments are monotonically increasing
        for i in 1..segs_vec.len() {
            if segs_vec[i] <= segs_vec[i - 1] {
                unsafe {
                    *status = SPIR_INVALID_ARGUMENT;
                }
                return SPIR_INVALID_ARGUMENT;
            }
        }

        // Generate base rule with DDouble precision
        let rule_dd = legendre::<sparse_ir::Df64>(n as usize);

        // Create piecewise rule
        let piecewise_rule = rule_dd.piecewise(&segs_vec);

        // Extract high and low parts
        for i in 0..piecewise_rule.x.len() {
            unsafe {
                *x_high.add(i) = piecewise_rule.x[i].hi();
                *x_low.add(i) = piecewise_rule.x[i].lo();
                *w_high.add(i) = piecewise_rule.w[i].hi();
                *w_low.add(i) = piecewise_rule.w[i].lo();
            }
        }

        unsafe {
            *status = SPIR_COMPUTATION_SUCCESS;
        }
        SPIR_COMPUTATION_SUCCESS
    });

    result.unwrap_or_else(|_| {
        unsafe {
            *status = SPIR_INTERNAL_ERROR;
        }
        SPIR_INTERNAL_ERROR
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_order_conversion() {
        assert_eq!(
            MemoryOrder::from_c_int(SPIR_ORDER_ROW_MAJOR),
            Ok(MemoryOrder::RowMajor)
        );
        assert_eq!(
            MemoryOrder::from_c_int(SPIR_ORDER_COLUMN_MAJOR),
            Ok(MemoryOrder::ColumnMajor)
        );
        assert_eq!(MemoryOrder::from_c_int(99), Err(()));
    }

    #[test]
    fn test_choose_working_type() {
        // Test with epsilon >= 1e-8 -> should return FLOAT64
        {
            let twork = spir_choose_working_type(1e-6);
            assert_eq!(twork, SPIR_TWORK_FLOAT64);
        }

        {
            let twork = spir_choose_working_type(1e-8);
            assert_eq!(twork, SPIR_TWORK_FLOAT64);
        }

        // Test with epsilon < 1e-8 -> should return FLOAT64X2
        {
            let twork = spir_choose_working_type(1e-10);
            assert_eq!(twork, SPIR_TWORK_FLOAT64X2);
        }

        {
            let twork = spir_choose_working_type(1e-15);
            assert_eq!(twork, SPIR_TWORK_FLOAT64X2);
        }

        // Test with NaN -> should return FLOAT64X2
        {
            let twork = spir_choose_working_type(f64::NAN);
            assert_eq!(twork, SPIR_TWORK_FLOAT64X2);
        }

        // Test boundary case: epsilon = 1e-8 exactly
        {
            let twork = spir_choose_working_type(1e-8);
            assert_eq!(twork, SPIR_TWORK_FLOAT64);
        }

        // Test boundary case: epsilon just below 1e-8
        {
            let twork = spir_choose_working_type(0.99e-8);
            assert_eq!(twork, SPIR_TWORK_FLOAT64X2);
        }
    }

    #[test]
    fn test_gauss_legendre_rule_piecewise_double() {
        // Test with single segment [-1, 1]
        {
            let n = 5;
            let segments = [-1.0, 1.0];
            let n_segments = 1;
            let mut x = vec![0.0; n as usize];
            let mut w = vec![0.0; n as usize];
            let mut status = SPIR_INTERNAL_ERROR;

            let result = spir_gauss_legendre_rule_piecewise_double(
                n,
                segments.as_ptr(),
                n_segments,
                x.as_mut_ptr(),
                w.as_mut_ptr(),
                &mut status,
            );
            assert_eq!(result, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Verify we got n points
            // Points should be in [-1, 1] and sorted
            assert!(x[0] >= -1.0);
            assert!(x[(n - 1) as usize] <= 1.0);
            for i in 1..(n as usize) {
                assert!(x[i] > x[i - 1]);
            }

            // Weights should be positive
            for i in 0..(n as usize) {
                assert!(w[i] > 0.0);
            }

            // Weight sum should be approximately 2.0 (integral over [-1, 1] is 2)
            let weight_sum: f64 = w.iter().sum();
            assert!((weight_sum - 2.0).abs() < 1e-10);
        }

        // Test with two segments [-1, 0, 1]
        {
            let n = 3;
            let segments = [-1.0, 0.0, 1.0];
            let n_segments = 2;
            let mut x = vec![0.0; (n * n_segments) as usize];
            let mut w = vec![0.0; (n * n_segments) as usize];
            let mut status = SPIR_INTERNAL_ERROR;

            let result = spir_gauss_legendre_rule_piecewise_double(
                n,
                segments.as_ptr(),
                n_segments,
                x.as_mut_ptr(),
                w.as_mut_ptr(),
                &mut status,
            );
            assert_eq!(result, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Verify we got n * n_segments points
            // Points should be sorted across segments
            assert!(x[0] >= -1.0);
            assert!(x[5] <= 1.0);
            for i in 1..6 {
                assert!(x[i] > x[i - 1]);
            }

            // Weights should be positive
            for i in 0..6 {
                assert!(w[i] > 0.0);
            }

            // Weight sum should be approximately 2.0 (integral over [-1, 1])
            let weight_sum: f64 = w.iter().sum();
            assert!((weight_sum - 2.0).abs() < 1e-10);
        }

        // Test error handling
        {
            let mut status = SPIR_INTERNAL_ERROR;
            let result = spir_gauss_legendre_rule_piecewise_double(
                5,
                std::ptr::null(),
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut status,
            );
            assert_ne!(result, SPIR_COMPUTATION_SUCCESS);
        }

        {
            let segments = [-1.0, 1.0];
            let mut x = vec![0.0; 5];
            let mut w = vec![0.0; 5];
            let mut status = SPIR_INTERNAL_ERROR;
            let result = spir_gauss_legendre_rule_piecewise_double(
                0,
                segments.as_ptr(),
                1,
                x.as_mut_ptr(),
                w.as_mut_ptr(),
                &mut status,
            );
            assert_ne!(result, SPIR_COMPUTATION_SUCCESS);
        }

        {
            let segments = [1.0, -1.0]; // Wrong order
            let mut x = vec![0.0; 5];
            let mut w = vec![0.0; 5];
            let mut status = SPIR_INTERNAL_ERROR;
            let result = spir_gauss_legendre_rule_piecewise_double(
                5,
                segments.as_ptr(),
                1,
                x.as_mut_ptr(),
                w.as_mut_ptr(),
                &mut status,
            );
            assert_ne!(result, SPIR_COMPUTATION_SUCCESS);
        }
    }

    #[test]
    fn test_convert_dims_for_col_major() {
        assert_eq!(
            convert_dims_for_col_major(&[5, 3, 2], 0, MemoryOrder::ColumnMajor),
            (vec![5, 3, 2], 0)
        );
        assert_eq!(
            convert_dims_for_col_major(&[5, 3, 2], 0, MemoryOrder::RowMajor),
            (vec![2, 3, 5], 2)
        );
    }

    #[test]
    fn test_read_matrix_both_orders() {
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let row = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let a = unsafe { read_matrix(row.as_ptr(), 2, 3, MemoryOrder::RowMajor) }.unwrap();
        let b = unsafe { read_matrix(col.as_ptr(), 2, 3, MemoryOrder::ColumnMajor) }.unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.host_data().unwrap(), &col);
        assert_eq!(b.host_data().unwrap(), &col);
    }

    #[test]
    fn test_views_from_ptr() {
        use num_complex::Complex64;
        let data: Vec<Complex64> = (0..24).map(|i| Complex64::new(i as f64, -1.0)).collect();
        let view = unsafe { create_view_from_ptr(data.as_ptr(), &[2, 3, 4]) }.unwrap();
        assert_eq!(view.shape(), &[2, 3, 4]);
        assert!(view.is_col_major_contiguous().unwrap());
        assert_eq!(view.get(&[1, 2, 3]), Some(&data[1 + 2 * 2 + 3 * 6]));

        let mut out = vec![0.0f64; 6];
        {
            let mut v = unsafe { create_view_mut_from_ptr(out.as_mut_ptr(), &[3, 2]) }.unwrap();
            assert_eq!(v.shape(), &[3, 2]);
            assert!(v.is_col_major_contiguous().unwrap());
        }
        let t = unsafe { read_tensor_col_major(data.as_ptr(), &[4, 6]) }.unwrap();
        let mut copy = vec![Complex64::new(0.0, 0.0); 24];
        unsafe { copy_tensor_to_c_array(&t, copy.as_mut_ptr()) }.unwrap();
        assert_eq!(copy, data);
    }

    #[test]
    fn test_gauss_legendre_rule_piecewise_ddouble() {
        // Test with single segment [-1, 1]
        {
            let n = 5;
            let segments = [-1.0, 1.0];
            let n_segments = 1;
            let mut x_high = vec![0.0; n as usize];
            let mut x_low = vec![0.0; n as usize];
            let mut w_high = vec![0.0; n as usize];
            let mut w_low = vec![0.0; n as usize];
            let mut status = SPIR_INTERNAL_ERROR;

            let result = spir_gauss_legendre_rule_piecewise_ddouble(
                n,
                segments.as_ptr(),
                n_segments,
                x_high.as_mut_ptr(),
                x_low.as_mut_ptr(),
                w_high.as_mut_ptr(),
                w_low.as_mut_ptr(),
                &mut status,
            );
            assert_eq!(result, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Verify we got n points
            // Points should be in [-1, 1] and sorted
            let x0 = x_high[0] + x_low[0];
            let x_last = x_high[(n - 1) as usize] + x_low[(n - 1) as usize];
            assert!(x0 >= -1.0);
            assert!(x_last <= 1.0);
            for i in 1..(n as usize) {
                let x_val = x_high[i] + x_low[i];
                let x_prev = x_high[i - 1] + x_low[i - 1];
                assert!(x_val > x_prev);
            }

            // Weights should be positive
            let mut weight_sum = 0.0;
            for i in 0..(n as usize) {
                let w_val = w_high[i] + w_low[i];
                assert!(w_val > 0.0);
                weight_sum += w_val;
            }
            // Weight sum should be approximately 2.0 (integral over [-1, 1])
            assert!((weight_sum - 2.0).abs() < 1e-10);
        }

        // Test with two segments [-1, 0, 1]
        {
            let n = 3;
            let segments = [-1.0, 0.0, 1.0];
            let n_segments = 2;
            let mut x_high = vec![0.0; (n * n_segments) as usize];
            let mut x_low = vec![0.0; (n * n_segments) as usize];
            let mut w_high = vec![0.0; (n * n_segments) as usize];
            let mut w_low = vec![0.0; (n * n_segments) as usize];
            let mut status = SPIR_INTERNAL_ERROR;

            let result = spir_gauss_legendre_rule_piecewise_ddouble(
                n,
                segments.as_ptr(),
                n_segments,
                x_high.as_mut_ptr(),
                x_low.as_mut_ptr(),
                w_high.as_mut_ptr(),
                w_low.as_mut_ptr(),
                &mut status,
            );
            assert_eq!(result, SPIR_COMPUTATION_SUCCESS);
            assert_eq!(status, SPIR_COMPUTATION_SUCCESS);

            // Verify points are sorted
            for i in 1..6 {
                let x_val = x_high[i] + x_low[i];
                let x_prev = x_high[i - 1] + x_low[i - 1];
                assert!(x_val > x_prev);
            }

            // Weight sum should be approximately 2.0 (integral over [-1, 1])
            let mut weight_sum = 0.0;
            for i in 0..6 {
                let w_val = w_high[i] + w_low[i];
                weight_sum += w_val;
            }
            assert!((weight_sum - 2.0).abs() < 1e-10);
        }

        // Test error handling
        {
            let mut status = SPIR_INTERNAL_ERROR;
            let result = spir_gauss_legendre_rule_piecewise_ddouble(
                5,
                std::ptr::null(),
                1,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut status,
            );
            assert_ne!(result, SPIR_COMPUTATION_SUCCESS);
        }
    }
}
