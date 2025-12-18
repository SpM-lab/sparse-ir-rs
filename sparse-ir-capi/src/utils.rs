//! Utility functions for C API
//!
//! This module provides helper functions for order conversion and dimension handling.

#[allow(unused_imports)] // Used in test code
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR,
    SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2,
};
#[allow(unused_imports)]
use mdarray::Shape;
use sparse_ir::numeric::CustomNumeric; // Used in test code for with_dims

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

/// Convert dimensions and target_dim for row-major mdarray
///
/// mdarray uses row-major (C order) by default. When the C-API caller
/// specifies column-major (Fortran/Julia order), we need to reverse
/// dimensions and adjust target_dim to match mdarray's row-major layout.
///
/// This follows libsparseir's pattern for order handling.
///
/// # Arguments
/// * `dims` - Original dimensions from C-API
/// * `target_dim` - Original target dimension from C-API
/// * `order` - Memory order specified by caller
///
/// # Returns
/// (mdarray_dims, mdarray_target_dim) - Dimensions and target_dim for row-major mdarray
///
/// # Example
/// ```text
/// // Julia: dims=[5, 3], target_dim=0, order=COLUMN_MAJOR
/// convert_dims_for_row_major(&[5, 3], 0, MemoryOrder::ColumnMajor)
/// → ([3, 5], 1)  // For row-major mdarray
/// ```
pub fn convert_dims_for_row_major(
    dims: &[usize],
    target_dim: usize,
    order: MemoryOrder,
) -> (Vec<usize>, usize) {
    match order {
        MemoryOrder::RowMajor => {
            // Already row-major, use as-is
            (dims.to_vec(), target_dim)
        }
        MemoryOrder::ColumnMajor => {
            // Convert column-major to row-major:
            // Reverse dims and flip target_dim
            let mut rev_dims = dims.to_vec();
            rev_dims.reverse();
            let rev_target_dim = dims.len() - 1 - target_dim;
            (rev_dims, rev_target_dim)
        }
    }
}

/// Read N-dimensional tensor from raw pointer (row-major layout)
///
/// Reads a tensor from a raw pointer assuming row-major (C order) memory layout.
/// The buffer is interpreted as a flat array and reshaped according to `dims`.
///
/// # Arguments
/// * `ptr` - Raw pointer to the data buffer
/// * `dims` - Dimensions of the tensor (e.g., `[num_points, basis_size]`)
///
/// # Returns
/// A `Tensor<T, DynRank>` with the specified dimensions
///
/// # Safety
/// Caller must ensure `ptr` is valid and points to at least `product(dims)` elements.
pub(crate) unsafe fn _read_tensor_nd_row_major<T: Copy>(
    ptr: *const T,
    dims: &[usize],
) -> sparse_ir::Tensor<T, sparse_ir::DynRank> {
    assert!(!dims.is_empty(), "dims must not be empty");
    let total: usize = dims.iter().product();

    // Read buffer as slice
    let slice = unsafe { std::slice::from_raw_parts(ptr, total) };
    let data: Vec<T> = slice.to_vec();

    // Create 1D tensor and reshape to specified dimensions
    let flat = sparse_ir::Tensor::<T, (usize,)>::from(data);
    flat.into_dyn().reshape(dims).to_tensor()
}

/// Read N-dimensional tensor from raw pointer (column-major layout)
///
/// Reads a tensor from a raw pointer assuming column-major (Fortran/Julia order) memory layout.
/// The buffer is interpreted as a flat array with reversed dimensions, then permuted
/// to restore the original axis order.
///
/// # Arguments
/// * `ptr` - Raw pointer to the data buffer
/// * `dims` - Dimensions of the tensor (e.g., `[num_points, basis_size]`)
///
/// # Returns
/// A `Tensor<T, DynRank>` with the specified dimensions and correct axis order
///
/// # Safety
/// Caller must ensure `ptr` is valid and points to at least `product(dims)` elements.
pub(crate) unsafe fn _read_tensor_nd_column_major<T: Copy>(
    ptr: *const T,
    dims: &[usize],
) -> sparse_ir::Tensor<T, sparse_ir::DynRank> {
    assert!(!dims.is_empty(), "dims must not be empty");

    // 1. Reverse dimensions to read as row-major
    let mut rev_dims = dims.to_vec();
    rev_dims.reverse();
    let tmp = unsafe { _read_tensor_nd_row_major(ptr, &rev_dims) };

    // 2. Permute axes to restore original order
    // For example: if dims=[5, 3], we read as [3, 5] (row-major),
    // then permute [0, 1] -> [1, 0] to get back [5, 3]
    let rank = dims.len();
    let perm: Vec<usize> = (0..rank).rev().collect();

    // Tensor implements Borrow<Slice>, so &tmp can be used as &Slice
    use mdarray::Slice;
    (&tmp as &Slice<T, sparse_ir::DynRank>)
        .permute(&perm[..])
        .to_tensor()
}

/// Read N-dimensional tensor from raw pointer
///
/// Reads a tensor from a raw pointer, handling both row-major and column-major memory layouts.
/// This is a convenience wrapper that dispatches to the appropriate internal function
/// based on the memory order.
///
/// # Arguments
/// * `ptr` - Raw pointer to the data buffer
/// * `dims` - Dimensions of the tensor (e.g., `[num_points, basis_size]`)
/// * `order` - Memory layout order (RowMajor or ColumnMajor)
///
/// # Returns
/// A `Tensor<T, DynRank>` with the specified dimensions
///
/// # Safety
/// Caller must ensure `ptr` is valid and points to at least `product(dims)` elements.
pub(crate) unsafe fn read_tensor_nd<T: Copy>(
    ptr: *const T,
    dims: &[usize],
    order: MemoryOrder,
) -> sparse_ir::Tensor<T, sparse_ir::DynRank> {
    match order {
        MemoryOrder::RowMajor => unsafe { _read_tensor_nd_row_major(ptr, dims) },
        MemoryOrder::ColumnMajor => unsafe { _read_tensor_nd_column_major(ptr, dims) },
    }
}

/// Copy N-dimensional tensor to C array
///
/// Flattens the tensor and copies all elements to the output pointer.
/// For column-major order, the tensor dimensions are permuted before flattening
/// to match the expected memory layout.
///
/// # Arguments
/// * `tensor` - Source tensor (any rank)
/// * `out` - Destination C array pointer
/// * `order` - Memory layout order for output (RowMajor or ColumnMajor)
///
/// # Safety
/// Caller must ensure `out` has space for `tensor.len()` elements
pub(crate) unsafe fn copy_tensor_to_c_array<T: Copy>(
    tensor: sparse_ir::Tensor<T, sparse_ir::DynRank>,
    out: *mut T,
    order: MemoryOrder,
) {
    let total = tensor.len();

    // For column-major, permute dimensions to reverse order before flattening
    let flat = match order {
        MemoryOrder::RowMajor => {
            // Row-major: flatten directly
            tensor.into_dyn().reshape(&[total]).to_tensor()
        }
        MemoryOrder::ColumnMajor => {
            // Column-major: permute dimensions to reverse order, then flatten
            // This is the inverse of read_tensor_nd_column_major
            use mdarray::Slice;
            let rank = tensor.rank();
            let perm: Vec<usize> = (0..rank).rev().collect();
            let permuted = (&tensor as &Slice<T, sparse_ir::DynRank>)
                .permute(&perm[..])
                .to_tensor();
            permuted.into_dyn().reshape(&[total]).to_tensor()
        }
    };

    for i in 0..total {
        unsafe {
            *out.add(i) = flat[i];
        }
    }
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

/// Create a DView (immutable) from raw pointer with DynRank dimensions
///
/// Zero-copy: directly interprets the buffer as a tensor with the given dimensions.
/// For column-major data, pass reversed dimensions (via convert_dims_for_row_major).
///
/// # Safety
/// - `ptr` must be valid and point to at least `product(dims)` elements
/// - The memory must remain valid for the lifetime of the returned view
pub(crate) unsafe fn create_dview_from_ptr<'a, T>(
    ptr: *const T,
    dims: &[usize],
) -> mdarray::View<'a, T, sparse_ir::DynRank, mdarray::Dense> {
    use mdarray::Shape;
    let shape = sparse_ir::DynRank::from_dims(dims);
    let mapping = mdarray::DenseMapping::new(shape);
    unsafe { mdarray::View::new_unchecked(ptr, mapping) }
}

/// Create a DViewMut (mutable) from raw pointer with DynRank dimensions
///
/// Zero-copy: directly interprets the buffer as a mutable tensor with the given dimensions.
/// For column-major data, pass reversed dimensions (via convert_dims_for_row_major).
///
/// # Safety
/// - `ptr` must be valid and point to at least `product(dims)` elements
/// - The memory must remain valid for the lifetime of the returned view
/// - The caller must ensure no aliasing occurs
pub(crate) unsafe fn create_dviewmut_from_ptr<'a, T>(
    ptr: *mut T,
    dims: &[usize],
) -> mdarray::ViewMut<'a, T, sparse_ir::DynRank> {
    use mdarray::Shape;
    let shape = sparse_ir::DynRank::from_dims(dims);
    let mapping = mdarray::DenseMapping::new(shape);
    unsafe { mdarray::ViewMut::new_unchecked(ptr, mapping) }
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
    fn test_read_tensor_nd_row_major() {
        use num_complex::Complex64;

        // Test 2D tensor: 3x4 matrix
        {
            // Create test data: row-major order
            // [[1, 2, 3, 4],
            //  [5, 6, 7, 8],
            //  [9, 10, 11, 12]]
            let data = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let tensor = unsafe { _read_tensor_nd_row_major(data.as_ptr(), &[3, 4]) };

            let shape_dims = tensor.shape().with_dims(|dims| dims.to_vec());
            assert_eq!(shape_dims, &[3, 4]);
            assert_eq!(tensor[[0, 0]], 1.0);
            assert_eq!(tensor[[0, 3]], 4.0);
            assert_eq!(tensor[[1, 0]], 5.0);
            assert_eq!(tensor[[2, 3]], 12.0);
        }

        // Test 3D tensor: 2x3x4
        {
            let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
            let tensor = unsafe { _read_tensor_nd_row_major(data.as_ptr(), &[2, 3, 4]) };

            let shape_dims = tensor.shape().with_dims(|dims| dims.to_vec());
            assert_eq!(shape_dims, &[2, 3, 4]);
            assert_eq!(tensor[[0, 0, 0]], 1.0);
            assert_eq!(tensor[[0, 0, 3]], 4.0);
            assert_eq!(tensor[[0, 1, 0]], 5.0);
            assert_eq!(tensor[[1, 2, 3]], 24.0);
        }

        // Test complex numbers
        {
            let data = vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, 6.0),
                Complex64::new(7.0, 8.0),
            ];
            let tensor = unsafe { _read_tensor_nd_row_major(data.as_ptr(), &[2, 2]) };

            let shape_dims = tensor.shape().with_dims(|dims| dims.to_vec());
            assert_eq!(shape_dims, &[2, 2]);
            assert_eq!(tensor[[0, 0]], Complex64::new(1.0, 2.0));
            assert_eq!(tensor[[1, 1]], Complex64::new(7.0, 8.0));
        }
    }

    #[test]
    fn test_read_tensor_nd_column_major() {
        use num_complex::Complex64;

        // Test 2D tensor: 3x4 matrix
        // Column-major order means:
        // [[1, 4, 7, 10],
        //  [2, 5, 8, 11],
        //  [3, 6, 9, 12]]
        // But we want to read it as [3, 4] shape
        {
            // Create test data: column-major order
            // First column: [1, 2, 3]
            // Second column: [4, 5, 6]
            // Third column: [7, 8, 9]
            // Fourth column: [10, 11, 12]
            let data = vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let tensor = unsafe { _read_tensor_nd_column_major(data.as_ptr(), &[3, 4]) };

            let shape_dims = tensor.shape().with_dims(|dims| dims.to_vec());
            assert_eq!(shape_dims, &[3, 4]);
            // After reading as [4, 3] (reversed) and permuting back, we should get:
            // [[1, 4, 7, 10],
            //  [2, 5, 8, 11],
            //  [3, 6, 9, 12]]
            assert_eq!(tensor[[0, 0]], 1.0);
            assert_eq!(tensor[[0, 1]], 4.0);
            assert_eq!(tensor[[0, 3]], 10.0);
            assert_eq!(tensor[[1, 0]], 2.0);
            assert_eq!(tensor[[2, 3]], 12.0);
        }

        // Test 3D tensor: 2x3x4
        // Column-major: first all elements with index [0,0,0], [1,0,0], then [0,1,0], [1,1,0], etc.
        {
            // For 2x3x4, column-major order:
            // [0,0,0]=1, [1,0,0]=2, [0,1,0]=3, [1,1,0]=4, [0,2,0]=5, [1,2,0]=6,
            // [0,0,1]=7, [1,0,1]=8, ...
            let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
            let tensor = unsafe { _read_tensor_nd_column_major(data.as_ptr(), &[2, 3, 4]) };

            let shape_dims = tensor.shape().with_dims(|dims| dims.to_vec());
            assert_eq!(shape_dims, &[2, 3, 4]);
            // Verify first few elements
            assert_eq!(tensor[[0, 0, 0]], 1.0);
            assert_eq!(tensor[[1, 0, 0]], 2.0);
            assert_eq!(tensor[[0, 1, 0]], 3.0);
        }

        // Test complex numbers
        {
            // Column-major: [1+2i, 3+4i] in first column, [5+6i, 7+8i] in second column
            let data = vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, 6.0),
                Complex64::new(7.0, 8.0),
            ];
            let tensor = unsafe { _read_tensor_nd_column_major(data.as_ptr(), &[2, 2]) };

            let shape_dims = tensor.shape().with_dims(|dims| dims.to_vec());
            assert_eq!(shape_dims, &[2, 2]);
            // After permute: [[1+2i, 5+6i], [3+4i, 7+8i]]
            assert_eq!(tensor[[0, 0]], Complex64::new(1.0, 2.0));
            assert_eq!(tensor[[1, 0]], Complex64::new(3.0, 4.0));
            assert_eq!(tensor[[0, 1]], Complex64::new(5.0, 6.0));
            assert_eq!(tensor[[1, 1]], Complex64::new(7.0, 8.0));
        }
    }

    #[test]
    fn test_read_tensor_nd_roundtrip() {
        // Test that row-major and column-major produce consistent results
        // when the data is transposed appropriately

        // Create a 3x4 matrix in row-major
        let row_major_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let row_tensor = unsafe { _read_tensor_nd_row_major(row_major_data.as_ptr(), &[3, 4]) };

        // Create the same matrix in column-major (transposed storage)
        // [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]] stored as:
        // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] (column-major)
        let col_major_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let col_tensor = unsafe { _read_tensor_nd_column_major(col_major_data.as_ptr(), &[3, 4]) };

        // They should have the same shape
        let row_shape = row_tensor.shape().with_dims(|dims| dims.to_vec());
        let col_shape = col_tensor.shape().with_dims(|dims| dims.to_vec());
        assert_eq!(row_shape, col_shape);

        // But different values (because storage order is different)
        // row_tensor: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        // col_tensor: [[1,4,7,10], [2,5,8,11], [3,6,9,12]]
        assert_eq!(row_tensor[[0, 0]], 1.0);
        assert_eq!(col_tensor[[0, 0]], 1.0);
        assert_eq!(row_tensor[[0, 1]], 2.0);
        assert_eq!(col_tensor[[0, 1]], 4.0); // Different!
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
