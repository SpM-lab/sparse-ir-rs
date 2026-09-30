//! Utility functions for C API
//!
//! This module provides helper functions for order conversion and dimension handling.

#[allow(unused_imports)] // Used in test code
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INTERNAL_ERROR, SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR,
    SPIR_TWORK_FLOAT64, SPIR_TWORK_FLOAT64X2,
};
use crate::{
    SPIR_INPUT_DIMENSION_MISMATCH, SPIR_INVALID_ARGUMENT, SPIR_INVALID_DIMENSION, StatusCode,
};
use sparse_ir::numeric::CustomNumeric; // Used in test code for with_dims

/// Check if the `SPARSEIR_DEBUG` environment variable enables debug output
///
/// Returns true only if `SPARSEIR_DEBUG` is `1`, `true`, `yes` or `on`, in any
/// letter case, the same values pylibsparseir accepts. Any other value,
/// including `0` or an empty string, and an unset variable return false.
///
/// This forwards to [`sparse_ir::is_debug_enabled`], so the debug macros of
/// this crate and of `sparse-ir` share one rule.
pub fn is_debug_enabled() -> bool {
    sparse_ir::is_debug_enabled()
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

/// Convert dimensions and target_dim to the column-major convention
///
/// sparse-ir tensors are column-major (the leftmost axis varies fastest).
/// When the C-API caller specifies row-major (C/Python order), reversing the
/// dimensions and mirroring target_dim describes the same buffer as a
/// column-major tensor, without moving any data.
///
/// # Arguments
/// * `dims` - Original dimensions from C-API
/// * `target_dim` - Original target dimension from C-API
/// * `order` - Memory order specified by caller
///
/// # Returns
/// (col_major_dims, col_major_target_dim) - Dimensions and target_dim of the
/// column-major tensor
///
/// # Example
/// ```text
/// // Python: dims=[5, 3], target_dim=0, order=ROW_MAJOR
/// convert_dims_for_col_major(&[5, 3], 0, MemoryOrder::RowMajor)
/// → ([3, 5], 1)  // For a column-major tensor
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

/// Column-major linear offset of the multi-index `idx` in an array of `dims`
fn col_major_offset(idx: &[usize], dims: &[usize]) -> usize {
    idx.iter()
        .zip(dims)
        .rev()
        .fold(0, |acc, (&i, &n)| acc * n + i)
}

/// Row-major linear offset of the multi-index `idx` in an array of `dims`
fn row_major_offset(idx: &[usize], dims: &[usize]) -> usize {
    idx.iter().zip(dims).fold(0, |acc, (&i, &n)| acc * n + i)
}

/// Advance the multi-index `idx` of an array of `dims` in column-major order
fn next_col_major(idx: &mut [usize], dims: &[usize]) {
    for (i, &n) in idx.iter_mut().zip(dims) {
        *i += 1;
        if *i < n {
            return;
        }
        *i = 0;
    }
}

/// Read an N-dimensional array from a raw pointer
///
/// Returns a column-major tensor with the logical shape `dims`, in the
/// caller's axis order, whatever the memory order of the buffer: a
/// column-major buffer is copied as is, a row-major one is transposed.
///
/// # Arguments
/// * `ptr` - Raw pointer to the data buffer
/// * `dims` - Dimensions of the array (e.g., `[num_points, basis_size]`)
/// * `order` - Memory layout order of the buffer (RowMajor or ColumnMajor)
///
/// # Safety
/// Caller must ensure `ptr` is valid and points to at least `product(dims)`
/// elements, and that `dims` passes [`checked_len`] for `T` (entry points
/// taking `input_dims` establish this with [`validate_dims`]); otherwise the
/// element count below can wrap.
pub(crate) unsafe fn read_tensor_nd<T: sparse_ir::TensorScalar + Copy>(
    ptr: *const T,
    dims: &[usize],
    order: MemoryOrder,
) -> sparse_ir::Result<sparse_ir::TypedTensor<T>> {
    assert!(!dims.is_empty(), "dims must not be empty");
    let total: usize = dims.iter().product();
    // SAFETY: the caller guarantees `ptr` is valid for `total` reads.
    let src = unsafe { std::slice::from_raw_parts(ptr, total) };
    let data = match order {
        MemoryOrder::ColumnMajor => src.to_vec(),
        MemoryOrder::RowMajor => {
            let mut data = Vec::with_capacity(total);
            let mut idx = vec![0usize; dims.len()];
            for _ in 0..total {
                data.push(src[row_major_offset(&idx, dims)]);
                next_col_major(&mut idx, dims);
            }
            data
        }
    };
    Ok(sparse_ir::TypedTensor::from_vec_col_major(
        dims.to_vec(),
        data,
    )?)
}

/// Read a `nrows × ncols` matrix from a raw pointer
///
/// [`read_tensor_nd`] with a rank-2 result, as the sampling constructors
/// that take a matrix need it.
///
/// # Safety
/// Same requirements as [`read_tensor_nd`] for `dims = [nrows, ncols]`.
pub(crate) unsafe fn read_matrix<T: sparse_ir::TensorScalar + Copy>(
    ptr: *const T,
    nrows: usize,
    ncols: usize,
    order: MemoryOrder,
) -> sparse_ir::Result<sparse_ir::Matrix<T>> {
    let tensor = unsafe { read_tensor_nd(ptr, &[nrows, ncols], order) }?;
    Ok(sparse_ir::Matrix::from_vec_col_major(
        [nrows, ncols],
        tensor.host_data()?.to_vec(),
    )?)
}

/// Copy an N-dimensional tensor to a C array
///
/// Writes the elements of `tensor` (column-major, logical axes in the
/// caller's order) to `out` in the requested memory order: a column-major
/// output is a plain copy, a row-major one is transposed.
///
/// # Arguments
/// * `tensor` - Source tensor (any rank)
/// * `out` - Destination C array pointer
/// * `order` - Memory layout order for output (RowMajor or ColumnMajor)
///
/// # Safety
/// Caller must ensure `out` has space for `tensor.n_elements()` elements
pub(crate) unsafe fn copy_tensor_to_c_array<T: sparse_ir::TensorScalar + Copy>(
    tensor: sparse_ir::TypedTensor<T>,
    out: *mut T,
    order: MemoryOrder,
) -> sparse_ir::Result<()> {
    let data = tensor.host_data()?;
    let dims = tensor.shape().to_vec();
    match order {
        // SAFETY: the caller guarantees `out` is valid for `data.len()`
        // writes, and `out` cannot alias the freshly computed tensor.
        MemoryOrder::ColumnMajor => unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), out, data.len())
        },
        MemoryOrder::RowMajor => {
            let mut idx = vec![0usize; dims.len()];
            for &x in data {
                // SAFETY: as above; the row-major offset is below data.len().
                unsafe { *out.add(row_major_offset(&idx, &dims)) = x };
                next_col_major(&mut idx, &dims);
            }
        }
    }
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

/// Number of elements of a dense array of `T` with extents `dims`.
///
/// Returns `None` if the element count overflows `usize` or the array would
/// span more than `isize::MAX` bytes, the size limit of any Rust allocation,
/// slice or tensor view.
pub(crate) fn checked_len<T>(dims: &[usize]) -> Option<usize> {
    let len = dims.iter().try_fold(1usize, |len, &d| len.checked_mul(d))?;
    let bytes = len.checked_mul(std::mem::size_of::<T>())?;
    (bytes <= isize::MAX as usize).then_some(len)
}

/// Validate the extents of a C API array of `T` and convert them to `usize`.
///
/// Every extent must be positive and the whole array must pass
/// [`checked_len`]. Run this before an extent reaches a slice, view or
/// allocation: a negative `c_int` cast to `usize` wraps to a huge value.
///
/// # Errors
/// `SPIR_INVALID_DIMENSION` if an extent is zero or negative, or if the
/// element count or byte size of the array is not representable.
pub(crate) fn validate_dims<T>(dims: &[libc::c_int]) -> Result<Vec<usize>, StatusCode> {
    let dims = dims
        .iter()
        .map(|&d| usize::try_from(d).ok().filter(|&d| d > 0))
        .collect::<Option<Vec<usize>>>()
        .ok_or(SPIR_INVALID_DIMENSION)?;
    checked_len::<T>(&dims).ok_or(SPIR_INVALID_DIMENSION)?;
    Ok(dims)
}

/// Validated column-major shapes of a transform along one axis of an array.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct TransformDims {
    /// Input extents in column-major order
    pub input: Vec<usize>,
    /// Output extents: `input` with the target extent replaced
    pub output: Vec<usize>,
    /// Target axis in column-major order
    pub target_dim: usize,
}

/// Validate the shape of a transform that maps `n_in` elements of `Tin` along
/// `target_dim` to `n_out` elements of `Tout`, and convert it to column-major
/// order. `input_dims` and `target_dim` are given in the caller's `order`.
///
/// # Errors
/// * `SPIR_INVALID_ARGUMENT` if `target_dim` is not an axis of `input_dims`
/// * `SPIR_INVALID_DIMENSION` if [`validate_dims`] rejects `input_dims`, or if
///   the output array does not pass [`checked_len`]
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `input_dims[target_dim] != n_in`
pub(crate) fn validate_transform_dims<Tin, Tout>(
    input_dims: &[libc::c_int],
    target_dim: usize,
    order: MemoryOrder,
    n_in: usize,
    n_out: usize,
) -> Result<TransformDims, StatusCode> {
    if target_dim >= input_dims.len() {
        return Err(SPIR_INVALID_ARGUMENT);
    }
    let dims = validate_dims::<Tin>(input_dims)?;
    transform_dims::<Tout>(&dims, target_dim, order, n_in, n_out)
}

/// [`validate_transform_dims`] for extents that [`validate_dims`] has
/// accepted, with `target_dim` an axis of `dims`
///
/// # Errors
/// * `SPIR_INPUT_DIMENSION_MISMATCH` if `dims[target_dim] != n_in`
/// * `SPIR_INVALID_DIMENSION` if the output array does not pass
///   [`checked_len`]
pub(crate) fn transform_dims<Tout>(
    dims: &[usize],
    target_dim: usize,
    order: MemoryOrder,
    n_in: usize,
    n_out: usize,
) -> Result<TransformDims, StatusCode> {
    let (input, target_dim) = convert_dims_for_col_major(dims, target_dim, order);
    if input[target_dim] != n_in {
        return Err(SPIR_INPUT_DIMENSION_MISMATCH);
    }
    let output = build_output_dims(&input, target_dim, n_out);
    checked_len::<Tout>(&output).ok_or(SPIR_INVALID_DIMENSION)?;
    Ok(TransformDims {
        input,
        output,
        target_dim,
    })
}

/// Column-major strides of an array of `dims`
///
/// The extents must pass [`checked_len`], so every stride fits `isize`.
fn col_major_strides(dims: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut acc: usize = 1;
    for &d in dims {
        strides.push(acc as isize);
        acc *= d;
    }
    strides
}

/// Create an immutable column-major view of a raw buffer
///
/// Zero-copy: directly interprets the buffer as a column-major tensor with
/// the given dimensions. For row-major data, pass reversed dimensions (via
/// [`convert_dims_for_col_major`]).
///
/// # Safety
/// - `ptr` must be valid and point to at least `product(dims)` elements
/// - `dims` must pass [`checked_len`] for `T` (C API entry points establish this
///   with [`validate_transform_dims`])
/// - The memory must remain valid for the lifetime of the returned view
pub(crate) unsafe fn create_view_from_ptr<'a, T: sparse_ir::TensorScalar>(
    ptr: *const T,
    dims: &[usize],
) -> sparse_ir::Result<sparse_ir::TypedTensorView<'a, T>> {
    let len: usize = dims.iter().product();
    // SAFETY: the caller guarantees `ptr` is valid for `len` reads for `'a`.
    let data = unsafe { std::slice::from_raw_parts(ptr, len) };
    Ok(sparse_ir::TypedTensorView::from_slice(
        dims,
        col_major_strides(dims),
        0,
        data,
    )?)
}

/// Create a mutable column-major view of a raw buffer
///
/// Zero-copy: directly interprets the buffer as a mutable column-major
/// tensor with the given dimensions. For row-major data, pass reversed
/// dimensions (via [`convert_dims_for_col_major`]).
///
/// # Safety
/// - `ptr` must be valid and point to at least `product(dims)` elements
/// - `dims` must pass [`checked_len`] for `T` (C API entry points establish this
///   with [`validate_transform_dims`])
/// - The memory must remain valid for the lifetime of the returned view
/// - The caller must ensure no aliasing occurs
pub(crate) unsafe fn create_view_mut_from_ptr<'a, T: sparse_ir::TensorScalar>(
    ptr: *mut T,
    dims: &[usize],
) -> sparse_ir::Result<sparse_ir::TypedTensorViewMut<'a, T>> {
    let len: usize = dims.iter().product();
    // SAFETY: the caller guarantees `ptr` is valid, unaliased and writable
    // for `len` elements for `'a`.
    let data = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
    Ok(sparse_ir::TypedTensorViewMut::from_slice(
        dims,
        col_major_strides(dims),
        0,
        data,
    )?)
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
/// * `segments` - Array of segment boundaries (n_segments + 1 elements):
///   finite and strictly increasing, with finite segment lengths and a
///   finite sum of the ends of each segment
/// * `n_segments` - Number of segments (must be >= 1)
/// * `x` - Output array for Gauss points (size n * n_segments). Must be pre-allocated.
/// * `w` - Output array for Gauss weights (size n * n_segments). Must be pre-allocated.
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Status code (also written to `*status`):
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if a pointer is NULL, `n` or `n_segments` < 1,
///   or `segments` does not meet the conditions above
/// - SPIR_INTERNAL_ERROR if an internal error occurs
#[unsafe(no_mangle)]
pub extern "C" fn spir_gauss_legendre_rule_piecewise_double(
    n: libc::c_int,
    segments: *const f64,
    n_segments: libc::c_int,
    x: *mut f64,
    w: *mut f64,
    status: *mut crate::StatusCode,
) -> crate::StatusCode {
    use crate::status::status_from;
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

        // Generate base rule with DDouble precision, then convert to double
        let rule_dd = legendre::<sparse_ir::Df64>(n as usize);
        let rule = sparse_ir::gauss::Rule::from_vectors(
            rule_dd.x().iter().map(|&x| x.to_f64()).collect(),
            rule_dd.w().iter().map(|&w| w.to_f64()).collect(),
            rule_dd.a().to_f64(),
            rule_dd.b().to_f64(),
        )
        .expect("a Gauss-Legendre rule has one weight per point");

        // Create piecewise rule; the core rejects boundaries that are not
        // finite and strictly increasing, and segment lengths that overflow
        let piecewise_rule = match rule.piecewise(&segs_vec) {
            Ok(rule) => rule,
            Err(e) => {
                let code = status_from(&e);
                unsafe {
                    *status = code;
                }
                return code;
            }
        };

        // Copy to output arrays
        for i in 0..piecewise_rule.x().len() {
            unsafe {
                *x.add(i) = piecewise_rule.x()[i];
                *w.add(i) = piecewise_rule.w()[i];
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
/// * `segments` - Array of segment boundaries (n_segments + 1 elements):
///   finite and strictly increasing, with finite segment lengths and a
///   finite sum of the ends of each segment
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
/// Status code (also written to `*status`):
/// - SPIR_COMPUTATION_SUCCESS (0) on success
/// - SPIR_INVALID_ARGUMENT if a pointer is NULL, `n` or `n_segments` < 1,
///   or `segments` does not meet the conditions above
/// - SPIR_INTERNAL_ERROR if an internal error occurs
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
    use crate::status::status_from;
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

        // Generate base rule with DDouble precision
        let rule_dd = legendre::<sparse_ir::Df64>(n as usize);

        // Create piecewise rule; the core rejects boundaries that are not
        // finite and strictly increasing, and segment lengths that overflow
        let piecewise_rule = match rule_dd.piecewise(&segs_vec) {
            Ok(rule) => rule,
            Err(e) => {
                let code = status_from(&e);
                unsafe {
                    *status = code;
                }
                return code;
            }
        };

        // Extract high and low parts
        for i in 0..piecewise_rule.x().len() {
            unsafe {
                *x_high.add(i) = piecewise_rule.x()[i].hi();
                *x_low.add(i) = piecewise_rule.x()[i].lo();
                *w_high.add(i) = piecewise_rule.w()[i].hi();
                *w_low.add(i) = piecewise_rule.w()[i].lo();
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
    fn test_checked_len() {
        assert_eq!(checked_len::<f64>(&[3, 4, 5]), Some(60));
        assert_eq!(checked_len::<f64>(&[usize::MAX, 2]), None);
        // The byte size, not just the element count, must be addressable.
        assert_eq!(checked_len::<f64>(&[(1 << 60) - 1]), Some((1 << 60) - 1));
        assert_eq!(checked_len::<f64>(&[1 << 60]), None);
        assert_eq!(
            checked_len::<u8>(&[isize::MAX as usize]),
            Some(isize::MAX as usize)
        );
        assert_eq!(checked_len::<u8>(&[isize::MAX as usize + 1]), None);
    }

    #[test]
    fn test_validate_dims() {
        use num_complex::Complex64;

        assert_eq!(validate_dims::<f64>(&[3, 4]), Ok(vec![3, 4]));
        assert_eq!(validate_dims::<f64>(&[1]), Ok(vec![1]));

        // Zero or negative extents on any axis
        for dims in [
            &[-1, 4][..],
            &[3, -1],
            &[3, i32::MIN],
            &[0, 4],
            &[3, 0],
            &[0],
        ] {
            assert_eq!(
                validate_dims::<f64>(dims),
                Err(SPIR_INVALID_DIMENSION),
                "dims = {dims:?}"
            );
        }

        // The element count overflows usize
        assert_eq!(
            validate_dims::<f64>(&[i32::MAX, i32::MAX, i32::MAX]),
            Err(SPIR_INVALID_DIMENSION)
        );

        // 2^59 elements: 2^62 bytes of f64 are addressable, 2^63 bytes of
        // Complex64 are not.
        let dims = [1 << 30, 1 << 29];
        assert_eq!(validate_dims::<f64>(&dims), Ok(vec![1 << 30, 1 << 29]));
        assert_eq!(
            validate_dims::<Complex64>(&dims),
            Err(SPIR_INVALID_DIMENSION)
        );
    }

    #[test]
    fn test_validate_transform_dims() {
        use MemoryOrder::{ColumnMajor, RowMajor};
        use num_complex::Complex64;

        // Column-major shapes are kept; the output replaces the target extent.
        assert_eq!(
            validate_transform_dims::<f64, f64>(&[5, 3, 2], 1, ColumnMajor, 3, 7),
            Ok(TransformDims {
                input: vec![5, 3, 2],
                output: vec![5, 7, 2],
                target_dim: 1,
            })
        );
        // Row-major shapes are reversed and the target axis is flipped.
        assert_eq!(
            validate_transform_dims::<f64, f64>(&[5, 3, 2], 0, RowMajor, 5, 7),
            Ok(TransformDims {
                input: vec![2, 3, 5],
                output: vec![2, 3, 7],
                target_dim: 2,
            })
        );

        // Malformed extents are rejected before the target extent is compared.
        for (dims, target_dim, order) in [
            (&[-1, 2][..], 0, RowMajor),
            (&[0, 2], 0, RowMajor),
            (&[3, -2], 0, ColumnMajor),
            (&[3, 0], 0, ColumnMajor),
        ] {
            assert_eq!(
                validate_transform_dims::<f64, f64>(dims, target_dim, order, 3, 7),
                Err(SPIR_INVALID_DIMENSION),
                "dims = {dims:?}"
            );
        }

        // A well-formed shape with the wrong target extent
        assert_eq!(
            validate_transform_dims::<f64, f64>(&[4, 2], 0, RowMajor, 3, 7),
            Err(SPIR_INPUT_DIMENSION_MISMATCH)
        );
        // A target axis outside the shape
        assert_eq!(
            validate_transform_dims::<f64, f64>(&[3, 2], 2, RowMajor, 3, 7),
            Err(SPIR_INVALID_ARGUMENT)
        );

        // The output can be unaddressable even when the input is not: 2^59 f64
        // inputs (2^62 bytes) become 2^61 f64 outputs (2^64 bytes) for n_out = 4,
        // and 2^59 Complex64 outputs (2^63 bytes) for n_out = 1.
        let dims = [1, 1 << 29, 1 << 30];
        assert!(validate_transform_dims::<f64, f64>(&dims, 0, RowMajor, 1, 1).is_ok());
        assert_eq!(
            validate_transform_dims::<f64, f64>(&dims, 0, RowMajor, 1, 4),
            Err(SPIR_INVALID_DIMENSION)
        );
        assert_eq!(
            validate_transform_dims::<f64, Complex64>(&dims, 0, RowMajor, 1, 1),
            Err(SPIR_INVALID_DIMENSION)
        );
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

    /// A NaN or infinite boundary, or a segment length or midpoint that
    /// overflows, passed the `<=` check; the core then panicked sorting NaN
    /// points (SPIR_INTERNAL_ERROR) or returned NaN or infinite points. They
    /// are invalid arguments now, for both precisions.
    #[test]
    fn test_gauss_legendre_rule_piecewise_rejects_non_finite_segments() {
        for segments in [
            [0.0, f64::NAN],
            [0.0, f64::INFINITY],
            [-1e308, 1e308],
            [1e308, 1.7e308],
        ] {
            for n in [1, 3] {
                let len = n as usize;
                let (mut x, mut w) = (vec![0.0; len], vec![0.0; len]);
                let mut status = SPIR_INTERNAL_ERROR;
                let result = spir_gauss_legendre_rule_piecewise_double(
                    n,
                    segments.as_ptr(),
                    1,
                    x.as_mut_ptr(),
                    w.as_mut_ptr(),
                    &mut status,
                );
                assert_eq!(
                    (result, status),
                    (SPIR_INVALID_ARGUMENT, SPIR_INVALID_ARGUMENT),
                    "double, {segments:?}, n = {n}"
                );

                let (mut xh, mut xl) = (vec![0.0; len], vec![0.0; len]);
                let (mut wh, mut wl) = (vec![0.0; len], vec![0.0; len]);
                let mut status = SPIR_INTERNAL_ERROR;
                let result = spir_gauss_legendre_rule_piecewise_ddouble(
                    n,
                    segments.as_ptr(),
                    1,
                    xh.as_mut_ptr(),
                    xl.as_mut_ptr(),
                    wh.as_mut_ptr(),
                    wl.as_mut_ptr(),
                    &mut status,
                );
                assert_eq!(
                    (result, status),
                    (SPIR_INVALID_ARGUMENT, SPIR_INVALID_ARGUMENT),
                    "ddouble, {segments:?}, n = {n}"
                );
            }
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
            let tensor =
                unsafe { read_tensor_nd(data.as_ptr(), &[3, 4], MemoryOrder::RowMajor) }.unwrap();

            let shape_dims = tensor.shape().to_vec();
            assert_eq!(shape_dims, &[3, 4]);
            assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
            assert_eq!(*tensor.get(&[0, 3]).unwrap(), 4.0);
            assert_eq!(*tensor.get(&[1, 0]).unwrap(), 5.0);
            assert_eq!(*tensor.get(&[2, 3]).unwrap(), 12.0);
        }

        // Test 3D tensor: 2x3x4
        {
            let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
            let tensor =
                unsafe { read_tensor_nd(data.as_ptr(), &[2, 3, 4], MemoryOrder::RowMajor) }
                    .unwrap();

            let shape_dims = tensor.shape().to_vec();
            assert_eq!(shape_dims, &[2, 3, 4]);
            assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), 1.0);
            assert_eq!(*tensor.get(&[0, 0, 3]).unwrap(), 4.0);
            assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 5.0);
            assert_eq!(*tensor.get(&[1, 2, 3]).unwrap(), 24.0);
        }

        // Test complex numbers
        {
            let data = vec![
                Complex64::new(1.0, 2.0),
                Complex64::new(3.0, 4.0),
                Complex64::new(5.0, 6.0),
                Complex64::new(7.0, 8.0),
            ];
            let tensor =
                unsafe { read_tensor_nd(data.as_ptr(), &[2, 2], MemoryOrder::RowMajor) }.unwrap();

            let shape_dims = tensor.shape().to_vec();
            assert_eq!(shape_dims, &[2, 2]);
            assert_eq!(*tensor.get(&[0, 0]).unwrap(), Complex64::new(1.0, 2.0));
            assert_eq!(*tensor.get(&[1, 1]).unwrap(), Complex64::new(7.0, 8.0));
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
            let tensor =
                unsafe { read_tensor_nd(data.as_ptr(), &[3, 4], MemoryOrder::ColumnMajor) }
                    .unwrap();

            let shape_dims = tensor.shape().to_vec();
            assert_eq!(shape_dims, &[3, 4]);
            // After reading as [4, 3] (reversed) and permuting back, we should get:
            // [[1, 4, 7, 10],
            //  [2, 5, 8, 11],
            //  [3, 6, 9, 12]]
            assert_eq!(*tensor.get(&[0, 0]).unwrap(), 1.0);
            assert_eq!(*tensor.get(&[0, 1]).unwrap(), 4.0);
            assert_eq!(*tensor.get(&[0, 3]).unwrap(), 10.0);
            assert_eq!(*tensor.get(&[1, 0]).unwrap(), 2.0);
            assert_eq!(*tensor.get(&[2, 3]).unwrap(), 12.0);
        }

        // Test 3D tensor: 2x3x4
        // Column-major: first all elements with index [0,0,0], [1,0,0], then [0,1,0], [1,1,0], etc.
        {
            // For 2x3x4, column-major order:
            // [0,0,0]=1, [1,0,0]=2, [0,1,0]=3, [1,1,0]=4, [0,2,0]=5, [1,2,0]=6,
            // [0,0,1]=7, [1,0,1]=8, ...
            let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
            let tensor =
                unsafe { read_tensor_nd(data.as_ptr(), &[2, 3, 4], MemoryOrder::ColumnMajor) }
                    .unwrap();

            let shape_dims = tensor.shape().to_vec();
            assert_eq!(shape_dims, &[2, 3, 4]);
            // Verify first few elements
            assert_eq!(*tensor.get(&[0, 0, 0]).unwrap(), 1.0);
            assert_eq!(*tensor.get(&[1, 0, 0]).unwrap(), 2.0);
            assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 3.0);
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
            let tensor =
                unsafe { read_tensor_nd(data.as_ptr(), &[2, 2], MemoryOrder::ColumnMajor) }
                    .unwrap();

            let shape_dims = tensor.shape().to_vec();
            assert_eq!(shape_dims, &[2, 2]);
            // After permute: [[1+2i, 5+6i], [3+4i, 7+8i]]
            assert_eq!(*tensor.get(&[0, 0]).unwrap(), Complex64::new(1.0, 2.0));
            assert_eq!(*tensor.get(&[1, 0]).unwrap(), Complex64::new(3.0, 4.0));
            assert_eq!(*tensor.get(&[0, 1]).unwrap(), Complex64::new(5.0, 6.0));
            assert_eq!(*tensor.get(&[1, 1]).unwrap(), Complex64::new(7.0, 8.0));
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
        let row_tensor =
            unsafe { read_tensor_nd(row_major_data.as_ptr(), &[3, 4], MemoryOrder::RowMajor) }
                .unwrap();

        // Create the same matrix in column-major (transposed storage)
        // [[1, 4, 7, 10], [2, 5, 8, 11], [3, 6, 9, 12]] stored as:
        // [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] (column-major)
        let col_major_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let col_tensor =
            unsafe { read_tensor_nd(col_major_data.as_ptr(), &[3, 4], MemoryOrder::ColumnMajor) }
                .unwrap();

        // They should have the same shape
        let row_shape = row_tensor.shape().to_vec();
        let col_shape = col_tensor.shape().to_vec();
        assert_eq!(row_shape, col_shape);

        // But different values (because storage order is different)
        // row_tensor: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
        // col_tensor: [[1,4,7,10], [2,5,8,11], [3,6,9,12]]
        assert_eq!(*row_tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*col_tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*row_tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(*col_tensor.get(&[0, 1]).unwrap(), 4.0); // Different!
    }

    /// copy_tensor_to_c_array writes the inverse of read_tensor_nd in both
    /// memory orders
    #[test]
    fn test_copy_tensor_to_c_array_roundtrip() {
        let data: Vec<f64> = (1..=24).map(|x| x as f64).collect();
        for order in [MemoryOrder::RowMajor, MemoryOrder::ColumnMajor] {
            let tensor = unsafe { read_tensor_nd(data.as_ptr(), &[2, 3, 4], order) }.unwrap();
            let mut out = vec![0.0; data.len()];
            unsafe { copy_tensor_to_c_array(tensor, out.as_mut_ptr(), order) }.unwrap();
            assert_eq!(out, data, "{order:?}");
        }
        // A row-major read of [2, 3, 4] puts element [1, 2, 3] at the end
        let tensor =
            unsafe { read_tensor_nd(data.as_ptr(), &[2, 3, 4], MemoryOrder::RowMajor) }.unwrap();
        assert_eq!(*tensor.get(&[1, 2, 3]).unwrap(), 24.0);
        assert_eq!(*tensor.get(&[0, 1, 0]).unwrap(), 5.0);
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
