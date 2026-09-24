//! Shared machinery for fitters.
//!
//! Every fitter operation is a contraction of a stored matrix with one axis of
//! a column-major tensor. The tensor is addressed as `[pre, n, post]`, where
//! `pre` is the product of the extents before the target axis and `post` the
//! product of those after it, so no data movement is needed to bring the
//! target axis to the front.
//!
//! This module provides the `InplaceFitter` trait, layout validation for
//! tenferro tensors and views, and the SVD-based least-squares factors.

use crate::error::{Error, Result};
use crate::fpu_check::FpuGuard;
use crate::gemm::{GemmBackendHandle, GemmScalar, apply_along_axis};
use num_complex::Complex;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

// ============================================================================
// InplaceFitter trait
// ============================================================================

/// Evaluation and fitting along one axis of an N-dimensional tensor, writing
/// into a caller-provided output view.
///
/// Method suffixes follow BLAS naming (`d` = `f64`, `z` = `Complex<f64>`):
/// `evaluate_nd_dz_to` maps real coefficients to complex values.
///
/// Inputs may have any strided host layout (non-contiguous inputs are copied
/// once). Outputs must be host-resident compact column-major views whose
/// shape equals the input shape with axis `dim` replaced by the output
/// extent.
///
/// # Errors
/// Every method returns [`Error::Unsupported`] when the scalar combination is
/// not provided by the implementor (the default), [`Error::AxisOutOfRange`]
/// or [`Error::ShapeMismatch`] for inconsistent extents, and
/// [`Error::Layout`] for an output view that is not compact column-major.
pub trait InplaceFitter {
    /// Number of sampling points.
    fn n_points(&self) -> usize;

    /// Number of basis functions.
    fn basis_size(&self) -> usize;

    /// Evaluate: `f64` coefficients to `f64` values.
    fn evaluate_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        let _ = (backend, coeffs, dim, out);
        Err(Error::Unsupported("evaluate_nd_dd_to"))
    }

    /// Evaluate: `f64` coefficients to `Complex<f64>` values.
    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        let _ = (backend, coeffs, dim, out);
        Err(Error::Unsupported("evaluate_nd_dz_to"))
    }

    /// Evaluate: `Complex<f64>` coefficients to `f64` values.
    fn evaluate_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        let _ = (backend, coeffs, dim, out);
        Err(Error::Unsupported("evaluate_nd_zd_to"))
    }

    /// Evaluate: `Complex<f64>` coefficients to `Complex<f64>` values.
    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        let _ = (backend, coeffs, dim, out);
        Err(Error::Unsupported("evaluate_nd_zz_to"))
    }

    /// Fit: `f64` values to `f64` coefficients.
    fn fit_nd_dd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        let _ = (backend, values, dim, out);
        Err(Error::Unsupported("fit_nd_dd_to"))
    }

    /// Fit: `f64` values to `Complex<f64>` coefficients.
    fn fit_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, f64>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        let _ = (backend, values, dim, out);
        Err(Error::Unsupported("fit_nd_dz_to"))
    }

    /// Fit: `Complex<f64>` values to `f64` coefficients.
    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, f64>,
    ) -> Result<()> {
        let _ = (backend, values, dim, out);
        Err(Error::Unsupported("fit_nd_zd_to"))
    }

    /// Fit: `Complex<f64>` values to `Complex<f64>` coefficients.
    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &TypedTensorView<'_, Complex<f64>>,
        dim: usize,
        out: &mut TypedTensorViewMut<'_, Complex<f64>>,
    ) -> Result<()> {
        let _ = (backend, values, dim, out);
        Err(Error::Unsupported("fit_nd_zz_to"))
    }
}

// ============================================================================
// Scalar reinterpretation
// ============================================================================

/// `f64` or `Complex<f64>`, viewable as interleaved `f64` storage.
///
/// A column-major complex tensor of shape `[pre, n, post]` is a real tensor
/// of shape `[2 * pre, n, post]`, so real-matrix contractions apply to it
/// with `pre` doubled.
pub trait FitScalar: GemmScalar {
    /// Number of `f64` values per element (1 or 2).
    const REAL_WIDTH: usize;
    /// Reinterpret as `f64` storage.
    fn as_f64(s: &[Self]) -> &[f64];
    /// Reinterpret mutably as `f64` storage.
    fn as_f64_mut(s: &mut [Self]) -> &mut [f64];
}

impl FitScalar for f64 {
    const REAL_WIDTH: usize = 1;
    fn as_f64(s: &[f64]) -> &[f64] {
        s
    }
    fn as_f64_mut(s: &mut [f64]) -> &mut [f64] {
        s
    }
}

impl FitScalar for Complex<f64> {
    const REAL_WIDTH: usize = 2;
    fn as_f64(s: &[Self]) -> &[f64] {
        // SAFETY: `Complex<f64>` is `#[repr(C)] { re: f64, im: f64 }`, so a
        // slice of N complex values is a valid slice of 2N f64 values with
        // the same lifetime and alignment.
        unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<f64>(), 2 * s.len()) }
    }
    fn as_f64_mut(s: &mut [Self]) -> &mut [f64] {
        // SAFETY: as above; the unique borrow is transferred.
        unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr().cast::<f64>(), 2 * s.len()) }
    }
}

// ============================================================================
// Axis geometry and layout validation
// ============================================================================

/// Extents of a tensor split around the target axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AxisSplit {
    pub pre: usize,
    pub post: usize,
}

/// Split `shape` around `dim` after checking `shape[dim] == expected`.
pub(crate) fn split_axis(shape: &[usize], dim: usize, expected: usize) -> Result<AxisSplit> {
    if dim >= shape.len() {
        return Err(Error::AxisOutOfRange {
            dim,
            rank: shape.len(),
        });
    }
    if shape[dim] != expected {
        return Err(Error::ShapeMismatch(format!(
            "axis {dim} has extent {}, expected {expected} (shape {shape:?})",
            shape[dim]
        )));
    }
    Ok(AxisSplit {
        pre: shape[..dim].iter().product(),
        post: shape[dim + 1..].iter().product(),
    })
}

/// `shape` with axis `dim` replaced by `extent`.
pub(crate) fn replace_axis(shape: &[usize], dim: usize, extent: usize) -> Vec<usize> {
    let mut out = shape.to_vec();
    out[dim] = extent;
    out
}

/// Compact column-major host slice of a view, copying a strided view once.
pub(crate) enum InputSlice<'a, T: tenferro_tensor::TensorScalar> {
    Borrowed(&'a [T]),
    Owned(Vec<T>),
}

impl<T: tenferro_tensor::TensorScalar> InputSlice<'_, T> {
    pub(crate) fn get(&self) -> Result<&[T]> {
        match self {
            InputSlice::Borrowed(s) => Ok(s),
            InputSlice::Owned(v) => Ok(v),
        }
    }
}

pub(crate) fn input_slice<'a, T: tenferro_tensor::TensorScalar>(
    view: &TypedTensorView<'a, T>,
) -> Result<InputSlice<'a, T>> {
    if view.is_col_major_contiguous()? {
        Ok(InputSlice::Borrowed(view.as_slice()?))
    } else {
        Ok(InputSlice::Owned(gather_col_major(view)?))
    }
}

/// Copy a strided host view into a compact column-major buffer.
fn gather_col_major<T: tenferro_tensor::TensorScalar>(
    view: &TypedTensorView<'_, T>,
) -> Result<Vec<T>> {
    let shape = view.shape();
    let strides = view.strides();
    let storage = view.host_storage()?;
    let len = view.n_elements();
    if len == 0 {
        return Ok(Vec::new());
    }
    // Validate the reachable offset range once, before allocating.
    let (mut lo, mut hi) = (view.offset(), view.offset());
    for (&n, &st) in shape.iter().zip(strides) {
        let span = isize::try_from(n - 1)
            .ok()
            .and_then(|m| m.checked_mul(st))
            .ok_or_else(|| Error::Layout("input view stride overflow".into()))?;
        if span < 0 {
            lo += span;
        } else {
            hi += span;
        }
    }
    if lo < 0 || usize::try_from(hi).map_or(true, |h| h >= storage.len()) {
        return Err(Error::Layout("input view exceeds its storage".into()));
    }
    let mut out = Vec::with_capacity(len);
    let mut idx = vec![0usize; shape.len()];
    let mut pos = view.offset();
    for _ in 0..len {
        // pos is within [lo, hi] by the range check above.
        out.push(storage[pos as usize].clone());
        for (k, i) in idx.iter_mut().enumerate() {
            *i += 1;
            pos += strides[k];
            if *i < shape[k] {
                break;
            }
            pos -= strides[k] * shape[k] as isize;
            *i = 0;
        }
    }
    Ok(out)
}

/// Compact column-major host slice of an output view.
pub(crate) fn output_slice<'a, T: tenferro_tensor::TensorScalar>(
    view: &'a mut TypedTensorViewMut<'_, T>,
) -> Result<&'a mut [T]> {
    if !view.is_col_major_contiguous()? {
        return Err(Error::Layout(format!(
            "output view must be compact column-major, got {}",
            view.layout_summary()
        )));
    }
    let offset = usize::try_from(view.offset())
        .map_err(|_| Error::Layout("output view has a negative offset".into()))?;
    let len = view.n_elements();
    let storage = view
        .host_storage_mut()
        .map_err(|e| Error::Layout(e.to_string()))?;
    storage
        .get_mut(offset..offset + len)
        .ok_or_else(|| Error::Layout("output view exceeds its storage".into()))
}

/// Run an axis operation `n_in -> n_out` from `input` into `out`.
///
/// `op` receives compact column-major `[pre, n_in, post]` and
/// `[pre, n_out, post]` slices.
pub(crate) fn run_to<Tin, Tout, F>(
    input: &TypedTensorView<'_, Tin>,
    dim: usize,
    n_in: usize,
    n_out: usize,
    out: &mut TypedTensorViewMut<'_, Tout>,
    op: F,
) -> Result<()>
where
    Tin: tenferro_tensor::TensorScalar,
    Tout: tenferro_tensor::TensorScalar,
    F: FnOnce(&[Tin], AxisSplit, &mut [Tout]) -> Result<()>,
{
    let split = split_axis(input.shape(), dim, n_in)?;
    let expected = replace_axis(input.shape(), dim, n_out);
    if out.shape() != expected.as_slice() {
        return Err(Error::ShapeMismatch(format!(
            "output shape {:?}, expected {expected:?}",
            out.shape()
        )));
    }
    let x = input_slice(input)?;
    let y = output_slice(out)?;
    op(x.get()?, split, y)
}

/// Run an axis operation `n_in -> n_out` into a newly allocated tensor.
pub(crate) fn run_alloc<Tin, Tout, F>(
    input: &TypedTensor<Tin>,
    dim: usize,
    n_in: usize,
    n_out: usize,
    op: F,
) -> Result<TypedTensor<Tout>>
where
    Tin: tenferro_tensor::TensorScalar,
    Tout: tenferro_tensor::TensorScalar + num_traits::Zero,
    F: FnOnce(&[Tin], AxisSplit, &mut [Tout]) -> Result<()>,
{
    let split = split_axis(input.shape(), dim, n_in)?;
    let out_shape = replace_axis(input.shape(), dim, n_out);
    // Owned tensors are compact column-major by tenferro's invariant, so the
    // host buffer is the tensor in order (and avoids building a view).
    let x = input.host_data()?;
    debug_assert_eq!(x.len(), input.n_elements());
    let mut y = vec![Tout::zero(); split.pre * n_out * split.post];
    op(x, split, &mut y)?;
    Ok(TypedTensor::from_vec_col_major(out_shape, y)?)
}

// ============================================================================
// Complex <-> stacked-real layout helpers
// ============================================================================

/// Convert real `[pre, 2n, post]` (rows `0..n` real parts, `n..2n` imaginary
/// parts) into complex `[pre, n, post]`.
pub(crate) fn stacked_to_complex(
    t: &[f64],
    pre: usize,
    n: usize,
    post: usize,
    y: &mut [Complex<f64>],
) {
    for q in 0..post {
        for i in 0..n {
            let re = &t[pre * (i + 2 * n * q)..][..pre];
            let im = &t[pre * (n + i + 2 * n * q)..][..pre];
            let dst = &mut y[pre * (i + n * q)..][..pre];
            for p in 0..pre {
                dst[p] = Complex::new(re[p], im[p]);
            }
        }
    }
}

/// Convert complex `[pre, n, post]` into real `[pre, 2n, post]` with real
/// parts in rows `0..n` and imaginary parts in rows `n..2n`.
pub(crate) fn complex_to_stacked(
    x: &[Complex<f64>],
    pre: usize,
    n: usize,
    post: usize,
) -> Vec<f64> {
    let mut t = vec![0.0; 2 * pre * n * post];
    for q in 0..post {
        for i in 0..n {
            let src = &x[pre * (i + n * q)..][..pre];
            let base_re = pre * (i + 2 * n * q);
            let base_im = pre * (n + i + 2 * n * q);
            for p in 0..pre {
                t[base_re + p] = src[p].re;
                t[base_im + p] = src[p].im;
            }
        }
    }
    t
}

// ============================================================================
// SVD-based least squares
// ============================================================================

/// Factors of the pseudo-inverse `A^+ = V diag(1/s) U^H` of an `n x m`
/// matrix, stored for axis contractions.
///
/// Rank is `r = min(n, m)` without truncation: sampling matrices are well
/// conditioned by construction.
pub(crate) struct PinvFactors<T> {
    /// `U^H`, contiguous `r x n`.
    pub uh: Vec<T>,
    /// `V diag(1/s)`, contiguous `m x r`.
    pub v_scaled: Vec<T>,
    pub n: usize,
    pub m: usize,
    pub rank: usize,
}

impl<T: GemmScalar> PinvFactors<T> {
    /// Least-squares solve along the middle axis of `[pre, n, post]` into
    /// `[pre, m, post]`.
    pub(crate) fn solve(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[T],
        pre: usize,
        post: usize,
        out: &mut [T],
    ) -> Result<()> {
        let mut tmp = vec![T::zero(); pre * self.rank * post];
        apply_along_axis(
            backend, &self.uh, self.rank, self.n, values, pre, post, &mut tmp,
        )?;
        apply_along_axis(
            backend,
            &self.v_scaled,
            self.m,
            self.rank,
            &tmp,
            pre,
            post,
            out,
        )?;
        Ok(())
    }
}

/// Scalars with an SVD through tenferro-linalg.
pub trait SvdScalar: GemmScalar + tenferro_linalg::LinalgScalar {
    fn conj_(self) -> Self;
    fn scale(self, s: f64) -> Self;
    fn real_part(v: <Self as tenferro_tensor::TensorScalar>::Real) -> f64;
}

impl SvdScalar for f64 {
    fn conj_(self) -> Self {
        self
    }
    fn scale(self, s: f64) -> Self {
        self * s
    }
    fn real_part(v: f64) -> f64 {
        v
    }
}

impl SvdScalar for Complex<f64> {
    fn conj_(self) -> Self {
        self.conj()
    }
    fn scale(self, s: f64) -> Self {
        self * s
    }
    fn real_part(v: f64) -> f64 {
        v
    }
}

/// Thin SVD of a contiguous column-major `n x m` matrix, returned as
/// pseudo-inverse factors.
///
/// # Errors
/// Propagates tenferro errors (for example a non-converging SVD).
pub(crate) fn compute_pinv<T: SvdScalar>(a: &[T], n: usize, m: usize) -> Result<PinvFactors<T>> {
    compute_pinv_impl(a, n, m, None)
}

/// Like [`compute_pinv`], but keeps only singular values `s_l > rtol * s_0`
/// (truncated-SVD regularization).
///
/// # Errors
/// Propagates tenferro errors.
pub(crate) fn compute_pinv_truncated<T: SvdScalar>(
    a: &[T],
    n: usize,
    m: usize,
    rtol: f64,
) -> Result<PinvFactors<T>> {
    compute_pinv_impl(a, n, m, Some(rtol))
}

fn compute_pinv_impl<T: SvdScalar>(
    a: &[T],
    n: usize,
    m: usize,
    rtol: Option<f64>,
) -> Result<PinvFactors<T>> {
    use tenferro_cpu::CpuBackend;
    use tenferro_linalg::TypedTensorLinalgExt;
    use tenferro_tensor::BackendSessionHost;

    // Protect FPU state during SVD computation (required for Intel Fortran compatibility)
    let _guard = FpuGuard::new_protect_computation();

    let rank = n.min(m);
    if rank == 0 {
        return Ok(PinvFactors {
            uh: Vec::new(),
            v_scaled: vec![T::zero(); m * rank],
            n,
            m,
            rank,
        });
    }
    let tensor = TypedTensor::<T>::from_vec_col_major(vec![n, m], a.to_vec())?;
    let mut host = CpuBackend::new();
    let (u, s, vt) = host.with_backend_session(|session| tensor.svd(session))?;

    let (u_shape, vt_shape) = (u.shape().to_vec(), vt.shape().to_vec());
    let (u, s, vt) = (u.host_data()?, s.host_data()?, vt.host_data()?);
    if u_shape[0] != n || u_shape[1] < rank || vt_shape[1] != m || vt_shape[0] < rank {
        return Err(Error::ShapeMismatch(format!(
            "SVD factors U {u_shape:?}, Vt {vt_shape:?} for a {n}x{m} matrix"
        )));
    }
    let (ldu, ldvt) = (u_shape[0], vt_shape[0]);
    let rank = match rtol {
        Some(rtol) => {
            let s0 = T::real_part(s[0]);
            (0..rank)
                .take_while(|&l| T::real_part(s[l]) > rtol * s0)
                .count()
        }
        None => rank,
    };

    // U^H: uh[l + r*i] = conj(U[i, l])
    let mut uh = vec![T::zero(); rank * n];
    for i in 0..n {
        for l in 0..rank {
            uh[l + rank * i] = u[i + ldu * l].conj_();
        }
    }
    // V diag(1/s): V[j, l] = conj(Vt[l, j])
    let mut v_scaled = vec![T::zero(); m * rank];
    for l in 0..rank {
        let inv_s = 1.0 / T::real_part(s[l]);
        for j in 0..m {
            v_scaled[j + m * l] = vt[l + ldvt * j].conj_().scale(inv_s);
        }
    }
    Ok(PinvFactors {
        uh,
        v_scaled,
        n,
        m,
        rank,
    })
}

/// Singular values of a contiguous column-major `n x m` matrix, descending.
///
/// # Errors
/// Propagates tenferro errors.
pub fn singular_values<T: SvdScalar>(a: &[T], n: usize, m: usize) -> Result<Vec<f64>> {
    use tenferro_cpu::CpuBackend;
    use tenferro_linalg::TypedTensorLinalgExt;
    use tenferro_tensor::BackendSessionHost;

    let _guard = FpuGuard::new_protect_computation();
    if n.min(m) == 0 {
        return Ok(Vec::new());
    }
    let tensor = TypedTensor::<T>::from_vec_col_major(vec![n, m], a.to_vec())?;
    let mut host = CpuBackend::new();
    let s = host.with_backend_session(|session| tensor.svdvals(session))?;
    Ok(s.host_data()?.iter().map(|&v| T::real_part(v)).collect())
}
