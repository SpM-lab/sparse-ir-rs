//! Minimal pole representation (MiniPole) API for C
//!
//! Compresses a Green's function into a small number of complex poles and
//! residues, from DLR coefficients (MPM-DLR) or from Matsubara data on a
//! uniform grid (MPM), with the algorithms of Green-Phys/MiniPole.
//!
//! Functions:
//! - Creation: spir_minipole_from_dlr, spir_minipole_from_matsubara
//! - Introspection: spir_pole_repr_get_npoles, spir_pole_repr_get_poles,
//!   spir_pole_repr_get_residues, spir_pole_repr_get_constant,
//!   spir_pole_repr_get_n0, spir_pole_repr_get_err_max
//! - Lifecycle: spir_pole_repr_release, spir_pole_repr_clone,
//!   spir_pole_repr_is_assigned

use num_complex::Complex64;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use crate::status::status_from;
use crate::types::{BasisType, spir_basis};
use crate::utils::{MemoryOrder, convert_dims_for_col_major, validate_dims};
use crate::{
    SPIR_COMPUTATION_SUCCESS, SPIR_INPUT_DIMENSION_MISMATCH, SPIR_INTERNAL_ERROR,
    SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED, StatusCode,
};
use sparse_ir::TypedTensor;
use sparse_ir::esprit::ErrType;
use sparse_ir::minipole::{
    MiniPoleDlrParams, MiniPoleParams, MiniPoleResult, N0, Plane, mini_pole, mini_pole_dlr_from,
};

/// Opaque minimal pole representation for C API
///
/// Holds complex poles `ξ_j`, residues `A_j` and a constant `C` with
/// `G(z) ≈ Σ_j A_j / (z - ξ_j) + C`.
#[repr(C)]
pub struct spir_pole_repr {
    pub(crate) _private: *const std::ffi::c_void,
}

/// Internal pole representation (not exposed to C)
pub(crate) struct PoleReprInner {
    poles: Vec<Complex64>,
    /// Residues in the caller's layout, with the pole axis at `target_dim`.
    residues: Vec<Complex64>,
    /// Constant term in the caller's layout without the target axis.
    constant: Vec<Complex64>,
    n0: usize,
    err_max: Option<f64>,
}

impl spir_pole_repr {
    fn new(inner: PoleReprInner) -> Self {
        Self {
            _private: Box::into_raw(Box::new(Arc::new(inner))) as *const std::ffi::c_void,
        }
    }

    fn inner(&self) -> &PoleReprInner {
        unsafe { &*(self._private as *const Arc<PoleReprInner>) }
    }
}

impl Clone for spir_pole_repr {
    fn clone(&self) -> Self {
        let inner = unsafe { &*(self._private as *const Arc<PoleReprInner>) }.clone();
        Self {
            _private: Box::into_raw(Box::new(inner)) as *const std::ffi::c_void,
        }
    }
}

impl Drop for spir_pole_repr {
    fn drop(&mut self) {
        if !self._private.is_null() {
            unsafe {
                let _ = Box::from_raw(self._private as *mut Arc<PoleReprInner>);
            }
        }
    }
}

/// Releases a pole representation
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_release(rep: *mut spir_pole_repr) {
    if !rep.is_null() {
        unsafe {
            let _ = Box::from_raw(rep);
        }
    }
}

/// Clones a pole representation (shared data, reference counted)
///
/// The returned pointer must be freed with `spir_pole_repr_release()`.
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_clone(src: *const spir_pole_repr) -> *mut spir_pole_repr {
    if src.is_null() {
        return std::ptr::null_mut();
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        Box::into_raw(Box::new((*src).clone()))
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Checks if the pointer is non-null (1) or null (0)
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_is_assigned(obj: *const spir_pole_repr) -> i32 {
    i32::from(!obj.is_null())
}

/// Column-major data with `axis` moved to the front.
fn move_axis_to_front(data: &[Complex64], dims: &[usize], axis: usize) -> Vec<Complex64> {
    let l: usize = dims[..axis].iter().product();
    let n = dims[axis];
    let r: usize = dims[axis + 1..].iter().product();
    let mut out = vec![Complex64::new(0.0, 0.0); data.len()];
    for k in 0..r {
        for j in 0..n {
            for i in 0..l {
                out[j + n * (i + l * k)] = data[i + l * (j + n * k)];
            }
        }
    }
    out
}

/// Inverse of [`move_axis_to_front`]: `data` has shape `[n, rest...]`.
fn move_front_to_axis(data: &[Complex64], n: usize, dims: &[usize], axis: usize) -> Vec<Complex64> {
    let l: usize = dims[..axis].iter().product();
    let r: usize = dims[axis + 1..].iter().product();
    let mut out = vec![Complex64::new(0.0, 0.0); data.len()];
    for k in 0..r {
        for j in 0..n {
            for i in 0..l {
                out[i + l * (j + n * k)] = data[j + n * (i + l * k)];
            }
        }
    }
    out
}

/// Caller array as a column-major tensor with the target axis first.
///
/// Returns the tensor and the column-major dims / target axis needed to
/// restore the caller's layout.
unsafe fn read_axis_first(
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    data: *const Complex64,
) -> Result<(TypedTensor<Complex64>, Vec<usize>, usize), StatusCode> {
    if input_dims.is_null() || data.is_null() || ndim <= 0 || target_dim < 0 || target_dim >= ndim {
        return Err(SPIR_INVALID_ARGUMENT);
    }
    let mem_order = MemoryOrder::from_c_int(order).map_err(|_| SPIR_INVALID_ARGUMENT)?;
    let dims_c = unsafe { std::slice::from_raw_parts(input_dims, ndim as usize) };
    // Positive extents with an addressable element count, as for every
    // array of the C API
    let dims = validate_dims::<Complex64>(dims_c)?;
    let (cm_dims, axis) = convert_dims_for_col_major(&dims, target_dim as usize, mem_order);
    let len: usize = cm_dims.iter().product();
    let src = unsafe { std::slice::from_raw_parts(data, len) };
    let moved = move_axis_to_front(src, &cm_dims, axis);
    let mut shape = vec![cm_dims[axis]];
    shape.extend(cm_dims[..axis].iter().chain(&cm_dims[axis + 1..]));
    let tensor =
        TypedTensor::from_vec_col_major(shape, moved).map_err(|_| SPIR_INVALID_ARGUMENT)?;
    Ok((tensor, cm_dims, axis))
}

fn err_type_code(code: libc::c_int) -> Result<ErrType, StatusCode> {
    match code {
        0 => Ok(ErrType::Abs),
        1 => Ok(ErrType::Rel),
        _ => Err(SPIR_INVALID_ARGUMENT),
    }
}

fn finish(
    rep: MiniPoleResult,
    cm_dims: &[usize],
    axis: usize,
) -> Result<PoleReprInner, StatusCode> {
    let np = rep.pole_location.len();
    let mut out_dims = cm_dims.to_vec();
    out_dims[axis] = np;
    let data = rep
        .pole_weight
        .host_data()
        .map_err(|e| status_from(&e.into()))?;
    Ok(PoleReprInner {
        residues: move_front_to_axis(data, np, &out_dims, axis),
        // The channels in column-major order of the remaining axes are the
        // caller's layout without the target axis.
        constant: rep.constant,
        poles: rep.pole_location,
        n0: rep.n0,
        err_max: rep.err_max,
    })
}

fn return_repr(
    result: std::thread::Result<Result<PoleReprInner, StatusCode>>,
    status: *mut StatusCode,
) -> *mut spir_pole_repr {
    let (ptr, code) = match result {
        Ok(Ok(inner)) => (
            Box::into_raw(Box::new(spir_pole_repr::new(inner))),
            SPIR_COMPUTATION_SUCCESS,
        ),
        Ok(Err(code)) => (std::ptr::null_mut(), code),
        Err(_) => (std::ptr::null_mut(), SPIR_INTERNAL_ERROR),
    };
    if !status.is_null() {
        unsafe {
            *status = code;
        }
    }
    ptr
}

/// Minimal pole representation from DLR coefficients (MPM-DLR)
///
/// The poles `ω_l` of the DLR and the residues `A_l = g_l w_l` (`w_l` the
/// pole weights) are compressed as in `MiniPoleDLR` of Green-Phys/MiniPole.
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order (`SPIR_ORDER_ROW_MAJOR` or `SPIR_ORDER_COLUMN_MAJOR`)
/// * `ndim` - Number of dimensions of `coeffs`
/// * `input_dims` - Dimensions of `coeffs`; `input_dims[target_dim]` must equal the number of DLR poles
/// * `target_dim` - Dimension holding the DLR coefficients
/// * `coeffs` - Complex DLR coefficients `g_l`
/// * `n0` - The contour starts at `(2 n0 + 1)π/β` (>= 0)
/// * `nmax` - The contour ends at `(2 nmax + 1)π/β` without symmetry, or <= 0 for `nmax = β`
/// * `err` - Error tolerance of ESPRIT, or <= 0 to use `n_poles` alone
/// * `err_type` - 0: `err` is absolute, 1: relative to the largest singular value
/// * `n_poles` - Number of poles, or <= 0 to choose it from `err`; one of `err` and `n_poles` is required
/// * `symmetry` - Impose up-down symmetry (gapless map)
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the pole representation, or NULL on failure. Residues share the
/// layout of `coeffs` with `input_dims[target_dim]` replaced by the number of poles.
#[unsafe(no_mangle)]
pub extern "C" fn spir_minipole_from_dlr(
    dlr: *const spir_basis,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    coeffs: *const Complex64,
    n0: libc::c_int,
    nmax: f64,
    err: f64,
    err_type: libc::c_int,
    n_poles: libc::c_int,
    symmetry: bool,
    status: *mut StatusCode,
) -> *mut spir_pole_repr {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<PoleReprInner, StatusCode> {
        if dlr.is_null() || n0 < 0 || !(err.is_finite() && nmax.is_finite()) {
            return Err(SPIR_INVALID_ARGUMENT);
        }
        let params = MiniPoleDlrParams {
            nmax: (nmax > 0.0).then_some(nmax),
            err: (err > 0.0).then_some(err),
            err_type: err_type_code(err_type)?,
            m: (n_poles > 0).then_some(n_poles as usize),
            symmetry,
            ..MiniPoleDlrParams::new(n0 as usize, 0.0)
        };
        let (g, cm_dims, axis) =
            unsafe { read_axis_first(order, ndim, input_dims, target_dim, coeffs)? };
        let basis = unsafe { &*dlr };
        if g.shape()[0] != basis.size() {
            return Err(SPIR_INPUT_DIMENSION_MISMATCH);
        }
        let rep = match basis.inner() {
            BasisType::DLRFermionic(d) => mini_pole_dlr_from(d.as_ref(), &g, &params),
            BasisType::DLRBosonic(d) => mini_pole_dlr_from(d.as_ref(), &g, &params),
            _ => return Err(SPIR_NOT_SUPPORTED),
        }
        .map_err(|e| status_from(&e))?;
        finish(rep, &cm_dims, axis)
    }));
    return_repr(result, status)
}

/// Minimal pole representation from Matsubara data (MPM)
///
/// The data are given on a uniform grid of non-negative Matsubara
/// frequencies `ω_n = nπ/β` and compressed as in `MiniPole` of
/// Green-Phys/MiniPole.
///
/// # Arguments
/// * `beta` - Inverse temperature
/// * `n_freqs` - Number of Matsubara frequencies (>= 3)
/// * `matsubara_indices` - Matsubara indices `n`, non-negative, increasing and uniformly spaced
///   (e.g. `1, 3, 5, ...` for fermions, `0, 2, 4, ...` for bosons)
/// * `order`, `ndim`, `input_dims`, `target_dim` - Layout of `values`; `input_dims[target_dim]`
///   must equal `n_freqs`, and the other dimensions must be none or two equal ones (a matrix)
/// * `values` - Complex values `G(iω_n)`
/// * `n0` - Number of low frequencies left out of the contour, or < 0 to choose it from the data
/// * `n0_shift` - Shift (>= 0) added to the automatic choice of `n0`
/// * `err` - Error tolerance (> 0), at least the noise level of the data
/// * `err_type` - 0: `err` is absolute, 1: relative
/// * `n_poles` - Number of poles, or <= 0 to use the precision of the first approximation
/// * `symmetry` - Preserve up-down symmetry
/// * `g_symmetric` - Symmetrize the data as `G_ij = G_ji`
/// * `compute_const` - Fit a constant term (not with `symmetry`)
/// * `plane` - Pole weights from a least-squares fit to the data (0), from the mapped plane (1),
///   or < 0 for the default (0 without, 1 with `symmetry`)
/// * `include_n0` - Include the first `n0` frequencies in the least-squares fit
/// * `k_max` - Maximum number of contour integrals, or <= 0 for 999
/// * `ratio_max` - Maximum ratio of oscillation when choosing `n0`, or <= 0 for 10
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the pole representation, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn spir_minipole_from_matsubara(
    beta: f64,
    n_freqs: libc::c_int,
    matsubara_indices: *const i64,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    values: *const Complex64,
    n0: libc::c_int,
    n0_shift: libc::c_int,
    err: f64,
    err_type: libc::c_int,
    n_poles: libc::c_int,
    symmetry: bool,
    g_symmetric: bool,
    compute_const: bool,
    plane: libc::c_int,
    include_n0: bool,
    k_max: libc::c_int,
    ratio_max: f64,
    status: *mut StatusCode,
) -> *mut spir_pole_repr {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<PoleReprInner, StatusCode> {
        if matsubara_indices.is_null()
            || n_freqs <= 0
            || n0_shift < 0
            || !(beta.is_finite() && beta > 0.0 && err.is_finite() && err > 0.0)
            || !ratio_max.is_finite()
        {
            return Err(SPIR_INVALID_ARGUMENT);
        }
        let params = MiniPoleParams {
            n0: if n0 < 0 {
                N0::Auto {
                    shift: n0_shift as usize,
                }
            } else {
                N0::Fixed(n0 as usize)
            },
            err_type: err_type_code(err_type)?,
            m: (n_poles > 0).then_some(n_poles as usize),
            symmetry,
            g_symmetric,
            compute_const,
            plane: match plane {
                p if p < 0 => None,
                0 => Some(Plane::Z),
                1 => Some(Plane::W),
                _ => return Err(SPIR_INVALID_ARGUMENT),
            },
            include_n0,
            k_max: if k_max > 0 { k_max as usize } else { 999 },
            ratio_max: if ratio_max > 0.0 { ratio_max } else { 10.0 },
            ..MiniPoleParams::new(err)
        };
        let (v, cm_dims, axis) =
            unsafe { read_axis_first(order, ndim, input_dims, target_dim, values)? };
        if v.shape()[0] != n_freqs as usize {
            return Err(SPIR_INPUT_DIMENSION_MISMATCH);
        }
        let ns = unsafe { std::slice::from_raw_parts(matsubara_indices, n_freqs as usize) };
        let w: Vec<f64> = ns
            .iter()
            .map(|&n| n as f64 * std::f64::consts::PI / beta)
            .collect();
        let rep = mini_pole(&v, &w, &params).map_err(|e| status_from(&e))?;
        finish(rep, &cm_dims, axis)
    }));
    return_repr(result, status)
}

/// Gets the number of poles
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_npoles(
    rep: *const spir_pole_repr,
    num_poles: *mut libc::c_int,
) -> StatusCode {
    if rep.is_null() || num_poles.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        *num_poles = (*rep).inner().poles.len() as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));
    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the complex poles (array of length npoles), sorted by real part
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_poles(
    rep: *const spir_pole_repr,
    poles: *mut Complex64,
) -> StatusCode {
    if rep.is_null() || poles.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let p = &(*rep).inner().poles;
        std::ptr::copy_nonoverlapping(p.as_ptr(), poles, p.len());
        SPIR_COMPUTATION_SUCCESS
    }));
    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the residues
///
/// The output has the layout (order, dims) of the input passed at creation,
/// with `input_dims[target_dim]` replaced by the number of poles.
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_residues(
    rep: *const spir_pole_repr,
    residues: *mut Complex64,
) -> StatusCode {
    if rep.is_null() || residues.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let r = &(*rep).inner().residues;
        std::ptr::copy_nonoverlapping(r.as_ptr(), residues, r.len());
        SPIR_COMPUTATION_SUCCESS
    }));
    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the constant term `C`
///
/// The output has the layout of the input passed at creation without the
/// target dimension (one value for scalar data). It is zero unless
/// `compute_const` was set.
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_constant(
    rep: *const spir_pole_repr,
    constant: *mut Complex64,
) -> StatusCode {
    if rep.is_null() || constant.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        let c = &(*rep).inner().constant;
        std::ptr::copy_nonoverlapping(c.as_ptr(), constant, c.len());
        SPIR_COMPUTATION_SUCCESS
    }));
    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets `n0`, the start of the contour (given or chosen from the data)
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_n0(
    rep: *const spir_pole_repr,
    n0: *mut libc::c_int,
) -> StatusCode {
    if rep.is_null() || n0.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        *n0 = (*rep).inner().n0 as libc::c_int;
        SPIR_COMPUTATION_SUCCESS
    }));
    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

/// Gets the precision of the first approximation of the Matsubara data
///
/// Available only for representations built by `spir_minipole_from_matsubara`;
/// otherwise returns `SPIR_NOT_SUPPORTED`.
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_err_max(
    rep: *const spir_pole_repr,
    err_max: *mut f64,
) -> StatusCode {
    if rep.is_null() || err_max.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        match (*rep).inner().err_max {
            Some(e) => {
                *err_max = e;
                SPIR_COMPUTATION_SUCCESS
            }
            None => SPIR_NOT_SUPPORTED,
        }
    }));
    result.unwrap_or(SPIR_INTERNAL_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SPIR_ORDER_COLUMN_MAJOR, SPIR_ORDER_ROW_MAJOR, spir_basis_release};

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn axis_moves_roundtrip() {
        let dims = [2, 3, 4];
        let data: Vec<Complex64> = (0..24).map(|i| c(i as f64, 0.0)).collect();
        for axis in 0..3 {
            let f = move_axis_to_front(&data, &dims, axis);
            assert_eq!(move_front_to_axis(&f, dims[axis], &dims, axis), data);
        }
        // axis 1 to front: element (i, j, k) -> (j, i, k)
        let f = move_axis_to_front(&data, &dims, 1);
        assert_eq!(f[2 + 3 * (1 + 2 * 3)], data[1 + 2 * (2 + 3 * 3)]);
    }

    fn green(amps: &[[f64; 2]; 2], xs: &[f64; 2], beta: f64, ch: usize, n: i64) -> Complex64 {
        let z = c(0.0, n as f64 * std::f64::consts::PI / beta);
        (0..2).map(|j| amps[ch][j] / (z - xs[j])).sum()
    }

    /// Two-channel data along axis 1 of a row-major [2, nf] array.
    #[test]
    fn minipole_from_matsubara_capi() {
        let beta = 40.0;
        let xs = [-0.5, 0.3];
        let amps = [[0.4, 0.6], [0.1, 0.2]]; // [channel][pole]
        let ns: Vec<i64> = (0..120).map(|k| 2 * k + 1).collect();
        let nf = ns.len();
        let mut row = vec![c(0.0, 0.0); 2 * nf];
        for ch in 0..2 {
            for (i, &n) in ns.iter().enumerate() {
                row[ch * nf + i] = green(&amps, &xs, beta, ch, n);
            }
        }
        let mut st = 0;
        let call = |n0: i32, err: f64, err_type: i32, plane: i32, st: &mut StatusCode| {
            spir_minipole_from_matsubara(
                beta,
                nf as i32,
                ns.as_ptr(),
                SPIR_ORDER_ROW_MAJOR,
                2,
                [2, nf as i32].as_ptr(),
                1,
                row.as_ptr(),
                n0,
                0,
                err,
                err_type,
                0,
                false,
                false,
                false,
                plane,
                false,
                0,
                0.0,
                st,
            )
        };
        // Two channels are not a matrix: the trailing dimensions must be
        // none or a square pair.
        let bad = call(-1, 1e-8, 0, -1, &mut st);
        assert!(bad.is_null());
        assert_eq!(st, SPIR_INPUT_DIMENSION_MISMATCH);

        // Scalar channel 0 through a [1, nf] row-major view is still two
        // dimensions; pass it as 1-D instead.
        let one = &row[..nf];
        let rep = spir_minipole_from_matsubara(
            beta,
            nf as i32,
            ns.as_ptr(),
            SPIR_ORDER_ROW_MAJOR,
            1,
            [nf as i32].as_ptr(),
            0,
            one.as_ptr(),
            -1,
            0,
            1e-8,
            0,
            0,
            false,
            false,
            false,
            -1,
            false,
            0,
            0.0,
            &mut st,
        );
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);
        let mut npoles = 0;
        spir_pole_repr_get_npoles(rep, &mut npoles);
        assert_eq!(npoles, 2);
        let mut poles = vec![c(0.0, 0.0); 2];
        spir_pole_repr_get_poles(rep, poles.as_mut_ptr());
        let mut res = vec![c(0.0, 0.0); 2];
        spir_pole_repr_get_residues(rep, res.as_mut_ptr());
        for j in 0..2 {
            assert!((poles[j] - xs[j]).norm() < 1e-6, "{poles:?}");
            assert!((res[j] - amps[0][j]).norm() < 1e-6, "{res:?}");
        }
        let (mut n0, mut err_max) = (0, 0.0);
        assert_eq!(
            spir_pole_repr_get_n0(rep, &mut n0),
            SPIR_COMPUTATION_SUCCESS
        );
        assert!(n0 > 0);
        assert_eq!(
            spir_pole_repr_get_err_max(rep, &mut err_max),
            SPIR_COMPUTATION_SUCCESS
        );
        assert!(err_max > 0.0 && err_max < 1e-8);
        let mut cst = c(1.0, 1.0);
        spir_pole_repr_get_constant(rep, &mut cst);
        assert_eq!(cst, c(0.0, 0.0));

        // Clone shares data.
        let rep2 = spir_pole_repr_clone(rep);
        spir_pole_repr_release(rep);
        let mut n2 = 0;
        spir_pole_repr_get_npoles(rep2, &mut n2);
        assert_eq!(n2, 2);
        spir_pole_repr_release(rep2);

        // Invalid err_type and plane.
        assert!(call(-1, 1e-8, 2, -1, &mut st).is_null());
        assert_eq!(st, SPIR_INVALID_ARGUMENT);
        assert!(call(-1, 1e-8, 0, 2, &mut st).is_null());
        assert_eq!(st, SPIR_INVALID_ARGUMENT);
        assert!(call(-1, 0.0, 0, -1, &mut st).is_null());
        assert_eq!(st, SPIR_INVALID_ARGUMENT);
    }

    /// A 2x2 matrix along axis 0 of a row-major [nf, 2, 2] array, with a
    /// constant term.
    #[test]
    fn minipole_from_matsubara_matrix_constant_capi() {
        let beta = 40.0;
        let xs = [-0.5, 0.3];
        // G_ab = Σ_j R_j[a][b] / (z - x_j) + C_ab
        let r = [[[0.3, 0.1], [0.1, 0.2]], [[0.2, -0.05], [-0.05, 0.4]]];
        let cst = [[0.05, 0.0], [0.0, -0.02]];
        let ns: Vec<i64> = (0..150).map(|k| 2 * k + 1).collect();
        let nf = ns.len();
        let mut v = vec![c(0.0, 0.0); nf * 4];
        for (i, &n) in ns.iter().enumerate() {
            let z = c(0.0, n as f64 * std::f64::consts::PI / beta);
            for a in 0..2 {
                for b in 0..2 {
                    v[(i * 2 + a) * 2 + b] =
                        (0..2).map(|j| r[j][a][b] / (z - xs[j])).sum::<Complex64>() + cst[a][b];
                }
            }
        }
        let mut st = 0;
        let rep = spir_minipole_from_matsubara(
            beta,
            nf as i32,
            ns.as_ptr(),
            SPIR_ORDER_ROW_MAJOR,
            3,
            [nf as i32, 2, 2].as_ptr(),
            0,
            v.as_ptr(),
            3,
            0,
            1e-10,
            0,
            0,
            false,
            false,
            true,
            -1,
            false,
            0,
            0.0,
            &mut st,
        );
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);
        let mut got = vec![c(0.0, 0.0); 4];
        spir_pole_repr_get_constant(rep, got.as_mut_ptr());
        for a in 0..2 {
            for b in 0..2 {
                // Row-major [2, 2].
                assert!((got[a * 2 + b] - cst[a][b]).norm() < 1e-6, "{got:?}");
            }
        }
        let mut n0 = 0;
        spir_pole_repr_get_n0(rep, &mut n0);
        assert_eq!(n0, 3);
        // The physical poles, with weights, are among the poles found.
        let mut np = 0;
        spir_pole_repr_get_npoles(rep, &mut np);
        let mut poles = vec![c(0.0, 0.0); np as usize];
        spir_pole_repr_get_poles(rep, poles.as_mut_ptr());
        let mut res = vec![c(0.0, 0.0); np as usize * 4];
        spir_pole_repr_get_residues(rep, res.as_mut_ptr());
        for (j, &x) in xs.iter().enumerate() {
            let k = (0..np as usize)
                .min_by(|&p, &q| (poles[p] - x).norm().total_cmp(&(poles[q] - x).norm()))
                .unwrap();
            assert!((poles[k] - x).norm() < 1e-6, "{poles:?}");
            for a in 0..2 {
                for b in 0..2 {
                    // Row-major [npoles, 2, 2].
                    assert!((res[(k * 2 + a) * 2 + b] - r[j][a][b]).norm() < 1e-6);
                }
            }
        }
        spir_pole_repr_release(rep);
    }

    /// DLR coefficients in a column-major [np, 2] array.
    #[test]
    fn minipole_from_dlr_capi() {
        let (beta, wmax) = (40.0, 1.0);
        let xs = [-0.5, 0.3];
        let amps = [[0.4, 0.6], [0.1, 0.2]];
        let mut st = 0;
        let dlr = spir_dlr_new_independent_for_test(beta, wmax, &mut st);
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);

        // Default DLR Matsubara and τ nodes: one per pole.
        let mut nm = 0;
        assert_eq!(
            crate::spir_basis_get_n_default_matsus(dlr, false, &mut nm),
            SPIR_COMPUTATION_SUCCESS
        );
        let mut np = 0;
        assert_eq!(
            crate::spir_dlr_get_npoles(dlr, &mut np),
            SPIR_COMPUTATION_SUCCESS
        );
        assert_eq!(nm, np);
        let mut ntau = 0;
        assert_eq!(
            crate::spir_basis_get_n_default_taus(dlr, &mut ntau),
            SPIR_COMPUTATION_SUCCESS
        );
        assert_eq!(ntau, np);

        // G(iν) = Σ_l g_l w_l / (iν - ω_l) fitted at the DLR Matsubara nodes.
        let mut nodes = vec![0i64; nm as usize];
        crate::spir_basis_get_default_matsus(dlr, false, nodes.as_mut_ptr());
        let mut samp_st = 0;
        let smpl = crate::spir_matsu_sampling_new(dlr, false, nm, nodes.as_ptr(), &mut samp_st);
        assert_eq!(samp_st, SPIR_COMPUTATION_SUCCESS);
        let mut gv = vec![c(0.0, 0.0); nm as usize * 2];
        for ch in 0..2 {
            for (i, &n) in nodes.iter().enumerate() {
                gv[i + nm as usize * ch] = green(&amps, &xs, beta, ch, n);
            }
        }
        let mut coeffs = vec![c(0.0, 0.0); np as usize * 2];
        let sdims = [nm, 2];
        assert_eq!(
            crate::spir_sampling_fit_zz(
                smpl,
                std::ptr::null(),
                SPIR_ORDER_COLUMN_MAJOR,
                2,
                sdims.as_ptr(),
                0,
                gv.as_ptr(),
                coeffs.as_mut_ptr()
            ),
            SPIR_COMPUTATION_SUCCESS
        );
        let cdims = [np, 2];
        let rep = spir_minipole_from_dlr(
            dlr,
            SPIR_ORDER_COLUMN_MAJOR,
            2,
            cdims.as_ptr(),
            0,
            coeffs.as_ptr(),
            5,
            0.0,
            1e-8,
            0,
            0,
            false,
            &mut st,
        );
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);
        let mut npoles = 0;
        spir_pole_repr_get_npoles(rep, &mut npoles);
        assert_eq!(npoles, 2);
        let mut poles = vec![c(0.0, 0.0); 2];
        spir_pole_repr_get_poles(rep, poles.as_mut_ptr());
        let mut res = vec![c(0.0, 0.0); 4];
        spir_pole_repr_get_residues(rep, res.as_mut_ptr());
        for j in 0..2 {
            assert!((poles[j] - xs[j]).norm() < 1e-6, "{poles:?}");
            for ch in 0..2 {
                // Column-major [npoles, 2].
                assert!((res[j + 2 * ch] - amps[ch][j]).norm() < 1e-6, "{res:?}");
            }
        }
        let mut e = 0.0;
        assert_eq!(spir_pole_repr_get_err_max(rep, &mut e), SPIR_NOT_SUPPORTED);
        let mut n0 = 0;
        spir_pole_repr_get_n0(rep, &mut n0);
        assert_eq!(n0, 5);
        spir_pole_repr_release(rep);

        let call = |dims: &[i32],
                    data: &[Complex64],
                    n0: i32,
                    err: f64,
                    n_poles: i32,
                    st: &mut StatusCode| {
            spir_minipole_from_dlr(
                dlr,
                SPIR_ORDER_COLUMN_MAJOR,
                2,
                dims.as_ptr(),
                0,
                data.as_ptr(),
                n0,
                0.0,
                err,
                0,
                n_poles,
                false,
                st,
            )
        };
        // Wrong number of coefficients, negative n0, neither err nor n_poles.
        let big = vec![c(0.0, 0.0); (np as usize + 1) * 2];
        assert!(call(&[np + 1, 2], &big, 5, 1e-8, 0, &mut st).is_null());
        assert_eq!(st, SPIR_INPUT_DIMENSION_MISMATCH);
        assert!(call(&cdims, &coeffs, -1, 1e-8, 0, &mut st).is_null());
        assert_eq!(st, SPIR_INVALID_ARGUMENT);
        assert!(call(&cdims, &coeffs, 5, 0.0, 0, &mut st).is_null());
        assert_eq!(st, SPIR_INVALID_ARGUMENT);
        // A fixed number of poles alone.
        let rep = call(&cdims, &coeffs, 5, 0.0, 2, &mut st);
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);
        spir_pole_repr_get_npoles(rep, &mut npoles);
        assert_eq!(npoles, 2);
        spir_pole_repr_release(rep);

        crate::spir_sampling_release(smpl);
        spir_basis_release(dlr);
    }

    fn spir_dlr_new_independent_for_test(
        beta: f64,
        wmax: f64,
        st: &mut StatusCode,
    ) -> *mut spir_basis {
        crate::spir_dlr_new_independent(crate::SPIR_STATISTICS_FERMIONIC, beta, wmax, 1e-12, st)
    }
}
