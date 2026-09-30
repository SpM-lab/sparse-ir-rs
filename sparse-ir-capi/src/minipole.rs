//! Minimal pole representation (MiniPole) API for C
//!
//! Compresses a Green's function into a small number of complex poles and
//! residues, either from DLR coefficients or directly from Matsubara data.
//!
//! Functions:
//! - Creation: spir_minipole_from_dlr, spir_minipole_from_matsubara
//! - Introspection: spir_pole_repr_get_npoles, spir_pole_repr_get_poles,
//!   spir_pole_repr_get_residues, spir_pole_repr_get_dlr_fit_residual
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
    SPIR_INVALID_ARGUMENT, SPIR_NOT_SUPPORTED, SPIR_STATISTICS_BOSONIC, SPIR_STATISTICS_FERMIONIC,
    StatusCode,
};
use sparse_ir::minipole::{
    MiniPoleOptions, PoleRepresentation, minipole_from_dlr, minipole_from_matsubara,
};
use sparse_ir::{Bosonic, Fermionic, MatsubaraFreq, StatisticsType, TypedTensor};

/// Opaque minimal pole representation for C API
///
/// Holds complex poles `ξ_j` and residues `A_j` with
/// `G(z) ≈ Σ_j A_j / (z - ξ_j)`.
#[repr(C)]
pub struct spir_pole_repr {
    pub(crate) _private: *const std::ffi::c_void,
}

/// Internal pole representation (not exposed to C)
pub(crate) struct PoleReprInner {
    poles: Vec<Complex64>,
    /// Residues in the caller's layout, with the pole axis at `target_dim`.
    residues: Vec<Complex64>,
    dlr_fit_residual: Option<f64>,
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

fn build_options(
    tolerance: f64,
    n_moments: libc::c_int,
    freq_min: f64,
    freq_max: f64,
    max_poles: libc::c_int,
) -> Result<MiniPoleOptions, StatusCode> {
    let mut opts = MiniPoleOptions::new(tolerance);
    if n_moments > 0 {
        opts = opts.with_moments(n_moments as usize);
    }
    match (freq_min > 0.0, freq_max > 0.0) {
        (true, true) => opts = opts.with_segment(freq_min, freq_max),
        (false, false) => {}
        _ => return Err(SPIR_INVALID_ARGUMENT),
    }
    if max_poles > 0 {
        opts = opts.with_max_poles(max_poles as usize);
    }
    Ok(opts)
}

fn finish(
    rep: PoleRepresentation,
    cm_dims: &[usize],
    axis: usize,
) -> Result<PoleReprInner, StatusCode> {
    let np = rep.poles.len();
    let mut out_dims = cm_dims.to_vec();
    out_dims[axis] = np;
    let data = rep
        .residues
        .host_data()
        .map_err(|e| status_from(&e.into()))?;
    Ok(PoleReprInner {
        residues: move_front_to_axis(data, np, &out_dims, axis),
        poles: rep.poles,
        dlr_fit_residual: rep.diagnostics.dlr_fit_residual,
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

/// Compresses DLR coefficients into a minimal pole representation
///
/// # Arguments
/// * `dlr` - Pointer to a DLR basis object
/// * `order` - Memory layout order (`SPIR_ORDER_ROW_MAJOR` or `SPIR_ORDER_COLUMN_MAJOR`)
/// * `ndim` - Number of dimensions of `coeffs`
/// * `input_dims` - Dimensions of `coeffs`; `input_dims[target_dim]` must equal the number of DLR poles
/// * `target_dim` - Dimension holding the DLR coefficients
/// * `coeffs` - Complex DLR coefficients
/// * `tolerance` - Accuracy target of the compression (must be >= 0)
/// * `n_moments` - Number of moments, or <= 0 for the default
/// * `freq_min`, `freq_max` - Imaginary-axis segment `[i freq_min, i freq_max]` of the
///   conformal map, or both <= 0 for the default
/// * `max_poles` - Maximum number of poles, or <= 0 for no limit
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
    tolerance: f64,
    n_moments: libc::c_int,
    freq_min: f64,
    freq_max: f64,
    max_poles: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_pole_repr {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<PoleReprInner, StatusCode> {
        if dlr.is_null() {
            return Err(SPIR_INVALID_ARGUMENT);
        }
        let opts = build_options(tolerance, n_moments, freq_min, freq_max, max_poles)?;
        let (g, cm_dims, axis) =
            unsafe { read_axis_first(order, ndim, input_dims, target_dim, coeffs)? };
        let basis = unsafe { &*dlr };
        if g.shape()[0] != basis.size() {
            return Err(SPIR_INPUT_DIMENSION_MISMATCH);
        }
        let rep = match basis.inner() {
            BasisType::DLRFermionic(d) => minipole_from_dlr(d.as_ref(), &g, &opts),
            BasisType::DLRBosonic(d) => minipole_from_dlr(d.as_ref(), &g, &opts),
            _ => return Err(SPIR_NOT_SUPPORTED),
        }
        .map_err(|e| status_from(&e))?;
        finish(rep, &cm_dims, axis)
    }));
    return_repr(result, status)
}

fn from_matsubara_typed<S: StatisticsType + 'static>(
    beta: f64,
    omega_max: f64,
    dlr_accuracy: f64,
    ns: &[i64],
    values: &TypedTensor<Complex64>,
    opts: &MiniPoleOptions,
) -> Result<PoleRepresentation, StatusCode> {
    let freqs = ns
        .iter()
        .map(|&n| MatsubaraFreq::<S>::new(n))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| SPIR_INVALID_ARGUMENT)?;
    minipole_from_matsubara(beta, omega_max, dlr_accuracy, &freqs, values, opts)
        .map_err(|e| status_from(&e))
}

/// Builds a minimal pole representation from Matsubara data
///
/// A DLR with cutoff `omega_max` and accuracy `dlr_accuracy` is fitted to the
/// data by regularized least squares and then compressed as in
/// `spir_minipole_from_dlr`. Frequencies need not be sorted.
///
/// # Arguments
/// * `statistics` - `SPIR_STATISTICS_FERMIONIC` or `SPIR_STATISTICS_BOSONIC`
/// * `beta` - Inverse temperature
/// * `omega_max` - Frequency cutoff of the intermediate DLR
/// * `dlr_accuracy` - Accuracy of the intermediate DLR
/// * `n_freqs` - Number of Matsubara frequencies
/// * `matsubara_indices` - Matsubara indices `n` (`ν = nπ/β`, odd for fermions, even for bosons)
/// * `order`, `ndim`, `input_dims`, `target_dim` - Layout of `values`; `input_dims[target_dim]` must equal `n_freqs`
/// * `values` - Complex values `G(iν_n)`
/// * `tolerance`, `n_moments`, `freq_min`, `freq_max`, `max_poles` - As in `spir_minipole_from_dlr`
/// * `status` - Pointer to store the status code
///
/// # Returns
/// Pointer to the pole representation, or NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn spir_minipole_from_matsubara(
    statistics: libc::c_int,
    beta: f64,
    omega_max: f64,
    dlr_accuracy: f64,
    n_freqs: libc::c_int,
    matsubara_indices: *const i64,
    order: libc::c_int,
    ndim: libc::c_int,
    input_dims: *const libc::c_int,
    target_dim: libc::c_int,
    values: *const Complex64,
    tolerance: f64,
    n_moments: libc::c_int,
    freq_min: f64,
    freq_max: f64,
    max_poles: libc::c_int,
    status: *mut StatusCode,
) -> *mut spir_pole_repr {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<PoleReprInner, StatusCode> {
        if matsubara_indices.is_null() || n_freqs <= 0 {
            return Err(SPIR_INVALID_ARGUMENT);
        }
        let opts = build_options(tolerance, n_moments, freq_min, freq_max, max_poles)?;
        let (v, cm_dims, axis) =
            unsafe { read_axis_first(order, ndim, input_dims, target_dim, values)? };
        if v.shape()[0] != n_freqs as usize {
            return Err(SPIR_INPUT_DIMENSION_MISMATCH);
        }
        let ns = unsafe { std::slice::from_raw_parts(matsubara_indices, n_freqs as usize) };
        let rep = match statistics {
            SPIR_STATISTICS_FERMIONIC => {
                from_matsubara_typed::<Fermionic>(beta, omega_max, dlr_accuracy, ns, &v, &opts)
            }
            SPIR_STATISTICS_BOSONIC => {
                from_matsubara_typed::<Bosonic>(beta, omega_max, dlr_accuracy, ns, &v, &opts)
            }
            _ => Err(SPIR_INVALID_ARGUMENT),
        }?;
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

/// Gets the complex poles (array of length npoles)
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

/// Gets the relative residual of the intermediate DLR fit
///
/// Available only for representations built by `spir_minipole_from_matsubara`;
/// otherwise returns `SPIR_NOT_SUPPORTED`.
#[unsafe(no_mangle)]
pub extern "C" fn spir_pole_repr_get_dlr_fit_residual(
    rep: *const spir_pole_repr,
    residual: *mut f64,
) -> StatusCode {
    if rep.is_null() || residual.is_null() {
        return SPIR_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        match (*rep).inner().dlr_fit_residual {
            Some(r) => {
                *residual = r;
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

    /// Two-channel data along axis 1 (row-major [2, nf] and column-major
    /// [nf, 2] must give the same residues).
    #[test]
    fn minipole_from_dlr_and_matsubara_capi() {
        let (beta, wmax) = (40.0, 1.0);
        let xs = [-0.5, 0.3];
        let amps = [[0.4, 0.6], [0.1, 0.2]]; // [channel][pole]
        let mut st = 0;
        let dlr = spir_dlr_new_independent_for_test(beta, wmax, &mut st);
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);

        // Default DLR Matsubara nodes are now reported.
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

        let ns: Vec<i64> = (-60..60).map(|k| 2 * k + 1).collect();
        let nf = ns.len();
        let g = |ch: usize, n: i64| -> Complex64 {
            let z = c(0.0, n as f64 * std::f64::consts::PI / beta);
            (0..2).map(|j| amps[ch][j] / (z - xs[j])).sum()
        };
        // Row-major [2, nf], target_dim = 1.
        let mut row = vec![c(0.0, 0.0); 2 * nf];
        for ch in 0..2 {
            for (i, &n) in ns.iter().enumerate() {
                row[ch * nf + i] = g(ch, n);
            }
        }
        let dims = [2, nf as i32];
        let rep = spir_minipole_from_matsubara(
            SPIR_STATISTICS_FERMIONIC,
            beta,
            wmax,
            1e-12,
            nf as i32,
            ns.as_ptr(),
            SPIR_ORDER_ROW_MAJOR,
            2,
            dims.as_ptr(),
            1,
            row.as_ptr(),
            1e-8,
            0,
            0.0,
            0.0,
            0,
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
                // Row-major [2, npoles].
                assert!((res[ch * 2 + j] - amps[ch][j]).norm() < 1e-6, "{res:?}");
            }
        }
        let mut resid = 0.0;
        assert_eq!(
            spir_pole_repr_get_dlr_fit_residual(rep, &mut resid),
            SPIR_COMPUTATION_SUCCESS
        );
        assert!(resid < 1e-8);

        // Clone shares data.
        let rep2 = spir_pole_repr_clone(rep);
        spir_pole_repr_release(rep);
        let mut n2 = 0;
        spir_pole_repr_get_npoles(rep2, &mut n2);
        assert_eq!(n2, 2);
        spir_pole_repr_release(rep2);

        // From DLR coefficients (column-major [np, 2], target_dim = 0):
        // G(iν) = Σ_l g_l w_l / (iν - ω_l) fitted at the DLR Matsubara nodes.
        let mut nodes = vec![0i64; nm as usize];
        crate::spir_basis_get_default_matsus(dlr, false, nodes.as_mut_ptr());
        let mut samp_st = 0;
        let smpl = crate::spir_matsu_sampling_new(dlr, false, nm, nodes.as_ptr(), &mut samp_st);
        assert_eq!(samp_st, SPIR_COMPUTATION_SUCCESS);
        let mut gv = vec![c(0.0, 0.0); nm as usize * 2];
        for ch in 0..2 {
            for (i, &n) in nodes.iter().enumerate() {
                gv[i + nm as usize * ch] = g(ch, n);
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
            1e-8,
            0,
            0.0,
            0.0,
            0,
            &mut st,
        );
        assert_eq!(st, SPIR_COMPUTATION_SUCCESS);
        let mut res = vec![c(0.0, 0.0); 4];
        spir_pole_repr_get_residues(rep, res.as_mut_ptr());
        for j in 0..2 {
            for ch in 0..2 {
                // Column-major [npoles, 2].
                assert!((res[j + 2 * ch] - amps[ch][j]).norm() < 1e-6, "{res:?}");
            }
        }
        assert_eq!(
            spir_pole_repr_get_dlr_fit_residual(rep, &mut resid),
            SPIR_NOT_SUPPORTED
        );
        spir_pole_repr_release(rep);

        // Invalid arguments.
        let bad = spir_minipole_from_dlr(
            dlr,
            SPIR_ORDER_COLUMN_MAJOR,
            2,
            [np + 1, 2].as_ptr(),
            0,
            vec![c(0.0, 0.0); (np as usize + 1) * 2].as_ptr(),
            1e-8,
            0,
            0.0,
            0.0,
            0,
            &mut st,
        );
        assert!(bad.is_null());
        assert_eq!(st, SPIR_INPUT_DIMENSION_MISMATCH);
        let bad = spir_minipole_from_dlr(
            dlr,
            SPIR_ORDER_COLUMN_MAJOR,
            2,
            cdims.as_ptr(),
            0,
            coeffs.as_ptr(),
            1e-8,
            0,
            1.0,
            0.0,
            0,
            &mut st,
        );
        assert!(bad.is_null());
        assert_eq!(st, SPIR_INVALID_ARGUMENT);

        crate::spir_sampling_release(smpl);
        spir_basis_release(dlr);
    }

    fn spir_dlr_new_independent_for_test(
        beta: f64,
        wmax: f64,
        st: &mut StatusCode,
    ) -> *mut spir_basis {
        crate::spir_dlr_new_independent(SPIR_STATISTICS_FERMIONIC, beta, wmax, 1e-12, st)
    }
}
