//! Dense linear algebra in the conventions of NumPy, column-major.

use crate::error::{Error, Result};
use crate::fitters::common::compute_pinv_truncated;
use num_complex::Complex;
use sparse_ir_core::fpu_check::FpuGuard;
use tenferro_cpu::CpuBackend;
use tenferro_linalg::TypedTensorLinalgExt;
use tenferro_tensor::{BackendSessionHost, TypedTensor};

type C64 = Complex<f64>;

/// Singular values and `Vᴴ` (`k x m`, `k = min(n, m)`) of an `n x m` matrix,
/// as `np.linalg.svd(a, full_matrices=False)[1:]`.
pub(crate) fn svd_s_vh(a: &[C64], n: usize, m: usize) -> Result<(Vec<f64>, Vec<C64>)> {
    let _guard = FpuGuard::new_protect_computation();
    let tensor = TypedTensor::<C64>::from_vec_col_major(vec![n, m], a.to_vec())?;
    let mut host = CpuBackend::new();
    let (_u, s, vt) = host
        .with_backend_session(|session| tensor.svd(session))
        .map_err(|e| Error::DecompositionFailed {
            reason: format!("the SVD of the {n} x {m} Hankel matrix failed: {e}"),
        })?;
    let k = n.min(m);
    let ld = vt.shape()[0];
    let vt = vt.host_data()?;
    let mut vh = vec![C64::new(0.0, 0.0); k * m];
    for j in 0..m {
        for i in 0..k {
            vh[i + k * j] = vt[i + ld * j];
        }
    }
    Ok((s.host_data()?[..k].to_vec(), vh))
}

/// Least-squares solution of `a x = b` (`a`: `n x m`, `b`: `n x d`), as
/// `np.linalg.lstsq(a, b, rcond=-1)[0]`: singular values below machine
/// precision relative to the largest are treated as zero.
pub(crate) fn lstsq(a: &[C64], n: usize, m: usize, b: &[C64], d: usize) -> Result<Vec<C64>> {
    let mut x = vec![C64::new(0.0, 0.0); m * d];
    if n == 0 || m == 0 || d == 0 {
        return Ok(x);
    }
    compute_pinv_truncated(a, n, m, f64::EPSILON)?.solve(None, b, 1, d, &mut x)?;
    Ok(x)
}

/// Eigenvalues of an `r x r` matrix, as `np.linalg.eigvals`.
pub(crate) fn eigvals(a: &[C64], r: usize) -> Result<Vec<C64>> {
    if r == 0 {
        return Ok(Vec::new());
    }
    let _guard = FpuGuard::new_protect_computation();
    let tensor = TypedTensor::<C64>::from_vec_col_major(vec![r, r], a.to_vec())?;
    let mut host = CpuBackend::new();
    let (values, _vectors) = host
        .with_backend_session(|session| tensor.eig(session))
        .map_err(|e| Error::DecompositionFailed {
            reason: format!("the eigendecomposition of the {r} x {r} shift matrix failed: {e}"),
        })?;
    Ok(values.host_data()?.to_vec())
}

/// `base ** exponent` for a complex base and a real exponent, as NumPy's
/// `npy_cpow`: integral exponents of modulus below 100 by repeated
/// multiplication, others through the principal logarithm.
pub(crate) fn cpow(base: C64, exponent: f64) -> C64 {
    if exponent == 0.0 {
        return C64::new(1.0, 0.0);
    }
    if exponent.fract() == 0.0 && exponent.abs() < 100.0 {
        let n = exponent.abs() as u32;
        let mut acc = C64::new(1.0, 0.0);
        let mut p = base;
        let mut k = n;
        while k > 0 {
            if k & 1 == 1 {
                acc *= p;
            }
            p *= p;
            k >>= 1;
        }
        return if exponent < 0.0 { acc.inv() } else { acc };
    }
    if base == C64::new(0.0, 0.0) {
        return C64::new(0.0, 0.0);
    }
    (base.ln() * exponent).exp()
}
