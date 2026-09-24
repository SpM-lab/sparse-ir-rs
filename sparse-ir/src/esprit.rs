//! ESPRIT: estimation of exponential sums from uniformly spaced samples.
//!
//! Given samples `y_k ∈ C^d`, `k = 0, …, N-1`, ESPRIT (Roy and Kailath,
//! IEEE Trans. ASSP 37, 984 (1989)) fits the model
//!
//! ```text
//! y_k ≈ Σ_{j=1}^{r} a_j z_j^k,      a_j ∈ C^d,
//! ```
//!
//! with nodes `z_j` shared by all `d` channels (matrix-valued data such as
//! `G_{ab}(k)` are flattened into channels). The procedure is:
//!
//! 1. form the `L x d(N-L+1)` block Hankel matrix `H[i, (c, j)] = y_{i+j, c}`;
//! 2. take the `r` leading left singular vectors `U_r` (signal subspace), with
//!    `r` fixed or chosen by a singular-value tolerance;
//! 3. solve the shift-invariance equation `U_r[0..L-1] Φ = U_r[1..L]` in the
//!    least-squares sense; the nodes are the eigenvalues of `Φ`;
//! 4. fit the amplitudes by least squares on the Vandermonde matrix
//!    `V[k, j] = z_j^k`.

use crate::error::{Error, Result};
use crate::fitters::common::compute_pinv;
use num_complex::Complex;
use tenferro_tensor::TypedTensor;

type C64 = Complex<f64>;

/// How the number of exponentials `r` is chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelOrder {
    /// Exactly `r` terms.
    Fixed(usize),
    /// Keep the singular values of the Hankel matrix with
    /// `σ_i > rtol · σ_0`.
    Tolerance(f64),
}

/// Options for [`esprit`].
#[derive(Debug, Clone, PartialEq)]
pub struct EspritOptions {
    /// Model order selection.
    pub order: ModelOrder,
    /// Upper bound on the model order (applies to [`ModelOrder::Tolerance`]).
    pub max_order: Option<usize>,
    /// Pencil parameter `L` (number of Hankel rows). `None` chooses
    /// `L ≈ N d / (d + 1)`, which balances rows and columns of the block
    /// Hankel matrix.
    pub pencil: Option<usize>,
}

impl EspritOptions {
    /// Fixed model order `r`.
    pub fn fixed(r: usize) -> Self {
        Self {
            order: ModelOrder::Fixed(r),
            max_order: None,
            pencil: None,
        }
    }

    /// Model order from the relative singular-value tolerance `rtol`.
    pub fn tolerance(rtol: f64) -> Self {
        Self {
            order: ModelOrder::Tolerance(rtol),
            max_order: None,
            pencil: None,
        }
    }

    /// Set the pencil parameter `L`.
    pub fn with_pencil(mut self, pencil: usize) -> Self {
        self.pencil = Some(pencil);
        self
    }

    /// Cap the model order.
    pub fn with_max_order(mut self, max_order: usize) -> Self {
        self.max_order = Some(max_order);
        self
    }
}

/// Diagnostics of an ESPRIT fit.
#[derive(Debug, Clone, PartialEq)]
pub struct EspritDiagnostics {
    /// All singular values of the Hankel matrix, descending.
    pub singular_values: Vec<f64>,
    /// Selected model order `r`.
    pub order: usize,
    /// Pencil parameter `L` used.
    pub pencil: usize,
    /// `σ_r / σ_0`: relative size of the largest discarded singular value
    /// (0 when nothing is discarded).
    pub truncation: f64,
    /// Maximum absolute deviation `max_{k,c} |y_{k,c} - model_{k,c}|`.
    pub max_residual: f64,
    /// `‖y - model‖_F / ‖y‖_F` (0 for all-zero data).
    pub relative_residual: f64,
}

/// Result of [`esprit`].
#[derive(Debug)]
pub struct EspritResult {
    /// Nodes `z_j`, sorted by decreasing modulus (ties by argument).
    pub nodes: Vec<C64>,
    /// Amplitudes with shape `[r, ...]`, the trailing dimensions being those
    /// of the input channels.
    pub amplitudes: TypedTensor<C64>,
    /// Fit diagnostics.
    pub diagnostics: EspritDiagnostics,
}

impl EspritResult {
    /// Evaluate the model `Σ_j a_j z_j^k` at (possibly non-integer) sample
    /// positions `k`; returns shape `[len(k), ...]`.
    ///
    /// # Errors
    /// Propagates tensor construction errors.
    pub fn evaluate(&self, k: &[f64]) -> Result<TypedTensor<C64>> {
        let shape = self.amplitudes.shape().to_vec();
        let r = self.nodes.len();
        let d: usize = shape[1..].iter().product();
        let a = self.amplitudes.host_data()?;
        let mut out = vec![C64::new(0.0, 0.0); k.len() * d];
        for (i, &x) in k.iter().enumerate() {
            for (j, z) in self.nodes.iter().enumerate() {
                let zk = if x.fract() == 0.0 && x.abs() < i32::MAX as f64 {
                    z.powi(x as i32)
                } else {
                    z.powf(x)
                };
                for c in 0..d {
                    out[i + k.len() * c] += a[j + r * c] * zk;
                }
            }
        }
        let mut out_shape = shape;
        out_shape[0] = k.len();
        Ok(TypedTensor::from_vec_col_major(out_shape, out)?)
    }
}

/// ESPRIT for a scalar sequence `y_0, …, y_{N-1}`.
///
/// # Errors
/// See [`esprit`].
pub fn esprit_scalar(samples: &[C64], options: &EspritOptions) -> Result<EspritResult> {
    let y = TypedTensor::from_vec_col_major(vec![samples.len()], samples.to_vec())?;
    esprit(&y, options)
}

/// ESPRIT with nodes shared across channels.
///
/// `samples` has shape `[N, ...]`: axis 0 is the uniformly spaced sample
/// index, and all trailing axes are channels (for example `[N, n_orb, n_orb]`
/// for a matrix-valued function).
///
/// # Errors
/// Returns [`Error::InvalidArgument`] for fewer than 2 samples, an invalid
/// pencil, tolerance or fixed order, and [`Error::Numerical`] if a
/// decomposition fails.
pub fn esprit(samples: &TypedTensor<C64>, options: &EspritOptions) -> Result<EspritResult> {
    let shape = samples.shape().to_vec();
    if shape.is_empty() {
        return Err(Error::InvalidArgument(
            "samples must have at least one axis".into(),
        ));
    }
    let n = shape[0];
    let d: usize = shape[1..].iter().product();
    if n < 2 || d == 0 {
        return Err(Error::InvalidArgument(format!(
            "need at least 2 samples and 1 channel, got shape {shape:?}"
        )));
    }
    let y = samples.host_data()?;

    let pencil = match options.pencil {
        Some(l) => l,
        None => ((n * d) as f64 / (d + 1) as f64).round() as usize,
    }
    .clamp(2, n);
    if options.pencil.is_some_and(|l| l < 2 || l > n) {
        return Err(Error::InvalidArgument(format!(
            "pencil {pencil} must lie in [2, {n}]"
        )));
    }
    let k = n - pencil + 1;
    let cols = d * k;

    // Block Hankel matrix, column-major L x (d K).
    let mut h = vec![C64::new(0.0, 0.0); pencil * cols];
    for c in 0..d {
        for j in 0..k {
            let col = &mut h[pencil * (c * k + j)..pencil * (c * k + j + 1)];
            col.copy_from_slice(&y[n * c + j..n * c + j + pencil]);
        }
    }
    let (u, s) = left_singular(&h, pencil, cols)?;

    let max_r = (pencil - 1).min(cols);
    let r = match options.order {
        ModelOrder::Fixed(r) => {
            if r > max_r {
                return Err(Error::InvalidArgument(format!(
                    "model order {r} exceeds min(L - 1, d (N - L + 1)) = {max_r}"
                )));
            }
            r
        }
        ModelOrder::Tolerance(rtol) => {
            if !(rtol.is_finite() && rtol >= 0.0) {
                return Err(Error::InvalidArgument(format!(
                    "tolerance {rtol} must be finite and non-negative"
                )));
            }
            let s0 = s.first().copied().unwrap_or(0.0);
            s.iter()
                .take_while(|&&v| s0 > 0.0 && v > rtol * s0)
                .count()
                .min(max_r)
                .min(options.max_order.unwrap_or(usize::MAX))
        }
    };
    let truncation = match (s.first(), s.get(r)) {
        (Some(&s0), Some(&sr)) if s0 > 0.0 => sr / s0,
        _ => 0.0,
    };

    let nodes = if r == 0 {
        Vec::new()
    } else {
        shift_invariance_nodes(&u, pencil, r)?
    };
    let (amplitudes, max_residual, relative_residual) = fit_amplitudes(y, n, d, &nodes)?;

    let mut amp_shape = shape.clone();
    amp_shape[0] = nodes.len();
    Ok(EspritResult {
        nodes,
        amplitudes: TypedTensor::from_vec_col_major(amp_shape, amplitudes)?,
        diagnostics: EspritDiagnostics {
            singular_values: s,
            order: r,
            pencil,
            truncation,
            max_residual,
            relative_residual,
        },
    })
}

/// Left singular vectors (column-major `n x min(n, m)`) and singular values.
fn left_singular(a: &[C64], n: usize, m: usize) -> Result<(Vec<C64>, Vec<f64>)> {
    use tenferro_cpu::CpuBackend;
    use tenferro_linalg::TypedTensorLinalgExt;
    use tenferro_tensor::BackendSessionHost;

    let tensor = TypedTensor::<C64>::from_vec_col_major(vec![n, m], a.to_vec())?;
    let mut host = CpuBackend::new();
    let (u, s, _vt) = host
        .with_backend_session(|session| tensor.svd(session))
        .map_err(|e| Error::Numerical(format!("Hankel SVD failed: {e}")))?;
    let k = n.min(m);
    let ld = u.shape()[0];
    let u = u.host_data()?;
    let mut out = Vec::with_capacity(n * k);
    for j in 0..k {
        out.extend_from_slice(&u[ld * j..ld * j + n]);
    }
    Ok((out, s.host_data()?.to_vec()))
}

/// Eigenvalues of `Φ = U_r[0..L-1]^+ U_r[1..L]`.
fn shift_invariance_nodes(u: &[C64], pencil: usize, r: usize) -> Result<Vec<C64>> {
    use tenferro_cpu::CpuBackend;
    use tenferro_linalg::TypedTensorLinalgExt;
    use tenferro_tensor::BackendSessionHost;

    let m = pencil - 1;
    let mut up = Vec::with_capacity(m * r);
    let mut down = Vec::with_capacity(m * r);
    for j in 0..r {
        let col = &u[pencil * j..pencil * (j + 1)];
        up.extend_from_slice(&col[..m]);
        down.extend_from_slice(&col[1..]);
    }
    let pinv = compute_pinv(&up, m, r)?;
    let mut phi = vec![C64::new(0.0, 0.0); r * r];
    pinv.solve(None, &down, 1, r, &mut phi)?;

    let phi = TypedTensor::<C64>::from_vec_col_major(vec![r, r], phi)?;
    let mut host = CpuBackend::new();
    let (values, _vectors) = host
        .with_backend_session(|session| phi.eig(session))
        .map_err(|e| Error::Numerical(format!("eigendecomposition failed: {e}")))?;
    let mut nodes = values.host_data()?.to_vec();
    nodes.sort_by(|a, b| {
        b.norm()
            .total_cmp(&a.norm())
            .then_with(|| a.arg().total_cmp(&b.arg()))
    });
    Ok(nodes)
}

/// Least-squares amplitudes on the Vandermonde matrix; returns the
/// column-major `r x d` amplitudes and the residual norms.
fn fit_amplitudes(y: &[C64], n: usize, d: usize, nodes: &[C64]) -> Result<(Vec<C64>, f64, f64)> {
    let r = nodes.len();
    let mut vand = vec![C64::new(0.0, 0.0); n * r];
    for (j, &z) in nodes.iter().enumerate() {
        let mut p = C64::new(1.0, 0.0);
        for k in 0..n {
            vand[k + n * j] = p;
            p *= z;
        }
    }
    let mut amp = vec![C64::new(0.0, 0.0); r * d];
    if r > 0 {
        compute_pinv(&vand, n, r)?.solve(None, y, 1, d, &mut amp)?;
    }
    let (mut max_res, mut res2, mut y2) = (0.0_f64, 0.0, 0.0);
    for c in 0..d {
        for k in 0..n {
            let model: C64 = (0..r).map(|j| vand[k + n * j] * amp[j + r * c]).sum();
            let yk = y[k + n * c];
            let e = (yk - model).norm();
            max_res = max_res.max(e);
            res2 += e * e;
            y2 += yk.norm_sqr();
        }
    }
    let rel = if y2 > 0.0 { (res2 / y2).sqrt() } else { 0.0 };
    Ok((amp, max_res, rel))
}

#[cfg(test)]
#[path = "esprit_tests.rs"]
mod tests;
