//! Minimal pole representation (MiniPole) from a DLR or from Matsubara data.
//!
//! Compresses a Green's function `G(z) = ∫ A(x) / (z - x) dx` (scalar or
//! matrix-valued) into a small number of poles,
//!
//! ```text
//! G(z) ≈ Σ_{j=1}^{r} A_j / (z - ξ_j),
//! ```
//!
//! following the minimal pole method of L. Zhang and E. Gull, Phys. Rev. B
//! 110, 035154 (2024), with the DLR in place of the first Prony step.
//!
//! # Construction
//!
//! The map `z = iν_m + (Δ/2)(w - 1/w)`, with `ν_m` the midpoint and `Δ` the
//! half-length of a segment `[iν_a, iν_b]` of the imaginary axis, sends the
//! unit circle `|w| = 1` onto that segment (traversed on both sides), and
//! every real pole `ξ` to one point `ξ̃ = g(ξ)` inside the disk. The moments
//!
//! ```text
//! h_k = (1/2πi) ∮_{|w|=1} G(z(w)) w^k dw = Σ_l Ã_l ξ̃_l^k,
//! Ã_l = A_l / z'(ξ̃_l),  z'(w) = (Δ/2)(1 + 1/w²),
//! ```
//!
//! form an exponential sum. Only values of `G` on the segment enter, so an
//! error `δ` of `G` there changes every `h_k` by at most `δ`, uniformly in
//! `k`. This matters because a DLR fitted on Matsubara frequencies is
//! accurate on the imaginary axis only away from the real axis: between the
//! lowest Matsubara frequencies (and near the real axis in general) it
//! deviates from the true function. The segment therefore starts at
//! `ν_a ~ ln(1/tol) / β`.
//!
//! 1. `h_k`, `k < M`, from the DLR `G(z) = Σ_i c_i / (z - ω_i)` exactly by
//!    residues; Matsubara data are first fitted by a DLR;
//! 2. ESPRIT with a relative singular-value tolerance on `h_k`;
//! 3. poles `ξ_l = z(ξ̃_l)` and residues `A_l = z'(ξ̃_l) Ã_l`; poles with
//!    `|Im ξ_l| >= ν_a / 2` are artifacts of errors near the segment ends and
//!    are dropped, with the remaining amplitudes refitted.

use crate::dlr::DiscreteLehmannRepresentation;
use crate::error::{Error, Result};
use crate::esprit::{EspritDiagnostics, EspritOptions, esprit};
use crate::fitters::common::{compute_pinv, compute_pinv_truncated};
use crate::freq::MatsubaraFreq;
use crate::traits::StatisticsType;
use num_complex::Complex;
use tenferro_tensor::{TensorScalar, TypedTensor};

type C64 = Complex<f64>;

/// Options for the MiniPole compression.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniPoleOptions {
    /// Relative singular-value tolerance of the ESPRIT step; set it to the
    /// accuracy (noise level) of the input.
    pub tolerance: f64,
    /// Number of moments `M`. `None` chooses `M` such that the slowest
    /// decaying node satisfies `|ξ̃|^M < tol / 10`, clamped to `[16, 2000]`.
    pub n_moments: Option<usize>,
    /// Lower end `ν_a` of the integration segment. `None` uses
    /// `max(2.5 ln(1/tol) / β, π / β)`.
    pub freq_min: Option<f64>,
    /// Upper end `ν_b` of the integration segment. `None` uses
    /// `ν_a + 10 ωmax`; a segment much longer than `ωmax` resolves
    /// the poles better.
    pub freq_max: Option<f64>,
    /// Upper bound on the number of poles.
    pub max_poles: Option<usize>,
}

impl MiniPoleOptions {
    /// Options with ESPRIT tolerance `tolerance` and defaults otherwise.
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            n_moments: None,
            freq_min: None,
            freq_max: None,
            max_poles: None,
        }
    }

    /// Set the number of moments `M`.
    pub fn with_moments(mut self, n_moments: usize) -> Self {
        self.n_moments = Some(n_moments);
        self
    }

    /// Set the integration segment `[iν_a, iν_b]`.
    pub fn with_segment(mut self, freq_min: f64, freq_max: f64) -> Self {
        self.freq_min = Some(freq_min);
        self.freq_max = Some(freq_max);
        self
    }

    /// Cap the number of poles.
    pub fn with_max_poles(mut self, max_poles: usize) -> Self {
        self.max_poles = Some(max_poles);
        self
    }
}

/// Diagnostics of a MiniPole compression.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniPoleDiagnostics {
    /// ESPRIT diagnostics on the moments.
    pub esprit: EspritDiagnostics,
    /// Number of moments `M`.
    pub n_moments: usize,
    /// Integration segment `[ν_a, ν_b]`.
    pub segment: (f64, f64),
    /// ESPRIT nodes discarded because their pole lies closer to the segment
    /// than to the real axis (`|Im ξ| >= ν_a / 2`).
    pub discarded: usize,
    /// Relative least-squares residual of the DLR fit (Matsubara input only).
    pub dlr_fit_residual: Option<f64>,
}

/// A pole representation `G(z) = Σ_j A_j / (z - ξ_j)`.
#[derive(Debug)]
pub struct PoleRepresentation {
    /// Poles `ξ_j`, sorted by real part. Imaginary parts are small for
    /// physical input and are kept rather than discarded.
    pub poles: Vec<C64>,
    /// Residues with shape `[r, ...]`, trailing axes being the channels of
    /// the input.
    pub residues: TypedTensor<C64>,
    /// Diagnostics.
    pub diagnostics: MiniPoleDiagnostics,
}

impl PoleRepresentation {
    /// Evaluate `G(z)` at complex frequencies; returns shape `[len(z), ...]`.
    ///
    /// # Errors
    /// Propagates tensor construction errors.
    pub fn evaluate(&self, z: &[C64]) -> Result<TypedTensor<C64>> {
        let shape = self.residues.shape().to_vec();
        let r = self.poles.len();
        let d: usize = shape[1..].iter().product();
        let a = self.residues.host_data()?;
        let mut out = vec![C64::new(0.0, 0.0); z.len() * d];
        for (i, &zi) in z.iter().enumerate() {
            for (j, &p) in self.poles.iter().enumerate() {
                let f = (zi - p).inv();
                for c in 0..d {
                    out[i + z.len() * c] += a[j + r * c] * f;
                }
            }
        }
        let mut out_shape = shape;
        out_shape[0] = z.len();
        Ok(TypedTensor::from_vec_col_major(out_shape, out)?)
    }

    /// Evaluate `G(iν)` at Matsubara frequencies.
    ///
    /// # Errors
    /// Propagates tensor construction errors.
    pub fn evaluate_matsubara<S: StatisticsType>(
        &self,
        beta: f64,
        freqs: &[MatsubaraFreq<S>],
    ) -> Result<TypedTensor<C64>> {
        let z: Vec<C64> = freqs.iter().map(|f| f.value_imaginary(beta)).collect();
        self.evaluate(&z)
    }
}

/// Compress DLR coefficients into a minimal pole representation.
///
/// `g_dlr` has shape `[n_poles, ...]` (DLR coefficients along axis 0, any
/// trailing channel axes), as returned for example by
/// [`MatsubaraSampling::fit_nd`](crate::MatsubaraSampling::fit_nd) with
/// `dim = 0` on a DLR.
///
/// # Errors
/// Returns [`Error::InvalidArgument`] if the shape does not match the DLR,
/// or the options are invalid; propagates ESPRIT errors.
pub fn minipole_from_dlr<S, T>(
    dlr: &DiscreteLehmannRepresentation<S>,
    g_dlr: &TypedTensor<T>,
    options: &MiniPoleOptions,
) -> Result<PoleRepresentation>
where
    S: StatisticsType,
    T: TensorScalar + Copy + Into<C64>,
{
    let shape = g_dlr.shape().to_vec();
    let n_dlr = dlr.poles.len();
    if shape.first() != Some(&n_dlr) {
        return Err(Error::InvalidArgument(format!(
            "coefficients of shape {shape:?} do not match {n_dlr} DLR poles along axis 0"
        )));
    }
    if !(options.tolerance.is_finite() && options.tolerance >= 0.0) {
        return Err(Error::InvalidArgument(format!(
            "tolerance {} must be finite and non-negative",
            options.tolerance
        )));
    }
    let log_tol = -options.tolerance.max(f64::EPSILON).ln();
    let nu_a = options
        .freq_min
        .unwrap_or((2.5 * log_tol / dlr.beta).max(std::f64::consts::PI / dlr.beta));
    let nu_b = options.freq_max.unwrap_or(nu_a + 10.0 * dlr.wmax);
    if !(nu_a.is_finite() && nu_b.is_finite() && 0.0 < nu_a && nu_a < nu_b) {
        return Err(Error::InvalidArgument(format!(
            "segment [{nu_a}, {nu_b}] must satisfy 0 < freq_min < freq_max"
        )));
    }
    let map = SegmentMap::new(nu_a, nu_b);

    // G(z) = Σ_i c_i / (z - ω_i), c_i = g_i * pole_weight_i; the moments of
    // each term are c_i ξ̃_i^k / z'(ξ̃_i).
    let nodes: Vec<C64> = dlr
        .poles
        .iter()
        .map(|&w| map.preimage(C64::new(w, 0.0)))
        .collect();
    let max_mod = nodes.iter().map(|q| q.norm()).fold(0.0, f64::max);
    let n_moments = match options.n_moments {
        Some(m) => m,
        None if max_mod > 0.0 && max_mod < 1.0 => {
            (((log_tol + 10f64.ln()) / -max_mod.ln()).ceil() as usize).clamp(16, 2000)
        }
        None => 16,
    };
    if n_moments < 3 {
        return Err(Error::InvalidArgument(format!(
            "at least 3 moments are required, got {n_moments}"
        )));
    }
    let d: usize = shape[1..].iter().product();
    let g = g_dlr.host_data()?;
    let weights = dlr.pole_weights();
    let mut h = vec![C64::new(0.0, 0.0); n_moments * d];
    for (i, &q) in nodes.iter().enumerate() {
        let f = weights[i] / map.derivative(q);
        for c in 0..d {
            let mut p: C64 = g[i + n_dlr * c].into() * f;
            for k in 0..n_moments {
                h[k + n_moments * c] += p;
                p *= q;
            }
        }
    }
    let mut mshape = shape;
    mshape[0] = n_moments;
    compress(h, mshape, &map, options)
}

/// Minimal pole representation from Matsubara data.
///
/// `values` has shape `[len(freqs), ...]`. A DLR with cutoff `wmax` and
/// accuracy `dlr_accuracy` is fitted to the data by least squares, and its
/// coefficients are compressed by [`minipole_from_dlr`]. The frequencies need
/// not be sorted, and fewer than `n_dlr` frequencies give a minimum-norm fit.
///
/// # Errors
/// Returns [`Error::InvalidArgument`] for a shape mismatch or invalid DLR
/// parameters; propagates fitting and ESPRIT errors.
pub fn minipole_from_matsubara<S>(
    beta: f64,
    wmax: f64,
    dlr_accuracy: f64,
    freqs: &[MatsubaraFreq<S>],
    values: &TypedTensor<C64>,
    options: &MiniPoleOptions,
) -> Result<PoleRepresentation>
where
    S: StatisticsType + 'static,
{
    let shape = values.shape().to_vec();
    if shape.first() != Some(&freqs.len()) || freqs.is_empty() {
        return Err(Error::InvalidArgument(format!(
            "values of shape {shape:?} do not match {} frequencies along axis 0",
            freqs.len()
        )));
    }
    let dlr = DiscreteLehmannRepresentation::<S>::new(beta, wmax, dlr_accuracy)
        .map_err(|e| Error::InvalidArgument(e.to_string()))?;

    // Truncated-SVD least squares on u_i(iν) = w_i / (iν - ω_i): without
    // regularization the fit follows the noise, and the DLR then deviates
    // strongly from G between the data points.
    let nf = freqs.len();
    let n_dlr = dlr.poles.len();
    let d: usize = shape[1..].iter().product();
    let weights = dlr.pole_weights();
    let mut a = vec![C64::new(0.0, 0.0); nf * n_dlr];
    for (i, &w) in dlr.poles.iter().enumerate() {
        for (row, f) in freqs.iter().enumerate() {
            a[row + nf * i] = weights[i] / (f.value_imaginary(beta) - w);
        }
    }
    let rtol = dlr_accuracy.max(options.tolerance * 1e-2);
    let pinv = compute_pinv_truncated(&a, nf, n_dlr, rtol)?;
    let y = values.host_data()?;
    let mut coeffs = vec![C64::new(0.0, 0.0); n_dlr * d];
    pinv.solve(None, y, 1, d, &mut coeffs)?;

    let (mut num, mut den) = (0.0, 0.0);
    for c in 0..d {
        for row in 0..nf {
            let f: C64 = (0..n_dlr)
                .map(|i| a[row + nf * i] * coeffs[i + n_dlr * c])
                .sum();
            num += (f - y[row + nf * c]).norm_sqr();
            den += y[row + nf * c].norm_sqr();
        }
    }
    let fit_residual = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let mut cshape = shape;
    cshape[0] = n_dlr;
    let g_dlr = TypedTensor::from_vec_col_major(cshape, coeffs)?;

    let mut result = minipole_from_dlr(&dlr, &g_dlr, options)?;
    result.diagnostics.dlr_fit_residual = Some(fit_residual);
    Ok(result)
}

/// `z = iν_m + (Δ/2)(w - 1/w)`: unit circle ↔ segment `[iν_a, iν_b]`.
struct SegmentMap {
    nu_a: f64,
    nu_b: f64,
    mid: f64,
    half: f64,
}

impl SegmentMap {
    fn new(nu_a: f64, nu_b: f64) -> Self {
        Self {
            nu_a,
            nu_b,
            mid: 0.5 * (nu_a + nu_b),
            half: 0.5 * (nu_b - nu_a),
        }
    }

    fn to_plane(&self, w: C64) -> C64 {
        C64::new(0.0, self.mid) + (w - w.inv()) * (0.5 * self.half)
    }

    fn derivative(&self, w: C64) -> C64 {
        (C64::new(1.0, 0.0) + (w * w).inv()) * (0.5 * self.half)
    }

    /// The preimage of `z` inside the unit disk.
    fn preimage(&self, z: C64) -> C64 {
        // w^2 - 2 z_s w - 1 = 0 with z_s = (z - iν_m) / Δ; the roots are w
        // and -1/w.
        let zs = (z - C64::new(0.0, self.mid)) / self.half;
        let r = (zs * zs + 1.0).sqrt();
        let (w1, w2) = (zs - r, zs + r);
        if w1.norm() <= w2.norm() { w1 } else { w2 }
    }
}

/// ESPRIT on the moments `h` (shape `[M, ...]`) and the map back to poles.
fn compress(
    h: Vec<C64>,
    shape: Vec<usize>,
    map: &SegmentMap,
    options: &MiniPoleOptions,
) -> Result<PoleRepresentation> {
    let m = shape[0];
    let d: usize = shape[1..].iter().product();
    let mut eopts = EspritOptions::tolerance(options.tolerance);
    if let Some(p) = options.max_poles {
        eopts = eopts.with_max_order(p);
    }
    let fit = esprit(
        &TypedTensor::from_vec_col_major(shape.clone(), h.clone())?,
        &eopts,
    )?;

    // Errors of G near the segment ends produce nodes close to the unit
    // circle, i.e. poles closer to the segment than to the real axis; drop
    // them and refit the amplitudes of the rest.
    let limit = 0.5 * map.nu_a;
    let keep: Vec<usize> = (0..fit.nodes.len())
        .filter(|&j| map.to_plane(fit.nodes[j]).im.abs() < limit)
        .collect();
    let discarded = fit.nodes.len() - keep.len();
    let nodes: Vec<C64> = keep.iter().map(|&j| fit.nodes[j]).collect();
    let r = nodes.len();
    let amp = if discarded == 0 {
        fit.amplitudes.host_data()?.to_vec()
    } else if r == 0 {
        Vec::new()
    } else {
        let mut vand = vec![C64::new(0.0, 0.0); m * r];
        for (j, &q) in nodes.iter().enumerate() {
            let mut p = C64::new(1.0, 0.0);
            for k in 0..m {
                vand[k + m * j] = p;
                p *= q;
            }
        }
        let mut amp = vec![C64::new(0.0, 0.0); r * d];
        compute_pinv(&vand, m, r)?.solve(None, &h, 1, d, &mut amp)?;
        amp
    };

    let poles_unsorted: Vec<C64> = nodes.iter().map(|&q| map.to_plane(q)).collect();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| poles_unsorted[a].re.total_cmp(&poles_unsorted[b].re));
    let poles: Vec<C64> = order.iter().map(|&j| poles_unsorted[j]).collect();
    let mut residues = vec![C64::new(0.0, 0.0); r * d];
    for (jn, &j) in order.iter().enumerate() {
        let f = map.derivative(nodes[j]);
        for c in 0..d {
            residues[jn + r * c] = amp[j + r * c] * f;
        }
    }
    let mut rshape = shape;
    rshape[0] = r;
    Ok(PoleRepresentation {
        poles,
        residues: TypedTensor::from_vec_col_major(rshape, residues)?,
        diagnostics: MiniPoleDiagnostics {
            esprit: fit.diagnostics,
            n_moments: m,
            segment: (map.nu_a, map.nu_b),
            discarded,
            dlr_fit_residual: None,
        },
    })
}

#[cfg(test)]
#[path = "minipole_tests.rs"]
mod tests;
