//! MPM-DLR: minimal poles from DLR coefficients (port of
//! `mini_pole/mini_pole_dlr.py`).
//!
//! Ported from Green-Phys/MiniPole (commit 15e4a54, MIT License,
//! Copyright (c) 2024 lzphy); see `LICENSE-THIRD-PARTY`.

use super::con_map::{ConMap, ConMapGapless, ConMapGeneric};
use super::esprit::{ErrType, Esprit, EspritParams};
use super::{MiniPoleResult, assemble};
use crate::basis_trait::Basis;
use crate::dlr::DiscreteLehmannRepresentation;
use crate::error::{ArrayRole, Error, Result};
use crate::traits::StatisticsType;
use num_complex::Complex;
use tenferro_tensor::{TensorScalar, TypedTensor};

type C64 = Complex<f64>;

/// Parameters of [`mini_pole_dlr`], with the defaults of the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniPoleDlrParams {
    /// Index of the lowest Matsubara frequency `ω_{n0} = (2 n0 + 1)π/β` of
    /// the contour, typically in `0..10`.
    pub n0: usize,
    /// Cutoff `nmax` of the upper end `(2 nmax + 1)π/β` of the contour when
    /// `symmetry` is false; `None` uses `β`.
    pub nmax: Option<f64>,
    /// Error tolerance of ESPRIT. One of `err` and `m` is required.
    pub err: Option<f64>,
    /// Interpretation of `err` (default absolute).
    pub err_type: ErrType,
    /// Number of poles.
    pub m: Option<usize>,
    /// Impose up-down symmetry (the gapless map).
    pub symmetry: bool,
    /// Maximum number of moments (default 200).
    pub k_max: usize,
    /// Ratio `L/N` of ESPRIT (default 0.4).
    pub lfactor: f64,
}

impl MiniPoleDlrParams {
    /// Parameters with contour start `n0`, tolerance `err` and defaults
    /// otherwise.
    pub fn new(n0: usize, err: f64) -> Self {
        Self {
            n0,
            nmax: None,
            err: Some(err),
            err_type: ErrType::Abs,
            m: None,
            symmetry: false,
            k_max: 200,
            lfactor: 0.4,
        }
    }
}

/// Minimal pole representation of `G(z) = Σ_l A_l / (z - x_l)`.
///
/// `al` has shape `[r, ...]` (residues along axis 0, trailing channel axes),
/// `xl` the `r` real pole locations.
///
/// # Errors
/// [`Error::ShapeMismatch`] if `al` does not have `len(xl)` rows,
/// [`Error::InvalidParameter`] for invalid parameters, and ESPRIT errors.
pub fn mini_pole_dlr<T>(
    al: &TypedTensor<T>,
    xl: &[f64],
    beta: f64,
    params: &MiniPoleDlrParams,
) -> Result<MiniPoleResult>
where
    T: TensorScalar + Copy + Into<C64>,
{
    let shape = al.shape().to_vec();
    let r = xl.len();
    if shape.first() != Some(&r) {
        let mut expected = shape.clone();
        match expected.first_mut() {
            Some(n) => *n = r,
            None => expected.push(r),
        }
        return Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected,
            actual: shape,
        });
    }
    if !(beta.is_finite() && beta > 0.0) {
        return Err(Error::InvalidParameter {
            name: "beta",
            value: format!("{beta:?}"),
            reason: "must be finite and positive".to_string(),
        });
    }
    let d: usize = shape[1..].iter().product();
    let al: Vec<C64> = al.host_data()?.iter().map(|&v| v.into()).collect();

    let pi = std::f64::consts::PI;
    let w_n0 = (2.0 * params.n0 as f64 + 1.0) * pi / beta;
    let generic;
    let gapless;
    let map: &dyn ConMap = if !params.symmetry {
        let nmax = params.nmax.unwrap_or(beta);
        let w_nmax = (2.0 * nmax + 1.0) * pi / beta;
        generic = ConMapGeneric {
            w_m: 0.5 * (w_n0 + w_nmax),
            dw_h: 0.5 * (w_nmax - w_n0),
        };
        if generic.dw_h.is_nan() || generic.dw_h <= 0.0 {
            return Err(Error::InvalidParameter {
                name: "nmax",
                value: format!("{nmax:?}"),
                reason: format!("must exceed n0 = {}", params.n0),
            });
        }
        &generic
    } else {
        gapless = ConMapGapless { w_min: w_n0 };
        &gapless
    };

    // h_k = Σ_l xl_p^k A_l / z'(xl_p), k < min(int((r + 1) / Lfactor), k_max).
    let xl_p: Vec<C64> = xl.iter().map(|&x| map.w(C64::new(x, 0.0))).collect();
    let n = (((r + 1) as f64 / params.lfactor) as usize).min(params.k_max);
    let mut h = vec![C64::new(0.0, 0.0); n * d];
    for (l, &q) in xl_p.iter().enumerate() {
        let f = map.dz(q).inv();
        let mut p = C64::new(1.0, 0.0);
        for k in 0..n {
            for c in 0..d {
                h[k + n * c] += p * (al[l + r * c] * f);
            }
            p *= q;
        }
    }

    let esprit = Esprit::new(
        &h,
        n,
        d,
        &EspritParams {
            err: params.err,
            err_type: params.err_type,
            m: params.m,
            lfactor: params.lfactor,
            ..EspritParams::default()
        },
    )?;
    let keep: Vec<usize> = (0..esprit.gamma.len())
        .filter(|&j| esprit.gamma[j].norm() < 1.0)
        .collect();
    let location: Vec<C64> = keep.iter().map(|&j| map.z(esprit.gamma[j])).collect();
    let mut weight = vec![C64::new(0.0, 0.0); keep.len() * d];
    for (jn, &j) in keep.iter().enumerate() {
        let f = map.dz(esprit.gamma[j]);
        for c in 0..d {
            weight[jn + keep.len() * c] = esprit.omega[j + esprit.m * c] * f;
        }
    }
    let mut hshape = shape;
    hshape[0] = n;
    assemble(
        location,
        weight,
        vec![C64::new(0.0, 0.0); d],
        h,
        hshape,
        esprit,
        params.n0,
        None,
    )
}

/// [`mini_pole_dlr`] of DLR coefficients: `A_l = g_l w_l` at the DLR poles
/// `x_l`, with `w_l` the pole weights of the DLR.
///
/// # Errors
/// See [`mini_pole_dlr`].
pub fn mini_pole_dlr_from<S, T>(
    dlr: &DiscreteLehmannRepresentation<S>,
    g_dlr: &TypedTensor<T>,
    params: &MiniPoleDlrParams,
) -> Result<MiniPoleResult>
where
    S: StatisticsType + 'static,
    T: TensorScalar + Copy + Into<C64>,
{
    let shape = g_dlr.shape().to_vec();
    let r = dlr.poles().len();
    if shape.first() != Some(&r) {
        let mut expected = shape.clone();
        match expected.first_mut() {
            Some(n) => *n = r,
            None => expected.push(r),
        }
        return Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected,
            actual: shape,
        });
    }
    let weights = dlr.pole_weights();
    let g = g_dlr.host_data()?;
    let al: Vec<C64> = g
        .iter()
        .enumerate()
        .map(|(i, &v)| v.into() * weights[i % r])
        .collect();
    let al = TypedTensor::from_vec_col_major(shape, al)?;
    mini_pole_dlr(&al, dlr.poles(), dlr.beta(), params)
}
