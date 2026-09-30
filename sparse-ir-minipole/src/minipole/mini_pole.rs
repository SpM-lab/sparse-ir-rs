//! MPM: minimal poles from Matsubara data on a uniform grid (port of
//! `mini_pole/mini_pole.py`).
//!
//! Ported from Green-Phys/MiniPole (commit 15e4a54, MIT License,
//! Copyright (c) 2024 lzphy); see `LICENSE-THIRD-PARTY`.

use super::con_map::{ConMap, ConMapGapless, ConMapGeneric};
use super::quad::oscillatory;
use super::{MiniPoleResult, assemble};
use crate::error::{ArrayRole, Error, Result};
use crate::esprit::{ErrType, Esprit, EspritParams, linspace};
use crate::linalg::lstsq;
use num_complex::Complex;
use std::f64::consts::PI;
use tenferro_tensor::TypedTensor;

type C64 = Complex<f64>;

/// Choice of `n0`, the number of low frequencies left out of the contour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum N0 {
    /// Chosen from the data, plus `shift`.
    Auto {
        /// Shift added to the automatic choice.
        shift: usize,
    },
    /// Fixed.
    Fixed(usize),
}

/// Plane in which the pole weights are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// Least squares on the Matsubara data.
    Z,
    /// From the ESPRIT weights in the mapped plane.
    W,
}

/// Parameters of [`mini_pole`], with the defaults of the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniPoleParams {
    /// Choice of `n0` (default automatic, shift 0).
    pub n0: N0,
    /// Error tolerance, at least the noise level. Required: the reference's
    /// default (knee detection) is not ported.
    pub err: f64,
    /// Interpretation of `err` (default absolute).
    pub err_type: ErrType,
    /// Number of poles; `None` uses the precision of the first ESPRIT.
    pub m: Option<usize>,
    /// Preserve up-down symmetry.
    pub symmetry: bool,
    /// Symmetrize the data as `G_ij(z) = G_ji(z)`.
    pub g_symmetric: bool,
    /// Fit a constant term of `G`.
    pub compute_const: bool,
    /// Plane for the pole weights; `None` uses `Z` without and `W` with
    /// symmetry.
    pub plane: Option<Plane>,
    /// Include the first `n0` points in the weight fit in the z plane.
    pub include_n0: bool,
    /// Maximum number of contour integrals (default 999).
    pub k_max: usize,
    /// Maximum ratio of oscillation when choosing `n0` (default 10).
    pub ratio_max: f64,
}

impl MiniPoleParams {
    /// Parameters with tolerance `err` and defaults otherwise.
    pub fn new(err: f64) -> Self {
        Self {
            n0: N0::Auto { shift: 0 },
            err,
            err_type: ErrType::Abs,
            m: None,
            symmetry: false,
            g_symmetric: false,
            compute_const: false,
            plane: None,
            include_n0: false,
            k_max: 999,
            ratio_max: 10.0,
        }
    }
}

/// Minimal pole representation of Matsubara data.
///
/// `g_w` has shape `[n_w]` or `[n_w, n_orb, n_orb]`, `w` the corresponding
/// finite, increasing, uniformly spaced non-negative Matsubara frequencies
/// `ω_n` (real).
///
/// # Errors
/// [`Error::ShapeMismatch`] for inconsistent shapes,
/// [`Error::InvalidParameter`] for a grid that is not finite, increasing,
/// uniform and non-negative or invalid parameters, and ESPRIT errors.
pub fn mini_pole(
    g_w: &TypedTensor<C64>,
    w: &[f64],
    params: &MiniPoleParams,
) -> Result<MiniPoleResult> {
    let shape = g_w.shape().to_vec();
    let nw = w.len();
    let n_orb = match shape.len() {
        1 => 1,
        3 if shape[1] == shape[2] => shape[1],
        _ => {
            return Err(Error::ShapeMismatch {
                which: ArrayRole::Input,
                expected: vec![
                    nw,
                    shape.get(1).copied().unwrap_or(1),
                    shape.get(1).copied().unwrap_or(1),
                ],
                actual: shape,
            });
        }
    };
    if shape[0] != nw {
        let mut expected = shape.clone();
        expected[0] = nw;
        return Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected,
            actual: shape,
        });
    }
    if nw < 3
        || w.iter().any(|x| !x.is_finite() || *x < 0.0)
        || w.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(Error::InvalidParameter {
            name: "w",
            value: format!("of length {nw} starting at {:?}", w.first()),
            reason: "must have at least 3 finite, non-negative, strictly increasing frequencies"
                .to_string(),
        });
    }
    let wabs = w.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let dd = (0..nw - 2)
        .map(|i| ((w[i + 2] - 2.0 * w[i + 1] + w[i]) / wabs).abs())
        .fold(0.0, f64::max);
    if dd.is_nan() || dd >= 1e-6 {
        return Err(Error::InvalidParameter {
            name: "w",
            value: format!("with second differences up to {dd:e}"),
            reason: "must be uniformly spaced".to_string(),
        });
    }
    if params.symmetry && params.compute_const {
        return Err(Error::InvalidParameter {
            name: "compute_const",
            value: "true".to_string(),
            reason: "set symmetry to false to calculate the overall constant".to_string(),
        });
    }
    let d = n_orb * n_orb;
    let tr = |c: usize| (c % n_orb) * n_orb + c / n_orb;
    let raw = g_w.host_data()?;
    let g: Vec<C64> = if params.g_symmetric {
        (0..nw * d)
            .map(|idx| {
                let (i, c) = (idx % nw, idx / nw);
                (raw[idx] + raw[i + nw * tr(c)]) * 0.5
            })
            .collect()
    } else {
        raw.to_vec()
    };
    let plane = params
        .plane
        .unwrap_or(if params.symmetry { Plane::W } else { Plane::Z });

    // First ESPRIT on each component.
    let first = |lfactor: f64| -> Result<Vec<Esprit>> {
        (0..d)
            .map(|c| {
                Esprit::new(
                    &g[nw * c..nw * (c + 1)],
                    nw,
                    1,
                    &EspritParams {
                        x_min: w[0],
                        x_max: w[nw - 1],
                        err: Some(params.err),
                        err_type: params.err_type,
                        lfactor,
                        ..EspritParams::default()
                    },
                )
            })
            .collect()
    };
    let p_o = first(0.4)?;
    let (n0, err_max) = match params.n0 {
        N0::Auto { shift } => {
            let p_o2 = first(0.5)?;
            let w_cont = linspace(w[0], w[nw - 1], 10 * nw - 9);
            let err_max = p_o
                .iter()
                .chain(&p_o2)
                .map(|p| p.err_max)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut n0 = 0;
            for c in 0..d {
                let l1 = p_o[c].get_value(&w_cont);
                let l2 = p_o2[c].get_value(&w_cont);
                let diff: Vec<f64> = (0..nw - 1)
                    .map(|j| {
                        (0..10)
                            .map(|t| (l2[10 * j + t] - l1[10 * j + t]).norm())
                            .fold(f64::NEG_INFINITY, f64::max)
                    })
                    .collect();
                let first_ok = (0..nw - 2)
                    .position(|j| diff[j] <= err_max && diff[j] / diff[j + 1] < params.ratio_max)
                    .unwrap_or(0);
                n0 = n0.max(first_ok);
            }
            (n0 + shift, err_max)
        }
        N0::Fixed(n0) => (
            n0,
            p_o.iter()
                .map(|p| p.err_max)
                .fold(f64::NEG_INFINITY, f64::max),
        ),
    };
    if n0 + 1 >= nw {
        return Err(Error::InvalidParameter {
            name: "n0",
            value: n0.to_string(),
            reason: format!("must be less than len(w) - 1 = {}", nw - 1),
        });
    }
    if params.symmetry && w[n0] <= 0.0 {
        // ConMapGapless needs ω_min > 0 (e.g. n0 = 0 on a bosonic grid).
        return Err(Error::InvalidParameter {
            name: "n0",
            value: n0.to_string(),
            reason: format!("must give w[n0] > 0 with symmetry, got {:?}", w[n0]),
        });
    }
    let head = |c: usize, x: f64| p_o[c].get_value_indiv(x, 0);
    let cutoff = err_max;
    let qerr = 0.01 * cutoff;

    let generic;
    let gapless;
    let mut constant = vec![C64::new(0.0, 0.0); d];
    let (map, h, nk): (&dyn ConMap, Vec<C64>, usize) = if !params.symmetry {
        generic = ConMapGeneric {
            w_m: 0.5 * (w[n0] + w[nw - 1]),
            dw_h: 0.5 * (w[nw - 1] - w[n0]),
        };
        let (w_m, dw_h) = (generic.w_m, generic.dw_h);
        let (h, nk) = moments(params.k_max, d, cutoff, |k, c| {
            let f = |x: f64| head(c, w_m + dw_h * x.sin());
            let v = oscillatory(&f, -0.5 * PI, 0.5 * PI, (k + 1) as f64, k & 1 == 0, qerr);
            if k & 1 == 0 {
                v * C64::new(0.0, 1.0 / PI)
            } else {
                v / PI
            }
        });
        (&generic, h, nk)
    } else {
        // Complex poles for the data in [iω_max, i∞).
        let sub = mini_pole(
            g_w,
            w,
            &MiniPoleParams {
                m: None,
                symmetry: false,
                plane: None,
                include_n0: false,
                ..params.clone()
            },
        )?;
        constant = sub.constant.clone();
        let sub_loc = sub.pole_location.clone();
        let sub_w = sub.pole_weight.host_data()?.to_vec();
        let r_sub = sub_loc.len();
        let tail = move |c: usize, x: f64| -> C64 {
            let z = C64::new(0.0, x);
            (0..r_sub)
                .map(|j| sub_w[j + r_sub * c] / (z - sub_loc[j]))
                .sum()
        };
        gapless = ConMapGapless { w_min: w[n0] };
        let w_min = gapless.w_min;
        let theta0 = (w_min / w[nw - 1]).asin();
        let (ha, hb) = (theta0 + 1e-12, 0.5 * PI);
        let (ta, tb) = (1e-6, theta0 - 1e-12);
        let integral = |f: &dyn Fn(f64) -> C64, a: f64, b: f64, k: usize| {
            oscillatory(f, a, b, (k + 1) as f64, k & 1 == 0, qerr)
        };
        // cal_hk_gapless_symmetric_indiv on one function
        let sym = |gf: &dyn Fn(f64) -> C64, k: usize, a: f64, b: f64| -> C64 {
            if k & 1 == 0 {
                let f = |x: f64| C64::new(gf(w_min / x.sin()).im, 0.0);
                integral(&f, a, b, k) * (-2.0 / PI)
            } else {
                let f = |x: f64| C64::new(gf(w_min / x.sin()).re, 0.0);
                integral(&f, a, b, k) * (2.0 / PI)
            }
        };
        let raw_int = |gf: &dyn Fn(f64) -> C64, k: usize, a: f64, b: f64| -> C64 {
            let f = |x: f64| gf(w_min / x.sin());
            integral(&f, a, b, k)
        };
        let sym_hk = |c: usize, k: usize| -> C64 {
            sym(&|x| head(c, x), k, ha, hb) + sym(&|x| tail(c, x), k, ta, tb)
        };
        let (h, nk) = if params.g_symmetric {
            moments(params.k_max, d, cutoff, |k, c| sym_hk(c, k))
        } else {
            moments_rows(params.k_max, d, cutoff, |k| {
                let mut row = vec![C64::new(0.0, 0.0); d];
                for i in 0..n_orb {
                    for j in i..n_orb {
                        // Channel of G_ij in the column-major layout.
                        let c1 = i + n_orb * j;
                        let c2 = j + n_orb * i;
                        if i == j {
                            row[c1] = sym_hk(c1, k);
                        } else {
                            let h1 = raw_int(&|x| head(c1, x), k, ha, hb)
                                + raw_int(&|x| tail(c1, x), k, ta, tb);
                            let h2 = raw_int(&|x| head(c2, x), k, ha, hb)
                                + raw_int(&|x| tail(c2, x), k, ta, tb);
                            if k & 1 == 0 {
                                let f = C64::new(0.0, 1.0 / PI);
                                row[c1] = f * (h1 - h2.conj());
                                row[c2] = f * (h2 - h1.conj());
                            } else {
                                row[c1] = (h1 + h2.conj()) / PI;
                                row[c2] = (h2 + h1.conj()) / PI;
                            }
                        }
                    }
                }
                row
            })
        };
        (&gapless, h, nk)
    };

    // find_poles: second ESPRIT on the contour integrals.
    let esprit = Esprit::new(
        &h,
        nk,
        d,
        &EspritParams {
            err: if params.m.is_none() {
                Some(0.5 * err_max)
            } else {
                None
            },
            m: params.m,
            lfactor: 0.5,
            ..EspritParams::default()
        },
    )?;
    let keep: Vec<usize> = (0..esprit.gamma.len())
        .filter(|&j| esprit.gamma[j].norm() < 1.0)
        .collect();
    let r = keep.len();
    let location: Vec<C64> = keep.iter().map(|&j| map.z(esprit.gamma[j])).collect();
    let mut weight = vec![C64::new(0.0, 0.0); r * d];
    for (jn, &j) in keep.iter().enumerate() {
        let f = map.dz(esprit.gamma[j]);
        for c in 0..d {
            weight[jn + r * c] = esprit.omega[j + esprit.m * c] * f;
        }
    }
    if params.compute_const {
        let nf = nw - n0;
        let mut cst = vec![C64::new(0.0, 0.0); d];
        for (c, v) in cst.iter_mut().enumerate() {
            for i in n0..nw {
                let z = C64::new(0.0, w[i]);
                let approx: C64 = (0..r).map(|j| weight[j + r * c] / (z - location[j])).sum();
                *v += g[i + nw * c] - approx;
            }
            *v /= nf as f64;
        }
        let big = cst.iter().map(|v| v.norm()).fold(0.0, f64::max) > 100.0 * err_max;
        constant = if big {
            cst
        } else {
            vec![C64::new(0.0, 0.0); d]
        };
    }
    if plane == Plane::Z {
        let start = if params.include_n0 { 0 } else { n0 };
        let mut ws: Vec<f64> = Vec::new();
        let mut rows: Vec<(usize, bool)> = Vec::new(); // (frequency index, mirrored)
        if params.symmetry {
            for i in (start..nw).rev() {
                ws.push(-w[i]);
                rows.push((i, true));
            }
        }
        ws.extend_from_slice(&w[start..]);
        rows.extend((start..nw).map(|i| (i, false)));
        let nr = ws.len();
        let mut a = vec![C64::new(0.0, 0.0); nr * r];
        for (j, &loc) in location.iter().enumerate() {
            for (i, &x) in ws.iter().enumerate() {
                a[i + nr * j] = (C64::new(0.0, x) - loc).inv();
            }
        }
        let mut b = vec![C64::new(0.0, 0.0); nr * d];
        for c in 0..d {
            for (i, &(fi, mirrored)) in rows.iter().enumerate() {
                let v = if mirrored {
                    g[fi + nw * tr(c)].conj()
                } else {
                    g[fi + nw * c]
                };
                b[i + nr * c] = v - constant[c];
            }
        }
        weight = lstsq(&a, nr, r, &b, d)?;
    }
    // Discard poles with negligible weights.
    let keep: Vec<usize> = (0..r)
        .filter(|&j| (0..d).map(|c| weight[j + r * c].norm()).fold(0.0, f64::max) > err_max)
        .collect();
    let location_k: Vec<C64> = keep.iter().map(|&j| location[j]).collect();
    let mut weight_k = vec![C64::new(0.0, 0.0); keep.len() * d];
    for (jn, &j) in keep.iter().enumerate() {
        for c in 0..d {
            weight_k[jn + keep.len() * c] = weight[j + r * c];
        }
    }
    let mut hshape = shape;
    hshape[0] = nk;
    assemble(
        location_k,
        weight_k,
        constant,
        h,
        hshape,
        esprit,
        n0,
        Some(err_max),
    )
}

/// Rows `h_k` for `k = 0, 1, …` until two successive rows are below
/// `cutoff` in every component, at most `k_max` rows; column-major `K x d`.
fn moments(
    k_max: usize,
    d: usize,
    cutoff: f64,
    f: impl Fn(usize, usize) -> C64,
) -> (Vec<C64>, usize) {
    moments_rows(k_max, d, cutoff, |k| (0..d).map(|c| f(k, c)).collect())
}

fn moments_rows(
    k_max: usize,
    d: usize,
    cutoff: f64,
    f: impl Fn(usize) -> Vec<C64>,
) -> (Vec<C64>, usize) {
    let mut rows: Vec<Vec<C64>> = Vec::new();
    for k in 0..k_max {
        rows.push(f(k));
        if k >= 1 && (0..d).all(|c| rows[k][c].norm() < cutoff && rows[k - 1][c].norm() < cutoff) {
            break;
        }
    }
    let nk = rows.len();
    let mut h = vec![C64::new(0.0, 0.0); nk * d];
    for (k, row) in rows.iter().enumerate() {
        for c in 0..d {
            h[k + nk * c] = row[c];
        }
    }
    (h, nk)
}
