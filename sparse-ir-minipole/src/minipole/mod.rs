//! Port of the minimal pole method of Green-Phys/MiniPole.
//!
//! Ported from <https://github.com/Green-Phys/MiniPole> at commit `15e4a54`
//! (MIT License, Copyright (c) 2024 lzphy; see `LICENSE-THIRD-PARTY` of this
//! crate): `con_map.py` (`ConMapGeneric`, `ConMapGapless`),
//! `mini_pole_dlr.py` and `mini_pole.py`. L. Zhang and E. Gull, Phys. Rev. B
//! 110, 035154 (2024); L. Zhang, Y. Yu and E. Gull, Phys. Rev. B 110, 235131
//! (2024).
//!
//! Differences from the reference:
//! - the error tolerance (or the number of poles) is required: the automatic
//!   choice by knee detection (`kneed`) is not ported;
//! - the contour integrals of [`mini_pole`] use composite Gauss–Legendre
//!   quadrature instead of QUADPACK's QAWO, to the same tolerance;
//! - arrays are column-major tensors, and the channels of a matrix-valued
//!   function are its trailing axes.

mod con_map;
mod mini_pole;
mod mini_pole_dlr;
mod quad;

pub use crate::esprit::{ErrType, Esprit, EspritParams};
pub use con_map::{ConMap, ConMapGapless, ConMapGeneric};
pub use mini_pole::{MiniPoleParams, N0, Plane, mini_pole};
pub use mini_pole_dlr::{MiniPoleDlrParams, mini_pole_dlr, mini_pole_dlr_from};

use crate::error::Result;
use num_complex::Complex;
use tenferro_tensor::TypedTensor;

type C64 = Complex<f64>;

/// A minimal pole representation `G(z) = Σ_j A_j / (z - ξ_j) + C`.
#[derive(Debug)]
pub struct MiniPoleResult {
    /// Pole locations `ξ_j`, sorted by real part.
    pub pole_location: Vec<C64>,
    /// Pole weights `A_j`, shape `[M, ...]`.
    pub pole_weight: TypedTensor<C64>,
    /// Constant term `C`, one entry per channel (column-major).
    pub constant: Vec<C64>,
    /// Moments (contour integrals) `h_k`, shape `[K, ...]`.
    pub h_k: TypedTensor<C64>,
    /// The ESPRIT approximation of `h_k`.
    pub esprit: Esprit,
    /// `n0` used for the contour.
    pub n0: usize,
    /// Precision of the first approximation of the Matsubara data
    /// ([`mini_pole`] only).
    pub err_max: Option<f64>,
}

impl MiniPoleResult {
    /// Evaluate `G(z)`; returns shape `[len(z), ...]`.
    ///
    /// # Errors
    /// Propagates tensor construction errors.
    pub fn evaluate(&self, z: &[C64]) -> Result<TypedTensor<C64>> {
        let shape = self.pole_weight.shape().to_vec();
        let r = self.pole_location.len();
        let d: usize = shape[1..].iter().product();
        let a = self.pole_weight.host_data()?;
        let mut out = vec![C64::new(0.0, 0.0); z.len() * d];
        for (i, &zi) in z.iter().enumerate() {
            for c in 0..d {
                out[i + z.len() * c] = self.constant[c];
            }
            for (j, &p) in self.pole_location.iter().enumerate() {
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
}

/// Sort the poles by real part and build the result.
#[allow(clippy::too_many_arguments)]
fn assemble(
    location: Vec<C64>,
    weight: Vec<C64>,
    constant: Vec<C64>,
    h: Vec<C64>,
    hshape: Vec<usize>,
    esprit: Esprit,
    n0: usize,
    err_max: Option<f64>,
) -> Result<MiniPoleResult> {
    let r = location.len();
    let d = constant.len();
    let mut order: Vec<usize> = (0..r).collect();
    order.sort_by(|&a, &b| location[a].re.total_cmp(&location[b].re));
    let pole_location: Vec<C64> = order.iter().map(|&j| location[j]).collect();
    let mut w = vec![C64::new(0.0, 0.0); r * d];
    for (jn, &j) in order.iter().enumerate() {
        for c in 0..d {
            w[jn + r * c] = weight[j + r * c];
        }
    }
    let mut wshape = hshape.clone();
    wshape[0] = r;
    Ok(MiniPoleResult {
        pole_location,
        pole_weight: TypedTensor::from_vec_col_major(wshape, w)?,
        constant,
        h_k: TypedTensor::from_vec_col_major(hshape, h)?,
        esprit,
        n0,
        err_max,
    })
}

#[cfg(test)]
mod tests;
