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
//!
//! # Entry points
//!
//! - [`mini_pole_dlr_from`] takes a DLR and the coefficients `g_l` that
//!   `MatsubaraSampling::fit_nd` returns for it, and forms the residues
//!   `A_l = g_l w_l` with `dlr.pole_weights()`. Bosonic DLR coefficients are
//!   therefore not residues; do not pass them to [`mini_pole_dlr`].
//! - [`mini_pole_dlr`] takes real poles `x_l`, their actual residues `A_l`
//!   (pole axis first, channels on the trailing axes, column-major) and `β`.
//! - [`mini_pole`] takes Matsubara data on a uniform grid of non-negative
//!   *physical* frequencies `ω_n` (real numbers, not indices and not `iω_n`).
//!
//! All three return a [`MiniPoleResult`]; [`MiniPoleResult::evaluate`] sums the
//! poles and the constant term at arbitrary complex `z`.
//!
//! # The contour of the DLR entry points
//!
//! Without symmetry ([`MiniPoleDlrParams::symmetry`] false), the conformal map
//! uses the imaginary-axis segment `[iω_{n0}, iω_{nmax}]` with
//! `ω_n = (2n + 1)π/β`, **also for a bosonic DLR**. This is a contour index,
//! not the physical bosonic grid and not the reduced Matsubara index `n`
//! (frequency `nπ/β`) of the rest of the library and of the C API.
//!
//! - `n0` is chosen by the caller; there is no automatic choice for DLR input.
//!   Increasing it raises the lower end of the contour.
//! - `nmax: None` uses the numerical value of `β` in the units of the input.
//!   An explicit `Some(nmax)` must exceed `n0`; it is a floating-point cutoff,
//!   not a number of samples.
//! - With symmetry the gapless map uses the lower end only, and `nmax` does
//!   not enter.
//!
//! Changing the segment changes the mapped poles and moments that ESPRIT
//! sees, so the same coefficients and tolerance need not give the same number
//! of poles.
//!
//! # Tolerance and number of poles
//!
//! `err` is the ESPRIT tolerance: absolute with [`ErrType::Abs`] (the
//! default), relative to the largest singular value with [`ErrType::Rel`].
//! It is **not** a bound on the reconstruction error of `G`. With the DLR
//! entry points one may instead set `m = Some(count)` and `err = None` when the
//! model order is known; one of the two is required, because the automatic
//! choice by knee detection is not ported.
//!
//! # Matsubara input ([`mini_pole`])
//!
//! - At least three finite, non-negative, strictly increasing, uniformly
//!   spaced frequencies, with data of shape `[n_w]` or `[n_w, n_orb, n_orb]`.
//!   Sparse DLR/IR sampling nodes, irregular grids and negative frequencies
//!   are not accepted; fit a DLR and use [`mini_pole_dlr_from`] for those.
//! - `n0` is a **position in the supplied array**: the contour starts at
//!   `w[n0]` and, without symmetry, ends at the last element. There is no
//!   `nmax`; extend the data for a larger upper end. The default is
//!   [`N0::Auto`] with `shift: 0`; inspect [`MiniPoleResult::n0`] and compare
//!   with [`N0::Fixed`] if needed. With symmetry `w[n0]` must be positive, so a
//!   bosonic zero frequency cannot be the lower end of the gapless map.
//! - `err` is required and should be at least the noise level of the data.
//!   [`MiniPoleResult::err_max`] reports the precision of the first ESPRIT
//!   interpolation, not a certified reconstruction error; it is `None` for
//!   DLR input.
//! - [`MiniPoleParams::g_symmetric`] symmetrizes matrix data as
//!   `G_ij = G_ji`, independently of the up-down `symmetry` flag.
//!   [`MiniPoleParams::compute_const`] fits a constant term and cannot be
//!   combined with `symmetry`.
//! - Residues default to a least-squares fit in [`Plane::Z`] without symmetry
//!   and to the mapped [`Plane::W`] with symmetry;
//!   [`MiniPoleParams::include_n0`] also includes the first `n0` points in the
//!   z-plane fit.
//!
//! # C API
//!
//! `spir_minipole_from_matsubara` takes reduced Matsubara indices `n`
//! (`1, 3, 5, ...` for fermions, `0, 2, 4, ...` for bosons) and converts them
//! as `ω = nπ/β`; a negative `n0` selects the automatic choice and `n0_shift`
//! adds to it. `spir_minipole_from_dlr` follows [`mini_pole_dlr_from`], with
//! `nmax <= 0` selecting `β`. The getters expose the poles, residues, constant,
//! the `n0` used and the Matsubara-only `err_max`.
//!
//! A worked example with figures is the MiniPole page of the sparse-ir Rust
//! user guide: <https://spm-lab.github.io/sparse-ir-rs/tutorials/minipole.html>.

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
