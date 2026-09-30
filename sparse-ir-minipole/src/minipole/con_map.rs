//! Holomorphic maps between the z plane and the unit disk (port of
//! `mini_pole/con_map.py`).
//!
//! Ported from Green-Phys/MiniPole (commit 15e4a54, MIT License,
//! Copyright (c) 2024 lzphy); see `LICENSE-THIRD-PARTY`.

use num_complex::Complex;

type C64 = Complex<f64>;

/// A holomorphic map `z(w)` from the unit disk onto the z plane cut along
/// the integration contour.
pub trait ConMap {
    /// `z(w)`.
    fn z(&self, w: C64) -> C64;
    /// The preimage `w(z)` inside the unit disk.
    fn w(&self, z: C64) -> C64;
    /// `dz/dw`.
    fn dz(&self, w: C64) -> C64;
}

/// `z = (Δω_h/2)(w - 1/w) + iω_m`: the unit circle onto the segment
/// `[i(ω_m - Δω_h), i(ω_m + Δω_h)]` (`ConMapGeneric`, `branch_in = True`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConMapGeneric {
    /// Midpoint `ω_m` of the segment.
    pub w_m: f64,
    /// Half-width `Δω_h` of the segment.
    pub dw_h: f64,
}

impl ConMap for ConMapGeneric {
    fn z(&self, w: C64) -> C64 {
        if w == C64::new(0.0, 0.0) {
            return C64::new(f64::INFINITY, 0.0);
        }
        (w - w.inv()) * (0.5 * self.dw_h) + C64::new(0.0, self.w_m)
    }

    fn w(&self, z: C64) -> C64 {
        let x = (z - C64::new(0.0, self.w_m)) / self.dw_h;
        let mut w = x - (x * x + 1.0).sqrt();
        if w.norm() > 1.0 {
            w = x * 2.0 - w;
        }
        w
    }

    fn dz(&self, w: C64) -> C64 {
        if w == C64::new(0.0, 0.0) {
            return C64::new(f64::INFINITY, 0.0);
        }
        (C64::new(1.0, 0.0) + (w * w).inv()) * (0.5 * self.dw_h)
    }
}

/// `z = 2 ω_min w / (1 - w²)`: the unit circle onto `i(-∞, -ω_min] ∪
/// i[ω_min, ∞)` (`ConMapGapless`), for data with up-down symmetry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConMapGapless {
    /// `ω_min > 0`.
    pub w_min: f64,
}

impl ConMap for ConMapGapless {
    fn z(&self, w: C64) -> C64 {
        if w == C64::new(1.0, 0.0) || w == C64::new(-1.0, 0.0) {
            return C64::new(f64::INFINITY, 0.0);
        }
        w * (2.0 * self.w_min) / (C64::new(1.0, 0.0) - w * w)
    }

    fn w(&self, z: C64) -> C64 {
        if z == C64::new(0.0, 0.0) {
            return C64::new(0.0, 0.0);
        }
        let wm = self.w_min;
        let mut w = ((z * z).inv() + 1.0 / (wm * wm)).sqrt() - z.inv();
        w *= wm;
        if w.norm() > 1.0 {
            w = -w - z.inv() * (2.0 * wm);
        }
        w
    }

    fn dz(&self, w: C64) -> C64 {
        if w == C64::new(1.0, 0.0) || w == C64::new(-1.0, 0.0) {
            return C64::new(f64::INFINITY, 0.0);
        }
        let one = C64::new(1.0, 0.0);
        let d = one - w * w;
        (one + w * w) * (2.0 * self.w_min) / (d * d)
    }
}
