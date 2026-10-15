//! Piecewise Legendre polynomial Fourier transform implementation for SparseIR
//!
//! This module provides Fourier transform functionality for piecewise Legendre
//! polynomials, enabling evaluation in Matsubara frequency domain.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::error::Error;
use crate::freq::MatsubaraFreq;
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::special_functions::spherical_bessel_j;
use crate::traits::{Bosonic, Fermionic, Statistics, StatisticsType};

/// Power model for asymptotic behavior
#[derive(Debug, Clone)]
pub struct PowerModel {
    pub(crate) moments: Vec<f64>,
}

impl PowerModel {
    /// Create a new power model with given moments
    pub fn new(moments: Vec<f64>) -> Self {
        Self { moments }
    }
}

/// Piecewise Legendre polynomial with Fourier transform capability
///
/// This represents a piecewise Legendre polynomial that can be evaluated
/// in the Matsubara frequency domain using Fourier transform.
#[derive(Debug, Clone)]
pub struct PiecewiseLegendreFT<S: StatisticsType> {
    /// The underlying piecewise Legendre polynomial
    pub(crate) poly: PiecewiseLegendrePoly,
    /// Asymptotic cutoff frequency index
    pub(crate) n_asymp: f64,
    /// Power model for asymptotic behavior
    pub(crate) model: PowerModel,
    // `fn() -> S` keeps the type Send + Sync whatever `S` is (it holds no S)
    _phantom: std::marker::PhantomData<fn() -> S>,
}

// Type aliases for convenience
pub type FermionicPiecewiseLegendreFT = PiecewiseLegendreFT<Fermionic>;
pub type BosonicPiecewiseLegendreFT = PiecewiseLegendreFT<Bosonic>;

impl<S: StatisticsType> PiecewiseLegendreFT<S> {
    /// Create a new PiecewiseLegendreFT from a polynomial and statistics
    ///
    /// # Arguments
    /// * `poly` - The underlying piecewise Legendre polynomial
    /// * `stat` - Statistics type (Fermionic or Bosonic)
    /// * `n_asymp` - Asymptotic cutoff frequency index (default: infinity)
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidParameter`] if the domain of `poly` is not [-1, 1]
    ///   within 1e-12, or `n_asymp` is NaN or negative (infinity, the
    ///   default, disables the asymptotic model)
    pub fn new(poly: PiecewiseLegendrePoly, _stat: S, n_asymp: Option<f64>) -> Result<Self, Error> {
        // Validate domain
        if !((poly.get_xmin() + 1.0).abs() <= 1e-12 && (poly.get_xmax() - 1.0).abs() <= 1e-12) {
            return Err(Error::InvalidParameter {
                name: "poly",
                value: format!(
                    "a polynomial on [{:?}, {:?}]",
                    poly.get_xmin(),
                    poly.get_xmax()
                ),
                reason: "must be defined on [-1, 1] (within 1e-12)".to_string(),
            });
        }
        if let Some(n) = n_asymp.filter(|n| !(*n >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "n_asymp",
                value: format!("{n:?}"),
                reason: "must be non-negative (infinity disables the asymptotic model)".to_string(),
            });
        }

        let n_asymp = n_asymp.unwrap_or(f64::INFINITY);
        let model = Self::power_model(&poly);

        Ok(Self {
            poly,
            n_asymp,
            model,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the asymptotic cutoff frequency index
    pub fn get_n_asymp(&self) -> f64 {
        self.n_asymp
    }

    /// Get the statistics type
    pub fn get_statistics(&self) -> Statistics {
        S::STATISTICS
    }

    /// Get the zeta value for this statistics type
    pub fn zeta(&self) -> i64 {
        match S::STATISTICS {
            Statistics::Fermionic => 1,
            Statistics::Bosonic => 0,
        }
    }

    /// Get a reference to the underlying polynomial
    pub fn get_poly(&self) -> &PiecewiseLegendrePoly {
        &self.poly
    }

    /// Evaluate the Fourier transform at a Matsubara frequency
    ///
    /// # Arguments
    /// * `omega` - Matsubara frequency
    ///
    /// # Returns
    /// The complex Fourier transform value
    pub fn evaluate(&self, omega: &MatsubaraFreq<S>) -> Complex64 {
        let n = omega.get_n();
        if (n as f64).abs() < self.n_asymp {
            self.compute_unl_inner(&self.poly, n)
        } else {
            self.giw(n)
        }
    }

    /// Evaluate at integer Matsubara index
    ///
    /// # Errors
    ///
    /// [`Error::InvalidMatsubaraIndex`] if `n` has the wrong parity for the
    /// statistics (it returned 0 before)
    pub fn evaluate_at_n(&self, n: i64) -> Result<Complex64, Error> {
        Ok(self.evaluate(&MatsubaraFreq::<S>::new(n)?))
    }

    /// Evaluate at multiple Matsubara indices
    ///
    /// # Errors
    ///
    /// The error of [`Self::evaluate_at_n`] for the first invalid index
    pub fn evaluate_at_ns(&self, ns: &[i64]) -> Result<Vec<Complex64>, Error> {
        ns.iter().map(|&n| self.evaluate_at_n(n)).collect()
    }

    /// Create power model for asymptotic behavior
    fn power_model(poly: &PiecewiseLegendrePoly) -> PowerModel {
        // Evaluate at the ends of the domain: `new` accepts an xmin/xmax
        // within 1e-12 of -1/1, where derivs(-1.0)/derivs(1.0) would be
        // outside the domain.
        PowerModel::new(Self::power_moments(
            &poly.derivs(poly.get_xmax()),
            &poly.derivs(poly.get_xmin()),
        ))
    }

    /// Asymptotic power moments from the boundary derivatives at both ends
    ///
    /// `M_m = -( statsign · (-1)^(m+1) · u^(m)(+1) + (-1)^m · u^(m)(-1) ) / sqrt(2)`,
    /// where `deriv_p[m]` / `deriv_m[m]` is the order-m derivative at x = +1 / -1.
    /// No parity is assumed, so this holds for functions without definite
    /// parity (`symm == 0`, e.g. from `compute_sve_general`); for a function
    /// of parity `(-1)^l` it reduces to the one-sided form with
    /// `u^(m)(-1) = (-1)^(l+m) u^(m)(+1)`.
    fn power_moments(deriv_p: &[f64], deriv_m: &[f64]) -> Vec<f64> {
        let statsign = match S::STATISTICS {
            Statistics::Fermionic => -1.0,
            Statistics::Bosonic => 1.0,
        };
        deriv_p
            .iter()
            .zip(deriv_m)
            .enumerate()
            .map(|(m, (&dp, &dm))| {
                let sign_p = (-1.0_f64).powi((m + 1) as i32);
                let sign_m = (-1.0_f64).powi(m as i32);
                -(statsign * sign_p * dp + sign_m * dm) / 2.0_f64.sqrt()
            })
            .collect()
    }

    /// Compute the inner Fourier transform (for small frequencies)
    fn compute_unl_inner(&self, poly: &PiecewiseLegendrePoly, wn: i64) -> Complex64 {
        Self::contract_inner(
            poly,
            &InnerFactors::new::<S>(poly, poly.get_polyorder(), wn),
        )
    }

    /// Inner transform from precomputed segment-wise factors
    ///
    /// Sums `data[order, j] · T_nl[order, j] · phase[j]` over the orders of each
    /// segment j first (real × complex over the contiguous column j of the
    /// column-major `data`), then divides each segment sum by `norms[j]` once.
    fn contract_inner(poly: &PiecewiseLegendrePoly, f: &InnerFactors) -> Complex64 {
        let polyorder = poly.get_polyorder();
        debug_assert_eq!(f.nseg, poly.get_knots().len() - 1);
        debug_assert!(polyorder <= f.polyorder);
        let mut res = Complex64::new(0.0, 0.0);
        for j in 0..f.nseg {
            let col = &poly.get_data().col(j)[..polyorder];
            let w_re = &f.w_re[j * f.polyorder..j * f.polyorder + polyorder];
            let w_im = &f.w_im[j * f.polyorder..j * f.polyorder + polyorder];
            let (mut acc_re, mut acc_im) = (0.0, 0.0);
            for o in 0..polyorder {
                acc_re += col[o] * w_re[o];
                acc_im += col[o] * w_im[o];
            }
            res += Complex64::new(acc_re, acc_im) / poly.get_norms()[j];
        }
        res / 2.0_f64.sqrt()
    }

    /// Compute asymptotic behavior (for large frequencies)
    fn giw(&self, wn: i64) -> Complex64 {
        let iw = Complex64::new(0.0, PI / 2.0 * wn as f64);
        if wn == 0 {
            return Complex64::new(0.0, 0.0);
        }

        let inv_iw = 1.0 / iw;

        inv_iw * Self::evalpoly(inv_iw, &self.model.moments)
    }

    /// Evaluate polynomial at complex point (Horner's method)
    fn evalpoly(x: Complex64, coeffs: &[f64]) -> Complex64 {
        let mut result = Complex64::new(0.0, 0.0);
        for i in (0..coeffs.len()).rev() {
            result = result * x + Complex64::new(coeffs[i], 0.0);
        }
        result
    }

    /// Compute midpoint relative to nearest integer
    ///
    /// Returns (xmid_diff, extra_shift) where:
    /// - xmid_diff: midpoint values for numerical stability
    /// - extra_shift: nearest integer shift (-1, 0, or 1)
    fn shift_xmid(knots: &[f64], delta_x: &[f64]) -> (Vec<f64>, Vec<i32>) {
        let n_segments = delta_x.len();
        let delta_x_half: Vec<f64> = delta_x.iter().map(|&dx| dx / 2.0).collect();

        // xmid_m1: cumsum(Δx) - Δx_half
        let mut xmid_m1 = Vec::with_capacity(n_segments);
        let mut cumsum = 0.0;
        for i in 0..n_segments {
            cumsum += delta_x[i];
            xmid_m1.push(cumsum - delta_x_half[i]);
        }

        // xmid_p1: -reverse(cumsum(reverse(Δx))) + Δx_half
        let mut xmid_p1 = Vec::with_capacity(n_segments);
        let mut cumsum_rev = 0.0;
        for i in (0..n_segments).rev() {
            cumsum_rev += delta_x[i];
            xmid_p1.insert(0, -cumsum_rev + delta_x_half[i]);
        }

        // xmid_0: knots[1:] - Δx_half
        let xmid_0: Vec<f64> = (0..n_segments)
            .map(|i| knots[i + 1] - delta_x_half[i])
            .collect();

        // Determine shift and diff
        let mut xmid_diff = Vec::with_capacity(n_segments);
        let mut extra_shift = Vec::with_capacity(n_segments);

        for i in 0..n_segments {
            let shift = xmid_0[i].round() as i32;
            extra_shift.push(shift);

            // Choose appropriate xmid based on shift
            let diff = match shift {
                -1 => xmid_m1[i],
                0 => xmid_0[i],
                1 => xmid_p1[i],
                _ => xmid_0[i], // Fallback
            };
            xmid_diff.push(diff);
        }

        (xmid_diff, extra_shift)
    }

    /// Compute stable phase factors
    ///
    /// Computes: im^mod(wn * (extra_shift + 1), 4) * cispi(wn * xmid_diff / 2)
    /// where cispi(x) = exp(i*π*x)
    fn phase_stable(poly: &PiecewiseLegendrePoly, wn: i64) -> Vec<Complex64> {
        let (xmid_diff, extra_shift) = Self::shift_xmid(poly.get_knots(), poly.get_delta_x());
        let mut phase_wi = Vec::with_capacity(xmid_diff.len());

        let im_unit = Complex64::new(0.0, 1.0);

        for j in 0..xmid_diff.len() {
            // Compute im^mod(wn * (extra_shift[j] + 1), 4)
            let power = ((wn * (extra_shift[j] as i64 + 1)) % 4 + 4) % 4; // Ensure positive mod
            let im_power = im_unit.powi(power as i32);

            // Compute cispi(wn * xmid_diff[j] / 2) = exp(i*π*wn*xmid_diff/2)
            let arg = PI * wn as f64 * xmid_diff[j] / 2.0;
            let cispi = Complex64::new(arg.cos(), arg.sin());

            phase_wi.push(im_power * cispi);
        }

        phase_wi
    }

    /// Get T_nl coefficient (special function)
    ///
    /// This implements the T_nl function which is related to spherical Bessel functions:
    /// T_nl(w) = 2 * i^l * j_l(|w|) * (w < 0 ? conj : identity)
    /// where j_l is the spherical Bessel function of the first kind.
    pub fn get_tnl(l: i32, w: f64) -> Complex64 {
        let abs_w = w.abs();

        // Use the high-precision spherical Bessel function from special_functions
        let sph_bessel = spherical_bessel_j(l, abs_w);

        // Compute 2 * i^l
        let im_unit = Complex64::new(0.0, 1.0);
        let im_power = im_unit.powi(l);
        let result = 2.0 * im_power * sph_bessel;

        // Apply conjugation for negative w
        if w < 0.0 { result.conj() } else { result }
    }
}

/// Segment-wise factors of the inner Fourier transform at one frequency:
/// `w[order, j] = T_nl[order, j] · phase[j]`, stored as separate real and
/// imaginary parts, segment-major (index `j * polyorder + order`, matching the
/// column-major polynomial data). They depend only on the knots, so
/// polynomials sharing knots can share them.
struct InnerFactors {
    nseg: usize,
    polyorder: usize,
    w_re: Vec<f64>,
    w_im: Vec<f64>,
}

impl InnerFactors {
    fn new<S: StatisticsType>(poly: &PiecewiseLegendrePoly, polyorder: usize, wn: i64) -> Self {
        let wred = PI / 4.0 * wn as f64;
        let phase = PiecewiseLegendreFT::<S>::phase_stable(poly, wn);
        let nseg = phase.len();
        let mut w_re = Vec::with_capacity(polyorder * nseg);
        let mut w_im = Vec::with_capacity(polyorder * nseg);
        for (&dx, &ph) in poly.get_delta_x().iter().zip(&phase) {
            for order in 0..polyorder {
                let w = PiecewiseLegendreFT::<S>::get_tnl(order as i32, wred * dx) * ph;
                w_re.push(w.re);
                w_im.push(w.im);
            }
        }
        Self {
            nseg,
            polyorder,
            w_re,
            w_im,
        }
    }
}

/// Vector of PiecewiseLegendreFT polynomials
#[derive(Debug, Clone)]
pub struct PiecewiseLegendreFTVector<S: StatisticsType> {
    pub(crate) polyvec: Vec<PiecewiseLegendreFT<S>>,
    // `fn() -> S` keeps the type Send + Sync whatever `S` is (it holds no S)
    _phantom: std::marker::PhantomData<fn() -> S>,
}

// Type aliases for convenience
pub type FermionicPiecewiseLegendreFTVector = PiecewiseLegendreFTVector<Fermionic>;
pub type BosonicPiecewiseLegendreFTVector = PiecewiseLegendreFTVector<Bosonic>;

impl<S: StatisticsType> PiecewiseLegendreFTVector<S> {
    /// Create an empty vector
    pub fn new() -> Self {
        Self {
            polyvec: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create from a vector of PiecewiseLegendreFT
    pub fn from_vector(polyvec: Vec<PiecewiseLegendreFT<S>>) -> Self {
        Self {
            polyvec,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the number of polynomials in the vector
    pub fn len(&self) -> usize {
        self.polyvec.len()
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.polyvec.is_empty()
    }

    /// Create from PiecewiseLegendrePolyVector and statistics
    ///
    /// # Errors
    ///
    /// The errors of [`PiecewiseLegendreFT::new`] for the first polynomial
    /// that it rejects
    pub fn from_poly_vector(
        polys: &PiecewiseLegendrePolyVector,
        _stat: S,
        n_asymp: Option<f64>,
    ) -> Result<Self, Error> {
        let mut polyvec = Vec::with_capacity(polys.size());

        for poly in polys.get_polys() {
            let ft_poly = PiecewiseLegendreFT::new(poly.clone(), _stat, n_asymp)?;
            polyvec.push(ft_poly);
        }

        Ok(Self {
            polyvec,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.polyvec.len()
    }

    /// The transformed polynomials, in order
    pub fn get_polys(&self) -> &[PiecewiseLegendreFT<S>] {
        &self.polyvec
    }

    /// Get element by index (immutable)
    pub fn get(&self, index: usize) -> Option<&PiecewiseLegendreFT<S>> {
        self.polyvec.get(index)
    }

    /// Get element by index (mutable)
    pub fn get_mut(&mut self, index: usize) -> Option<&mut PiecewiseLegendreFT<S>> {
        self.polyvec.get_mut(index)
    }

    /// Set element at index
    ///
    /// # Errors
    /// [`Error::InvalidParameter`] if `index` is not less than the length;
    /// the vector is then unchanged
    pub fn set(&mut self, index: usize, poly: PiecewiseLegendreFT<S>) -> Result<(), Error> {
        let len = self.polyvec.len();
        if index >= len {
            return Err(Error::InvalidParameter {
                name: "index",
                value: index.to_string(),
                reason: format!("must be less than the length {len}"),
            });
        }
        self.polyvec[index] = poly;
        Ok(())
    }

    /// Create a similar empty vector
    pub fn similar(&self) -> Self {
        Self::new()
    }

    /// Get n_asymp from the first element (if any)
    pub fn n_asymp(&self) -> f64 {
        self.polyvec.first().map_or(f64::INFINITY, |p| p.n_asymp)
    }

    /// Evaluate all polynomials at a Matsubara frequency
    ///
    /// When all polynomials share their knots (as the functions of one basis
    /// do), the segment-wise factors of the inner transform are computed once.
    pub fn evaluate_at(&self, omega: &MatsubaraFreq<S>) -> Vec<Complex64> {
        let Some(first) = self.polyvec.first() else {
            return Vec::new();
        };
        let shared_knots = self.polyvec.iter().all(|p| {
            p.poly.get_knots() == first.poly.get_knots()
                && p.poly.get_delta_x() == first.poly.get_delta_x()
        });
        if !shared_knots {
            return self.polyvec.iter().map(|p| p.evaluate(omega)).collect();
        }
        let n = omega.get_n();
        let polyorder = self
            .polyvec
            .iter()
            .map(|p| p.poly.get_polyorder())
            .max()
            .unwrap_or(0);
        let mut factors = None;
        self.polyvec
            .iter()
            .map(|p| {
                if (n as f64).abs() < p.n_asymp {
                    let f = factors
                        .get_or_insert_with(|| InnerFactors::new::<S>(&first.poly, polyorder, n));
                    PiecewiseLegendreFT::<S>::contract_inner(&p.poly, f)
                } else {
                    p.giw(n)
                }
            })
            .collect()
    }

    /// Evaluate all polynomials at multiple Matsubara frequencies
    pub fn evaluate_at_many(&self, omegas: &[MatsubaraFreq<S>]) -> Vec<Vec<Complex64>> {
        omegas.iter().map(|omega| self.evaluate_at(omega)).collect()
    }
}

// Indexing operators
impl<S: StatisticsType> std::ops::Index<usize> for PiecewiseLegendreFTVector<S> {
    type Output = PiecewiseLegendreFT<S>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.polyvec[index]
    }
}

impl<S: StatisticsType> std::ops::IndexMut<usize> for PiecewiseLegendreFTVector<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.polyvec[index]
    }
}

// Default implementations
impl<S: StatisticsType> Default for PiecewiseLegendreFTVector<S> {
    fn default() -> Self {
        Self::new()
    }
}

// ===== Matsubara sampling point selection =====

/// Grid of reduced Matsubara indices k (n = 2k + ζ) for sampling candidates
/// Matches C++ DEFAULT_GRID: [0:2^6-1] followed by exponential spacing up to 2^25
/// Generated from Julia: [range(0; length=2^6); trunc.(Int, exp2.(range(6, 25; length=32 * (25 - 6) + 1)))]
const DEFAULT_GRID: &[i64] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 68, 69, 71, 72, 74, 76, 77,
    79, 81, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 103, 105, 107, 109, 112, 114, 117, 119, 122,
    125, 128, 130, 133, 136, 139, 142, 145, 148, 152, 155, 158, 162, 165, 169, 173, 177, 181, 184,
    189, 193, 197, 201, 206, 210, 215, 219, 224, 229, 234, 239, 245, 250, 256, 261, 267, 273, 279,
    285, 291, 297, 304, 311, 317, 324, 331, 339, 346, 354, 362, 369, 378, 386, 394, 403, 412, 421,
    430, 439, 449, 459, 469, 479, 490, 501, 512, 523, 534, 546, 558, 570, 583, 595, 608, 622, 635,
    649, 663, 678, 693, 708, 724, 739, 756, 772, 789, 806, 824, 842, 861, 879, 899, 918, 939, 959,
    980, 1002, 1024, 1046, 1069, 1092, 1116, 1141, 1166, 1191, 1217, 1244, 1271, 1299, 1327, 1357,
    1386, 1417, 1448, 1479, 1512, 1545, 1579, 1613, 1649, 1685, 1722, 1759, 1798, 1837, 1878, 1919,
    1961, 2004, 2048, 2092, 2138, 2185, 2233, 2282, 2332, 2383, 2435, 2488, 2543, 2599, 2655, 2714,
    2773, 2834, 2896, 2959, 3024, 3090, 3158, 3227, 3298, 3370, 3444, 3519, 3596, 3675, 3756, 3838,
    3922, 4008, 4096, 4185, 4277, 4371, 4466, 4564, 4664, 4766, 4870, 4977, 5086, 5198, 5311, 5428,
    5547, 5668, 5792, 5919, 6049, 6181, 6316, 6455, 6596, 6741, 6888, 7039, 7193, 7351, 7512, 7676,
    7844, 8016, 8192, 8371, 8554, 8742, 8933, 9129, 9328, 9533, 9741, 9955, 10173, 10396, 10623,
    10856, 11094, 11336, 11585, 11838, 12098, 12363, 12633, 12910, 13193, 13482, 13777, 14078,
    14387, 14702, 15024, 15353, 15689, 16032, 16384, 16742, 17109, 17484, 17866, 18258, 18657,
    19066, 19483, 19910, 20346, 20792, 21247, 21712, 22188, 22673, 23170, 23677, 24196, 24726,
    25267, 25820, 26386, 26964, 27554, 28157, 28774, 29404, 30048, 30706, 31378, 32065, 32768,
    33485, 34218, 34968, 35733, 36516, 37315, 38132, 38967, 39821, 40693, 41584, 42494, 43425,
    44376, 45347, 46340, 47355, 48392, 49452, 50535, 51641, 52772, 53928, 55108, 56315, 57548,
    58809, 60096, 61412, 62757, 64131, 65536, 66971, 68437, 69936, 71467, 73032, 74631, 76265,
    77935, 79642, 81386, 83168, 84989, 86850, 88752, 90695, 92681, 94711, 96785, 98904, 101070,
    103283, 105545, 107856, 110217, 112631, 115097, 117618, 120193, 122825, 125514, 128263, 131072,
    133942, 136875, 139872, 142935, 146064, 149263, 152531, 155871, 159284, 162772, 166337, 169979,
    173701, 177504, 181391, 185363, 189422, 193570, 197809, 202140, 206566, 211090, 215712, 220435,
    225262, 230195, 235236, 240387, 245650, 251029, 256526, 262144, 267884, 273750, 279744, 285870,
    292129, 298526, 305063, 311743, 318569, 325545, 332674, 339958, 347402, 355009, 362783, 370727,
    378845, 387141, 395618, 404281, 413133, 422180, 431424, 440871, 450525, 460390, 470472, 480774,
    491301, 502059, 513053, 524288, 535768, 547500, 559488, 571740, 584259, 597053, 610126, 623487,
    637139, 651091, 665348, 679917, 694805, 710019, 725567, 741455, 757690, 774282, 791236, 808562,
    826267, 844360, 862849, 881743, 901051, 920781, 940944, 961548, 982603, 1004119, 1026107,
    1048576, 1071536, 1095000, 1118977, 1143480, 1168519, 1194106, 1220253, 1246974, 1274279,
    1302182, 1330696, 1359834, 1389611, 1420039, 1451134, 1482910, 1515381, 1548564, 1582473,
    1617125, 1652535, 1688721, 1725699, 1763487, 1802102, 1841563, 1881888, 1923096, 1965207,
    2008239, 2052214, 2097152, 2143073, 2190000, 2237955, 2286960, 2337038, 2388212, 2440507,
    2493948, 2548558, 2604364, 2661392, 2719669, 2779222, 2840079, 2902269, 2965820, 3030763,
    3097128, 3164947, 3234250, 3305071, 3377443, 3451399, 3526975, 3604205, 3683127, 3763777,
    3846193, 3930414, 4016479, 4104428, 4194304, 4286147, 4380001, 4475911, 4573920, 4674076,
    4776425, 4881015, 4987896, 5097116, 5208729, 5322785, 5439339, 5558445, 5680159, 5804538,
    5931641, 6061527, 6194257, 6329894, 6468501, 6610142, 6754886, 6902798, 7053950, 7208411,
    7366255, 7527555, 7692387, 7860828, 8032958, 8208857, 8388608, 8572294, 8760003, 8951822,
    9147841, 9348153, 9552851, 9762031, 9975792, 10194233, 10417458, 10645571, 10878678, 11116890,
    11360318, 11609077, 11863283, 12123055, 12388515, 12659788, 12937002, 13220285, 13509772,
    13805597, 14107900, 14416823, 14732510, 15055110, 15384774, 15721657, 16065917, 16417714,
    16777216, 17144589, 17520006, 17903645, 18295683, 18696307, 19105702, 19524063, 19951584,
    20388467, 20834916, 21291142, 21757357, 22233781, 22720637, 23218155, 23726566, 24246110,
    24777031, 25319577, 25874004, 26440571, 27019544, 27611195, 28215801, 28833647, 29465021,
    30110221, 30769549, 31443315, 32131834, 32835429, 33554432,
];

/// Find sign changes of a Matsubara basis function
///
/// Returns Matsubara frequencies where the function changes sign.
///
/// # Errors
///
/// [`Error::NotSupported`] if `u_hat` has no definite parity (symm = 0);
/// such functions get their points from [`pair_sampling_points`] instead
pub(crate) fn sign_changes<S: StatisticsType + 'static>(
    u_hat: &PiecewiseLegendreFT<S>,
    positive_only: bool,
) -> Result<Vec<MatsubaraFreq<S>>, Error> {
    let f = func_for_part(u_hat)?;
    let x0 = find_all(&f, DEFAULT_GRID);

    // Convert to Matsubara indices: n = 2*x + zeta
    let mut indices: Vec<i64> = x0.iter().map(|&x| 2 * x + u_hat.zeta()).collect();

    if !positive_only {
        symmetrize_matsubara_inplace(&mut indices);
    }

    Ok(indices
        .iter()
        .filter_map(|&n| MatsubaraFreq::<S>::new(n).ok())
        .collect())
}

/// Find extrema of a Matsubara basis function
///
/// Returns Matsubara frequencies where the function has local extrema.
///
/// # Errors
///
/// [`Error::NotSupported`] if `u_hat` has no definite parity (symm = 0);
/// such functions get their points from [`pair_sampling_points`] instead
pub(crate) fn find_extrema<S: StatisticsType + 'static>(
    u_hat: &PiecewiseLegendreFT<S>,
    positive_only: bool,
) -> Result<Vec<MatsubaraFreq<S>>, Error> {
    let f = func_for_part(u_hat)?;
    let x0 = discrete_extrema(&f, DEFAULT_GRID);

    // Convert to Matsubara indices: n = 2*x + zeta
    let mut indices: Vec<i64> = x0.iter().map(|&x| 2 * x + u_hat.zeta()).collect();

    if !positive_only {
        symmetrize_matsubara_inplace(&mut indices);
    }

    Ok(indices
        .iter()
        .filter_map(|&n| MatsubaraFreq::<S>::new(n).ok())
        .collect())
}

/// The part of a Matsubara basis function whose sign changes choose the
/// default sampling points
#[derive(Clone, Copy)]
enum Part {
    Real,
    Imaginary,
}

/// The part of `û(iν)` that carries its sign changes, from the parity `symm`
/// of `u` and the statistics: for an even `u` (symm = 1) the fermionic `û`
/// is imaginary and the bosonic `û` real, for an odd `u` (symm = -1) the
/// other way round
///
/// # Errors
///
/// [`Error::NotSupported`] if `symm` is not ±1: the singular functions of an
/// SVE that is not centrosymmetric have no parity.
fn part_with_the_sign_changes<S: StatisticsType>(symm: i32) -> Result<Part, Error> {
    match (symm, S::STATISTICS) {
        (1, Statistics::Bosonic) | (-1, Statistics::Fermionic) => Ok(Part::Real),
        (1, Statistics::Fermionic) | (-1, Statistics::Bosonic) => Ok(Part::Imaginary),
        _ => Err(Error::NotSupported {
            what: format!(
                "default Matsubara sampling points, which need basis functions of definite \
                 parity (symm = ±1, from a centrosymmetric SVE); got symm = {symm}"
            ),
        }),
    }
}

impl Part {
    fn of(self, value: Complex64) -> f64 {
        match self {
            Part::Real => value.re,
            Part::Imaginary => value.im,
        }
    }
}

/// Create a function that extracts the part of `poly_ft` that carries its
/// sign changes (see [`part_with_the_sign_changes`])
fn func_for_part<S: StatisticsType + 'static>(
    poly_ft: &PiecewiseLegendreFT<S>,
) -> Result<Box<dyn Fn(i64) -> f64>, Error> {
    let part = part_with_the_sign_changes::<S>(poly_ft.poly.symm())?;
    let zeta = poly_ft.zeta();
    let poly_ft_clone = poly_ft.clone();

    Ok(Box::new(move |n: i64| {
        let omega = MatsubaraFreq::<S>::new(2 * n + zeta).unwrap();
        part.of(poly_ft_clone.evaluate(&omega))
    }))
}

/// Integer bisection: find the zero crossing of f in [a, b] where f(a) and f(b)
/// have different signs. Returns the largest integer x such that f(x) has the
/// same sign as f(a). Matches Julia's `bisect` implementation.
fn bisect(f: &dyn Fn(i64) -> f64, a: i64, b: i64, fa: f64) -> i64 {
    let mut lo = a;
    let mut hi = b;
    let mut flo = fa;

    while (hi - lo).abs() > 1 {
        let mid = lo + (hi - lo) / 2;
        let fmid = f(mid);
        if flo.signum() != fmid.signum() {
            hi = mid;
        } else {
            lo = mid;
            flo = fmid;
        }
    }
    lo
}

/// Integer bisection for extrema: find the discrete extremum of f in [a, b].
/// Returns the integer x in [a, b] that maximizes |f(x)|.
/// Matches Julia's `bisect_discr_extremum` implementation.
fn bisect_discr_extremum(f: &dyn Fn(i64) -> f64, a: i64, b: i64) -> i64 {
    let mut lo = a;
    let mut hi = b;

    while hi - lo > 2 {
        let mid1 = lo + (hi - lo) / 3;
        let mid2 = hi - (hi - lo) / 3;
        if f(mid1).abs() < f(mid2).abs() {
            lo = mid1;
        } else {
            hi = mid2;
        }
    }

    // Check all remaining candidates
    let mut best = lo;
    let mut best_val = f(lo).abs();
    for x in (lo + 1)..=hi {
        let val = f(x).abs();
        if val > best_val {
            best = x;
            best_val = val;
        }
    }
    best
}

/// Find all sign changes of a function on a grid, using bisection to locate
/// the precise integer position within each grid interval.
/// Matches Julia's `find_all` with `bisect`.
fn find_all(f: &dyn Fn(i64) -> f64, xgrid: &[i64]) -> Vec<i64> {
    if xgrid.is_empty() {
        return Vec::new();
    }

    let mut results = Vec::new();
    let mut prev_val = f(xgrid[0]);

    for i in 1..xgrid.len() {
        let val = f(xgrid[i]);
        // Detect sign change (both values must be nonzero)
        if prev_val.signum() != val.signum() && prev_val != 0.0 && val != 0.0 {
            // Bisect to find precise integer location of the zero
            let root = bisect(f, xgrid[i - 1], xgrid[i], prev_val);
            results.push(root);
        }
        prev_val = val;
    }

    results
}

/// Find discrete extrema of a function on a grid, using bisection to locate
/// the precise integer position within each grid interval.
/// Also checks boundary extrema. Matches Julia v1 / Python 1.x `discrete_extrema`.
fn discrete_extrema(f: &dyn Fn(i64) -> f64, xgrid: &[i64]) -> Vec<i64> {
    if xgrid.len() < 3 {
        return Vec::new();
    }

    let fx: Vec<f64> = xgrid.iter().map(|&x| f(x)).collect();
    let absfx: Vec<f64> = fx.iter().map(|v| v.abs()).collect();

    // Forward differences: gx[i] = fx[i+1] - fx[i]
    // signdfdx[i] = signbit(gx[i])
    // A derivative sign change at index i means the extremum is strictly
    // between xgrid[i] and xgrid[i+2].
    let signdfdx: Vec<bool> = fx
        .windows(2)
        .map(|w| (w[1] - w[0]).is_sign_negative())
        .collect();

    let mut results = Vec::new();

    for i in 0..signdfdx.len() - 1 {
        if signdfdx[i] != signdfdx[i + 1] {
            // Extremum between xgrid[i] and xgrid[i+2]
            let refined = bisect_discr_extremum(f, xgrid[i], xgrid[i + 2]);
            results.push(refined);
        }
    }

    // Boundary: first point is extremum if |f| decreases or sign changes inwards
    let sfx: Vec<bool> = fx.iter().map(|v| v.is_sign_negative()).collect();
    if absfx[0] > absfx[1] || sfx[0] != sfx[1] {
        results.insert(0, xgrid[0]);
    }

    // Boundary: last point is extremum if |f| decreases or sign changes inwards
    let n = fx.len();
    if absfx[n - 1] > absfx[n - 2] || sfx[n - 1] != sfx[n - 2] {
        results.push(xgrid[n - 1]);
    }

    results
}

/// Symmetrize Matsubara indices by adding negative frequencies
fn symmetrize_matsubara_inplace(xs: &mut Vec<i64>) {
    if xs.is_empty() {
        return;
    }

    // Remove zero if present
    xs.retain(|&x| x != 0);

    // Add negative frequencies
    let positives: Vec<i64> = xs.iter().filter(|&&x| x > 0).copied().collect();
    let mut negatives: Vec<i64> = positives.iter().map(|&x| -x).collect();

    xs.append(&mut negatives);
    xs.sort();
    xs.dedup();
}

/// Matsubara sampling frequencies for fitting the first `n_points` functions
/// of `uhat`, selected as symmetric pairs; the default points of functions
/// without a definite parity
///
/// The sign changes of `û_L` that choose the default points of functions with
/// a parity (as in SparseIR.jl) need `û_L` to be purely real or purely
/// imaginary. This selection needs no parity.
///
/// The selected set is symmetric, `{±n_k}`. Since `û_l(-n) = conj(û_l(n))`
/// for a real `u_l(τ)`, fitting at `±n_k` is equivalent to fitting the real
/// and imaginary parts at `n_k`, so the selection is made on the real matrix
/// whose rows are `Re û_l(n_k)` and `Im û_l(n_k)`: two rows per frequency
/// `n_k > 0`, and for bosons one row, `Re û_l(0)`, for `n = 0`, which is
/// always selected. The rows are chosen by an interpolative decomposition
/// (the pivoted QR of the rows, taking pairs; see the `pair_select` module),
/// as Kaye, Chen and Parcollet do for the DLR (Phys. Rev. B 105, 235115
/// (2022)), until there are at least `n_points` rows: an even number for
/// fermions, an odd one for bosons, as for the sign changes. When `uhat` has
/// the next function, the selection is made on `n_points` rounded up to that
/// parity, so that it is square.
///
/// Candidates are the frequencies `n = 2k + ζ >= 0` for `k` on the grid of
/// the sign-change search (as in SparseIR.jl): every `k` below 64, then
/// about 32 points per octave (spacing about 2%). The selected frequencies
/// are points of this grid, unlike the sign changes, which are bisected to
/// the exact integer; at high frequency, where the grid is sparse, the
/// functions vary slowly in `n`.
///
/// With `mitigate`, about 10% more frequencies are added (oversampling), in
/// ± pairs, each pair the one that most decreases the noise variance
/// `trace((FᵀF)⁻¹)` of the fit (A-optimal).
///
/// With `positive_only`, the frequencies `n_k >= 0` are returned (as used
/// for fitting real-valued imaginary-time functions); otherwise `±n_k`.
/// Returned sorted; empty if `n_points` is 0.
///
/// # Errors
///
/// * [`Error::EmptyInput`] if `uhat` is empty
/// * [`Error::InvalidParameter`] if `n_points` exceeds the number of
///   functions of `uhat`
/// * [`Error::DecompositionFailed`] if the rows of the functions on the grid
///   have rank below `n_points`, so that fewer points than requested would
///   be returned
pub(crate) fn pair_sampling_points<S: StatisticsType + 'static>(
    uhat: &PiecewiseLegendreFTVector<S>,
    n_points: usize,
    positive_only: bool,
    mitigate: bool,
) -> Result<Vec<MatsubaraFreq<S>>, Error> {
    use crate::pair_select::select_pairs;
    use nalgebra::DMatrix;

    if uhat.is_empty() {
        return Err(Error::EmptyInput { name: "uhat" });
    }
    if n_points > uhat.len() {
        return Err(Error::InvalidParameter {
            name: "n_points",
            value: n_points.to_string(),
            reason: format!(
                "the pair selection fits at most the {} functions given",
                uhat.len()
            ),
        });
    }
    if n_points == 0 {
        return Ok(Vec::new());
    }
    let bosonic = S::STATISTICS == Statistics::Bosonic;
    // Rows of the selection: n_points rounded up to the parity of the
    // statistics; as many functions are fitted when uhat has them.
    let n_rows = if bosonic {
        n_points + 1 - n_points % 2
    } else {
        n_points + n_points % 2
    };
    let n_fit = if n_rows <= uhat.len() {
        n_rows
    } else {
        n_points
    };
    let zeta = i64::from(!bosonic);
    // The candidates 2k + ζ have the parity of the statistics.
    let freqs: Vec<MatsubaraFreq<S>> = DEFAULT_GRID
        .iter()
        .map(|&k| MatsubaraFreq::new(2 * k + zeta))
        .collect::<Result<_, _>>()?;
    // Only the first n_fit functions enter the selection: evaluate just
    // those. û_l(-n) = conj(û_l(n)) as u_l(τ) is real: only n >= 0 is
    // evaluated.
    let used = PiecewiseLegendreFTVector::from_vector(uhat.polyvec[..n_fit].to_vec());
    let values: Vec<Vec<Complex64>> = used.evaluate_at_many(&freqs);
    // n = 0 (bosons only) is the first candidate; it gives one fixed row,
    // Re û(0)
    let first = usize::from(bosonic);
    let pairs = DMatrix::<f64>::from_fn(2 * (freqs.len() - first), n_fit, |i, l| {
        let v = values[first + i / 2][l];
        if i % 2 == 0 { v.re } else { v.im }
    });
    let fixed = DMatrix::<f64>::from_fn(first, n_fit, |_, l| values[0][l].re);
    // Oversampling adds 10% more frequencies, in ± pairs
    let n_extra = if mitigate {
        n_points.div_ceil(10).div_ceil(2)
    } else {
        0
    };
    let selected_pairs = select_pairs(&pairs, &fixed, n_extra);
    if first + 2 * selected_pairs.len() < n_points {
        return Err(Error::DecompositionFailed {
            reason: format!(
                "the functions have rank {} below {n_points} on the candidate frequencies",
                first + 2 * selected_pairs.len()
            ),
        });
    }
    let mut selected: Vec<MatsubaraFreq<S>> = selected_pairs
        .into_iter()
        .map(|p| freqs[first + p])
        .collect();
    if bosonic {
        selected.push(freqs[0]);
    }
    if !positive_only {
        let negative: Vec<MatsubaraFreq<S>> = selected
            .iter()
            .filter(|f| f.get_n() != 0)
            .map(|&f| -f)
            .collect();
        selected.extend(negative);
    }
    selected.sort();
    Ok(selected)
}

#[cfg(test)]
#[path = "polyfourier_tests.rs"]
mod polyfourier_tests;
