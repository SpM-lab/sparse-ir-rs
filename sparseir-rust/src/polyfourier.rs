//! Piecewise Legendre polynomial Fourier transform implementation for SparseIR
//!
//! This module provides Fourier transform functionality for piecewise Legendre
//! polynomials, enabling evaluation in Matsubara frequency domain.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::freq::MatsubaraFreq;
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use crate::special_functions::spherical_bessel_j;
use crate::traits::{Bosonic, Fermionic, Statistics, StatisticsType};

/// Power model for asymptotic behavior
#[derive(Debug, Clone)]
pub struct PowerModel {
    pub moments: Vec<f64>,
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
    pub poly: PiecewiseLegendrePoly,
    /// Asymptotic cutoff frequency index
    pub n_asymp: f64,
    /// Power model for asymptotic behavior
    pub model: PowerModel,
    _phantom: std::marker::PhantomData<S>,
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
    /// # Panics
    /// Panics if the polynomial domain is not [-1, 1]
    pub fn new(poly: PiecewiseLegendrePoly, _stat: S, n_asymp: Option<f64>) -> Self {
        // Validate domain
        if (poly.xmin - (-1.0)).abs() > 1e-12 || (poly.xmax - 1.0).abs() > 1e-12 {
            panic!("Only interval [-1, 1] is supported for Fourier transform");
        }

        let n_asymp = n_asymp.unwrap_or(f64::INFINITY);
        let model = Self::power_model(&poly);

        Self {
            poly,
            n_asymp,
            model,
            _phantom: std::marker::PhantomData,
        }
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
        let n = omega.get_n() as i32;
        if (n as f64).abs() < self.n_asymp {
            self.compute_unl_inner(&self.poly, n)
        } else {
            self.giw(n)
        }
    }

    /// Evaluate at integer Matsubara index
    pub fn evaluate_at_n(&self, n: i64) -> Complex64 {
        match MatsubaraFreq::<S>::new(n) {
            Ok(omega) => self.evaluate(&omega),
            Err(_) => Complex64::new(0.0, 0.0), // Return zero for invalid frequencies
        }
    }

    /// Evaluate at multiple Matsubara indices
    pub fn evaluate_at_ns(&self, ns: &[i64]) -> Vec<Complex64> {
        ns.iter().map(|&n| self.evaluate_at_n(n)).collect()
    }

    /// Create power model for asymptotic behavior
    fn power_model(poly: &PiecewiseLegendrePoly) -> PowerModel {
        let deriv_x1 = poly.derivs(1.0);
        let moments = Self::power_moments(&deriv_x1, poly.l);
        PowerModel::new(moments)
    }

    /// Compute power moments for asymptotic expansion
    fn power_moments(deriv_x1: &[f64], l: i32) -> Vec<f64> {
        let statsign = match S::STATISTICS {
            Statistics::Fermionic => -1.0,
            Statistics::Bosonic => 1.0,
        };

        let mut moments = deriv_x1.to_vec();
        for (m, moment) in moments.iter_mut().enumerate() {
            let m_f64 = (m + 1) as f64; // Julia uses 1-based indexing
            *moment *=
                -(statsign * (-1.0_f64).powi(m_f64 as i32) + (-1.0_f64).powi(l)) / 2.0_f64.sqrt();
        }
        moments
    }

    /// Compute the inner Fourier transform (for small frequencies)
    fn compute_unl_inner(&self, poly: &PiecewiseLegendrePoly, wn: i32) -> Complex64 {
        let wred = PI / 4.0 * wn as f64;
        let phase_wi = Self::phase_stable(poly, wn);
        let mut res = Complex64::new(0.0, 0.0);

        let order_max = poly.polyorder;
        let segment_count = poly.knots.len() - 1;

        for order in 0..order_max {
            for j in 0..segment_count {
                let data_oj = poly.data[[order, j]];
                let tnl = Self::get_tnl(order as i32, wred * poly.delta_x[j]);
                res += data_oj * tnl * phase_wi[j] / poly.norms[j];
            }
        }

        res / 2.0_f64.sqrt()
    }

    /// Compute asymptotic behavior (for large frequencies)
    fn giw(&self, wn: i32) -> Complex64 {
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
    fn phase_stable(poly: &PiecewiseLegendrePoly, wn: i32) -> Vec<Complex64> {
        let (xmid_diff, extra_shift) = Self::shift_xmid(&poly.knots, &poly.delta_x);
        let mut phase_wi = Vec::with_capacity(xmid_diff.len());

        let im_unit = Complex64::new(0.0, 1.0);

        for j in 0..xmid_diff.len() {
            // Compute im^mod(wn * (extra_shift[j] + 1), 4)
            let power = ((wn * (extra_shift[j] + 1)) % 4 + 4) % 4; // Ensure positive mod
            let im_power = im_unit.powi(power);

            // Compute cispi(wn * xmid_diff[j] / 2) = exp(i*π*wn*xmid_diff/2)
            let arg = PI * wn as f64 * xmid_diff[j] / 2.0;
            let cispi = Complex64::new(arg.cos(), arg.sin());

            phase_wi.push(im_power * cispi);
        }

        phase_wi
    }

    /// Find sign changes in the Fourier transform
    ///
    /// # Arguments
    /// * `positive_only` - If true, only return positive frequency sign changes
    ///
    /// # Returns
    /// Vector of Matsubara frequencies where sign changes occur
    pub fn sign_changes(&self, positive_only: bool) -> Vec<MatsubaraFreq<S>> {
        let f = Self::func_for_part(self);
        let x0 = Self::find_all_roots(&f, DEFAULT_GRID);

        // Transform grid indices to Matsubara frequencies
        let mut matsubara_indices: Vec<i64> = x0.into_iter().map(|x| 2 * x + self.zeta()).collect();

        if !positive_only {
            Self::symmetrize_matsubara_inplace(&mut matsubara_indices);
        }

        matsubara_indices
            .into_iter()
            .filter_map(|n| MatsubaraFreq::<S>::new(n).ok())
            .collect()
    }

    /// Find extrema in the Fourier transform
    ///
    /// # Arguments
    /// * `positive_only` - If true, only return positive frequency extrema
    ///
    /// # Returns
    /// Vector of Matsubara frequencies where extrema occur
    pub fn find_extrema(&self, positive_only: bool) -> Vec<MatsubaraFreq<S>> {
        let f = Self::func_for_part(self);
        let x0 = Self::discrete_extrema(&f, DEFAULT_GRID);

        // Transform grid indices to Matsubara frequencies
        let mut matsubara_indices: Vec<i64> = x0.into_iter().map(|x| 2 * x + self.zeta()).collect();

        if !positive_only {
            Self::symmetrize_matsubara_inplace(&mut matsubara_indices);
        }

        matsubara_indices
            .into_iter()
            .filter_map(|n| MatsubaraFreq::<S>::new(n).ok())
            .collect()
    }

    /// Create function for extracting real or imaginary part based on parity
    fn func_for_part(&self) -> impl Fn(i64) -> f64 + '_ {
        let parity = self.poly.symm;
        let poly_ft = self.clone();

        move |n| {
            let omega = match MatsubaraFreq::<S>::new(n) {
                Ok(omega) => omega,
                Err(_) => return 0.0,
            };
            let value = poly_ft.evaluate(&omega);

            let result = match parity {
                1 => match S::STATISTICS {
                    Statistics::Fermionic => value.im,
                    Statistics::Bosonic => value.re,
                },
                -1 => match S::STATISTICS {
                    Statistics::Fermionic => value.re,
                    Statistics::Bosonic => value.im,
                },
                0 => {
                    // For symm = 0, use real part for both statistics
                    value.re
                }
                _ => panic!("Cannot detect parity for symm = {}", parity),
            };

            // Debug: print values for constant polynomial
            if n == 0 || n == 1 || n == 2 {
                println!("n={}, value={}, result={}", n, value, result);
            }

            result
        }
    }

    /// Find all roots using the same algorithm as the poly module
    fn find_all_roots<F>(f: &F, xgrid: &[i64]) -> Vec<i64>
    where
        F: Fn(i64) -> f64,
    {
        if xgrid.is_empty() {
            return Vec::new();
        }

        // Evaluate function at all grid points
        let fx: Vec<f64> = xgrid.iter().map(|&x| f(x)).collect();

        // Find exact zeros (direct hits)
        let mut x_hit = Vec::new();
        for i in 0..fx.len() {
            if fx[i] == 0.0 {
                x_hit.push(xgrid[i]);
            }
        }

        // Find sign changes
        let mut sign_change = Vec::new();
        for i in 0..fx.len() - 1 {
            let has_sign_change = fx[i].signum() != fx[i + 1].signum();
            let not_hit = fx[i] != 0.0 && fx[i + 1] != 0.0;
            let both_nonzero = fx[i].abs() > 1e-12 && fx[i + 1].abs() > 1e-12;
            sign_change.push(has_sign_change && not_hit && both_nonzero);
        }

        // If no sign changes, return only direct hits
        if sign_change.iter().all(|&sc| !sc) {
            x_hit.sort();
            return x_hit;
        }

        // Find intervals with sign changes
        let mut a_intervals = Vec::new();
        let mut b_intervals = Vec::new();
        let mut fa_values = Vec::new();

        for i in 0..sign_change.len() {
            if sign_change[i] {
                a_intervals.push(xgrid[i]);
                b_intervals.push(xgrid[i + 1]);
                fa_values.push(fx[i]);
            }
        }

        // Use bisection for each interval with sign change
        for i in 0..a_intervals.len() {
            let root = Self::bisect(&f, a_intervals[i], b_intervals[i], fa_values[i]);
            x_hit.push(root);
        }

        // Sort and return
        x_hit.sort();
        x_hit
    }

    /// Bisection method for integer grid
    fn bisect<F>(f: &F, a: i64, b: i64, fa: f64) -> i64
    where
        F: Fn(i64) -> f64,
    {
        let mut a = a;
        let mut b = b;
        let mut fa = fa;

        loop {
            if (b - a).abs() <= 1 {
                return a;
            }

            let mid = (a + b) / 2;
            let fmid = f(mid);

            if fa.signum() != fmid.signum() {
                b = mid;
            } else {
                a = mid;
                fa = fmid;
            }
        }
    }

    /// Find discrete extrema
    fn discrete_extrema<F>(f: &F, xgrid: &[i64]) -> Vec<i64>
    where
        F: Fn(i64) -> f64,
    {
        if xgrid.len() < 3 {
            return Vec::new();
        }

        let fx: Vec<f64> = xgrid.iter().map(|&x| f(x)).collect();
        let mut extrema = Vec::new();

        // Check for extrema (local maxima or minima)
        for i in 1..fx.len() - 1 {
            let prev = fx[i - 1];
            let curr = fx[i];
            let next = fx[i + 1];

            // Local maximum
            if curr > prev && curr > next {
                extrema.push(xgrid[i]);
            }
            // Local minimum
            else if curr < prev && curr < next {
                extrema.push(xgrid[i]);
            }
        }

        extrema
    }

    /// Symmetrize Matsubara indices (remove zero if present and add negatives)
    fn symmetrize_matsubara_inplace(xs: &mut Vec<i64>) {
        // Remove zero if present
        xs.retain(|&x| x != 0);

        // Sort in ascending order
        xs.sort();

        // Create negative counterparts
        let negatives: Vec<i64> = xs.iter().rev().map(|&x| -x).collect();

        // Combine negatives with originals
        xs.splice(0..0, negatives);
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

/// Vector of PiecewiseLegendreFT polynomials
#[derive(Debug, Clone)]
pub struct PiecewiseLegendreFTVector<S: StatisticsType> {
    pub polyvec: Vec<PiecewiseLegendreFT<S>>,
    _phantom: std::marker::PhantomData<S>,
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
    pub fn from_poly_vector(
        polys: &PiecewiseLegendrePolyVector,
        _stat: S,
        n_asymp: Option<f64>,
    ) -> Self {
        let mut polyvec = Vec::with_capacity(polys.size());

        for i in 0..polys.size() {
            let poly = polys.get(i).unwrap().clone();
            let ft_poly = PiecewiseLegendreFT::new(poly, _stat, n_asymp);
            polyvec.push(ft_poly);
        }

        Self {
            polyvec,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the size of the vector
    pub fn size(&self) -> usize {
        self.polyvec.len()
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
    pub fn set(&mut self, index: usize, poly: PiecewiseLegendreFT<S>) -> Result<(), String> {
        if index >= self.polyvec.len() {
            return Err(format!("Index {} out of range", index));
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
    pub fn evaluate_at(&self, omega: &MatsubaraFreq<S>) -> Vec<Complex64> {
        self.polyvec
            .iter()
            .map(|poly| poly.evaluate(omega))
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

/// Default grid for finding extrema/sign changes
/// Matches C++ DEFAULT_GRID: [0:2^6-1] followed by exponential spacing up to 2^25
/// Generated from Julia: [range(0; length=2^6); trunc.(Int, exp2.(range(6, 25; length=32 * (25 - 6) + 1)))]
const DEFAULT_GRID: &[i64] = &[
    0,        1,        2,        3,        4,        5,        6,
    7,        8,        9,        10,       11,       12,       13,
    14,       15,       16,       17,       18,       19,       20,
    21,       22,       23,       24,       25,       26,       27,
    28,       29,       30,       31,       32,       33,       34,
    35,       36,       37,       38,       39,       40,       41,
    42,       43,       44,       45,       46,       47,       48,
    49,       50,       51,       52,       53,       54,       55,
    56,       57,       58,       59,       60,       61,       62,
    63,       64,       65,       66,       68,       69,       71,
    72,       74,       76,       77,       79,       81,       82,
    84,       86,       88,       90,       92,       94,       96,
    98,       100,      103,      105,      107,      109,      112,
    114,      117,      119,      122,      125,      128,      130,
    133,      136,      139,      142,      145,      148,      152,
    155,      158,      162,      165,      169,      173,      177,
    181,      184,      189,      193,      197,      201,      206,
    210,      215,      219,      224,      229,      234,      239,
    245,      250,      256,      261,      267,      273,      279,
    285,      291,      297,      304,      311,      317,      324,
    331,      339,      346,      354,      362,      369,      378,
    386,      394,      403,      412,      421,      430,      439,
    449,      459,      469,      479,      490,      501,      512,
    523,      534,      546,      558,      570,      583,      595,
    608,      622,      635,      649,      663,      678,      693,
    708,      724,      739,      756,      772,      789,      806,
    824,      842,      861,      879,      899,      918,      939,
    959,      980,      1002,     1024,     1046,     1069,     1092,
    1116,     1141,     1166,     1191,     1217,     1244,     1271,
    1299,     1327,     1357,     1386,     1417,     1448,     1479,
    1512,     1545,     1579,     1613,     1649,     1685,     1722,
    1759,     1798,     1837,     1878,     1919,     1961,     2004,
    2048,     2092,     2138,     2185,     2233,     2282,     2332,
    2383,     2435,     2488,     2543,     2599,     2655,     2714,
    2773,     2834,     2896,     2959,     3024,     3090,     3158,
    3227,     3298,     3370,     3444,     3519,     3596,     3675,
    3756,     3838,     3922,     4008,     4096,     4185,     4277,
    4371,     4466,     4564,     4664,     4766,     4870,     4977,
    5086,     5198,     5311,     5428,     5547,     5668,     5792,
    5919,     6049,     6181,     6316,     6455,     6596,     6741,
    6888,     7039,     7193,     7351,     7512,     7676,     7844,
    8016,     8192,     8371,     8554,     8742,     8933,     9129,
    9328,     9533,     9741,     9955,     10173,    10396,    10623,
    10856,    11094,    11336,    11585,    11838,    12098,    12363,
    12633,    12910,    13193,    13482,    13777,    14078,    14387,
    14702,    15024,    15353,    15689,    16032,    16384,    16742,
    17109,    17484,    17866,    18258,    18657,    19066,    19483,
    19910,    20346,    20792,    21247,    21712,    22188,    22673,
    23170,    23677,    24196,    24726,    25267,    25820,    26386,
    26964,    27554,    28157,    28774,    29404,    30048,    30706,
    31378,    32065,    32768,    33485,    34218,    34968,    35733,
    36516,    37315,    38132,    38967,    39821,    40693,    41584,
    42494,    43425,    44376,    45347,    46340,    47355,    48392,
    49452,    50535,    51641,    52772,    53928,    55108,    56315,
    57548,    58809,    60096,    61412,    62757,    64131,    65536,
    66971,    68437,    69936,    71467,    73032,    74631,    76265,
    77935,    79642,    81386,    83168,    84989,    86850,    88752,
    90695,    92681,    94711,    96785,    98904,    101070,   103283,
    105545,   107856,   110217,   112631,   115097,   117618,   120193,
    122825,   125514,   128263,   131072,   133942,   136875,   139872,
    142935,   146064,   149263,   152531,   155871,   159284,   162772,
    166337,   169979,   173701,   177504,   181391,   185363,   189422,
    193570,   197809,   202140,   206566,   211090,   215712,   220435,
    225262,   230195,   235236,   240387,   245650,   251029,   256526,
    262144,   267884,   273750,   279744,   285870,   292129,   298526,
    305063,   311743,   318569,   325545,   332674,   339958,   347402,
    355009,   362783,   370727,   378845,   387141,   395618,   404281,
    413133,   422180,   431424,   440871,   450525,   460390,   470472,
    480774,   491301,   502059,   513053,   524288,   535768,   547500,
    559488,   571740,   584259,   597053,   610126,   623487,   637139,
    651091,   665348,   679917,   694805,   710019,   725567,   741455,
    757690,   774282,   791236,   808562,   826267,   844360,   862849,
    881743,   901051,   920781,   940944,   961548,   982603,   1004119,
    1026107,  1048576,  1071536,  1095000,  1118977,  1143480,  1168519,
    1194106,  1220253,  1246974,  1274279,  1302182,  1330696,  1359834,
    1389611,  1420039,  1451134,  1482910,  1515381,  1548564,  1582473,
    1617125,  1652535,  1688721,  1725699,  1763487,  1802102,  1841563,
    1881888,  1923096,  1965207,  2008239,  2052214,  2097152,  2143073,
    2190000,  2237955,  2286960,  2337038,  2388212,  2440507,  2493948,
    2548558,  2604364,  2661392,  2719669,  2779222,  2840079,  2902269,
    2965820,  3030763,  3097128,  3164947,  3234250,  3305071,  3377443,
    3451399,  3526975,  3604205,  3683127,  3763777,  3846193,  3930414,
    4016479,  4104428,  4194304,  4286147,  4380001,  4475911,  4573920,
    4674076,  4776425,  4881015,  4987896,  5097116,  5208729,  5322785,
    5439339,  5558445,  5680159,  5804538,  5931641,  6061527,  6194257,
    6329894,  6468501,  6610142,  6754886,  6902798,  7053950,  7208411,
    7366255,  7527555,  7692387,  7860828,  8032958,  8208857,  8388608,
    8572294,  8760003,  8951822,  9147841,  9348153,  9552851,  9762031,
    9975792,  10194233, 10417458, 10645571, 10878678, 11116890, 11360318,
    11609077, 11863283, 12123055, 12388515, 12659788, 12937002, 13220285,
    13509772, 13805597, 14107900, 14416823, 14732510, 15055110, 15384774,
    15721657, 16065917, 16417714, 16777216, 17144589, 17520006, 17903645,
    18295683, 18696307, 19105702, 19524063, 19951584, 20388467, 20834916,
    21291142, 21757357, 22233781, 22720637, 23218155, 23726566, 24246110,
    24777031, 25319577, 25874004, 26440571, 27019544, 27611195, 28215801,
    28833647, 29465021, 30110221, 30769549, 31443315, 32131834, 32835429,
    33554432,
];

/// Find sign changes of a Matsubara basis function
///
/// Returns Matsubara frequencies where the function changes sign.
pub fn sign_changes<S: StatisticsType + 'static>(
    u_hat: &PiecewiseLegendreFT<S>,
    positive_only: bool,
) -> Vec<MatsubaraFreq<S>> {
    let f = func_for_part(u_hat);
    let x0 = find_all(&f, DEFAULT_GRID);

    // Convert to Matsubara indices: n = 2*x + zeta
    let mut indices: Vec<i64> = x0.iter().map(|&x| 2 * x + u_hat.zeta()).collect();

    if !positive_only {
        symmetrize_matsubara_inplace(&mut indices);
    }

    indices
        .iter()
        .filter_map(|&n| MatsubaraFreq::<S>::new(n).ok())
        .collect()
}

/// Find extrema of a Matsubara basis function
///
/// Returns Matsubara frequencies where the function has local extrema.
pub fn find_extrema<S: StatisticsType + 'static>(
    u_hat: &PiecewiseLegendreFT<S>,
    positive_only: bool,
) -> Vec<MatsubaraFreq<S>> {
    let f = func_for_part(u_hat);
    let x0 = discrete_extrema(&f, DEFAULT_GRID);

    // Convert to Matsubara indices: n = 2*x + zeta
    let mut indices: Vec<i64> = x0.iter().map(|&x| 2 * x + u_hat.zeta()).collect();

    if !positive_only {
        symmetrize_matsubara_inplace(&mut indices);
    }

    indices
        .iter()
        .filter_map(|&n| MatsubaraFreq::<S>::new(n).ok())
        .collect()
}

/// Create a function that extracts the appropriate part (real/imag) based on parity
fn func_for_part<S: StatisticsType + 'static>(
    poly_ft: &PiecewiseLegendreFT<S>,
) -> Box<dyn Fn(i64) -> f64> {
    let parity = poly_ft.poly.symm();
    let zeta = poly_ft.zeta();

    // Clone what we need
    let poly_ft_clone = poly_ft.clone();

    Box::new(move |n: i64| {
        let omega = MatsubaraFreq::<S>::new(2 * n + zeta).unwrap();
        let value = poly_ft_clone.evaluate(&omega);

        // Select real or imaginary part based on parity and statistics
        if parity == 1 {
            // Even parity
            if S::STATISTICS == Statistics::Bosonic {
                value.re
            } else {
                value.im
            }
        } else if parity == -1 {
            // Odd parity
            if S::STATISTICS == Statistics::Bosonic {
                value.im
            } else {
                value.re
            }
        } else {
            panic!("Cannot detect parity");
        }
    })
}

/// Find all sign changes of a function on a grid
fn find_all(f: &dyn Fn(i64) -> f64, xgrid: &[i64]) -> Vec<i64> {
    if xgrid.is_empty() {
        return Vec::new();
    }

    let mut results = Vec::new();
    let mut prev_val = f(xgrid[0]);

    for i in 1..xgrid.len() {
        let val = f(xgrid[i]);
        if prev_val.signum() != val.signum() && val != 0.0 {
            results.push(xgrid[i - 1]);
        }
        prev_val = val;
    }

    results
}

/// Find discrete extrema of a function on a grid
fn discrete_extrema(f: &dyn Fn(i64) -> f64, xgrid: &[i64]) -> Vec<i64> {
    let mut results = Vec::new();

    if xgrid.len() < 3 {
        return results;
    }

    let mut prev_val = f(xgrid[0]);
    let mut curr_val = f(xgrid[1]);

    for i in 2..xgrid.len() {
        let next_val = f(xgrid[i]);

        // Check if curr_val is a local extremum
        if (curr_val > prev_val && curr_val > next_val)
            || (curr_val < prev_val && curr_val < next_val)
        {
            results.push(xgrid[i - 1]);
        }

        prev_val = curr_val;
        curr_val = next_val;
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

#[cfg(test)]
#[path = "polyfourier_tests.rs"]
mod polyfourier_tests;

