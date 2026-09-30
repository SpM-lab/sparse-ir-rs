//! Matrix ESPRIT (port of `mini_pole/esprit.py`).
//!
//! Approximates `h_k ∈ C^d`, sampled at `N` uniformly spaced points
//! `x_k` of `[x_min, x_max]`, by `Σ_j ω_j γ_j^k` with nodes `γ_j` shared by
//! the `d` columns.
//!
//! Ported from Green-Phys/MiniPole (commit 15e4a54, MIT License,
//! Copyright (c) 2024 lzphy); see `LICENSE-THIRD-PARTY`.

use crate::error::{Error, Result};
use crate::linalg::{cpow, eigvals, lstsq, svd_s_vh};
use num_complex::Complex;

type C64 = Complex<f64>;

/// Interpretation of an error tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrType {
    /// Absolute error.
    #[default]
    Abs,
    /// Error relative to the largest singular value.
    Rel,
}

/// Parameters of [`Esprit::new`], with the defaults of the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct EspritParams {
    /// Lower end of the sampling interval (default 0).
    pub x_min: f64,
    /// Upper end of the sampling interval (default 1).
    pub x_max: f64,
    /// Error tolerance that selects the number of nodes `M`. One of `err`
    /// and `m` is required: the reference's knee detection for neither is
    /// not ported.
    pub err: Option<f64>,
    /// Interpretation of `err` (default absolute).
    pub err_type: ErrType,
    /// Number of nodes `M`; overrides `err`.
    pub m: Option<usize>,
    /// Ratio `L / (N - 1)` of the Hankel matrix (default 0.4).
    pub lfactor: f64,
    /// Threshold below which the imaginary (real) part of the input counts
    /// as zero (default 1e-15).
    pub tol: f64,
    /// The approximation is accepted when its maximum error is below
    /// `ctrl_ratio` times the first discarded singular value (default 10).
    pub ctrl_ratio: f64,
}

impl Default for EspritParams {
    fn default() -> Self {
        Self {
            x_min: 0.0,
            x_max: 1.0,
            err: None,
            err_type: ErrType::Abs,
            m: None,
            lfactor: 0.4,
            tol: 1e-15,
            ctrl_ratio: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DataType {
    Real,
    Imag,
    Cplx,
}

/// Result of the ESPRIT approximation.
#[derive(Debug, Clone)]
pub struct Esprit {
    /// Number of samples `N`.
    pub n: usize,
    /// Number of columns `d`.
    pub dim: usize,
    /// Hankel parameter `L`.
    pub l: usize,
    /// Sampling interval.
    pub x_min: f64,
    /// Sampling interval.
    pub x_max: f64,
    /// Singular values of the Hankel matrix (empty for zero input).
    pub s: Vec<f64>,
    /// Number of nodes `M`.
    pub m: usize,
    /// `S[M]`, the first discarded singular value (0 for zero input).
    pub sigma: f64,
    /// Nodes `γ_j`.
    pub gamma: Vec<C64>,
    /// Weights `ω_j`, column-major `M x d`.
    pub omega: Vec<C64>,
    /// Maximum error of the approximation on the samples.
    pub err_max: f64,
    /// Largest mean (over samples) error among the columns.
    pub err_ave: f64,
    data_type: DataType,
}

impl Esprit {
    /// ESPRIT of `h` (column-major `n x dim`).
    ///
    /// # Errors
    /// [`Error::InvalidParameter`] for invalid parameters, and
    /// [`Error::DecompositionFailed`] if no controlled approximation exists.
    pub fn new(h: &[C64], n: usize, dim: usize, p: &EspritParams) -> Result<Self> {
        if h.len() != n * dim || dim == 0 {
            return Err(Error::InvalidParameter {
                name: "h_k",
                value: format!("of length {} for N = {n}, d = {dim}", h.len()),
                reason: "must have N d entries and d >= 1".to_string(),
            });
        }
        let mut l = (p.lfactor * (n as f64 - 1.0)) as usize;
        if n < 2 || n - l < l + 1 {
            return Err(Error::InvalidParameter {
                name: "lfactor",
                value: format!("{} for N = {n}", p.lfactor),
                reason: "must satisfy N - L >= L + 1 with L = int(lfactor (N - 1))".to_string(),
            });
        }
        if p.x_min.is_nan() || p.x_max.is_nan() || p.x_min >= p.x_max {
            return Err(Error::InvalidParameter {
                name: "x_min, x_max",
                value: format!("[{:?}, {:?}]", p.x_min, p.x_max),
                reason: "must satisfy x_min < x_max".to_string(),
            });
        }
        let max_im = h.iter().map(|v| v.im.abs()).fold(0.0, f64::max);
        let max_re = h.iter().map(|v| v.re.abs()).fold(0.0, f64::max);
        let (data_type, h): (DataType, Vec<C64>) = if max_im < p.tol {
            (
                DataType::Real,
                h.iter().map(|v| C64::new(v.re, 0.0)).collect(),
            )
        } else if max_re < p.tol {
            (
                DataType::Imag,
                h.iter().map(|v| C64::new(0.0, v.im)).collect(),
            )
        } else {
            (DataType::Cplx, h.to_vec())
        };
        let mut out = Self {
            n,
            dim,
            l,
            x_min: p.x_min,
            x_max: p.x_max,
            s: Vec::new(),
            m: 0,
            sigma: 0.0,
            gamma: Vec::new(),
            omega: Vec::new(),
            err_max: 0.0,
            err_ave: 0.0,
            data_type,
        };
        if h.iter().all(|v| *v == C64::new(0.0, 0.0)) {
            return Ok(out);
        }

        // H[(d l + c), j] = h[l + j, c], of size d (N - L) x (L + 1); shrink L
        // if the SVD fails.
        let (s, vh) = loop {
            let rows = dim * (n - l);
            let cols = l + 1;
            let mut hm = vec![C64::new(0.0, 0.0); rows * cols];
            for ll in 0..(n - l) {
                for c in 0..dim {
                    for j in 0..cols {
                        hm[(dim * ll + c) + rows * j] = h[(ll + j) + n * c];
                    }
                }
            }
            match svd_s_vh(&hm, rows, cols) {
                Ok(f) => break f,
                Err(e) if l <= 1 => return Err(e),
                Err(_) => l -= 1,
            }
        };
        out.l = l;
        let k = s.len();

        let mut m = match p.m {
            Some(m) => m.min(k - 1),
            None => {
                let err = p.err.ok_or_else(|| Error::InvalidParameter {
                    name: "err, M",
                    value: "None".to_string(),
                    reason:
                        "one of them is required (the knee detection for neither is not ported)"
                            .to_string(),
                })?;
                let mut m = find_m_with_err(&s, err, p.err_type);
                if s[m] / s[0] < 1e-14 {
                    m = find_m_with_err(&s, 1e-14, ErrType::Rel);
                }
                m
            }
        };
        let x_k: Vec<f64> = linspace(p.x_min, p.x_max, n);
        loop {
            out.sigma = s[m];
            // F = lstsq(W_0ᵀ, W_1ᵀ), W_0 = Vh[:M, :-1], W_1 = Vh[:M, 1:].
            let mut w0t = vec![C64::new(0.0, 0.0); l * m];
            let mut w1t = vec![C64::new(0.0, 0.0); l * m];
            for i in 0..m {
                for j in 0..l {
                    w0t[j + l * i] = vh[i + k * j];
                    w1t[j + l * i] = vh[i + k * (j + 1)];
                }
            }
            let f = lstsq(&w0t, l, m, &w1t, m)?;
            out.gamma = eigvals(&f, m)?;
            out.m = m;
            // omega = lstsq(V, h), V[i, j] = γ_j^i.
            let mut v = vec![C64::new(0.0, 0.0); n * m];
            for (j, &g) in out.gamma.iter().enumerate() {
                for i in 0..n {
                    v[i + n * j] = cpow(g, i as f64);
                }
            }
            out.omega = lstsq(&v, n, m, &h, dim)?;
            // cal_err
            let approx = out.get_value(&x_k);
            let (mut err_max, mut err_ave) = (0.0_f64, 0.0_f64);
            for c in 0..dim {
                let mut sum = 0.0;
                for i in 0..n {
                    let e = (approx[i + n * c] - h[i + n * c]).norm();
                    err_max = err_max.max(e);
                    sum += e;
                }
                err_ave = err_ave.max(sum / n as f64);
            }
            out.err_max = err_max;
            out.err_ave = err_ave;
            if err_max < (p.ctrl_ratio * out.sigma).max(1e-14 * s[0]) {
                break;
            }
            m = m.wrapping_sub(1);
            if m == 0 || m == usize::MAX {
                return Err(Error::DecompositionFailed {
                    reason: "ESPRIT could not find a controlled approximation".to_string(),
                });
            }
        }
        out.s = s;
        Ok(out)
    }

    /// The approximation at points `x` (column-major `len(x) x d`).
    pub fn get_value(&self, x: &[f64]) -> Vec<C64> {
        let nx = x.len();
        let mut out = vec![C64::new(0.0, 0.0); nx * self.dim];
        for (i, &xi) in x.iter().enumerate() {
            let e = (self.n as f64 - 1.0) * ((xi - self.x_min) / (self.x_max - self.x_min));
            for (j, &g) in self.gamma.iter().enumerate() {
                let vj = cpow(g, e);
                for c in 0..self.dim {
                    out[i + nx * c] += vj * self.omega[j + self.m * c];
                }
            }
        }
        for v in &mut out {
            *v = self.project(*v);
        }
        out
    }

    /// The approximation of column `col` at `x`.
    pub fn get_value_indiv(&self, x: f64, col: usize) -> C64 {
        let e = (self.n as f64 - 1.0) * ((x - self.x_min) / (self.x_max - self.x_min));
        let v: C64 = self
            .gamma
            .iter()
            .enumerate()
            .map(|(j, &g)| cpow(g, e) * self.omega[j + self.m * col])
            .sum();
        self.project(v)
    }

    fn project(&self, v: C64) -> C64 {
        match self.data_type {
            DataType::Cplx => v,
            DataType::Real => C64::new(v.re, 0.0),
            DataType::Imag => C64::new(0.0, v.im),
        }
    }
}

/// First index with `S[idx] < cutoff`, or the last index.
fn find_m_with_err(s: &[f64], err: f64, err_type: ErrType) -> usize {
    let cutoff = match err_type {
        ErrType::Abs => err,
        ErrType::Rel => s[0] * err,
    };
    s.iter().position(|&v| v < cutoff).unwrap_or(s.len() - 1)
}

/// `np.linspace(a, b, n)`.
pub(crate) fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n as f64 - 1.0);
    let mut x: Vec<f64> = (0..n).map(|i| a + i as f64 * step).collect();
    x[n - 1] = b;
    x
}

#[cfg(test)]
#[path = "esprit_tests.rs"]
mod tests;
