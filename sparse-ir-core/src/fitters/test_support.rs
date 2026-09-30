//! Shared helpers for fitter unit tests.

use crate::matrix::Mat;
use num_complex::Complex;
use num_traits::Zero;
use std::ops::{Add, Mul};

pub(crate) type C64 = Complex<f64>;

/// Well-conditioned real test matrix (`n >= m`).
pub(crate) fn real_matrix(n: usize, m: usize) -> Mat<f64> {
    Mat::from_fn([n, m], |idx| {
        let x = (idx[0] as f64 + 0.5) / n as f64;
        (std::f64::consts::PI * (idx[1] as f64 + 0.5) * x).cos()
            + if idx[0] == idx[1] { 1.0 } else { 0.0 }
    })
}

/// Well-conditioned complex test matrix (`n >= m`).
pub(crate) fn complex_matrix(n: usize, m: usize) -> Mat<C64> {
    Mat::from_fn([n, m], |idx| {
        let x = (idx[0] as f64 + 0.5) / n as f64;
        let t = std::f64::consts::PI * (idx[1] as f64 + 0.5) * x;
        C64::new(t.cos(), 0.7 * (t + 0.3).sin()) + if idx[0] == idx[1] { 1.0 } else { 0.0 }
    })
}

pub(crate) fn real_data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (((i + seed) * 37 + 11) % 23) as f64 / 7.0 - 1.5)
        .collect()
}

pub(crate) fn complex_data(len: usize, seed: usize) -> Vec<C64> {
    let re = real_data(len, seed);
    let im = real_data(len, seed + 101);
    re.into_iter()
        .zip(im)
        .map(|(r, i)| C64::new(r, i))
        .collect()
}

/// Naive column-major contraction of `a` (`rows x cols`) with axis `dim` of
/// `x` (shape `shape`, `shape[dim] == cols`).
pub(crate) fn naive_apply<A, X, Y>(
    a: &[A],
    rows: usize,
    cols: usize,
    x: &[X],
    shape: &[usize],
    dim: usize,
) -> Vec<Y>
where
    A: Copy + Mul<X, Output = Y>,
    X: Copy,
    Y: Copy + Zero + Add<Output = Y>,
{
    assert_eq!(shape[dim], cols);
    let pre: usize = shape[..dim].iter().product();
    let post: usize = shape[dim + 1..].iter().product();
    let mut y = vec![Y::zero(); pre * rows * post];
    for q in 0..post {
        for i in 0..rows {
            for p in 0..pre {
                let mut acc = Y::zero();
                for j in 0..cols {
                    acc = acc + a[i + rows * j] * x[p + pre * (j + cols * q)];
                }
                y[p + pre * (i + rows * q)] = acc;
            }
        }
    }
    y
}

pub(crate) trait Abs {
    fn abs_(self) -> f64;
}

impl Abs for f64 {
    fn abs_(self) -> f64 {
        self.abs()
    }
}

impl Abs for C64 {
    fn abs_(self) -> f64 {
        self.norm()
    }
}

pub(crate) fn max_diff<T: Copy + std::ops::Sub<Output = T> + Abs>(a: &[T], b: &[T]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs_())
        .fold(0.0, f64::max)
}

/// Row-major copy of a column-major buffer with shape `shape`, and the
/// strides of a view that reads it as the same logical tensor.
pub(crate) fn to_row_major<T: Copy>(x: &[T], shape: &[usize]) -> (Vec<T>, Vec<isize>) {
    let rank = shape.len();
    let mut rm_strides = vec![1isize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        rm_strides[d] = rm_strides[d + 1] * shape[d + 1] as isize;
    }
    let mut out = x.to_vec();
    let mut idx = vec![0usize; rank];
    for (lin, &v) in x.iter().enumerate() {
        let mut r = lin;
        for d in 0..rank {
            idx[d] = r % shape[d];
            r /= shape[d];
        }
        let off: isize = idx
            .iter()
            .zip(&rm_strides)
            .map(|(&i, &s)| i as isize * s)
            .sum();
        out[off as usize] = v;
    }
    (out, rm_strides)
}
