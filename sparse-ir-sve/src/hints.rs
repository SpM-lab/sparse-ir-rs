//! SVE hints given by the segments of the domain
//!
//! A kernel implements [`SVEKernel::sve_hints`] by returning a type that
//! implements [`SVEHints`]. [`SegmentHints`] is such a type for any kernel:
//! it holds the segment boundaries and the number of Gauss points per
//! segment, so that a kernel need not define a hints type of its own.

use crate::error::Error;
use crate::kernel::{SVEHints, SVEKernel};
use crate::numeric::CustomNumeric;
use std::fmt::Debug;

/// [`SVEHints`] given by the segment boundaries of the domain
///
/// The boundaries are given on the full domain `[xmin, xmax] × [ymin, ymax]`
/// of the kernel, whether it is centrosymmetric or not. A centrosymmetric
/// kernel is discretized on `[0, xmax] × [0, ymax]` and the mirror images of
/// its segments ([`SVEHints`]): the boundaries in `(0, xmax]` (`(0, ymax]`),
/// with 0 added if it is not one of them.
///
/// The accuracy of the SVE depends on the segments: on each of them, the
/// kernel must be resolved by polynomials of degree `ngauss - 1` in each
/// variable. [`sve_residual`](crate::sve::sve_residual) measures how well the
/// result reproduces the kernel.
///
/// # Example
///
/// ```
/// use sparse_ir_sve::{AbstractKernel, CustomNumeric, SVEKernel, SVEStrategyKernel, SegmentHints};
/// use std::fmt::Debug;
///
/// /// K(x, y) = exp(-(x - y)²) on [-1, 1] × [0, 2]
/// #[derive(Clone)]
/// struct Gaussian;
///
/// impl AbstractKernel for Gaussian {
///     fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
///         (-(x - y) * (x - y)).exp()
///     }
/// }
///
/// impl SVEKernel for Gaussian {
///     type SVEHintsType<T>
///         = SegmentHints
///     where
///         T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
///
///     fn xmax(&self) -> f64 {
///         1.0
///     }
///
///     fn ymax(&self) -> f64 {
///         2.0
///     }
///
///     fn ymin(&self) -> f64 {
///         0.0
///     }
///
///     fn sve_hints<T>(&self, _epsilon: f64) -> SegmentHints
///     where
///         T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
///     {
///         // Four segments on each axis, 16 Gauss points on each segment
///         SegmentHints::uniform(self, 4, 4, 16).unwrap()
///     }
/// }
///
/// // Not centrosymmetric: the SVE is taken over the full domain
/// impl SVEStrategyKernel for Gaussian {}
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentHints {
    /// Boundaries on x
    segments_x: Vec<f64>,
    /// Boundaries on y
    segments_y: Vec<f64>,
    /// Gauss points per segment
    ngauss: usize,
}

impl SegmentHints {
    /// Hints with the segment boundaries `segments_x` and `segments_y` of the
    /// full domain of `kernel` and `ngauss` Gauss points per segment
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `ngauss` is 0, or the boundaries on an
    /// axis are not finite and strictly increasing from the lower to the
    /// upper bound of the domain of `kernel` on that axis (as
    /// [`SVEKernel::xmin`] and [`SVEKernel::xmax`] return them)
    pub fn new<K: SVEKernel>(
        kernel: &K,
        segments_x: &[f64],
        segments_y: &[f64],
        ngauss: usize,
    ) -> Result<Self, Error> {
        if ngauss == 0 {
            return Err(Error::InvalidParameter {
                name: "ngauss",
                value: "0".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        check_segments("segments_x", segments_x, kernel.xmin(), kernel.xmax())?;
        check_segments("segments_y", segments_y, kernel.ymin(), kernel.ymax())?;
        Ok(Self {
            segments_x: segments_x.to_vec(),
            segments_y: segments_y.to_vec(),
            ngauss,
        })
    }

    /// Hints with `nseg_x` and `nseg_y` segments of equal length on the full
    /// domain of `kernel` and `ngauss` Gauss points per segment
    ///
    /// On a domain symmetric about 0, the boundaries are symmetric as well,
    /// so a centrosymmetric kernel with an even number of segments is
    /// discretized on half of them.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidParameter`] if `nseg_x`, `nseg_y` or `ngauss` is 0, or
    /// the domain of `kernel` is not a finite, non-empty interval on an axis
    pub fn uniform<K: SVEKernel>(
        kernel: &K,
        nseg_x: usize,
        nseg_y: usize,
        ngauss: usize,
    ) -> Result<Self, Error> {
        let x = uniform_segments("nseg_x", nseg_x, kernel.xmin(), kernel.xmax())?;
        let y = uniform_segments("nseg_y", nseg_y, kernel.ymin(), kernel.ymax())?;
        Self::new(kernel, &x, &y, ngauss)
    }
}

/// `n` segments of equal length on `[min, max]`: the boundaries
/// `c + h (2i - n) / n` with `c` and `h` the midpoint and half-length,
/// exactly symmetric about `c` and ending exactly at `min` and `max`
fn uniform_segments(name: &'static str, n: usize, min: f64, max: f64) -> Result<Vec<f64>, Error> {
    if n == 0 {
        return Err(Error::InvalidParameter {
            name,
            value: "0".to_string(),
            reason: "must be positive".to_string(),
        });
    }
    let (c, h) = (0.5 * (min + max), 0.5 * (max - min));
    let mut segments: Vec<f64> = (0..=n)
        .map(|i| c + h * ((2 * i) as f64 - n as f64) / n as f64)
        .collect();
    segments[0] = min;
    segments[n] = max;
    Ok(segments)
}

/// `Ok` if `segments` has at least 2 finite, strictly increasing entries from
/// `min` to `max`
fn check_segments(name: &'static str, segments: &[f64], min: f64, max: f64) -> Result<(), Error> {
    let invalid = |value: String, reason: String| Error::InvalidParameter {
        name,
        value,
        reason,
    };
    if segments.len() < 2 {
        return Err(invalid(
            format!("{} boundaries", segments.len()),
            "must have at least 2 entries".to_string(),
        ));
    }
    if let Some((i, s)) = segments.iter().enumerate().find(|(_, s)| !s.is_finite()) {
        return Err(invalid(
            format!("{s:?} at index {i}"),
            "must be finite".to_string(),
        ));
    }
    if let Some(i) = (1..segments.len()).find(|&i| segments[i] <= segments[i - 1]) {
        return Err(invalid(
            format!("{:?} after {:?} at index {i}", segments[i], segments[i - 1]),
            "must be strictly increasing".to_string(),
        ));
    }
    let (first, last) = (segments[0], segments[segments.len() - 1]);
    if first != min || last != max {
        return Err(invalid(
            format!("[{first:?}, {last:?}]"),
            format!("must run from the lower to the upper bound of the domain, [{min:?}, {max:?}]"),
        ));
    }
    Ok(())
}

impl<T> SVEHints<T> for SegmentHints
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        self.segments_x
            .iter()
            .map(|&s| T::from_f64_unchecked(s))
            .collect()
    }

    fn segments_y(&self) -> Vec<T> {
        self.segments_y
            .iter()
            .map(|&s| T::from_f64_unchecked(s))
            .collect()
    }

    /// The number of singular values the discretization can resolve, at
    /// most `ngauss` per segment on the axis with fewer segments; the SVE
    /// does not use it
    fn nsvals(&self) -> usize {
        let segments = (self.segments_x.len()).min(self.segments_y.len()) - 1;
        self.ngauss * segments
    }

    fn ngauss(&self) -> usize {
        self.ngauss
    }
}

#[cfg(test)]
#[path = "hints_tests.rs"]
mod hints_tests;
