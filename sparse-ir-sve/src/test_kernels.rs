//! Kernels for the tests of this crate
//!
//! The base kernel is K(x, y) = exp(a x y) on [-1, 1] × [-1, 1]: smooth,
//! centrosymmetric, with singular values that decay faster than
//! geometrically, about (a/2)^l / l!. Its variants break the symmetry
//! ([`TiltedExpKernel`]) or give the discretization zero rows
//! ([`HalfExpKernel`]), or claim centrosymmetry on an asymmetric domain
//! ([`AsymmetricDomainExpKernel`]). The SVE hints are uniform segments, `nseg` on [0, 1]
//! (mirrored onto [-1, 1] for the full-domain variants), so that a test sets
//! the size of the discretization.

use crate::kernel::{
    AbstractKernel, ReducedKernel, SVEHints, SVEKernel, SVEStrategyKernel, SymmetryType,
};
use crate::numeric::CustomNumeric;
use crate::sve::mirror_segments_to_full_domain;
use std::fmt::Debug;

/// K(x, y) = exp(a x y) on [-1, 1] × [-1, 1]
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpKernel {
    /// Coefficient of x y in the exponent; the larger, the more singular
    /// values above a given threshold
    pub(crate) a: f64,
    /// Number of uniform segments of [0, 1] in the SVE hints
    pub(crate) nseg: usize,
}

impl ExpKernel {
    pub(crate) fn new(a: f64) -> Self {
        Self { a, nseg: 4 }
    }
}

impl AbstractKernel for ExpKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        (T::from_f64_unchecked(self.a) * x * y).exp()
    }

    fn is_centrosymmetric(&self) -> bool {
        true
    }
}

impl ReducedKernel for ExpKernel {
    /// 2 cosh(a x y) and 2 sinh(a x y), without the cancellation of the odd
    /// part near x y = 0
    fn compute_reduced<T: CustomNumeric + Copy + Debug>(
        &self,
        x: T,
        y: T,
        symmetry: SymmetryType,
    ) -> T {
        let axy = T::from_f64_unchecked(self.a) * x * y;
        let two = T::from_f64_unchecked(2.0);
        match symmetry {
            SymmetryType::Even => two * axy.cosh(),
            SymmetryType::Odd => two * axy.sinh(),
        }
    }
}

/// Uniform segments: `nseg` on [0, 1], or `2 nseg` on [-1, 1] when `full`
#[derive(Debug, Clone)]
pub(crate) struct UniformHints {
    nseg: usize,
    full: bool,
}

impl<T> SVEHints<T> for UniformHints
where
    T: Copy + Debug + Send + Sync + CustomNumeric,
{
    fn segments_x(&self) -> Vec<T> {
        let half: Vec<T> = (0..=self.nseg)
            .map(|i| T::from_f64_unchecked(i as f64 / self.nseg as f64))
            .collect();
        if self.full {
            mirror_segments_to_full_domain(&half)
        } else {
            half
        }
    }

    fn segments_y(&self) -> Vec<T> {
        self.segments_x()
    }

    fn nsvals(&self) -> usize {
        40
    }

    fn ngauss(&self) -> usize {
        16
    }
}

impl SVEKernel for ExpKernel {
    type SVEHintsType<T>
        = UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn sve_hints<T>(&self, _epsilon: f64) -> UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        UniformHints {
            nseg: self.nseg,
            full: false,
        }
    }
}

/// [`ExpKernel`] tilted by `1 + x/2`, so that `K(-x, -y) != K(x, y)`
#[derive(Debug, Clone, Copy)]
pub(crate) struct TiltedExpKernel(pub(crate) ExpKernel);

impl AbstractKernel for TiltedExpKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        let tilt = T::from_f64_unchecked(1.0) + T::from_f64_unchecked(0.5) * x;
        self.0.compute(x, y) * tilt
    }
}

impl SVEKernel for TiltedExpKernel {
    type SVEHintsType<T>
        = UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn sve_hints<T>(&self, _epsilon: f64) -> UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        UniformHints {
            nseg: self.0.nseg,
            full: true,
        }
    }
}

impl SVEStrategyKernel for TiltedExpKernel {}

/// [`ExpKernel`] cut to zero for x ≤ 0: not centrosymmetric, with zero rows
/// in its discretization (the first of which is ACA's first pivot)
#[derive(Debug, Clone, Copy)]
pub(crate) struct HalfExpKernel(pub(crate) ExpKernel);

impl AbstractKernel for HalfExpKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        if x > T::from_f64_unchecked(0.0) {
            self.0.compute(x, y)
        } else {
            T::from_f64_unchecked(0.0)
        }
    }
}

impl SVEKernel for HalfExpKernel {
    type SVEHintsType<T>
        = UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn sve_hints<T>(&self, _epsilon: f64) -> UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        UniformHints {
            nseg: self.0.nseg,
            full: true,
        }
    }
}

impl SVEStrategyKernel for HalfExpKernel {}

/// [`ExpKernel`] claiming centrosymmetry on the asymmetric domain
/// y ∈ [-0.5, 1]
#[derive(Debug, Clone, Copy)]
pub(crate) struct AsymmetricDomainExpKernel(pub(crate) ExpKernel);

impl AbstractKernel for AsymmetricDomainExpKernel {
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
        self.0.compute(x, y)
    }

    fn is_centrosymmetric(&self) -> bool {
        true
    }
}

impl ReducedKernel for AsymmetricDomainExpKernel {
    fn compute_reduced<T: CustomNumeric + Copy + Debug>(
        &self,
        x: T,
        y: T,
        symmetry: SymmetryType,
    ) -> T {
        self.0.compute_reduced(x, y, symmetry)
    }
}

impl SVEKernel for AsymmetricDomainExpKernel {
    type SVEHintsType<T>
        = UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    fn xmax(&self) -> f64 {
        1.0
    }

    fn ymax(&self) -> f64 {
        1.0
    }

    fn ymin(&self) -> f64 {
        -0.5
    }

    fn sve_hints<T>(&self, epsilon: f64) -> UniformHints
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static,
    {
        self.0.sve_hints::<T>(epsilon)
    }
}
