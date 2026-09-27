//! The pieces the applied examples share: the τ convention, the momentum
//! grid, and the transforms between the representations a diagrammatic
//! calculation moves through.
//!
//! A self-consistent calculation on a lattice keeps the same object in four
//! places — `(iν, k)`, `(iν, r)`, `(τ, r)` and the IR coefficients — and does
//! each step where it is cheap: the Dyson equation in `(iν, k)`, the products
//! that build `χ₀` and `Σ` in `(τ, r)`, and the basis fits in between.

use std::ops::Neg;

use mdarray::{DynRank, Tensor};
use num_complex::Complex64;
use rustfft::FftPlanner;
use sparse_ir::{
    Basis, Error, MatsubaraFreq, MatsubaraSampling, Statistics, StatisticsType, TauSampling,
};

/// Reorders `values`, sampled on `points` and laid out row-major as
/// `(points.len(), ncols)`, so that row `i` of the result holds the value at
/// `β − points[i]`.
///
/// The sampling times of `sparse-ir` lie on `[−β/2, β/2]` and the grid is
/// closed under `τ → −τ`, so reversing the rows gives `G(−τ)`. The value at
/// `β − τ` follows from the (anti)periodicity of a Green's function,
///
/// ```text
///     G(β − τ) = ζ G(−τ),     ζ = −1 (fermionic), +1 (bosonic),
/// ```
///
/// which is the only difference from the `grit[::-1]` of the Python
/// notebooks, where the sampling times have been folded into `[0, β)` and the
/// grid is symmetric about `β/2` instead.
///
/// Panics if the grid is not symmetric — the expressions this function exists
/// for would silently return a different quantity.
pub fn reverse_tau_rows<S, T>(points: &[f64], values: &[T], ncols: usize) -> Vec<T>
where
    S: StatisticsType,
    T: Copy + Neg<Output = T>,
{
    let nrows = points.len();
    assert!(ncols > 0, "a row needs at least one column");
    assert_eq!(
        values.len(),
        nrows * ncols,
        "expected {nrows} rows of {ncols} values, got {} values",
        values.len()
    );
    assert_symmetric_tau_grid(points);

    let flip = matches!(S::STATISTICS, Statistics::Fermionic);
    let mut out = Vec::with_capacity(values.len());
    for i in 0..nrows {
        let row = &values[(nrows - 1 - i) * ncols..][..ncols];
        out.extend(row.iter().map(|&v| if flip { -v } else { v }));
    }
    out
}

/// Panics unless `points` is sorted and closed under `τ → −τ`.
pub fn assert_symmetric_tau_grid(points: &[f64]) {
    let n = points.len();
    assert!(n > 0, "an empty grid cannot be reversed");
    let scale = points.iter().fold(0.0f64, |acc, t| acc.max(t.abs()));
    for (i, &tau) in points.iter().enumerate() {
        let mirrored = points[n - 1 - i];
        assert!(
            (tau + mirrored).abs() <= 1e-12 * scale.max(1.0),
            "the sampling times must be closed under τ → −τ for G(β − τ) to be a \
             reordering of G(τ), but {tau} is paired with {mirrored}"
        );
    }
}

/// A square momentum grid and the Fourier transforms between `k` and `r`.
///
/// The grid is `nk1 × nk2` points of the Brillouin zone, flattened row-major
/// into a single index so that every quantity is a row-major `(n, nk)` array
/// whose first axis is imaginary time or Matsubara frequency.
pub struct MomentumGrid {
    nk1: usize,
    nk2: usize,
}

impl MomentumGrid {
    pub fn new(nk1: usize, nk2: usize) -> Self {
        assert!(
            nk1 > 0 && nk2 > 0,
            "the grid needs at least one point per axis"
        );
        Self { nk1, nk2 }
    }

    pub fn nk1(&self) -> usize {
        self.nk1
    }

    pub fn nk2(&self) -> usize {
        self.nk2
    }

    /// Number of momentum points.
    pub fn len(&self) -> usize {
        self.nk1 * self.nk2
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// The fractional coordinates `(k₁, k₂)` of point `index`, each in `[0, 1)`.
    pub fn coordinates(&self, index: usize) -> (f64, f64) {
        let (i, j) = (index / self.nk2, index % self.nk2);
        (i as f64 / self.nk1 as f64, j as f64 / self.nk2 as f64)
    }

    /// `ε(k) = −2t (cos 2πk₁ + cos 2πk₂)`, the nearest-neighbour square lattice.
    pub fn square_lattice_dispersion(&self, t: f64) -> Vec<f64> {
        use std::f64::consts::TAU;
        (0..self.len())
            .map(|index| {
                let (k1, k2) = self.coordinates(index);
                -2.0 * t * ((TAU * k1).cos() + (TAU * k2).cos())
            })
            .collect()
    }

    /// `A(r) = (1/nk) Σ_k e^{−ik·r} A(k)`: the zone average.
    ///
    /// Matches `np.fft.fftn(..., axes=(1, 2)) / nk` of the Python notebooks.
    /// The `1/nk` belongs here rather than in [`r_to_k`], because it is the
    /// momentum sum that is an average over the zone; with it, a product in
    /// real space is the convolution `(1/nk) Σ_q A(q) B(k − q)` that the
    /// diagrams are written with, and the two transforms are inverses.
    ///
    /// [`r_to_k`]: Self::r_to_k
    pub fn k_to_r(&self, values: &[Complex64]) -> Vec<Complex64> {
        let nk = self.len() as f64;
        self.transform(values, Direction::Forward, 1.0 / nk)
    }

    /// `A(k) = Σ_r e^{ik·r} A(r)`, with no prefactor.
    ///
    /// Matches `np.fft.ifftn(..., axes=(1, 2)) * nk` of the Python notebooks,
    /// and undoes [`k_to_r`] exactly.
    ///
    /// [`k_to_r`]: Self::k_to_r
    pub fn r_to_k(&self, values: &[Complex64]) -> Vec<Complex64> {
        self.transform(values, Direction::Inverse, 1.0)
    }

    fn transform(&self, values: &[Complex64], direction: Direction, scale: f64) -> Vec<Complex64> {
        let nk = self.len();
        assert_eq!(
            values.len() % nk,
            0,
            "expected whole rows of {nk} momentum points, got {} values",
            values.len()
        );
        let mut planner = FftPlanner::new();
        let fft1 = match direction {
            Direction::Forward => planner.plan_fft_forward(self.nk1),
            Direction::Inverse => planner.plan_fft_inverse(self.nk1),
        };
        let fft2 = match direction {
            Direction::Forward => planner.plan_fft_forward(self.nk2),
            Direction::Inverse => planner.plan_fft_inverse(self.nk2),
        };

        let mut out = values.to_vec();
        let mut column = vec![Complex64::default(); self.nk1];
        for row in out.chunks_mut(nk) {
            // The second axis is contiguous; the first one has to be gathered.
            for line in row.chunks_mut(self.nk2) {
                fft2.process(line);
            }
            for j in 0..self.nk2 {
                for (i, slot) in column.iter_mut().enumerate() {
                    *slot = row[i * self.nk2 + j];
                }
                fft1.process(&mut column);
                for (i, &value) in column.iter().enumerate() {
                    row[i * self.nk2 + j] = value;
                }
            }
            if scale != 1.0 {
                for value in row.iter_mut() {
                    *value *= scale;
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
}

/// The sampling objects of one statistics, and the transforms between the
/// three representations a diagrammatic calculation moves through.
///
/// Everything is a flat row-major `(n, ncols)` array whose first axis is the
/// sampled one: `ncols` is the number of momentum points, or `1` for a purely
/// local calculation. That is the layout the momentum transforms of
/// [`MomentumGrid`] expect too, so the two compose without a reshape.
pub struct IrMesh<S: StatisticsType + 'static> {
    tau: TauSampling<S>,
    wn: MatsubaraSampling<S>,
}

impl<S: StatisticsType + 'static> IrMesh<S> {
    pub fn new(basis: &impl Basis<S>) -> Result<Self, Error> {
        Ok(Self {
            tau: TauSampling::<S>::new(basis)?,
            wn: MatsubaraSampling::<S>::new(basis)?,
        })
    }

    /// The sampling times, on `[−β/2, β/2]`.
    pub fn tau_points(&self) -> &[f64] {
        self.tau.sampling_points()
    }

    /// The sampling frequencies, as the integer index `n` of `iνₙ`.
    pub fn wn(&self) -> &[MatsubaraFreq<S>] {
        self.wn.sampling_points()
    }

    pub fn n_tau(&self) -> usize {
        self.tau.n_sampling_points()
    }

    pub fn n_wn(&self) -> usize {
        self.wn.n_sampling_points()
    }

    pub fn tau_sampling(&self) -> &TauSampling<S> {
        &self.tau
    }

    pub fn wn_sampling(&self) -> &MatsubaraSampling<S> {
        &self.wn
    }

    /// `G(τ) → G(iν)`, one column at a time, through the IR coefficients.
    pub fn tau_to_wn(&self, values: &[Complex64], ncols: usize) -> Result<Vec<Complex64>, Error> {
        self.l_to_wn(&self.tau_to_l(values, ncols)?, ncols)
    }

    /// `G(iν) → G(τ)`, the inverse of [`tau_to_wn`](Self::tau_to_wn).
    pub fn wn_to_tau(&self, values: &[Complex64], ncols: usize) -> Result<Vec<Complex64>, Error> {
        self.l_to_tau(&self.wn_to_l(values, ncols)?, ncols)
    }

    /// The IR coefficients behind values given at the sampling frequencies.
    pub fn wn_to_l(&self, values: &[Complex64], ncols: usize) -> Result<Vec<Complex64>, Error> {
        let input = to_tensor(values, self.n_wn(), ncols);
        Ok(from_tensor(&self.wn.fit_nd(None, &input, 0)?))
    }

    /// The IR coefficients behind values given at the sampling times.
    pub fn tau_to_l(&self, values: &[Complex64], ncols: usize) -> Result<Vec<Complex64>, Error> {
        let input = to_tensor(values, self.n_tau(), ncols);
        Ok(from_tensor(&self.tau.fit_nd_zz(None, &input.expr(), 0)?))
    }

    /// IR coefficients → values at the sampling times.
    pub fn l_to_tau(&self, coeffs: &[Complex64], ncols: usize) -> Result<Vec<Complex64>, Error> {
        let input = to_tensor_rows(coeffs, ncols);
        Ok(from_tensor(&self.tau.evaluate_nd_zz(
            None,
            &input.expr(),
            0,
        )?))
    }

    /// IR coefficients → values at the sampling frequencies.
    pub fn l_to_wn(&self, coeffs: &[Complex64], ncols: usize) -> Result<Vec<Complex64>, Error> {
        let input = to_tensor_rows(coeffs, ncols);
        Ok(from_tensor(&self.wn.evaluate_nd(None, &input.expr(), 0)?))
    }

    /// `G(τ) → G(β − τ)`; see [`reverse_tau_rows`].
    pub fn reverse_tau(&self, values: &[Complex64], ncols: usize) -> Vec<Complex64> {
        reverse_tau_rows::<S, Complex64>(self.tau_points(), values, ncols)
    }
}

/// Wraps `values` as an `nrows × ncols` tensor, taking the number of rows
/// from the length — which is how many IR coefficients there turned out to be.
fn to_tensor_rows(values: &[Complex64], ncols: usize) -> Tensor<Complex64, DynRank> {
    assert!(ncols > 0, "a row needs at least one column");
    assert_eq!(
        values.len() % ncols,
        0,
        "expected whole rows of {ncols} values, got {} values",
        values.len()
    );
    to_tensor(values, values.len() / ncols, ncols)
}

fn to_tensor(values: &[Complex64], nrows: usize, ncols: usize) -> Tensor<Complex64, DynRank> {
    assert_eq!(
        values.len(),
        nrows * ncols,
        "expected {nrows} rows of {ncols} values, got {} values",
        values.len()
    );
    Tensor::<Complex64, DynRank>::from_fn(&[nrows, ncols][..], |index| {
        values[index[0] * ncols + index[1]]
    })
}

fn from_tensor(tensor: &Tensor<Complex64, DynRank>) -> Vec<Complex64> {
    tensor.iter().copied().collect()
}
