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

/// Reorders and re-signs `values`, sampled on `points` and laid out row-major
/// as `(points.len(), ncols)`, so that row `i` of the result holds the value
/// at `β − points[i]`.
///
/// The sampling times of `sparse-ir` lie on `[−β/2, β/2]`, where the value at
/// `β − τ` follows from the (anti)periodicity of a Green's function,
///
/// ```text
///     G(β − τ) = ζ G(−τ),     ζ = −1 (fermionic), +1 (bosonic),
/// ```
///
/// so the operation is a permutation with a sign. That is the only difference
/// from the `grit[::-1]` of the Python notebooks, where the sampling times
/// have been folded into `[0, β)` and the grid is symmetric about `β/2`.
///
/// The grid is *nearly* symmetric about zero, but not quite: it can carry a
/// point at `τ = β/2`, whose mirror `−β/2` is the same point seen one period
/// away. [`tau_reversal`] resolves that, and this function applies it.
///
/// `S` is the statistics of the *function*, which need not be the statistics
/// of the grid: a fermionic `G` sampled at the bosonic times is still
/// anti-periodic.
pub fn reverse_tau_rows<S, T>(points: &[f64], beta: f64, values: &[T], ncols: usize) -> Vec<T>
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

    let mut out = Vec::with_capacity(values.len());
    for (source, flip) in tau_reversal::<S>(points, beta) {
        let row = &values[source * ncols..][..ncols];
        out.extend(row.iter().map(|&v| if flip { -v } else { v }));
    }
    out
}

/// The permutation and signs behind `G(τ) → G(β − τ)` on a sampling grid.
///
/// Entry `i` says which row holds `G(β − points[i])` and whether it has to
/// change sign. Both come from one identity applied twice: `G(β − τ)` is
/// `ζ G(−τ)`, and `−τ` is itself on the grid only up to a period, picking up
/// another `ζ` when it is not.
///
/// Panics if some `−τ` is neither on the grid nor one period away from it,
/// which would mean `G(β − τ)` is not any of the values that were sampled.
pub fn tau_reversal<S: StatisticsType>(points: &[f64], beta: f64) -> Vec<(usize, bool)> {
    assert!(beta > 0.0, "β must be positive, got {beta}");
    assert!(!points.is_empty(), "an empty grid cannot be reversed");
    let zeta_flips = matches!(S::STATISTICS, Statistics::Fermionic);
    let tolerance = 1e-12 * beta;

    points
        .iter()
        .map(|&tau| {
            let target = -tau;
            let found = points
                .iter()
                .position(|&t| (t - target).abs() <= tolerance)
                .map(|index| (index, false))
                .or_else(|| {
                    // −τ off the grid means it is one period away from a point
                    // that is on it, and crossing a period costs another ζ.
                    points
                        .iter()
                        .position(|&t| {
                            (t - target - beta).abs() <= tolerance
                                || (t - target + beta).abs() <= tolerance
                        })
                        .map(|index| (index, zeta_flips))
                });
            let (index, wrapped) = found.unwrap_or_else(|| {
                panic!(
                    "G(β − τ) is not among the sampled values: no sampling time equals \
                     {target} or {target} ± {beta}"
                )
            });
            (index, zeta_flips != wrapped)
        })
        .collect()
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
    beta: f64,
    tau: TauSampling<S>,
    wn: MatsubaraSampling<S>,
}

impl<S: StatisticsType + 'static> IrMesh<S> {
    pub fn new(basis: &impl Basis<S>) -> Result<Self, Error> {
        Ok(Self {
            beta: basis.beta(),
            tau: TauSampling::<S>::new(basis)?,
            wn: MatsubaraSampling::<S>::new(basis)?,
        })
    }

    pub fn beta(&self) -> f64 {
        self.beta
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

    /// `G(τ) → G(β − τ)` for a function of this mesh's statistics.
    pub fn reverse_tau(&self, values: &[Complex64], ncols: usize) -> Vec<Complex64> {
        self.reverse_tau_as::<S>(values, ncols)
    }

    /// `G(τ) → G(β − τ)` for a function of statistics `Z` sampled at *this*
    /// mesh's times.
    ///
    /// GW needs exactly this: the polarization is built from a fermionic `G`
    /// at the bosonic sampling times, and it is `G` that is anti-periodic, not
    /// the times.
    pub fn reverse_tau_as<Z: StatisticsType>(
        &self,
        values: &[Complex64],
        ncols: usize,
    ) -> Vec<Complex64> {
        reverse_tau_rows::<Z, Complex64>(self.tau_points(), self.beta, values, ncols)
    }
}

/// `Σₗ cₗ f(xᵢ)` for a matrix of basis functions `f[i][l]` and a row-major
/// `(size, ncols)` block of IR coefficients.
///
/// This is how the applied examples leave the basis for points the sampling
/// did not choose: a fermionic function at the bosonic sampling times, or the
/// Matsubara sum of a product, which is that product at `τ = 0`. The matrix
/// comes from [`Basis::evaluate_tau`] or [`Basis::evaluate_matsubara`], both
/// of which apply the statistics of the basis they belong to — which is what
/// makes a cross-statistics evaluation safe.
pub fn evaluate_rows(
    functions: &mdarray::DTensor<f64, 2>,
    coefficients: &[Complex64],
    ncols: usize,
) -> Vec<Complex64> {
    let (n_points, size) = *functions.shape();
    assert!(ncols > 0, "a row needs at least one column");
    assert_eq!(
        coefficients.len(),
        size * ncols,
        "expected {size} rows of {ncols} coefficients, got {} values",
        coefficients.len()
    );
    let mut out = vec![Complex64::default(); n_points * ncols];
    for i in 0..n_points {
        for l in 0..size {
            let f = functions[[i, l]];
            if f == 0.0 {
                continue;
            }
            for column in 0..ncols {
                out[i * ncols + column] += coefficients[l * ncols + column] * f;
            }
        }
    }
    out
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
