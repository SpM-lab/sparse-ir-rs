//! How well an SVE reproduces its kernel

use nalgebra::DMatrix;

use super::SVEResult;
use crate::kernel::AbstractKernel;

/// Position of the points next to the ends of a segment at which
/// [`sve_residual`] compares, relative to the segment length. Strictly inside
/// the segment, so that a kernel discontinuous at a knot is compared with the
/// expansion on the same side of it.
const END_OFFSET: f64 = 1.0 / 1048576.0;

/// The points of each segment between `knots` at which [`sve_residual`]
/// compares: next to both ends, where a polynomial interpolant departs most
/// from the function it interpolates, and the midpoint
fn residual_points(knots: &[f64]) -> Vec<f64> {
    knots
        .windows(2)
        .flat_map(|w| {
            let (a, b) = (w[0], w[1]);
            let d = (b - a) * END_OFFSET;
            [a + d, 0.5 * (a + b), b - d]
        })
        .collect()
}

/// The residual of the SVE `sve` of `kernel`, relative to the largest
/// singular value: max |K(x, y) - Σ_l s_l u_l(x) v_l(y)| / s_0
///
/// The maximum is taken over the points next to both ends and at the
/// midpoint of every segment of the singular functions, on both axes, where
/// the expansion is evaluated in `f64`. The result measures the
/// discretization of the kernel by the segments of its SVE hints (which
/// [`compute_sve`](super::compute_sve) does not check) together with the
/// truncation of the expansion, which contributes about the first singular
/// value left out, below `epsilon`·s_0.
///
/// As a maximum over finitely many points, it is a lower bound of the
/// maximum over the whole domain. For resolved and under-resolved kernels
/// (exp(a x y) and the same times 1 + x/2, a ridge (1 + x/2) / (1 + 100 (x -
/// y)²), √(1 + x) e^{xy}, and e^{-Λy(x+1)/2} / (1 + e^{-Λy}), on 1 to 32
/// uniform segments per half axis), it was between 0.65 and 1.2 times the
/// maximum over a 201 × 201 grid.
///
/// The evaluation in `f64` limits it from below: rounding x and y moves
/// K(x, y) by about `f64::EPSILON` times |x ∂K/∂x| + |y ∂K/∂y|. For
/// e^{-Λy(x+1)/2} / (1 + e^{-Λy}) with the segments of the IR basis, it is
/// 8·10⁻¹⁴ at Λ = 10², 9·10⁻¹² at Λ = 10⁴ and 10⁻⁸ at Λ = 10⁶, for
/// `epsilon` = 10⁻⁶ and 10⁻¹² alike, and does not decrease with more
/// segments or Gauss points.
///
/// It evaluates the kernel in `f64` at 9 points per pair of segments, and
/// the singular functions at these points; for the same kernel (Λ = 10² to
/// 10⁶), that took 0.1 to 0.3 times as long as the SVE on 8 threads, and
/// 0.06 to 0.12 times on one (it runs on the calling thread).
pub fn sve_residual<K: AbstractKernel>(kernel: &K, sve: &SVEResult) -> f64 {
    let s = sve.s();
    let (u, v) = (sve.u().polys(), sve.v().polys());
    let xs = residual_points(u[0].knots());
    let ys = residual_points(v[0].knots());
    // s_l u_l(x_i) and v_l(y_j), so that their product is the expansion
    let mut us = DMatrix::<f64>::zeros(xs.len(), s.len());
    let mut vs = DMatrix::<f64>::zeros(ys.len(), s.len());
    for l in 0..s.len() {
        for (i, ux) in u[l].evaluate_many(&xs).into_iter().enumerate() {
            us[(i, l)] = s[l] * ux;
        }
        for (j, vy) in v[l].evaluate_many(&ys).into_iter().enumerate() {
            vs[(j, l)] = vy;
        }
    }
    let expansion = us * vs.transpose();
    let mut residual: f64 = 0.0;
    for (j, &y) in ys.iter().enumerate() {
        for (i, &x) in xs.iter().enumerate() {
            let k: f64 = kernel.compute(x, y);
            let r = (k - expansion[(i, j)]).abs();
            // NaN (from a kernel value that is not finite) must not be lost
            // to max, which ignores it
            if r.is_nan() {
                return f64::NAN;
            }
            residual = residual.max(r);
        }
    }
    residual / s[0]
}
