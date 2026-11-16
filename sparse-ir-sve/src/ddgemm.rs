//! Matrix products in double-double ([`Df64`]) precision.
//!
//! nalgebra's generic product accumulates each output entry by full
//! double-double multiply-adds, one entry at a time. Here a product term
//! `a·b` is formed exactly to first order by FMA (`a_hi·b_hi` and its rounding
//! error, plus the cross terms `a_hi·b_lo + a_lo·b_hi`), and the sum is
//! accumulated as an unevaluated pair: the high parts by error-free
//! transformation (TwoSum), the low parts in plain `f64`, normalized once at
//! the end (the scheme of Ogita, Rump & Oishi's Dot2, SIAM J. Sci. Comput. 26,
//! 2005). The error stays at the double-double level, `O(k·u²)·Σ|a||b|` for
//! inner dimension `k` and `u = 2⁻⁵³`. The high and low parts are kept in
//! separate arrays so that the inner loop over output rows vectorizes.

use crate::Df64;
use nalgebra::DMatrix;

/// Output rows accumulated together; the accumulators (2 × 8 bytes each) and
/// the block of `a` rows stay in cache.
const ROW_BLOCK: usize = 256;

/// `a · b` in double-double precision, computed on the thread pool for large
/// products.
///
/// Each output entry is accumulated over the inner index in order by the same
/// operations whatever the threading, so the result does not depend on the
/// number of threads.
pub(crate) fn matmul_df64(a: &DMatrix<Df64>, b: &DMatrix<Df64>) -> DMatrix<Df64> {
    let (m, k) = a.shape();
    assert_eq!(k, b.nrows(), "matmul_df64: inner dimensions differ");
    let n = b.ncols();
    let mut out = vec![Df64::new(0.0); m * n];
    if m == 0 || n == 0 {
        return DMatrix::from_vec(m, n, out);
    }
    let (a_hi, a_lo): (Vec<f64>, Vec<f64>) = a.iter().map(|x| (x.hi(), x.lo())).unzip();
    let mut cols: Vec<&mut [Df64]> = out.chunks_mut(m).collect();
    crate::parallel::for_each_chunk(
        &mut cols,
        crate::cost::DF64_PRODUCT_TERM * m * k * n,
        |j0, chunk| {
            for (dj, col) in chunk.iter_mut().enumerate() {
                column(&a_hi, &a_lo, m, b.column(j0 + dj).as_slice(), col);
            }
        },
    );
    DMatrix::from_vec(m, n, out)
}

/// `out = A · b_col` for the `m × k` matrix `A` given by its column-major
/// high and low parts.
fn column(a_hi: &[f64], a_lo: &[f64], m: usize, b_col: &[Df64], out: &mut [Df64]) {
    let mut acc_hi = [0.0f64; ROW_BLOCK];
    let mut acc_lo = [0.0f64; ROW_BLOCK];
    for i0 in (0..m).step_by(ROW_BLOCK) {
        let len = ROW_BLOCK.min(m - i0);
        let (ch, cl) = (&mut acc_hi[..len], &mut acc_lo[..len]);
        ch.fill(0.0);
        cl.fill(0.0);
        for (l, b) in b_col.iter().enumerate() {
            let (bh, bl) = (b.hi(), b.lo());
            let ah = &a_hi[l * m + i0..l * m + i0 + len];
            let al = &a_lo[l * m + i0..l * m + i0 + len];
            for i in 0..len {
                // Product: ph + pe = a·b to first order, ph = fl(a_hi·b_hi)
                let ph = ah[i] * bh;
                let pe = ah[i].mul_add(bh, -ph);
                let pe = ah[i].mul_add(bl, pe);
                let pe = al[i].mul_add(bh, pe);
                // TwoSum of the high parts; errors go to the low part
                let s = ch[i] + ph;
                let t = s - ch[i];
                let e = (ch[i] - (s - t)) + (ph - t);
                ch[i] = s;
                cl[i] += e + pe;
            }
        }
        for (o, (&h, &l)) in out[i0..i0 + len].iter_mut().zip(ch.iter().zip(cl.iter())) {
            // FastTwoSum normalization: |h| ≥ |l| up to rounding
            let s = h + l;
            let e = l - (s - h);
            // (s, e) is normalized: s = fl(s + e)
            *o = Df64::new_full(s, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Df64 matrix with non-trivial low parts: entries are ratios of small
    /// integers computed in double-double.
    fn ratio_matrix(m: usize, n: usize, seed: f64) -> DMatrix<Df64> {
        DMatrix::from_fn(m, n, |i, j| {
            let num = Df64::from(((i * 7 + j * 13) % 17) as f64 - 8.0 + seed);
            num / Df64::from((i + 2 * j + 3) as f64)
        })
    }

    #[test]
    fn matches_nalgebra_product_to_double_double_accuracy() {
        for (m, k, n) in [
            (1, 1, 1),
            (5, 3, 4),
            (300, 97, 70),
            (257, 40, 3),
            (3, 513, 2),
        ] {
            let a = ratio_matrix(m, k, 0.25);
            let b = ratio_matrix(k, n, -0.5);
            let c = matmul_df64(&a, &b);
            let reference = &a * &b;
            let abs_a = a.map(|x| x.hi().abs());
            let abs_b = b.map(|x| x.hi().abs());
            let bound = &abs_a * &abs_b;
            for i in 0..m {
                for j in 0..n {
                    let err = (c[(i, j)] - reference[(i, j)]).hi().abs();
                    assert!(
                        err <= 1e-30 * bound[(i, j)].max(f64::MIN_POSITIVE),
                        "({m},{k},{n}) entry ({i},{j}): error {err:e}, Σ|a||b| = {:e}",
                        bound[(i, j)]
                    );
                }
            }
        }
    }

    #[test]
    fn handles_empty_dimensions() {
        let a = ratio_matrix(4, 0, 0.0);
        let b = ratio_matrix(0, 3, 0.0);
        let c = matmul_df64(&a, &b);
        assert_eq!(c.shape(), (4, 3));
        assert!(c.iter().all(|x| x.hi() == 0.0 && x.lo() == 0.0));
        assert_eq!(
            matmul_df64(&ratio_matrix(0, 2, 0.0), &ratio_matrix(2, 3, 0.0)).shape(),
            (0, 3)
        );
    }

    /// Large products are split over the columns on the thread pool; each
    /// column must equal the product of `a` with that column alone (computed
    /// on the calling thread).
    #[test]
    fn pooled_product_matches_column_by_column() {
        let (m, k, n) = (300, 97, 70);
        let a = ratio_matrix(m, k, 0.25);
        let b = ratio_matrix(k, n, -0.5);
        let c = matmul_df64(&a, &b);
        for j in 0..n {
            let c_j = matmul_df64(&a, &b.columns(j, 1).into_owned());
            assert_eq!(c.column(j), c_j.column(0), "column {j}");
        }
    }
}
