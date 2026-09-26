//! Utility functions for SVE computation

use crate::error::Error;
use crate::gauss::Rule;
use crate::interpolation1d::legendre_collocation_matrix;
use crate::kernel::SymmetryType;
use crate::numeric::CustomNumeric;
use crate::poly::{PiecewiseLegendrePoly, PiecewiseLegendrePolyVector};
use mdarray::DTensor;

/// Remove Gauss weights from SVD matrix
///
/// This function removes the square root of Gauss quadrature weights that were
/// applied before SVD computation. This is the inverse operation of
/// `DiscretizedKernel::apply_weights_for_sve()`.
///
/// # Arguments
///
/// * `matrix` - SVD result matrix (U or V^T)
/// * `weights` - Gauss quadrature weights
/// * `is_row` - If true, remove from rows; if false, remove from columns
///
/// # Returns
///
/// Matrix with weights removed
pub fn remove_weights<T: CustomNumeric>(
    matrix: &DTensor<T, 2>,
    weights: &[T],
    is_row: bool,
) -> DTensor<T, 2> {
    let mut result = matrix.clone();

    let shape = *result.shape();
    if is_row {
        // Remove weights from rows (for U matrix)
        for i in 0..shape.0 {
            let sqrt_weight = weights[i].sqrt();
            for j in 0..shape.1 {
                result[[i, j]] = result[[i, j]] / sqrt_weight;
            }
        }
    } else {
        // Remove weights from columns (for V matrix)
        for j in 0..shape.1 {
            let sqrt_weight = weights[j].sqrt();
            for i in 0..shape.0 {
                result[[i, j]] = result[[i, j]] / sqrt_weight;
            }
        }
    }

    result
}

/// Mirror half-domain segment boundaries onto the full domain
///
/// Maps `[0, s_1, ..., s_n]` to `[-s_n, ..., -s_1, -0, s_1, ..., s_n]`.
/// This is the knot layout that [`extend_to_full_domain`] gives the singular
/// functions of a centrosymmetric SVE, so a full-domain discretization built
/// from these segments uses the mirror images of the half-domain segments.
///
/// The input must start at zero, as [`crate::kernel::SVEHints`] documents
/// for the segments of centrosymmetric kernels.
pub(crate) fn mirror_segments_to_full_domain<T: CustomNumeric>(half: &[T]) -> Vec<T> {
    let mut full = Vec::with_capacity((2 * half.len()).saturating_sub(1));
    full.extend(half.iter().rev().map(|&s| -s));
    full.extend(half.iter().skip(1).copied());
    full
}

/// Extend polynomials from [0, xmax] to [-xmax, xmax] using symmetry
///
/// Following the C++ implementation logic from sve.rs.bak:856-888
///
/// # Arguments
///
/// * `polys` - Polynomials defined on [0, xmax]
/// * `symmetry` - Even or Odd symmetry type
/// * `xmax` - Maximum value of the domain
///
/// # Returns
///
/// Polynomials extended to full domain [-xmax, xmax], with `symm` set from
/// `symmetry` and `l` kept unchanged
///
/// # Mathematical Background
///
/// For Even symmetry (sign = +1): f(-x) = f(x)
/// For Odd symmetry (sign = -1): f(-x) = -f(x)
///
/// Legendre polynomial parity: P_n(-x) = (-1)^n P_n(x)
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if a polynomial does not start at x = 0
///   (the half domain [0, xmax]): its mirrored knots would decrease or
///   overlap
/// * The errors of [`PiecewiseLegendrePoly::new`] for the extended knots
pub fn extend_to_full_domain(
    polys: Vec<PiecewiseLegendrePoly>,
    symmetry: SymmetryType,
    _xmax: f64,
) -> Result<Vec<PiecewiseLegendrePoly>, Error> {
    let sign = symmetry.sign() as f64;
    let symm = symmetry.sign(); // Preserve symmetry: +1 for even, -1 for odd

    // Create poly_flip_x: alternating signs for Legendre polynomials
    // This accounts for P_n(-x) = (-1)^n P_n(x)
    let n_poly_coeffs = if !polys.is_empty() {
        polys[0].data.shape().0
    } else {
        return Ok(Vec::new());
    };

    let poly_flip_x: Vec<f64> = (0..n_poly_coeffs)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    polys
        .into_iter()
        .map(|poly| {
            if poly.knots[0] != 0.0 {
                return Err(Error::InvalidParameter {
                    name: "polys",
                    value: format!("a polynomial on [{:?}, {:?}]", poly.xmin, poly.xmax),
                    reason: "must be defined on [0, xmax] (the half domain)".to_string(),
                });
            }
            // Create full segments from this polynomial's knots: [-xmax, ..., 0, ..., xmax]
            let full_segments = mirror_segments_to_full_domain(&poly.knots);

            // Normalize by 1/sqrt(2) and convert to f64
            let pos_data = DTensor::<f64, 2>::from_fn(*poly.data.shape(), |idx| {
                poly.data[idx] / 2.0_f64.sqrt()
            });

            // Create negative part by reversing columns and applying signs
            let pos_shape = *pos_data.shape();
            let mut neg_data = DTensor::<f64, 2>::from_fn([pos_shape.0, pos_shape.1], |idx| {
                // Reverse column order: map column j to column (n_cols - 1 - j)
                let reversed_col = pos_shape.1 - 1 - idx[1];
                pos_data[[idx[0], reversed_col]]
            });

            // Apply poly_flip_x and sign to negative part
            for (i, &flip_sign) in poly_flip_x.iter().enumerate() {
                let coeff_sign = flip_sign * sign;
                for j in 0..pos_shape.1 {
                    neg_data[[i, j]] *= coeff_sign;
                }
            }

            // Combine negative and positive parts (concatenate along axis 1)
            let combined_data = DTensor::<f64, 2>::from_fn([pos_shape.0, pos_shape.1 * 2], |idx| {
                if idx[1] < pos_shape.1 {
                    neg_data[[idx[0], idx[1]]]
                } else {
                    pos_data[[idx[0], idx[1] - pos_shape.1]]
                }
            });

            // Create complete polynomial with full segments
            // Preserve symmetry from even/odd decomposition
            PiecewiseLegendrePoly::new(
                combined_data,
                full_segments,
                poly.l,
                None, // delta_x will be computed automatically
                symm, // Preserve symmetry: +1 for even, -1 for odd
            )
        })
        .collect()
}

/// Convert SVD matrix to piecewise Legendre polynomials
///
/// This function converts SVD results (U or V matrices) to piecewise Legendre
/// polynomial representation.
///
/// # Arguments
///
/// * `u_or_v` - SVD result matrix (rows = Gauss points, cols = singular values)
/// * `segments` - Segment boundaries
/// * `gauss_rule` - Gauss quadrature rule
/// * `n_gauss` - Number of Gauss points per segment
///
/// # Returns
///
/// Vector of piecewise Legendre polynomials. The `k`-th polynomial has `l = k`,
/// its column in `u_or_v`; for a centrosymmetric SVE that is the index within
/// the even or odd block, which `merge_results` renumbers.
///
/// # Errors
///
/// * [`Error::InvalidParameter`] if `segments` has fewer than 2 entries
/// * The errors of [`PiecewiseLegendrePoly::new`] for the segments
pub fn svd_to_polynomials<T: CustomNumeric>(
    u_or_v: &DTensor<T, 2>,
    segments: &[T],
    gauss_rule: &Rule<f64>,
    n_gauss: usize,
) -> Result<Vec<PiecewiseLegendrePoly>, Error> {
    if segments.len() < 2 {
        return Err(Error::InvalidParameter {
            name: "segments",
            value: format!("{} boundaries", segments.len()),
            reason: "must have at least 2 entries".to_string(),
        });
    }
    let n_segments = segments.len() - 1;
    let n_svals = u_or_v.shape().1;
    let n_rows = u_or_v.shape().0;

    // Reshape to 3D: (n_gauss, n_segments, n_svals)
    // Note: Due to QR early termination, u_or_v may have fewer rows than expected
    // We need to handle the case where row_idx exceeds the actual number of rows
    let mut tensor_3d = DTensor::<f64, 3>::zeros([n_gauss, n_segments, n_svals]);
    for i in 0..n_gauss {
        for j in 0..n_segments {
            for k in 0..n_svals {
                let row_idx = j * n_gauss + i;
                if row_idx < n_rows {
                    tensor_3d[[i, j, k]] = u_or_v[[row_idx, k]].to_f64();
                } else {
                    // If row_idx is out of bounds, set to zero (due to early termination)
                    tensor_3d[[i, j, k]] = 0.0;
                }
            }
        }
    }

    // Create Legendre collocation matrix
    let cmat = legendre_collocation_matrix(gauss_rule);

    // Transform to Legendre basis
    let cmat_shape = *cmat.shape();
    let mut u_data = DTensor::<f64, 3>::zeros([cmat_shape.0, n_segments, n_svals]);
    for j in 0..n_segments {
        for k in 0..n_svals {
            for i in 0..cmat_shape.0 {
                let mut sum = 0.0;
                for l in 0..n_gauss {
                    sum += cmat[[i, l]] * tensor_3d[[l, j, k]];
                }
                u_data[[i, j, k]] = sum;
            }
        }
    }

    // Apply segment length normalization: sqrt(0.5 * delta_segment)
    let mut dsegs = Vec::new();
    for i in 0..segments.len() - 1 {
        dsegs.push(segments[i + 1].to_f64() - segments[i].to_f64());
    }

    let u_data_shape = *u_data.shape();
    for j in 0..n_segments {
        let norm = (0.5 * dsegs[j]).sqrt();
        for i in 0..u_data_shape.0 {
            for k in 0..n_svals {
                u_data[[i, j, k]] *= norm;
            }
        }
    }

    // Create polynomials
    let mut polys = Vec::new();
    let knots: Vec<f64> = segments.iter().map(|&x| x.to_f64()).collect();
    let delta_x: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();

    for k in 0..n_svals {
        // Extract data for this singular value: (n_coeffs, n_segments)
        let u_data_shape = u_data.shape();
        let mut data = DTensor::<f64, 2>::zeros([u_data_shape.0, n_segments]);
        for i in 0..u_data_shape.0 {
            for j in 0..n_segments {
                data[[i, j]] = u_data[[i, j, k]];
            }
        }

        polys.push(PiecewiseLegendrePoly::new(
            data,
            knots.clone(),
            k as i32,
            Some(delta_x.clone()),
            0, // no symmetry
        )?);
    }

    Ok(polys)
}

// Note: legendre_collocation_matrix is imported from interpolation1d module
// Note: legendre_vandermonde is available in gauss module

/// Canonicalize singular function signs
///
/// Fix the gauge freedom in SVD by demanding u[l](xmax) >= 0, where xmax is
/// the right end of the domain of u[l]. The pairs (u[l], v[l]) are flipped
/// together.
pub(crate) fn canonicalize_signs(
    u_polys: Vec<PiecewiseLegendrePoly>,
    v_polys: Vec<PiecewiseLegendrePoly>,
) -> (Vec<PiecewiseLegendrePoly>, Vec<PiecewiseLegendrePoly>) {
    u_polys
        .into_iter()
        .zip(v_polys)
        .map(|(u, v)| {
            if u.evaluate(u.xmax) < 0.0 {
                (u.negated(), v.negated())
            } else {
                (u, v)
            }
        })
        .unzip()
}

/// Singular functions and values of one SVD block: `(u, s, v)`
///
/// Plain vectors, unlike [`PiecewiseLegendrePolyVector`], can be empty.
pub(crate) type SvdBlock = (
    Vec<PiecewiseLegendrePoly>,
    Vec<f64>,
    Vec<PiecewiseLegendrePoly>,
);

/// Merge even and odd SVE results
///
/// Either block may be empty (a `PiecewiseLegendrePolyVector` built through
/// its public field): truncating an SVE can remove all singular values of one
/// parity.
///
/// # Arguments
///
/// * `result_even` - (u, s, v) for even symmetry
/// * `result_odd` - (u, s, v) for odd symmetry
/// * `epsilon` - Accuracy parameter
///
/// # Returns
///
/// Merged SVEResult with singular values sorted in decreasing order. The `l`
/// of each returned polynomial is its position in this merged result, not
/// its index within the even or odd block.
///
/// # Errors
///
/// [`Error::EmptyInput`] if both blocks are empty: an SVE result has at least
/// one singular value
pub fn merge_results(
    result_even: (
        PiecewiseLegendrePolyVector,
        Vec<f64>,
        PiecewiseLegendrePolyVector,
    ),
    result_odd: (
        PiecewiseLegendrePolyVector,
        Vec<f64>,
        PiecewiseLegendrePolyVector,
    ),
    epsilon: f64,
) -> Result<crate::sve::SVEResult, Error> {
    let (u_even, s_even, v_even) = result_even;
    let (u_odd, s_odd, v_odd) = result_odd;
    merge_blocks(
        (u_even.polyvec, s_even, v_even.polyvec),
        (u_odd.polyvec, s_odd, v_odd.polyvec),
        epsilon,
    )
}

/// [`merge_results`] on plain vectors
///
/// The SVE strategy keeps each block in plain vectors until this merge, so
/// that a block emptied by truncation is merged like any other: keeping only
/// the largest singular value (`max_num_svals = Some(1)`, or `cutoff =
/// Some(1.0)`) leaves the odd block empty.
///
/// # Errors
///
/// [`Error::EmptyInput`] if both blocks are empty
pub(crate) fn merge_blocks(
    result_even: SvdBlock,
    result_odd: SvdBlock,
    epsilon: f64,
) -> Result<crate::sve::SVEResult, Error> {
    use crate::sve::SVEResult;

    let (u_even, s_even, v_even) = result_even;
    let (u_odd, s_odd, v_odd) = result_odd;
    if s_even.is_empty() && s_odd.is_empty() {
        return Err(Error::EmptyInput {
            name: "singular values",
        });
    }

    // Debug output
    // Create indices with symmetry info
    let mut indices: Vec<(usize, bool)> = Vec::new();
    for i in 0..s_even.len() {
        indices.push((i, true)); // true = even
    }
    for i in 0..s_odd.len() {
        indices.push((i, false)); // false = odd
    }

    // Sort by singular values (descending)
    indices.sort_by(|a, b| {
        let s_a = if a.1 { s_even[a.0] } else { s_odd[a.0] };
        let s_b = if b.1 { s_even[b.0] } else { s_odd[b.0] };
        s_b.partial_cmp(&s_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build sorted arrays.
    //
    // Each polynomial still carries `l = idx`, its column in the even or odd
    // SVD block (see `svd_to_polynomials`). Renumber it to its position in the
    // merged result, so that `l` is the index of the singular function for
    // every SVE result (convention-matched with SparseIR.jl v1
    // `postprocess(::CentrosymmSVE)`, which sets `l = i - 1` after sorting).
    // `PiecewiseLegendreFT` takes the parity (-1)^l from it; that equals
    // `symm` because the even and odd singular values interlace.
    let mut u_polys = Vec::new();
    let mut v_polys = Vec::new();
    let mut s_sorted = Vec::new();

    for (l, (idx, is_even)) in indices.into_iter().enumerate() {
        let (u_block, s_block, v_block) = if is_even {
            (&u_even, &s_even, &v_even)
        } else {
            (&u_odd, &s_odd, &v_odd)
        };
        let l = l as i32;
        u_polys.push(PiecewiseLegendrePoly {
            l,
            ..u_block[idx].clone()
        });
        v_polys.push(PiecewiseLegendrePoly {
            l,
            ..v_block[idx].clone()
        });
        s_sorted.push(s_block[idx]);
    }

    // Canonicalize signs: ensure u[l](xmax) >= 0. The merged blocks are not
    // empty (checked above) and share their knots, so the vectors are built
    // through their field; the field becomes private in part 7.
    let (canonical_u, canonical_v) = canonicalize_signs(u_polys, v_polys);
    SVEResult::new(
        PiecewiseLegendrePolyVector {
            polyvec: canonical_u,
        },
        s_sorted,
        PiecewiseLegendrePolyVector {
            polyvec: canonical_v,
        },
        epsilon,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_weights() {
        let matrix = DTensor::<f64, 2>::from_fn([2, 2], |idx| (idx[0] * 2 + idx[1] + 1) as f64);
        let weights = vec![1.0, 4.0];

        let result = remove_weights(&matrix, &weights, true);

        // Both U and V: remove from rows (Gauss points)
        // First row: [1.0, 2.0] / sqrt(1.0) = [1.0, 2.0]
        // Second row: [3.0, 4.0] / sqrt(4.0) = [1.5, 2.0]
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((result[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((result[[1, 0]] - 1.5).abs() < 1e-10);
        assert!((result[[1, 1]] - 2.0).abs() < 1e-10);
    }
}
