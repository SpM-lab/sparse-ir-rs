//! Utility functions for SVE computation

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
/// Polynomials extended to full domain [-xmax, xmax]
///
/// # Mathematical Background
///
/// For Even symmetry (sign = +1): f(-x) = f(x)
/// For Odd symmetry (sign = -1): f(-x) = -f(x)
///
/// Legendre polynomial parity: P_n(-x) = (-1)^n P_n(x)
pub fn extend_to_full_domain(
    polys: Vec<PiecewiseLegendrePoly>,
    symmetry: SymmetryType,
    _xmax: f64,
) -> Vec<PiecewiseLegendrePoly> {
    let sign = symmetry.sign() as f64;
    let symm = symmetry.sign(); // Preserve symmetry: +1 for even, -1 for odd

    // Create poly_flip_x: alternating signs for Legendre polynomials
    // This accounts for P_n(-x) = (-1)^n P_n(x)
    let n_poly_coeffs = if !polys.is_empty() {
        polys[0].data.shape().0
    } else {
        return Vec::new();
    };

    let poly_flip_x: Vec<f64> = (0..n_poly_coeffs)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();

    polys
        .into_iter()
        .map(|poly| {
            // Create full segments from this polynomial's knots: [-xmax, ..., 0, ..., xmax]
            let knots_pos = &poly.knots;
            let mut full_segments = Vec::new();
            for i in (0..knots_pos.len()).rev() {
                full_segments.push(-knots_pos[i]);
            }
            for i in 1..knots_pos.len() {
                full_segments.push(knots_pos[i]);
            }

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
/// Vector of piecewise Legendre polynomials
pub fn svd_to_polynomials<T: CustomNumeric>(
    u_or_v: &DTensor<T, 2>,
    segments: &[T],
    gauss_rule: &Rule<f64>,
    n_gauss: usize,
) -> Vec<PiecewiseLegendrePoly> {
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
        ));
    }

    polys
}

// Note: legendre_collocation_matrix is imported from interpolation1d module
// Note: legendre_vandermonde is available in gauss module

/// Canonicalize singular function signs
///
/// Fix the gauge freedom in SVD by demanding u[l](xmax) > 0.
/// This ensures consistent signs across different implementations.
///
/// # Arguments
///
/// * `u_polys` - Left singular functions
/// * `v_polys` - Right singular functions
/// * `xmax` - Maximum value to evaluate at (typically 1.0)
fn canonicalize_signs(
    u_polys: PiecewiseLegendrePolyVector,
    v_polys: PiecewiseLegendrePolyVector,
    xmax: f64,
) -> (PiecewiseLegendrePolyVector, PiecewiseLegendrePolyVector) {
    let u_vec = u_polys.get_polys();
    let v_vec = v_polys.get_polys();

    let mut new_u_vec = Vec::new();
    let mut new_v_vec = Vec::new();

    for i in 0..u_vec.len().min(v_vec.len()) {
        // Evaluate u[i] at xmax
        let u_at_xmax = u_vec[i].evaluate(xmax);

        if u_at_xmax < 0.0 {
            // Flip sign of both u and v
            let u_data_flipped =
                DTensor::<f64, 2>::from_fn(*u_vec[i].data.shape(), |idx| -u_vec[i].data[idx]);
            let v_data_flipped =
                DTensor::<f64, 2>::from_fn(*v_vec[i].data.shape(), |idx| -v_vec[i].data[idx]);

            new_u_vec.push(PiecewiseLegendrePoly::new(
                u_data_flipped,
                u_vec[i].knots.clone(),
                u_vec[i].l,
                Some(u_vec[i].delta_x.clone()),
                u_vec[i].symm,
            ));
            new_v_vec.push(PiecewiseLegendrePoly::new(
                v_data_flipped,
                v_vec[i].knots.clone(),
                v_vec[i].l,
                Some(v_vec[i].delta_x.clone()),
                v_vec[i].symm,
            ));
        } else {
            // Keep as is
            new_u_vec.push(u_vec[i].clone());
            new_v_vec.push(v_vec[i].clone());
        }
    }

    (
        PiecewiseLegendrePolyVector::new(new_u_vec),
        PiecewiseLegendrePolyVector::new(new_v_vec),
    )
}

/// Merge even and odd SVE results
///
/// # Arguments
///
/// * `result_even` - (u, s, v) for even symmetry
/// * `result_odd` - (u, s, v) for odd symmetry
/// * `epsilon` - Accuracy parameter
///
/// # Returns
///
/// Merged SVEResult with singular values sorted in decreasing order
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
) -> crate::sve::SVEResult {
    use crate::sve::SVEResult;

    let (u_even, s_even, v_even) = result_even;
    let (u_odd, s_odd, v_odd) = result_odd;

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

    // Build sorted arrays
    let mut u_polys = Vec::new();
    let mut v_polys = Vec::new();
    let mut s_sorted = Vec::new();

    for (idx, is_even) in indices {
        if is_even {
            u_polys.push(u_even.get_polys()[idx].clone());
            v_polys.push(v_even.get_polys()[idx].clone());
            s_sorted.push(s_even[idx]);
        } else {
            u_polys.push(u_odd.get_polys()[idx].clone());
            v_polys.push(v_odd.get_polys()[idx].clone());
            s_sorted.push(s_odd[idx]);
        }
    }

    // Canonicalize signs: ensure u[l](1) > 0
    let (canonical_u, canonical_v) = canonicalize_signs(
        PiecewiseLegendrePolyVector::new(u_polys),
        PiecewiseLegendrePolyVector::new(v_polys),
        1.0,
    );

    SVEResult::new(canonical_u, s_sorted, canonical_v, epsilon)
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
