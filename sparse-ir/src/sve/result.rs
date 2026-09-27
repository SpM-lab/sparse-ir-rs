//! SVE result container

use crate::error::{Error, require_nonzero_size, require_threshold};
use crate::poly::PiecewiseLegendrePolyVector;

/// Result of Singular Value Expansion computation
#[derive(Debug, Clone)]
pub struct SVEResult {
    /// Left singular functions (u)
    pub(crate) u: PiecewiseLegendrePolyVector,
    /// Singular values in non-increasing order
    pub(crate) s: Vec<f64>,
    /// Right singular functions (v)
    pub(crate) v: PiecewiseLegendrePolyVector,
    /// Accuracy parameter used for computation
    pub(crate) epsilon: f64,
}

impl SVEResult {
    /// Left singular functions, one per singular value
    pub fn u(&self) -> &PiecewiseLegendrePolyVector {
        &self.u
    }

    /// Singular values, in non-increasing order
    pub fn s(&self) -> &[f64] {
        &self.s
    }

    /// Right singular functions, one per singular value
    pub fn v(&self) -> &PiecewiseLegendrePolyVector {
        &self.v
    }

    /// Accuracy the expansion was computed to
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Build an SVE from a kernel matrix discretized on the Gauss points of
    /// `segments_x` x `segments_y`
    ///
    /// `matrix[i][j]` is `sqrt(w_x[i]) * K(x[i], y[j]) * sqrt(w_y[j])` for the
    /// piecewise Gauss-Legendre rules `gauss_x` and `gauss_y` of `n_gauss`
    /// points per segment, so `matrix` has `n_gauss * n_segments` rows and
    /// columns. The weights are divided out of the singular vectors before
    /// they become piecewise Legendre polynomials.
    ///
    /// # Errors
    ///
    /// * The errors of [`compute_svd_dtensor`](crate::tsvd::compute_svd_dtensor):
    ///   [`Error::EmptyInput`], [`Error::NonFiniteInput`] and
    ///   [`Error::DecompositionFailed`]
    /// * [`Error::InvalidParameter`] if `matrix` does not have one row
    ///   (column) per Gauss point of `segments_x` (`segments_y`), or if a
    ///   segment array has fewer than 2 entries
    /// * The errors of [`Self::new`], in particular [`Error::EmptyInput`] for
    ///   a matrix of rank 0, which has no singular functions
    pub fn from_discretized_matrix<T: crate::numeric::CustomNumeric + 'static>(
        matrix: &mdarray::DTensor<T, 2>,
        gauss_x: &crate::gauss::Rule<T>,
        gauss_y: &crate::gauss::Rule<T>,
        segments_x: &[f64],
        segments_y: &[f64],
        n_gauss: usize,
        epsilon: f64,
    ) -> Result<Self, Error> {
        require_gauss_point_count(matrix.shape().0, segments_x, n_gauss, "rows")?;
        require_gauss_point_count(matrix.shape().1, segments_y, n_gauss, "columns")?;

        let (u, s, v) = crate::tsvd::compute_svd_dtensor(matrix)?;

        // The matrix carries the weights: divide them out of the singular
        // vectors before they become polynomials.
        let u_unweighted = crate::sve::utils::remove_weights(&u, gauss_x.w.as_slice(), true);
        let v_unweighted = crate::sve::utils::remove_weights(&v, gauss_y.w.as_slice(), true);

        let u_f64 = mdarray::DTensor::<f64, 2>::from_fn(*u_unweighted.shape(), |idx| {
            u_unweighted[idx].to_f64()
        });
        let v_f64 = mdarray::DTensor::<f64, 2>::from_fn(*v_unweighted.shape(), |idx| {
            v_unweighted[idx].to_f64()
        });

        let gauss_rule_f64 = crate::gauss::legendre::<f64>(n_gauss);
        let u_polys =
            crate::sve::utils::svd_to_polynomials(&u_f64, segments_x, &gauss_rule_f64, n_gauss)?;
        let v_polys =
            crate::sve::utils::svd_to_polynomials(&v_f64, segments_y, &gauss_rule_f64, n_gauss)?;

        let s_f64: Vec<f64> = s.iter().map(|sv| sv.to_f64()).collect();

        // A matrix of rank 0 has no singular functions. The vectors are built
        // through the field (`PiecewiseLegendrePolyVector::new` rejects an
        // empty vector), so that `Self::new` reports the empty `s`.
        Self::new(
            PiecewiseLegendrePolyVector { polyvec: u_polys },
            s_f64,
            PiecewiseLegendrePolyVector { polyvec: v_polys },
            epsilon,
        )
    }

    /// [`Self::from_discretized_matrix`] for a centrosymmetric kernel, whose
    /// even and odd parts are discretized on the half domains
    /// `segments_x` x `segments_y` and extended to `[-xmax, xmax]` and
    /// `[-ymax, ymax]`
    ///
    /// A block of rank 0 (the odd part of a kernel that is even in y, say)
    /// contributes no singular function; only both blocks being empty is an
    /// error.
    ///
    /// # Errors
    ///
    /// * The errors of [`Self::from_discretized_matrix`] for each block,
    ///   except that an empty block is accepted
    /// * The errors of
    ///   [`extend_to_full_domain`](crate::sve::utils::extend_to_full_domain)
    ///   and of [`merge_results`](crate::sve::utils::merge_results), in
    ///   particular [`Error::EmptyInput`] if both blocks are empty
    #[allow(clippy::too_many_arguments)]
    pub fn from_discretized_matrices_centrosymmetric(
        even: &mdarray::DTensor<f64, 2>,
        odd: &mdarray::DTensor<f64, 2>,
        gauss_x: &crate::gauss::Rule<f64>,
        gauss_y: &crate::gauss::Rule<f64>,
        segments_x: &[f64],
        segments_y: &[f64],
        n_gauss: usize,
        xmax: f64,
        ymax: f64,
        epsilon: f64,
    ) -> Result<Self, Error> {
        use crate::kernel::SymmetryType;
        use crate::sve::utils::{extend_to_full_domain, merge_results, svd_to_polynomials};

        for matrix in [even, odd] {
            require_gauss_point_count(matrix.shape().0, segments_x, n_gauss, "rows")?;
            require_gauss_point_count(matrix.shape().1, segments_y, n_gauss, "columns")?;
        }

        let gauss_rule_f64 = crate::gauss::legendre::<f64>(n_gauss);
        let block = |matrix: &mdarray::DTensor<f64, 2>, symmetry: SymmetryType| {
            let (u, s, v) = crate::tsvd::compute_svd_dtensor(matrix)?;
            let u_unweighted = crate::sve::utils::remove_weights(&u, gauss_x.w.as_slice(), true);
            let v_unweighted = crate::sve::utils::remove_weights(&v, gauss_y.w.as_slice(), true);
            let u_polys = svd_to_polynomials(&u_unweighted, segments_x, &gauss_rule_f64, n_gauss)?;
            let v_polys = svd_to_polynomials(&v_unweighted, segments_y, &gauss_rule_f64, n_gauss)?;
            let u_full = extend_to_full_domain(u_polys, symmetry, xmax)?;
            let v_full = extend_to_full_domain(v_polys, symmetry, ymax)?;
            Ok::<_, Error>((
                PiecewiseLegendrePolyVector { polyvec: u_full },
                s,
                PiecewiseLegendrePolyVector { polyvec: v_full },
            ))
        };

        let result_even = block(even, SymmetryType::Even)?;
        let result_odd = block(odd, SymmetryType::Odd)?;
        merge_results(result_even, result_odd, epsilon)
    }

    /// Create a new SVEResult
    ///
    /// # Errors
    ///
    /// * [`Error::EmptyInput`] if `s` is empty
    /// * [`Error::InvalidParameter`] if `u` or `v` does not have one function
    ///   per singular value, a singular value is not positive, the singular
    ///   values are not non-increasing, or `epsilon` is not in [0, 1)
    /// * [`Error::NonFiniteInput`] if a singular value is NaN or infinite
    pub fn new(
        u: PiecewiseLegendrePolyVector,
        s: Vec<f64>,
        v: PiecewiseLegendrePolyVector,
        epsilon: f64,
    ) -> Result<Self, Error> {
        let result = Self { u, s, v, epsilon };
        result.check()?;
        Ok(result)
    }

    /// Check the invariants that [`Self::new`] guarantees. The fields are
    /// public, so [`Self::part`] checks them again.
    fn check(&self) -> Result<(), Error> {
        let n = self.s.len();
        if n == 0 {
            return Err(Error::EmptyInput { name: "s" });
        }
        for (name, funcs) in [("u", &self.u), ("v", &self.v)] {
            let len = funcs.get_polys().len();
            if len != n {
                return Err(Error::InvalidParameter {
                    name,
                    value: format!("{len} functions"),
                    reason: format!("must have one function per singular value ({n})"),
                });
            }
        }
        for (i, &x) in self.s.iter().enumerate() {
            if !x.is_finite() {
                return Err(Error::NonFiniteInput {
                    name: "s",
                    index: vec![i],
                    value: x,
                });
            }
            if x <= 0.0 {
                return Err(Error::InvalidParameter {
                    name: "s",
                    value: format!("{x:?} at index {i}"),
                    reason: "singular values must be positive".to_string(),
                });
            }
            if i > 0 && x > self.s[i - 1] {
                return Err(Error::InvalidParameter {
                    name: "s",
                    value: format!("{x:?} at index {i}, after {:?}", self.s[i - 1]),
                    reason: "singular values must be non-increasing".to_string(),
                });
            }
        }
        require_threshold("epsilon", Some(self.epsilon))
    }

    /// Extract a subset of the SVE result based on epsilon and max_size
    ///
    /// # Arguments
    ///
    /// * `eps` - Relative threshold for singular values (default: self.epsilon)
    /// * `max_size` - Maximum number of singular values to keep
    ///
    /// # Returns
    ///
    /// Tuple of (u_subset, s_subset, v_subset)
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidParameter`] if `eps` is not in [0, 1) (0 keeps every
    ///   singular value) or `max_size` is `Some(0)`
    /// * The errors of [`Self::new`] and [`PiecewiseLegendrePolyVector::new`]
    ///   if the (public) fields break their invariants
    pub fn part(
        &self,
        eps: Option<f64>,
        max_size: Option<usize>,
    ) -> Result<
        (
            PiecewiseLegendrePolyVector,
            Vec<f64>,
            PiecewiseLegendrePolyVector,
        ),
        Error,
    > {
        self.check()?;
        require_threshold("eps", eps)?;
        require_nonzero_size("max_size", max_size)?;
        let eps = eps.unwrap_or(self.epsilon);
        let threshold = eps * self.s[0];

        let mut cut = 0;
        for &val in self.s.iter() {
            if val >= threshold {
                cut += 1;
            } else {
                break;
            }
        }

        if let Some(max) = max_size {
            cut = cut.min(max);
        }

        // Extract subsets
        let u_part = PiecewiseLegendrePolyVector::new(self.u.get_polys()[..cut].to_vec())?;
        let s_part = self.s[..cut].to_vec();
        let v_part = PiecewiseLegendrePolyVector::new(self.v.get_polys()[..cut].to_vec())?;

        Ok((u_part, s_part, v_part))
    }
}

/// The rows (columns) of a discretized kernel matrix are the Gauss points of
/// the segments: `n_gauss` per segment.
///
/// # Errors
/// * [`Error::InvalidParameter`] if `segments` has fewer than 2 entries or
///   `len` is not `n_gauss * (segments.len() - 1)`
fn require_gauss_point_count(
    len: usize,
    segments: &[f64],
    n_gauss: usize,
    what: &'static str,
) -> Result<(), Error> {
    if segments.len() < 2 {
        return Err(Error::InvalidParameter {
            name: "segments",
            value: format!("{} boundaries", segments.len()),
            reason: "must have at least 2 entries".to_string(),
        });
    }
    let expected = n_gauss * (segments.len() - 1);
    if len != expected {
        return Err(Error::InvalidParameter {
            name: "matrix",
            value: format!("{len} {what}"),
            reason: format!("must have one per Gauss point of the segments ({expected})"),
        });
    }
    Ok(())
}
