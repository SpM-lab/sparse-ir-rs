//! SVE result container

use crate::error::{Error, require_nonzero_size, require_threshold};
use crate::poly::PiecewiseLegendrePolyVector;

/// Result of Singular Value Expansion computation
#[derive(Debug, Clone)]
pub struct SVEResult {
    /// Left singular functions (u)
    pub u: PiecewiseLegendrePolyVector,
    /// Singular values in non-increasing order
    pub s: Vec<f64>,
    /// Right singular functions (v)
    pub v: PiecewiseLegendrePolyVector,
    /// Accuracy parameter used for computation
    pub epsilon: f64,
}

impl SVEResult {
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
