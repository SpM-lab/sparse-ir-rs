//! SVE result container

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
    pub fn new(
        u: PiecewiseLegendrePolyVector,
        s: Vec<f64>,
        v: PiecewiseLegendrePolyVector,
        epsilon: f64,
    ) -> Self {
        Self { u, s, v, epsilon }
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
    pub fn part(
        &self,
        eps: Option<f64>,
        max_size: Option<usize>,
    ) -> (
        PiecewiseLegendrePolyVector,
        Vec<f64>,
        PiecewiseLegendrePolyVector,
    ) {
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
        let u_part = PiecewiseLegendrePolyVector::new(self.u.get_polys()[..cut].to_vec());
        let s_part = self.s[..cut].to_vec();
        let v_part = PiecewiseLegendrePolyVector::new(self.v.get_polys()[..cut].to_vec());

        (u_part, s_part, v_part)
    }
}
