//! Kernels K(x, y) and the traits the singular value expansion needs
//!
//! The SVE expands a kernel in the Fredholm integral equation of the first
//! kind
//!
//! u(x) = integral of K(x, y) v(y) dy
//!
//! where x ∈ [xmin, xmax] and y ∈ [ymin, ymax]. A kernel implements
//! [`AbstractKernel`] (its values) and [`SVEKernel`] (its domain and the
//! discretization hints of its SVE). A centrosymmetric kernel implements
//! [`ReducedKernel`] as well, whose SVE is block-diagonalized into the even
//! and odd reduced kernels; any other kernel implements [`ExpandableKernel`]
//! (no methods), whose SVE is taken over the full domain. Either is what
//! [`compute_sve`](crate::sve::compute_sve) accepts.

use crate::error::Error;
use crate::numeric::CustomNumeric;
use crate::sve::strategy::{CentrosymmSVE, NonCentrosymmSVE, SVEStrategy};
use std::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryType {
    Even,
    Odd,
}

impl SymmetryType {
    pub fn sign(self) -> i32 {
        match self {
            SymmetryType::Even => 1,
            SymmetryType::Odd => -1,
        }
    }
}

/// Trait for SVE (Singular Value Expansion) hints
///
/// Provides discretization hints for singular value expansion of a given kernel.
/// This includes segment information and numerical parameters for efficient computation.
pub trait SVEHints<T>: Debug + Send + Sync
where
    T: Copy + Debug + Send + Sync,
{
    /// Segment boundaries on x: strictly increasing from `xmin` to `xmax`
    ///
    /// The SVE of a centrosymmetric kernel ([`ReducedKernel`]) discretizes
    /// the half `[0, xmax]`: the boundaries in `(0, xmax]` with 0 added. The
    /// mirror images of these are its boundaries on `[-xmax, 0]`.
    fn segments_x(&self) -> Vec<T>;

    /// Segment boundaries on y: strictly increasing from `ymin` to `ymax`
    ///
    /// The SVE of a centrosymmetric kernel ([`ReducedKernel`]) discretizes
    /// the half `[0, ymax]`, as for [`Self::segments_x`].
    fn segments_y(&self) -> Vec<T>;

    /// Get the number of singular values hint
    fn nsvals(&self) -> usize;

    /// Get the number of Gauss points for quadrature
    fn ngauss(&self) -> usize;
}

/// The domain of a kernel and the discretization hints of its SVE
///
/// This is what [`crate::sve::compute_sve`] needs of a kernel besides its
/// values ([`AbstractKernel`]) and its strategy ([`ExpandableKernel`]).
pub trait SVEKernel: AbstractKernel {
    /// Associated type for SVE hints
    type SVEHintsType<T>: SVEHints<T> + Clone
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;

    /// Get the upper bound of the x domain
    fn xmax(&self) -> f64;

    /// Get the upper bound of the y domain
    fn ymax(&self) -> f64;

    /// Lower bound of the x domain. Defaults to the symmetric case `-xmax`.
    fn xmin(&self) -> f64 {
        -self.xmax()
    }

    /// Lower bound of the y domain. Defaults to the symmetric case `-ymax`.
    fn ymin(&self) -> f64 {
        -self.ymax()
    }

    /// Create SVE hints for this kernel type.
    ///
    /// Provides discretization hints for singular value expansion computation.
    /// The hints include segment information and numerical parameters optimized
    /// for the specific kernel type.
    ///
    /// @param epsilon Target accuracy for the SVE computation
    /// @return SVE hints specific to this kernel type
    fn sve_hints<T>(&self, epsilon: f64) -> Self::SVEHintsType<T>
    where
        T: Copy + Debug + Send + Sync + CustomNumeric + 'static;
}

/// Trait for general kernels (both centrosymmetric and non-centrosymmetric)
///
/// This trait provides the basic interface for computing kernel values.
/// Centrosymmetric kernels implement [`ReducedKernel`] as well, which
/// provides additional optimizations.
pub trait AbstractKernel: Send + Sync {
    /// Compute the kernel value K(x, y) with high precision
    ///
    /// # Arguments
    ///
    /// * `x` - The x coordinate (typically in [xmin, xmax])
    /// * `y` - The y coordinate (typically in [ymin, ymax])
    ///
    /// # Returns
    ///
    /// The kernel value K(x, y)
    fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T;

    /// Whether the kernel is centrosymmetric
    ///
    /// The SVE exploits the symmetry of the kernels that implement
    /// [`ReducedKernel`]. Only `compute_sve_general`, which takes any kernel,
    /// reads this, to keep the dense decomposition for a centrosymmetric
    /// kernel.
    #[deprecated(note = "a kernel is centrosymmetric when it implements ReducedKernel")]
    #[doc(hidden)]
    fn is_centrosymmetric(&self) -> bool {
        false
    }
}

/// Trait for centrosymmetric kernels, through their reduced kernels
///
/// Centrosymmetric kernels satisfy K(x, y) = K(-x, -y) on a symmetric domain
/// and can be decomposed into even and odd components for efficient
/// computation.
pub trait ReducedKernel: AbstractKernel {
    /// Compute the reduced kernel value
    ///
    /// K_red(x, y) = K(x, y) + sign * K(x, -y)
    /// where sign = 1 for even symmetry and sign = -1 for odd symmetry
    fn compute_reduced<T: CustomNumeric + Copy + Debug>(
        &self,
        x: T,
        y: T,
        symmetry: SymmetryType,
    ) -> T;
}

/// The kernels whose SVE [`compute_sve`](crate::sve::compute_sve) computes
///
/// Implemented for every [`ReducedKernel`], whose SVE is block-diagonalized
/// into the even and odd reduced kernels. Other kernels implement it with
/// no methods (`impl ExpandableKernel for MyKernel {}`; Rust cannot provide
/// it for every kernel that is not a [`ReducedKernel`]); their SVE is taken
/// over the full domain
/// `[xmin, xmax] × [ymin, ymax]` and their singular functions have no
/// definite parity.
pub trait ExpandableKernel: SVEKernel + Clone + 'static {
    /// The SVE strategy of the kernel: the even/odd blocks for a
    /// [`ReducedKernel`], otherwise the full domain
    ///
    /// # Errors
    ///
    /// The errors of [`NonCentrosymmSVE::new`] or [`CentrosymmSVE::new`]
    #[doc(hidden)]
    fn sve_strategy<T>(&self, epsilon: f64) -> Result<Box<dyn SVEStrategy<T>>, Error>
    where
        T: CustomNumeric + Send + Sync + Clone + 'static,
    {
        Ok(Box::new(NonCentrosymmSVE::new(self.clone(), epsilon)?))
    }
}

impl<K> ExpandableKernel for K
where
    K: ReducedKernel + SVEKernel + Clone + 'static,
{
    fn sve_strategy<T>(&self, epsilon: f64) -> Result<Box<dyn SVEStrategy<T>>, Error>
    where
        T: CustomNumeric + Send + Sync + Clone + 'static,
    {
        Ok(Box::new(CentrosymmSVE::new(self.clone(), epsilon)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple non-centrosymmetric kernel for testing AbstractKernel trait
    // K(x, y) = x + y (clearly non-centrosymmetric: K(-x, -y) = -x - y = -(x + y) != x + y = K(x, y))
    #[derive(Debug, Clone, Copy)]
    struct SimpleNonCentrosymmKernel {
        lambda: f64,
    }

    impl SimpleNonCentrosymmKernel {
        fn new(lambda: f64) -> Self {
            Self { lambda }
        }
    }

    impl AbstractKernel for SimpleNonCentrosymmKernel {
        fn compute<T: CustomNumeric + Copy + Debug>(&self, x: T, y: T) -> T {
            x + y
        }

        fn is_centrosymmetric(&self) -> bool {
            false
        }
    }

    #[test]
    fn test_noncentrosymm_kernel_compute() {
        let kernel = SimpleNonCentrosymmKernel::new(10.0);

        // Test basic computation: K(x, y) = x + y
        let result1: f64 = kernel.compute(0.5, 0.5);
        assert!((result1 - 1.0).abs() < 1e-10);

        let result2: f64 = kernel.compute(-0.5, 0.5);
        assert!((result2 - 0.0).abs() < 1e-10);

        // Verify non-centrosymmetry: K(-x, -y) != K(x, y)
        // K(0.5, 0.5) = 1.0, K(-0.5, -0.5) = -1.0
        let result3: f64 = kernel.compute(0.5, 0.5);
        let result4: f64 = kernel.compute(-0.5, -0.5);
        assert!((result3 - result4).abs() > 1e-10); // Should be different (1.0 vs -1.0)
    }
}
