//! SVE computation strategies

use crate::gauss::{Rule, legendre_generic};
use crate::kernel::{AbstractKernel, CentrosymmKernel, KernelProperties, SVEHints, SymmetryType};
use crate::kernelmatrix::{matrix_from_gauss_noncentrosymmetric, matrix_from_gauss_with_segments};
use crate::numeric::CustomNumeric;
use crate::poly::PiecewiseLegendrePolyVector;
use mdarray::DTensor;

use super::result::SVEResult;
use super::utils::{extend_to_full_domain, merge_results, remove_weights, svd_to_polynomials};

/// Trait for SVE computation strategies
pub trait SVEStrategy<T: CustomNumeric> {
    /// Compute the discretized matrices for SVD
    fn matrices(&self) -> Vec<DTensor<T, 2>>;

    /// Post-process SVD results to create SVEResult
    fn postprocess(
        &self,
        u_list: Vec<DTensor<T, 2>>,
        s_list: Vec<Vec<T>>,
        v_list: Vec<DTensor<T, 2>>,
    ) -> SVEResult;
}

/// Sampling-based SVE computation
///
/// This is the general SVE computation strategy that works with discretized kernels.
/// It does NOT know about symmetry - it just processes a given discretized kernel matrix.
///
/// # Responsibility
///
/// - Remove weights from SVD results
/// - Convert to polynomials on the domain specified by segments
/// - Domain extension is caller's responsibility
pub struct SamplingSVE<T>
where
    T: CustomNumeric + Send + Sync + 'static,
{
    segments_x: Vec<T>,
    segments_y: Vec<T>,
    gauss_x: Rule<T>,
    gauss_y: Rule<T>,
    #[allow(dead_code)]
    epsilon: f64,
    n_gauss: usize,
}

impl<T> SamplingSVE<T>
where
    T: CustomNumeric + Send + Sync + 'static,
{
    /// Create a new SamplingSVE
    ///
    /// This takes only the geometric information needed for polynomial conversion,
    /// not the kernel itself.
    pub fn new(
        segments_x: Vec<T>,
        segments_y: Vec<T>,
        gauss_x: Rule<T>,
        gauss_y: Rule<T>,
        epsilon: f64,
        n_gauss: usize,
    ) -> Self {
        Self {
            segments_x,
            segments_y,
            gauss_x,
            gauss_y,
            epsilon,
            n_gauss,
        }
    }

    /// Post-process a single SVD result to create polynomials
    ///
    /// This converts SVD results to piecewise Legendre polynomials
    /// on the domain specified by segments (e.g., [0, xmax] for reduced kernels).
    pub fn postprocess_single(
        &self,
        u: &DTensor<T, 2>,
        s: &[T],
        v: &DTensor<T, 2>,
    ) -> (
        PiecewiseLegendrePolyVector,
        Vec<f64>,
        PiecewiseLegendrePolyVector,
    ) {
        // 1. Remove weights
        // Both U and V have rows corresponding to Gauss points, so is_row=true for both
        let u_unweighted = remove_weights(u, self.gauss_x.w.as_slice(), true);
        let v_unweighted = remove_weights(v, self.gauss_y.w.as_slice(), true);

        // 2. Convert to polynomials
        let gauss_rule_f64 = legendre_generic::<f64>(self.n_gauss);
        let u_polys = svd_to_polynomials(
            &u_unweighted,
            &self.segments_x,
            &gauss_rule_f64,
            self.n_gauss,
        );
        let v_polys = svd_to_polynomials(
            &v_unweighted,
            &self.segments_y,
            &gauss_rule_f64,
            self.n_gauss,
        );

        // Note: No domain extension here - that's the caller's responsibility
        (
            PiecewiseLegendrePolyVector::new(u_polys),
            s.iter().map(|&x| x.to_f64()).collect(),
            PiecewiseLegendrePolyVector::new(v_polys),
        )
    }
}

/// Centrosymmetric SVE computation
///
/// Exploits even/odd symmetry for efficient computation.
/// This manages the symmetry: creating reduced kernels, extending to full domain, and merging.
pub struct CentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + 'static,
    K: CentrosymmKernel + KernelProperties,
{
    kernel: K,
    epsilon: f64,
    hints: K::SVEHintsType<T>,
    #[allow(dead_code)]
    n_gauss: usize,

    // Geometric information (positive domain [0, xmax])
    #[allow(dead_code)]
    segments_x: Vec<T>,
    #[allow(dead_code)]
    segments_y: Vec<T>,
    gauss_x: Rule<T>,
    gauss_y: Rule<T>,

    // The general SVE processor (no symmetry knowledge)
    sampling_sve: SamplingSVE<T>,
}

impl<T, K> CentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: CentrosymmKernel + KernelProperties + Clone,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    /// Create a new CentrosymmSVE
    pub fn new(kernel: K, epsilon: f64) -> Self {
        let hints = kernel.sve_hints::<T>(epsilon);

        // Get segments for positive domain [0, xmax]
        let segments_x = hints.segments_x();
        let segments_y = hints.segments_y();
        let n_gauss = hints.ngauss();

        // Create composite Gauss rules
        let rule = legendre_generic::<T>(n_gauss);
        let gauss_x = rule.piecewise(&segments_x);
        let gauss_y = rule.piecewise(&segments_y);

        // Create the general SVE processor
        let sampling_sve = SamplingSVE::new(
            segments_x.clone(),
            segments_y.clone(),
            gauss_x.clone(),
            gauss_y.clone(),
            epsilon,
            n_gauss,
        );

        Self {
            kernel,
            epsilon,
            hints,
            n_gauss,
            segments_x,
            segments_y,
            gauss_x,
            gauss_y,
            sampling_sve,
        }
    }

    /// Compute reduced kernel matrix for given symmetry
    fn compute_reduced_matrix(&self, symmetry: SymmetryType) -> DTensor<T, 2> {
        // Compute K_red(x, y) = K(x, y) + sign * K(x, -y)
        // where x, y are in [0, xmax] and [0, ymax]
        let discretized = matrix_from_gauss_with_segments(
            &self.kernel,
            &self.gauss_x,
            &self.gauss_y,
            symmetry,
            &self.hints,
        );

        // Apply weights for SVE

        discretized.apply_weights_for_sve()
    }

    /// Extend polynomials from [0, xmax] to [-xmax, xmax]
    fn extend_result_to_full_domain(
        &self,
        result: (
            PiecewiseLegendrePolyVector,
            Vec<f64>,
            PiecewiseLegendrePolyVector,
        ),
        symmetry: SymmetryType,
    ) -> (
        PiecewiseLegendrePolyVector,
        Vec<f64>,
        PiecewiseLegendrePolyVector,
    ) {
        let (u, s, v) = result;

        // Extend u and v from [0, xmax] to [-xmax, xmax]
        let u_full = extend_to_full_domain(u.get_polys().to_vec(), symmetry, self.kernel.xmax());
        let v_full = extend_to_full_domain(v.get_polys().to_vec(), symmetry, self.kernel.ymax());

        (
            PiecewiseLegendrePolyVector::new(u_full),
            s,
            PiecewiseLegendrePolyVector::new(v_full),
        )
    }
}

impl<T, K> SVEStrategy<T> for CentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: CentrosymmKernel + KernelProperties + Clone,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    fn matrices(&self) -> Vec<DTensor<T, 2>> {
        // Compute reduced kernels for even and odd symmetries
        let even_matrix = self.compute_reduced_matrix(SymmetryType::Even);
        let odd_matrix = self.compute_reduced_matrix(SymmetryType::Odd);

        vec![even_matrix, odd_matrix]
    }

    fn postprocess(
        &self,
        u_list: Vec<DTensor<T, 2>>,
        s_list: Vec<Vec<T>>,
        v_list: Vec<DTensor<T, 2>>,
    ) -> SVEResult {
        // Process even and odd results using SamplingSVE (which doesn't know about symmetry)
        let result_even = self
            .sampling_sve
            .postprocess_single(&u_list[0], &s_list[0], &v_list[0]);
        let result_odd = self
            .sampling_sve
            .postprocess_single(&u_list[1], &s_list[1], &v_list[1]);

        // Now extend to full domain (this is where symmetry comes in)
        let result_even_full = self.extend_result_to_full_domain(result_even, SymmetryType::Even);
        let result_odd_full = self.extend_result_to_full_domain(result_odd, SymmetryType::Odd);

        // Merge the results
        merge_results(result_even_full, result_odd_full, self.epsilon)
    }
}

/// Non-centrosymmetric SVE computation
///
/// This strategy works with non-centrosymmetric kernels by directly computing
/// the kernel matrix over the full domain [-xmax, xmax] × [-ymax, ymax].
/// No symmetry exploitation is performed.
#[allow(dead_code)]
pub struct NonCentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + 'static,
    K: AbstractKernel + KernelProperties,
{
    kernel: K,
    epsilon: f64,
    hints: K::SVEHintsType<T>,
    n_gauss: usize,

    // Geometric information (full domain [-xmax, xmax])
    segments_x: Vec<T>,
    segments_y: Vec<T>,
    gauss_x: Rule<T>,
    gauss_y: Rule<T>,

    // The general SVE processor
    sampling_sve: SamplingSVE<T>,
}

impl<T, K> NonCentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: AbstractKernel + KernelProperties + Clone,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    /// Create a new NonCentrosymmSVE
    pub fn new(kernel: K, epsilon: f64) -> Self {
        let hints = kernel.sve_hints::<T>(epsilon);

        // Get segments for full domain [-xmax, xmax]
        let segments_x = hints.segments_x();
        let segments_y = hints.segments_y();
        let n_gauss = hints.ngauss();

        // Create composite Gauss rules for full domain
        let rule = legendre_generic::<T>(n_gauss);
        let gauss_x = rule.piecewise(&segments_x);
        let gauss_y = rule.piecewise(&segments_y);

        // Create the general SVE processor
        let sampling_sve = SamplingSVE::new(
            segments_x.clone(),
            segments_y.clone(),
            gauss_x.clone(),
            gauss_y.clone(),
            epsilon,
            n_gauss,
        );

        Self {
            kernel,
            epsilon,
            hints,
            n_gauss,
            segments_x,
            segments_y,
            gauss_x,
            gauss_y,
            sampling_sve,
        }
    }

    /// Compute kernel matrix for non-centrosymmetric kernel
    fn compute_matrix(&self) -> DTensor<T, 2> {
        // Compute K(x, y) directly over full domain
        let discretized = matrix_from_gauss_noncentrosymmetric(
            &self.kernel,
            &self.gauss_x,
            &self.gauss_y,
            &self.hints,
        );

        // Apply weights for SVE
        discretized.apply_weights_for_sve()
    }
}

impl<T, K> SVEStrategy<T> for NonCentrosymmSVE<T, K>
where
    T: CustomNumeric + Send + Sync + Clone + 'static,
    K: AbstractKernel + KernelProperties + Clone,
    K::SVEHintsType<T>: SVEHints<T> + Clone,
{
    fn matrices(&self) -> Vec<DTensor<T, 2>> {
        // Single matrix for non-centrosymmetric kernel
        vec![self.compute_matrix()]
    }

    fn postprocess(
        &self,
        u_list: Vec<DTensor<T, 2>>,
        s_list: Vec<Vec<T>>,
        v_list: Vec<DTensor<T, 2>>,
    ) -> SVEResult {
        // Process single result using SamplingSVE
        let (u_polys, s, v_polys) = self
            .sampling_sve
            .postprocess_single(&u_list[0], &s_list[0], &v_list[0]);

        // No domain extension needed - already on full domain
        SVEResult::new(u_polys, s, v_polys, self.epsilon)
    }
}
