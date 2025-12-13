//! Complex matrix fitter: A ∈ C^{n×m}
//!
//! This module provides `ComplexMatrixFitter` for solving least-squares problems
//! where the matrix, coefficients, and values are all complex.

use crate::gemm::GemmBackendHandle;
use mdarray::{DTensor, DView, DynRank, Shape, Slice, ViewMut};
use num_complex::Complex;
use std::cell::RefCell;

use super::common::{
    combine_complex, compute_complex_svd, copy_from_contiguous, copy_to_contiguous,
    extract_real_parts_coeffs, make_perm_to_front, ComplexSVD, InplaceFitter,
};

/// Fitter for complex matrix with complex coefficients: A ∈ C^{n×m}
///
/// Solves: min ||A * coeffs - values||^2
/// where A, coeffs, values are all complex
///
/// # Example
/// ```ignore
/// let matrix = DTensor::from_fn([10, 5], |idx| Complex::new(...));
/// let fitter = ComplexMatrixFitter::new(matrix);
///
/// let coeffs: Vec<Complex<f64>> = vec![...];
/// let values = fitter.evaluate(&coeffs);  // → Vec<Complex<f64>>
/// let fitted_coeffs = fitter.fit(&values);  // ← Vec<Complex<f64>>, → Vec<Complex<f64>>
/// ```
pub(crate) struct ComplexMatrixFitter {
    pub matrix: DTensor<Complex<f64>, 2>,   // (n_points, basis_size)
    matrix_t: DTensor<Complex<f64>, 2>,     // (basis_size, n_points) - transposed
    // Real/Imag parts for evaluate_dz operations (pre-computed)
    matrix_re: DTensor<f64, 2>,             // (n_points, basis_size)
    matrix_im: DTensor<f64, 2>,             // (n_points, basis_size)
    matrix_re_t: DTensor<f64, 2>,           // (basis_size, n_points) - transposed
    matrix_im_t: DTensor<f64, 2>,           // (basis_size, n_points) - transposed
    svd: RefCell<Option<ComplexSVDExtended>>,
}

/// Extended SVD structure with pre-computed transposes for dim=1 operations
struct ComplexSVDExtended {
    svd: ComplexSVD,
    /// u_conj = ut^T (no conjugate): shape (n_points, min_dim)
    u_conj: DTensor<Complex<f64>, 2>,
    /// vt = v^T (no conjugate): shape (min_dim, basis_size)
    vt: DTensor<Complex<f64>, 2>,
}

impl ComplexSVDExtended {
    fn from_svd(svd: ComplexSVD, n_points: usize, basis_size: usize) -> Self {
        let min_dim = svd.s.len();

        // u_conj[i,j] = ut[j,i] (transpose, no conjugate)
        let u_conj = DTensor::<Complex<f64>, 2>::from_fn([n_points, min_dim], |idx| {
            svd.ut[[idx[1], idx[0]]]
        });

        // vt[i,j] = v[j,i] (transpose, no conjugate)
        let vt = DTensor::<Complex<f64>, 2>::from_fn([min_dim, basis_size], |idx| {
            svd.v[[idx[1], idx[0]]]
        });

        Self { svd, u_conj, vt }
    }
}

impl ComplexMatrixFitter {
    /// Create a new fitter with the given complex matrix
    pub fn new(matrix: DTensor<Complex<f64>, 2>) -> Self {
        let (n_points, basis_size) = *matrix.shape();

        // Pre-compute transposed matrix for dim=1 operations
        let matrix_t = DTensor::<Complex<f64>, 2>::from_fn([basis_size, n_points], |idx| {
            matrix[[idx[1], idx[0]]]
        });

        // Pre-compute real/imaginary parts for evaluate_dz operations
        let matrix_re = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| matrix[idx].re);
        let matrix_im = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| matrix[idx].im);

        // Pre-compute transposed versions
        let matrix_re_t =
            DTensor::<f64, 2>::from_fn([basis_size, n_points], |idx| matrix_re[[idx[1], idx[0]]]);
        let matrix_im_t =
            DTensor::<f64, 2>::from_fn([basis_size, n_points], |idx| matrix_im[[idx[1], idx[0]]]);

        Self {
            matrix,
            matrix_t,
            matrix_re,
            matrix_im,
            matrix_re_t,
            matrix_im_t,
            svd: RefCell::new(None),
        }
    }

    /// Number of data points
    pub fn n_points(&self) -> usize {
        self.matrix.shape().0
    }

    /// Number of basis functions (coefficients)
    pub fn basis_size(&self) -> usize {
        self.matrix.shape().1
    }

    /// Evaluate: coeffs (complex) → values (complex)
    ///
    /// Computes: values = A * coeffs using GEMM
    pub fn evaluate(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        assert_eq!(
            coeffs.len(),
            self.basis_size(),
            "coeffs.len()={} must equal basis_size={}",
            coeffs.len(),
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = vec![Complex::new(0.0, 0.0); n_points];
        self.evaluate_to(backend, coeffs, &mut out);
        out
    }

    /// Evaluate: coeffs (complex) → values (complex), writing to output slice
    ///
    /// Computes: out = A * coeffs
    pub fn evaluate_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) {
        assert_eq!(coeffs.len(), self.basis_size());
        assert_eq!(out.len(), self.n_points());

        // Create views treating slices as column vectors [N, 1]
        let coeffs_view = unsafe {
            let mapping = mdarray::DenseMapping::new((coeffs.len(), 1));
            mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(coeffs.as_ptr(), mapping)
        };
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((out.len(), 1));
            mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
        };
        self.evaluate_2d_to(backend, &coeffs_view, &mut out_view);
    }

    /// Fit: values (complex) → coeffs (complex)
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
    pub fn fit(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[Complex<f64>],
    ) -> Vec<Complex<f64>> {
        assert_eq!(
            values.len(),
            self.n_points(),
            "values.len()={} must equal n_points={}",
            values.len(),
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = vec![Complex::new(0.0, 0.0); basis_size];
        self.fit_to(backend, values, &mut out);
        out
    }

    /// Fit: values (complex) → coeffs (complex), writing to output slice
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
    pub fn fit_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) {
        assert_eq!(values.len(), self.n_points());
        assert_eq!(out.len(), self.basis_size());

        // Create views treating slices as column vectors [N, 1]
        let values_view = unsafe {
            let mapping = mdarray::DenseMapping::new((values.len(), 1));
            mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(values.as_ptr(), mapping)
        };
        let mut out_view = unsafe {
            let mapping = mdarray::DenseMapping::new((out.len(), 1));
            mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
        };
        self.fit_2d_to(backend, &values_view, &mut out_view);
    }

    /// Evaluate 2D complex tensor (along dim=0) using matrix multiplication
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Values tensor, shape: [n_points, extra_size]
    ///
    /// # Note
    /// This is a wrapper around `evaluate_2d_to` that allocates the output tensor.
    pub fn evaluate_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        let (basis_size, extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        let n_points = self.n_points();
        let mut out = DTensor::<Complex<f64>, 2>::zeros([n_points, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.evaluate_2d_to(backend, coeffs_2d, &mut out_view);
        out
    }

    /// Evaluate 2D complex tensor (along dim=0), writing to a mutable view
    ///
    /// Computes: out = matrix * coeffs_2d
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape: [basis_size, extra_size]
    /// * `out` - Output mutable view, shape: [n_points, extra_size] (will be overwritten)
    pub fn evaluate_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        use crate::gemm::matmul_par_to_viewmut;

        let (basis_size, extra_size) = *coeffs_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );
        assert_eq!(
            out_rows,
            self.n_points(),
            "out.shape().0={} must equal n_points={}",
            out_rows,
            self.n_points()
        );
        assert_eq!(
            out_cols, extra_size,
            "out.shape().1={} must equal extra_size={}",
            out_cols, extra_size
        );

        let matrix_view = self.matrix.view(.., ..);
        matmul_par_to_viewmut(&matrix_view, coeffs_2d, out, backend);
    }

    /// Evaluate 2D real tensor to complex values (along dim=0) using matrix multiplication
    ///
    /// Computes: values_2d = A * coeffs_2d where A is complex and coeffs_2d is real
    ///
    /// # Arguments
    /// * `coeffs_2d` - Real coefficients, shape: [basis_size, extra_size]
    ///
    /// # Returns
    /// Complex values, shape: [n_points, extra_size]
    pub fn evaluate_2d_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, f64, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        use crate::gemm::matmul_par_view;

        let (basis_size, _extra_size) = *coeffs_2d.shape();
        assert_eq!(
            basis_size,
            self.basis_size(),
            "coeffs_2d.shape().0={} must equal basis_size={}",
            basis_size,
            self.basis_size()
        );

        // Split matrix into real and imaginary parts
        let matrix_re = DTensor::<f64, 2>::from_fn(*self.matrix.shape(), |idx| self.matrix[idx].re);
        let matrix_im = DTensor::<f64, 2>::from_fn(*self.matrix.shape(), |idx| self.matrix[idx].im);

        // Compute real and imaginary parts separately
        let matrix_re_view = matrix_re.view(.., ..);
        let matrix_im_view = matrix_im.view(.., ..);
        let values_re = matmul_par_view(&matrix_re_view, coeffs_2d, backend);
        let values_im = matmul_par_view(&matrix_im_view, coeffs_2d, backend);

        // Combine to complex
        combine_complex(&values_re, &values_im)
    }

    /// Fit 2D complex tensor (along dim=0) using matrix multiplication
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Complex coefficients tensor, shape: [basis_size, extra_size]
    ///
    /// # Note
    /// This is a wrapper around `fit_2d_to` that allocates the output tensor.
    pub fn fit_2d(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<Complex<f64>, 2> {
        let (n_points, extra_size) = *values_2d.shape();
        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );

        let basis_size = self.basis_size();
        let mut out = DTensor::<Complex<f64>, 2>::zeros([basis_size, extra_size]);
        let mut out_view = out.view_mut(.., ..);
        self.fit_2d_to(backend, values_2d, &mut out_view);
        out
    }

    /// Fit 2D complex tensor (along dim=0), writing to a mutable view
    ///
    /// Solves: min ||A * coeffs - values||^2 using complex SVD
    ///
    /// # Arguments
    /// * `values_2d` - Shape: [n_points, extra_size]
    /// * `out` - Output mutable view, shape: [basis_size, extra_size] (will be overwritten)
    ///
    /// # Note
    /// An intermediate buffer of size [min_dim, extra_size] is still allocated internally.
    pub fn fit_2d_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
    ) {
        use crate::gemm::{matmul_par_to_viewmut, matmul_par_view};

        let (n_points, extra_size) = *values_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        assert_eq!(
            n_points,
            self.n_points(),
            "values_2d.shape().0={} must equal n_points={}",
            n_points,
            self.n_points()
        );
        assert_eq!(
            out_rows,
            self.basis_size(),
            "out.shape().0={} must equal basis_size={}",
            out_rows,
            self.basis_size()
        );
        assert_eq!(
            out_cols, extra_size,
            "out.shape().1={} must equal extra_size={}",
            out_cols, extra_size
        );

        // Compute SVD lazily
        self.ensure_svd();

        let svd_ext = self.svd.borrow();
        let svd_ext = svd_ext.as_ref().unwrap();
        let svd = &svd_ext.svd;

        // coeffs_2d = V * S^{-1} * U^H * values_2d

        // 1. U^H * values_2d (ut is already U^H)
        let ut_view = svd.ut.view(.., ..);
        let mut uh_values = matmul_par_view(&ut_view, values_2d, backend); // [min_dim, extra_size]

        // 2. S^{-1} * (U^H * values_2d) - in-place division
        let min_dim = svd.s.len();
        for i in 0..min_dim {
            for j in 0..extra_size {
                uh_values[[i, j]] /= svd.s[i];
            }
        }

        // 3. V * (S^{-1} * U^H * values_2d) → out
        let v_view = svd.v.view(.., ..);
        let uh_values_view = uh_values.view(.., ..);
        matmul_par_to_viewmut(&v_view, &uh_values_view, out, backend);
    }

    /// Fit 2D complex values to real coefficients (along dim=0)
    ///
    /// This method fits complex values at Matsubara frequencies to real IR coefficients.
    /// It takes the real part of the least-squares solution.
    ///
    /// # Arguments
    /// * `values_2d` - Complex values, shape: [n_points, extra_size]
    ///
    /// # Returns
    /// Real coefficients, shape: [basis_size, extra_size]
    pub fn fit_2d_real(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
    ) -> DTensor<f64, 2> {
        // Fit as complex, then take real part
        let coeffs_complex = self.fit_2d(backend, values_2d);

        // Extract real part
        extract_real_parts_coeffs(&coeffs_complex)
    }

    /// Evaluate 2D complex tensor with configurable target dimension
    ///
    /// # Arguments
    /// * `coeffs_2d` - Shape depends on dim: [basis_size, extra] if dim=0, [extra, basis_size] if dim=1
    /// * `out` - Output mutable view
    /// * `dim` - Target dimension (0 or 1)
    fn evaluate_2d_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
        dim: usize,
    ) {
        use crate::gemm::matmul_par_to_viewmut;

        let (coeffs_rows, coeffs_cols) = *coeffs_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        if dim == 0 {
            // out[n_points, extra] = matrix[n_points, basis_size] * coeffs[basis_size, extra]
            let basis_size = coeffs_rows;
            let extra_size = coeffs_cols;

            assert_eq!(
                basis_size,
                self.basis_size(),
                "coeffs_2d.shape().0={} must equal basis_size={}",
                basis_size,
                self.basis_size()
            );
            assert_eq!(
                out_rows,
                self.n_points(),
                "out.shape().0={} must equal n_points={}",
                out_rows,
                self.n_points()
            );
            assert_eq!(
                out_cols, extra_size,
                "out.shape().1={} must equal extra_size={}",
                out_cols, extra_size
            );

            let matrix_view = self.matrix.view(.., ..);
            matmul_par_to_viewmut(&matrix_view, coeffs_2d, out, backend);
        } else {
            // dim == 1
            // out[extra, n_points] = coeffs[extra, basis_size] * matrix^T[basis_size, n_points]
            let extra_size = coeffs_rows;
            let basis_size = coeffs_cols;

            assert_eq!(
                basis_size,
                self.basis_size(),
                "coeffs_2d.shape().1={} must equal basis_size={}",
                basis_size,
                self.basis_size()
            );
            assert_eq!(
                out_rows, extra_size,
                "out.shape().0={} must equal extra_size={}",
                out_rows, extra_size
            );
            assert_eq!(
                out_cols,
                self.n_points(),
                "out.shape().1={} must equal n_points={}",
                out_cols,
                self.n_points()
            );

            // Use pre-computed transposed matrix
            let matrix_t_view = self.matrix_t.view(.., ..);
            matmul_par_to_viewmut(coeffs_2d, &matrix_t_view, out, backend);
        }
    }

    /// Fit 2D complex tensor with configurable target dimension
    ///
    /// # Arguments
    /// * `values_2d` - Shape depends on dim: [n_points, extra] if dim=0, [extra, n_points] if dim=1
    /// * `out` - Output mutable view
    /// * `dim` - Target dimension (0 or 1)
    fn fit_2d_to_dim(
        &self,
        backend: Option<&GemmBackendHandle>,
        values_2d: &DView<'_, Complex<f64>, 2>,
        out: &mut mdarray::DViewMut<'_, Complex<f64>, 2>,
        dim: usize,
    ) {
        use crate::gemm::{matmul_par_to_viewmut, matmul_par_view};

        let (values_rows, values_cols) = *values_2d.shape();
        let (out_rows, out_cols) = *out.shape();

        // Compute SVD lazily
        self.ensure_svd();

        let svd_ext = self.svd.borrow();
        let svd_ext = svd_ext.as_ref().unwrap();
        let svd = &svd_ext.svd;
        let min_dim = svd.s.len();

        if dim == 0 {
            // values[n_points, extra] → coeffs[basis_size, extra]
            let n_points = values_rows;
            let extra_size = values_cols;

            assert_eq!(
                n_points,
                self.n_points(),
                "values_2d.shape().0={} must equal n_points={}",
                n_points,
                self.n_points()
            );
            assert_eq!(
                out_rows,
                self.basis_size(),
                "out.shape().0={} must equal basis_size={}",
                out_rows,
                self.basis_size()
            );
            assert_eq!(
                out_cols, extra_size,
                "out.shape().1={} must equal extra_size={}",
                out_cols, extra_size
            );

            // coeffs = V * S^{-1} * U^H * values
            let ut_view = svd.ut.view(.., ..);
            let mut uh_values = matmul_par_view(&ut_view, values_2d, backend);

            for i in 0..min_dim {
                for j in 0..extra_size {
                    uh_values[[i, j]] /= svd.s[i];
                }
            }

            let v_view = svd.v.view(.., ..);
            let uh_values_view = uh_values.view(.., ..);
            matmul_par_to_viewmut(&v_view, &uh_values_view, out, backend);
        } else {
            // dim == 1
            // values[extra, n_points] → coeffs[extra, basis_size]
            let extra_size = values_rows;
            let n_points = values_cols;

            assert_eq!(
                n_points,
                self.n_points(),
                "values_2d.shape().1={} must equal n_points={}",
                n_points,
                self.n_points()
            );
            assert_eq!(
                out_rows, extra_size,
                "out.shape().0={} must equal extra_size={}",
                out_rows, extra_size
            );
            assert_eq!(
                out_cols,
                self.basis_size(),
                "out.shape().1={} must equal basis_size={}",
                out_cols,
                self.basis_size()
            );

            // coeffs = values * conj(U) * S^{-1} * V^T
            // Use pre-computed u_conj and vt
            let u_conj_view = svd_ext.u_conj.view(.., ..);
            let mut values_u = matmul_par_view(values_2d, &u_conj_view, backend); // [extra, min_dim]

            // Apply S^{-1}
            for i in 0..extra_size {
                for j in 0..min_dim {
                    values_u[[i, j]] /= svd.s[j];
                }
            }

            // Use pre-computed V^T
            let vt_view = svd_ext.vt.view(.., ..);
            let values_u_view = values_u.view(.., ..);
            matmul_par_to_viewmut(&values_u_view, &vt_view, out, backend);
        }
    }

    /// Ensure SVD is computed (lazy initialization)
    fn ensure_svd(&self) {
        if self.svd.borrow().is_none() {
            let svd = compute_complex_svd(&self.matrix);
            let svd_ext = ComplexSVDExtended::from_svd(svd, self.n_points(), self.basis_size());
            *self.svd.borrow_mut() = Some(svd_ext);
        }
    }

    /// Evaluate ND complex tensor (along specified dim)
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional complex tensor with coeffs.shape().dim(dim) == basis_size
    /// * `dim` - Target dimension
    /// * `out` - Output mutable view with out.shape().dim(dim) == n_points
    pub fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(coeffs.shape().dim(dim), basis_size);
        assert_eq!(out.shape().dim(dim), n_points);

        let total = coeffs.len();
        let extra_size = total / basis_size;

        if dim == 0 {
            // Fast path 1: dim == 0, no movedim needed
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1, no movedim needed
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 1);
        } else {
            // General path: use temporary buffers for movedim
            let perm = make_perm_to_front(rank, dim);
            let permuted_view = coeffs.permute(&perm[..]);

            let mut input_buffer: Vec<Complex<f64>> = Vec::with_capacity(total);
            unsafe { input_buffer.set_len(total); }
            copy_to_contiguous(&permuted_view.into_dyn(), &mut input_buffer);

            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(input_buffer.as_ptr(), mapping)
            };

            let out_total = out.len();
            let mut output_buffer: Vec<Complex<f64>> = Vec::with_capacity(out_total);
            unsafe { output_buffer.set_len(out_total); }

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(
                    output_buffer.as_mut_ptr(),
                    mapping,
                )
            };

            self.evaluate_2d_to_dim(backend, &coeffs_2d, &mut out_2d, 0);

            // Copy from contiguous buffer to strided output view
            let out_permuted = (&mut *out).permute_mut(&perm[..]);
            copy_from_contiguous(&output_buffer, &mut out_permuted.into_dyn());
        }
        true
    }

    /// Fit ND complex tensor (along specified dim)
    ///
    /// # Arguments
    /// * `values` - N-dimensional complex tensor with values.shape().dim(dim) == n_points
    /// * `dim` - Target dimension
    /// * `out` - Output mutable view with out.shape().dim(dim) == basis_size
    pub fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        let rank = values.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(values.shape().dim(dim), n_points);
        assert_eq!(out.shape().dim(dim), basis_size);

        let total = values.len();
        let extra_size = total / n_points;

        if dim == 0 {
            // Fast path 1: dim == 0, no movedim needed
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(values.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 0);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1, no movedim needed
            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, n_points));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(values.as_ptr(), mapping)
            };

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(out.as_mut_ptr(), mapping)
            };

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 1);
        } else {
            // General path: use temporary buffers for movedim
            let perm = make_perm_to_front(rank, dim);
            let permuted_view = values.permute(&perm[..]);

            let mut input_buffer: Vec<Complex<f64>> = Vec::with_capacity(total);
            unsafe { input_buffer.set_len(total); }
            copy_to_contiguous(&permuted_view.into_dyn(), &mut input_buffer);

            let values_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((n_points, extra_size));
                mdarray::DView::<'_, Complex<f64>, 2>::new_unchecked(input_buffer.as_ptr(), mapping)
            };

            let out_total = out.len();
            let mut output_buffer: Vec<Complex<f64>> = Vec::with_capacity(out_total);
            unsafe { output_buffer.set_len(out_total); }

            let mut out_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DViewMut::<'_, Complex<f64>, 2>::new_unchecked(
                    output_buffer.as_mut_ptr(),
                    mapping,
                )
            };

            self.fit_2d_to_dim(backend, &values_2d, &mut out_2d, 0);

            // Copy from contiguous buffer to strided output view
            let out_permuted = (&mut *out).permute_mut(&perm[..]);
            copy_from_contiguous(&output_buffer, &mut out_permuted.into_dyn());
        }
        true
    }

    /// Evaluate ND real tensor to complex (along specified dim)
    ///
    /// Computes: out = matrix_re * coeffs + i * matrix_im * coeffs
    ///
    /// # Arguments
    /// * `coeffs` - N-dimensional real tensor with coeffs.shape().dim(dim) == basis_size
    /// * `dim` - Target dimension
    /// * `out` - Output mutable view (complex) with out.shape().dim(dim) == n_points
    pub fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        use crate::gemm::matmul_par_view;

        let rank = coeffs.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(coeffs.shape().dim(dim), basis_size);
        assert_eq!(out.shape().dim(dim), n_points);

        let total = coeffs.len();
        let extra_size = total / basis_size;

        // Helper function to combine real results into complex output
        fn combine_to_complex(
            values_re: &mdarray::DTensor<f64, 2>,
            values_im: &mdarray::DTensor<f64, 2>,
            out: &mut ViewMut<'_, Complex<f64>, DynRank>,
            perm: Option<&[usize]>,
        ) {
            let (rows, cols) = *values_re.shape();
            let total = rows * cols;

            if let Some(perm) = perm {
                // General case: need to permute back
                let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(total);
                for i in 0..rows {
                    for j in 0..cols {
                        buffer.push(Complex::new(values_re[[i, j]], values_im[[i, j]]));
                    }
                }
                let out_permuted = (&mut *out).permute_mut(perm);
                copy_from_contiguous(&buffer, &mut out_permuted.into_dyn());
            } else {
                // Fast path: direct copy
                let out_ptr = out.as_mut_ptr();
                for i in 0..rows {
                    for j in 0..cols {
                        let idx = i * cols + j;
                        unsafe {
                            *out_ptr.add(idx) = Complex::new(values_re[[i, j]], values_im[[i, j]]);
                        }
                    }
                }
            }
        }

        if dim == 0 {
            // Fast path 1: dim == 0
            // out[n_points, extra] = matrix * coeffs[basis_size, extra]
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let matrix_re_view = self.matrix_re.view(.., ..);
            let matrix_im_view = self.matrix_im.view(.., ..);
            let values_re = matmul_par_view(&matrix_re_view, &coeffs_2d, backend);
            let values_im = matmul_par_view(&matrix_im_view, &coeffs_2d, backend);

            combine_to_complex(&values_re, &values_im, out, None);
        } else if dim == rank - 1 {
            // Fast path 2: dim == N-1
            // out[extra, n_points] = coeffs[extra, basis_size] * matrix^T
            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((extra_size, basis_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(coeffs.as_ptr(), mapping)
            };

            let matrix_re_t_view = self.matrix_re_t.view(.., ..);
            let matrix_im_t_view = self.matrix_im_t.view(.., ..);
            let values_re = matmul_par_view(&coeffs_2d, &matrix_re_t_view, backend);
            let values_im = matmul_par_view(&coeffs_2d, &matrix_im_t_view, backend);

            combine_to_complex(&values_re, &values_im, out, None);
        } else {
            // General path: use temporary buffers for movedim
            let perm = make_perm_to_front(rank, dim);
            let permuted_view = coeffs.permute(&perm[..]);

            let mut input_buffer: Vec<f64> = Vec::with_capacity(total);
            unsafe { input_buffer.set_len(total); }
            copy_to_contiguous(&permuted_view.into_dyn(), &mut input_buffer);

            let coeffs_2d = unsafe {
                let mapping = mdarray::DenseMapping::new((basis_size, extra_size));
                mdarray::DView::<'_, f64, 2>::new_unchecked(input_buffer.as_ptr(), mapping)
            };

            let matrix_re_view = self.matrix_re.view(.., ..);
            let matrix_im_view = self.matrix_im.view(.., ..);
            let values_re = matmul_par_view(&matrix_re_view, &coeffs_2d, backend);
            let values_im = matmul_par_view(&matrix_im_view, &coeffs_2d, backend);

            combine_to_complex(&values_re, &values_im, out, Some(&perm));
        }
        true
    }

    /// Fit ND complex tensor to real coefficients (along specified dim)
    ///
    /// Computes complex fit using fit_nd_zz_to, then extracts real parts.
    ///
    /// # Arguments
    /// * `values` - N-dimensional complex tensor with values.shape().dim(dim) == n_points
    /// * `dim` - Target dimension
    /// * `out` - Output mutable view (real) with out.shape().dim(dim) == basis_size
    pub fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        let rank = values.rank();
        let basis_size = self.basis_size();
        let n_points = self.n_points();

        // Validate
        assert!(dim < rank, "dim={} must be < rank={}", dim, rank);
        assert_eq!(out.rank(), rank);
        assert_eq!(values.shape().dim(dim), n_points);
        assert_eq!(out.shape().dim(dim), basis_size);

        // Build output shape for complex temp buffer
        let mut temp_shape: Vec<usize> = Vec::with_capacity(rank);
        for i in 0..rank {
            if i == dim {
                temp_shape.push(basis_size);
            } else {
                temp_shape.push(values.shape().dim(i));
            }
        }

        // Allocate temp buffer for complex coefficients
        let mut temp_coeffs: mdarray::Tensor<Complex<f64>, DynRank> =
            mdarray::Tensor::zeros(&temp_shape[..]);

        // Fit to complex coefficients
        self.fit_nd_zz_to(backend, values, dim, &mut temp_coeffs.expr_mut());

        // Extract real parts to output
        let total = out.len();
        let temp_slice = temp_coeffs.expr();

        // Copy real parts - iterate in same order
        if dim == 0 {
            // Fast path: contiguous in memory
            let out_ptr = out.as_mut_ptr();
            let temp_ptr = temp_coeffs.as_ptr();
            for i in 0..total {
                unsafe {
                    *out_ptr.add(i) = (*temp_ptr.add(i)).re;
                }
            }
        } else if dim == rank - 1 {
            // Fast path: contiguous in memory
            let out_ptr = out.as_mut_ptr();
            let temp_ptr = temp_coeffs.as_ptr();
            for i in 0..total {
                unsafe {
                    *out_ptr.add(i) = (*temp_ptr.add(i)).re;
                }
            }
        } else {
            // General case: need to handle strided access
            let perm = make_perm_to_front(rank, dim);
            let temp_permuted = temp_slice.permute(&perm[..]);
            let out_permuted = (&mut *out).permute_mut(&perm[..]);

            // Both are now contiguous after permutation
            for (o, t) in out_permuted.into_dyn().iter_mut().zip(temp_permuted.into_dyn().iter()) {
                *o = t.re;
            }
        }
        true
    }
}

/// InplaceFitter implementation for ComplexMatrixFitter
///
/// Supported operations:
/// - `zz`: Complex input → Complex output (full support)
/// - `dz`: Real input → Complex output (evaluate only)
/// - `zd`: Complex input → Real output (fit only, takes real part)
impl InplaceFitter for ComplexMatrixFitter {
    fn n_points(&self) -> usize {
        self.n_points()
    }

    fn basis_size(&self) -> usize {
        self.basis_size()
    }

    fn evaluate_nd_dz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<f64, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        ComplexMatrixFitter::evaluate_nd_dz_to(self, backend, coeffs, dim, out)
    }

    fn evaluate_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        coeffs: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        ComplexMatrixFitter::evaluate_nd_zz_to(self, backend, coeffs, dim, out)
    }

    fn fit_nd_zd_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, f64, DynRank>,
    ) -> bool {
        ComplexMatrixFitter::fit_nd_zd_to(self, backend, values, dim, out)
    }

    fn fit_nd_zz_to(
        &self,
        backend: Option<&GemmBackendHandle>,
        values: &Slice<Complex<f64>, DynRank>,
        dim: usize,
        out: &mut ViewMut<'_, Complex<f64>, DynRank>,
    ) -> bool {
        ComplexMatrixFitter::fit_nd_zz_to(self, backend, values, dim, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::DTensor;
    use num_complex::Complex;

    trait ErrorNorm {
        fn error_norm(&self) -> f64;
    }

    impl ErrorNorm for Complex<f64> {
        fn error_norm(&self) -> f64 {
            self.norm()
        }
    }

    #[test]
    fn test_roundtrip() {
        let n_points = 10;
        let basis_size = 5;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            let mag = i.powi(j);
            let phase = (j as f64) * 0.5;
            Complex::new(mag * phase.cos(), mag * phase.sin())
        });

        let fitter = ComplexMatrixFitter::new(matrix);

        let coeffs: Vec<Complex<f64>> = (0..basis_size)
            .map(|i| Complex::new((i as f64 + 1.0) * 0.5, (i as f64) * 0.3))
            .collect();

        let values = fitter.evaluate(None, &coeffs);
        assert_eq!(values.len(), n_points);

        let fitted_coeffs = fitter.fit(None, &values);
        assert_eq!(fitted_coeffs.len(), basis_size);

        for (i, (orig, fitted)) in coeffs.iter().zip(fitted_coeffs.iter()).enumerate() {
            let error = (orig - fitted).error_norm();
            assert!(error < 1e-8, "Complex matrix roundtrip error at {}: {}", i, error);
        }
    }

    #[test]
    fn test_vs_complex_to_real() {
        use crate::fitters::ComplexToRealFitter;

        let n_points = 8;
        let basis_size = 4;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            Complex::new(i.powi(j), (i * (j as f64) * 0.1).sin())
        });

        let fitter_c2r = ComplexToRealFitter::new(&matrix);
        let fitter_complex = ComplexMatrixFitter::new(matrix);

        let coeffs_real: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();

        let values = fitter_c2r.evaluate(None, &coeffs_real);

        let fitted_complex = fitter_complex.fit(None, &values);

        for (i, &coeff_real) in coeffs_real.iter().enumerate() {
            let diff_re = (coeff_real - fitted_complex[i].re).abs();
            let im = fitted_complex[i].im.abs();

            assert!(diff_re < 1e-10, "Real part mismatch at {}: {}", i, diff_re);
            assert!(im < 1e-10, "Imaginary part should be small at {}: {}", i, im);
        }
    }

    #[test]
    fn test_nd_roundtrip() {
        use mdarray::Tensor;

        let n_points = 8;
        let basis_size = 4;
        let extra1 = 3;
        let extra2 = 2;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            let mag = i.powi(j);
            let phase = (j as f64) * 0.5;
            Complex::new(mag * phase.cos(), mag * phase.sin())
        });

        let fitter = ComplexMatrixFitter::new(matrix);

        // Test dim=0
        {
            let coeffs =
                Tensor::<Complex<f64>, mdarray::DynRank>::from_fn(&[basis_size, extra1][..], |idx| {
                    Complex::new((idx[0] + idx[1]) as f64, (idx[0] * idx[1]) as f64 * 0.1)
                });

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[n_points, extra1][..]);
            let mut fitted =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[basis_size, extra1][..]);

            fitter.evaluate_nd_zz_to(None, &coeffs.expr(), 0, &mut values.expr_mut());
            fitter.fit_nd_zz_to(None, &values.expr(), 0, &mut fitted.expr_mut());

            for i in 0..basis_size {
                for j in 0..extra1 {
                    let error = (coeffs[&[i, j][..]] - fitted[&[i, j][..]]).norm();
                    assert!(error < 1e-8, "dim=0 roundtrip error at [{}, {}]: {}", i, j, error);
                }
            }
        }

        // Test dim=1
        {
            let coeffs =
                Tensor::<Complex<f64>, mdarray::DynRank>::from_fn(&[extra1, basis_size][..], |idx| {
                    Complex::new((idx[0] + idx[1]) as f64, (idx[0] * idx[1]) as f64 * 0.1)
                });

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra1, n_points][..]);
            let mut fitted =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra1, basis_size][..]);

            fitter.evaluate_nd_zz_to(None, &coeffs.expr(), 1, &mut values.expr_mut());
            fitter.fit_nd_zz_to(None, &values.expr(), 1, &mut fitted.expr_mut());

            for i in 0..extra1 {
                for j in 0..basis_size {
                    let error = (coeffs[&[i, j][..]] - fitted[&[i, j][..]]).norm();
                    assert!(error < 1e-8, "dim=1 roundtrip error at [{}, {}]: {}", i, j, error);
                }
            }
        }

        // Test dim=1 in 3D array (middle dimension)
        {
            let coeffs = Tensor::<Complex<f64>, mdarray::DynRank>::from_fn(
                &[extra1, basis_size, extra2][..],
                |idx| Complex::new((idx[0] + idx[1] + idx[2]) as f64, (idx[0] * idx[2]) as f64 * 0.1),
            );

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra1, n_points, extra2][..]);
            let mut fitted =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra1, basis_size, extra2][..]);

            fitter.evaluate_nd_zz_to(None, &coeffs.expr(), 1, &mut values.expr_mut());
            fitter.fit_nd_zz_to(None, &values.expr(), 1, &mut fitted.expr_mut());

            for i in 0..extra1 {
                for j in 0..basis_size {
                    for k in 0..extra2 {
                        let error = (coeffs[&[i, j, k][..]] - fitted[&[i, j, k][..]]).norm();
                        assert!(
                            error < 1e-8,
                            "dim=1 (3D) roundtrip error at [{}, {}, {}]: {}",
                            i,
                            j,
                            k,
                            error
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_dz_zd_roundtrip() {
        use mdarray::Tensor;

        let n_points = 8;
        let basis_size = 4;
        let extra = 3;

        let matrix = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            let mag = i.powi(j);
            let phase = (j as f64) * 0.5;
            Complex::new(mag * phase.cos(), mag * phase.sin())
        });

        let fitter = ComplexMatrixFitter::new(matrix);

        // Test evaluate_nd_dz_to and fit_nd_zd_to roundtrip
        // Real coeffs → Complex values → Real coeffs

        // dim=0
        {
            let coeffs_real = Tensor::<f64, mdarray::DynRank>::from_fn(&[basis_size, extra][..], |idx| {
                (idx[0] + idx[1]) as f64 * 0.5 + 1.0
            });

            let mut values = Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[n_points, extra][..]);
            let mut fitted_real = Tensor::<f64, mdarray::DynRank>::zeros(&[basis_size, extra][..]);

            fitter.evaluate_nd_dz_to(None, &coeffs_real.expr(), 0, &mut values.expr_mut());
            fitter.fit_nd_zd_to(None, &values.expr(), 0, &mut fitted_real.expr_mut());

            for i in 0..basis_size {
                for j in 0..extra {
                    let error = (coeffs_real[&[i, j][..]] - fitted_real[&[i, j][..]]).abs();
                    assert!(
                        error < 1e-8,
                        "dz/zd dim=0 roundtrip error at [{}, {}]: {}",
                        i,
                        j,
                        error
                    );
                }
            }
        }

        // dim=1
        {
            let coeffs_real = Tensor::<f64, mdarray::DynRank>::from_fn(&[extra, basis_size][..], |idx| {
                (idx[0] + idx[1]) as f64 * 0.5 + 1.0
            });

            let mut values = Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra, n_points][..]);
            let mut fitted_real = Tensor::<f64, mdarray::DynRank>::zeros(&[extra, basis_size][..]);

            fitter.evaluate_nd_dz_to(None, &coeffs_real.expr(), 1, &mut values.expr_mut());
            fitter.fit_nd_zd_to(None, &values.expr(), 1, &mut fitted_real.expr_mut());

            for i in 0..extra {
                for j in 0..basis_size {
                    let error = (coeffs_real[&[i, j][..]] - fitted_real[&[i, j][..]]).abs();
                    assert!(
                        error < 1e-8,
                        "dz/zd dim=1 roundtrip error at [{}, {}]: {}",
                        i,
                        j,
                        error
                    );
                }
            }
        }

        // dim=1 in 3D (middle dimension)
        {
            let extra2 = 2;
            let coeffs_real = Tensor::<f64, mdarray::DynRank>::from_fn(
                &[extra, basis_size, extra2][..],
                |idx| (idx[0] + idx[1] + idx[2]) as f64 * 0.3 + 0.5,
            );

            let mut values =
                Tensor::<Complex<f64>, mdarray::DynRank>::zeros(&[extra, n_points, extra2][..]);
            let mut fitted_real =
                Tensor::<f64, mdarray::DynRank>::zeros(&[extra, basis_size, extra2][..]);

            fitter.evaluate_nd_dz_to(None, &coeffs_real.expr(), 1, &mut values.expr_mut());
            fitter.fit_nd_zd_to(None, &values.expr(), 1, &mut fitted_real.expr_mut());

            for i in 0..extra {
                for j in 0..basis_size {
                    for k in 0..extra2 {
                        let error =
                            (coeffs_real[&[i, j, k][..]] - fitted_real[&[i, j, k][..]]).abs();
                        assert!(
                            error < 1e-8,
                            "dz/zd dim=1 (3D) roundtrip error at [{}, {}, {}]: {}",
                            i,
                            j,
                            k,
                            error
                        );
                    }
                }
            }
        }
    }
}
