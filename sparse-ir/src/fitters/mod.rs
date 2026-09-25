//! Fitters for least-squares problems with various matrix types
//!
//! This module provides fitters for solving min ||A * coeffs - values||^2
//! where the matrix A and value types can vary:
//!
//! - [`RealMatrixFitter`]: Real matrix A ∈ R^{n×m}
//! - [`ComplexToRealFitter`]: Complex matrix A ∈ C^{n×m}, real coefficients
//! - [`ComplexMatrixFitter`]: Complex matrix A ∈ C^{n×m}, complex coefficients

pub mod common;
mod complex;
mod complex_to_real;
mod real;

pub use common::InplaceFitter;
pub(crate) use complex::ComplexMatrixFitter;
pub(crate) use complex_to_real::ComplexToRealFitter;
pub(crate) use real::RealMatrixFitter;

#[cfg(test)]
mod tests {
    use super::*;
    use mdarray::DTensor;
    use num_complex::Complex;

    #[test]
    fn test_fitter_dimensions() {
        let n_points = 8;
        let basis_size = 4;

        let matrix_real =
            DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| (idx[0] + idx[1]) as f64);
        let fitter_real = RealMatrixFitter::new(matrix_real);

        assert_eq!(fitter_real.n_points(), n_points);
        assert_eq!(fitter_real.basis_size(), basis_size);

        let matrix_complex = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            Complex::new((idx[0] + idx[1]) as f64, 0.0)
        });
        let fitter_complex = ComplexToRealFitter::new(&matrix_complex);

        assert_eq!(fitter_complex.n_points(), n_points);
        assert_eq!(fitter_complex.basis_size(), basis_size);
    }

    #[test]
    fn test_condition_number_from_singular_values_conventions() {
        use super::common::condition_number_from_singular_values as cond;

        assert_eq!(cond(&[4.0, 2.0, 0.5]), 8.0);
        // Independent of the order of the singular values
        assert_eq!(cond(&[0.5, 4.0, 2.0]), 8.0);
        assert_eq!(cond(&[3.0]), 1.0);
        // No singular values (a matrix with a zero dimension)
        assert_eq!(cond(&[]), 1.0);
        // Numerically singular
        assert_eq!(cond(&[1.0, 1e-16]), f64::INFINITY);
        assert_eq!(cond(&[1.0, 0.0]), f64::INFINITY);
        // A NaN singular value is never turned into a finite value
        assert!(cond(&[1.0, f64::NAN]).is_nan());
        assert!(cond(&[f64::NAN, 1.0]).is_nan());
    }

    #[test]
    fn test_complex_fitter_real_matrix_equivalence() {
        let n_points = 8;
        let basis_size = 4;

        let matrix_real = DTensor::<f64, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            i.powi(j)
        });

        let matrix_complex = DTensor::<Complex<f64>, 2>::from_fn([n_points, basis_size], |idx| {
            let i = idx[0] as f64 / (n_points as f64);
            let j = idx[1] as i32;
            Complex::new(i.powi(j), 0.0)
        });

        let fitter_real = RealMatrixFitter::new(matrix_real);
        let fitter_complex = ComplexToRealFitter::new(&matrix_complex);

        let coeffs: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();

        let values_real = fitter_real.evaluate(None, &coeffs);
        let values_complex = fitter_complex.evaluate(None, &coeffs);

        for (v_real, v_complex) in values_real.iter().zip(values_complex.iter()) {
            assert!((v_real - v_complex.re).abs() < 1e-14, "Real part mismatch");
            assert!(v_complex.im.abs() < 1e-14, "Imaginary part should be ~0");
        }

        let values_complex_zero_im: Vec<Complex<f64>> =
            values_real.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let fitted_real = fitter_real.fit(None, &values_real);
        let fitted_complex = fitter_complex.fit(None, &values_complex_zero_im);

        for (real, complex) in fitted_real.iter().zip(fitted_complex.iter()) {
            assert!((real - complex).abs() < 1e-12, "Fitted coeffs mismatch");
        }
    }

    // ------------------------------------------------------------------
    // Shape of `out` in the N-D InplaceFitter methods
    // ------------------------------------------------------------------

    use super::common::InplaceFitter;
    use mdarray::{DenseMapping, DynRank, Shape, Slice, Tensor, ViewMut};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    const N_POINTS: usize = 5;
    const BASIS_SIZE: usize = 3;
    const CANARY: f64 = -12345.0;

    fn test_matrix(i: usize, j: usize) -> f64 {
        1.0 / (1.0 + i as f64 + 2.0 * j as f64) + if i == j { 1.0 } else { 0.0 }
    }

    fn real_matrix() -> DTensor<f64, 2> {
        DTensor::<f64, 2>::from_fn([N_POINTS, BASIS_SIZE], |idx| test_matrix(idx[0], idx[1]))
    }

    fn complex_matrix() -> DTensor<Complex<f64>, 2> {
        DTensor::<Complex<f64>, 2>::from_fn([N_POINTS, BASIS_SIZE], |idx| {
            Complex::new(
                test_matrix(idx[0], idx[1]),
                0.5 * test_matrix(idx[1], idx[0]),
            )
        })
    }

    trait Canary: Copy + PartialEq + std::fmt::Debug {
        fn canary() -> Self;
    }
    impl Canary for f64 {
        fn canary() -> Self {
            CANARY
        }
    }
    impl Canary for Complex<f64> {
        fn canary() -> Self {
            Complex::new(CANARY, -CANARY)
        }
    }

    type NdMethod<'f, Tin, Tout> =
        &'f dyn Fn(&Slice<Tin, DynRank>, usize, &mut ViewMut<'_, Tout, DynRank>) -> bool;

    /// Call `method` along `dim` of a rank-3 input with extent `n_in` there,
    /// with an output whose extent along `bad_axis` is off by `delta` from
    /// that of the input. The output view is backed by a buffer filled with
    /// a canary and large enough for the correct output, so writes past the
    /// end of the view stay inside the buffer and are detected.
    ///
    /// Returns the panic message (or `None` if `method` returned) and whether
    /// the buffer is untouched.
    fn call_with_mismatched_out<Tin: Canary + Default, Tout: Canary>(
        method: NdMethod<'_, Tin, Tout>,
        dim: usize,
        n_in: usize,
        n_out: usize,
        bad_axis: usize,
        delta: isize,
    ) -> (Option<String>, bool) {
        let mut in_shape = vec![2usize, 3, 4];
        in_shape[dim] = n_in;
        let input = Tensor::<Tin, DynRank>::from_elem(&in_shape[..], Tin::default());

        let mut out_shape = in_shape.clone();
        out_shape[dim] = n_out;
        out_shape[bad_axis] = out_shape[bad_axis].checked_add_signed(delta).unwrap();
        let mut full_shape = in_shape.clone();
        full_shape[dim] = n_out;
        let capacity = full_shape
            .iter()
            .product::<usize>()
            .max(out_shape.iter().product());
        let mut buffer = vec![Tout::canary(); capacity];

        let result = {
            // SAFETY: the view covers the first `out_shape.product()` elements
            // of `buffer`, which holds at least that many.
            let mut out = unsafe {
                ViewMut::<'_, Tout, DynRank>::new_unchecked(
                    buffer.as_mut_ptr(),
                    DenseMapping::new(DynRank::from_dims(&out_shape[..])),
                )
            };
            catch_unwind(AssertUnwindSafe(|| method(&input, dim, &mut out)))
        };
        let message = match result {
            Ok(_) => None,
            Err(payload) => Some(
                payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default(),
            ),
        };
        let untouched = buffer.iter().all(|&x| x == Tout::canary());
        (message, untouched)
    }

    /// A mismatched non-target extent of `out` must be rejected before
    /// anything is written. Before the fix only the rank and the target
    /// extent were checked: the unchecked views were sized from the input,
    /// so a smaller output was written past its end (and, for the zd fit of
    /// `ComplexMatrixFitter`, a larger one was read past a temporary).
    fn assert_rejects_mismatched_out<Tin: Canary + Default, Tout: Canary>(
        name: &str,
        method: NdMethod<'_, Tin, Tout>,
        n_in: usize,
        n_out: usize,
    ) {
        for dim in 0..3 {
            for bad_axis in (0..3).filter(|&axis| axis != dim) {
                for delta in [-1, 1] {
                    let (message, untouched) =
                        call_with_mismatched_out(method, dim, n_in, n_out, bad_axis, delta);
                    let case = format!("{name}: dim={dim}, bad_axis={bad_axis}, delta={delta}");
                    let message = message.unwrap_or_else(|| panic!("{case}: not rejected"));
                    assert!(
                        message.contains(&format!("out.shape().dim({bad_axis})=")),
                        "{case}: unexpected panic {message:?}"
                    );
                    assert!(untouched, "{case}: output buffer was written");
                }
            }
        }
    }

    #[test]
    fn test_real_fitter_rejects_mismatched_out() {
        let f = RealMatrixFitter::new(real_matrix());
        let b = None;
        assert_rejects_mismatched_out::<f64, f64>(
            "RealMatrixFitter::evaluate_nd_dd_to",
            &|c, d, o| f.evaluate_nd_dd_to(b, c, d, o),
            BASIS_SIZE,
            N_POINTS,
        );
        assert_rejects_mismatched_out::<Complex<f64>, Complex<f64>>(
            "RealMatrixFitter::evaluate_nd_zz_to",
            &|c, d, o| f.evaluate_nd_zz_to(b, c, d, o),
            BASIS_SIZE,
            N_POINTS,
        );
        assert_rejects_mismatched_out::<f64, f64>(
            "RealMatrixFitter::fit_nd_dd_to",
            &|v, d, o| f.fit_nd_dd_to(b, v, d, o),
            N_POINTS,
            BASIS_SIZE,
        );
        assert_rejects_mismatched_out::<Complex<f64>, Complex<f64>>(
            "RealMatrixFitter::fit_nd_zz_to",
            &|v, d, o| f.fit_nd_zz_to(b, v, d, o),
            N_POINTS,
            BASIS_SIZE,
        );
    }

    #[test]
    fn test_complex_fitter_rejects_mismatched_out() {
        let f = ComplexMatrixFitter::new(complex_matrix());
        let b = None;
        assert_rejects_mismatched_out::<Complex<f64>, Complex<f64>>(
            "ComplexMatrixFitter::evaluate_nd_zz_to",
            &|c, d, o| f.evaluate_nd_zz_to(b, c, d, o),
            BASIS_SIZE,
            N_POINTS,
        );
        assert_rejects_mismatched_out::<f64, Complex<f64>>(
            "ComplexMatrixFitter::evaluate_nd_dz_to",
            &|c, d, o| f.evaluate_nd_dz_to(b, c, d, o),
            BASIS_SIZE,
            N_POINTS,
        );
        assert_rejects_mismatched_out::<Complex<f64>, Complex<f64>>(
            "ComplexMatrixFitter::fit_nd_zz_to",
            &|v, d, o| f.fit_nd_zz_to(b, v, d, o),
            N_POINTS,
            BASIS_SIZE,
        );
        assert_rejects_mismatched_out::<Complex<f64>, f64>(
            "ComplexMatrixFitter::fit_nd_zd_to",
            &|v, d, o| f.fit_nd_zd_to(b, v, d, o),
            N_POINTS,
            BASIS_SIZE,
        );
    }

    #[test]
    fn test_complex_to_real_fitter_rejects_mismatched_out() {
        let f = ComplexToRealFitter::new(&complex_matrix());
        let b = None;
        assert_rejects_mismatched_out::<f64, Complex<f64>>(
            "ComplexToRealFitter::evaluate_nd_dz_to",
            &|c, d, o| f.evaluate_nd_dz_to(b, c, d, o),
            BASIS_SIZE,
            N_POINTS,
        );
        assert_rejects_mismatched_out::<Complex<f64>, Complex<f64>>(
            "ComplexToRealFitter::evaluate_nd_zz_to",
            &|c, d, o| InplaceFitter::evaluate_nd_zz_to(&f, b, c, d, o),
            BASIS_SIZE,
            N_POINTS,
        );
        assert_rejects_mismatched_out::<Complex<f64>, f64>(
            "ComplexToRealFitter::fit_nd_zd_to",
            &|v, d, o| f.fit_nd_zd_to(b, v, d, o),
            N_POINTS,
            BASIS_SIZE,
        );
        assert_rejects_mismatched_out::<Complex<f64>, Complex<f64>>(
            "ComplexToRealFitter::fit_nd_zz_to",
            &|v, d, o| InplaceFitter::fit_nd_zz_to(&f, b, v, d, o),
            N_POINTS,
            BASIS_SIZE,
        );
    }

    /// With the right shape the methods still succeed and write every element
    #[test]
    fn test_nd_methods_accept_matching_out() {
        let f = RealMatrixFitter::new(real_matrix());
        for dim in 0..3 {
            let mut shape = vec![2usize, 3, 4];
            shape[dim] = BASIS_SIZE;
            let coeffs = Tensor::<f64, DynRank>::from_fn(&shape[..], |idx| {
                1.0 + idx[0] as f64 + 10.0 * idx[1] as f64 + 100.0 * idx[2] as f64
            });
            shape[dim] = N_POINTS;
            let mut out = Tensor::<f64, DynRank>::from_elem(&shape[..], CANARY);
            assert!(f.evaluate_nd_dd_to(None, &coeffs, dim, &mut out.expr_mut()));
            assert!(out.iter().all(|&x| x != CANARY), "dim={dim}");

            let mut fitted = Tensor::<f64, DynRank>::from_elem(coeffs.shape().dims(), CANARY);
            assert!(f.fit_nd_dd_to(None, &out, dim, &mut fitted.expr_mut()));
            for (x, y) in fitted.iter().zip(coeffs.iter()) {
                assert!((x - y).abs() < 1e-12 * y.abs(), "dim={dim}: {x} vs {y}");
            }
        }
    }

    /// The SVD of a matrix with a zero dimension has no singular values.
    /// Before the fix, transposing its [n, 0] factor U segfaulted in mdarray
    /// 0.7.2 (https://github.com/fre-hu/mdarray/issues/21). The public
    /// constructors now reject such matrices; the SVD of the fitter must still
    /// not touch memory out of bounds.
    #[test]
    fn test_real_fitter_svd_with_zero_columns_does_not_crash() {
        let fitter = RealMatrixFitter::new(DTensor::<f64, 2>::zeros([3, 0]));
        // No singular values: the documented convention for a zero dimension
        assert_eq!(fitter.condition_number(), 1.0);
    }
}
