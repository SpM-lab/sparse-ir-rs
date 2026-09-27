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

        let values_real = fitter_real.evaluate(None, &coeffs).unwrap();
        let values_complex = fitter_complex.evaluate(None, &coeffs).unwrap();

        for (v_real, v_complex) in values_real.iter().zip(values_complex.iter()) {
            assert!((v_real - v_complex.re).abs() < 1e-14, "Real part mismatch");
            assert!(v_complex.im.abs() < 1e-14, "Imaginary part should be ~0");
        }

        let values_complex_zero_im: Vec<Complex<f64>> =
            values_real.iter().map(|&v| Complex::new(v, 0.0)).collect();

        let fitted_real = fitter_real.fit(None, &values_real).unwrap();
        let fitted_complex = fitter_complex.fit(None, &values_complex_zero_im).unwrap();

        for (real, complex) in fitted_real.iter().zip(fitted_complex.iter()) {
            assert!((real - complex).abs() < 1e-12, "Fitted coeffs mismatch");
        }
    }

    // ------------------------------------------------------------------
    // Shape of `out` in the N-D InplaceFitter methods
    // ------------------------------------------------------------------

    use super::common::InplaceFitter;
    use crate::error::{ArrayRole, Error};
    use mdarray::{DenseMapping, DynRank, Shape, Slice, Tensor, ViewMut};

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

    type NdMethod<'f, Tin, Tout> = &'f dyn Fn(
        &Slice<Tin, DynRank>,
        usize,
        &mut ViewMut<'_, Tout, DynRank>,
    ) -> Result<(), Error>;

    /// Call `method` along `dim` of a rank-3 input with extent `n_in` there,
    /// with an output whose extent along `bad_axis` is off by `delta` from
    /// that of the input. The output view is backed by a buffer filled with
    /// a canary and large enough for the correct output, so writes past the
    /// end of the view stay inside the buffer and are detected.
    ///
    /// Returns the result, the correct output shape, the shape of the view
    /// and whether the buffer is untouched.
    fn call_with_mismatched_out<Tin: Canary + Default, Tout: Canary>(
        method: NdMethod<'_, Tin, Tout>,
        dim: usize,
        n_in: usize,
        n_out: usize,
        bad_axis: usize,
        delta: isize,
    ) -> (Result<(), Error>, Vec<usize>, Vec<usize>, bool) {
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
            method(&input, dim, &mut out)
        };
        let untouched = buffer.iter().all(|&x| x == Tout::canary());
        (result, full_shape, out_shape, untouched)
    }

    /// A mismatched non-target extent of `out` must be rejected before
    /// anything is written, as ShapeMismatch of the output with both shapes.
    /// Before PR-0 only the rank and the target extent were checked and a
    /// smaller output was written past its end; PR-0 made it a panic.
    fn assert_rejects_mismatched_out<Tin: Canary + Default, Tout: Canary>(
        name: &str,
        method: NdMethod<'_, Tin, Tout>,
        n_in: usize,
        n_out: usize,
    ) {
        for dim in 0..3 {
            for bad_axis in (0..3).filter(|&axis| axis != dim) {
                for delta in [-1, 1] {
                    let (result, expected, actual, untouched) =
                        call_with_mismatched_out(method, dim, n_in, n_out, bad_axis, delta);
                    let case = format!("{name}: dim={dim}, bad_axis={bad_axis}, delta={delta}");
                    assert_eq!(
                        result,
                        Err(Error::ShapeMismatch {
                            which: ArrayRole::Output,
                            expected,
                            actual,
                        }),
                        "{case}"
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
            f.evaluate_nd_dd_to(None, &coeffs, dim, &mut out.expr_mut())
                .unwrap();
            assert!(out.iter().all(|&x| x != CANARY), "dim={dim}");

            let mut fitted = Tensor::<f64, DynRank>::from_elem(coeffs.shape().dims(), CANARY);
            f.fit_nd_dd_to(None, &out, dim, &mut fitted.expr_mut())
                .unwrap();
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
        assert_eq!(fitter.condition_number().unwrap(), 1.0);
    }

    /// The N-D methods check the axis first, then the input, then `out`, and
    /// write nothing on an error. An unsupported pair of types is
    /// NotSupported (it was `false`).
    #[test]
    fn test_nd_methods_report_the_axis_and_the_input_shape() {
        let f = RealMatrixFitter::new(real_matrix());
        let coeffs = Tensor::<f64, DynRank>::from_elem(&[BASIS_SIZE, 2][..], 1.0);
        let mut out = Tensor::<f64, DynRank>::from_elem(&[N_POINTS, 2][..], CANARY);

        assert_eq!(
            f.evaluate_nd_dd_to(None, &coeffs, 2, &mut out.expr_mut()),
            Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
        );
        let wrong = Tensor::<f64, DynRank>::from_elem(&[BASIS_SIZE + 1, 2][..], 1.0);
        assert_eq!(
            f.evaluate_nd_dd_to(None, &wrong, 0, &mut out.expr_mut()),
            Err(Error::ShapeMismatch {
                which: ArrayRole::Input,
                expected: vec![BASIS_SIZE, 2],
                actual: vec![BASIS_SIZE + 1, 2],
            })
        );
        let mut rank3 = Tensor::<f64, DynRank>::from_elem(&[N_POINTS, 2, 1][..], CANARY);
        assert_eq!(
            f.evaluate_nd_dd_to(None, &coeffs, 0, &mut rank3.expr_mut()),
            Err(Error::ShapeMismatch {
                which: ArrayRole::Output,
                expected: vec![N_POINTS, 2],
                actual: vec![N_POINTS, 2, 1],
            })
        );
        assert!(out.iter().chain(rank3.iter()).all(|&x| x == CANARY));

        let mut out_z = Tensor::<Complex<f64>, DynRank>::zeros(&[N_POINTS, 2][..]);
        let err = InplaceFitter::evaluate_nd_dz_to(&f, None, &coeffs, 0, &mut out_z.expr_mut())
            .unwrap_err();
        assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");

        // An empty batch of the right shape is fine; a wrong empty `out` is not.
        let empty = Tensor::<f64, DynRank>::zeros(&[BASIS_SIZE, 0][..]);
        let mut out0 = Tensor::<f64, DynRank>::zeros(&[N_POINTS, 0][..]);
        f.evaluate_nd_dd_to(None, &empty, 0, &mut out0.expr_mut())
            .unwrap();
        let mut out_bad = Tensor::<f64, DynRank>::zeros(&[N_POINTS + 1, 0][..]);
        assert!(matches!(
            f.evaluate_nd_dd_to(None, &empty, 0, &mut out_bad.expr_mut()),
            Err(Error::ShapeMismatch {
                which: ArrayRole::Output,
                ..
            })
        ));
    }

    /// The SVD of a matrix with a NaN or an infinity does not converge. It
    /// panicked ("SVD computation failed") at the first fit or condition
    /// number; it is DecompositionFailed now, the same on the second call,
    /// and a fit writes nothing. (The public constructors reject such
    /// matrices; the fitters are built directly here.)
    #[test]
    fn test_fitters_report_a_failed_svd() {
        let failed = |err: Error| {
            assert!(
                matches!(&err, Error::DecompositionFailed { reason } if reason.contains("SVD")),
                "{err:?}"
            );
        };

        let mut m = real_matrix();
        m[[1, 1]] = f64::NAN;
        let f = RealMatrixFitter::new(m);
        failed(f.condition_number().unwrap_err());
        failed(f.condition_number().unwrap_err());
        let mut out = vec![CANARY; BASIS_SIZE];
        failed(f.fit_to(None, &[1.0; N_POINTS], &mut out).unwrap_err());
        assert!(out.iter().all(|&x| x == CANARY));
        // Evaluating does not need the SVD.
        assert_eq!(
            f.evaluate(None, &[1.0; BASIS_SIZE]).unwrap().len(),
            N_POINTS
        );

        let mut mc = complex_matrix();
        mc[[2, 0]] = Complex::new(0.0, f64::INFINITY);
        failed(
            ComplexMatrixFitter::new(mc.clone())
                .condition_number()
                .unwrap_err(),
        );
        failed(
            ComplexMatrixFitter::new(mc.clone())
                .fit(None, &[Complex::new(1.0, 0.0); N_POINTS])
                .unwrap_err(),
        );
        failed(
            ComplexToRealFitter::new(&mc)
                .condition_number()
                .unwrap_err(),
        );
        failed(
            ComplexToRealFitter::new(&mc)
                .fit(None, &[Complex::new(1.0, 0.0); N_POINTS])
                .unwrap_err(),
        );

        // The N-D fit writes nothing either.
        let values = Tensor::<f64, DynRank>::from_elem(&[N_POINTS, 2][..], 1.0);
        let mut out_nd = Tensor::<f64, DynRank>::from_elem(&[BASIS_SIZE, 2][..], CANARY);
        failed(
            f.fit_nd_dd_to(None, &values, 0, &mut out_nd.expr_mut())
                .unwrap_err(),
        );
        assert!(out_nd.iter().all(|&x| x == CANARY));
    }

    /// The error of a failed SVD names the matrix. The positive-only
    /// (complex-to-real) fitter decomposes the real 2n x m matrix stacked from
    /// the n x m complex sampling matrix; it reported that stacked matrix as
    /// "the 2n x m sampling matrix", which is not the matrix the user gave.
    #[test]
    fn test_svd_errors_name_the_matrix() {
        let reason = |err: Error| match err {
            Error::DecompositionFailed { reason } => reason,
            err => panic!("{err:?}"),
        };
        let sampling = format!("the SVD of the {N_POINTS} x {BASIS_SIZE} sampling matrix failed: ");

        let mut m = real_matrix();
        m[[1, 1]] = f64::NAN;
        let r = reason(RealMatrixFitter::new(m).condition_number().unwrap_err());
        assert!(r.starts_with(&sampling), "{r}");

        let mut mc = complex_matrix();
        mc[[2, 0]] = Complex::new(0.0, f64::INFINITY);
        let r = reason(
            ComplexMatrixFitter::new(mc.clone())
                .condition_number()
                .unwrap_err(),
        );
        assert!(r.starts_with(&sampling), "{r}");

        let r = reason(
            ComplexToRealFitter::new(&mc)
                .condition_number()
                .unwrap_err(),
        );
        let stacked = format!(
            "the SVD of the {} x {BASIS_SIZE} real matrix stacked from the \
             {N_POINTS} x {BASIS_SIZE} complex sampling matrix failed: ",
            2 * N_POINTS
        );
        assert!(r.starts_with(&stacked), "{r}");
    }

    /// ComplexToRealFitter::evaluate_nd_zz_to reports a wrong axis and a
    /// wrong input shape with the same errors, in the same order, as the
    /// evaluate_nd_dz_to it delegates to.
    #[test]
    fn test_complex_to_real_zz_checks_the_input_first() {
        let f = ComplexToRealFitter::new(&complex_matrix());
        let mut out = Tensor::<Complex<f64>, DynRank>::zeros(&[N_POINTS, 2][..]);
        let coeffs = Tensor::<Complex<f64>, DynRank>::zeros(&[BASIS_SIZE, 2][..]);
        assert_eq!(
            InplaceFitter::evaluate_nd_zz_to(&f, None, &coeffs, 2, &mut out.expr_mut()),
            Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
        );
        let wrong = Tensor::<Complex<f64>, DynRank>::zeros(&[BASIS_SIZE + 1, 2][..]);
        assert_eq!(
            InplaceFitter::evaluate_nd_zz_to(&f, None, &wrong, 0, &mut out.expr_mut()),
            Err(Error::ShapeMismatch {
                which: ArrayRole::Input,
                expected: vec![BASIS_SIZE, 2],
                actual: vec![BASIS_SIZE + 1, 2],
            })
        );
    }
}
