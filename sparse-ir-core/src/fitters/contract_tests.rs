//! Contract of the fitters: dimensions, conditioning, the shape of `out` in
//! the N-D methods, and failed decompositions.

use super::common::InplaceFitter;
use super::*;
use crate::error::{ArrayRole, Error};
use crate::matrix::Mat;
use num_complex::Complex;
use tenferro_tensor::{TypedTensor, TypedTensorView, TypedTensorViewMut};

#[test]
fn test_fitter_dimensions() {
    let n_points = 8;
    let basis_size = 4;

    let matrix_real = Mat::<f64>::from_fn([n_points, basis_size], |idx| (idx[0] + idx[1]) as f64);
    let fitter_real = RealMatrixFitter::new(matrix_real);

    assert_eq!(fitter_real.n_points(), n_points);
    assert_eq!(fitter_real.basis_size(), basis_size);

    let matrix_complex = Mat::<Complex<f64>>::from_fn([n_points, basis_size], |idx| {
        Complex::new((idx[0] + idx[1]) as f64, 0.0)
    });
    let fitter_complex = ComplexToRealFitter::new(matrix_complex);

    assert_eq!(InplaceFitter::n_points(&fitter_complex), n_points);
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

    let matrix_real = Mat::<f64>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        i.powi(j)
    });

    let matrix_complex = Mat::<Complex<f64>>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        Complex::new(i.powi(j), 0.0)
    });

    let fitter_real = RealMatrixFitter::new(matrix_real);
    let fitter_complex = ComplexToRealFitter::new(matrix_complex);

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

const N_POINTS: usize = 5;
const BASIS_SIZE: usize = 3;
const CANARY: f64 = -12345.0;

fn test_matrix(i: usize, j: usize) -> f64 {
    1.0 / (1.0 + i as f64 + 2.0 * j as f64) + if i == j { 1.0 } else { 0.0 }
}

fn real_matrix() -> Mat<f64> {
    Mat::<f64>::from_fn([N_POINTS, BASIS_SIZE], |idx| test_matrix(idx[0], idx[1]))
}

fn complex_matrix() -> Mat<Complex<f64>> {
    Mat::<Complex<f64>>::from_fn([N_POINTS, BASIS_SIZE], |idx| {
        Complex::new(
            test_matrix(idx[0], idx[1]),
            0.5 * test_matrix(idx[1], idx[0]),
        )
    })
}

trait Canary: tenferro_tensor::TensorScalar + Copy + PartialEq + std::fmt::Debug {
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
    &TypedTensorView<'_, Tin>,
    usize,
    &mut TypedTensorViewMut<'_, Tout>,
) -> crate::Result<()>;

/// Column-major strides of `shape`
fn col_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut acc = 1isize;
    for &n in shape {
        strides.push(acc);
        acc *= n.max(1) as isize;
    }
    strides
}

/// A compact column-major tensor of `shape` filled with `value`
fn filled<T: Canary>(shape: &[usize], value: T) -> TypedTensor<T> {
    TypedTensor::from_vec_col_major(shape.to_vec(), vec![value; shape.iter().product()]).unwrap()
}

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
) -> (crate::Result<()>, Vec<usize>, Vec<usize>, bool) {
    let mut in_shape = vec![2usize, 3, 4];
    in_shape[dim] = n_in;
    let input = filled(&in_shape, Tin::default());

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
        let strides = col_major_strides(&out_shape);
        let mut out =
            TypedTensorViewMut::from_slice(out_shape.clone(), strides, 0, &mut buffer).unwrap();
        method(&input.as_view(), dim, &mut out)
    };
    let untouched = buffer.iter().all(|&x| x == Tout::canary());
    (result, full_shape, out_shape, untouched)
}

/// A mismatched non-target extent of `out` must be rejected before
/// anything is written, as ShapeMismatch of the output with both shapes.
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
    let f = ComplexToRealFitter::new(complex_matrix());
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
        let coeffs_data: Vec<f64> = (0..shape.iter().product::<usize>())
            .map(|i| 1.0 + i as f64)
            .collect();
        let coeffs = TypedTensor::from_vec_col_major(shape.clone(), coeffs_data.clone()).unwrap();
        let coeff_shape = shape.clone();
        shape[dim] = N_POINTS;
        let mut out = filled(&shape, CANARY);
        f.evaluate_nd_dd_to(None, &coeffs.as_view(), dim, &mut out.as_view_mut())
            .unwrap();
        assert!(
            out.host_data().unwrap().iter().all(|&x| x != CANARY),
            "dim={dim}"
        );

        let mut fitted = filled(&coeff_shape, CANARY);
        f.fit_nd_dd_to(None, &out.as_view(), dim, &mut fitted.as_view_mut())
            .unwrap();
        for (x, y) in fitted.host_data().unwrap().iter().zip(&coeffs_data) {
            assert!((x - y).abs() < 1e-12 * y.abs(), "dim={dim}: {x} vs {y}");
        }
    }
}

/// The SVD of a matrix with a zero dimension has no singular values. The
/// public constructors reject such matrices; the fitter must still not touch
/// memory out of bounds.
#[test]
fn test_real_fitter_svd_with_zero_columns_does_not_crash() {
    let fitter = RealMatrixFitter::new(Mat::<f64>::zeros([3, 0]));
    // No singular values: the documented convention for a zero dimension
    assert_eq!(fitter.condition_number().unwrap(), 1.0);
}

/// The N-D methods check the axis first, then the input, then `out`, and
/// write nothing on an error. An unsupported pair of types is NotSupported.
#[test]
fn test_nd_methods_report_the_axis_and_the_input_shape() {
    let f = RealMatrixFitter::new(real_matrix());
    let coeffs = filled(&[BASIS_SIZE, 2], 1.0);
    let mut out = filled(&[N_POINTS, 2], CANARY);

    assert_eq!(
        f.evaluate_nd_dd_to(None, &coeffs.as_view(), 2, &mut out.as_view_mut()),
        Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
    );
    let wrong = filled(&[BASIS_SIZE + 1, 2], 1.0);
    assert_eq!(
        f.evaluate_nd_dd_to(None, &wrong.as_view(), 0, &mut out.as_view_mut()),
        Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected: vec![BASIS_SIZE, 2],
            actual: vec![BASIS_SIZE + 1, 2],
        })
    );
    let mut rank3 = filled(&[N_POINTS, 2, 1], CANARY);
    assert_eq!(
        f.evaluate_nd_dd_to(None, &coeffs.as_view(), 0, &mut rank3.as_view_mut()),
        Err(Error::ShapeMismatch {
            which: ArrayRole::Output,
            expected: vec![N_POINTS, 2],
            actual: vec![N_POINTS, 2, 1],
        })
    );
    assert!(
        out.host_data()
            .unwrap()
            .iter()
            .chain(rank3.host_data().unwrap())
            .all(|&x| x == CANARY)
    );

    let mut out_z = filled(&[N_POINTS, 2], Complex::new(0.0, 0.0));
    let err =
        InplaceFitter::evaluate_nd_dz_to(&f, None, &coeffs.as_view(), 0, &mut out_z.as_view_mut())
            .unwrap_err();
    assert!(matches!(err, Error::NotSupported { .. }), "{err:?}");

    // An empty batch of the right shape is fine; a wrong empty `out` is not.
    let empty = filled(&[BASIS_SIZE, 0], 0.0);
    let mut out0 = filled(&[N_POINTS, 0], 0.0);
    f.evaluate_nd_dd_to(None, &empty.as_view(), 0, &mut out0.as_view_mut())
        .unwrap();
    let mut out_bad = filled(&[N_POINTS + 1, 0], 0.0);
    assert!(matches!(
        f.evaluate_nd_dd_to(None, &empty.as_view(), 0, &mut out_bad.as_view_mut()),
        Err(Error::ShapeMismatch {
            which: ArrayRole::Output,
            ..
        })
    ));
}

/// The SVD of a matrix with a NaN or an infinity does not converge. It is
/// DecompositionFailed, the same on the second call, and a fit writes
/// nothing. (The public constructors reject such matrices; the fitters are
/// built directly here.)
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
        ComplexToRealFitter::new(mc.clone())
            .condition_number()
            .unwrap_err(),
    );
    failed(
        ComplexToRealFitter::new(mc)
            .fit(None, &[Complex::new(1.0, 0.0); N_POINTS])
            .unwrap_err(),
    );

    // The N-D fit writes nothing either.
    let values = filled(&[N_POINTS, 2], 1.0);
    let mut out_nd = filled(&[BASIS_SIZE, 2], CANARY);
    failed(
        f.fit_nd_dd_to(None, &values.as_view(), 0, &mut out_nd.as_view_mut())
            .unwrap_err(),
    );
    assert!(out_nd.host_data().unwrap().iter().all(|&x| x == CANARY));
}

/// The error of a failed SVD names the matrix. The positive-only
/// (complex-to-real) fitter decomposes the real 2n x m matrix stacked from
/// the n x m complex sampling matrix, and names both.
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

    let r = reason(ComplexToRealFitter::new(mc).condition_number().unwrap_err());
    let stacked = format!(
        "the SVD of the {} x {BASIS_SIZE} real matrix stacked from the \
         {N_POINTS} x {BASIS_SIZE} complex sampling matrix failed: ",
        2 * N_POINTS
    );
    assert!(r.starts_with(&stacked), "{r}");
}

/// ComplexToRealFitter::evaluate_nd_zz_to reports a wrong axis and a wrong
/// input shape with the same errors, in the same order, as the
/// evaluate_nd_dz_to it delegates to.
#[test]
fn test_complex_to_real_zz_checks_the_input_first() {
    let f = ComplexToRealFitter::new(complex_matrix());
    let zero = Complex::new(0.0, 0.0);
    let mut out = filled(&[N_POINTS, 2], zero);
    let coeffs = filled(&[BASIS_SIZE, 2], zero);
    assert_eq!(
        InplaceFitter::evaluate_nd_zz_to(&f, None, &coeffs.as_view(), 2, &mut out.as_view_mut()),
        Err(Error::AxisOutOfRange { axis: 2, rank: 2 })
    );
    let wrong = filled(&[BASIS_SIZE + 1, 2], zero);
    assert_eq!(
        InplaceFitter::evaluate_nd_zz_to(&f, None, &wrong.as_view(), 0, &mut out.as_view_mut()),
        Err(Error::ShapeMismatch {
            which: ArrayRole::Input,
            expected: vec![BASIS_SIZE, 2],
            actual: vec![BASIS_SIZE + 1, 2],
        })
    );
}

// ------------------------------------------------------------------
// Tests carried over from the in-file tests of the mdarray fitters
// ------------------------------------------------------------------

/// Real coefficients evaluated by the positive-only fitter are recovered by
/// the general complex fitter, with vanishing imaginary parts.
#[test]
fn test_complex_fitter_vs_complex_to_real() {
    let (n_points, basis_size) = (8, 4);
    let matrix = Mat::<Complex<f64>>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        Complex::new(i.powi(j), (i * (j as f64) * 0.1).sin())
    });
    let fitter_c2r = ComplexToRealFitter::new(matrix.clone());
    let fitter_complex = ComplexMatrixFitter::new(matrix);

    let coeffs_real: Vec<f64> = (0..basis_size).map(|i| i as f64 * 0.4).collect();
    let values = fitter_c2r.evaluate(None, &coeffs_real).unwrap();
    let fitted_complex = fitter_complex.fit(None, &values).unwrap();
    for (i, &c) in coeffs_real.iter().enumerate() {
        assert!((c - fitted_complex[i].re).abs() < 1e-10, "real part at {i}");
        assert!(fitted_complex[i].im.abs() < 1e-10, "imaginary part at {i}");
    }
}

/// Real coefficients -> complex values -> real coefficients through the
/// in-place dz/zd methods of the complex fitter, along every axis of a rank-2
/// and the middle axis of a rank-3 array.
#[test]
fn test_complex_fitter_dz_zd_inplace_roundtrip() {
    let (n_points, basis_size, extra) = (8, 4, 3);
    let matrix = Mat::<Complex<f64>>::from_fn([n_points, basis_size], |idx| {
        let i = idx[0] as f64 / (n_points as f64);
        let j = idx[1] as i32;
        let mag = i.powi(j);
        let phase = (j as f64) * 0.5;
        Complex::new(mag * phase.cos(), mag * phase.sin())
    });
    let fitter = ComplexMatrixFitter::new(matrix);

    for (shape, dim) in [
        (vec![basis_size, extra], 0),
        (vec![extra, basis_size], 1),
        (vec![extra, basis_size, 2], 1),
    ] {
        let len: usize = shape.iter().product();
        let coeffs_data: Vec<f64> = (0..len).map(|k| 0.5 + 0.3 * k as f64).collect();
        let coeffs = TypedTensor::from_vec_col_major(shape.clone(), coeffs_data.clone()).unwrap();
        let mut values_shape = shape.clone();
        values_shape[dim] = n_points;
        let mut values = filled(&values_shape, Complex::new(0.0, 0.0));
        let mut fitted = filled(&shape, 0.0);
        fitter
            .evaluate_nd_dz_to(None, &coeffs.as_view(), dim, &mut values.as_view_mut())
            .unwrap();
        fitter
            .fit_nd_zd_to(None, &values.as_view(), dim, &mut fitted.as_view_mut())
            .unwrap();
        for (x, y) in fitted.host_data().unwrap().iter().zip(&coeffs_data) {
            assert!(
                (x - y).abs() < 1e-8,
                "shape {shape:?}, dim {dim}: {x} vs {y}"
            );
        }
    }
}

/// Least squares on a tall (20 x 5) system recovers exact coefficients.
#[test]
fn test_overdetermined_roundtrips() {
    let (n_points, basis_size) = (20, 5);
    let coeffs: Vec<f64> = (0..basis_size).map(|i| (i as f64) * 0.3).collect();

    let real = RealMatrixFitter::new(Mat::<f64>::from_fn([n_points, basis_size], |idx| {
        ((idx[0] as f64 + 1.0) / (n_points as f64)).powi(idx[1] as i32)
    }));
    let fitted = real
        .fit(None, &real.evaluate(None, &coeffs).unwrap())
        .unwrap();
    for (orig, f) in coeffs.iter().zip(&fitted) {
        assert!((orig - f).abs() < 1e-10, "real fitter: {orig} vs {f}");
    }

    let c2r = ComplexToRealFitter::new(Mat::<Complex<f64>>::from_fn(
        [n_points, basis_size],
        |idx| {
            let (i, j) = (idx[0] as f64, idx[1] as f64);
            let phase = 2.0 * std::f64::consts::PI * i * j / (n_points as f64);
            Complex::new(phase.cos(), phase.sin()) / (j + 1.0)
        },
    ));
    let fitted = c2r
        .fit(None, &c2r.evaluate(None, &coeffs).unwrap())
        .unwrap();
    for (orig, f) in coeffs.iter().zip(&fitted) {
        assert!(
            (orig - f).abs() < 1e-10,
            "complex-to-real fitter: {orig} vs {f}"
        );
    }
}

/// A 2x3 times 3x1 product (non-square operands)
#[test]
fn test_matmul_non_square() {
    let a = crate::matrix::mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_typed();
    let b = crate::matrix::mat![[7.0], [8.0], [9.0]].into_typed();
    let c = crate::gemm::matmul(None, &a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 1]);
    assert!((c.get(&[0, 0]).unwrap() - 50.0).abs() < 1e-10);
    assert!((c.get(&[1, 0]).unwrap() - 122.0).abs() < 1e-10);
}
