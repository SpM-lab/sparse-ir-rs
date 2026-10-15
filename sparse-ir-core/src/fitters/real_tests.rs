use super::*;
use crate::error::Error;
use crate::fitters::test_support::*;

const N: usize = 9;
const M: usize = 5;

#[test]
fn vector_roundtrip_real_and_complex() {
    let fitter = RealMatrixFitter::new(real_matrix(N, M));
    let x = real_data(M, 0);
    let y = fitter.evaluate(None, &x).unwrap();
    let a = real_matrix(N, M);
    assert!(max_diff(&y, &naive_apply(a.as_slice(), N, M, &x, &[M], 0)) < 1e-13);
    assert!(max_diff(&fitter.fit(None, &y).unwrap(), &x) < 1e-12);

    let z = complex_data(M, 3);
    let yz = fitter.evaluate(None, &z).unwrap();
    assert!(max_diff(&yz, &naive_apply(a.as_slice(), N, M, &z, &[M], 0)) < 1e-13);
    assert!(max_diff(&fitter.fit(None, &yz).unwrap(), &z) < 1e-12);
}

#[test]
fn vector_length_mismatch_is_error() {
    let fitter = RealMatrixFitter::new(real_matrix(N, M));
    assert!(matches!(
        fitter.evaluate(None, &[1.0; M + 1]),
        Err(Error::ShapeMismatch { .. })
    ));
    assert!(matches!(
        fitter.fit(None, &[1.0; N - 1]),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn nd_all_dims_real_and_complex() {
    let a = real_matrix(N, M);
    let fitter = RealMatrixFitter::new(a.clone());
    for dim in 0..3 {
        let mut shape = vec![3, 4, 2];
        shape[dim] = M;
        let len: usize = shape.iter().product();

        let x = real_data(len, dim);
        let xt = TypedTensor::from_vec_col_major(shape.clone(), x.clone()).unwrap();
        let yt = fitter.evaluate_nd(None, &xt, dim).unwrap();
        let expected = naive_apply(a.as_slice(), N, M, &x, &shape, dim);
        assert!(max_diff(yt.host_data().unwrap(), &expected) < 1e-13);
        let back = fitter.fit_nd(None, &yt, dim).unwrap();
        assert!(max_diff(back.host_data().unwrap(), &x) < 1e-12);

        let z = complex_data(len, dim);
        let zt = TypedTensor::from_vec_col_major(shape.clone(), z.clone()).unwrap();
        let yz = fitter.evaluate_nd(None, &zt, dim).unwrap();
        let expected = naive_apply(a.as_slice(), N, M, &z, &shape, dim);
        assert!(max_diff(yz.host_data().unwrap(), &expected) < 1e-13);
        let back = fitter.fit_nd(None, &yz, dim).unwrap();
        assert!(max_diff(back.host_data().unwrap(), &z) < 1e-12);
    }
}

#[test]
fn inplace_accepts_row_major_input() {
    let a = real_matrix(N, M);
    let fitter = RealMatrixFitter::new(a.clone());
    let shape = [3, M, 2];
    let x = real_data(shape.iter().product(), 7);
    let (x_rm, strides) = to_row_major(&x, &shape);
    let view = TypedTensorView::from_slice(shape, strides, 0, &x_rm).unwrap();
    let mut y = vec![0.0; 3 * N * 2];
    let mut out =
        TypedTensorViewMut::from_slice([3, N, 2], [1, 3, 3 * N as isize], 0, &mut y).unwrap();
    InplaceFitter::evaluate_nd_dd_to(&fitter, None, &view, 1, &mut out).unwrap();
    let expected = naive_apply(a.as_slice(), N, M, &x, &shape, 1);
    assert!(max_diff(&y, &expected) < 1e-13);
}

#[test]
fn inplace_rejects_row_major_output_and_bad_shape() {
    let fitter = RealMatrixFitter::new(real_matrix(N, M));
    let x = real_data(3 * M, 0);
    let view = TypedTensorView::from_slice([3, M], [1, 3], 0, &x).unwrap();
    let mut y = vec![0.0; 3 * N];
    let mut out = TypedTensorViewMut::from_slice([3, N], [N as isize, 1], 0, &mut y).unwrap();
    assert!(matches!(
        fitter.evaluate_nd_to(None, &view, 1, &mut out),
        Err(Error::NotSupported { .. })
    ));
    let mut out = TypedTensorViewMut::from_slice([3, N - 1], [1, 3], 0, &mut y).unwrap();
    assert!(matches!(
        fitter.evaluate_nd_to(None, &view, 1, &mut out),
        Err(Error::ShapeMismatch { .. })
    ));
    let mut out = TypedTensorViewMut::from_slice([3, N], [1, 3], 0, &mut y).unwrap();
    assert!(matches!(
        fitter.evaluate_nd_to(None, &view, 2, &mut out),
        Err(Error::AxisOutOfRange { .. })
    ));
}

#[test]
fn unsupported_combination_is_reported() {
    let fitter = RealMatrixFitter::new(real_matrix(N, M));
    let x = real_data(M, 0);
    let view = TypedTensorView::from_slice([M], [1], 0, &x).unwrap();
    let mut y = vec![C64::new(0.0, 0.0); N];
    let mut out = TypedTensorViewMut::from_slice([N], [1], 0, &mut y).unwrap();
    assert!(matches!(
        InplaceFitter::evaluate_nd_dz_to(&fitter, None, &view, 0, &mut out),
        Err(Error::NotSupported { .. })
    ));
}

/// A rank-deficient matrix (two equal columns) gives the minimum-norm
/// least-squares solution, not inf/NaN; so does the zero matrix.
#[test]
fn fit_rank_deficient_matrix() {
    let a = Mat::<f64>::from_fn([5, 2], |idx| (idx[0] + 1) as f64);
    let fitter = RealMatrixFitter::new(a);
    let values: Vec<f64> = (1..=5).map(|i| i as f64).collect();
    let coeffs = fitter.fit(None, &values).unwrap();
    assert!((coeffs[0] - 0.5).abs() < 1e-14 && (coeffs[1] - 0.5).abs() < 1e-14);

    let zero = RealMatrixFitter::new(Mat::<f64>::from_elem([4, 3], 0.0));
    assert_eq!(zero.fit(None, &[1.0, 2.0, 3.0, 4.0]).unwrap(), vec![0.0; 3]);
}

/// The rank cut `s_l > rtol * s_0` is false for every l when s_0 is NaN; it
/// must fail instead of giving a rank-0 pseudo-inverse whose fits are zeros.
/// (The SVD backend reports non-convergence for a non-finite matrix before
/// this check, so it is tested directly.)
#[test]
fn pinv_rank_rejects_non_finite_singular_values() {
    use crate::fitters::common::pinv_rank;
    assert_eq!(pinv_rank(&[2.0, 1.0, 1e-20], 1e-15), Ok(2));
    assert_eq!(pinv_rank(&[], 1e-15), Ok(0));
    for bad in [f64::NAN, f64::INFINITY] {
        assert_eq!(
            pinv_rank(&[bad, 1.0], 1e-15).unwrap_err(),
            format!("non-finite singular value s[0] = {bad}")
        );
        assert!(pinv_rank(&[2.0, bad], 1e-15).is_err());
    }
}

/// A fit with a non-finite matrix fails instead of returning zeros.
#[test]
fn fit_non_finite_matrix_is_error() {
    for bad in [f64::NAN, f64::INFINITY] {
        let mut a = real_matrix(N, M);
        a[[2, 1]] = bad;
        let fitter = RealMatrixFitter::new(a);
        assert!(
            matches!(
                fitter.fit(None, &real_data(N, 0)),
                Err(Error::DecompositionFailed { .. })
            ),
            "{bad}"
        );
    }
}
