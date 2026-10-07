#[allow(unused_imports)]
use super::C64;
use super::*;
use crate::fitters::test_support::*;

const N: usize = 8;
const M: usize = 5;

#[test]
fn vector_roundtrip() {
    let a = complex_matrix(N, M);
    let fitter = ComplexMatrixFitter::new(a.clone());

    let z = complex_data(M, 1);
    let y = fitter.evaluate(None, &z).unwrap();
    assert!(max_diff(&y, &naive_apply(a.as_slice(), N, M, &z, &[M], 0)) < 1e-13);
    assert!(max_diff(&fitter.fit(None, &y).unwrap(), &z) < 1e-12);

    let x = real_data(M, 2);
    let y = fitter.evaluate_real(None, &x).unwrap();
    assert!(max_diff(&y, &naive_apply(a.as_slice(), N, M, &x, &[M], 0)) < 1e-13);
    assert!(max_diff(&fitter.fit_real(None, &y).unwrap(), &x) < 1e-12);
}

#[test]
fn nd_all_dims() {
    let a = complex_matrix(N, M);
    let fitter = ComplexMatrixFitter::new(a.clone());
    for dim in 0..3 {
        let mut shape = vec![2, 3, 4];
        shape[dim] = M;
        let len: usize = shape.iter().product();

        let z = complex_data(len, dim);
        let zt = TypedTensor::from_vec_col_major(shape.clone(), z.clone()).unwrap();
        let y = fitter.evaluate_nd_zz(None, &zt, dim).unwrap();
        let expected = naive_apply(a.as_slice(), N, M, &z, &shape, dim);
        assert!(max_diff(y.host_data().unwrap(), &expected) < 1e-13);
        let back = fitter.fit_nd_zz(None, &y, dim).unwrap();
        assert!(max_diff(back.host_data().unwrap(), &z) < 1e-12);

        let x = real_data(len, dim);
        let xt = TypedTensor::from_vec_col_major(shape.clone(), x.clone()).unwrap();
        let y = fitter.evaluate_nd_dz(None, &xt, dim).unwrap();
        let expected = naive_apply(a.as_slice(), N, M, &x, &shape, dim);
        assert!(max_diff(y.host_data().unwrap(), &expected) < 1e-13);
        let back = fitter.fit_nd_zd(None, &y, dim).unwrap();
        assert!(max_diff(back.host_data().unwrap(), &x) < 1e-12);
    }
}

#[test]
fn inplace_row_major_input() {
    let a = complex_matrix(N, M);
    let fitter = ComplexMatrixFitter::new(a.clone());
    let shape = [M, 3];
    let x = real_data(M * 3, 4);
    let (x_rm, strides) = to_row_major(&x, &shape);
    let view = TypedTensorView::from_slice(shape, strides, 0, &x_rm).unwrap();
    let mut y = vec![C64::new(0.0, 0.0); N * 3];
    let mut out = TypedTensorViewMut::from_slice([N, 3], [1, N as isize], 0, &mut y).unwrap();
    InplaceFitter::evaluate_nd_dz_to(&fitter, None, &view, 0, &mut out).unwrap();
    let expected = naive_apply(a.as_slice(), N, M, &x, &shape, 0);
    assert!(max_diff(&y, &expected) < 1e-13);

    let mut back = vec![0.0; M * 3];
    let yview = TypedTensorView::from_slice([N, 3], [1, N as isize], 0, &y).unwrap();
    let mut out = TypedTensorViewMut::from_slice([M, 3], [1, M as isize], 0, &mut back).unwrap();
    InplaceFitter::fit_nd_zd_to(&fitter, None, &yview, 0, &mut out).unwrap();
    assert!(max_diff(&back, &x) < 1e-12);
}
