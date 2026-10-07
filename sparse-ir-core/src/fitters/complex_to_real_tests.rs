#[allow(unused_imports)]
use super::C64;
use super::*;
use crate::fitters::test_support::*;

const N: usize = 4;
const M: usize = 7; // 2N >= M: the stacked real problem is overdetermined.

#[test]
fn vector_roundtrip() {
    let a = complex_matrix(N, M);
    let fitter = ComplexToRealFitter::new(a.clone());
    let x = real_data(M, 0);
    let y = fitter.evaluate(None, &x).unwrap();
    assert!(max_diff(&y, &naive_apply(a.as_slice(), N, M, &x, &[M], 0)) < 1e-13);
    assert!(y.iter().any(|z| z.im.abs() > 1e-3));
    assert!(max_diff(&fitter.fit(None, &y).unwrap(), &x) < 1e-11);
}

#[test]
fn nd_all_dims() {
    let a = complex_matrix(N, M);
    let fitter = ComplexToRealFitter::new(a.clone());
    for dim in 0..3 {
        let mut shape = vec![2, 3, 5];
        shape[dim] = M;
        let len: usize = shape.iter().product();
        let x = real_data(len, dim);
        let xt = TypedTensor::from_vec_col_major(shape.clone(), x.clone()).unwrap();
        let y = fitter.evaluate_nd_dz(None, &xt, dim).unwrap();
        let expected = naive_apply(a.as_slice(), N, M, &x, &shape, dim);
        assert!(max_diff(y.host_data().unwrap(), &expected) < 1e-13);
        let back = fitter.fit_nd_zd(None, &y, dim).unwrap();
        assert!(max_diff(back.host_data().unwrap(), &x) < 1e-11);

        // zz variants: real parts in, zero imaginary parts out.
        let xz: Vec<C64> = x.iter().map(|&r| C64::new(r, 0.25)).collect();
        let xzt = TypedTensor::from_vec_col_major(shape.clone(), xz).unwrap();
        let yz = fitter.evaluate_nd_zz(None, &xzt, dim).unwrap();
        assert!(max_diff(yz.host_data().unwrap(), y.host_data().unwrap()) < 1e-14);
        let back = fitter.fit_nd_zz(None, &yz, dim).unwrap();
        let expected: Vec<C64> = x.iter().map(|&r| C64::new(r, 0.0)).collect();
        assert!(max_diff(back.host_data().unwrap(), &expected) < 1e-11);
    }
}

#[test]
fn inplace_strided_input() {
    let a = complex_matrix(N, M);
    let fitter = ComplexToRealFitter::new(a.clone());
    let shape = [3, N];
    let x = real_data(M * 3, 1);
    let y = naive_apply::<C64, f64, C64>(a.as_slice(), N, M, &x, &[3, M], 1);
    let (y_rm, strides) = to_row_major(&y, &shape);
    let view = TypedTensorView::from_slice(shape, strides, 0, &y_rm).unwrap();
    let mut back = vec![0.0; 3 * M];
    let mut out = TypedTensorViewMut::from_slice([3, M], [1, 3], 0, &mut back).unwrap();
    InplaceFitter::fit_nd_zd_to(&fitter, None, &view, 1, &mut out).unwrap();
    assert!(max_diff(&back, &x) < 1e-11);
}
