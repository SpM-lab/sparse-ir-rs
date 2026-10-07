//! Tests for kernel matrix discretization

use crate::gauss::legendre;
use crate::kernel::{LogisticKernel, SymmetryType};
use crate::kernelmatrix::matrix_from_gauss;

// ========================================================================
// Basic matrix_from_gauss tests
// ========================================================================

#[test]
fn test_matrix_from_gauss_basic() {
    // 2x2の小さな行列で基本動作確認
    let kernel = LogisticKernel::new(1.0).unwrap();
    let gauss_x = legendre::<f64>(2).reseat(0.0, 1.0);
    let gauss_y = legendre::<f64>(2).reseat(0.0, 1.0);
    let matrix = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);

    assert_eq!(matrix.matrix.shape().0, 2);
    assert_eq!(matrix.matrix.shape().1, 2);
}

#[test]
fn test_matrix_from_gauss_sizes() {
    let kernel = LogisticKernel::new(1.0).unwrap();

    for n in [2, 4, 8] {
        let gauss_x = legendre::<f64>(n).reseat(0.0, 1.0);
        let gauss_y = legendre::<f64>(n).reseat(0.0, 1.0);
        let matrix = matrix_from_gauss(&kernel, &gauss_x, &gauss_y, SymmetryType::Even);

        assert_eq!(matrix.matrix.shape().0, n);
        assert_eq!(matrix.matrix.shape().1, n);
    }
}
