use crate::gauss::{legendre_generic, legendre_vandermonde};
use crate::interpolation1d::legendre_collocation_matrix;
use crate::matrix::Mat;

#[test]
fn test_legendre_collocation_matrix_inverse() {
    // Test with different sizes
    for n in [2, 3, 5, 10] {
        let gauss_rule = legendre_generic::<f64>(n).reseat(-1.0, 1.0);

        // Create Vandermonde matrix
        let vandermonde = legendre_vandermonde(&gauss_rule.x.to_vec(), n - 1);

        // Create collocation matrix
        let collocation = legendre_collocation_matrix(&gauss_rule);

        // Compute V * C and check if it's approximately the identity matrix
        let mut product = Mat::<f64>::from_elem([n, n], 0.0);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    product[[i, j]] += vandermonde[[i, k]] * collocation[[k, j]];
                }
            }
        }

        // Check that V * C ≈ I
        let mut error = 0.0;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                error += (product[[i, j]] - expected).abs();
            }
        }
        error /= (n * n) as f64;

        println!("n={}, error={}", n, error);
        assert!(
            error < 1e-10,
            "Collocation matrix is not inverse of Vandermonde matrix for n={}: error={}",
            n,
            error
        );
    }
}
