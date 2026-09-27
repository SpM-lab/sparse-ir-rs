//! The one piece of dense linear algebra the applied examples need: the
//! eigendecomposition of a 2×2 Hermitian matrix.
//!
//! Graphene has two sites per cell, so its Hamiltonian at every momentum is a
//! 2×2 Hermitian matrix that has to be diagonalised before the Green's
//! function can be written down. At that size the answer is closed form, and
//! writing it out keeps the result exactly reproducible — a general eigenvalue
//! routine is free to return the eigenvectors in either order or with either
//! phase.
//!
//! The susceptibility the example computes is a trace of products of matrices
//! rotated into the eigenbasis, so it does not depend on the phase convention
//! chosen here; the ordering does matter, and eigenvalues come back ascending.

use num_complex::Complex64;

/// A 2×2 Hermitian matrix `[[a, c], [c*, b]]`, with `a` and `b` real.
#[derive(Clone, Copy, Debug)]
pub struct Hermitian2 {
    pub a: f64,
    pub b: f64,
    pub c: Complex64,
}

/// Eigenvalues in ascending order, and the matrix of eigenvectors as columns.
#[derive(Clone, Copy, Debug)]
pub struct Eigen2 {
    pub values: [f64; 2],
    /// `vectors[i][j]` is component `i` of eigenvector `j`.
    pub vectors: [[Complex64; 2]; 2],
}

impl Hermitian2 {
    pub fn new(a: f64, b: f64, c: Complex64) -> Self {
        Self { a, b, c }
    }

    /// `V† M V`, the matrix in the eigenbasis of `self`.
    pub fn rotate(eigen: &Eigen2, m: &[[Complex64; 2]; 2]) -> [[Complex64; 2]; 2] {
        let mut out = [[Complex64::default(); 2]; 2];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                let mut sum = Complex64::default();
                for (p, m_row) in m.iter().enumerate() {
                    for (q, entry) in m_row.iter().enumerate() {
                        sum += eigen.vectors[p][i].conj() * entry * eigen.vectors[q][j];
                    }
                }
                *slot = sum;
            }
        }
        out
    }

    /// The eigenvalues and eigenvectors, in closed form.
    pub fn eigen(&self) -> Eigen2 {
        let half_trace = 0.5 * (self.a + self.b);
        let gap = (0.25 * (self.a - self.b).powi(2) + self.c.norm_sqr()).sqrt();
        let values = [half_trace - gap, half_trace + gap];

        // `c == 0` leaves the eigenvectors undetermined by the formula below;
        // the matrix is already diagonal there.
        if self.c == Complex64::default() {
            let (lower, upper) = if self.a <= self.b {
                (
                    [Complex64::new(1.0, 0.0), Complex64::default()],
                    [Complex64::default(), Complex64::new(1.0, 0.0)],
                )
            } else {
                (
                    [Complex64::default(), Complex64::new(1.0, 0.0)],
                    [Complex64::new(1.0, 0.0), Complex64::default()],
                )
            };
            return Eigen2 {
                values,
                vectors: [[lower[0], upper[0]], [lower[1], upper[1]]],
            };
        }

        let mut vectors = [[Complex64::default(); 2]; 2];
        for (j, &value) in values.iter().enumerate() {
            // (M − λ) v = 0 gives v ∝ (c, λ − a) for a Hermitian 2×2 with c ≠ 0.
            let v0 = self.c;
            let v1 = Complex64::new(value - self.a, 0.0);
            let norm = (v0.norm_sqr() + v1.norm_sqr()).sqrt();
            vectors[0][j] = v0 / norm;
            vectors[1][j] = v1 / norm;
        }
        Eigen2 { values, vectors }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(eigen: &Eigen2) -> [[Complex64; 2]; 2] {
        let mut out = [[Complex64::default(); 2]; 2];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, slot) in row.iter_mut().enumerate() {
                for k in 0..2 {
                    *slot += eigen.vectors[i][k] * eigen.values[k] * eigen.vectors[j][k].conj();
                }
            }
        }
        out
    }

    fn cases() -> Vec<Hermitian2> {
        vec![
            Hermitian2::new(0.0, 0.0, Complex64::new(0.3, -0.7)),
            Hermitian2::new(1.5, -2.5, Complex64::new(0.0, 1.0)),
            Hermitian2::new(-1.0, -1.0, Complex64::new(2.0, 0.0)),
            Hermitian2::new(3.0, 1.0, Complex64::default()),
            Hermitian2::new(1.0, 3.0, Complex64::default()),
        ]
    }

    #[test]
    fn the_decomposition_reproduces_the_matrix() {
        for matrix in cases() {
            let eigen = matrix.eigen();
            let back = reconstruct(&eigen);
            let expected = [
                [Complex64::new(matrix.a, 0.0), matrix.c],
                [matrix.c.conj(), Complex64::new(matrix.b, 0.0)],
            ];
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (back[i][j] - expected[i][j]).norm() < 1e-14,
                        "{matrix:?}: entry ({i}, {j}) came back as {} instead of {}",
                        back[i][j],
                        expected[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn the_eigenvectors_are_orthonormal_and_the_values_ascending() {
        for matrix in cases() {
            let eigen = matrix.eigen();
            assert!(eigen.values[0] <= eigen.values[1], "{matrix:?}");
            for j in 0..2 {
                for k in 0..2 {
                    let overlap: Complex64 = (0..2)
                        .map(|i| eigen.vectors[i][j].conj() * eigen.vectors[i][k])
                        .sum();
                    let want = if j == k { 1.0 } else { 0.0 };
                    assert!(
                        (overlap - want).norm() < 1e-14,
                        "{matrix:?}: ⟨{j}|{k}⟩ = {overlap}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_graphene_hamiltonian_has_symmetric_bands() {
        // [[0, h], [h*, 0]] has eigenvalues ±|h| for any h.
        let h = Complex64::new(-0.4, 1.1);
        let eigen = Hermitian2::new(0.0, 0.0, h).eigen();
        assert!((eigen.values[0] + h.norm()).abs() < 1e-15);
        assert!((eigen.values[1] - h.norm()).abs() < 1e-15);
    }
}
