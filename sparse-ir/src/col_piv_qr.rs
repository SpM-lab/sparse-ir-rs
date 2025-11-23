//! Column-pivoted QR decomposition with early termination support
//!
//! This module provides a column-pivoted QR decomposition implementation based on nalgebra.
//! It includes support for early termination based on relative tolerance (rtol).
//!
//! # License
//!
//! This file is based on code from the nalgebra library (Apache 2.0 license).
//! Original source: nalgebra/src/linalg/col_piv_qr.rs
//!
//! Copyright 2020 Sébastien Crozet
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//! <http://www.apache.org/licenses/LICENSE-2.0>
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//!
//! Modifications and additions (including early termination support) are licensed under
//! the MIT License as part of the sparse-ir project.

use num_traits::Zero;

use nalgebra::ComplexField;
use nalgebra::allocator::{Allocator, Reallocator};
use nalgebra::base::{Const, DefaultAllocator, Matrix, OMatrix, OVector, Unit};
use nalgebra::constraint::{SameNumberOfRows, ShapeConstraint};
use nalgebra::dimension::{Dim, DimMin, DimMinimum};
use nalgebra::storage::StorageMut;

use nalgebra::geometry::Reflection;
use nalgebra::linalg::{PermutationSequence, householder};
use std::mem::MaybeUninit;

/// The QR decomposition (with column pivoting) of a general matrix.
#[derive(Clone, Debug)]
pub struct ColPivQR<T: ComplexField, R: DimMin<C>, C: Dim>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
{
    col_piv_qr: OMatrix<T, R, C>,
    p: PermutationSequence<DimMinimum<R, C>>,
    diag: OVector<T, DimMinimum<R, C>>,
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> Copy for ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<DimMinimum<R, C>>,
    OMatrix<T, R, C>: Copy,
    PermutationSequence<DimMinimum<R, C>>: Copy,
    OVector<T, DimMinimum<R, C>>: Copy,
{
}

impl<T: ComplexField, R: DimMin<C>, C: Dim> ColPivQR<T, R, C>
where
    DefaultAllocator: Allocator<R, C> + Allocator<R> + Allocator<DimMinimum<R, C>>,
{
    /// Computes the `ColPivQR` decomposition using householder reflections.
    pub fn new(matrix: OMatrix<T, R, C>) -> Self {
        Self::new_with_rtol(matrix, None)
    }

    /// Computes the `ColPivQR` decomposition using householder reflections with early termination.
    ///
    /// # Arguments
    /// * `matrix` - Input matrix to decompose
    /// * `rtol` - Optional relative tolerance for early termination.
    ///            If `Some(rtol)`, the decomposition stops when `abs(diag[i]) < rtol * abs(diag[0])`.
    ///            If `None`, all columns are processed (no early termination).
    ///
    /// # Returns
    /// * `ColPivQR` - QR decomposition result with column pivoting. If early termination occurred,
    ///                remaining diagonal elements are set to zero.
    pub fn new_with_rtol(mut matrix: OMatrix<T, R, C>, rtol: Option<T::RealField>) -> Self
    where
        T: ComplexField,
    {
        let (nrows, ncols) = matrix.shape_generic();
        let min_nrows_ncols = nrows.min(ncols);
        let mut p = PermutationSequence::identity_generic(min_nrows_ncols);

        if min_nrows_ncols.value() == 0 {
            return ColPivQR {
                col_piv_qr: matrix,
                p,
                diag: Matrix::zeros_generic(min_nrows_ncols, Const::<1>),
            };
        }

        let mut diag = Matrix::uninit(min_nrows_ncols, Const::<1>);
        let mut first_diag_abs = None;

        for i in 0..min_nrows_ncols.value() {
            let piv = matrix.view_range(i.., i..).icamax_full();
            let col_piv = piv.1 + i;
            matrix.swap_columns(i, col_piv);
            p.append_permutation(i, col_piv);

            let diag_value = householder::clear_column_unchecked(&mut matrix, i, 0, None);
            let diag_abs = diag_value.clone().modulus();

            // Store first diagonal element's absolute value for early termination check
            if i == 0 {
                first_diag_abs = Some(diag_abs.clone());
            }

            // Check for early termination if rtol is provided
            if let Some(ref rtol_val) = rtol {
                if let Some(ref first_abs) = first_diag_abs {
                    if diag_abs < rtol_val.clone() * first_abs.clone() {
                        // Early termination: set remaining diagonal elements to zero
                        for j in i..min_nrows_ncols.value() {
                            diag[j] = MaybeUninit::new(T::zero());
                        }
                        break;
                    }
                }
            }

            diag[i] = MaybeUninit::new(diag_value);
        }

        // Safety: diag is now fully initialized (either with values or zeros).
        let diag = unsafe { diag.assume_init() };

        ColPivQR {
            col_piv_qr: matrix,
            p,
            diag,
        }
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    #[inline]
    #[must_use]
    pub fn r(&self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Allocator<DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.shape_generic();
        let mut res = self
            .col_piv_qr
            .rows_generic(0, nrows.min(ncols))
            .upper_triangle();
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.clone().modulus())));
        res
    }

    /// Retrieves the upper trapezoidal submatrix `R` of this decomposition.
    ///
    /// This is usually faster than `r` but consumes `self`.
    #[inline]
    pub fn unpack_r(self) -> OMatrix<T, DimMinimum<R, C>, C>
    where
        DefaultAllocator: Reallocator<T, R, C, DimMinimum<R, C>, C>,
    {
        let (nrows, ncols) = self.col_piv_qr.shape_generic();
        let mut res = self
            .col_piv_qr
            .resize_generic(nrows.min(ncols), ncols, T::zero());
        res.fill_lower_triangle(T::zero(), 1);
        res.set_partial_diagonal(self.diag.iter().map(|e| T::from_real(e.clone().modulus())));
        res
    }

    /// Computes the orthogonal matrix `Q` of this decomposition.
    #[must_use]
    pub fn q(&self) -> OMatrix<T, R, DimMinimum<R, C>>
    where
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>,
    {
        let (nrows, ncols) = self.col_piv_qr.shape_generic();

        // NOTE: we could build the identity matrix and call q_mul on it.
        // Instead we don't so that we take in account the matrix sparseness.
        let mut res = Matrix::identity_generic(nrows, nrows.min(ncols));
        let dim = self.diag.len();

        // Find the effective rank (first zero diagonal element)
        let mut effective_rank = dim;
        for i in 0..dim {
            if self.diag[i].is_zero() {
                effective_rank = i;
                break;
            }
        }

        // Apply householder reflections only up to effective rank
        for i in (0..effective_rank).rev() {
            let axis = self.col_piv_qr.view_range(i.., i);
            // TODO: sometimes, the axis might have a zero magnitude.
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut res_rows = res.view_range_mut(i.., i..);
            refl.reflect_with_sign(&mut res_rows, self.diag[i].clone().signum());
        }

        // Set remaining columns to zero if early termination occurred
        if effective_rank < dim {
            for j in effective_rank..dim {
                res.column_mut(j).fill(T::zero());
            }
        }

        res
    }
    /// Retrieves the column permutation of this decomposition.
    #[inline]
    #[must_use]
    pub const fn p(&self) -> &PermutationSequence<DimMinimum<R, C>> {
        &self.p
    }

    /// Unpacks this decomposition into its two matrix factors.
    pub fn unpack(
        self,
    ) -> (
        OMatrix<T, R, DimMinimum<R, C>>,
        OMatrix<T, DimMinimum<R, C>, C>,
        PermutationSequence<DimMinimum<R, C>>,
    )
    where
        DimMinimum<R, C>: DimMin<C, Output = DimMinimum<R, C>>,
        DefaultAllocator: Allocator<R, DimMinimum<R, C>>
            + Reallocator<T, R, C, DimMinimum<R, C>, C>
            + Allocator<DimMinimum<R, C>>,
    {
        (self.q(), self.r(), self.p)
    }

    #[doc(hidden)]
    pub const fn col_piv_qr_internal(&self) -> &OMatrix<T, R, C> {
        &self.col_piv_qr
    }

    #[must_use]
    pub(crate) const fn diag_internal(&self) -> &OVector<T, DimMinimum<R, C>> {
        &self.diag
    }

    /// Multiplies the provided matrix by the transpose of the `Q` matrix of this decomposition.
    pub fn q_tr_mul<R2: Dim, C2: Dim, S2>(&self, rhs: &mut Matrix<T, R2, C2, S2>)
    where
        S2: StorageMut<T, R2, C2>,
    {
        let dim = self.diag.len();

        for i in 0..dim {
            let axis = self.col_piv_qr.view_range(i.., i);
            let refl = Reflection::new(Unit::new_unchecked(axis), T::zero());

            let mut rhs_rows = rhs.rows_range_mut(i..);
            refl.reflect_with_sign(&mut rhs_rows, self.diag[i].clone().signum().conjugate());
        }
    }

    /// Returns the effective rank of the QR decomposition.
    ///
    /// The effective rank is the number of non-zero diagonal elements in R,
    /// or the number of diagonal elements that are above the relative tolerance
    /// if early termination was used.
    ///
    /// # Returns
    /// * `usize` - Effective rank (number of significant diagonal elements)
    pub fn rank(&self) -> usize {
        let dim = self.diag.len();
        if dim == 0 {
            return 0;
        }

        // Find the first non-zero diagonal element to use as reference
        let first_diag_abs = self.diag[0].clone().modulus();
        if first_diag_abs.is_zero() {
            return 0;
        }

        // Count diagonal elements that are non-zero
        // For early termination, we count until we hit a zero or very small value
        let mut rank = 0;
        for i in 0..dim {
            let diag_abs = self.diag[i].clone().modulus();
            if diag_abs.is_zero() {
                break;
            }
            rank += 1;
        }

        rank
    }

    /// Returns the effective rank based on a relative tolerance.
    ///
    /// # Arguments
    /// * `rtol` - Relative tolerance. Elements with `abs(diag[i]) < rtol * abs(diag[0])`
    ///            are considered zero.
    ///
    /// # Returns
    /// * `usize` - Effective rank
    pub fn rank_with_rtol(&self, rtol: T::RealField) -> usize
    where
        T: ComplexField,
    {
        let dim = self.diag.len();
        if dim == 0 {
            return 0;
        }

        let first_diag_abs = self.diag[0].clone().modulus();
        if first_diag_abs.is_zero() {
            return 0;
        }

        let threshold = rtol * first_diag_abs;
        let mut rank = 0;

        for i in 0..dim {
            let diag_abs = self.diag[i].clone().modulus();
            if diag_abs < threshold {
                break;
            }
            rank += 1;
        }

        rank
    }
}

impl<T: ComplexField, D: DimMin<D, Output = D>> ColPivQR<T, D, D>
where
    DefaultAllocator: Allocator<D, D> + Allocator<D> + Allocator<DimMinimum<D, D>>,
{
    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// Returns `None` if `self` is not invertible.
    #[must_use = "Did you mean to use solve_mut()?"]
    pub fn solve<R2: Dim, C2: Dim, S2>(
        &self,
        b: &Matrix<T, R2, C2, S2>,
    ) -> Option<OMatrix<T, R2, C2>>
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
        DefaultAllocator: Allocator<R2, C2>,
    {
        let mut res = b.clone_owned();

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Solves the linear system `self * x = b`, where `x` is the unknown to be determined.
    ///
    /// If the decomposed matrix is not invertible, this returns `false` and its input `b` is
    /// overwritten with garbage.
    pub fn solve_mut<R2: Dim, C2: Dim, S2>(&self, b: &mut Matrix<T, R2, C2, S2>) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        assert_eq!(
            self.col_piv_qr.nrows(),
            b.nrows(),
            "ColPivQR solve matrix dimension mismatch."
        );
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR solve: unable to solve a non-square system."
        );

        self.q_tr_mul(b);
        let solved = self.solve_upper_triangular_mut(b);
        self.p.inv_permute_rows(b);

        solved
    }

    // TODO: duplicate code from the `solve` module.
    fn solve_upper_triangular_mut<R2: Dim, C2: Dim, S2>(
        &self,
        b: &mut Matrix<T, R2, C2, S2>,
    ) -> bool
    where
        S2: StorageMut<T, R2, C2>,
        ShapeConstraint: SameNumberOfRows<R2, D>,
    {
        let dim = self.col_piv_qr.nrows();

        for k in 0..b.ncols() {
            let mut b = b.column_mut(k);
            for i in (0..dim).rev() {
                let coeff;

                unsafe {
                    let diag = self.diag.vget_unchecked(i).clone().modulus();

                    if diag.is_zero() {
                        return false;
                    }

                    coeff = b.vget_unchecked(i).clone().unscale(diag);
                    *b.vget_unchecked_mut(i) = coeff.clone();
                }

                b.rows_range_mut(..i)
                    .axpy(-coeff, &self.col_piv_qr.view_range(..i, i), T::one());
            }
        }

        true
    }

    /// Computes the inverse of the decomposed matrix.
    ///
    /// Returns `None` if the decomposed matrix is not invertible.
    #[must_use]
    pub fn try_inverse(&self) -> Option<OMatrix<T, D, D>> {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR inverse: unable to compute the inverse of a non-square matrix."
        );

        // TODO: is there a less naive method ?
        let (nrows, ncols) = self.col_piv_qr.shape_generic();
        let mut res = OMatrix::identity_generic(nrows, ncols);

        if self.solve_mut(&mut res) {
            Some(res)
        } else {
            None
        }
    }

    /// Indicates if the decomposed matrix is invertible.
    #[must_use]
    pub fn is_invertible(&self) -> bool {
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR: unable to test the invertibility of a non-square matrix."
        );

        for i in 0..self.diag.len() {
            if self.diag[i].is_zero() {
                return false;
            }
        }

        true
    }

    /// Computes the determinant of the decomposed matrix.
    #[must_use]
    pub fn determinant(&self) -> T {
        let dim = self.col_piv_qr.nrows();
        assert!(
            self.col_piv_qr.is_square(),
            "ColPivQR determinant: unable to compute the determinant of a non-square matrix."
        );

        let mut res = T::one();
        for i in 0..dim {
            res *= unsafe { self.diag.vget_unchecked(i).clone() };
        }

        res * self.p.determinant()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, Dyn};

    /// Create Hilbert matrix of size nrows x ncols
    /// H[i,j] = 1 / (i + j + 1)
    fn create_hilbert_matrix(nrows: usize, ncols: usize) -> DMatrix<f64> {
        DMatrix::from_fn(nrows, ncols, |i, j| 1.0 / ((i + j + 1) as f64))
    }

    /// Reconstruct matrix from QR decomposition: A = Q * R * P^T
    /// where P is the permutation matrix
    fn reconstruct_matrix_from_col_piv_qr(
        q: &DMatrix<f64>,
        r: &DMatrix<f64>,
        p: &nalgebra::linalg::PermutationSequence<Dyn>,
    ) -> DMatrix<f64> {
        // A = Q * R * P^T
        // First compute Q * R
        let qr = q * r;

        // Apply inverse permutation to columns (P^T applied to columns)
        let mut result = qr.clone();
        p.inv_permute_columns(&mut result);
        result
    }

    /// Calculate Frobenius norm of matrix
    fn frobenius_norm(matrix: &DMatrix<f64>) -> f64 {
        let mut sum = 0.0;
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                let val = matrix[(i, j)];
                sum += val * val;
            }
        }
        sum.sqrt()
    }

    #[test]
    fn test_col_piv_qr_hilbert_with_rtol() {
        let rtol = 1e-10;

        // Test square matrix 20x20
        test_hilbert_matrix_with_rtol(20, 20, rtol, "20x20");

        // Test rectangular matrices
        test_hilbert_matrix_with_rtol(20, 30, rtol, "20x30");
        test_hilbert_matrix_with_rtol(30, 20, rtol, "30x20");
    }

    fn test_hilbert_matrix_with_rtol(nrows: usize, ncols: usize, rtol: f64, label: &str) {
        let h = create_hilbert_matrix(nrows, ncols);
        let min_dim = nrows.min(ncols);

        println!("\n=== Testing Hilbert {}x{} ({}) ===", nrows, ncols, label);

        // Compute QR with and without early termination
        let qr_with_rtol = ColPivQR::new_with_rtol(h.clone(), Some(rtol));
        let qr_without_rtol = ColPivQR::new_with_rtol(h.clone(), None);

        // Check that early termination reduces the effective rank
        let rank_with_rtol = qr_with_rtol.rank_with_rtol(rtol);
        let rank_without_rtol = qr_without_rtol.rank();

        println!(
            "Hilbert {}x{} ({}): rank with rtol={} is {}, without rtol is {}",
            nrows, ncols, label, rtol, rank_with_rtol, rank_without_rtol
        );

        // Early termination should give a rank <= full rank
        assert!(
            rank_with_rtol <= rank_without_rtol,
            "Early termination rank {} should be <= full rank {}",
            rank_with_rtol,
            rank_without_rtol
        );

        // For Hilbert matrix, early termination should reduce rank significantly
        // due to numerical rank deficiency
        assert!(
            rank_with_rtol < min_dim,
            "Early termination should reduce rank for ill-conditioned matrix"
        );

        // Check that diagonal values satisfy rtol condition
        let diag = qr_with_rtol.diag_internal();
        let first_diag_abs = diag[0].clone().modulus();
        let threshold = rtol * first_diag_abs;

        println!("First diagonal element abs: {}", first_diag_abs);
        println!("Threshold (rtol * first_diag_abs): {}", threshold);

        // All elements before rank should be >= threshold
        for i in 0..rank_with_rtol {
            let diag_abs = diag[i].clone().modulus();
            assert!(
                diag_abs >= threshold,
                "Diagonal element [{}] abs={} should be >= threshold {}",
                i,
                diag_abs,
                threshold
            );
        }

        // If rank < full dimension, the element at rank should be < threshold (if early termination occurred)
        if rank_with_rtol < diag.len() {
            let diag_abs_at_rank = diag[rank_with_rtol].clone().modulus();
            println!(
                "Diagonal element [{}] abs={}",
                rank_with_rtol, diag_abs_at_rank
            );

            // If early termination occurred, this element should be below threshold
            // (or zero if it was set to zero during early termination)
            if diag_abs_at_rank > 0.0 {
                assert!(
                    diag_abs_at_rank < threshold,
                    "Diagonal element [{}] abs={} should be < threshold {} (early termination check)",
                    rank_with_rtol,
                    diag_abs_at_rank,
                    threshold
                );
            }
        }

        // Reconstruct matrix and check error
        let q = qr_with_rtol.q();
        let r = qr_with_rtol.r();
        let p = qr_with_rtol.p();

        // Check that Q's remaining columns are zero after early termination
        if rank_with_rtol < q.ncols() {
            println!(
                "Checking Q matrix columns after rank {} (total columns: {})",
                rank_with_rtol,
                q.ncols()
            );
            for j in rank_with_rtol..q.ncols() {
                let q_col = q.column(j);
                let col_norm = q_col.norm();
                println!("Q column [{}] norm: {}", j, col_norm);
                assert!(
                    col_norm < 1e-12,
                    "Q column [{}] should be zero after early termination, but norm is {}",
                    j,
                    col_norm
                );
            }
        }

        let h_reconstructed = reconstruct_matrix_from_col_piv_qr(&q, &r, p);

        // Calculate reconstruction error
        let h_norm = frobenius_norm(&h);
        let error_matrix = &h - &h_reconstructed;
        let error_norm = frobenius_norm(&error_matrix);
        let relative_error = error_norm / h_norm;

        println!(
            "Hilbert {}x{} ({}): relative reconstruction error = {}",
            nrows, ncols, label, relative_error
        );

        // Check that reconstruction error is reasonable
        // Note: 20x20 Hilbert matrix has very large condition number, so we expect larger errors
        assert!(
            relative_error < 1e-6,
            "Relative reconstruction error {} exceeds 1e-6",
            relative_error
        );
    }

    #[test]
    fn test_col_piv_qr_identity_matrix() {
        let matrix = DMatrix::<f64>::identity(5, 5);
        let rtol = 1e-10;

        let qr = ColPivQR::new_with_rtol(matrix.clone(), Some(rtol));

        // Identity matrix should have full rank
        let rank = qr.rank_with_rtol(rtol);
        assert_eq!(rank, 5, "Identity matrix should have full rank");

        // Reconstruct and verify
        let q = qr.q();
        let r = qr.r();
        let p = qr.p();
        let reconstructed = reconstruct_matrix_from_col_piv_qr(&q, &r, p);

        let error = frobenius_norm(&(&matrix - &reconstructed));
        assert!(
            error < 1e-12,
            "Reconstruction error {} should be small",
            error
        );
    }
}
