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
//! the same dual license as the sparse-ir crate (MIT OR Apache-2.0).

use num_traits::Zero;

use nalgebra::ComplexField;
use nalgebra::allocator::Allocator;
use nalgebra::base::{Const, DefaultAllocator, Matrix, OMatrix, OVector, Unit};
use nalgebra::dimension::{Dim, DimMin, DimMinimum};

use nalgebra::base::storage::{RawStorage, StorageMut};
use nalgebra::geometry::Reflection;
use nalgebra::linalg::{PermutationSequence, householder};
use std::mem::MaybeUninit;

/// Index of the entry of `m` with the largest `norm1`, first one in
/// column-major order on a tie.
///
/// This is nalgebra's `Matrix::icamax_full` (`src/base/min_max.rs`), copied so
/// that it is inlined into the caller and therefore compiled with the caller's
/// target features.
#[inline(always)]
fn icamax_full_inline<T: ComplexField, R: Dim, C: Dim, S: RawStorage<T, R, C>>(
    m: &Matrix<T, R, C, S>,
) -> (usize, usize) {
    let mut the_max = unsafe { m.get_unchecked((0, 0)).clone().norm1() };
    let mut the_ij = (0, 0);
    for j in 0..m.ncols() {
        for i in 0..m.nrows() {
            let val = unsafe { m.get_unchecked((i, j)).clone().norm1() };
            if val > the_max {
                the_max = val;
                the_ij = (i, j);
            }
        }
    }
    the_ij
}

/// The Householder step of nalgebra's `householder::clear_column_unchecked`,
/// with `reflection_axis_mut` and `Reflection::reflect_with_sign` applied
/// inline.
///
/// The bodies mirror nalgebra 0.33 (`src/linalg/householder.rs` and
/// `src/geometry/reflection.rs`, Apache-2.0, see the file header) operation for
/// operation, so the result is unchanged.  `#[inline(always)]` is what makes
/// the arithmetic compile inside the entry point that calls it, and hence with
/// that entry point's target features - without a fused multiply-add the
/// double-double arithmetic of the scalar type calls the software `fma` in
/// libm, which dominates the cost of the SVE.
///
/// # Safety
/// `icol < matrix.ncols()` and `icol + shift < matrix.nrows()`.
#[inline(always)]
unsafe fn clear_column_inline<T: ComplexField, R: Dim, C: Dim>(
    matrix: &mut OMatrix<T, R, C>,
    icol: usize,
    shift: usize,
) -> T
where
    DefaultAllocator: Allocator<R, C> + Allocator<R>,
{
    let (mut left, mut right) = matrix.columns_range_pair_mut(icol, icol + 1..);
    let mut axis = left.rows_range_mut(icol + shift..);

    // nalgebra::linalg::householder::reflection_axis_mut
    let reflection_sq_norm = axis.norm_squared();
    let reflection_norm = reflection_sq_norm.clone().sqrt();
    let (modulus, sign) = axis[0].clone().to_exp();
    let signed_norm = sign.scale(reflection_norm.clone());
    let factor = (reflection_sq_norm + modulus * reflection_norm) * nalgebra::convert(2.0);
    axis[0] += signed_norm.clone();
    let not_zero = !factor.is_zero();
    if not_zero {
        axis.unscale_mut(factor.sqrt());
        let _ = axis.normalize_mut();
    }
    let reflection_norm = -signed_norm;

    if not_zero {
        // nalgebra::geometry::Reflection::reflect_with_sign with zero bias:
        // column = sign * column + factor * axis
        let sign = reflection_norm.clone().signum().conjugate();
        let m_two = sign.clone().scale(nalgebra::convert(-2.0f64));
        let mut tail = right.rows_range_mut(icol + shift..);
        for i in 0..tail.ncols() {
            let factor = axis.dotc(&tail.column(i)) * m_two.clone();
            let mut col = tail.column_mut(i);
            for (value, elem) in col.iter_mut().zip(axis.iter()) {
                *value = sign.clone() * value.clone() + factor.clone() * elem.clone();
            }
        }
    }

    reflection_norm
}

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
    pub fn new_with_rtol(matrix: OMatrix<T, R, C>, rtol: Option<T::RealField>) -> Self
    where
        T: ComplexField,
    {
        // SAFETY: the body only uses `assume_init` on the diagonal it fills.
        unsafe { Self::new_with_rtol_impl(matrix, rtol) }
    }

    /// [`Self::new_with_rtol`] compiled for a target that has the `fma`
    /// instruction.
    ///
    /// Marking this entry point is what decides whether the arithmetic below it
    /// is compiled with FMA: everything inlined into it - the trailing
    /// Householder update, the pivot search and the arithmetic of the scalar
    /// type - then uses the instruction, whereas without it `f64::mul_add`
    /// calls the software implementation in libm, which is what dominates the
    /// cost of the SVE.
    ///
    /// # Safety
    /// The caller must have established that the target supports `fma`
    /// (see [`crate::numeric::fma_available`]).
    #[target_feature(enable = "fma")]
    pub unsafe fn new_with_rtol_fma(matrix: OMatrix<T, R, C>, rtol: Option<T::RealField>) -> Self
    where
        T: ComplexField,
    {
        // SAFETY: as in `new_with_rtol`.
        unsafe { Self::new_with_rtol_impl(matrix, rtol) }
    }

    /// The body shared by [`Self::new_with_rtol`] and
    /// [`Self::new_with_rtol_fma`], `#[inline(always)]` so that it is compiled
    /// inside whichever entry point calls it.
    ///
    /// # Safety
    /// The body only uses `assume_init` on the diagonal that the loop fills.
    #[inline(always)]
    unsafe fn new_with_rtol_impl(mut matrix: OMatrix<T, R, C>, rtol: Option<T::RealField>) -> Self
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
            let piv = icamax_full_inline(&matrix.view_range(i.., i..));
            let col_piv = piv.1 + i;
            matrix.swap_columns(i, col_piv);
            p.append_permutation(i, col_piv);

            // SAFETY: i < min(nrows, ncols), so i < nrows and i < ncols.
            let diag_value = unsafe { clear_column_inline(&mut matrix, i, 0) };
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

    #[must_use]
    #[allow(dead_code)] // Used in tests
    pub(crate) const fn diag_internal(&self) -> &OVector<T, DimMinimum<R, C>> {
        &self.diag
    }

    /// Returns the effective rank of the QR decomposition.
    ///
    /// The effective rank is the number of non-zero diagonal elements in R,
    /// or the number of diagonal elements that are above the relative tolerance
    /// if early termination was used.
    ///
    /// # Returns
    /// * `usize` - Effective rank (number of significant diagonal elements)
    #[allow(dead_code)] // Used in tests
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
    #[allow(dead_code)] // Used in tests
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
