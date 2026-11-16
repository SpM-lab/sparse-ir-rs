//! Small column-major dense containers for internal numerics.
//!
//! [`Mat`] and [`Mat3`] come from `sparse-ir-sve`, where they hold generic
//! scalars (including `Df64`) for the SVE, polynomial, and kernel-matrix
//! code paths. They are plain column-major `Vec<T>` buffers with infallible
//! indexing: the first index varies fastest in memory, matching Fortran,
//! BLAS, and tenferro.
//!
//! Public APIs that expose `f64`/`Complex<f64>` matrices use
//! [`crate::Matrix`] (`TypedTensor<T, Rank<2>>`); [`MatTyped::into_typed`]
//! and [`MatTyped::from_typed`] move between the two without copying.

use tenferro_tensor::{Rank, TensorScalar, TypedTensor};

pub use sparse_ir_sve::matrix::{Mat, Mat3, mat};

/// Conversions between [`Mat`] and tenferro rank-2 tensors
pub trait MatTyped<T: TensorScalar>: Sized {
    /// Move into a tenferro rank-2 tensor without copying.
    fn into_typed(self) -> TypedTensor<T, Rank<2>>;

    /// Copy a host-resident tenferro matrix.
    ///
    /// # Errors
    /// Returns an error when the tensor is not host-resident compact
    /// column-major storage.
    fn from_typed(t: &TypedTensor<T, Rank<2>>) -> Result<Self, tenferro_tensor::Error>;
}

impl<T: TensorScalar> MatTyped<T> for Mat<T> {
    fn into_typed(self) -> TypedTensor<T, Rank<2>> {
        let dims = self.dims();
        TypedTensor::from_vec_col_major(dims, self.into_vec())
            .expect("Mat invariant: buffer length equals nrows * ncols")
    }

    fn from_typed(t: &TypedTensor<T, Rank<2>>) -> Result<Self, tenferro_tensor::Error> {
        let view = t.host_col_major_view()?;
        Ok(Mat::from_vec_col_major(
            *view.shape(),
            view.as_slice().to_vec(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_roundtrip_is_zero_copy_layout() {
        let a = Mat::from_fn([2, 2], |idx| (idx[0] + 2 * idx[1]) as f64);
        let t = a.clone().into_typed();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(*t.get2(1, 0).unwrap(), 1.0);
        assert_eq!(Mat::from_typed(&t).unwrap(), a);
    }
}
