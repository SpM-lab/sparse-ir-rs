//! Small column-major dense containers for internal numerics.
//!
//! [`Mat`] and [`Mat3`] hold generic scalars (including [`crate::Df64`]) for
//! the SVE, polynomial, and kernel-matrix code paths. They are plain
//! column-major `Vec<T>` buffers with infallible indexing: the first index
//! varies fastest in memory, matching Fortran, BLAS, and tenferro.
//!
//! Public APIs that expose `f64`/`Complex<f64>` matrices use
//! [`crate::Matrix`] (`TypedTensor<T, Rank<2>>`); [`Mat::into_typed`] and
//! [`Mat::from_typed`] move between the two without copying.

use std::ops::{Index, IndexMut};

use num_traits::Zero;
use tenferro_tensor::{Rank, TensorScalar, TypedTensor};

/// Column-major dense matrix with infallible indexing.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat<T> {
    data: Vec<T>,
    shape: (usize, usize),
}

impl<T> Mat<T> {
    /// Wrap a column-major buffer.
    ///
    /// # Panics
    /// Panics if `data.len() != nrows * ncols`; callers construct the buffer
    /// with the matching size.
    pub fn from_vec_col_major(shape: [usize; 2], data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            shape[0] * shape[1],
            "Mat::from_vec_col_major: buffer length does not match shape"
        );
        Self {
            data,
            shape: (shape[0], shape[1]),
        }
    }

    /// Build a matrix by evaluating `f(&[i, j])` for every element.
    pub fn from_fn<F: FnMut(&[usize]) -> T>(shape: [usize; 2], mut f: F) -> Self {
        let (m, n) = (shape[0], shape[1]);
        let mut data = Vec::with_capacity(m * n);
        for j in 0..n {
            for i in 0..m {
                data.push(f(&[i, j]));
            }
        }
        Self {
            data,
            shape: (m, n),
        }
    }

    /// Shape as `(nrows, ncols)`.
    #[inline]
    pub fn shape(&self) -> &(usize, usize) {
        &self.shape
    }

    /// Shape as `[nrows, ncols]`, the form [`Mat::from_fn`] takes.
    #[inline]
    pub fn dims(&self) -> [usize; 2] {
        [self.shape.0, self.shape.1]
    }

    /// Extent of axis `axis` (0 = rows, 1 = columns).
    #[inline]
    pub fn dim(&self, axis: usize) -> usize {
        match axis {
            0 => self.shape.0,
            1 => self.shape.1,
            _ => panic!("Mat::dim: axis {axis} out of range for a matrix"),
        }
    }

    #[inline]
    pub fn nrows(&self) -> usize {
        self.shape.0
    }

    #[inline]
    pub fn ncols(&self) -> usize {
        self.shape.1
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Column-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable column-major storage.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume into the column-major buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Contiguous column `j`.
    #[inline]
    pub fn col(&self, j: usize) -> &[T] {
        let m = self.shape.0;
        &self.data[j * m..(j + 1) * m]
    }

    /// Mutable contiguous column `j`.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [T] {
        let m = self.shape.0;
        &mut self.data[j * m..(j + 1) * m]
    }

    /// Iterate over elements in column-major order.
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }

    /// Iterate mutably over elements in column-major order.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }

    /// Apply `f` element-wise.
    pub fn map<U, F: FnMut(&T) -> U>(&self, f: F) -> Mat<U> {
        Mat {
            data: self.data.iter().map(f).collect(),
            shape: self.shape,
        }
    }
}

impl<T: Clone> Mat<T> {
    /// Matrix filled with `elem`.
    pub fn from_elem(shape: [usize; 2], elem: T) -> Self {
        Self {
            data: vec![elem; shape[0] * shape[1]],
            shape: (shape[0], shape[1]),
        }
    }

    /// Transposed copy.
    pub fn transpose(&self) -> Self {
        let (m, n) = self.shape;
        Self::from_fn([n, m], |idx| self.data[idx[1] + m * idx[0]].clone())
    }

    /// Copy of the leading `ncols` columns.
    pub fn leading_cols(&self, ncols: usize) -> Self {
        assert!(ncols <= self.shape.1, "Mat::leading_cols: too many columns");
        Self {
            data: self.data[..self.shape.0 * ncols].to_vec(),
            shape: (self.shape.0, ncols),
        }
    }
}

impl<T: Clone> Mat<T> {
    /// Build from a list of rows (row-literal order, column-major storage).
    ///
    /// # Panics
    /// Panics if the rows have different lengths.
    pub fn from_rows(rows: Vec<Vec<T>>) -> Self {
        let m = rows.len();
        let n = rows.first().map_or(0, Vec::len);
        assert!(
            rows.iter().all(|r| r.len() == n),
            "Mat::from_rows: ragged rows"
        );
        Self::from_fn([m, n], |idx| rows[idx[0]][idx[1]].clone())
    }
}

/// Row-literal matrix constructor: `mat![[a, b], [c, d]]`.
#[allow(unused_macros)]
macro_rules! mat {
    ($([$($x:expr),* $(,)?]),+ $(,)?) => {
        $crate::matrix::Mat::from_rows(vec![$(vec![$($x),*]),+])
    };
}
#[allow(unused_imports)]
pub(crate) use mat;

impl<T: Clone + Zero> Mat<T> {
    /// Zero matrix.
    pub fn zeros(shape: [usize; 2]) -> Self {
        Self::from_elem(shape, T::zero())
    }
}

impl<T: TensorScalar> Mat<T> {
    /// Move into a tenferro rank-2 tensor without copying.
    pub fn into_typed(self) -> TypedTensor<T, Rank<2>> {
        let (m, n) = self.shape;
        TypedTensor::from_vec_col_major([m, n], self.data)
            .expect("Mat invariant: buffer length equals nrows * ncols")
    }

    /// Copy a host-resident tenferro matrix.
    ///
    /// # Errors
    /// Returns an error when the tensor is not host-resident compact
    /// column-major storage.
    pub fn from_typed(t: &TypedTensor<T, Rank<2>>) -> Result<Self, tenferro_tensor::Error> {
        let view = t.host_col_major_view()?;
        let shape = *view.shape();
        Ok(Self {
            data: view.as_slice().to_vec(),
            shape: (shape[0], shape[1]),
        })
    }
}

impl<T> Index<[usize; 2]> for Mat<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: [usize; 2]) -> &T {
        debug_assert!(idx[0] < self.shape.0 && idx[1] < self.shape.1);
        &self.data[idx[0] + self.shape.0 * idx[1]]
    }
}

impl<T> IndexMut<[usize; 2]> for Mat<T> {
    #[inline(always)]
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        debug_assert!(idx[0] < self.shape.0 && idx[1] < self.shape.1);
        &mut self.data[idx[0] + self.shape.0 * idx[1]]
    }
}

impl<T> Index<&[usize]> for Mat<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: &[usize]) -> &T {
        &self[[idx[0], idx[1]]]
    }
}

/// Column-major dense rank-3 array with infallible indexing.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat3<T> {
    data: Vec<T>,
    shape: (usize, usize, usize),
}

impl<T> Mat3<T> {
    /// Shape as `(n0, n1, n2)`.
    #[inline]
    pub fn shape(&self) -> &(usize, usize, usize) {
        &self.shape
    }

    /// Column-major storage.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Consume into the column-major buffer.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Wrap a column-major buffer.
    ///
    /// # Panics
    /// Panics if the buffer length does not match the shape.
    pub fn from_vec_col_major(shape: [usize; 3], data: Vec<T>) -> Self {
        assert_eq!(
            data.len(),
            shape[0] * shape[1] * shape[2],
            "Mat3::from_vec_col_major: buffer length does not match shape"
        );
        Self {
            data,
            shape: (shape[0], shape[1], shape[2]),
        }
    }
}

impl<T: Clone> Mat3<T> {
    /// Array filled with `elem`.
    pub fn from_elem(shape: [usize; 3], elem: T) -> Self {
        Self {
            data: vec![elem; shape[0] * shape[1] * shape[2]],
            shape: (shape[0], shape[1], shape[2]),
        }
    }
}

impl<T: Clone + Zero> Mat3<T> {
    /// Zero array.
    pub fn zeros(shape: [usize; 3]) -> Self {
        Self::from_elem(shape, T::zero())
    }
}

impl<T> Index<[usize; 3]> for Mat3<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, idx: [usize; 3]) -> &T {
        let (a, b, _) = self.shape;
        &self.data[idx[0] + a * (idx[1] + b * idx[2])]
    }
}

impl<T> IndexMut<[usize; 3]> for Mat3<T> {
    #[inline(always)]
    fn index_mut(&mut self, idx: [usize; 3]) -> &mut T {
        let (a, b, _) = self.shape;
        &mut self.data[idx[0] + a * (idx[1] + b * idx[2])]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn col_major_layout_and_transpose() {
        let a = Mat::from_fn([2, 3], |idx| (10 * idx[0] + idx[1]) as f64);
        assert_eq!(a.as_slice(), &[0.0, 10.0, 1.0, 11.0, 2.0, 12.0]);
        assert_eq!(a[[1, 2]], 12.0);
        assert_eq!(a.col(1), &[1.0, 11.0]);
        let t = a.transpose();
        assert_eq!(*t.shape(), (3, 2));
        assert_eq!(t[[2, 1]], 12.0);
    }

    #[test]
    fn typed_roundtrip_is_zero_copy_layout() {
        let a = Mat::from_fn([2, 2], |idx| (idx[0] + 2 * idx[1]) as f64);
        let t = a.clone().into_typed();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(*t.get2(1, 0).unwrap(), 1.0);
        assert_eq!(Mat::from_typed(&t).unwrap(), a);
    }

    #[test]
    fn rank3_indexing() {
        let mut a = Mat3::<f64>::zeros([2, 3, 4]);
        a[[1, 2, 3]] = 5.0;
        assert_eq!(a.as_slice()[1 + 2 * (2 + 3 * 3)], 5.0);
    }
}
